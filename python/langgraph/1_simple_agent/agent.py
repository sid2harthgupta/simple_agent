# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

"""LangGraph-based conversational agent with automatic state management."""
import uuid
from typing import Annotated, Dict, List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage, BaseMessageType
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage as LangchainBaseMessage, AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from tools import check_supplier_compliance, assess_disruption_risk


class AgentState(TypedDict):
    """Agent state with automatic message accumulation."""
    messages: Annotated[List, add_messages]


class Agent(BaseAgent):
    """LangGraph wrapper for stateful agents with tool support and conversation memory.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_tavily import TavilySearch
        >>> from galileo.handlers.langchain import GalileoCallback
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [TavilySearch(max_results=3)]
        >>> callbacks = [GalileoCallback()]
        >>> agent = Agent(llm, tools, callbacks)
        >>> response = agent.invoke("What are supply chain risks?")
    """

    def __init__(
            self,
            llm: BaseChatModel,
            tools: List[BaseTool],
            callbacks: List
    ) -> None:
        """Initialize agent with LangGraph-managed state.

        Args:
            llm: Chat model for generating responses.
            tools: Available tools for the agent.
            callbacks: Monitoring/logging callbacks.
        """
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools)
        self.callbacks = callbacks
        self.config = self._create_config(callbacks)

        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return "Supply Chain Agent"

    @property
    def capabilities(self) -> List[str]:
        return ["Web search", "Tool calling", "Memory"]

    @property
    def example_queries(self) -> List[str]:
        return [
            "Who has won the most men's singles tennis matches?",
            "What is the compliance status for SUP001?",
            "What is the square root of 34 multiplied by 5?",
        ]

    def reset(self) -> None:
        self.graph = self._build_graph()
        self.config = self._create_config(self.callbacks)

    def invoke(self, user_message: str) -> str:
        """Send message to agent and get response.

        LangGraph automatically loads conversation history, processes the message
        through the graph (with tools if needed), and saves the updated state.

        Args:
            user_message: User's input message.

        Returns:
            Agent's response as string.
        """
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=self.config
        )
        return result["messages"][-1].content

    def get_message_history(self) -> List[BaseMessage]:
        """Get current conversation history."""
        current_state = self.graph.get_state(config=self.config)
        messages = [m for m in current_state.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
        return [AgentFactory.langchain_to_base_message(message) for message in messages]

    def _invoke_chatbot(self, state: AgentState) -> Dict[str, List]:
        """Internal chatbot node implementation."""
        message = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    def _build_graph(self) -> CompiledStateGraph:
        """Build LangGraph with automatic state persistence."""
        graph_builder = StateGraph(AgentState)

        # Nodes
        graph_builder.add_node("chatbot", self._invoke_chatbot)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        # Edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        return graph_builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _create_config(callbacks: List) -> dict:
        return {
            "configurable": {"thread_id": str(uuid.uuid4())[:8]},
            "callbacks": callbacks
        }


# TODO: Move to a better place
class AgentFactory:
    @staticmethod
    def create_reference_agent(callbacks: List) -> BaseAgent:
        return Agent(
            llm=ChatOpenAI(model="gpt-4"),
            tools=[TavilySearch(max_results=2), assess_disruption_risk, check_supplier_compliance],
            callbacks=callbacks
        )

    @staticmethod
    def langchain_to_base_message(langchain_message: LangchainBaseMessage) -> BaseMessage:
        if isinstance(langchain_message, AIMessage):
            return BaseMessage(message_type=BaseMessageType.AiMessage, content=str(langchain_message.content))
        elif isinstance(langchain_message, HumanMessage):
            return BaseMessage(message_type=BaseMessageType.HumanMessage, content=str(langchain_message.content))
        else:
            raise NotImplementedError(f"Unexpected message type: {type(langchain_message)}")
