# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

"""LangGraph-based conversational agent with automatic state management."""

from typing import Annotated, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent state with automatic message accumulation."""
    messages: Annotated[List, add_messages]


class Agent:
    """LangGraph wrapper for stateful agents with tool support and conversation memory.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_tavily import TavilySearch
        >>> from galileo.handlers.langchain import GalileoCallback
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [TavilySearch(max_results=3)]
        >>> callbacks = [GalileoCallback()]
        >>> agent = Agent(llm, tools, "thread_1", callbacks)
        >>> response = agent.invoke("What are supply chain risks?")
    """

    def __init__(
            self,
            llm: BaseChatModel,
            tools: List[BaseTool],
            thread_id: str,
            callbacks: List,
            history: Optional[List[BaseMessage]] = None
    ) -> None:
        """Initialize agent with LangGraph-managed state.

        Args:
            llm: Chat model for generating responses.
            tools: Available tools for the agent.
            thread_id: Unique conversation thread identifier.
            callbacks: Monitoring/logging callbacks.
            history: Optional initial message history.
        """
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools)
        self.thread_id = thread_id
        self.config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks
        }

        self.graph = self._build_graph()

        if history:
            self.graph.invoke({"messages": history}, config=self.config)

    @property
    def state(self) -> AgentState:
        """Current conversation state from LangGraph."""
        current_state = self.graph.get_state(config=self.config)
        return AgentState(messages=current_state.values.get("messages", []))

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

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get current conversation history."""
        return self.state["messages"]

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
