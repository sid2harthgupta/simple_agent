# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

import uuid
import warnings
from typing import Annotated, Dict, List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage, BaseMessageType
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from pinecone_retrieval_tool import PineconeRetrievalTool
from tools import check_supplier_compliance, assess_disruption_risk

warnings.filterwarnings("ignore", category=SyntaxWarning, module="langchain_tavily")


class SupplyChainAgentState(TypedDict):
    """Agent state with automatic message accumulation."""
    messages: Annotated[List, add_messages]


class SupplyChainAgent(BaseAgent):
    """LangGraph wrapper for stateful agents with tool support and conversation memory.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_tavily import TavilySearch
        >>> from galileo.handlers.langchain import GalileoCallback
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [TavilySearch(max_results=3)]
        >>> callbacks = [GalileoCallback()]
        >>> agent = SupplyChainAgent(llm, callbacks)
        >>> response = agent.invoke("What are supply chain risks?")
    """
    _tools: List[BaseTool] = [PineconeRetrievalTool, assess_disruption_risk, check_supplier_compliance]

    def __init__(self, llm: BaseChatModel, callbacks: List) -> None:
        """Initialize agent with LangGraph-managed state.

        Args:
            llm: Chat model for generating responses.
            callbacks: Monitoring/logging callbacks.
        """
        self.tools = SupplyChainAgent._tools
        self.llm_with_tools = llm.bind_tools(SupplyChainAgent._tools)
        self.callbacks = callbacks
        self.config = self._create_config(callbacks)

        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return "Supply Chain Agent"

    @property
    def capabilities(self) -> List[str]:
        return ["Web search", "Tool calling", "Memory", "RAG"]

    @property
    def example_queries(self) -> List[str]:
        return [
            "Give me a short intro to the fundamentals of supply chain.",
            "What is the compliance status for SUP001?",
            "Who has won the most men's singles tennis matches?",
            "What is the square root of 34 multiplied by 5?",
        ]

    def reset(self) -> None:
        self.config = self._create_config(self.callbacks)

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 1 and messages[0].message_type == BaseMessageType.HumanMessage
        result = self.graph.invoke(
            {"messages": [LangGraphUtils.to_langchain_message(messages[0])]},
            config=self.config
        )["messages"][-1]
        return [LangGraphUtils.to_base_message(result)]

    def get_message_history(self) -> List[BaseMessage]:
        """Get current conversation history."""
        current_state = self.graph.get_state(config=self.config)
        messages = [m for m in current_state.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
        return [LangGraphUtils.to_base_message(message) for message in messages]

    def _invoke_agent(self, state: SupplyChainAgentState) -> Dict[str, List]:
        """Internal chatbot node implementation."""
        message = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    def _build_graph(self) -> CompiledStateGraph:
        """Build LangGraph with automatic state persistence."""
        graph_builder = StateGraph(SupplyChainAgentState)

        # Nodes
        graph_builder.add_node("chatbot", self._invoke_agent)
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
