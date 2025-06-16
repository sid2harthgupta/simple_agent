# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

import uuid
from typing import TypedDict, Annotated, List, Dict

from api.base_agent import BaseAgent
from api.base_message import BaseMessage
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from financial_agent_tools import calculate_tco, analyze_financial_risk, compare_supplier_costs

FINANCIAL_TOOLS = [calculate_tco, analyze_financial_risk, compare_supplier_costs]

class FinancialAgentState(TypedDict):
    """Agent state with automatic message accumulation."""
    messages: Annotated[List, add_messages]


class FinancialAgent(BaseAgent):

    def __init__(
            self,
            llm: BaseChatModel,
            callbacks: List
    ) -> None:
        """Initialize agent with LangGraph-managed state.

        Args:
            llm: Chat model for generating responses.
            callbacks: Monitoring/logging callbacks.
        """
        self.tools = FINANCIAL_TOOLS
        self.llm_with_tools = llm.bind_tools(FINANCIAL_TOOLS)
        self.callbacks = callbacks
        self.config = self._create_config(callbacks)

        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return "Financial Agent"

    @property
    def capabilities(self) -> List[str]:
        return ["Tool calls"]

    @property
    def example_queries(self) -> List[str]:
        return [
            "What is the total cost of ownership for SUP001 for a volume of 10000 and unit price $0.5?",
            "What is the financial risk for SUP001?",
            "Compare costs for SUP001 and SUP002 assuming SUP001 is in Mexico and SUP002 is in China."
        ]

    def reset(self) -> None:
        self.config = self._create_config(self.callbacks)

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 1
        result = self.graph.invoke(
            {"messages": [LangGraphUtils.to_langchain_message(messages[0])]},
            config=self.config
        )
        return [LangGraphUtils.to_base_message(result["messages"][-1])]

    def get_message_history(self) -> List[BaseMessage]:
        current_state = self.graph.get_state(config=self.config)
        messages = [m for m in current_state.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
        return [LangGraphUtils.to_base_message(message) for message in messages]

    def _invoke_chatbot(self, state: FinancialAgentState) -> Dict[str, List]:
        """Internal chatbot node implementation."""
        message = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    def _build_graph(self) -> CompiledStateGraph:
        """Build LangGraph with automatic state persistence."""
        graph_builder = StateGraph(FinancialAgentState)
        graph_builder.add_node("financial_chatbot", self._invoke_chatbot)

        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        graph_builder.add_conditional_edges("financial_chatbot", tools_condition)
        graph_builder.add_edge("tools", "financial_chatbot")
        graph_builder.add_edge(START, "financial_chatbot")

        return graph_builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _create_config(callbacks: List) -> dict:
        return {
            "configurable": {"thread_id": str(uuid.uuid4())[:8]},
            "callbacks": callbacks
        }
