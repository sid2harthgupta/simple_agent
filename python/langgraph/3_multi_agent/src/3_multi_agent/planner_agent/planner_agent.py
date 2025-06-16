# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from typing import List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage, BaseMessageType
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel


class PlannerAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, callbacks: List):
        self.llm = llm
        self.callbacks = callbacks
        self.message_history = []

    @property
    def name(self) -> str:
        return 'Planner Agent'

    @property
    def capabilities(self) -> List[str]:
        return ["Decides which sub-agent/s to be called."]

    @property
    def example_queries(self) -> List[str]:
        return ["What agent should be called for questions related to supply chain?"]

    def reset(self) -> None:
        self.message_history = []

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 1 and messages[0].message_type == BaseMessageType.HumanMessage
        message = LangGraphUtils.to_langchain_message(base_message=messages[0])
        self.message_history.append(message)
        response = self.llm.invoke([message])["messages"][-1]
        self.message_history.append(response)
        return [LangGraphUtils.to_base_message(response)]

    def get_message_history(self) -> List[BaseMessage]:
        return self.message_history

    @staticmethod
    def _get_planning_prompt(user_query: str):
        return f"""
        Analyze this user query and classify which agent(s) should handle it.

        User Query: "{user_query}"

        Available Agents:
        - supply_chain_agent: Handles supply chain management, risk assessment, supplier compliance, logistics, operational analysis. This agent can answer questions about everything that impacts supply chains including but not limited to weather and news.
        - financial_agent: Handles cost analysis, financial risk, ROI calculations, budget planning, TCO analysis
        - both_agents: Requires collaboration between both agents

        Respond with ONLY a JSON object:
        {{
            "primary_agent": "supply_chain_agent" | "financial_agent" | "both_agents",
            "reasoning": "brief explanation of routing decision"
        }}

        Examples:
        - "Check supplier compliance" → supply_chain_agent
        - "Calculate TCO" → financial_agent  
        - "Should we switch suppliers?" → both_agents
        """
