# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.
from typing import List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class ResponseSynthesisAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, callbacks: List):
        self.llm = llm
        self.callbacks = callbacks
        self.messages: List[BaseMessage] = []

    @property
    def name(self) -> str:
        return "Response Synthesis Agent"

    @property
    def capabilities(self) -> List[str]:
        return ["Provides a coherent summary of supply chain and financial analysis."]

    @property
    def example_queries(self) -> List[str]:
        return []

    def reset(self) -> None:
        pass

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 2 or len(messages) == 3
        if len(messages) == 2:
            self.messages.append(messages[1])
            return [messages[1]]
        human_message = messages[0]
        supply_chain_message = messages[1]
        financial_message = messages[2]
        response = self.llm.invoke([
            HumanMessage(content=self._get_synthesis_prompt(
                human_message.content,
                supply_chain_message.content,
                financial_message.content
            ))
        ])["messages"][-1]
        self.messages.append(response)
        return [LangGraphUtils.to_base_message(response)]

    def get_message_history(self) -> List[BaseMessage]:
        pass

    @staticmethod
    def _get_synthesis_prompt(human_message: str, supply_chain_message: str, financial_message: str) -> str:
        return f"""
        Synthesize a comprehensive response by combining insights from both specialized agents.
    
        Original User Query: "{human_message}"
    
        SUPPLY CHAIN AGENT ANALYSIS:
        {supply_chain_message}
    
        FINANCIAL AGENT ANALYSIS:
        {financial_message}
    
        Please provide a unified response that:
        1. Directly answers the user's original query
        2. Integrates insights from both agents seamlessly
        3. Highlights any trade-offs between operational and financial considerations
        4. Provides clear, actionable recommendations
        5. Is well-structured and easy to understand
    
        Do not simply concatenate the responses - create a cohesive synthesis that balances both perspectives.
        """
