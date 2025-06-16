# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from typing import List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


class TranslationAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, callbacks: List, language: str) -> None:
        self.llm = llm
        self.callbacks = callbacks
        self.message_history = []
        self.language = language

    @property
    def name(self) -> str:
        return "Translation Agent"

    @property
    def capabilities(self) -> List[str]:
        return [f"Accepts English messages and translates them to {self.language}"]

    @property
    def example_queries(self) -> List[str]:
        return [f"How do you say 'Good morning!' in {self.language}?"]

    def reset(self) -> None:
        self.message_history = []

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 1
        message = SystemMessage(content=self._get_translation_prompt(messages[0].content))
        self.message_history.append(message)
        translation_response = self.llm.invoke([message])["messages"][-1]
        self.message_history.append(translation_response)
        return [LangGraphUtils.to_base_message(translation_response)]

    def get_message_history(self) -> List[BaseMessage]:
        return self.message_history

    def _get_translation_prompt(self, english_response):
        return f"""
            Translate the following English response into {self.language}. 
            Maintain the professional tone and technical accuracy. 
            Keep any technical terms and metrics in their original form where appropriate.
            Use the appropriate script for translation.

            English Response:
            {english_response}

            Please provide only the {self.language} translation, maintaining the same structure and formatting.
        """
