# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from abc import ABC, abstractmethod
from typing import List

from api.base_message import BaseMessage


class BaseAgent(ABC):
    """Abstract base class for agents with state management.

    This interface defines the core contract that all agent implementations
    must follow, regardless of the underlying framework (LangGraph, LangChain, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the agent."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """The list of capabilities this agent supports."""
        pass

    @property
    @abstractmethod
    def example_queries(self) -> List[str]:
        """A list of suggested example queries."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the state of the agent."""
        pass

    @abstractmethod
    def invoke(self, user_message: str) -> str:
        """Process user message and return agent response.

        Args:
            user_message: User's input message.

        Returns:
            Agent's response as string.
        """
        pass

    @abstractmethod
    def get_message_history(self) -> List[BaseMessage]:
        """Retrieve current conversation history.

        Returns:
            List of messages in chronological order.
        """
        pass
