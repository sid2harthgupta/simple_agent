# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from api.base_message import BaseMessage, BaseMessageType
from langchain_core.messages import BaseMessage as LangchainBaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage


class LangGraphUtils:
    @staticmethod
    def to_base_message(langchain_message: LangchainBaseMessage) -> BaseMessage:
        if isinstance(langchain_message, AIMessage):
            return BaseMessage(message_type=BaseMessageType.AiMessage, content=str(langchain_message.content))
        elif isinstance(langchain_message, HumanMessage):
            return BaseMessage(message_type=BaseMessageType.HumanMessage, content=str(langchain_message.content))
        else:
            raise NotImplementedError(f"Unexpected message type: {type(langchain_message)}")

    @staticmethod
    def to_langchain_message(base_message: BaseMessage) -> LangchainBaseMessage:
        if base_message.message_type == BaseMessageType.AiMessage:
            return AIMessage(content=base_message.content)
        elif base_message.message_type == BaseMessageType.HumanMessage:
            return HumanMessage(content=base_message.content)
        elif base_message.message_type == BaseMessageType.SystemMessage:
            return SystemMessage(content=base_message.content)
        elif base_message.message_type == BaseMessageType.ToolMessage:
            return ToolMessage(content=base_message.content)
        else:
            raise NotImplementedError(f"Unexpected message type: {base_message.message_type}")
