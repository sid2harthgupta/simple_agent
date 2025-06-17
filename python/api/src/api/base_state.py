# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from typing import TypedDict, List

from api.base_message import BaseMessage


class BaseState(TypedDict):
    messages: List[BaseMessage]


class BaseStateFactory:
    @staticmethod
    def from_message(message: BaseMessage) -> BaseState:
        return BaseStateFactory.from_messages([message])

    @staticmethod
    def from_messages(messages: List[BaseMessage]) -> BaseState:
        return BaseState(messages=messages)
