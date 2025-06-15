# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

import enum
from dataclasses import dataclass


class BaseMessageType(enum.Enum):
    """Framework agnostic message types."""
    AiMessage = 1
    HumanMessage = 2
    SystemMessage = 3
    ToolMessage = 4


@dataclass
class BaseMessage:
    """Framework agnostic message data."""
    message_type: BaseMessageType
    content: str
