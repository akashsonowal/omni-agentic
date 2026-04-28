"""Agent memory: short-term (message history) and long-term (key-value store)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single entry in the conversation history."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """Sliding-window message history for a single agent session."""

    def __init__(self, max_messages: int = 100) -> None:
        self._messages: List[Message] = []
        self._max = max_messages

    def add(self, role: str, content: str, **metadata: Any) -> None:
        msg = Message(role=role, content=content, metadata=metadata)
        self._messages.append(msg)
        if len(self._messages) > self._max:
            # Remove oldest non-system messages first
            non_system = [i for i, m in enumerate(self._messages) if m.role != "system"]
            if non_system:
                self._messages.pop(non_system[0])

    def messages(self) -> List[Message]:
        return list(self._messages)

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def clear(self) -> None:
        system_msgs = [m for m in self._messages if m.role == "system"]
        self._messages = system_msgs

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """Simple key-value store for persistent facts between sessions.

    In production this can be backed by a vector store or database;
    here we use an in-process dict for portability.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def remember(self, key: str, value: Any) -> None:
        self._store[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def forget(self, key: str) -> None:
        self._store.pop(key, None)

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store


class AgentMemory:
    """Combined short- and long-term memory for an agent."""

    def __init__(self, max_short_term_messages: int = 100) -> None:
        self.short_term = ShortTermMemory(max_messages=max_short_term_messages)
        self.long_term = LongTermMemory()

    # Convenience delegation ------------------------------------------------

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        self.short_term.add(role, content, **metadata)

    def messages(self) -> List[Message]:
        return self.short_term.messages()

    def remember(self, key: str, value: Any) -> None:
        self.long_term.remember(key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        return self.long_term.recall(key, default)

    def clear_session(self) -> None:
        self.short_term.clear()
