"""Tool registry: define, register and call agent tools."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """The result of executing a tool."""

    tool_name: str
    output: Any
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if self.error:
            return f"[Tool error] {self.tool_name}: {self.error}"
        return f"[Tool result] {self.tool_name}: {self.output}"


@dataclass
class Tool:
    """A callable tool that an agent can invoke."""

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments."""
        try:
            result = self.fn(**kwargs)
            return ToolResult(tool_name=self.name, output=result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %r raised: %s", self.name, exc)
            return ToolResult(tool_name=self.name, output=None, error=str(exc))

    def schema(self) -> Dict[str, Any]:
        """Return a JSON-schema-compatible description of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in self.parameters.items()
                },
            },
        }


# ------------------------------------------------------------------ #
# @tool decorator                                                     #
# ------------------------------------------------------------------ #


def tool(
    name: Optional[str] = None,
    description: str = "",
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that converts a plain function into a :class:`Tool`.

    Usage::

        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(a=1, b=2)   # ToolResult
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        doc = description or (inspect.getdoc(fn) or "")

        # Build parameter descriptions from type hints / annotations
        sig = inspect.signature(fn)
        params: Dict[str, str] = {}
        hints = fn.__annotations__
        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            hint = hints.get(param_name, "")
            type_name = hint.__name__ if isinstance(hint, type) else str(hint)
            params[param_name] = type_name

        return Tool(name=tool_name, description=doc, fn=fn, parameters=params)

    return decorator


# ------------------------------------------------------------------ #
# ToolRegistry                                                        #
# ------------------------------------------------------------------ #


class ToolRegistry:
    """Keeps a named collection of :class:`Tool` objects."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        t = self.get(name)
        if t is None:
            return ToolResult(
                tool_name=name, output=None, error=f"Unknown tool: {name!r}"
            )
        return t(**kwargs)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({sorted(self._tools.keys())})"
