"""Base Agent: react-style loop with tool-calling on heterogeneous hardware."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omni_agentic.agents.memory import AgentMemory
from omni_agentic.agents.tools import Tool, ToolRegistry, ToolResult
from omni_agentic.hardware.device import Device

logger = logging.getLogger(__name__)

# Regex to detect a tool-call block in the model's output.
# Format:  <tool_call>{"name": "...", "args": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful AI assistant.
You may call tools by emitting a JSON block wrapped in <tool_call>…</tool_call> tags.
Respond in plain text when no tool is needed.

Available tools:
{tool_schemas}
"""


@dataclass
class AgentConfig:
    """Configuration for a single agent instance."""

    system_prompt: Optional[str] = None
    max_iterations: int = 10
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""

    iteration: int
    thought: str
    tool_calls: List[ToolResult] = field(default_factory=list)
    final_answer: Optional[str] = None


class Agent:
    """A ReAct-style agent that calls tools to answer user queries.

    The agent is **model-agnostic** and **hardware-agnostic**: it accepts any
    :class:`~omni_agentic.models.base.ModelBackend` and delegates hardware
    decisions to the backend/loader layer.

    Usage::

        from omni_agentic import Agent, tool
        from omni_agentic.models.loader import ModelLoader

        loader = ModelLoader.from_auto()
        backend = loader.load("Qwen/Qwen2-0.5B-Instruct")

        @tool(description="Multiply two numbers")
        def multiply(a: float, b: float) -> float:
            return float(a) * float(b)

        agent = Agent(backend=backend)
        agent.register_tool(multiply)

        answer = agent.run("What is 6 times 7?")
        print(answer)
    """

    def __init__(
        self,
        backend: Any,  # ModelBackend (avoiding circular import at runtime)
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._backend = backend
        self._config = config or AgentConfig()
        self._tools = ToolRegistry()
        self._memory = AgentMemory()
        self._trace: List[AgentStep] = []

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> Device:
        return self._backend.device

    @property
    def memory(self) -> AgentMemory:
        return self._memory

    @property
    def trace(self) -> List[AgentStep]:
        return list(self._trace)

    # ------------------------------------------------------------------ #
    # Tool management                                                     #
    # ------------------------------------------------------------------ #

    def register_tool(self, t: Tool) -> None:
        """Register a tool the agent can call."""
        self._tools.register(t)
        logger.debug("Registered tool %r", t.name)

    def register_tools(self, tools: List[Tool]) -> None:
        for t in tools:
            self.register_tool(t)

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self, query: str) -> str:
        """Run the agent on a user *query* and return the final answer."""
        self._trace = []
        self._memory.clear_session()

        system_prompt = self._config.system_prompt or self._build_system_prompt()
        self._memory.add_message("system", system_prompt)
        self._memory.add_message("user", query)

        for iteration in range(self._config.max_iterations):
            prompt = self._build_prompt()
            logger.debug("Agent iteration %d – generating …", iteration)

            raw_output = self._backend.generate(
                prompt,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
            )

            step = AgentStep(iteration=iteration, thought=raw_output)
            self._trace.append(step)

            # Check for tool calls
            tool_calls_found = self._process_tool_calls(raw_output, step)

            if tool_calls_found:
                # Append assistant turn + tool results, then continue
                self._memory.add_message("assistant", raw_output)
                for tr in step.tool_calls:
                    self._memory.add_message("tool", str(tr), tool_name=tr.tool_name)
            else:
                # No tool call: treat the output as the final answer
                step.final_answer = raw_output.strip()
                self._memory.add_message("assistant", raw_output)
                return step.final_answer

        # Exhausted iterations: return the last output as-is
        logger.warning("Agent exhausted max iterations (%d)", self._config.max_iterations)
        last_thought = self._trace[-1].thought if self._trace else ""
        return last_thought.strip()

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        schemas = json.dumps(self._tools.schemas(), indent=2)
        return SYSTEM_PROMPT_TEMPLATE.format(tool_schemas=schemas)

    def _build_prompt(self) -> str:
        """Concatenate the message history into a single prompt string."""
        parts: List[str] = []
        for msg in self._memory.messages():
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _process_tool_calls(self, text: str, step: AgentStep) -> bool:
        """Parse and execute any tool calls found in *text*.

        Returns True if at least one tool call was found.
        """
        matches = _TOOL_CALL_RE.findall(text)
        if not matches:
            return False

        for raw_json in matches:
            try:
                call = json.loads(raw_json)
                name = call.get("name", "")
                args = call.get("args", {})
            except json.JSONDecodeError as exc:
                logger.warning("Malformed tool_call JSON: %s – %s", raw_json, exc)
                step.tool_calls.append(
                    ToolResult(tool_name="?", output=None, error=f"JSON error: {exc}")
                )
                continue

            result = self._tools.call(name, **args)
            step.tool_calls.append(result)
            logger.debug("Tool %r(%s) → %s", name, args, result.output)

        return True
