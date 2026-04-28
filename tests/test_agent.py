"""Tests for the agent tool and memory layers (no model required)."""

from __future__ import annotations

import pytest

from omni_agentic.agents.memory import AgentMemory, LongTermMemory, ShortTermMemory
from omni_agentic.agents.tools import Tool, ToolRegistry, ToolResult, tool


# ------------------------------------------------------------------ #
# Tool & ToolResult                                                   #
# ------------------------------------------------------------------ #


class TestTool:
    def test_basic_call(self):
        t = Tool(name="add", description="Add two numbers", fn=lambda a, b: a + b)
        result = t(a=1, b=2)
        assert result.ok
        assert result.output == 3
        assert result.tool_name == "add"

    def test_error_captured(self):
        def boom(**_):
            raise ValueError("oops")

        t = Tool(name="boom", description="Explodes", fn=boom)
        result = t()
        assert not result.ok
        assert "oops" in result.error

    def test_schema(self):
        t = Tool(
            name="greet",
            description="Say hello",
            fn=lambda name: f"Hello {name}",
            parameters={"name": "str"},
        )
        schema = t.schema()
        assert schema["name"] == "greet"
        assert "name" in schema["parameters"]["properties"]

    def test_tool_result_str_ok(self):
        r = ToolResult(tool_name="x", output=42)
        assert "42" in str(r)

    def test_tool_result_str_error(self):
        r = ToolResult(tool_name="x", output=None, error="boom")
        assert "boom" in str(r)


class TestToolDecorator:
    def test_decorator_creates_tool(self):
        @tool(description="Multiply two integers")
        def multiply(a: int, b: int) -> int:
            return a * b

        assert isinstance(multiply, Tool)
        assert multiply.name == "multiply"
        assert multiply.description == "Multiply two integers"

    def test_decorator_execution(self):
        @tool(description="Double a value")
        def double(x: int) -> int:
            return x * 2

        result = double(x=5)
        assert result.ok
        assert result.output == 10

    def test_decorator_uses_fn_name(self):
        @tool()
        def my_func():
            pass

        assert my_func.name == "my_func"

    def test_decorator_uses_docstring_when_no_description(self):
        @tool()
        def documented():
            """Does something useful."""

        assert "useful" in documented.description


class TestToolRegistry:
    def test_register_and_call(self):
        registry = ToolRegistry()
        t = Tool(name="add", description="Add", fn=lambda a, b: int(a) + int(b))
        registry.register(t)
        result = registry.call("add", a="3", b="4")
        assert result.ok
        assert result.output == 7

    def test_call_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.call("nonexistent")
        assert not result.ok
        assert "Unknown tool" in result.error

    def test_schemas(self):
        registry = ToolRegistry()
        registry.register(Tool(name="t1", description="d1", fn=lambda: None))
        registry.register(Tool(name="t2", description="d2", fn=lambda: None))
        schemas = registry.schemas()
        names = {s["name"] for s in schemas}
        assert names == {"t1", "t2"}

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(Tool(name="x", description="", fn=lambda: None))
        assert "x" in registry
        assert "y" not in registry

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(Tool(name="x", description="", fn=lambda: None))
        assert len(registry) == 1


# ------------------------------------------------------------------ #
# Memory                                                              #
# ------------------------------------------------------------------ #


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        messages = mem.messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content == "Hi there"

    def test_max_messages_eviction(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        # Should have at most 3 messages
        assert len(mem) <= 3

    def test_clear_preserves_system(self):
        mem = ShortTermMemory()
        mem.add("system", "System instructions")
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        mem.clear()
        remaining = mem.messages()
        assert all(m.role == "system" for m in remaining)

    def test_as_dicts(self):
        mem = ShortTermMemory()
        mem.add("user", "test")
        dicts = mem.as_dicts()
        assert dicts[0] == {"role": "user", "content": "test"}


class TestLongTermMemory:
    def test_remember_and_recall(self):
        mem = LongTermMemory()
        mem.remember("name", "Alice")
        assert mem.recall("name") == "Alice"

    def test_recall_default(self):
        mem = LongTermMemory()
        assert mem.recall("missing", default="fallback") == "fallback"

    def test_forget(self):
        mem = LongTermMemory()
        mem.remember("key", "value")
        mem.forget("key")
        assert "key" not in mem

    def test_keys(self):
        mem = LongTermMemory()
        mem.remember("a", 1)
        mem.remember("b", 2)
        assert set(mem.keys()) == {"a", "b"}


class TestAgentMemory:
    def test_combined_interface(self):
        mem = AgentMemory()
        mem.add_message("user", "Hello")
        mem.remember("fact", "sky is blue")
        assert len(mem.messages()) == 1
        assert mem.recall("fact") == "sky is blue"

    def test_clear_session(self):
        mem = AgentMemory()
        mem.add_message("system", "sys")
        mem.add_message("user", "hi")
        mem.clear_session()
        remaining = mem.messages()
        assert all(m.role == "system" for m in remaining)
