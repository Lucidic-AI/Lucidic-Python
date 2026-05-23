"""Framework-specific adapter modules for the v2 tool-backed dispatch chain.

Each adapter exposes two surfaces:

- ``register_*_tools(tools)`` — discover tool surfaces from the
  framework's native shape (OpenAI / Anthropic JSON specs, LangChain
  ``BaseTool`` instances) and register them via
  ``lucidicai.sdk.tools.registry.register_tool``. Buffers when no
  ``LucidicAI`` client is alive yet; the next client init drains.

- ``dispatch_*_tool_call(call, impls)`` (+ async sibling) — route one
  tool invocation through ``emit_call_through_backend`` when a
  ``MockContext`` is bound to the current execution context, else
  invoke ``impls[name]`` directly. This is the user-side dispatch
  helper for frameworks (OpenAI, Anthropic) where the framework
  doesn't own the Python function pointer.

LangChain is the exception: ``register_langchain_agent`` transparently
replaces ``tool.func`` (or ``tool._run``) with a mockable wrapper, so
the framework's own dispatch loop routes through us — no user-side
``dispatch_*`` call needed.

All adapters share the helpers in ``..registry`` (``ToolSurface``,
``_params_from_json_schema``, ``_params_from_callable``,
``_compute_source_hash``, ``register_tool``) and the transport in
``..transport`` (``emit_call_through_backend`` /
``aemit_call_through_backend``) so the on-wire shape and the
dispatch behavior are uniform across frameworks.
"""

from .anthropic import (
    adispatch_anthropic_tool_call,
    dispatch_anthropic_tool_call,
    register_anthropic_tools,
)
from .langchain import register_langchain_tools
from .openai import (
    adispatch_openai_tool_call,
    dispatch_openai_tool_call,
    register_openai_tools,
)


__all__ = [
    # OpenAI (LUC-579)
    "adispatch_openai_tool_call",
    "dispatch_openai_tool_call",
    "register_openai_tools",
    # Anthropic (LUC-580)
    "adispatch_anthropic_tool_call",
    "dispatch_anthropic_tool_call",
    "register_anthropic_tools",
    # LangChain (LUC-578)
    "register_langchain_tools",
]
