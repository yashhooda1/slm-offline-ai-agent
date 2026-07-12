"""The agent loop: think -> call tools -> observe -> repeat (ReAct), written by hand.

No framework. The loop is a `for` over MAX_STEPS. Each pass:
  1. Send the full message history + tool schemas to the local model.
  2. If the model returned no tool calls, its content is the final answer - stop.
  3. Otherwise run each requested tool locally, append the result as a 'tool' message,
     and loop again so the model can read what happened.
"""

from typing import Any, Callable, Optional

from src.config import MAX_STEPS, MODEL, OLLAMA_HOST, SYSTEM_PROMPT
from src.tools import TOOL_FUNCTIONS, TOOLS


def build_client():
    """Create an Ollama client pointed at the local server (imported lazily)."""
    from ollama import Client

    return Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()


def _as_history(message: Any) -> dict:
    """Normalize an Ollama Message object (or dict) into a plain history dict."""
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return dict(message)


def _tool_calls(message: Any) -> list:
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    return calls or []


def _call_name_and_args(call: Any) -> tuple[str, dict]:
    fn = getattr(call, "function", None) or call["function"]
    name = getattr(fn, "name", None) or fn["name"]
    args = getattr(fn, "arguments", None)
    if args is None:
        args = fn.get("arguments", {})
    return name, dict(args or {})


def _content(message: Any) -> str:
    value = getattr(message, "content", None)
    if value is None and isinstance(message, dict):
        value = message.get("content")
    return (value or "").strip()


def run_agent(
    task: str,
    client: Optional[Any] = None,
    on_event: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Run one task to completion and return the model's final answer.

    Args:
      task: The user's request, in plain English.
      client: An Ollama-compatible client. Defaults to a real local client.
      on_event: Optional callback(kind, text) for tracing - kinds are
        'think', 'tool_call', 'tool_result', 'final'.
    Returns:
      The final answer text.
    """
    client = client or build_client()
    emit = on_event or (lambda kind, text: None)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(MAX_STEPS):
        response = client.chat(model=MODEL, messages=messages, tools=TOOLS)
        message = response.message if hasattr(response, "message") else response["message"]
        messages.append(_as_history(message))

        calls = _tool_calls(message)
        if not calls:
            answer = _content(message)
            emit("final", answer)
            return answer

        thought = _content(message)
        if thought:
            emit("think", thought)

        for call in calls:
            name, args = _call_name_and_args(call)
            emit("tool_call", f"{name}({args})")

            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = f"Error: no tool named '{name}'. Available: {', '.join(TOOL_FUNCTIONS)}."
            else:
                try:
                    result = str(fn(**args))
                except Exception as exc:  # tool errors go back to the model as text
                    result = f"Error running {name}: {exc}"

            emit("tool_result", result)
            messages.append({"role": "tool", "name": name, "content": result})

    # Loop cap hit. Ask once for a final answer with no tools available.
    messages.append(
        {
            "role": "user",
            "content": "Stop calling tools. Give your final answer now using the results above.",
        }
    )
    response = client.chat(model=MODEL, messages=messages)
    message = response.message if hasattr(response, "message") else response["message"]
    answer = _content(message) or f"Stopped after {MAX_STEPS} steps without a final answer."
    emit("final", answer)
    return answer
