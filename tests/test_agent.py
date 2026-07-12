"""Tests that run with no Ollama and no network - the model is a scripted fake."""

import pytest

from src import tools
from src.agent import run_agent
from src.config import MAX_STEPS


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}


class FakeResponse:
    def __init__(self, message):
        self.message = message


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class FakeClient:
    """Replays a scripted list of assistant turns and records what it was sent."""

    def __init__(self, script, final="done"):
        self.script = list(script)
        self.final = final
        self.seen = []

    def chat(self, model, messages, tools=None):
        self.seen.append(messages[-1])
        if tools is None:
            # A real model with no tools offered can only reply with text.
            return FakeResponse(FakeMessage(content=self.final))
        if not self.script:
            return FakeResponse(FakeMessage(content=self.final))
        return FakeResponse(self.script.pop(0))


# --- tools -------------------------------------------------------------------

def test_calculator_does_real_math():
    assert tools.calculator("0.15 * 1240") == "186.0"
    assert tools.calculator("(2 + 3) ** 2 % 7") == "4"


def test_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        tools.calculator("__import__('os').system('echo pwned')")
    with pytest.raises(ValueError):
        tools.calculator("open('/etc/passwd').read()")


def test_note_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SANDBOX_DIR", tmp_path)
    assert "Saved note" in tools.write_note("tip.txt", "186.0")
    assert tools.read_note("tip.txt") == "186.0"
    assert tools.list_notes() == "tip.txt"


def test_read_missing_note_returns_text_not_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SANDBOX_DIR", tmp_path)
    assert "No note named" in tools.read_note("nope.txt")


def test_path_traversal_is_refused():
    with pytest.raises(ValueError, match="escapes the sandbox"):
        tools.write_note("../../important_file.txt", "gotcha")
    with pytest.raises(ValueError, match="escapes the sandbox"):
        tools.read_note("/etc/passwd")


def test_word_count():
    assert tools.word_count("one two three") == "3 words, 13 characters."


# --- agent loop ---------------------------------------------------------------

def test_loop_chains_two_tools_then_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SANDBOX_DIR", tmp_path)
    client = FakeClient([
        FakeMessage(tool_calls=[call("calculator", expression="0.15 * 1240")]),
        FakeMessage(tool_calls=[call("write_note", filename="tip.txt", content="186.0")]),
        FakeMessage(content="The tip is 186.0 and I saved it to tip.txt."),
    ])
    events = []
    answer = run_agent("15% of 1240, save it", client=client, on_event=lambda k, t: events.append((k, t)))

    assert answer == "The tip is 186.0 and I saved it to tip.txt."
    assert ("tool_result", "186.0") in events
    assert (tmp_path / "tip.txt").read_text() == "186.0"


def test_tool_error_is_fed_back_to_the_model_not_raised():
    client = FakeClient([
        FakeMessage(tool_calls=[call("calculator", expression="1 / 0")]),
        FakeMessage(content="That expression divides by zero."),
    ])
    answer = run_agent("divide 1 by 0", client=client)
    assert "divides by zero" in answer
    # the model was handed the error text as a tool observation
    assert any(m.get("role") == "tool" and "Error running calculator" in m["content"] for m in client.seen)


def test_unknown_tool_name_does_not_crash():
    client = FakeClient([
        FakeMessage(tool_calls=[call("send_email", to="ceo@example.com")]),
        FakeMessage(content="I don't have that tool."),
    ])
    assert run_agent("email the CEO", client=client) == "I don't have that tool."


def test_runaway_model_is_capped_by_max_steps():
    forever = [FakeMessage(tool_calls=[call("current_datetime")]) for _ in range(MAX_STEPS + 5)]
    client = FakeClient(forever, final="Fine: the time.")
    answer = run_agent("loop forever", client=client)
    assert answer == "Fine: the time."
    # MAX_STEPS tool turns + one forced final turn
    assert len(client.seen) == MAX_STEPS + 1
