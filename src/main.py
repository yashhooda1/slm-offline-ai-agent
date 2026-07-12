"""CLI entry point.

  uv run python -m src.main "what is 15% of 1240? save it to tip.txt"
  uv run python -m src.main              # interactive REPL
  uv run python -m src.main --trace ...  # show every tool call as it happens
"""

import argparse
import sys

from src.agent import build_client, run_agent
from src.config import MAX_STEPS, MODEL


def _tracer(kind: str, text: str) -> None:
    prefix = {
        "think": "  ...",
        "tool_call": "  -> ",
        "tool_result": "  <- ",
        "final": "",
    }.get(kind, "  ")
    if kind == "final":
        return
    snippet = text if len(text) < 300 else text[:297] + "..."
    print(f"{prefix}{snippet}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="A fully offline ReAct agent on a local SLM.")
    parser.add_argument("task", nargs="*", help="The task. Omit for an interactive REPL.")
    parser.add_argument("--trace", action="store_true", help="Print each tool call and result.")
    args = parser.parse_args()

    try:
        client = build_client()
    except Exception as exc:
        print(f"Could not reach Ollama: {exc}\nIs `ollama serve` running?", file=sys.stderr)
        return 1

    on_event = _tracer if args.trace else None

    if args.task:
        print(run_agent(" ".join(args.task), client=client, on_event=on_event))
        return 0

    print(f"Offline agent - model={MODEL}, max_steps={MAX_STEPS}. Ctrl-C to quit.\n")
    while True:
        try:
            task = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        if task in {"exit", "quit"}:
            return 0
        try:
            print(f"\nagent > {run_agent(task, client=client, on_event=on_event)}\n")
        except Exception as exc:
            print(f"\nagent > error: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
