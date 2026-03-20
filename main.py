"""
Orion — minimal CLI for testing the agent locally.

Usage:
    uv run python main.py
    uv run python main.py --session my-test-1
"""

import argparse
import uuid

from dotenv import load_dotenv

load_dotenv()

from agent.graph import graph  # noqa: E402 — load_dotenv must run first


def run(session_id: str) -> None:
    config = {"configurable": {"thread_id": session_id}}
    print(f"Orion agent ready. Session: {session_id}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input or user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        response = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"\nOrion: {response['messages'][-1].content}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="orion", description="Orion customer support agent.")
    parser.add_argument("--session", default=str(uuid.uuid4()), help="Session ID for conversation memory.")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
