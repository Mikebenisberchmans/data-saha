"""
Simple CLI test client for the conversational agent.

Phase 3 change: uses a stable per-user session id (not a fresh random one
per run) and app.agent.runner.run_turn, so the conversation — including
the rolling summary once triggered — persists across restarts. Delete the
corresponding file under data/conversations/ to start fresh.

Usage:
    python -m scripts.chat_cli
"""

from __future__ import annotations

from app.agent.runner import run_turn
from app.dependencies import get_default_session_id


def main() -> None:
    session_id = get_default_session_id()
    print(f"Session: {session_id}  (Ctrl+C to quit)")
    print("This conversation persists across restarts.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not user_input:
            continue

        reply = run_turn(session_id, user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()