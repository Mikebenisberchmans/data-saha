"""
Simple CLI test client for the conversational agent.

Usage:
    python -m scripts.chat_cli
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from app.agent.graph import get_graph


def main() -> None:
    graph = get_graph()
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}  (Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": session_id}},
        )
        print(f"Agent: {result['last_response']}\n")


if __name__ == "__main__":
    main()