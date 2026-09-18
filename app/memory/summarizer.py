"""
Summarizer logic (product spec section 13).

Not every message is summarized — only once the accumulated conversation
exceeds SUMMARY_TRIGGER_TOKENS (an approximate token count; see
`estimate_tokens`). When triggered, everything except the most recent
RECENT_MESSAGES_KEEP messages is folded into an updated rolling summary via
a dedicated Groq call, and those older messages are dropped from the
message list — only the summary carries their content forward from then on.

Split out from app/agent/nodes/summarizer.py so this logic is directly
unit-testable without going through a full graph.invoke().
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.config import get_settings
from app.core.logging import get_logger
from app.llm.groq_client import get_groq_client

logger = get_logger(__name__)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars/token). Good enough for a trigger
    threshold; not used for anything billing-sensitive. Swap for a real
    tokenizer (e.g. tiktoken) later if more precision is needed."""
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


SUMMARIZER_SYSTEM_PROMPT = (
    "You maintain a rolling summary of an ongoing analytics conversation "
    "between a user and an AI assistant. Given the previous summary (if "
    "any) and a batch of older messages being dropped from active context, "
    "produce an updated summary. Preserve: the user's objective, important "
    "metrics and previous results discussed, selected data sources, "
    "filters, time periods, entities mentioned, prior analytical "
    "conclusions, unresolved questions, and any dashboard/report context. "
    "Be concise — this is context for the assistant, not a transcript. "
    "Output only the updated summary text, with no preamble."
)


def _messages_to_text(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "Assistant" if isinstance(m, AIMessage) else "User"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def summarize(previous_summary: str, messages_to_drop: list[BaseMessage]) -> str:
    """Calls Groq to fold `messages_to_drop` into an updated summary. Pure
    function w.r.t. its inputs (aside from the Groq call), so it's easy to
    test with a mocked client."""
    if not messages_to_drop:
        return previous_summary

    client = get_groq_client()
    user_content = (
        f"Previous summary:\n{previous_summary or '(none yet)'}\n\n"
        f"Older messages to fold in:\n{_messages_to_text(messages_to_drop)}"
    )
    response = client.chat(
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or previous_summary


def maybe_summarize(
    messages: list[BaseMessage], previous_summary: str
) -> tuple[list[BaseMessage], str]:
    """Applies the trigger-threshold check, and if it fires, summarizes and
    trims. Returns (possibly-trimmed messages, possibly-updated summary).
    Below threshold (or too few messages to usefully trim), returns the
    input unchanged."""
    settings = get_settings()
    if estimate_tokens(messages) <= settings.summary_trigger_tokens:
        return messages, previous_summary

    keep_n = settings.recent_messages_keep
    if len(messages) <= keep_n:
        return messages, previous_summary

    to_drop = messages[:-keep_n]
    to_keep = messages[-keep_n:]

    logger.info(
        "Summary trigger hit: dropping %d older messages, keeping %d recent",
        len(to_drop),
        len(to_keep),
    )
    new_summary = summarize(previous_summary, to_drop)
    return to_keep, new_summary