from __future__ import annotations

from app.dependencies import get_conversation_store
from app.memory.conversation import StoredMessage


def test_load_missing_session_returns_empty_record(isolated_env):
    store = get_conversation_store()
    record = store.load("does-not-exist-yet")
    assert record.session_id == "does-not-exist-yet"
    assert record.messages == []
    assert record.summary == ""


def test_save_and_reload_round_trip(isolated_env):
    store = get_conversation_store()
    record = store.load("session-1")
    record.messages = [
        StoredMessage(role="user", content="hi"),
        StoredMessage(role="assistant", content="hello there"),
    ]
    record.summary = "User said hi."
    store.save(record)

    reloaded = store.load("session-1")
    assert len(reloaded.messages) == 2
    assert reloaded.messages[0].role == "user"
    assert reloaded.messages[1].content == "hello there"
    assert reloaded.summary == "User said hi."


def test_sessions_are_isolated_on_disk(isolated_env):
    store = get_conversation_store()

    record_a = store.load("session-a")
    record_a.messages = [StoredMessage(role="user", content="from A")]
    store.save(record_a)

    record_b = store.load("session-b")
    assert record_b.messages == []  # unaffected by session-a's save


def test_delete_removes_persisted_record(isolated_env):
    store = get_conversation_store()
    record = store.load("temp-session")
    record.messages = [StoredMessage(role="user", content="x")]
    store.save(record)
    assert store.exists("temp-session")

    store.delete("temp-session")
    assert not store.exists("temp-session")
    # load() after delete returns a fresh empty record, not an error
    assert store.load("temp-session").messages == []