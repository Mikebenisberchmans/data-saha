from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.llm.groq_client import reset_groq_client


@pytest.fixture
def client(isolated_env, monkeypatch):
    """A TestClient wired to a fully isolated app instance: isolated data
    dir (from isolated_env), a fake Groq key/model so GroqClient
    construction succeeds, and a fresh app import so it doesn't share
    cached FastAPI internals with other tests."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()

    from app.agent.graph import reset_graph
    reset_graph()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    reset_groq_client()
    reset_graph()


# --- health / me --------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_me(client):
    resp = client.get("/me")
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Mike"


# --- sources CRUD ---------------------------------------------------------------


def test_list_sources_empty_initially(client):
    resp = client.get("/sources")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_source_never_returns_pat(client):
    resp = client.post(
        "/sources",
        json={
            "display_name": "Test Source",
            "provider": "generic_mcp",
            "mcp_url": "https://example.com/mcp",
            "pat": "super-secret-pat-value",
            "description": "test",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "pat" not in body
    assert "super-secret-pat-value" not in json.dumps(body)
    assert body["id"].startswith("generic_mcp-")


def test_create_then_list_source(client):
    client.post(
        "/sources",
        json={
            "display_name": "Sales Snowflake",
            "provider": "snowflake",
            "mcp_url": "https://example.com/sales",
            "pat": "fake-pat",
            "description": "sales data",
            "business_domain": "sales",
        },
    )
    resp = client.get("/sources")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["display_name"] == "Sales Snowflake"


def test_delete_source(client):
    created = client.post(
        "/sources",
        json={
            "display_name": "Temp Source",
            "provider": "generic_mcp",
            "mcp_url": "https://example.com/mcp",
            "pat": "fake-pat",
        },
    ).json()

    resp = client.delete(f"/sources/{created['id']}")
    assert resp.status_code == 204

    resp = client.get("/sources")
    assert resp.json() == []


def test_delete_nonexistent_source_returns_404(client):
    resp = client.delete("/sources/does-not-exist")
    assert resp.status_code == 404


def test_source_health_check_for_unreachable_source(client):
    created = client.post(
        "/sources",
        json={
            "display_name": "Unreachable Source",
            "provider": "generic_mcp",
            "mcp_url": "https://127.0.0.1:1/mcp",  # nothing listens here
            "pat": "fake-pat",
        },
    ).json()

    resp = client.get(f"/sources/{created['id']}/health")
    assert resp.status_code == 200
    assert resp.json()["healthy"] is False


def test_source_health_check_for_missing_source(client):
    resp = client.get("/sources/does-not-exist/health")
    assert resp.status_code == 200
    assert resp.json()["healthy"] is False


# --- chat -----------------------------------------------------------------------


def test_chat_returns_reply(client):
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content="Hello Mike!", tool_calls=None))
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        resp = client.post("/chat", json={"message": "Hi there"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Hello Mike!"
    assert body["session_id"] == "local-user-main"
    assert body["sources_used"] == []
    assert body["dashboard_available"] is False
    assert body["report_available"] is False


def test_chat_persists_and_is_retrievable_via_conversation_endpoint(client):
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content="Got it.", tool_calls=None))
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        chat_resp = client.post("/chat", json={"message": "Remember this"})
        session_id = chat_resp.json()["session_id"]

        conv_resp = client.get(f"/conversation/{session_id}")

    assert conv_resp.status_code == 200
    messages = conv_resp.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == "Remember this"
    assert messages[1]["content"] == "Got it."


def test_chat_accepts_explicit_session_id(client):
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content="ok", tool_calls=None))
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        resp = client.post(
            "/chat", json={"session_id": "custom-session", "message": "hi"}
        )

    assert resp.json()["session_id"] == "custom-session"


def test_conversation_endpoint_for_unknown_session_returns_empty_record(client):
    resp = client.get("/conversation/never-seen-before")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


# --- dashboard / report -----------------------------------------------------------


def test_dashboard_endpoint_no_sources_configured(client):
    with patch("app.llm.groq_client.Groq"):
        resp = client.post("/dashboard", json={"question": "Show me sales"})

    assert resp.status_code == 200
    assert resp.json()["specification"]["title"] == "No data available"


def test_report_endpoint_json_no_sources_configured(client):
    with patch("app.llm.groq_client.Groq"):
        resp = client.post(
            "/report", json={"question": "Write a report", "format": "json"}
        )

    assert resp.status_code == 200
    assert resp.json()["specification"]["title"] == "No data available"


def test_report_endpoint_pdf_no_sources_configured(client):
    """Even the 'no data' fallback spec must render a real PDF, not error
    out — matches test_generate_report_pdf_handles_no_data_specification
    in test_report_generator.py, exercised here through the actual route."""
    with patch("app.llm.groq_client.Groq"):
        resp = client.post(
            "/report", json={"question": "Write a report", "format": "pdf"}
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"