from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(isolated_env):
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_preflight_allows_tauri_and_vite_origins(client):
    for origin in ("http://tauri.localhost", "tauri://localhost", "http://localhost:1420"):
        resp = client.options(
            "/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200, origin
        assert resp.headers["access-control-allow-origin"] == origin


def test_preflight_rejects_unlisted_origin(client):
    resp = client.options(
        "/chat",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in resp.headers


def test_content_disposition_is_exposed_for_pdf_downloads(client):
    resp = client.get("/health", headers={"Origin": "http://tauri.localhost"})
    exposed = resp.headers.get("access-control-expose-headers", "")
    assert "Content-Disposition" in exposed