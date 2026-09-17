from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from app.core.credentials import (
    CredentialNotFoundError,
    EnvCredentialStore,
    LocalFileCredentialStore,
    build_credential_store,
)
from app.config import reload_settings


def test_env_credential_store_roundtrip():
    store = EnvCredentialStore()
    store.set("snowflake-prod", "super-secret-pat")
    assert store.exists("snowflake-prod")
    assert store.get("snowflake-prod") == "super-secret-pat"
    store.delete("snowflake-prod")
    assert not store.exists("snowflake-prod")


def test_env_credential_store_missing_raises():
    store = EnvCredentialStore()
    with pytest.raises(CredentialNotFoundError):
        store.get("does-not-exist")


def test_local_file_credential_store_roundtrip(tmp_path):
    key = Fernet.generate_key().decode()
    store = LocalFileCredentialStore(
        file_path=tmp_path / "creds.local", encryption_key=key
    )
    store.set("redshift-marketing", "another-secret")
    assert store.get("redshift-marketing") == "another-secret"

    # File on disk must not contain the plaintext secret.
    raw_bytes = (tmp_path / "creds.local").read_bytes()
    assert b"another-secret" not in raw_bytes


def test_local_file_credential_store_missing_raises(tmp_path):
    key = Fernet.generate_key().decode()
    store = LocalFileCredentialStore(
        file_path=tmp_path / "creds.local", encryption_key=key
    )
    with pytest.raises(CredentialNotFoundError):
        store.get("nope")


def test_build_credential_store_env_backend(isolated_env):
    settings = reload_settings()
    assert settings.credential_store_backend == "env"
    store = build_credential_store()
    assert isinstance(store, EnvCredentialStore)


def test_build_credential_store_local_file_requires_key(isolated_env, monkeypatch):
    monkeypatch.setenv("CREDENTIAL_STORE_BACKEND", "local_file")
    monkeypatch.delenv("LOCAL_CREDENTIAL_STORE_KEY", raising=False)
    reload_settings()
    with pytest.raises(ValueError):
        build_credential_store()
