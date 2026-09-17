"""
Credential abstraction.

Design goal (see product spec section 3): PATs must never be stored in plain
text in normal application config, and must never reach logs, error
messages, API responses, LangGraph state, prompts, or LLM context.

The rest of the application only ever handles a `credential_ref` (an opaque
string key). Resolving a `credential_ref` to the actual secret happens only
at the point of use — inside the MCP connection layer (added in a later
phase) — and that resolved value must not be persisted, logged, or passed
into any object that flows into LangGraph state or an LLM prompt.

Backends:
- EnvCredentialStore: reads `MCP_PAT_<credential_ref>` from the environment.
  Good for local development and CI.
- LocalFileCredentialStore: Fernet-encrypted JSON file on disk. Also for
  local development — it is NOT a substitute for a real OS credential
  store, and is clearly labeled as such below.
- WindowsCredentialManagerStore / MacKeychainStore: stub interfaces for the
  future Tauri application. Tauri (via its Rust layer) is expected to own
  the actual OS keychain calls; these Python-side stubs exist so the
  `CredentialStore` interface already accounts for that shape, and so the
  desktop app has a documented extension point rather than needing to
  redesign this abstraction later.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path


class CredentialNotFoundError(KeyError):
    """Raised when a credential_ref cannot be resolved. Never include the
    resolved secret value in this error — only the ref (a non-secret id)."""


class CredentialStore(ABC):
    """Abstract interface for storing/retrieving per-source secrets (PATs).

    Implementations must guarantee:
    - `get` returns the raw secret string, or raises CredentialNotFoundError.
    - No method logs or echoes the secret value anywhere.
    - `__repr__`/`__str__` on the store itself never dumps secret contents.
    """

    @abstractmethod
    def get(self, credential_ref: str) -> str:
        ...

    @abstractmethod
    def set(self, credential_ref: str, secret: str) -> None:
        ...

    @abstractmethod
    def delete(self, credential_ref: str) -> None:
        ...

    @abstractmethod
    def exists(self, credential_ref: str) -> bool:
        ...

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.__class__.__name__}>"


class EnvCredentialStore(CredentialStore):
    """Reads secrets from environment variables named `MCP_PAT_<ref>`.

    Read-only by design for `set`/`delete` in most real usage (env vars set
    outside the process), but we support in-process mutation for tests and
    for a smoother local dev loop.
    """

    _PREFIX = "MCP_PAT_"

    def _env_key(self, credential_ref: str) -> str:
        return f"{self._PREFIX}{credential_ref}"

    def get(self, credential_ref: str) -> str:
        value = os.environ.get(self._env_key(credential_ref))
        if value is None:
            raise CredentialNotFoundError(credential_ref)
        return value

    def set(self, credential_ref: str, secret: str) -> None:
        os.environ[self._env_key(credential_ref)] = secret

    def delete(self, credential_ref: str) -> None:
        os.environ.pop(self._env_key(credential_ref), None)

    def exists(self, credential_ref: str) -> bool:
        return self._env_key(credential_ref) in os.environ


class LocalFileCredentialStore(CredentialStore):
    """Fernet-encrypted local file store, for local development only.

    NOT recommended for production use. Production/desktop deployments
    should use an OS-backed store (see the stub classes below), which the
    future Tauri application will provide.
    """

    def __init__(self, file_path: Path, encryption_key: str):
        from cryptography.fernet import Fernet

        self._path = file_path
        self._fernet = Fernet(
            encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        raw = self._path.read_bytes()
        if not raw:
            return {}
        decrypted = self._fernet.decrypt(raw)
        return json.loads(decrypted.decode("utf-8"))

    def _write(self, data: dict[str, str]) -> None:
        encoded = json.dumps(data).encode("utf-8")
        encrypted = self._fernet.encrypt(encoded)
        self._path.write_bytes(encrypted)

    def get(self, credential_ref: str) -> str:
        data = self._read()
        if credential_ref not in data:
            raise CredentialNotFoundError(credential_ref)
        return data[credential_ref]

    def set(self, credential_ref: str, secret: str) -> None:
        data = self._read()
        data[credential_ref] = secret
        self._write(data)

    def delete(self, credential_ref: str) -> None:
        data = self._read()
        data.pop(credential_ref, None)
        self._write(data)

    def exists(self, credential_ref: str) -> bool:
        return credential_ref in self._read()


class WindowsCredentialManagerStore(CredentialStore):
    """Reserved for the future Tauri desktop app (Windows Credential
    Manager). Not implemented on the Python side — the Tauri/Rust layer is
    expected to own this, exposing it to Python only via a secure local
    channel that has not been designed yet."""

    def get(self, credential_ref: str) -> str:
        raise NotImplementedError(
            "Windows Credential Manager integration is provided by the "
            "Tauri desktop shell, not the Python backend."
        )

    def set(self, credential_ref: str, secret: str) -> None:
        raise NotImplementedError

    def delete(self, credential_ref: str) -> None:
        raise NotImplementedError

    def exists(self, credential_ref: str) -> bool:
        raise NotImplementedError


class MacKeychainStore(CredentialStore):
    """Reserved for the future Tauri desktop app (macOS Keychain). Same
    caveat as WindowsCredentialManagerStore."""

    def get(self, credential_ref: str) -> str:
        raise NotImplementedError(
            "macOS Keychain integration is provided by the Tauri desktop "
            "shell, not the Python backend."
        )

    def set(self, credential_ref: str, secret: str) -> None:
        raise NotImplementedError

    def delete(self, credential_ref: str) -> None:
        raise NotImplementedError

    def exists(self, credential_ref: str) -> bool:
        raise NotImplementedError


def build_credential_store() -> CredentialStore:
    """Factory that builds the configured CredentialStore from settings."""
    from app.config import get_settings

    settings = get_settings()
    backend = settings.credential_store_backend

    if backend == "env":
        return EnvCredentialStore()

    if backend == "local_file":
        if not settings.local_credential_store_key:
            raise ValueError(
                "CREDENTIAL_STORE_BACKEND=local_file requires "
                "LOCAL_CREDENTIAL_STORE_KEY to be set."
            )
        return LocalFileCredentialStore(
            file_path=settings.local_credential_store_file,
            encryption_key=settings.local_credential_store_key,
        )

    if backend == "windows_credential_manager":
        return WindowsCredentialManagerStore()

    if backend == "macos_keychain":
        return MacKeychainStore()

    raise ValueError(f"Unknown credential store backend: {backend}")
