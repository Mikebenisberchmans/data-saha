"""
Repository layer for data sources and the user profile.

Responsibilities:
- Persist DataSourceConfig objects (no secrets) to a local JSON file.
- On create/update, route the raw PAT to the configured CredentialStore and
  never let it touch the JSON file, logs, or the returned object.
- Provide simple CRUD + lookup used by the future API layer (Phase 11) and
  the source-selection node (Phase 5).

This is intentionally a flat-file JSON repository for Phase 1 — it matches
the "local desktop app, single user" framing in the product spec. Swapping
this for SQLite/another store later does not require changing the
DataSourceConfig/UserProfile models or the CredentialStore interface.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.credentials import CredentialStore
from app.core.logging import get_logger
from app.sources.models import (
    DataSourceConfig,
    DataSourceCreate,
    DataSourceUpdate,
    UserProfile,
)

logger = get_logger(__name__)


class SourceNotFoundError(KeyError):
    pass


class SourceRepository:
    def __init__(self, sources_file: Path, credential_store: CredentialStore):
        self._sources_file = sources_file
        self._credentials = credential_store
        self._sources_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._sources_file.exists():
            self._write_all({})

    # -- persistence helpers -------------------------------------------------

    def _read_all(self) -> dict[str, DataSourceConfig]:
        if not self._sources_file.exists():
            return {}
        raw = self._sources_file.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return {sid: DataSourceConfig.model_validate(cfg) for sid, cfg in data.items()}

    def _write_all(self, sources: dict[str, DataSourceConfig]) -> None:
        serializable = {
            sid: json.loads(cfg.model_dump_json()) for sid, cfg in sources.items()
        }
        self._sources_file.write_text(
            json.dumps(serializable, indent=2), encoding="utf-8"
        )

    # -- CRUD -----------------------------------------------------------------

    def list_sources(self, enabled_only: bool = False) -> list[DataSourceConfig]:
        sources = list(self._read_all().values())
        if enabled_only:
            sources = [s for s in sources if s.enabled]
        return sources

    def get_source(self, source_id: str) -> DataSourceConfig:
        sources = self._read_all()
        if source_id not in sources:
            raise SourceNotFoundError(source_id)
        return sources[source_id]

    def create_source(self, data: DataSourceCreate) -> DataSourceConfig:
        config = DataSourceConfig.from_create(data)

        # Route the secret to the credential store FIRST. If this fails,
        # we never write a public config that points at a missing secret.
        self._credentials.set(config.credential_ref, data.pat)

        sources = self._read_all()
        sources[config.id] = config
        self._write_all(sources)

        logger.info(
            "Created data source id=%s provider=%s display_name=%s "
            "(pat not logged)",
            config.id,
            config.provider.value,
            config.display_name,
        )
        return config

    def update_source(self, source_id: str, data: DataSourceUpdate) -> DataSourceConfig:
        sources = self._read_all()
        if source_id not in sources:
            raise SourceNotFoundError(source_id)

        existing = sources[source_id]
        patch = data.model_dump(exclude_unset=True, exclude={"pat"})
        updated = existing.model_copy(update=patch)

        if data.pat is not None:
            self._credentials.set(existing.credential_ref, data.pat)
            logger.info(
                "Rotated credential for source id=%s (pat not logged)", source_id
            )

        sources[source_id] = updated
        self._write_all(sources)
        return updated

    def delete_source(self, source_id: str) -> None:
        sources = self._read_all()
        if source_id not in sources:
            raise SourceNotFoundError(source_id)
        credential_ref = sources[source_id].credential_ref
        del sources[source_id]
        self._write_all(sources)
        self._credentials.delete(credential_ref)
        logger.info("Deleted data source id=%s", source_id)

    def resolve_credential(self, source_id: str) -> str:
        """Resolve a source's PAT. This is the ONLY place a raw secret
        should be pulled out of storage — callers (the future MCP
        connection layer) must use it immediately and must not pass the
        result into logs, prompts, LangGraph state, or API responses."""
        config = self.get_source(source_id)
        return self._credentials.get(config.credential_ref)


class UserProfileRepository:
    def __init__(self, profile_file: Path):
        self._profile_file = profile_file
        self._profile_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self, default: UserProfile) -> UserProfile:
        if not self._profile_file.exists():
            self.save(default)
            return default
        raw = self._profile_file.read_text(encoding="utf-8").strip()
        if not raw:
            self.save(default)
            return default
        return UserProfile.model_validate(json.loads(raw))

    def save(self, profile: UserProfile) -> None:
        self._profile_file.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def add_source(self, profile: UserProfile, source_id: str) -> UserProfile:
        if source_id not in profile.configured_data_sources:
            profile.configured_data_sources.append(source_id)
            self.save(profile)
        return profile

    def remove_source(self, profile: UserProfile, source_id: str) -> UserProfile:
        if source_id in profile.configured_data_sources:
            profile.configured_data_sources.remove(source_id)
            self.save(profile)
        return profile
