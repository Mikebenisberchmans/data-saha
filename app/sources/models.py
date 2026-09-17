"""
Data source and user profile models.

Critical invariant: `DataSourceConfig` — the object that gets persisted to
disk, returned from the API, put into LangGraph state, and referenced in
prompts — NEVER contains a secret field. It only contains `credential_ref`,
an opaque, non-secret lookup key into a CredentialStore.

`DataSourceCreate` is the only model allowed to carry a raw `pat`, and it
exists solely as API/CLI input; the repository layer (app/sources/
repository.py) consumes it, writes the secret into the CredentialStore, and
immediately discards it in favor of a `DataSourceConfig`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ProviderType(str, Enum):
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    BIGQUERY = "bigquery"
    DATABRICKS = "databricks"
    FABRIC = "fabric"
    GENERIC_MCP = "generic_mcp"


def _new_source_id(provider: ProviderType) -> str:
    # Human-glanceable but still unique: <provider>-<short uuid>.
    # Never derived from display_name, so "Sales Snowflake" and
    # "Finance Snowflake" can never collide or be confused with each other.
    return f"{provider.value}-{uuid.uuid4().hex[:8]}"


class DataSourceBase(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=200)
    provider: ProviderType
    mcp_url: str = Field(..., min_length=1)
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Free-text description used by the source selector, "
        "e.g. 'Contains CRM opportunities and sales activity'.",
    )
    business_domain: str | None = Field(
        default=None,
        max_length=100,
        description="e.g. 'sales', 'finance', 'marketing' — used as a hint "
        "for source selection, not a hard filter.",
    )
    enabled: bool = True


class DataSourceCreate(DataSourceBase):
    """API/CLI input model. `pat` is write-only: it is consumed by the
    repository and routed into a CredentialStore, and is never stored in
    this shape or echoed back."""

    pat: str = Field(..., min_length=1, repr=False)

    @field_validator("pat")
    @classmethod
    def _pat_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pat must not be blank")
        return v


class DataSourceUpdate(BaseModel):
    """Partial update. `pat`, if provided, rotates the stored credential;
    all other fields patch the public config."""

    display_name: str | None = None
    description: str | None = None
    business_domain: str | None = None
    enabled: bool | None = None
    mcp_url: str | None = None
    pat: str | None = Field(default=None, repr=False)


class DataSourceConfig(DataSourceBase):
    """The persisted, public representation of a configured data source.
    Safe to log, return from the API, store in LangGraph state, and
    reference in LLM prompts — it never contains a secret."""

    id: str
    credential_ref: str = Field(
        ...,
        description="Opaque key into a CredentialStore. Not a secret itself.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_create(cls, data: DataSourceCreate) -> "DataSourceConfig":
        source_id = _new_source_id(data.provider)
        return cls(
            id=source_id,
            credential_ref=source_id,  # 1:1 by default; kept as a distinct
            # field rather than reusing `id` directly in calling code, so
            # the credential layer could later be repointed without an ID
            # migration.
            display_name=data.display_name,
            provider=data.provider,
            mcp_url=data.mcp_url,
            description=data.description,
            business_domain=data.business_domain,
            enabled=data.enabled,
        )

    def selector_context(self) -> dict:
        """The subset of fields the source-selection LLM node is allowed to
        see. Explicitly excludes mcp_url and credential_ref to keep the
        selection prompt minimal and to avoid putting connection details in
        front of the model unnecessarily (added here so future callers have
        an obvious, safe method to reach for instead of hand-picking
        fields)."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "description": self.description,
            "business_domain": self.business_domain,
            "enabled": self.enabled,
        }


class UserProfile(BaseModel):
    user_id: str
    display_name: str
    timezone: str = "UTC"
    preferred_language: str = "en"
    configured_data_sources: list[str] = Field(
        default_factory=list,
        description="List of DataSourceConfig.id values owned by this user. "
        "Full source configs live in the SourceRepository, not embedded "
        "here, so there is a single source of truth.",
    )

    def system_context(self) -> dict:
        """Fields injected into the agent's system context so the model can
        address the user naturally (e.g. 'Mike, your 2025 revenue...')
        without every prompt manually re-adding the name."""
        return {
            "user_display_name": self.display_name,
            "user_timezone": self.timezone,
            "preferred_language": self.preferred_language,
        }
