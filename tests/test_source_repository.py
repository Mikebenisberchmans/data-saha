from __future__ import annotations

import pytest

from app.sources.models import DataSourceCreate, DataSourceUpdate, ProviderType
from app.sources.repository import SourceNotFoundError
from app.dependencies import get_source_repository


def _create_input(display_name: str, provider: ProviderType, pat: str = "secret-pat") -> DataSourceCreate:
    return DataSourceCreate(
        display_name=display_name,
        provider=provider,
        mcp_url="https://example.com/mcp",
        pat=pat,
        description="test",
        business_domain="sales",
    )


def test_create_and_get_source(isolated_env):
    repo = get_source_repository()
    created = repo.create_source(_create_input("Sales Snowflake", ProviderType.SNOWFLAKE))

    fetched = repo.get_source(created.id)
    assert fetched.id == created.id
    assert fetched.display_name == "Sales Snowflake"


def test_pat_not_written_to_sources_file(isolated_env, tmp_path):
    repo = get_source_repository()
    repo.create_source(_create_input("Sales Snowflake", ProviderType.SNOWFLAKE, pat="do-not-leak"))

    sources_file_contents = repo._sources_file.read_text(encoding="utf-8")
    assert "do-not-leak" not in sources_file_contents


def test_resolve_credential_returns_original_pat(isolated_env):
    repo = get_source_repository()
    created = repo.create_source(
        _create_input("Sales Snowflake", ProviderType.SNOWFLAKE, pat="the-real-pat")
    )
    assert repo.resolve_credential(created.id) == "the-real-pat"


def test_multiple_sources_same_provider_coexist(isolated_env):
    repo = get_source_repository()
    sales = repo.create_source(_create_input("Sales Snowflake", ProviderType.SNOWFLAKE, pat="pat-a"))
    finance = repo.create_source(_create_input("Finance Snowflake", ProviderType.SNOWFLAKE, pat="pat-b"))

    all_sources = repo.list_sources()
    ids = {s.id for s in all_sources}
    assert sales.id in ids and finance.id in ids
    assert repo.resolve_credential(sales.id) == "pat-a"
    assert repo.resolve_credential(finance.id) == "pat-b"


def test_multi_provider_configuration(isolated_env):
    repo = get_source_repository()
    repo.create_source(_create_input("Prod Snowflake", ProviderType.SNOWFLAKE, pat="p1"))
    repo.create_source(_create_input("Marketing Redshift", ProviderType.REDSHIFT, pat="p2"))
    repo.create_source(_create_input("Analytics BigQuery", ProviderType.BIGQUERY, pat="p3"))

    providers = {s.provider for s in repo.list_sources()}
    assert providers == {ProviderType.SNOWFLAKE, ProviderType.REDSHIFT, ProviderType.BIGQUERY}


def test_update_source_rotates_credential(isolated_env):
    repo = get_source_repository()
    created = repo.create_source(_create_input("Sales Snowflake", ProviderType.SNOWFLAKE, pat="old-pat"))

    repo.update_source(created.id, DataSourceUpdate(pat="new-pat", description="updated desc"))

    assert repo.resolve_credential(created.id) == "new-pat"
    assert repo.get_source(created.id).description == "updated desc"


def test_delete_source_removes_config_and_credential(isolated_env):
    repo = get_source_repository()
    created = repo.create_source(_create_input("Sales Snowflake", ProviderType.SNOWFLAKE, pat="pat"))
    repo.delete_source(created.id)

    with pytest.raises(SourceNotFoundError):
        repo.get_source(created.id)
    with pytest.raises(Exception):
        repo.resolve_credential(created.id)


def test_get_missing_source_raises(isolated_env):
    repo = get_source_repository()
    with pytest.raises(SourceNotFoundError):
        repo.get_source("does-not-exist")


def test_enabled_only_filter(isolated_env):
    repo = get_source_repository()
    a = repo.create_source(_create_input("A", ProviderType.SNOWFLAKE, pat="pa"))
    repo.update_source(a.id, DataSourceUpdate(enabled=False))
    repo.create_source(_create_input("B", ProviderType.REDSHIFT, pat="pb"))

    enabled = repo.list_sources(enabled_only=True)
    assert len(enabled) == 1
    assert enabled[0].display_name == "B"
