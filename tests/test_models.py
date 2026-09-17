from __future__ import annotations

from app.sources.models import DataSourceConfig, DataSourceCreate, ProviderType, UserProfile


def _create(display_name: str, provider: ProviderType = ProviderType.SNOWFLAKE) -> DataSourceCreate:
    return DataSourceCreate(
        display_name=display_name,
        provider=provider,
        mcp_url="https://example.com/mcp",
        pat="raw-secret-value",
        description="test source",
        business_domain="sales",
    )


def test_data_source_config_never_serializes_pat():
    config = DataSourceConfig.from_create(_create("Sales Snowflake"))
    dumped = config.model_dump()
    assert "pat" not in dumped
    assert "raw-secret-value" not in config.model_dump_json()


def test_multiple_instances_of_same_provider_get_distinct_ids():
    sales = DataSourceConfig.from_create(_create("Sales Snowflake"))
    finance = DataSourceConfig.from_create(_create("Finance Snowflake"))
    assert sales.id != finance.id
    assert sales.provider == finance.provider == ProviderType.SNOWFLAKE
    assert sales.display_name != finance.display_name


def test_source_id_not_derived_from_display_name():
    a = DataSourceConfig.from_create(_create("Sales Snowflake"))
    b = DataSourceConfig.from_create(_create("Sales Snowflake"))
    # Same display name, still must not collide.
    assert a.id != b.id


def test_selector_context_excludes_connection_details():
    config = DataSourceConfig.from_create(_create("Sales Snowflake"))
    ctx = config.selector_context()
    assert "mcp_url" not in ctx
    assert "credential_ref" not in ctx
    assert ctx["display_name"] == "Sales Snowflake"


def test_user_profile_system_context():
    profile = UserProfile(
        user_id="local-user",
        display_name="Mike",
        timezone="Asia/Kolkata",
        preferred_language="en",
    )
    ctx = profile.system_context()
    assert ctx["user_display_name"] == "Mike"
    assert ctx["user_timezone"] == "Asia/Kolkata"
