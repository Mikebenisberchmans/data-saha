from __future__ import annotations

from app.sources.models import UserProfile
from app.dependencies import get_user_profile_repository


def test_load_creates_default_profile_if_missing(isolated_env):
    repo = get_user_profile_repository()
    default = UserProfile(user_id="local-user", display_name="Mike", timezone="Asia/Kolkata")
    loaded = repo.load(default=default)
    assert loaded.display_name == "Mike"
    assert repo._profile_file.exists()


def test_add_and_remove_source(isolated_env):
    repo = get_user_profile_repository()
    profile = repo.load(default=UserProfile(user_id="local-user", display_name="Mike"))

    repo.add_source(profile, "snowflake-abcd1234")
    assert "snowflake-abcd1234" in profile.configured_data_sources

    reloaded = repo.load(default=profile)
    assert "snowflake-abcd1234" in reloaded.configured_data_sources

    repo.remove_source(profile, "snowflake-abcd1234")
    assert "snowflake-abcd1234" not in profile.configured_data_sources
