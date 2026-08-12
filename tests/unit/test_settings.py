"""Unit tests for application settings — no infrastructure required."""

import pytest

from travel_ai_search.config.settings import Settings, get_settings


def test_default_opensearch_host() -> None:
    assert Settings().opensearch_host == "localhost"


def test_default_opensearch_port() -> None:
    assert Settings().opensearch_port == 9200


def test_default_ssl_disabled() -> None:
    s = Settings()
    assert s.opensearch_use_ssl is False
    assert s.opensearch_verify_certs is False


def test_default_environment() -> None:
    assert Settings().environment == "development"


def test_env_var_overrides_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_HOST", "my-cluster.example.com")
    s = Settings()
    assert s.opensearch_host == "my-cluster.example.com"


def test_env_var_overrides_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_PORT", "9300")
    s = Settings()
    assert s.opensearch_port == 9300


def test_env_var_enables_ssl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_USE_SSL", "true")
    s = Settings()
    assert s.opensearch_use_ssl is True


def test_default_index_name() -> None:
    assert Settings().opensearch_index_name == "travel_hotels"


def test_env_var_overrides_index_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_INDEX_NAME", "travel_hotels_staging")
    s = Settings()
    assert s.opensearch_index_name == "travel_hotels_staging"


def test_get_settings_returns_same_instance() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
