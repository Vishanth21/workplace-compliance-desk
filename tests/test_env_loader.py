"""Tests for app/env.py -- the .env loader.

The loader exists so that ``cp .env.example .env`` does what the README says.
It has one non-obvious rule worth pinning: it must not fire under pytest, or a
developer's ``ANALYST_MODE=real`` would put the suite on a live endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import env as env_module


def test_loads_a_file_and_sets_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_module, "_running_under_pytest", lambda: False)
    monkeypatch.delenv("COMPLIANCE_TEST_VAR", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("COMPLIANCE_TEST_VAR=from-the-file\n", encoding="utf-8")

    assert env_module.load_env(env_file) is True
    assert env_module.os.environ["COMPLIANCE_TEST_VAR"] == "from-the-file"


def test_existing_environment_wins(tmp_path: Path, monkeypatch) -> None:
    """A shell export must beat the file, or `docker compose` values break."""
    monkeypatch.setattr(env_module, "_running_under_pytest", lambda: False)
    monkeypatch.setenv("COMPLIANCE_TEST_VAR", "from-the-shell")

    env_file = tmp_path / ".env"
    env_file.write_text("COMPLIANCE_TEST_VAR=from-the-file\n", encoding="utf-8")

    env_module.load_env(env_file)
    assert env_module.os.environ["COMPLIANCE_TEST_VAR"] == "from-the-shell"


def test_missing_file_is_not_an_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_module, "_running_under_pytest", lambda: False)
    assert env_module.load_env(tmp_path / "absent.env") is False


def test_does_nothing_under_pytest(tmp_path: Path, monkeypatch) -> None:
    """The guard: importing app.main in a test must not read a real .env."""
    monkeypatch.setattr(env_module, "_running_under_pytest", lambda: True)
    monkeypatch.delenv("COMPLIANCE_TEST_VAR", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("COMPLIANCE_TEST_VAR=from-the-file\n", encoding="utf-8")

    assert env_module.load_env(env_file) is False
    assert "COMPLIANCE_TEST_VAR" not in env_module.os.environ


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_pytest_detection_positive(value: str, monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/test_x.py::t ({value})")
    assert env_module._running_under_pytest() is True