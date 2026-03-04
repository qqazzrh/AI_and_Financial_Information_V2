import os
import importlib
from pathlib import Path
from unittest.mock import patch
import uuid
import pytest

TEST_FILES_DIR = (Path.cwd() / "Test_files").resolve()


@pytest.fixture
def local_tmp_dir() -> Path:
    TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = TEST_FILES_DIR / f"test_tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir.resolve()


def test_env_defaults(local_tmp_dir, monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("PENRS_CACHE_DIR", raising=False)
    monkeypatch.delenv("PENRS_LOG_DIR", raising=False)
    monkeypatch.chdir(local_tmp_dir)

    # Patch at the source so that `from dotenv import load_dotenv`
    # during reload binds the mock, not the real function.
    with patch("dotenv.load_dotenv"):
        import penrs_mcp_server
        importlib.reload(penrs_mcp_server)

        assert penrs_mcp_server.ALPHA_VANTAGE_API_KEY == "demo"
        assert penrs_mcp_server.PENRS_CACHE_DIR == (local_tmp_dir / ".penrs_cache").resolve()
        assert penrs_mcp_server.PENRS_LOG_DIR == (local_tmp_dir / ".penrs_logs").resolve()


def test_dirs_created(local_tmp_dir, monkeypatch):
    monkeypatch.setenv("PENRS_CACHE_DIR", str(local_tmp_dir / "cache"))
    monkeypatch.setenv("PENRS_LOG_DIR", str(local_tmp_dir / "logs"))
    monkeypatch.chdir(local_tmp_dir)

    import penrs_mcp_server
    importlib.reload(penrs_mcp_server)

    assert (local_tmp_dir / "cache").is_dir()
    assert (local_tmp_dir / "logs").is_dir()


def test_api_key_from_env(monkeypatch, local_tmp_dir):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "TESTKEY123")
    monkeypatch.setenv("PENRS_CACHE_DIR", str(local_tmp_dir / "cache"))
    monkeypatch.setenv("PENRS_LOG_DIR", str(local_tmp_dir / "logs"))
    monkeypatch.chdir(local_tmp_dir)

    import penrs_mcp_server
    importlib.reload(penrs_mcp_server)

    assert penrs_mcp_server.ALPHA_VANTAGE_API_KEY == "TESTKEY123"


def test_mcp_server_named(local_tmp_dir, monkeypatch):
    monkeypatch.setenv("PENRS_CACHE_DIR", str(local_tmp_dir / "cache"))
    monkeypatch.setenv("PENRS_LOG_DIR", str(local_tmp_dir / "logs"))
    monkeypatch.chdir(local_tmp_dir)

    import penrs_mcp_server
    importlib.reload(penrs_mcp_server)

    assert penrs_mcp_server.mcp.name == "penrs_mcp"
