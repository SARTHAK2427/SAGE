"""
Tests for config portability and discovery logic.

Verifies:
1. Discovery priority order: explicit env var > PATH / repo-relative > unresolved ("")
2. Complete absence of machine-specific / user-specific hardcoded paths
3. Backward-compatible config API
4. .env.example cleanliness
"""

import os
import re
import pytest
from pathlib import Path
import config


class TestConfigPortability:
    def test_no_hardcoded_user_paths_in_config_source(self):
        """Ensure config.py source code contains zero user-specific or machine-specific absolute paths."""
        config_path = Path(config.__file__).resolve()
        content = config_path.read_text(encoding="utf-8")

        # Must not contain C:\Users or /home/ or /Users/
        assert not re.search(r"[A-Za-z]:\\Users\\[a-zA-Z0-9_-]+", content, re.IGNORECASE), (
            "Found hardcoded Windows user path in config.py"
        )
        assert not re.search(r"/home/[a-zA-Z0-9_-]+", content), (
            "Found hardcoded Linux user path in config.py"
        )
        assert not re.search(r"/Users/[a-zA-Z0-9_-]+", content), (
            "Found hardcoded macOS user path in config.py"
        )
        assert "sunbu" not in content.lower(), "Found legacy user 'sunbu' in config.py"
        assert "sarth" not in content.lower(), "Found legacy user 'sarth' in config.py"

    def test_llama_server_env_priority(self, monkeypatch):
        """Explicit LLAMA_SERVER_PATH takes top priority."""
        fake_path = r"D:\CustomLlama\llama-server.exe"
        monkeypatch.setenv("LLAMA_SERVER_PATH", fake_path)
        assert config._discover_llama_server() == fake_path

    def test_llama_server_which_discovery(self, monkeypatch, tmp_path):
        """When LLAMA_SERVER_PATH is not set, discovers via PATH."""
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising=False)
        fake_binary = tmp_path / "llama-server.exe"
        fake_binary.write_text("fake binary")

        # Mock shutil.which to return fake_binary
        monkeypatch.setattr("shutil.which", lambda name: str(fake_binary) if "llama-server" in name else None)
        assert config._discover_llama_server() == str(fake_binary.resolve())

    def test_llama_server_unresolved_fallback(self, monkeypatch):
        """When neither env nor PATH nor repo candidate exists, returns empty string."""
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        # Ensure repo candidates don't match
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        assert config._discover_llama_server() == ""

    def test_model_dir_env_priority(self, monkeypatch):
        """Explicit MODEL_DIR takes top priority."""
        fake_dir = r"D:\CustomModels"
        monkeypatch.setenv("MODEL_DIR", fake_dir)
        assert config._discover_model_dir() == fake_dir

    def test_model_dir_repo_discovery(self, monkeypatch, tmp_path):
        """When MODEL_DIR is not set, discovers repo-relative models dir if present."""
        monkeypatch.delenv("MODEL_DIR", raising=False)
        fake_models = config.BASE_DIR / "models"
        if not fake_models.exists():
            # If repo doesn't have models/ currently, test the discovery logic with is_dir patch
            monkeypatch.setattr("pathlib.Path.is_dir", lambda self: self == fake_models)
            assert config._discover_model_dir() == str(fake_models.resolve())

    def test_model_dir_unresolved_fallback(self, monkeypatch):
        """When neither env nor repo-relative dir exists, returns empty string."""
        monkeypatch.delenv("MODEL_DIR", raising=False)
        monkeypatch.setattr("pathlib.Path.is_dir", lambda self: False)
        assert config._discover_model_dir() == ""

    def test_config_api_backward_compatibility(self):
        """Ensure all standard config attributes and types expected by SAGE remain intact."""
        assert isinstance(config.BASE_DIR, Path)
        assert isinstance(config.TEMP_DIR, Path)
        assert isinstance(config.PROMPTS_DIR, Path)
        assert isinstance(config.STATIC_DIR, Path)
        assert isinstance(config.ARTIFACTS_ROOT, Path)
        assert isinstance(config.CHROMA_ROOT, Path)

        assert isinstance(config.LLAMA_SERVER_PATH, str)
        assert isinstance(config.MODEL_DIR, str)

        assert isinstance(config.SERVER_HOST, str)
        assert isinstance(config.LLAMA_PORT, int)
        assert isinstance(config.APP_PORT, int)
        assert isinstance(config.LLAMA_BASE_URL, str)

        assert isinstance(config.MAX_AGENT_LOOPS, int)
        assert isinstance(config.MODEL_START_TIMEOUT, (int, float))
        assert isinstance(config.REQUEST_TIMEOUT, (int, float))

        assert isinstance(config.SANDBOX, dict)
        assert "image" in config.SANDBOX
        assert "timeout_seconds" in config.SANDBOX

        assert isinstance(config.MODELS, dict)
        assert "agent" in config.MODELS
        assert "coder" in config.MODELS
        assert "document_analyzer" in config.MODELS

        for key in ("agent", "coder", "document_analyzer"):
            cfg = config.MODELS[key]
            assert "key" in cfg
            assert "name" in cfg
            assert "model_path" in cfg
            assert "context" in cfg
            assert "ngl" in cfg
            assert "temperature" in cfg
            assert "max_tokens" in cfg

    def test_env_example_file(self):
        """.env.example must exist and contain only placeholders, no real user paths."""
        env_example_path = config.BASE_DIR / ".env.example"
        assert env_example_path.is_file(), ".env.example must exist in project root"

        content = env_example_path.read_text(encoding="utf-8")
        assert "LLAMA_SERVER_PATH" in content
        assert "MODEL_DIR" in content

        # Verify placeholders only
        assert "C:\\path\\to\\" in content
        assert "Users\\sunbu" not in content
        assert "Users\\sarth" not in content
