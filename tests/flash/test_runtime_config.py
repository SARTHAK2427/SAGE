import pytest

from flash.runtime_config import RuntimeConfigStore


def test_remote_keys_are_never_exposed_by_public_snapshot():
    store = RuntimeConfigStore()
    snapshot = store.configure({
        "connections": [{"id": "primary", "base_url": "https://gpu.example", "api_key": "secret"}],
        "roles": {
            "gemma": {"provider": "remote", "connection_id": "primary", "model_id": "gemma"},
            "qwen": {"provider": "remote", "connection_id": "primary", "model_id": "qwen"},
            "memory": {"provider": "remote", "connection_id": "primary", "model_id": "memory"},
        },
    })
    assert snapshot["connections"]["primary"]["has_api_key"] is True
    assert "api_key" not in snapshot["connections"]["primary"]
    assert store.role("gemma")["connection"]["api_key"] == "secret"


def test_openai_v1_suffix_is_normalized_to_server_root():
    store = RuntimeConfigStore()
    store.configure({
        "connections": [{"id": "primary", "base_url": "https://gpu.example/v1", "api_key": ""}],
        "roles": {
            "gemma": {"provider": "remote", "connection_id": "primary"},
            "qwen": {"provider": "remote", "connection_id": "primary"},
            "memory": {"provider": "disabled"},
        },
    })
    assert store.role("gemma")["connection"]["base_url"] == "https://gpu.example"


def test_local_pair_and_cpu_memory_are_valid():
    store = RuntimeConfigStore()
    snapshot = store.configure({
        "connections": [],
        "roles": {
            "gemma": {"provider": "local_gpu"},
            "qwen": {"provider": "local_gpu"},
            "memory": {"provider": "local_cpu"},
        },
    })
    assert snapshot["roles"]["memory"]["provider"] == "local_cpu"


def test_unknown_remote_binding_is_rejected():
    store = RuntimeConfigStore()
    with pytest.raises(ValueError, match="unknown remote connection"):
        store.configure({
            "roles": {
                "gemma": {"provider": "remote", "connection_id": "missing"},
                "qwen": {"provider": "local_gpu"},
                "memory": {"provider": "disabled"},
            }
        })
