import json

import httpx

from flash.transport import FlashTransport, _origin


def test_role_model_id_suggestions_are_unique_and_role_specific():
    models = ["SAGE Memory Compressor 2B", "Qwen3-VL Flash Vision", "Gemma Flash Controller"]
    transport = FlashTransport()
    assert transport._suggest_model_id("memory", models) == "SAGE Memory Compressor 2B"
    assert transport._suggest_model_id("qwen", models) == "Qwen3-VL Flash Vision"
    assert transport._suggest_model_id("gemma", models) == "Gemma Flash Controller"


def test_ambiguous_model_id_is_not_guessed():
    transport = FlashTransport()
    assert transport._suggest_model_id("gemma", ["gemma-a", "gemma-b"]) is None


def test_origin_strips_path():
    assert _origin("https://gpu.example/v1/chat/completions") == "https://gpu.example"


def test_pooled_client_binds_ipv4():
    transport = FlashTransport()
    client = transport._build_client()
    try:
        inner = client._transport
        while hasattr(inner, "_pool"):
            inner = inner._pool
        assert getattr(inner, "_local_address", None) == "0.0.0.0"
    finally:
        client.close()


def test_http_client_is_reused_across_requests(monkeypatch):
    calls = {"build": 0, "hits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["hits"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"flash_case":"A","gemma_answer":"ok"}'}}]},
        )

    class CountingTransport(FlashTransport):
        def _build_client(self) -> httpx.Client:
            calls["build"] += 1
            return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(
        "flash.transport.runtime_config.role",
        lambda _role: {
            "provider": "remote",
            "model_id": "gemma",
            "connection": {"base_url": "https://gpu.example", "api_key": "secret"},
        },
    )
    monkeypatch.setattr(
        "flash.transport.load_catalog",
        lambda: {"gemma": {"temperature": 0.2, "max_tokens": 1024, "reasoning": "off"}},
    )

    transport = CountingTransport()
    transport.invoke("gemma", [{"role": "user", "content": "hi"}], json_mode=True)
    transport.invoke("gemma", [{"role": "user", "content": "again"}])
    assert calls["build"] == 1
    assert calls["hits"] == 2
    transport.close()


def test_retry_rebuilds_client_after_cloudflare_530(monkeypatch):
    calls = {"build": 0, "hits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["hits"] += 1
        if calls["hits"] == 1:
            return httpx.Response(530, text="origin error")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    class CountingTransport(FlashTransport):
        def _build_client(self) -> httpx.Client:
            calls["build"] += 1
            return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(
        "flash.transport.runtime_config.role",
        lambda _role: {
            "provider": "remote",
            "model_id": "gemma",
            "connection": {"base_url": "https://gpu.example", "api_key": ""},
        },
    )
    monkeypatch.setattr(
        "flash.transport.load_catalog",
        lambda: {"gemma": {"temperature": 0.2, "max_tokens": 1024, "reasoning": "off"}},
    )

    transport = CountingTransport()
    result = transport.invoke("gemma", [{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"
    assert calls["hits"] == 2
    assert calls["build"] == 2
    transport.close()


def test_invoke_disables_thinking_when_catalog_reasoning_is_off(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    class MockedTransport(FlashTransport):
        def _build_client(self) -> httpx.Client:
            return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(
        "flash.transport.runtime_config.role",
        lambda _role: {
            "provider": "remote",
            "model_id": "gemma",
            "connection": {"base_url": "https://gpu.example", "api_key": ""},
        },
    )
    monkeypatch.setattr(
        "flash.transport.load_catalog",
        lambda: {"gemma": {"temperature": 0.2, "max_tokens": 1024, "reasoning": "off"}},
    )

    transport = MockedTransport()
    transport.invoke("gemma", [{"role": "user", "content": "hi"}], json_mode=True)
    assert captured["payload"]["max_tokens"] == 1024
    assert captured["payload"]["reasoning_budget"] == 0
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    transport.close()
