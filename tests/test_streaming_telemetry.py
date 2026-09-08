"""
tests/test_streaming_telemetry.py

Automated tests covering:
1. Orchestrator live event_callback hook (order of events, presence of input and output)
2. FastAPI /api/chat/stream endpoint (SSE headers and streaming event emission)
3. Backward compatibility with standard /api/chat
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app import app
from orchestrator import Orchestrator
from core.dispatcher import ToolRegistry, ToolResult
from core.run_state import RunState


class TestStreamingTelemetry:
    """Verify live telemetry event streaming and model inspection."""

    def test_orchestrator_event_callback_emits_stages_and_inspect_data(self, monkeypatch):
        """Verify that Orchestrator.run emits structured events with inputs and outputs."""
        monkeypatch.setenv("SAGE_MOCK_MODE", "0")
        registry = ToolRegistry()
        registry.register(
            "math", "calculate",
            lambda expression: {"expression": expression, "value": 42}
        )

        orch = Orchestrator(registry=registry)
        events_emitted = []

        def callback(evt):
            events_emitted.append(evt)

        # Mock model_client to return tool_calls on turn 1, and final on turn 2
        with patch("orchestrator.model_client") as mock_client, \
             patch("orchestrator.model_manager") as mock_mgr:
            mock_mgr.switch_count = 0
            mock_mgr.ensure_model.return_value = True

            mock_client.chat_completion.side_effect = [
                # Turn 1: tool call
                {
                    "content": json.dumps({
                        "type": "tool_calls",
                        "calls": [{"tool": "math", "function": "calculate", "arguments": {"expression": "6*7"}}]
                    }),
                    "duration": 0.5,
                    "usage": {"completion_tokens": 20},
                },
                # Turn 2: Gemma final brief
                {
                    "content": json.dumps({
                        "type": "final",
                        "answer": "The answer to 6*7 is 42."
                    }),
                    "duration": 0.4,
                    "usage": {"completion_tokens": 15},
                },
                # Turn 3: Qwen3.5 final_synthesizer
                {
                    "content": "The answer to 6*7 is 42.",
                    "duration": 0.2,
                    "usage": {"completion_tokens": 10},
                },
            ]

            result = orch.run(
                user_objective="What is 6*7?",
                attachments_manifest=[],
                file_map={},
                event_callback=callback,
            )

        assert result["status"] == "success"
        assert "42" in result["answer"]

        # Verify event sequence
        event_types = [e.get("event") for e in events_emitted]
        assert "model_invoking" in event_types
        assert "model_output" in event_types
        assert "tools_requested" in event_types
        assert "tool_executing" in event_types
        assert "tool_result" in event_types
        assert "final_synthesis" in event_types
        assert "run_complete" in event_types

        # Verify inspection payload (input and output are present)
        invoking_evt = next(e for e in events_emitted if e["event"] == "model_invoking")
        assert "What is 6*7?" in invoking_evt["input"]

        output_evt = next(e for e in events_emitted if e["event"] == "model_output")
        assert output_evt["parsed_type"] == "tool_calls"
        assert "6*7" in output_evt["output"]

        tool_exec_evt = next(e for e in events_emitted if e["event"] == "tool_executing")
        assert tool_exec_evt["tool"] == "math"

    def test_api_chat_stream_sse_endpoint(self):
        """Verify that POST /api/chat/stream returns valid text/event-stream."""
        client = TestClient(app)

        with patch("orchestrator.orchestrator.run") as mock_run:
            mock_run.return_value = {
                "status": "success",
                "answer": "Streamed answer",
                "telemetry": {"total_wall_time": 1.23, "agent_calls": 1},
                "trace": [],
                "run_state": {},
            }

            response = client.post(
                "/api/chat/stream",
                data={"objective": "Test streaming objective"}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            body_text = response.text
            assert "event:" in body_text
            assert "data:" in body_text
            assert "done" in body_text

    def test_legacy_api_chat_endpoint_intact(self):
        """Verify that standard non-streaming POST /api/chat remains 100% intact."""
        client = TestClient(app)

        with patch("orchestrator.orchestrator.run") as mock_run:
            mock_run.return_value = {
                "status": "success",
                "answer": "Standard JSON answer",
                "telemetry": {"total_wall_time": 0.5, "agent_calls": 1},
                "trace": [],
                "run_state": {},
            }

            response = client.post(
                "/api/chat",
                data={"objective": "Test standard objective"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["answer"] == "Standard JSON answer"
