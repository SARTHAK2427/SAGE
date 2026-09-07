"""
SAGE Code Executor — LLM-Powered Error Fixer
=============================================
Provides:
  • CoderInterface  — abstract contract for any code-generating model
  • MockCoder       — deterministic mock; testing without a live LLM
  • RealCoder       — production adapter for Qwen2.5-Coder (model_client / model_manager)
  • FixAttempt      — record of one repair cycle
  • CodeFixer       — orchestrates repair-prompt construction + code extraction
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from .extractor import extract_code
from .sandbox import SandboxResult


# ─── Abstract Coder Interface ────────────────────────────────────────────────

class CoderInterface(ABC):
    """Abstract interface for any code-generating / code-repairing model."""

    @abstractmethod
    def generate(self, task: str, context: str) -> str:
        """
        Generate or repair Python code.

        Args:
            task:    Short description of the coding goal.
            context: Full context — failing code, error traceback, etc.

        Returns:
            Raw model response text (may contain fences, prose, etc.).
        """
        ...


# ─── Mock Coder (for testing) ─────────────────────────────────────────────────

class MockCoder(CoderInterface):
    """
    Deterministic mock coder for standalone testing without a live LLM.

    Usage::

        mock = MockCoder()
        mock.add_fix_response("```python\\nprint('fixed!')\\n```")
        # First call to .generate() returns the queued response.
        # Subsequent calls return the default response.
    """

    def __init__(
        self,
        default_response: str = "```python\nprint('mock default fix')\n```",
    ) -> None:
        self._queue: List[str] = []
        self._default = default_response
        self._call_count = 0

    def add_fix_response(self, response: str) -> None:
        """Queue a raw LLM-style response (FIFO) for the next generate() call."""
        self._queue.append(response)

    def generate(self, task: str, context: str) -> str:  # noqa: ARG002
        self._call_count += 1
        return self._queue.pop(0) if self._queue else self._default

    @property
    def call_count(self) -> int:
        """Total number of generate() calls made."""
        return self._call_count


# ─── Real Coder (production) ──────────────────────────────────────────────────

class RealCoder(CoderInterface):
    """
    Production coder: calls Qwen2.5-Coder via the existing model_client
    and model_manager infrastructure.

    Does NOT modify model_client.py or model_manager.py.
    Loads the model on demand via ensure_model("coder").
    """

    def __init__(self, model_client, model_manager) -> None:
        self._client = model_client
        self._manager = model_manager

    def generate(self, task: str, context: str) -> str:
        import config  # Imported lazily so the module works standalone

        self._manager.ensure_model("coder")
        coder_cfg = config.MODELS["coder"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are SAGE's specialized Coding Specialist.\n"
                    "Fix the provided Python code based on the reported error.\n"
                    "Return ONLY a ```python ... ``` fenced code block. "
                    "No explanation, prose, or text outside the fence."
                ),
            },
            {
                "role": "user",
                "content": f"TASK:\n{task}\n\nCONTEXT / ERROR:\n{context}",
            },
        ]

        res = self._client.chat_completion(
            messages=messages,
            temperature=coder_cfg.get("temperature", 0.10),
            max_tokens=coder_cfg.get("max_tokens", 4096),
        )
        return res["content"]


# ─── Fix Attempt record ───────────────────────────────────────────────────────

@dataclass
class FixAttempt:
    """Records one complete repair cycle for audit / telemetry purposes."""
    attempt_number: int
    original_code: str         # Code that failed
    fixed_code: str            # Code returned by the fixer
    sandbox_result: SandboxResult  # Result of running fixed_code
    repair_prompt: str         # The prompt sent to the coder
    raw_response: str = ""
    extraction: Optional[dict] = None
    failure_reason: Optional[str] = None
    retryable: bool = True

    def to_socket(self) -> dict:
        sb_socket = self.sandbox_result.to_socket() if hasattr(self.sandbox_result, "to_socket") else {
            "stdout": self.sandbox_result.stdout,
            "stderr": self.sandbox_result.stderr,
            "exit_code": self.sandbox_result.exit_code,
            "timed_out": self.sandbox_result.timed_out,
            "wall_time_ms": self.sandbox_result.wall_time_ms,
        }
        return {
            "attempt_number": self.attempt_number,
            "original_code": self.original_code,
            "generated_code": self.fixed_code,
            "repair_prompt": self.repair_prompt,
            "model": {
                "raw_response": self.raw_response,
                "repair_prompt": self.repair_prompt,
            },
            "extraction": self.extraction or {
                "code": self.fixed_code,
                "language": "python",
                "had_fence": True,
                "block_count": 1,
            },
            "sandbox": sb_socket,
            "failure": {
                "reason": self.failure_reason or (f"Non-zero exit code {self.sandbox_result.exit_code}" if self.sandbox_result.exit_code != 0 else None),
                "stderr": self.sandbox_result.stderr,
                "retryable": self.retryable,
            },
            "repair": {
                "repair_prompt": self.repair_prompt,
                "prompt_length": len(self.repair_prompt),
            },
        }

    def to_dict(self) -> dict:
        return self.to_socket()


# ─── Code Fixer ──────────────────────────────────────────────────────────────

class CodeFixer:
    """
    Sends failing code + sandbox result to a CoderInterface for repair,
    extracts the returned code block, and returns the ready-to-run string.
    """

    def __init__(self, coder: CoderInterface) -> None:
        self.coder = coder

    def build_repair_prompt(
        self,
        task: str,
        failing_code: str,
        result: SandboxResult,
        attempt: int,
    ) -> str:
        """
        Construct a structured repair prompt from the task + failing code
        + sandbox result details.
        """
        lines: List[str] = [
            f"=== SAGE CODE REPAIR — ATTEMPT {attempt} ===",
            "",
            "ORIGINAL TASK:",
            task,
            "",
            "FAILING CODE:",
            "```python",
            failing_code,
            "```",
            "",
        ]

        if result.timed_out:
            lines += [
                "ERROR TYPE: Execution timeout",
                "The code contains an infinite loop or is computationally too expensive.",
                "Rewrite it so it terminates within the allowed wall-clock limit.",
            ]
        elif result.error:
            lines += [
                "ERROR TYPE: Docker infrastructure error (not a code bug):",
                result.error,
                "Rewrite the code to avoid triggering infrastructure limits.",
            ]
        else:
            if result.stderr:
                lines += [
                    f"STDERR / TRACEBACK (exit code {result.exit_code}):",
                    result.stderr,
                ]
            else:
                lines.append(
                    f"EXIT CODE: {result.exit_code} (non-zero exit, no stderr output)"
                )

        lines += [
            "",
            "Fix the code so it runs correctly and exits with code 0.",
            "Return ONLY a ```python ... ``` fenced code block. No prose outside the fence.",
        ]
        return "\n".join(lines)

    def fix(
        self,
        task: str,
        failing_code: str,
        result: SandboxResult,
        attempt: int,
    ) -> str:
        """
        Ask the coder interface to repair *failing_code*.

        Returns:
            The extracted fixed code string (ready to pass to sandbox.run()).
        """
        prompt = self.build_repair_prompt(task, failing_code, result, attempt)
        raw_response = self.coder.generate(
            task=f"Fix Python code (attempt {attempt})",
            context=prompt,
        )
        return extract_code(raw_response).code
