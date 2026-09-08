"""
SAGE/orchestrator.py
Sequential multi-model agent orchestrator.

Architectural rules:
    - Gemma is the sole semantic controller
    - All capability executions route through the generic ToolRegistry / Dispatcher
    - The legacy DocumentProcessor is disconnected; document retrieval uses SageDocumentDB
    - Backend execution state is tracked in RunState
    - Supports SAGE_MOCK_MODE=1 for CPU/local testing without GPU or Docker
"""

from __future__ import annotations
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from model_manager import model_manager
from model_client import model_client
from code_executor.pipeline import CodeExecutionPipeline
from code_executor.fixer import CodeFixer, RealCoder
from code_executor.sandbox import DockerSandbox

from core.dispatcher import ToolRegistry, ToolResult
from core.run_state import RunState, RegisteredDocument
from core.json_repair import clean_json_string, parse_agent_json, build_repair_prompt
from tools.registry import create_default_registry
from core.mappers import (
    map_tool_result_for_gemma,
    build_tool_results_packet,
    map_rag_result,
    map_exact_result,
    map_artifact_fetch_result,
    map_list_artifacts_result,
    map_vision_result,
    map_coder_result,
    map_math_result,
    map_error_for_gemma,
    strip_internal_fields,
)

logger = logging.getLogger(__name__)

# Initialize UTF-8 safe Rich Console
console = Console(highlight=False, legacy_windows=False)


def _is_mock_mode() -> bool:
    return os.environ.get("SAGE_MOCK_MODE", "0") == "1"


class Orchestrator:
    def __init__(self, registry: Optional[ToolRegistry] = None):
        base_prompt = self._load_prompt("agent_system.txt")
        tools_def = self._load_prompt("tools.json")
        abilities_def = self._load_prompt("abilities.json")
        if tools_def:
            base_prompt += f"\n\nAUTHORITATIVE tools.json:\n{tools_def}"
        if abilities_def:
            base_prompt += f"\n\nAUTHORITATIVE abilities.json:\n{abilities_def}"
        self.agent_system_prompt = base_prompt
        self.coder_system_prompt = self._load_prompt("coder_system.txt")
        self.document_system_prompt = self._load_prompt("document_system.txt")

        # Code Execution Pipeline (Docker sandbox + LLM auto-repair)
        try:
            _sandbox = DockerSandbox(config.SANDBOX)
            _real_coder = RealCoder(model_client, model_manager)
            _fixer = CodeFixer(_real_coder)
            self._code_pipeline = CodeExecutionPipeline(
                sandbox=_sandbox,
                fixer=_fixer,
                max_fix_attempts=config.SANDBOX.get("max_fix_attempts", 3)
            )
        except Exception as e:
            logger.warning("CodeExecutionPipeline initialization warning: %s", e)
            self._code_pipeline = None

        # Generic Tool Registry (Wires document_db, vision, coder, math)
        if registry is not None:
            self.registry = registry
        else:
            self.registry = create_default_registry(
                code_pipeline=self._code_pipeline,
                model_manager=model_manager,
                model_client=model_client,
            )

    def _load_prompt(self, filename: str) -> str:
        path = config.PROMPTS_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _clean_json_str(self, text: str) -> str:
        return clean_json_string(text)

    def _parse_gemma_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        return parse_agent_json(raw_text)

    def _dispatch_tool_call(
        self,
        call: Dict[str, Any],
        run_state: RunState,
        telemetry: Dict[str, Any],
        trace: List[Dict[str, Any]],
        file_map: Dict[str, Dict[str, Any]],
        call_index: int = 0,
    ) -> Dict[str, Any]:
        """Dispatch a single tool call and return a Gemma-facing mapped result.

        Flow:
            Gemma tool call
                ↓
            ToolRegistry.dispatch() → rich internal result
                ↓
            tool-specific mapper function
                ↓
            map_tool_result_for_gemma() → outer {call_index, tool, function,
                                                   status, result|error|warning}

        Gemma NEVER receives raw deterministic sockets.
        Status values: "success" | "partial" | "error".
        """
        tool_name = call.get("tool", "")
        func_name = call.get("function", "")
        task = call.get("task", "")
        arguments = call.get("arguments", {})
        # Support both top-level args and nested arguments dict
        if not arguments:
            arguments = {k: v for k, v in call.items() if k not in ("tool", "function", "task")}

        # ── 1. document_database group ────────────────────────────────────────
        if tool_name in ("document_database", "document_db"):
            resolved_fn = func_name or "rag_search"
            telemetry["document_calls"] = telemetry.get("document_calls", 0) + 1

            console.print(Panel(
                f"[bold]Tool:[/bold] document_database\n[bold]Function:[/bold] {resolved_fn}\n[bold]Args:[/bold] {str(arguments)[:200]}",
                title="[bold cyan]EXECUTING TOOL: DOCUMENT DB[/bold cyan]",
                border_style="cyan"
            ))

            disp_res: ToolResult = self.registry.dispatch(
                "document_database", resolved_fn, **arguments
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}
            if disp_res.status not in ("success", "partial"):
                rich_socket["status"] = disp_res.status
                rich_socket["error"] = disp_res.error

            trace.append({"actor": "document_db", "action": resolved_fn,
                           "status": disp_res.status, "duration_ms": disp_res.duration_ms})

            # Select mapper by function name
            mapper_map = {
                "rag_search": (map_rag_result, {}),
                "exact_search": (map_exact_result, {}),
                "artifact_fetch": (map_artifact_fetch_result, {
                    "doc_id": arguments.get("doc_id", ""),
                    "element_id": arguments.get("element_id", ""),
                }),
                "list_artifacts": (map_list_artifacts_result, {}),
            }
            if resolved_fn not in mapper_map:
                return {
                    "call_index": call_index,
                    "tool": "document_database",
                    "function": resolved_fn,
                    "status": "error",
                    "error": {
                        "code": "UNMAPPED_TOOL_RESULT",
                        "message": "No Gemma-facing mapper exists for this tool result.",
                        "retryable": False,
                    },
                }

            mapper_fn, extra_kwargs = mapper_map[resolved_fn]
            return map_tool_result_for_gemma(
                "document_database", resolved_fn, rich_socket,
                call_index=call_index, mapper_fn=mapper_fn, **extra_kwargs
            )

        # ── 2. vision_ocr group ───────────────────────────────────────────────
        elif tool_name == "vision_ocr":
            resolved_fn = func_name or "analyze_image"
            doc_id = arguments.get("doc_id") or (
                run_state.registered_documents[0].doc_id
                if run_state.registered_documents else ""
            )
            image_id = arguments.get("image_id") or (arguments.get("image_ids")[0] if arguments.get("image_ids") else "")
            if not image_id and doc_id:
                try:
                    from db_service import document_db
                    arts = document_db.list_artifacts(doc_id=doc_id, element_type="image")
                    if arts.get("artifacts"):
                        image_id = arts["artifacts"][0]["element_id"]
                except Exception:
                    pass
            instruction = arguments.get("instruction") or arguments.get("prompt") or arguments.get("task") or task or "Extract all text, numbers, and details from this image."
            image_ref = arguments.get("image_ref")

            telemetry["document_calls"] = telemetry.get("document_calls", 0) + 1
            console.print(Panel(
                f"[bold]Tool:[/bold] vision_ocr\n[bold]Image:[/bold] {image_id}\n[bold]Task:[/bold] {instruction[:200]}",
                title="[bold magenta]EXECUTING TOOL: VISION OCR[/bold magenta]",
                border_style="magenta"
            ))

            disp_res: ToolResult = self.registry.dispatch(
                "vision_ocr", "analyze_image",
                instruction=instruction,
                doc_id=doc_id,
                image_id=image_id,
                image_ref=image_ref,
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}

            trace.append({"actor": "vision_ocr", "action": "analyze_image",
                           "image_id": image_id, "status": disp_res.status,
                           "duration_ms": disp_res.duration_ms})

            return map_tool_result_for_gemma(
                tool_name, resolved_fn, rich_socket,
                call_index=call_index,
                mapper_fn=map_vision_result,
                doc_id=doc_id, image_id=image_id,
            )

        # ── 3. code_specialist group ──────────────────────────────────────────
        elif tool_name in ("code_specialist", "coder"):
            resolved_fn = func_name or "solve_code_task"
            instruction = arguments.get("instruction") or arguments.get("prompt") or arguments.get("task") or task
            code = arguments.get("code")
            language = arguments.get("language")

            telemetry["coder_calls"] = telemetry.get("coder_calls", 0) + 1
            console.print(Panel(
                f"[bold]Tool:[/bold] code_specialist\n[bold]Instruction:[/bold] {instruction[:200]}\n[bold]Language:[/bold] {language or 'python'}",
                title="[bold yellow]EXECUTING TOOL: CODE SPECIALIST[/bold yellow]",
                border_style="yellow"
            ))

            disp_res: ToolResult = self.registry.dispatch(
                "code_specialist", "solve_code_task",
                instruction=instruction, code=code, language=language,
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}

            res_dict = rich_socket
            succeeded = res_dict.get("succeeded", disp_res.status == "success")
            exec_status = res_dict.get("execution_status", "success" if succeeded else "error")
            trace.append({
                "actor": "coder", "action": "executed_code",
                "status": exec_status,
                "language": res_dict.get("language", language or "python"),
                "exit_code": res_dict.get("exit_code"),
                "wall_time_ms": res_dict.get("wall_time_ms", disp_res.duration_ms),
                "attempts": res_dict.get("attempts", 1),
                "stdout_preview": (res_dict.get("stdout") or "")[:200],
            })

            return map_tool_result_for_gemma(
                "code_specialist", "solve_code_task", rich_socket,
                call_index=call_index, mapper_fn=map_coder_result,
            )

        # ── 4. math group ─────────────────────────────────────────────────────
        elif tool_name in ("math", "calculator"):
            expr = arguments.get("expression") or arguments.get("expr") or task
            console.print(Panel(
                f"[bold]Tool:[/bold] math\n[bold]Expression:[/bold] {expr}",
                title="[bold blue]EXECUTING TOOL: MATH[/bold blue]",
                border_style="blue"
            ))

            disp_res: ToolResult = self.registry.dispatch(
                "math", "calculate", expression=expr
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {
                "expression": expr, "status": disp_res.status,
                "error": disp_res.error,
            }

            return map_tool_result_for_gemma(
                "math", "calculate", rich_socket,
                call_index=call_index, mapper_fn=map_math_result,
            )

        # ── 5. Legacy document_analyzer bridge (→ document_database/rag_search)
        elif tool_name == "document_analyzer":
            doc_ids = run_state.list_document_ids() or None
            query = arguments.get("query") or arguments.get("task") or task
            telemetry["document_calls"] = telemetry.get("document_calls", 0) + 1

            console.print(Panel(
                f"[bold]Tool:[/bold] document_analyzer → rag_search\n[bold]Query:[/bold] {query}",
                title="[bold cyan]EXECUTING TOOL: DOCUMENT DB SEARCH[/bold cyan]",
                border_style="cyan"
            ))

            disp_res: ToolResult = self.registry.dispatch(
                "document_database", "rag_search",
                query=query, doc_ids=doc_ids, top_k=5,
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}

            trace.append({"actor": "document_db", "action": "retrieved_content",
                           "query": query, "status": disp_res.status,
                           "duration_ms": disp_res.duration_ms})

            return map_tool_result_for_gemma(
                "document_database", "rag_search", rich_socket,
                call_index=call_index, mapper_fn=map_rag_result,
            )

        # ── 6. Legacy direct DB methods (bare rag_search, etc.) ───────────────
        elif tool_name in ("rag_search", "exact_search", "artifact_fetch", "list_artifacts"):
            args = {k: v for k, v in arguments.items()}
            if "query" not in args and task:
                args["query"] = task
            telemetry["document_calls"] = telemetry.get("document_calls", 0) + 1

            disp_res: ToolResult = self.registry.dispatch(
                "document_database", tool_name, **args
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}

            mapper_map = {
                "rag_search": (map_rag_result, {}),
                "exact_search": (map_exact_result, {}),
                "artifact_fetch": (map_artifact_fetch_result, {
                    "doc_id": args.get("doc_id", ""),
                    "element_id": args.get("element_id", ""),
                }),
                "list_artifacts": (map_list_artifacts_result, {}),
            }
            if tool_name not in mapper_map:
                return {
                    "call_index": call_index,
                    "tool": "document_database",
                    "function": tool_name,
                    "status": "error",
                    "error": {
                        "code": "UNMAPPED_TOOL_RESULT",
                        "message": "No Gemma-facing mapper exists for this tool result.",
                        "retryable": False,
                    },
                }
            mapper_fn, extra_kwargs = mapper_map[tool_name]
            return map_tool_result_for_gemma(
                "document_database", tool_name, rich_socket,
                call_index=call_index, mapper_fn=mapper_fn, **extra_kwargs
            )

        # ── 7. Legacy vision alias ────────────────────────────────────────────
        elif tool_name == "vision":
            doc_id = arguments.get("doc_id") or (
                run_state.registered_documents[0].doc_id
                if run_state.registered_documents else ""
            )
            image_id = arguments.get("image_id", "")
            image_ids = arguments.get("image_ids") or ([image_id] if image_id else [])
            instruction = arguments.get("instruction") or task
            telemetry["document_calls"] = telemetry.get("document_calls", 0) + 1

            disp_res: ToolResult = self.registry.dispatch(
                "vision", "analyze",
                doc_id=doc_id, image_ids=image_ids, instruction=instruction,
            )
            rich_socket = disp_res.result if isinstance(disp_res.result, dict) else {"status": disp_res.status}

            first_img_id = image_ids[0] if image_ids else image_id
            return map_tool_result_for_gemma(
                "vision_ocr", "analyze_image", rich_socket,
                call_index=call_index, mapper_fn=map_vision_result,
                doc_id=doc_id, image_id=first_img_id,
            )

        # ── 8. Generic fallback (Fail closed: no unmapped tool result reaches Gemma) ───
        else:
            resolved_fn = func_name or "execute"
            disp_res: ToolResult = self.registry.dispatch(
                tool_name, resolved_fn, **arguments
            )

            if disp_res.status == "unknown_tool":
                avail = list(self.registry.available_tools.keys())
                return {
                    "call_index": call_index,
                    "tool": tool_name or "unknown",
                    "function": resolved_fn,
                    "status": "error",
                    "error": {
                        "code": "UNKNOWN_TOOL",
                        "message": f"Unknown tool '{tool_name}'. Available: {avail}",
                        "retryable": False,
                    },
                }

            # Fail closed: Even if a registered tool executed, if it has no explicit
            # Gemma mapper, NEVER expose rich socket or sanitized socket to Gemma.
            return {
                "call_index": call_index,
                "tool": tool_name,
                "function": resolved_fn,
                "status": "error",
                "error": {
                    "code": "UNMAPPED_TOOL_RESULT",
                    "message": "No Gemma-facing mapper exists for this tool result.",
                    "retryable": False,
                },
            }

    def run(
        self,
        user_objective: str,
        attachments_manifest: List[Dict[str, Any]],
        file_map: Dict[str, Dict[str, Any]],
        run_state: Optional[RunState] = None,
    ) -> Dict[str, Any]:
        """Execute the sequential multi-model orchestration loop."""
        wall_start = time.time()
        telemetry = {
            "agent_calls": 0,
            "coder_calls": 0,
            "document_calls": 0,
            "model_switches_at_start": model_manager.switch_count if model_manager else 0,
            "total_wall_time": 0.0,
            "final_response_length": 0,
        }
        trace: List[Dict[str, Any]] = []

        # Initialize RunState if not passed
        if run_state is None:
            run_state = RunState(
                request_id=f"req_{int(time.time())}_{uuid.uuid4().hex[:6]}",
                user_text=user_objective,
            )
            for att in attachments_manifest:
                run_state.register_document(
                    doc_id=att.get("doc_id") or att.get("ref", "unknown"),
                    display_name=att.get("name", "document"),
                    file_type=att.get("type"),
                    source_name=att.get("name"),
                )

        console.print("\n" + "="*70, style="bold cyan")
        console.print("[bold cyan][START] SAGE SEQUENTIAL MULTI-MODEL ORCHESTRATION[/bold cyan]")
        console.print("="*70 + "\n", style="bold cyan")
        console.print(f"[bold]User Objective:[/bold] {user_objective}")
        if attachments_manifest:
            console.print(f"[bold]Attachments:[/bold] {len(attachments_manifest)} registered file(s)")
            for att in attachments_manifest:
                doc_str = f" [doc_id: {att.get('doc_id')}]" if att.get("doc_id") else ""
                console.print(f"  * [yellow]{att['ref']}[/yellow]: {att['name']} ({att['type']}, {att['size']} bytes){doc_str}")

        # Initial conversation for Gemma
        user_payload = f"Objective:\n{user_objective}\n\nATTACHMENTS:\n{json.dumps(attachments_manifest, indent=2)}"
        history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.agent_system_prompt},
            {"role": "user", "content": user_payload}
        ]

        final_answer = None
        loop_count = 0

        # Mock mode fast-path if enabled
        if _is_mock_mode():
            logger.info("Orchestrator running in SAGE_MOCK_MODE=1")
            final_answer = f"MOCK_RESPONSE: Successfully processed objective: '{user_objective}'"
            if run_state.registered_documents:
                doc_names = [d.display_name for d in run_state.registered_documents]
                final_answer += f" using documents: {doc_names}"
            trace.append({
                "actor": "gemma",
                "action": "final_synthesis",
                "loop": 1,
                "answer_preview": final_answer,
            })
            telemetry["agent_calls"] = 1
            wall_end = time.time()
            telemetry["total_wall_time"] = wall_end - wall_start
            telemetry["final_response_length"] = len(final_answer)

            return {
                "status": "success",
                "answer": final_answer,
                "telemetry": telemetry,
                "trace": trace,
                "run_state": run_state.to_dict(),
            }

        # ── Main Agent Loop ───────────────────────────────────────────────────
        while loop_count < config.MAX_AGENT_LOOPS:
            loop_count += 1
            run_state.loop_index = loop_count

            console.print("\n" + "="*60, style="bold blue")
            console.print(f"[bold blue]AGENT LOOP {loop_count} / {config.MAX_AGENT_LOOPS}[/bold blue]", style="bold blue")
            console.print("="*60, style="bold blue")

            # Ensure Gemma is loaded
            model_manager.ensure_model("agent")
            telemetry["agent_calls"] += 1
            agent_cfg = config.MODELS.get("agent", {})

            last_msg = history[-1]
            console.print(Panel(
                f"[dim]Role: {last_msg['role']}[/dim]\n{last_msg['content'][:500]}...",
                title=f"[bold blue]INPUT -> GEMMA (Loop {loop_count})[/bold blue]",
                border_style="blue"
            ))

            # Call Gemma
            res = model_client.chat_completion(
                messages=history,
                temperature=agent_cfg.get("temperature", 0.20),
                max_tokens=agent_cfg.get("max_tokens", 2048)
            )

            raw_output = res["content"]
            duration = res["duration"]
            usage = res.get("usage", {})
            tok_info = f"{duration:.2f}s | {usage.get('completion_tokens', 'N/A')} tokens"

            console.print(Panel(
                raw_output,
                title=f"[bold blue]OUTPUT <- GEMMA ({tok_info})[/bold blue]",
                border_style="blue"
            ))

            # Parse JSON with decoupled json_repair logic
            parsed = self._parse_gemma_json(raw_output)

            # Defensive 1-turn repair if JSON is invalid
            if parsed is None:
                console.print("[bold red][WARN] Gemma output was not valid JSON. Attempting 1 repair turn...[/bold red]")
                repair_prompt = build_repair_prompt()
                history.append({"role": "assistant", "content": raw_output})
                history.append({"role": "user", "content": repair_prompt})

                res = model_client.chat_completion(
                    messages=history,
                    temperature=0.10,
                    max_tokens=agent_cfg.get("max_tokens", 2048)
                )
                raw_output = res["content"]
                parsed = self._parse_gemma_json(raw_output)

                # Remove repair turn from history
                history.pop()
                history.pop()

                if parsed is None:
                    raise RuntimeError(f"Gemma failed to produce valid JSON after repair attempt. Raw: {raw_output}")

            # Process Response
            res_type = parsed.get("type")

            if res_type == "final":
                final_answer = parsed.get("answer", "")
                trace.append({
                    "actor": "gemma",
                    "action": "final_synthesis",
                    "loop": loop_count,
                    "answer_preview": final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
                })
                break

            elif res_type == "tool_calls":
                calls = parsed.get("calls", [])
                if not calls:
                    raise RuntimeError("Gemma returned tool_calls with an empty 'calls' array.")

                trace.append({
                    "actor": "gemma",
                    "action": "requested_tools",
                    "loop": loop_count,
                    "calls": calls
                })

                history.append({"role": "assistant", "content": json.dumps(parsed, indent=2)})
                tool_results_list = []

                for call in calls:
                    t_call_start = time.time()
                    call_record = run_state.record_tool_call(
                        call_id=f"call_{uuid.uuid4().hex[:8]}",
                        tool_name=call.get("tool", "unknown"),
                        function_name=call.get("function", "execute"),
                        sanitized_args={k: v for k, v in call.items() if k != "tool"},
                        start_time=t_call_start,
                    )

                    mapped_result = self._dispatch_tool_call(
                        call=call,
                        run_state=run_state,
                        telemetry=telemetry,
                        trace=trace,
                        file_map=file_map,
                        call_index=len(tool_results_list),
                    )

                    t_call_end = time.time()
                    run_state.record_tool_result(
                        call_id=call_record.call_id,
                        status=mapped_result.get("status", "success"),
                        result_summary=str(mapped_result.get("result", ""))[:200],
                        error=str(mapped_result.get("error", "")) if mapped_result.get("error") else None,
                        end_time=t_call_end,
                    )

                    tool_results_list.append(mapped_result)

                # Build compact Gemma-facing packet (never raw sockets)
                tool_result_payload = build_tool_results_packet(tool_results_list)

                payload_str = json.dumps(tool_result_payload, indent=2)
                console.print(Panel(
                    payload_str[:800] + ("..." if len(payload_str) > 800 else ""),
                    title="[bold green]RETURNING TOOL RESULTS TO GEMMA[/bold green]",
                    border_style="green"
                ))

                history.append({"role": "user", "content": json.dumps(tool_result_payload, indent=2)})

            else:
                raise RuntimeError(f"Unexpected response type from Gemma: {res_type}")

        if final_answer is None:
            raise RuntimeError(f"Maximum agent loops ({config.MAX_AGENT_LOOPS}) reached without generating a final response.")

        wall_end = time.time()
        telemetry["total_wall_time"] = wall_end - wall_start
        telemetry["final_response_length"] = len(final_answer)
        telemetry["model_switches"] = (model_manager.switch_count - telemetry["model_switches_at_start"]) if model_manager else 0

        # Telemetry Summary Table
        table = Table(title="[bold green]SAGE EXECUTION TELEMETRY[/bold green]", border_style="green")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")

        table.add_row("Agent (Gemma) Calls", str(telemetry["agent_calls"]))
        table.add_row("Document Analyzer / DB Calls", str(telemetry["document_calls"]))
        table.add_row("Coder Calls", str(telemetry["coder_calls"]))
        table.add_row("Model Switches", str(telemetry["model_switches"]))
        table.add_row("Total Wall Time", f"{telemetry['total_wall_time']:.2f} seconds")
        table.add_row("Final Answer Length", f"{telemetry['final_response_length']} chars")

        console.print("\n" + "="*60, style="bold green")
        console.print("[bold green][SUCCESS] REQUEST COMPLETE[/bold green]")
        console.print("="*60, style="bold green")
        console.print(table)
        console.print("="*60 + "\n", style="bold green")

        return {
            "status": "success",
            "answer": final_answer,
            "telemetry": telemetry,
            "trace": trace,
            "run_state": run_state.to_dict(),
        }


orchestrator = Orchestrator()
