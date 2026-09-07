"""
SAGE/tools/math_tool.py
Safe deterministic arithmetic evaluator.

Requirements:
    - No unrestricted eval()
    - Support basic arithmetic: + - * / // % **
    - Support parentheses
    - Support percentages / formulas
    - Return structured success / error
    - Do not require loading Qwen Coder
    - Do not autonomously invoke other tools

Implementation:
    AST-based safe expression evaluator with whitelisted
    operators and functions only.
"""

from __future__ import annotations
import ast
import math
import operator
from typing import Any


# ── Whitelisted operators ─────────────────────────────────────────────────────

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# ── Whitelisted functions ─────────────────────────────────────────────────────

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
}

# ── Whitelisted constants ────────────────────────────────────────────────────

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


# ── AST Evaluator ─────────────────────────────────────────────────────────────

def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with whitelisted operations only."""

    # Numeric literal
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value!r}")

    # Whitelisted name (pi, e)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name: '{node.id}'")

    # Unary operator (+x, -x)
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))

    # Binary operator (a + b, a * b, etc.)
    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_fn(left, right)

    # Whitelisted function call: abs(x), round(x, n), min(a, b, c), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed (e.g. abs(x))")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Non-whitelisted function: '{func_name}'. "
                f"Allowed: {sorted(_ALLOWED_FUNCTIONS.keys())}"
            )
        fn = _ALLOWED_FUNCTIONS[func_name]
        args = [_safe_eval(arg) for arg in node.args]
        # No keyword args supported for safety
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in math expressions")
        return fn(*args)

    # List literal (for min/max/sum with lists)
    if isinstance(node, ast.List):
        return [_safe_eval(el) for el in node.elts]

    # Tuple literal
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(el) for el in node.elts)

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# ── Public API ────────────────────────────────────────────────────────────────

def tool_calculate(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A string containing an arithmetic expression.
                    Examples: "2 + 3", "sqrt(144)", "round(3.14159, 2)",
                              "min(10, 20, 5)", "(100 * 0.15) + 50"

    Returns:
        {"status": "success", "expression": <str>, "result": <number>}
        or
        {"status": "error", "expression": <str>, "error": <str>}
    """
    expression = expression.strip()
    if not expression:
        return {
            "status": "error",
            "expression": expression,
            "error": "Empty expression",
        }

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {
            "status": "error",
            "expression": expression,
            "error": f"Invalid syntax: {exc}",
        }

    try:
        result = _safe_eval(tree.body)
    except ZeroDivisionError:
        return {
            "status": "error",
            "expression": expression,
            "error": "Division by zero",
        }
    except (ValueError, TypeError, OverflowError) as exc:
        return {
            "status": "error",
            "expression": expression,
            "error": str(exc),
        }

    return {
        "status": "success",
        "expression": expression,
        "result": result,
    }


def _inspect_ast(node: ast.AST) -> tuple[list[str], list[str], list[str]]:
    """Extract operators, functions, and constants from AST."""
    operators: list[str] = []
    functions: list[str] = []
    constants: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp):
            operators.append(type(n.op).__name__)
        elif isinstance(n, ast.UnaryOp):
            operators.append(type(n.op).__name__)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            functions.append(n.func.id)
        elif isinstance(n, ast.Name) and n.id in _ALLOWED_NAMES:
            constants.append(n.id)
    return sorted(set(operators)), sorted(set(functions)), sorted(set(constants))


def tool_calculate_socket(expression: str) -> dict:
    """Evaluate math expression and return complete Component M socket."""
    import time
    from core.sockets import build_math_socket, build_timing_payload, build_error_payload

    t0 = time.perf_counter()
    expr = expression.strip()
    if not expr:
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        err = build_error_payload("EMPTY_EXPRESSION", "ValueError", "Empty expression", recoverable=False)
        return build_math_socket(expression=expression, timing=timing, error=err)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        err = build_error_payload("SYNTAX_ERROR", "SyntaxError", str(exc), recoverable=True, retryable=True, exc=exc)
        return build_math_socket(expression=expression, timing=timing, error=err)

    ops, fns, consts = _inspect_ast(tree.body)

    try:
        val = _safe_eval(tree.body)
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        return build_math_socket(
            expression=expression,
            value=val,
            operators_used=ops,
            functions_used=fns,
            constants_used=consts,
            timing=timing,
        )
    except ZeroDivisionError as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        err = build_error_payload("DIVISION_BY_ZERO", "ZeroDivisionError", "Division by zero", exc=exc)
        return build_math_socket(expression=expression, operators_used=ops, functions_used=fns, constants_used=consts, timing=timing, error=err)
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        err = build_error_payload("MATH_EVAL_ERROR", type(exc).__name__, str(exc), exc=exc)
        return build_math_socket(expression=expression, operators_used=ops, functions_used=fns, constants_used=consts, timing=timing, error=err)


# ─── Registration Helper ─────────────────────────────────────────────────────

def register_math_tools(registry) -> None:
    """Register safe math tools with a ToolRegistry.

    Canonical registration (matches tools.json, Gemma-visible):
        math / calculate
    """
    registry.register("math", "calculate", tool_calculate)

