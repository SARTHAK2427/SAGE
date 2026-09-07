"""
Tests for tools/math_tool.py — safe deterministic calculator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from tools.math_tool import tool_calculate


class TestMathTool:

    # ── Valid expressions ─────────────────────────────────────────────

    def test_basic_addition(self):
        r = tool_calculate("2 + 3")
        assert r["status"] == "success"
        assert r["result"] == 5

    def test_basic_subtraction(self):
        r = tool_calculate("10 - 4")
        assert r["status"] == "success"
        assert r["result"] == 6

    def test_multiplication(self):
        r = tool_calculate("6 * 7")
        assert r["status"] == "success"
        assert r["result"] == 42

    def test_division(self):
        r = tool_calculate("15 / 4")
        assert r["status"] == "success"
        assert r["result"] == 3.75

    def test_floor_division(self):
        r = tool_calculate("15 // 4")
        assert r["status"] == "success"
        assert r["result"] == 3

    def test_modulo(self):
        r = tool_calculate("17 % 5")
        assert r["status"] == "success"
        assert r["result"] == 2

    def test_power(self):
        r = tool_calculate("2 ** 10")
        assert r["status"] == "success"
        assert r["result"] == 1024

    def test_parentheses(self):
        r = tool_calculate("(2 + 3) * 4")
        assert r["status"] == "success"
        assert r["result"] == 20

    def test_negative_numbers(self):
        r = tool_calculate("-5 + 3")
        assert r["status"] == "success"
        assert r["result"] == -2

    def test_percentage_formula(self):
        r = tool_calculate("100 * 0.15 + 50")
        assert r["status"] == "success"
        assert r["result"] == 65.0

    def test_complex_expression(self):
        r = tool_calculate("(100 + 200) * 0.08 / 2")
        assert r["status"] == "success"
        assert abs(r["result"] - 12.0) < 0.001

    # ── Whitelisted functions ─────────────────────────────────────────

    def test_abs_function(self):
        r = tool_calculate("abs(-42)")
        assert r["status"] == "success"
        assert r["result"] == 42

    def test_round_function(self):
        r = tool_calculate("round(3.14159, 2)")
        assert r["status"] == "success"
        assert r["result"] == 3.14

    def test_min_function(self):
        r = tool_calculate("min(10, 20, 5)")
        assert r["status"] == "success"
        assert r["result"] == 5

    def test_max_function(self):
        r = tool_calculate("max(1, 99, 50)")
        assert r["status"] == "success"
        assert r["result"] == 99

    def test_sum_function(self):
        r = tool_calculate("sum([1, 2, 3, 4])")
        assert r["status"] == "success"
        assert r["result"] == 10

    def test_sqrt_function(self):
        r = tool_calculate("sqrt(144)")
        assert r["status"] == "success"
        assert r["result"] == 12.0

    def test_pi_constant(self):
        r = tool_calculate("pi * 2")
        assert r["status"] == "success"
        assert abs(r["result"] - 6.283185) < 0.001

    # ── Error cases ───────────────────────────────────────────────────

    def test_divide_by_zero(self):
        r = tool_calculate("10 / 0")
        assert r["status"] == "error"
        assert "zero" in r["error"].lower()

    def test_empty_expression(self):
        r = tool_calculate("")
        assert r["status"] == "error"
        assert "Empty" in r["error"]

    def test_invalid_syntax(self):
        r = tool_calculate("2 + + +")
        assert r["status"] == "error"

    def test_non_whitelisted_function_rejected(self):
        r = tool_calculate("exec('print(1)')")
        assert r["status"] == "error"
        assert "Non-whitelisted" in r["error"] or "not allowed" in r["error"].lower() or "syntax" in r["error"].lower()

    def test_import_rejected(self):
        r = tool_calculate("__import__('os').system('ls')")
        assert r["status"] == "error"

    def test_arbitrary_name_rejected(self):
        r = tool_calculate("foo + 1")
        assert r["status"] == "error"
        assert "Unknown name" in r["error"]

    def test_string_literal_rejected(self):
        r = tool_calculate("'hello' + 'world'")
        assert r["status"] == "error"
