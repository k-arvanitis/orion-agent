"""
Unit tests for the guard layer.

Tests cover:
  - PII stripping (CPF, phone numbers)
  - Clean responses pass through unchanged
  - GuardResult.clean is always True (hallucination check removed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "guard", pathlib.Path(__file__).parents[1] / "agent" / "guard.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
apply = _mod.apply


def test_cpf_is_stripped():
    result = apply("Your CPF 123.456.789-00 has been verified.", "")
    assert "123.456.789-00" not in result.text


def test_phone_is_stripped():
    result = apply("Contact us at (11) 91234-5678 for help.", "")
    assert "(11) 91234-5678" not in result.text


def test_clean_text_unchanged():
    text = "Your order has been delivered."
    result = apply(text, "delivered")
    assert result.text == text


def test_clean_is_always_true():
    result = apply("Your total is R$999.99.", "total: 38.71")
    assert result.clean is True
    assert result.hallucinated == []


def test_tool_output_optional():
    result = apply("Order delivered.")
    assert result.text == "Order delivered."
    assert result.clean is True
