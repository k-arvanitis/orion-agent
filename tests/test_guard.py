"""
Unit tests for the guard layer.

Tests cover:
  - PII stripping (CPF, phone numbers)
  - Hallucination detection (numbers in response not in tool output)
  - Clean responses pass through unchanged
  - GuardResult.clean flag behaviour
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util, pathlib

_spec = importlib.util.spec_from_file_location(
    "guard", pathlib.Path(__file__).parents[1] / "agent" / "guard.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
apply = _mod.apply


# ---------------------------------------------------------------------------
# PII stripping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


def test_hallucinated_number_detected():
    result = apply("Your total is R$999.99.", "total: 38.71")
    assert not result.clean
    assert "999.99" in result.hallucinated


def test_correct_number_passes():
    result = apply("Your total is R$38.71.", "total: 38.71")
    assert result.clean


def test_multiple_correct_numbers_pass():
    result = apply(
        "Order placed on 2017-10-02, delivered on 2017-10-10.",
        "2017-10-02 2017-10-10",
    )
    assert result.clean


def test_one_hallucinated_among_correct():
    result = apply(
        "Your total is R$38.71 but the refund is R$500.00.",
        "total: 38.71",
    )
    assert not result.clean
    assert "500.00" in result.hallucinated


# ---------------------------------------------------------------------------
# GuardResult properties
# ---------------------------------------------------------------------------


def test_clean_flag_true_when_no_hallucination():
    result = apply("Order delivered.", "delivered")
    assert result.clean is True


def test_clean_flag_false_when_hallucination():
    result = apply("Total is R$999.99.", "38.71")
    assert result.clean is False
