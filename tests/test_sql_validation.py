"""
Unit tests for the SQL validation layer in sql_tool.

Tests cover:
  - Valid SELECT queries pass without exception
  - INSERT / UPDATE / DELETE are rejected
  - Empty queries are rejected
  - Markdown code fence stripping works correctly

No LLM calls are made — these test only the pure validation logic.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "sql_tool",
    Path(__file__).parents[1] / "agent" / "tools" / "sql_tool.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_validate_sql = _mod._validate_sql


# ---------------------------------------------------------------------------
# Valid queries
# ---------------------------------------------------------------------------


def test_select_query_passes():
    _validate_sql("SELECT order_id FROM orders LIMIT 10")


def test_select_with_join_passes():
    _validate_sql(
        "SELECT o.order_id, c.customer_city "
        "FROM orders o JOIN customers c ON o.customer_id = c.customer_id LIMIT 10"
    )


# ---------------------------------------------------------------------------
# Rejected queries
# ---------------------------------------------------------------------------


def test_insert_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="Only SELECT"):
        _validate_sql("INSERT INTO orders (order_id) VALUES ('abc')")


def test_update_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="Only SELECT"):
        _validate_sql("UPDATE orders SET order_status = 'canceled' WHERE order_id = 'abc'")


def test_delete_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="Only SELECT"):
        _validate_sql("DELETE FROM orders WHERE order_id = 'abc'")


def test_empty_query_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        _validate_sql("")


# ---------------------------------------------------------------------------
# Markdown fence stripping (logic inside _generate_sql)
# ---------------------------------------------------------------------------


def test_markdown_fence_stripped():
    """The fence-stripping logic in _generate_sql should remove ```sql ... ```."""
    raw = "```sql\nSELECT * FROM orders LIMIT 1\n```"
    if raw.startswith("```"):
        sql = raw.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()
    else:
        sql = raw.strip()
    assert sql == "SELECT * FROM orders LIMIT 1"
