"""
Text2SQL tool — translates natural language to SQL and queries Supabase.

Flow:
  1. Groq generates a SELECT query from the question + schema context.
  2. sqlparse validates it (SELECT only, no dangerous statements).
  3. SQLAlchemy executes it with a 100-row limit and 5s timeout.
  4. Groq interprets the raw rows into a plain-language answer.
  5. On SQL error, one retry with the error message fed back to Groq.
"""

import os
import textwrap

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sqlparse

ALLOWED_STATEMENTS = {"SELECT"}
ROW_LIMIT = 100
QUERY_TIMEOUT = 5  # seconds

_engine = None
_llm: ChatGroq | None = None

SCHEMA = textwrap.dedent("""
    orders(order_id, customer_id, order_status, order_purchase_timestamp,
           order_delivered_customer_date, order_estimated_delivery_date)
    order_items(order_id, product_id, seller_id, price, freight_value, shipping_limit_date)
    customers(customer_id, customer_unique_id, customer_city, customer_state)
    order_payments(order_id, payment_sequential, payment_type, payment_installments, payment_value)
    order_reviews(review_id, order_id, review_score, review_comment_message, review_creation_date)
    products(product_id, product_category_name)
    product_category_translations(product_category_name, product_category_name_english)
    sellers(seller_id, seller_city, seller_state)
""").strip()


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            os.environ["DATABASE_URL"],
            connect_args={"connect_timeout": QUERY_TIMEOUT, "options": "-c statement_timeout=5000"},
        )
    return _engine


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    return _llm


def _generate_sql(question: str, error_context: str = "") -> str:
    error_hint = f"\n\nPrevious attempt failed with: {error_context}\nFix the query." if error_context else ""
    prompt = textwrap.dedent(f"""
        You are a SQL expert. Generate a single PostgreSQL SELECT query to answer the question.

        Schema:
        {SCHEMA}

        Rules:
        - SELECT only. No INSERT, UPDATE, DELETE, DROP, or any DDL.
        - Use table aliases for clarity.
        - LIMIT {ROW_LIMIT} rows maximum.
        - Return ONLY the raw SQL query — no markdown, no explanation.
        {error_hint}

        Question: {question}
    """).strip()

    response = _get_llm().invoke(prompt)
    sql = response.content.strip()
    # Strip markdown code fences if model wraps the query
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip()


def _validate_sql(sql: str) -> None:
    statements = sqlparse.parse(sql)
    if not statements:
        raise ValueError("Empty query.")
    for stmt in statements:
        kind = stmt.get_type()
        if kind not in ALLOWED_STATEMENTS:
            raise ValueError(f"Only SELECT queries are allowed. Got: {kind}")


def _execute_sql(sql: str) -> list[dict]:
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def _interpret(question: str, rows: list[dict]) -> str:
    if not rows:
        return "No matching records were found in the database."

    prompt = textwrap.dedent(f"""
        A customer asked: "{question}"

        Here are the database results:
        {rows}

        Write a clear, concise answer for the customer. Do not expose technical details.
        Mention the data source (e.g., "According to our order records...").
    """).strip()

    return _get_llm().invoke(prompt).content.strip()


@tool
def query_database(question: str) -> str:
    """
    Query the ShopNova orders database to answer questions about order status,
    delivery dates, payments, products, sellers, or customer information.

    Args:
        question: The customer's question about their order or account.

    Returns:
        A plain-language answer based on the database results.
    """
    sql = _generate_sql(question)

    try:
        _validate_sql(sql)
        rows = _execute_sql(sql)
    except (ValueError, SQLAlchemyError) as e:
        # One retry with error context
        sql = _generate_sql(question, error_context=str(e))
        try:
            _validate_sql(sql)
            rows = _execute_sql(sql)
        except (ValueError, SQLAlchemyError) as retry_err:
            return f"I was unable to retrieve that information. Error: {retry_err}"

    return _interpret(question, rows)
