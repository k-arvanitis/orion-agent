"""
Load Olist CSV files into Supabase (PostgreSQL).

Infers column types from each CSV (numerics, timestamps, text), creates the
table with a proper primary key if it doesn't exist, and upserts rows in
batches. Table names are derived by stripping the 'olist_' prefix and
'_dataset' suffix from the filename.

  olist_customers_dataset.csv           -> customers
  olist_order_items_dataset.csv         -> order_items
  olist_order_payments_dataset.csv      -> order_payments
  olist_order_reviews_dataset.csv       -> order_reviews
  olist_orders_dataset.csv              -> orders
  olist_products_dataset.csv            -> products
  olist_sellers_dataset.csv             -> sellers
  olist_geolocation_dataset.csv         -> geolocation
  product_category_name_translation.csv -> product_category_translations

Usage:
    uv run python load_customer_data.py
    uv run python load_customer_data.py --data data/customer-data
    uv run python load_customer_data.py --data data/customer-data --batch-size 500
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

DEFAULT_DATA_DIR = "data/customer-data"
DEFAULT_BATCH_SIZE = 1000

# Explicit primary keys per table (composite keys as tuples)
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "customers":                    ("customer_id",),
    "orders":                       ("order_id",),
    "order_items":                  ("order_id", "order_item_id"),
    "order_payments":               ("order_id", "payment_sequential"),
    "order_reviews":                ("review_id",),
    "products":                     ("product_id",),
    "sellers":                      ("seller_id",),
    "geolocation":                  ("geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"),
    "product_category_translations": ("product_category_name",),
}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def csv_to_table_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"^olist_", "", name)
    name = re.sub(r"_dataset$", "", name)
    return name


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def infer_series_type(series: pd.Series) -> tuple[pd.Series, str]:
    """
    Attempt to cast a text series to a more specific type.
    Returns the cast series and its PostgreSQL type string.
    """
    sample = series.dropna()

    # Timestamp — column name contains a date/time keyword
    if any(k in series.name for k in ("date", "timestamp", "time")):
        try:
            return pd.to_datetime(series, errors="raise"), "TIMESTAMP"
        except Exception:
            pass

    # Integer
    try:
        cast = pd.to_numeric(sample, errors="raise")
        if (cast == cast.astype("int64")).all():
            return pd.to_numeric(series, errors="coerce").astype("Int64"), "BIGINT"
    except Exception:
        pass

    # Float
    try:
        pd.to_numeric(sample, errors="raise")
        return pd.to_numeric(series, errors="coerce"), "DOUBLE PRECISION"
    except Exception:
        pass

    return series, "TEXT"


def infer_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a typed DataFrame and a {column: pg_type} mapping."""
    typed_cols = {}
    pg_types = {}
    for col in df.columns:
        typed_series, pg_type = infer_series_type(df[col])
        typed_cols[col] = typed_series
        pg_types[col] = pg_type
    return pd.DataFrame(typed_cols), pg_types


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def create_table_sql(table: str, pg_types: dict[str, str], pk: tuple[str, ...]) -> str:
    col_defs = [f'  "{col}" {pg_types[col]}' for col in pg_types]
    if pk:
        pk_cols = ", ".join(f'"{c}"' for c in pk)
        col_defs.append(f"  PRIMARY KEY ({pk_cols})")
    cols = ",\n".join(col_defs)
    return f'CREATE TABLE IF NOT EXISTS "{table}" (\n{cols}\n);'


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def load_csv(path: Path, conn, batch_size: int) -> None:
    table = csv_to_table_name(path.name)
    pk = PRIMARY_KEYS.get(table, ())

    print(f"  {path.name:<50} -> {table}")

    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip().lower().replace(" ", "_").lstrip("\ufeff") for c in raw.columns]

    df, pg_types = infer_dataframe(raw)
    df = df.where(pd.notna(df), None)

    conflict_clause = ""
    if pk:
        pk_cols = ", ".join(f'"{c}"' for c in pk)
        conflict_clause = f"ON CONFLICT ({pk_cols}) DO NOTHING"

    col_names = ", ".join(f'"{c}"' for c in df.columns)
    sql = f'INSERT INTO "{table}" ({col_names}) VALUES %s {conflict_clause}'

    def _to_python(v):
        try:
            if pd.isnull(v):
                return None
        except (TypeError, ValueError):
            pass
        return v.item() if hasattr(v, "item") else v

    rows = [tuple(_to_python(v) for v in row) for row in df.itertuples(index=False, name=None)]
    total = len(rows)

    with conn.cursor() as cur:
        cur.execute(create_table_sql(table, pg_types, pk))

        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            execute_values(cur, sql, batch)
            done = min(start + batch_size, total)
            print(f"    {done:>7,} / {total:,} rows", end="\r")

    conn.commit()
    print(f"    {total:>7,} rows loaded  ")


def load_all(data_dir: Path, conn, batch_size: int) -> None:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in '{data_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(csv_files)} files from '{data_dir}' ...\n")
    for path in csv_files:
        load_csv(path, conn, batch_size)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="load_customer_data",
        description="Load Olist CSV files into Supabase with type inference and upsert.",
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help=f"Directory containing CSV files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per insert batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"Data directory '{args.data}' does not exist.")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL must be set in .env", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(database_url)
    try:
        load_all(args.data, conn, args.batch_size)
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
