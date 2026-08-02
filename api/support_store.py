# ruff: noqa: E501
"""Database-backed support demo data and conversation operations."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection, Engine

from api.policy_store import search_policy_documents

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = f"sqlite:///{ROOT_DIR / 'data' / 'orion_support.db'}"

metadata = MetaData()

customers = Table(
    "support_customers",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(160), nullable=False),
    Column("initials", String(8), nullable=False),
    Column("email", String(320), nullable=False, unique=True),
    Column("phone", String(60), nullable=False),
    Column("location", String(160), nullable=False),
    Column("customer_since", Date, nullable=False),
    Column("match_method", String(120), nullable=False),
)

customer_tags = Table(
    "support_customer_tags",
    metadata,
    Column("customer_id", ForeignKey("support_customers.id", ondelete="CASCADE"), primary_key=True),
    Column("tag", String(80), primary_key=True),
)

products = Table(
    "support_products",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("sku", String(80), nullable=False, unique=True),
    Column("name", String(200), nullable=False),
    Column("category", String(120), nullable=False),
)

orders = Table(
    "support_orders",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("parcel_id", String(80), nullable=False, unique=True),
    Column("customer_id", ForeignKey("support_customers.id"), nullable=False),
    Column("product_id", ForeignKey("support_products.id"), nullable=False),
    Column("ordered_at", Date, nullable=False),
    Column("total_cents", Integer, nullable=False),
    Column("currency", String(3), nullable=False, default="USD"),
    Column("status", String(40), nullable=False),
    Column("detail", String(240), nullable=False),
)

conversations = Table(
    "support_conversations",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("customer_id", ForeignKey("support_customers.id"), nullable=False),
    Column("subject", String(200), nullable=False),
    Column("preview", String(400), nullable=False),
    Column("status", String(40), nullable=False),
    Column("channel", String(20), nullable=False),
    Column("action_needed", String(240)),
    Column("source", String(20), nullable=False, default="demo"),
    Column("unread", Integer, nullable=False, default=0),
    Column("resolved_by", String(20)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "status IN ('Waiting for support', 'Resolved by Orion')",
        name="support_conversation_status",
    ),
    CheckConstraint(
        "channel IN ('Email', 'Chat', 'Voice')",
        name="support_conversation_channel",
    ),
)

messages = Table(
    "support_messages",
    metadata,
    Column("id", String(40), primary_key=True),
    Column(
        "conversation_id",
        ForeignKey("support_conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", String(20), nullable=False),
    Column("sender", String(120), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "role IN ('customer', 'orion', 'agent', 'handoff')",
        name="support_message_role",
    ),
    UniqueConstraint("conversation_id", "id", name="support_message_conversation_id"),
)

conversation_traces = Table(
    "support_conversation_traces",
    metadata,
    Column(
        "conversation_id",
        ForeignKey("support_conversations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("details_json", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

VISITOR_CUSTOMER_ID = "CUS-VISITOR"

_engine: Engine | None = None
_engine_url: str | None = None
_init_lock = Lock()
_initialized = False


def _database_url() -> str:
    return os.getenv("SUPPORT_DATABASE_URL", DEFAULT_DATABASE_URL)


def _get_engine() -> Engine:
    global _engine, _engine_url
    url = _database_url()
    if _engine is None or _engine_url != url:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        _engine_url = url
    return _engine


def configure_database(database_url: str) -> None:
    """Switch the support repository URL; primarily useful for tests."""
    global _engine, _engine_url, _initialized
    os.environ["SUPPORT_DATABASE_URL"] = database_url
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _engine_url = None
    _initialized = False


CUSTOMER_SEED = [
    ("CUS-2048-MT", "Maya Torres", "MT", "maya.torres@example.com", "+55 11 98234 0182", "São Paulo, Brazil", date(2023, 5, 1), "Email and order number"),
    ("CUS-8814-LC", "Liam Chen", "LC", "liam.chen@example.com", "+1 415 555 0138", "San Francisco, USA", date(2026, 1, 1), "Email address"),
    ("CUS-3341-SM", "Sofia Martinez", "SM", "sofia.m@example.com", "+34 612 44 98 21", "Madrid, Spain", date(2024, 9, 1), "Email and payment reference"),
    ("CUS-7290-NW", "Noah Williams", "NW", "noah.w@example.com", "+44 7700 900214", "Bristol, UK", date(2025, 3, 1), "Email and order number"),
    ("CUS-6102-LC", "Lucas Costa", "LC", "lucas.costa@example.com", "+55 21 99128 4110", "Rio de Janeiro, Brazil", date(2025, 7, 1), "Parcel number"),
    ("CUS-4458-EO", "Emma Oliveira", "EO", "emma.o@example.com", "+55 31 98874 2031", "Belo Horizonte, Brazil", date(2023, 12, 1), "Email address"),
]

PRODUCT_SEED = [
    ("PROD-BAG-01", "BAG-LINEN-WKND", "Linen weekend bag", "Travel"),
    ("PROD-POUCH-01", "POUCH-CANVAS", "Canvas travel pouch", "Travel"),
    ("PROD-JACKET-01", "JACKET-HARBOR-M", "Harbor field jacket · M", "Apparel"),
    ("PROD-CERAMIC-01", "SERVING-CERAMIC", "Ceramic serving set", "Home"),
    ("PROD-WALLET-01", "WALLET-LEATHER", "Leather card wallet", "Accessories"),
    ("PROD-THROW-01", "THROW-WOVEN-BLUE", "Woven cotton throw · Blue", "Home"),
    ("PROD-TRAY-01", "TRAY-OAK-BEDSIDE", "Oak bedside tray", "Home"),
]

ORDER_SEED = [
    ("416e49799e9260d93c8f636ce6661a55", "BR-SP-8042-1197", "CUS-2048-MT", "PROD-BAG-01", date(2026, 7, 24), 12840, "USD", "Delayed", "Expected Jul 30 · Standard shipping"),
    ("SN-4108", "BR-SP-7421-0084", "CUS-2048-MT", "PROD-POUCH-01", date(2026, 4, 12), 8400, "USD", "Delivered", "Delivered Apr 17"),
    ("SN-4472", "US-CA-5520-1841", "CUS-8814-LC", "PROD-JACKET-01", date(2026, 7, 21), 14600, "USD", "Delivered", "Delivered Jul 27"),
    ("SN-4588", "ES-MD-1170-9924", "CUS-3341-SM", "PROD-CERAMIC-01", date(2026, 7, 29), 9650, "USD", "Processing", "Payment review in progress"),
    ("SN-3911", "UK-BR-2401-7736", "CUS-7290-NW", "PROD-WALLET-01", date(2025, 11, 16), 7200, "USD", "Delivered", "Delivered Nov 21"),
    ("SN-4521", "BR-RJ-5914-2248", "CUS-6102-LC", "PROD-THROW-01", date(2026, 7, 19), 11800, "USD", "Delivered", "Delivered Jul 25"),
    ("SN-4490", "BR-MG-6820-1453", "CUS-4458-EO", "PROD-TRAY-01", date(2026, 7, 17), 16400, "USD", "Delivered", "Delivered Jul 23"),
]

CONVERSATION_SEED = [
    ("SN-4821", "CUS-2048-MT", "Order still hasn’t arrived", "It was meant to arrive on Thursday and the tracking hasn’t moved…", "Waiting for support", "Chat", "Approve a shipping refund or cancellation", 2, 2),
    ("SN-4819", "CUS-8814-LC", "Need a different size", "The jacket is lovely, but it runs smaller than expected.", "Resolved by Orion", "Email", None, 1, 12),
    ("SN-4816", "CUS-3341-SM", "Charged twice at checkout", "I can see two identical charges on my card statement.", "Resolved by Orion", "Chat", None, 3, 18),
    ("SN-4812", "CUS-7290-NW", "Warranty coverage question", "Does the two-year warranty cover a broken clasp?", "Waiting for support", "Email", "Review the warranty photo and approve replacement", 0, 32),
    ("SN-4806", "CUS-6102-LC", "Wrong item in my parcel", "I ordered the blue throw but received the beige one.", "Waiting for support", "Voice", "Approve a free replacement for the wrong item", 1, 46),
    ("SN-4798", "CUS-4458-EO", "Everything arrived perfectly", "Just wanted to say thank you for the quick help!", "Resolved by Orion", "Email", None, 0, 60),
]

MESSAGE_SEED = {
    "SN-4821": [
        ("customer", "Maya", "Hi, my order was supposed to arrive on Thursday, but the tracking hasn’t moved in four days. I need it for a trip this weekend."),
        ("orion", "Orion", "I’m sorry this is running late, Maya. I checked your order and the carrier has reported a delay at the São Paulo distribution center. The latest estimate is August 3."),
        ("customer", "Maya", "That’s after I leave. Can I get a refund for the shipping or cancel it?"),
        ("handoff", "Orion", "Maya leaves on August 2. The parcel is four days late, and Orion cannot approve the requested refund. Support needs to approve a shipping refund or cancel the order."),
    ],
    "SN-4819": [
        ("customer", "Liam", "The jacket is lovely, but it runs smaller than expected. Can I exchange the medium for a large?"),
        ("orion", "Orion", "Absolutely. Your order is within the 30-day exchange window and the large is in stock. I’ve started the exchange and emailed your prepaid return label."),
    ],
    "SN-4816": [
        ("customer", "Sofia", "I can see two identical charges from ShopNova on my card. Can you confirm I wasn’t billed twice?"),
        ("orion", "Orion", "I found one completed payment and one temporary authorization. The pending authorization should disappear automatically within 3–5 business days."),
    ],
    "SN-4812": [
        ("customer", "Noah", "The clasp on my wallet has stopped closing. Is that covered by the two-year warranty?"),
        ("orion", "Orion", "A failed clasp is normally covered. I received your photo, but I can’t confirm whether the damage is a manufacturing fault, so I’ve passed it to our support team for review."),
        ("handoff", "Orion", "Warranty and purchase date are valid. The customer photo needs human review before a replacement can be approved."),
    ],
    "SN-4806": [
        ("customer", "Lucas", "I ordered the blue woven throw but the one in the parcel is beige. What’s the quickest way to swap it?"),
        ("orion", "Orion", "I’m sorry about the mix-up. I found the order from your parcel number, but replacing the incorrect item needs approval from our support team."),
        ("handoff", "Orion", "Parcel BR-RJ-5914-2248 contains the wrong colour. Support needs to approve a free replacement and return label."),
    ],
    "SN-4798": [
        ("customer", "Emma", "Everything arrived perfectly. Thank you for sorting out the address change so quickly!"),
        ("orion", "Orion", "You’re very welcome, Emma. I’m glad it reached the right address. I’ve marked this conversation as resolved."),
    ],
}


def _seed(conn: Connection) -> None:
    if conn.execute(select(func.count()).select_from(customers)).scalar_one() > 0:
        return

    conn.execute(
        insert(customers),
        [
            dict(zip(("id", "name", "initials", "email", "phone", "location", "customer_since", "match_method"), row))
            for row in CUSTOMER_SEED
        ],
    )
    conn.execute(
        insert(customer_tags),
        [
            {"customer_id": customer_id, "tag": tag}
            for customer_id, tag in [
                ("CUS-2048-MT", "Repeat customer"),
                ("CUS-8814-LC", "New customer"),
                ("CUS-3341-SM", "Repeat customer"),
                ("CUS-7290-NW", "Warranty"),
                ("CUS-6102-LC", "Voice customer"),
                ("CUS-4458-EO", "Loyal customer"),
            ]
        ],
    )
    conn.execute(
        insert(products),
        [dict(zip(("id", "sku", "name", "category"), row)) for row in PRODUCT_SEED],
    )
    conn.execute(
        insert(orders),
        [
            dict(zip(("id", "parcel_id", "customer_id", "product_id", "ordered_at", "total_cents", "currency", "status", "detail"), row))
            for row in ORDER_SEED
        ],
    )

    now = datetime.now(timezone.utc)
    conversation_rows = []
    message_rows = []
    for conversation in CONVERSATION_SEED:
        (
            conversation_id,
            customer_id,
            subject,
            preview,
            status,
            channel,
            action_needed,
            unread,
            minutes_ago,
        ) = conversation
        created_at = now - timedelta(minutes=minutes_ago + 4)
        updated_at = now - timedelta(minutes=minutes_ago)
        conversation_rows.append(
            {
                "id": conversation_id,
                "customer_id": customer_id,
                "subject": subject,
                "preview": preview,
                "status": status,
                "channel": channel,
                "action_needed": action_needed,
                "source": "seed",
                "unread": unread,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        seeded_messages = MESSAGE_SEED[conversation_id]
        for index, (role, sender, content) in enumerate(seeded_messages):
            message_rows.append(
                {
                    "id": f"{conversation_id.lower()}-{index + 1}",
                    "conversation_id": conversation_id,
                    "role": role,
                    "sender": sender,
                    "content": content,
                    "created_at": created_at + timedelta(minutes=index),
                }
            )

    conn.execute(insert(conversations), conversation_rows)
    conn.execute(insert(messages), message_rows)


def _ensure_visitor_customer(conn: Connection) -> None:
    exists = conn.execute(
        select(customers.c.id).where(customers.c.id == VISITOR_CUSTOMER_ID)
    ).first()
    if exists:
        return
    conn.execute(
        insert(customers).values(
            id=VISITOR_CUSTOMER_ID,
            name="ShopNova visitor",
            initials="SV",
            email="visitor@shopnova.invalid",
            phone="Not provided",
            location="Not provided",
            customer_since=date.today(),
            match_method="No account required",
        )
    )


def _ensure_conversation_columns(engine: Engine) -> None:
    """Add backward-compatible columns to an existing demo database."""
    column_names = {
        column["name"]
        for column in inspect(engine).get_columns("support_conversations")
    }
    if "resolved_by" not in column_names:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE support_conversations "
                    "ADD COLUMN resolved_by VARCHAR(20)"
                )
            )


def initialize_database() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        engine = _get_engine()
        if _engine_url and _engine_url.startswith("sqlite"):
            Path(_engine_url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        metadata.create_all(engine)
        _ensure_conversation_columns(engine)
        with engine.begin() as conn:
            _seed(conn)
            _ensure_visitor_customer(conn)
        _initialized = True


def _format_money(cents: int, currency: str) -> str:
    amount = cents / 100
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{amount:,.2f}" if cents % 100 else f"{symbol}{amount:,.0f}"


def _relative_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    minutes = max(0, int((datetime.now(timezone.utc) - value).total_seconds() / 60))
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h" if hours < 24 else f"{hours // 24}d"


def _customer_payload(conn: Connection, customer_row) -> dict:
    customer = customer_row._mapping
    order_rows = conn.execute(
        select(orders, products.c.name.label("product_name"))
        .join(products, products.c.id == orders.c.product_id)
        .where(orders.c.customer_id == customer["id"])
        .order_by(orders.c.ordered_at.desc())
    ).all()
    tags = conn.execute(
        select(customer_tags.c.tag).where(customer_tags.c.customer_id == customer["id"])
    ).scalars().all()
    total_cents = sum(row._mapping["total_cents"] for row in order_rows)
    currency = order_rows[0]._mapping["currency"] if order_rows else "USD"
    return {
        "id": customer["id"],
        "name": customer["name"],
        "initials": customer["initials"],
        "email": customer["email"],
        "phone": customer["phone"],
        "location": customer["location"],
        "since": customer["customer_since"].strftime("%B %Y"),
        "orderCount": len(order_rows),
        "totalSpent": _format_money(total_cents, currency),
        "tags": list(tags),
        "matchedBy": customer["match_method"],
        "orders": [
            {
                "id": row._mapping["id"],
                "parcelId": row._mapping["parcel_id"],
                "date": row._mapping["ordered_at"].strftime("%b %-d, %Y"),
                "total": _format_money(row._mapping["total_cents"], row._mapping["currency"]),
                "status": row._mapping["status"],
                "item": row._mapping["product_name"],
                "detail": row._mapping["detail"],
            }
            for row in order_rows
        ],
    }


def _database_technical_details(customer: dict) -> dict:
    order = customer["orders"][0] if customer["orders"] else None
    return {
        "tools": [
            {
                "name": "customer_lookup",
                "label": "Customer lookup",
                "result": customer["id"],
            },
            *(
                [
                    {
                        "name": "order_lookup",
                        "label": "Order lookup",
                        "result": order["id"],
                    }
                ]
                if order
                else []
            ),
        ],
        "records": [
            {"source": "support_customers", "record": customer["id"]},
            *(
                [{"source": "support_orders", "record": order["id"]}]
                if order
                else []
            ),
        ],
        "documents": [],
    }


def _conversation_payload(conn: Connection, conversation_row) -> dict:
    conversation = conversation_row._mapping
    customer_row = conn.execute(
        select(customers).where(customers.c.id == conversation["customer_id"])
    ).one()
    customer = _customer_payload(conn, customer_row)
    message_rows = conn.execute(
        select(messages)
        .where(messages.c.conversation_id == conversation["id"])
        .order_by(messages.c.created_at, messages.c.id)
    ).all()
    trace_row = conn.execute(
        select(conversation_traces.c.details_json).where(
            conversation_traces.c.conversation_id == conversation["id"]
        )
    ).first()
    technical_details = (
        json.loads(trace_row._mapping["details_json"])
        if trace_row
        else _database_technical_details(customer)
    )
    display_status = (
        "Resolved by support"
        if conversation["resolved_by"] == "support"
        else conversation["status"]
    )
    return {
        "id": conversation["id"],
        "subject": conversation["subject"],
        "preview": conversation["preview"],
        "time": _relative_time(conversation["updated_at"]),
        "unread": conversation["unread"],
        "status": display_status,
        "channel": conversation["channel"],
        "actionNeeded": conversation["action_needed"],
        "source": conversation["source"],
        "customer": customer,
        "technicalDetails": technical_details,
        "messages": [
            {
                "id": row._mapping["id"],
                "role": row._mapping["role"],
                "sender": row._mapping["sender"],
                "time": row._mapping["created_at"].strftime("%-I:%M %p"),
                "content": row._mapping["content"],
            }
            for row in message_rows
        ],
    }


def list_customers() -> list[dict]:
    initialize_database()
    with _get_engine().connect() as conn:
        rows = conn.execute(
            select(customers)
            .where(customers.c.id != VISITOR_CUSTOMER_ID)
            .order_by(customers.c.name)
        ).all()
        return [_customer_payload(conn, row) for row in rows]


def list_products() -> list[dict]:
    initialize_database()
    with _get_engine().connect() as conn:
        rows = conn.execute(select(products).order_by(products.c.name)).all()
        return [dict(row._mapping) for row in rows]


def get_demo_overview() -> dict:
    """Return a small, database-derived snapshot for the portfolio landing page."""
    initialize_database()
    engine = _get_engine()
    with engine.connect() as conn:
        sample = conn.execute(
            select(
                customers.c.name,
                customers.c.email,
                customers.c.id.label("customer_id"),
                orders.c.id.label("order_id"),
                orders.c.parcel_id,
                orders.c.status.label("order_status"),
                products.c.name.label("product_name"),
            )
            .join(orders, orders.c.customer_id == customers.c.id)
            .join(products, products.c.id == orders.c.product_id)
            .where(orders.c.status == "Delayed")
            .order_by(orders.c.ordered_at.desc())
            .limit(1)
        ).first()
        if sample is None:
            sample = conn.execute(
                select(
                    customers.c.name,
                    customers.c.email,
                    customers.c.id.label("customer_id"),
                    orders.c.id.label("order_id"),
                    orders.c.parcel_id,
                    orders.c.status.label("order_status"),
                    products.c.name.label("product_name"),
                )
                .join(orders, orders.c.customer_id == customers.c.id)
                .join(products, products.c.id == orders.c.product_id)
                .order_by(orders.c.ordered_at.desc())
                .limit(1)
            ).one()

        waiting = conn.execute(
            select(func.count())
            .select_from(conversations)
            .where(conversations.c.status == "Waiting for support")
        ).scalar_one()
        resolved = conn.execute(
            select(func.count())
            .select_from(conversations)
            .where(conversations.c.status == "Resolved by Orion")
        ).scalar_one()
        row = sample._mapping
        database_name = {
            "sqlite": "SQLite",
            "postgresql": "PostgreSQL",
        }.get(engine.dialect.name, engine.dialect.name.title())

        return {
            "database": database_name,
            "counts": {
                "customers": conn.execute(
                    select(func.count())
                    .select_from(customers)
                    .where(customers.c.id != VISITOR_CUSTOMER_ID)
                ).scalar_one(),
                "products": conn.execute(
                    select(func.count()).select_from(products)
                ).scalar_one(),
                "orders": conn.execute(
                    select(func.count()).select_from(orders)
                ).scalar_one(),
                "conversations": waiting + resolved,
                "waiting": waiting,
                "resolved": resolved,
            },
            "sample": {
                "name": row["name"],
                "email": row["email"],
                "customerId": row["customer_id"],
                "orderId": row["order_id"],
                "parcelId": row["parcel_id"],
                "orderStatus": row["order_status"],
                "productName": row["product_name"],
            },
            "examples": [
                {
                    "label": "Order status",
                    "outcome": "Resolved by Orion",
                    "message": f"Where is order {row['order_id']}?",
                },
                {
                    "label": "Refund approval",
                    "outcome": "Waiting for support",
                    "message": f"My email is {row['email']}. I need a refund.",
                },
                {
                    "label": "Damaged parcel",
                    "outcome": "Waiting for support",
                    "message": f"Parcel {row['parcel_id']} arrived damaged.",
                },
            ],
        }


def lookup_customer(identifier: str) -> dict | None:
    initialize_database()
    normalized = identifier.strip().lower()
    with _get_engine().connect() as conn:
        row = conn.execute(
            select(customers)
            .outerjoin(orders, orders.c.customer_id == customers.c.id)
            .where(
                or_(
                    func.lower(customers.c.id) == normalized,
                    func.lower(customers.c.email) == normalized,
                    func.lower(orders.c.id) == normalized,
                    func.lower(orders.c.parcel_id) == normalized,
                )
            )
            .limit(1)
        ).first()
        return _customer_payload(conn, row) if row else None


def list_conversations() -> list[dict]:
    initialize_database()
    with _get_engine().connect() as conn:
        rows = conn.execute(
            select(conversations).order_by(conversations.c.updated_at.desc())
        ).all()
        return [_conversation_payload(conn, row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    initialize_database()
    with _get_engine().connect() as conn:
        row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).first()
        return _conversation_payload(conn, row) if row else None


def _is_policy_question(content: str) -> bool:
    normalized = content.lower().strip()
    if "policy" in normalized:
        return True
    topic = any(
        word in normalized
        for word in (
            "return",
            "warranty",
            "shipping",
            "delivery",
            "payment method",
            "pay with",
        )
    )
    question = "?" in normalized or normalized.startswith(
        ("what", "when", "where", "how", "can", "do", "does", "is", "are")
    )
    return topic and question


def _policy_response(content: str) -> str:
    normalized = content.lower()
    if "return" in normalized or "refund" in normalized or "send back" in normalized:
        answer = (
            "Most items can be returned within 30 calendar days of confirmed "
            "delivery if they are unused and include their original packaging and tags."
        )
    elif "warranty" in normalized or "broken" in normalized or "defect" in normalized:
        answer = (
            "Warranty coverage depends on the product category and covers manufacturing "
            "defects rather than accidental damage. Electronics have 12 months of "
            "ShopNova platform coverage."
        )
    elif "shipping" in normalized or "delivery" in normalized or "parcel" in normalized:
        answer = (
            "Standard shipping normally takes 5–8 business days after dispatch. "
            "Express shipping takes 1–2 business days when the order and destination qualify."
        )
    elif "payment" in normalized or "pay" in normalized:
        answer = (
            "ShopNova accepts major credit and debit cards, Boleto Bancário, and "
            "ShopNova vouchers. Payment timing and instalment rules vary by method."
        )
    else:
        answer = "I found the most relevant section in ShopNova’s policy library."
    return answer


def _build_orion_turn(customer: dict, content: str) -> dict:
    normalized = content.lower()
    first_name = customer["name"].split(" ")[0]
    if _is_policy_question(content):
        documents = search_policy_documents(content)
        database_details = (
            _database_technical_details(customer)
            if customer["id"] != VISITOR_CUSTOMER_ID
            else {"tools": [], "records": []}
        )
        return {
            "status": "Resolved by Orion",
            "subject": "Policy question",
            "action_needed": None,
            "response": _policy_response(content),
            "handoff": None,
            "technical_details": {
                "tools": [
                    *database_details["tools"],
                    {
                        "name": "policy_vector_search",
                        "label": "Policy vector search",
                        "result": f"{len(documents)} chunks",
                    }
                ],
                "records": database_details["records"],
                "documents": [
                    {
                        "source": document["source"],
                        "heading": document["heading"],
                        "score": document["score"],
                    }
                    for document in documents
                ],
            },
        }

    if not customer["orders"]:
        raise LookupError(
            "Please share your email, customer ID, order number, or parcel number "
            "so I can find the right order."
        )

    order = customer["orders"][0]
    technical_details = _database_technical_details(customer)
    if "cancel" in normalized or "refund" in normalized:
        return {
            "status": "Waiting for support",
            "subject": "Refund or cancellation request",
            "action_needed": "Approve the refund or cancellation request",
            "response": f"I found order {order['id']}. I can see the request, but a support teammate needs to approve a refund or cancellation. I’ve sent them the conversation and your order details.",
            "handoff": f"{first_name} requested a refund or cancellation for {order['id']}. Orion matched the customer and order, but approval is required from support.",
            "technical_details": technical_details,
        }
    if any(word in normalized for word in ("wrong", "damag", "broken", "warranty", "missing item", "replacement")):
        return {
            "status": "Waiting for support",
            "subject": "Item needs review",
            "action_needed": "Review the item issue and approve a replacement",
            "response": f"I matched parcel {order['parcelId']}. A support teammate needs to review the item issue before a replacement can be approved, so I’ve passed this conversation to them.",
            "handoff": f"{first_name} reported an item problem for parcel {order['parcelId']}. Support needs to review it and approve the appropriate replacement.",
            "technical_details": technical_details,
        }
    if any(word in normalized for word in ("human", "person", "agent", "support team", "complaint")):
        return {
            "status": "Waiting for support",
            "subject": "Customer requested support",
            "action_needed": "Read the conversation and reply to the customer",
            "response": "Of course. I’ve sent this conversation and your matched customer record to our support team. A teammate can continue here.",
            "handoff": f"{first_name} asked to speak with a person. The customer and latest order were matched successfully.",
            "technical_details": technical_details,
        }
    if any(word in normalized for word in ("exchange", "size", "larger", "smaller")):
        return {
            "status": "Resolved by Orion",
            "subject": "Size exchange",
            "action_needed": None,
            "response": f"Your {order['item']} is within the 30-day exchange window. I’ve started the exchange and sent the return instructions to {customer['email']}.",
            "handoff": None,
            "technical_details": technical_details,
        }
    if any(word in normalized for word in ("charg", "payment", "billed", "authorization")):
        return {
            "status": "Resolved by Orion",
            "subject": "Payment status",
            "action_needed": None,
            "response": f"I checked order {order['id']}. There is one completed payment; any second pending entry is a temporary authorization and should disappear within 3–5 business days.",
            "handoff": None,
            "technical_details": technical_details,
        }
    return {
        "status": "Resolved by Orion",
        "subject": "Order status",
        "action_needed": None,
        "response": f"Your order is currently {order['status'].lower()}: {order['detail']}. Is there anything else I can help you with regarding your order?",
        "handoff": None,
        "technical_details": technical_details,
    }


def _match_customer_from_message(conn: Connection, content: str):
    normalized = content.lower()
    customer_rows = conn.execute(select(customers)).all()
    for row in customer_rows:
        customer = row._mapping
        if customer["email"].lower() in normalized or customer["id"].lower() in normalized:
            return row

    order_rows = conn.execute(select(orders)).all()
    for order_row in order_rows:
        order = order_row._mapping
        if order["id"].lower() in normalized or order["parcel_id"].lower() in normalized:
            return conn.execute(
                select(customers).where(customers.c.id == order["customer_id"])
            ).one()
    return None


def send_customer_message(
    content: str,
    conversation_id: str | None = None,
) -> dict:
    initialize_database()
    engine = _get_engine()
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        existing = None
        if conversation_id:
            existing = conn.execute(
                select(conversations).where(
                    conversations.c.id == conversation_id,
                    conversations.c.source == "demo",
                )
            ).first()
            if not existing:
                raise LookupError("Demo conversation not found")

        if existing:
            existing_customer_id = existing._mapping["customer_id"]
            if existing_customer_id == VISITOR_CUSTOMER_ID:
                matched_customer = _match_customer_from_message(conn, content)
                if matched_customer:
                    customer_row = matched_customer
                    conn.execute(
                        update(conversations)
                        .where(conversations.c.id == conversation_id)
                        .values(customer_id=matched_customer._mapping["id"])
                    )
                elif _is_policy_question(content):
                    customer_row = conn.execute(
                        select(customers).where(
                            customers.c.id == VISITOR_CUSTOMER_ID
                        )
                    ).one()
                else:
                    raise LookupError(
                        "Please share your email, customer ID, order number, or "
                        "parcel number so I can find the right order."
                    )
            else:
                customer_row = conn.execute(
                    select(customers).where(customers.c.id == existing_customer_id)
                ).one()
        else:
            customer_row = _match_customer_from_message(conn, content)
            if not customer_row:
                if _is_policy_question(content):
                    customer_row = conn.execute(
                        select(customers).where(
                            customers.c.id == VISITOR_CUSTOMER_ID
                        )
                    ).one()
                else:
                    raise LookupError(
                        "Include your email, customer ID, order ID, or parcel ID so Orion can find your account."
                    )

        customer = _customer_payload(conn, customer_row)
        customer_id = customer["id"]
        turn = _build_orion_turn(customer, content)

        if existing:
            target_id = conversation_id
            conn.execute(
                update(conversations)
                .where(conversations.c.id == target_id)
                .values(
                    subject=turn["subject"],
                    preview=content,
                    status=turn["status"],
                    action_needed=turn["action_needed"],
                    unread=conversations.c.unread + 1,
                    resolved_by=None,
                    updated_at=now,
                )
            )
        else:
            target_id = f"DEMO-{uuid4().hex[:8].upper()}"
            conn.execute(
                insert(conversations).values(
                    id=target_id,
                    customer_id=customer_id,
                    subject=turn["subject"],
                    preview=content,
                    status=turn["status"],
                    channel="Chat",
                    action_needed=turn["action_needed"],
                    source="demo",
                    unread=1,
                    created_at=now,
                    updated_at=now,
                )
            )

        conn.execute(
            delete(conversation_traces).where(
                conversation_traces.c.conversation_id == target_id
            )
        )
        conn.execute(
            insert(conversation_traces).values(
                conversation_id=target_id,
                details_json=json.dumps(turn["technical_details"]),
                updated_at=now,
            )
        )

        new_messages = [
            {
                "id": uuid4().hex,
                "conversation_id": target_id,
                "role": "customer",
                "sender": (
                    "Visitor"
                    if customer_id == VISITOR_CUSTOMER_ID
                    else customer["name"].split(" ")[0]
                ),
                "content": content,
                "created_at": now,
            },
            {
                "id": uuid4().hex,
                "conversation_id": target_id,
                "role": "orion",
                "sender": "Orion",
                "content": turn["response"],
                "created_at": now + timedelta(milliseconds=1),
            },
        ]
        if turn["handoff"]:
            new_messages.append(
                {
                    "id": uuid4().hex,
                    "conversation_id": target_id,
                    "role": "handoff",
                    "sender": "Orion",
                    "content": turn["handoff"],
                    "created_at": now + timedelta(milliseconds=2),
                }
            )
        conn.execute(insert(messages), new_messages)
        row = conn.execute(
            select(conversations).where(conversations.c.id == target_id)
        ).one()
        return _conversation_payload(conn, row)


def send_support_reply(conversation_id: str, content: str) -> dict:
    initialize_database()
    now = datetime.now(timezone.utc)
    with _get_engine().begin() as conn:
        row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).first()
        if not row:
            raise LookupError("Conversation not found")
        conn.execute(
            insert(messages).values(
                id=uuid4().hex,
                conversation_id=conversation_id,
                role="agent",
                sender="Alex Kim",
                content=content,
                created_at=now,
            )
        )
        conn.execute(
            update(conversations)
            .where(conversations.c.id == conversation_id)
            .values(unread=0, updated_at=now)
        )
        updated_row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).one()
        return _conversation_payload(conn, updated_row)


def resolve_conversation_by_support(conversation_id: str) -> dict:
    """Close an escalated conversation as resolved by the support teammate."""
    initialize_database()
    now = datetime.now(timezone.utc)
    with _get_engine().begin() as conn:
        row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).first()
        if not row:
            raise LookupError("Conversation not found")
        conn.execute(
            update(conversations)
            .where(conversations.c.id == conversation_id)
            .values(
                status="Resolved by Orion",
                resolved_by="support",
                action_needed=None,
                unread=0,
                updated_at=now,
            )
        )
        updated_row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).one()
        return _conversation_payload(conn, updated_row)


def mark_conversation_read(conversation_id: str) -> dict:
    """Clear a conversation's unread count and return its updated payload."""
    initialize_database()
    with _get_engine().begin() as conn:
        row = conn.execute(
            select(conversations).where(conversations.c.id == conversation_id)
        ).first()
        if not row:
            raise LookupError("Conversation not found")
        if row._mapping["unread"]:
            conn.execute(
                update(conversations)
                .where(conversations.c.id == conversation_id)
                .values(unread=0)
            )
            row = conn.execute(
                select(conversations).where(conversations.c.id == conversation_id)
            ).one()
        return _conversation_payload(conn, row)


def delete_conversation(conversation_id: str) -> None:
    """Delete one support conversation while preserving its customer record."""
    initialize_database()
    with _get_engine().begin() as conn:
        exists = conn.execute(
            select(conversations.c.id).where(conversations.c.id == conversation_id)
        ).first()
        if not exists:
            raise LookupError("Conversation not found")
        conn.execute(
            delete(conversation_traces).where(
                conversation_traces.c.conversation_id == conversation_id
            )
        )
        conn.execute(
            delete(messages).where(messages.c.conversation_id == conversation_id)
        )
        conn.execute(
            delete(conversations).where(conversations.c.id == conversation_id)
        )


def reset_demo_conversations() -> list[dict]:
    initialize_database()
    with _get_engine().begin() as conn:
        demo_ids = select(conversations.c.id).where(conversations.c.source == "demo")
        conn.execute(
            delete(conversation_traces).where(
                conversation_traces.c.conversation_id.in_(demo_ids)
            )
        )
        conn.execute(delete(messages).where(messages.c.conversation_id.in_(demo_ids)))
        conn.execute(delete(conversations).where(conversations.c.source == "demo"))
    return list_conversations()
