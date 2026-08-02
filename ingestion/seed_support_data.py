"""Create and seed the persistent support demo database."""

from api.support_store import initialize_database, list_conversations, list_customers


def main() -> None:
    initialize_database()
    print(
        f"Support database ready: {len(list_customers())} customers, "
        f"{len(list_conversations())} conversations."
    )


if __name__ == "__main__":
    main()
