"""Operational commands.

``python -m policy_comparator.cli <command>``

Uses argparse rather than a CLI framework so the sub-project keeps a small
dependency surface and runs from a bare checkout.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from sqlalchemy import select

from .config import get_settings
from .crypto import generate_key
from .db import create_all, session_scope
from .models import StaffUser
from .security import hash_password

DEMO_TENANT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


def cmd_genkey(_: argparse.Namespace) -> int:
    print(generate_key())
    print(
        "\nAdd this to your environment as PC_ENCRYPTION_KEY. "
        "Changing it later makes existing encrypted data unreadable.",
        file=sys.stderr,
    )
    return 0


def cmd_init_db(_: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.is_production:
        print(
            "Refusing to run create_all in production — use the Alembic "
            "migrations instead (`alembic -c policy_comparator/alembic.ini upgrade head`).",
            file=sys.stderr,
        )
        return 1
    create_all()
    print(f"Schema created at {settings.database_url}")
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    tenant_id = uuid.UUID(args.tenant_id) if args.tenant_id else DEMO_TENANT_ID
    email = args.email.strip().lower()

    with session_scope() as db:
        existing = db.execute(
            select(StaffUser).where(StaffUser.email == email)
        ).scalar_one_or_none()
        if existing is not None:
            print(f"User {email} already exists", file=sys.stderr)
            return 1
        user = StaffUser(
            tenant_id=tenant_id,
            email=email,
            hashed_password=hash_password(args.password),
            full_name=args.name,
            role=args.role,
        )
        db.add(user)
        db.commit()
        print(f"Created {email} (role={args.role}, tenant={tenant_id})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """One command to get a runnable demo: schema plus a staff login."""
    create_all()
    email = args.email.strip().lower()
    with session_scope() as db:
        existing = db.execute(
            select(StaffUser).where(StaffUser.email == email)
        ).scalar_one_or_none()
        if existing is None:
            db.add(
                StaffUser(
                    tenant_id=DEMO_TENANT_ID,
                    email=email,
                    hashed_password=hash_password(args.password),
                    full_name="Demo Staff",
                    role="admin",
                )
            )
            db.commit()

    settings = get_settings()
    print("Demo environment ready.")
    print(f"  database : {settings.database_url}")
    print(f"  login    : {email} / {args.password}")
    print(f"  tenant   : {DEMO_TENANT_ID}")
    print("\nStart the API and the worker in two terminals:")
    print("  uvicorn policy_comparator.api.app:app --reload --port 8100")
    print("  python -m policy_comparator.worker")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy_comparator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("genkey", help="Generate a PC_ENCRYPTION_KEY").set_defaults(func=cmd_genkey)
    sub.add_parser("init-db", help="Create the schema (development only)").set_defaults(
        func=cmd_init_db
    )

    create = sub.add_parser("create-user", help="Create a staff login")
    create.add_argument("--email", required=True)
    create.add_argument("--password", required=True)
    create.add_argument("--tenant-id", default=None)
    create.add_argument("--name", default=None)
    create.add_argument("--role", default="staff", choices=["staff", "admin"])
    create.set_defaults(func=cmd_create_user)

    demo = sub.add_parser("demo", help="Create the schema and a demo admin login")
    demo.add_argument("--email", default="staff@example.com")
    demo.add_argument("--password", default="demo-password")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
