"""
Create the 4 platform superadmin accounts.
Usage: python -m scripts.seed_superadmins
Reads credentials from environment variables:
    SA1_EMAIL, SA1_PASSWORD
    SA2_EMAIL, SA2_PASSWORD
    SA3_EMAIL, SA3_PASSWORD
    SA4_EMAIL, SA4_PASSWORD
Run after: alembic upgrade head
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.base import SessionLocal
from src.db.crud import create_superadmin_user


def main():
    db = SessionLocal()
    try:
        for i in range(1, 5):
            email = os.environ.get(f"SA{i}_EMAIL")
            password = os.environ.get(f"SA{i}_PASSWORD")
            if not email or not password:
                print(f"Skipping SA{i}: SA{i}_EMAIL or SA{i}_PASSWORD not set")
                continue
            user = create_superadmin_user(db, email, password)
            print(f"Created superadmin: {user.email} ({user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
