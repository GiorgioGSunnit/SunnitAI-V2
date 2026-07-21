"""Remove a redundant legacy Alembic head after the branches were reconnected."""

import os

from sqlalchemy import create_engine, inspect, text


def main() -> None:
    database_url = os.environ.get("MIGRATION_DATABASE_URL")
    if not database_url:
        raise RuntimeError("MIGRATION_DATABASE_URL is required")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        if not inspect(connection).has_table("alembic_version"):
            print("Alembic legacy heads reconciled: 0")
            return
        result = connection.execute(
            text(
                """
                DELETE FROM alembic_version
                WHERE version_num = '8eca0a3dd1e6'
                  AND EXISTS (
                      SELECT 1
                      FROM alembic_version
                      WHERE version_num IN ('0001', '0002', '0003', '0004', '0005')
                  )
                """
            )
        )
    print(f"Alembic legacy heads reconciled: {result.rowcount}")


if __name__ == "__main__":
    main()
