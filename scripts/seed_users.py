import os
import uuid
import psycopg2
import bcrypt

PG_CONN = {
    "host": os.getenv("PG_HOST", "127.0.0.1"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DATABASE", "astrea"),
    "user": os.getenv("PG_USER", "astrea_admin"),
    "password": os.getenv("PG_PASSWORD", "+Astrea01%IT"),
}

TENANT_ID = "00000000-0000-0000-0000-000000000001"
TENANT_EMAIL = "studio@studiorossi.it"
TENANT_PASSWORD = "changeme"

USERS = [
    {
        "id": "00000000-0000-0000-0000-000000000010",
        "email": "admin@studiorossi.it",
        "password": "changeme",
        "role": "admin",
        "first_name": "Mario",
        "last_name": "Rossi",
    },
    {
        "id": "00000000-0000-0000-0000-000000000011",
        "email": "user1@studiorossi.it",
        "password": "changeme",
        "role": "user",
        "first_name": "Anna",
        "last_name": "Bianchi",
    },
]

STUDIO_NAME = "Studio Legale Mario Rossi"

def seed():
    conn = psycopg2.connect(**PG_CONN)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # Tenant
            cur.execute("""
                INSERT INTO tenants (id, email, hashed_password, plan, is_active)
                VALUES (%s, %s, %s, 'pro', true)
                ON CONFLICT (id) DO NOTHING
            """, (TENANT_ID, TENANT_EMAIL,
                  bcrypt.hashpw(TENANT_PASSWORD.encode(), bcrypt.gensalt()).decode()))

            # Tenant profile
            cur.execute("""
                INSERT INTO tenant_profiles (id, tenant_id, legal_name, display_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id) DO NOTHING
            """, (str(uuid.uuid4()), TENANT_ID, STUDIO_NAME, STUDIO_NAME))

            # Users
            for u in USERS:
                cur.execute("""
                    INSERT INTO users
                    (id, tenant_id, email, hashed_password, role, is_active, email_verified)
                    VALUES (%s, %s, %s, %s, %s, true, true)
                    ON CONFLICT (id) DO NOTHING
                """, (u["id"], TENANT_ID, u["email"],
                      bcrypt.hashpw(u["password"].encode(), bcrypt.gensalt()).decode(), u["role"]))

                # User profile
                cur.execute("""
                    INSERT INTO user_profiles
                    (id, user_id, tenant_id, first_name, last_name, display_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (str(uuid.uuid4()), u["id"], TENANT_ID,
                      u["first_name"], u["last_name"],
                      f"{u['first_name']} {u['last_name']}"))

        conn.commit()
        print("Seed complete.")
    except Exception as e:
        conn.rollback()
        print(f"Seed failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    seed()
