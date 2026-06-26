import os
import psycopg2
import psycopg2.pool
from typing import Optional

_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("PG_HOST", "127.0.0.1"),
            port=int(os.getenv("PG_PORT", 5432)),
            dbname=os.getenv("PG_DATABASE", "astrea"),
            user=os.getenv("PG_USER", "astrea_admin"),
            password=os.getenv("PG_PASSWORD", ""),
        )
    return _pool

def _conn():
    return _get_pool().getconn()

def _release(conn):
    _get_pool().putconn(conn)

def get_user_by_email(email: str) -> Optional[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.id, u.tenant_id, u.email, u.hashed_password,
                       u.role, u.is_active,
                       up.first_name, up.last_name, up.display_name,
                       tp.legal_name as studio_name
                FROM users u
                LEFT JOIN user_profiles up ON up.user_id = u.id
                LEFT JOIN tenant_profiles tp ON tp.tenant_id = u.tenant_id
                WHERE u.email = %s
            """, (email,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        _release(conn)

def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.id, u.tenant_id, u.email,
                       u.role, u.is_active,
                       up.first_name, up.last_name, up.display_name,
                       tp.legal_name as studio_name
                FROM users u
                LEFT JOIN user_profiles up ON up.user_id = u.id
                LEFT JOIN tenant_profiles tp ON tp.tenant_id = u.tenant_id
                WHERE u.id = %s
            """, (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        _release(conn)

def get_user_settings(user_id: str) -> dict:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tone, standing, response_length
                FROM user_settings WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            if not row:
                return {"tone": 2, "standing": 2, "response_length": 2}
            return {"tone": row[0], "standing": row[1], "response_length": row[2]}
    finally:
        _release(conn)


def get_user_settings_by_email(email: str) -> dict:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.tone, s.standing, s.response_length
                FROM user_settings s
                JOIN users u ON u.id = s.user_id
                WHERE u.email = %s
            """, (email,))
            row = cur.fetchone()
            if not row:
                return {"tone": 2, "standing": 2, "response_length": 2}
            return {"tone": row[0], "standing": row[1], "response_length": row[2]}
    finally:
        _release(conn)


def get_tenant_by_id(tenant_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.id, t.email, t.is_active, t.plan,
                       tp.legal_name, tp.display_name
                FROM tenants t
                LEFT JOIN tenant_profiles tp ON tp.tenant_id = t.id
                WHERE t.id = %s
            """, (tenant_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        _release(conn)


_TENANT_PROFILE_COLUMNS = [
    "legal_name", "display_name", "vat_number", "phone", "website",
    "address_street", "address_city", "address_postal_code",
    "address_country", "logo_path", "document_expiry_hours",
]


def get_tenant_profile_full(tenant_id: str) -> Optional[dict]:
    """Fetch the full tenant_profiles row (letterhead-relevant fields)
    for a given tenant. Returns None if no profile row exists yet."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {', '.join(_TENANT_PROFILE_COLUMNS)}
                FROM tenant_profiles
                WHERE tenant_id = %s
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        _release(conn)


def get_user_document_for_generation(
    user_id: str, tenant_id: str, doc_id: str
) -> Optional[dict]:
    """Verify ownership and return storage_path + original_filename for
    a user document. Returns None if not found, expired, or access denied."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT storage_path, original_filename
                FROM user_documents
                WHERE id = %s
                  AND (
                    (scope = 'personal' AND user_id = %s)
                    OR (scope = 'tenant' AND tenant_id = %s)
                    OR scope = 'platform'
                  )
                  AND (expires_at IS NULL OR expires_at > NOW())
                """,
                (doc_id, user_id, tenant_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"storage_path": row[0], "original_filename": row[1]}
    finally:
        _release(conn)


def upsert_tenant_profile(tenant_id: str, fields: dict) -> dict:
    """Insert or update tenant_profiles fields for the given tenant.
    Only keys in _TENANT_PROFILE_COLUMNS are written; others are ignored."""
    update_fields = {k: v for k, v in fields.items() if k in _TENANT_PROFILE_COLUMNS}
    if not update_fields:
        return get_tenant_profile_full(tenant_id) or {}

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM tenant_profiles WHERE tenant_id = %s", (tenant_id,))
            existing = cur.fetchone()
            if existing:
                set_clause = ", ".join(f"{k} = %s" for k in update_fields)
                values = list(update_fields.values()) + [tenant_id]
                cur.execute(
                    f"UPDATE tenant_profiles SET {set_clause}, updated_at = now() WHERE tenant_id = %s",
                    values,
                )
            else:
                import uuid as _uuid
                cols = ["id", "tenant_id"] + list(update_fields.keys())
                placeholders = ", ".join(["%s"] * len(cols))
                values = [str(_uuid.uuid4()), tenant_id] + list(update_fields.values())
                cur.execute(
                    f"INSERT INTO tenant_profiles ({', '.join(cols)}) VALUES ({placeholders})",
                    values,
                )
            conn.commit()
        return get_tenant_profile_full(tenant_id) or {}
    finally:
        _release(conn)


def create_studio_and_admin(
    email: str,
    hashed_password: str,
    first_name: str,
    last_name: str,
    studio_name: str,
) -> dict:
    """Create a new tenant, tenant_profile, admin user and user_profile.
    Returns the created user dict including tenant_id and invite_code."""
    import uuid, secrets, string

    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    invite_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits)
                          for _ in range(10))

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tenants (id, email, hashed_password, plan, is_active, invite_code)
                VALUES (%s, %s, %s, 'basic', true, %s)
            """, (tenant_id, email, hashed_password, invite_code))

            cur.execute("""
                INSERT INTO tenant_profiles (id, tenant_id, legal_name, display_name)
                VALUES (%s, %s, %s, %s)
            """, (str(uuid.uuid4()), tenant_id, studio_name, studio_name))

            cur.execute("""
                INSERT INTO users
                (id, tenant_id, email, hashed_password, role, is_active, email_verified)
                VALUES (%s, %s, %s, %s, 'admin', true, true)
            """, (user_id, tenant_id, email, hashed_password))

            cur.execute("""
                INSERT INTO user_profiles
                (id, user_id, tenant_id, first_name, last_name, display_name)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(uuid.uuid4()), user_id, tenant_id,
                  first_name, last_name, f"{first_name} {last_name}"))

            cur.execute("""
                INSERT INTO user_settings (id, user_id, tenant_id, tone, standing, response_length)
                VALUES (%s, %s, %s, 2, 2, 2)
            """, (str(uuid.uuid4()), user_id, tenant_id))

            cur.execute("""
                INSERT INTO user_preferences (id, user_id, tenant_id, preferences)
                VALUES (%s, %s, %s, %s)
            """, (str(uuid.uuid4()), user_id, tenant_id, '{}'))

        conn.commit()
        return {
            "id": user_id,
            "tenant_id": tenant_id,
            "email": email,
            "role": "admin",
            "invite_code": invite_code,
            "studio_name": studio_name,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        _release(conn)


def create_user_with_invite(
    email: str,
    hashed_password: str,
    first_name: str,
    last_name: str,
    invite_code: str,
) -> dict:
    """Create a new user under an existing tenant using an invite code.
    Returns the created user dict, or raises ValueError if invite code invalid."""
    import uuid

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.id, tp.legal_name
                FROM tenants t
                LEFT JOIN tenant_profiles tp ON tp.tenant_id = t.id
                WHERE t.invite_code = %s AND t.is_active = true
            """, (invite_code,))
            row = cur.fetchone()
            if not row:
                raise ValueError("Invalid or expired invite code")

            tenant_id, studio_name = row
            cur.execute(
                "SELECT seats, status FROM tenant_subscriptions WHERE tenant_id = %s",
                (str(tenant_id),),
            )
            subscription_row = cur.fetchone()
            seat_capacity = 1
            if subscription_row and subscription_row[1] in ("active", "trialing"):
                seat_capacity = max(1, int(subscription_row[0] or 1))

            cur.execute(
                "SELECT COUNT(*) FROM users WHERE tenant_id = %s AND is_active = true",
                (str(tenant_id),),
            )
            seats_used = int(cur.fetchone()[0])
            if seats_used >= seat_capacity:
                raise ValueError("No available team seats for this tenant")

            user_id = str(uuid.uuid4())

            cur.execute("""
                INSERT INTO users
                (id, tenant_id, email, hashed_password, role, is_active, email_verified)
                VALUES (%s, %s, %s, %s, 'user', true, true)
            """, (user_id, str(tenant_id), email, hashed_password))

            cur.execute("""
                INSERT INTO user_profiles
                (id, user_id, tenant_id, first_name, last_name, display_name)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(uuid.uuid4()), user_id, str(tenant_id),
                  first_name, last_name, f"{first_name} {last_name}"))

            cur.execute("""
                INSERT INTO user_settings (id, user_id, tenant_id, tone, standing, response_length)
                VALUES (%s, %s, %s, 2, 2, 2)
            """, (str(uuid.uuid4()), user_id, str(tenant_id)))

            cur.execute("""
                INSERT INTO user_preferences (id, user_id, tenant_id, preferences)
                VALUES (%s, %s, %s, %s)
            """, (str(uuid.uuid4()), user_id, str(tenant_id), '{}'))

        conn.commit()
        return {
            "id": user_id,
            "tenant_id": str(tenant_id),
            "email": email,
            "role": "user",
            "studio_name": studio_name,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        _release(conn)


def get_tenant_invite_code(tenant_id: str) -> Optional[str]:
    """Get the invite code for a tenant."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT invite_code FROM tenants WHERE id = %s", (tenant_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        _release(conn)


def update_user_profile(
    user_id: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    display_name: Optional[str] = None,
    professional_title: Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[dict]:
    """Update user profile fields. Only updates fields that are provided."""
    fields = []
    values = []
    if first_name is not None:
        fields.append("first_name = %s")
        values.append(first_name)
    if last_name is not None:
        fields.append("last_name = %s")
        values.append(last_name)
    if display_name is not None:
        fields.append("display_name = %s")
        values.append(display_name)
    if professional_title is not None:
        fields.append("professional_title = %s")
        values.append(professional_title)
    if phone is not None:
        fields.append("phone = %s")
        values.append(phone)

    if not fields:
        return get_user_by_id(user_id)

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE user_profiles SET {', '.join(fields)} "
                f"WHERE user_id = %s",
                values + [user_id]
            )
        conn.commit()
        return get_user_by_id(user_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        _release(conn)
