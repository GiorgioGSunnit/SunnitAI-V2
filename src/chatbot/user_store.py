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
