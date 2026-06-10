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
