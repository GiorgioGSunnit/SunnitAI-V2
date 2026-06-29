import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session

from .models import (
    Tenant, TenantProfile, TenantSubscription,
    User, UserProfile,
    UserSettings, UserPreferences,
    Conversation, UserDocument
)
from ..chatbot.auth import hash_password


# ---------------------------------------------------------------------------
# Tenant operations
# ---------------------------------------------------------------------------

def get_tenant_by_email(db: Session, email: str) -> Optional[Tenant]:
    return db.query(Tenant).filter(Tenant.email == email).first()


def create_tenant(db: Session, email: str, plain_password: str) -> Tenant:
    tenant = Tenant(
        email=email,
        hashed_password=hash_password(plain_password)
    )
    db.add(tenant)
    db.flush()
    db.add(TenantProfile(tenant_id=tenant.id))
    return tenant


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_users_by_tenant(db: Session, tenant_id: uuid.UUID) -> list:
    return db.query(User).filter(
        User.tenant_id == tenant_id,
        User.is_active == True
    ).all()


def create_user(
    db: Session,
    tenant_id: uuid.UUID,
    email: str,
    plain_password: str,
    role: str = "member"
) -> User:
    user = User(
        tenant_id=tenant_id,
        email=email,
        hashed_password=hash_password(plain_password),
        role=role
    )
    db.add(user)
    db.flush()

    # Create related records automatically
    db.add(UserProfile(user_id=user.id, tenant_id=tenant_id))
    db.add(UserSettings(user_id=user.id, tenant_id=tenant_id))
    db.add(UserPreferences(user_id=user.id, tenant_id=tenant_id, preferences={}))

    return user


def deactivate_user(db: Session, user_id: uuid.UUID) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if user:
        user.is_active = False
        db.commit()
    return user


# ---------------------------------------------------------------------------
# Full registration — tenant + admin user in one transaction
# ---------------------------------------------------------------------------

def register_tenant_with_admin(
    db: Session,
    email: str,
    plain_password: str,
    company_name: str = ""
) -> tuple[Tenant, User]:
    """
    Creates a tenant and their admin user atomically.
    If anything fails, the whole transaction rolls back.
    """
    # Check if email already exists
    if get_tenant_by_email(db, email):
        raise ValueError("Email already registered")

    # Create tenant
    tenant = create_tenant(db, email, plain_password)

    # Set company name in profile if provided
    if company_name:
        tenant.profile.display_name = company_name
        tenant.profile.legal_name = company_name

    # Create admin user
    admin = create_user(db, tenant.id, email, plain_password, role="admin")

    db.commit()
    db.refresh(tenant)
    db.refresh(admin)

    return tenant, admin


# ---------------------------------------------------------------------------
# Settings operations
# ---------------------------------------------------------------------------

def get_user_settings(db: Session, user_id: uuid.UUID) -> Optional[UserSettings]:
    return db.query(UserSettings).filter(
        UserSettings.user_id == user_id
    ).first()


def update_user_settings(
    db: Session,
    user_id: uuid.UUID,
    tone: int,
    standing: int,
    response_length: int
) -> UserSettings:
    settings = get_user_settings(db, user_id)
    if not settings:
        user = db.query(User).filter(User.id == user_id).first()
        settings = UserSettings(
            user_id=user_id,
            tenant_id=user.tenant_id,
            tone=tone,
            standing=standing,
            response_length=response_length,
        )
        db.add(settings)
    else:
        settings.tone = tone
        settings.standing = standing
        settings.response_length = response_length
    db.commit()
    db.refresh(settings)
    return settings


def get_user_preferences(db: Session, user_id: uuid.UUID) -> Optional[UserPreferences]:
    return db.query(UserPreferences).filter(
        UserPreferences.user_id == user_id
    ).first()


def update_user_preferences(
    db: Session,
    user_id: uuid.UUID,
    preferences: dict
) -> UserPreferences:
    prefs = get_user_preferences(db, user_id)
    if not prefs:
        user = db.query(User).filter(User.id == user_id).first()
        prefs = UserPreferences(
            user_id=user_id,
            tenant_id=user.tenant_id,
            preferences=preferences,
        )
        db.add(prefs)
    else:
        prefs.preferences = preferences
    db.commit()
    db.refresh(prefs)
    return prefs


# ---------------------------------------------------------------------------
# Conversation operations
# ---------------------------------------------------------------------------

def save_conversation(
    db: Session,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    title: str,
    messages: list,
    session_language: str = "it"
) -> Conversation:
    conv = Conversation(
        user_id=user_id,
        tenant_id=tenant_id,
        title=title,
        messages=messages,
        session_language=session_language
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_user_conversations(db: Session, user_id: uuid.UUID) -> list:
    return db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).all()


def get_conversation_by_id(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> Optional[Conversation]:
    return db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()


def update_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    messages: list,
    title: str = None
) -> Optional[Conversation]:
    conv = get_conversation_by_id(db, conversation_id, user_id)
    if conv:
        conv.messages = messages
        if title:
            conv.title = title
        db.commit()
        db.refresh(conv)
    return conv


def delete_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> bool:
    conv = get_conversation_by_id(db, conversation_id, user_id)
    if conv:
        db.delete(conv)
        db.commit()
        return True
    return False


# ---------------------------------------------------------------------------
# Superadmin operations
# ---------------------------------------------------------------------------

def create_superadmin_user(
    db: Session, email: str, plain_password: str
) -> User:
    from ..constants import PLATFORM_TENANT_ID
    return create_user(db, PLATFORM_TENANT_ID, email, plain_password, role="superadmin")


# ---------------------------------------------------------------------------
# User document operations
# ---------------------------------------------------------------------------

def get_user_documents(
    db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID,
    include_platform: bool = False,
) -> list:
    """
    Return documents visible to this user.

    include_platform=False (default): returns only personal + tenant scope.
        Used by /api/user/documents (FE listing and picker) — platform skeleton
        templates are system-internal and should not appear in the user's document list.
    include_platform=True: includes platform scope.
        Used only by get_user_document_for_generation which needs to resolve
        platform skeleton doc_ids for template filling.
    """
    now = datetime.now(timezone.utc)
    not_expired = or_(UserDocument.expires_at.is_(None), UserDocument.expires_at > now)

    scope_filter = or_(
        and_(UserDocument.scope == "personal", UserDocument.user_id == user_id),
        and_(UserDocument.scope == "tenant", UserDocument.tenant_id == tenant_id),
    )
    if include_platform:
        scope_filter = or_(scope_filter, UserDocument.scope == "platform")

    return (
        db.query(UserDocument)
        .filter(scope_filter, not_expired)
        .order_by(UserDocument.scope, UserDocument.uploaded_at.desc())
        .all()
    )


def get_user_document(
    db: Session, doc_id: uuid.UUID, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> Optional[UserDocument]:
    now = datetime.now(timezone.utc)
    return (
        db.query(UserDocument)
        .filter(
            UserDocument.id == doc_id,
            or_(
                and_(UserDocument.scope == "personal", UserDocument.user_id == user_id),
                and_(UserDocument.scope == "tenant", UserDocument.tenant_id == tenant_id),
                UserDocument.scope == "platform",
            ),
            or_(UserDocument.expires_at.is_(None), UserDocument.expires_at > now),
        )
        .first()
    )


def find_user_document_by_name(
    db: Session,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    query: str,
) -> Optional["UserDocument"]:
    """
    Fuzzy-match a user's uploaded documents by filename.
    Returns the best-matching non-expired document the user can access,
    or None if nothing scores above the threshold.

    Matching strategy (in priority order):
      1. Exact filename match (case-insensitive)
      2. Query is a substring of filename or vice-versa
      3. Jaccard token overlap >= 0.25 (strips extension and separators)
    """
    import re as _re

    def _tokens(s: str) -> set:
        s = _re.sub(r"[^a-z0-9]", " ", s.lower())
        return {t for t in s.split() if len(t) > 1}

    now = datetime.now(timezone.utc)
    candidates = (
        db.query(UserDocument)
        .filter(
            or_(
                and_(UserDocument.scope == "personal", UserDocument.user_id == user_id),
                and_(UserDocument.scope == "tenant", UserDocument.tenant_id == tenant_id),
                UserDocument.scope == "platform",
            ),
            or_(UserDocument.expires_at.is_(None), UserDocument.expires_at > now),
        )
        .all()
    )

    q_clean = _re.sub(r"\.[a-z]{2,4}$", "", query.lower()).strip()
    q_tokens = _tokens(q_clean)

    best_doc = None
    best_score = -1.0

    for doc in candidates:
        fname = _re.sub(r"\.[a-z]{2,4}$", "", doc.original_filename.lower()).strip()

        # Exact match
        if fname == q_clean or doc.original_filename.lower() == query.lower():
            return doc

        # Substring match
        if q_clean in fname or fname in q_clean:
            score = 0.8
        else:
            f_tokens = _tokens(fname)
            union = q_tokens | f_tokens
            score = len(q_tokens & f_tokens) / len(union) if union else 0.0

        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc if best_score >= 0.25 else None


def find_expired_user_document_by_name(
    db: Session,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    query: str,
) -> Optional["UserDocument"]:
    """
    Same fuzzy matching as find_user_document_by_name but includes expired documents.
    Used to detect when a user references a document that has expired, so we can
    return a specific expiry message instead of falling through to RAG.
    """
    import re as _re

    def _tokens(s: str) -> set:
        s = _re.sub(r"[^a-z0-9]", " ", s.lower())
        return {t for t in s.split() if len(t) > 1}

    candidates = (
        db.query(UserDocument)
        .filter(
            or_(
                and_(UserDocument.scope == "personal", UserDocument.user_id == user_id),
                and_(UserDocument.scope == "tenant", UserDocument.tenant_id == tenant_id),
            ),
            UserDocument.expires_at.isnot(None),
            UserDocument.expires_at <= datetime.now(timezone.utc),
        )
        .all()
    )

    q_clean = _re.sub(r"\.[a-z]{2,4}$", "", query.lower()).strip()
    q_tokens = _tokens(q_clean)

    best_doc = None
    best_score = -1.0

    for doc in candidates:
        fname = _re.sub(r"\.[a-z]{2,4}$", "", doc.original_filename.lower()).strip()
        if fname == q_clean or doc.original_filename.lower() == query.lower():
            return doc
        if q_clean in fname or fname in q_clean:
            score = 0.8
        else:
            f_tokens = _tokens(fname)
            union = q_tokens | f_tokens
            score = len(q_tokens & f_tokens) / len(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc if best_score >= 0.25 else None


def create_user_document(
    db: Session,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    original_filename: str,
    storage_path: str,
    file_size_bytes: Optional[int] = None,
    scope: str = "personal",
    expires_at=None,
    document_role: str = "document",
) -> UserDocument:
    doc = UserDocument(
        user_id=user_id,
        tenant_id=tenant_id,
        original_filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=file_size_bytes,
        scope=scope,
        expires_at=expires_at,
        document_role=document_role,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def delete_user_document(
    db: Session, doc_id: uuid.UUID, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> bool:
    doc = get_user_document(db, doc_id, user_id, tenant_id)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True


DEFAULT_EXPIRY_HOURS = 24


def get_tenant_expiry_hours(db: Session, tenant_id: uuid.UUID) -> int:
    profile = db.query(TenantProfile).filter(TenantProfile.tenant_id == tenant_id).first()
    if profile and profile.document_expiry_hours is not None:
        return profile.document_expiry_hours
    return DEFAULT_EXPIRY_HOURS


def set_tenant_expiry_hours(db: Session, tenant_id: uuid.UUID, hours: int) -> Optional[TenantProfile]:
    profile = db.query(TenantProfile).filter(TenantProfile.tenant_id == tenant_id).first()
    if not profile:
        return None
    profile.document_expiry_hours = hours
    db.commit()
    db.refresh(profile)
    return profile


def cleanup_expired_documents(db: Session) -> int:
    now = datetime.now(timezone.utc)
    expired = (
        db.query(UserDocument)
        .filter(UserDocument.expires_at.isnot(None), UserDocument.expires_at <= now)
        .all()
    )
    count = 0
    for doc in expired:
        import os
        if os.path.exists(doc.storage_path):
            os.remove(doc.storage_path)
        db.delete(doc)
        count += 1
    if count:
        db.commit()
    return count


# ---------------------------------------------------------------------------
# Billing / subscriptions
# ---------------------------------------------------------------------------

ACTIVE_TENANT_SUBSCRIPTION_STATUSES = {"active", "trialing"}


def count_active_users_for_tenant(db: Session, tenant_id: uuid.UUID) -> int:
    return (
        db.query(User)
        .filter(User.tenant_id == tenant_id, User.is_active == True)
        .count()
    )


def get_tenant_subscription(db: Session, tenant_id: uuid.UUID) -> Optional[TenantSubscription]:
    return (
        db.query(TenantSubscription)
        .filter(TenantSubscription.tenant_id == tenant_id)
        .first()
    )


def get_tenant_subscription_by_customer_id(
    db: Session, stripe_customer_id: str
) -> Optional[TenantSubscription]:
    return (
        db.query(TenantSubscription)
        .filter(TenantSubscription.stripe_customer_id == stripe_customer_id)
        .first()
    )


def get_tenant_subscription_by_stripe_subscription_id(
    db: Session, stripe_subscription_id: str
) -> Optional[TenantSubscription]:
    return (
        db.query(TenantSubscription)
        .filter(TenantSubscription.stripe_subscription_id == stripe_subscription_id)
        .first()
    )


def get_tenant_subscription_by_checkout_session_id(
    db: Session, stripe_checkout_session_id: str
) -> Optional[TenantSubscription]:
    return (
        db.query(TenantSubscription)
        .filter(TenantSubscription.stripe_checkout_session_id == stripe_checkout_session_id)
        .first()
    )


def upsert_tenant_subscription(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    user_id: Optional[uuid.UUID] = None,
    plan_id: Optional[str] = None,
    status: Optional[str] = None,
    seats: Optional[int] = None,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    stripe_checkout_session_id: Optional[str] = None,
    trial_started_at=None,
    trial_ends_at=None,
    current_period_end=None,
    cancel_at_period_end: Optional[bool] = None,
    last_payment_status: Optional[str] = None,
) -> TenantSubscription:
    subscription = get_tenant_subscription(db, tenant_id)
    if not subscription:
        subscription = TenantSubscription(tenant_id=tenant_id)
        db.add(subscription)
        db.flush()

    if user_id is not None:
        subscription.user_id = user_id
    if plan_id is not None:
        subscription.plan_id = plan_id
    if status is not None:
        subscription.status = status
    if seats is not None:
        subscription.seats = max(1, int(seats))
    if stripe_customer_id is not None:
        subscription.stripe_customer_id = stripe_customer_id
    if stripe_subscription_id is not None:
        subscription.stripe_subscription_id = stripe_subscription_id
    if stripe_checkout_session_id is not None:
        subscription.stripe_checkout_session_id = stripe_checkout_session_id
    if trial_started_at is not None:
        subscription.trial_started_at = trial_started_at
    if trial_ends_at is not None:
        subscription.trial_ends_at = trial_ends_at
    if current_period_end is not None:
        subscription.current_period_end = current_period_end
    if cancel_at_period_end is not None:
        subscription.cancel_at_period_end = cancel_at_period_end
    if last_payment_status is not None:
        subscription.last_payment_status = last_payment_status

    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant:
        now = datetime.now(timezone.utc)
        if subscription.status in ACTIVE_TENANT_SUBSCRIPTION_STATUSES:
            if subscription.plan_id:
                tenant.plan = subscription.plan_id
            if tenant.subscription_start is None:
                tenant.subscription_start = now
            tenant.subscription_end = subscription.current_period_end
        else:
            tenant.subscription_end = subscription.current_period_end
            if subscription.status in {"canceled", "cancelled", "incomplete_expired", "inactive"}:
                tenant.plan = "basic"

    db.commit()
    db.refresh(subscription)
    return subscription