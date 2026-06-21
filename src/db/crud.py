import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session

from .models import (
    Tenant, TenantProfile,
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
    db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> list:
    now = datetime.now(timezone.utc)
    not_expired = or_(UserDocument.expires_at.is_(None), UserDocument.expires_at > now)
    return (
        db.query(UserDocument)
        .filter(
            or_(
                and_(UserDocument.scope == "personal", UserDocument.user_id == user_id),
                and_(UserDocument.scope == "tenant", UserDocument.tenant_id == tenant_id),
            ),
            not_expired,
        )
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
            ),
            or_(UserDocument.expires_at.is_(None), UserDocument.expires_at > now),
        )
        .first()
    )


def create_user_document(
    db: Session,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    original_filename: str,
    storage_path: str,
    file_size_bytes: Optional[int] = None,
    scope: str = "personal",
    expires_at=None,
) -> UserDocument:
    doc = UserDocument(
        user_id=user_id,
        tenant_id=tenant_id,
        original_filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=file_size_bytes,
        scope=scope,
        expires_at=expires_at,
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