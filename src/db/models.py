"""Database models for Astrea SaaS platform."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, Integer,
    DateTime, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .base import Base


def utcnow():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tenants — the company / law firm
# ---------------------------------------------------------------------------

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    plan = Column(String(50), default="basic")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    subscription_start = Column(DateTime(timezone=True), nullable=True)
    subscription_end = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    users = relationship("User", back_populates="tenant",
                         cascade="all, delete-orphan")
    profile = relationship("TenantProfile", back_populates="tenant",
                           uselist=False, cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Tenant Profile — company identity and branding
# ---------------------------------------------------------------------------

class TenantProfile(Base):
    __tablename__ = "tenant_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False, unique=True)
    display_name = Column(String(255), nullable=True)
    legal_name = Column(String(255), nullable=True)
    vat_number = Column(String(50), nullable=True)
    registration_number = Column(String(100), nullable=True)
    company_type = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    website = Column(String(255), nullable=True)
    address_street = Column(String(255), nullable=True)
    address_city = Column(String(100), nullable=True)
    address_postal_code = Column(String(20), nullable=True)
    address_country = Column(String(10), default="IT")
    logo_path = Column(String(500), nullable=True)
    primary_color = Column(String(20), nullable=True)
    billing_email = Column(String(255), nullable=True)
    billing_address = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="profile")


# ---------------------------------------------------------------------------
# Users — individual people under a tenant
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="member")     # admin / member
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255), nullable=True)
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires = Column(DateTime(timezone=True), nullable=True)
    totp_secret = Column(String(255), nullable=True)
    totp_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    profile = relationship("UserProfile", back_populates="user",
                           uselist=False, cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user",
                            uselist=False, cascade="all, delete-orphan")
    preferences = relationship("UserPreferences", back_populates="user",
                               uselist=False, cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user",
                                 cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# User Profile — personal identity
# ---------------------------------------------------------------------------

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                     nullable=False, unique=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    display_name = Column(String(255), nullable=True)
    professional_title = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    profile_image_path = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    user = relationship("User", back_populates="profile")


# ---------------------------------------------------------------------------
# User Settings — AI behavior (injected into every prompt)
# ---------------------------------------------------------------------------

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                     nullable=False, unique=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False)
    tone = Column(Integer, default=2)               # 1-4
    standing = Column(Integer, default=2)           # 1-4
    response_length = Column(Integer, default=2)    # 1-4
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    user = relationship("User", back_populates="settings")


# ---------------------------------------------------------------------------
# User Preferences — app experience (theme, language, etc.)
# ---------------------------------------------------------------------------

class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                     nullable=False, unique=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False)
    preferences = Column(JSONB, default=dict)
    # stores: theme, language, timezone, date_format,
    # sidebar_collapsed, email_notifications
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    user = relationship("User", back_populates="preferences")


# ---------------------------------------------------------------------------
# Conversations — chat history per user
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"),
                     nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"),
                       nullable=False)
    title = Column(String(255), nullable=True, default="Nuova conversazione")
    messages = Column(JSONB, default=list)
    session_language = Column(String(10), default="it")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")