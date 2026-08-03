"""Initial policy comparator schema.

Creates every ``pc_*`` table. This sub-project owns its own tables and its own
Alembic history; it does not touch the parent platform's schema.

Revision ID: 0001_initial
Revises:
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# The models use portable custom types (GUID, JSONColumn, Money,
# EncryptedString) that map to native PostgreSQL types and to portable ones on
# SQLite, so the module has to be importable from the migration.
import policy_comparator.db


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('pc_audit_events',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('actor_user_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('actor_email', sa.String(length=255), nullable=True),
    sa.Column('action', sa.String(length=64), nullable=False),
    sa.Column('entity_type', sa.String(length=48), nullable=True),
    sa.Column('entity_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('provider_id', sa.String(length=48), nullable=True),
    sa.Column('metadata_json', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_audit_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_audit_events_action'), ['action'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_audit_events_actor_user_id'), ['actor_user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_audit_events_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_audit_events_entity_id'), ['entity_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_audit_events_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_coverage_preferences',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('base_rc', sa.Boolean(), nullable=False),
    sa.Column('min_liability_limit_people', sa.String(length=32), nullable=True),
    sa.Column('min_liability_limit_property', sa.String(length=32), nullable=True),
    sa.Column('driving_formula', sa.String(length=24), nullable=True),
    sa.Column('max_acceptable_deductible', sa.String(length=32), nullable=True),
    sa.Column('required_optional_covers', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('accepts_black_box', sa.Boolean(), nullable=True),
    sa.Column('accepts_approved_repair_network', sa.Boolean(), nullable=True),
    sa.Column('payment_frequency', sa.String(length=24), nullable=True),
    sa.Column('field_sources', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_coverage_preferences', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_coverage_preferences_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_customers',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('created_by_user_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('email', policy_comparator.db.EncryptedString(), nullable=False),
    sa.Column('email_fingerprint', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_customers', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_customers_email_fingerprint'), ['email_fingerprint'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_customers_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_insurance_histories',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('current_insurer', sa.String(length=120), nullable=True),
    sa.Column('existing_policy_expiry', sa.Date(), nullable=True),
    sa.Column('universal_merit_class', sa.Integer(), nullable=True),
    sa.Column('first_insurance', sa.Boolean(), nullable=True),
    sa.Column('claims_last_5_years', sa.Integer(), nullable=True),
    sa.Column('claims_full_responsibility', sa.Integer(), nullable=True),
    sa.Column('claims_partial_responsibility', sa.Integer(), nullable=True),
    sa.Column('bersani_applicable', sa.Boolean(), nullable=True),
    sa.Column('bersani_source_plate', sa.String(length=16), nullable=True),
    sa.Column('bersani_source_merit_class', sa.Integer(), nullable=True),
    sa.Column('risk_certificate_reference', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('field_sources', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_insurance_histories', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_insurance_histories_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_provider_health',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_id', sa.String(length=48), nullable=False),
    sa.Column('consecutive_failures', sa.Integer(), nullable=False),
    sa.Column('circuit_open_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_failure_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_error_category', sa.String(length=64), nullable=True),
    sa.Column('total_successes', sa.Integer(), nullable=False),
    sa.Column('total_failures', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'provider_id', name='uq_pc_health_tenant_provider')
    )
    with op.batch_alter_table('pc_provider_health', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_provider_health_provider_id'), ['provider_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_health_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_quote_jobs',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('quote_request_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_attempt_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_id', sa.String(length=48), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('dedupe_key', sa.String(length=160), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('max_attempts', sa.Integer(), nullable=False),
    sa.Column('run_after', sa.DateTime(timezone=True), nullable=False),
    sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('claimed_by', sa.String(length=80), nullable=True),
    sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('payload', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dedupe_key', name='uq_pc_job_dedupe_key')
    )
    with op.batch_alter_table('pc_quote_jobs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_lease_expires_at'), ['lease_expires_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_provider_attempt_id'), ['provider_attempt_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_quote_request_id'), ['quote_request_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_run_after'), ['run_after'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_jobs_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_staff_users',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=160), nullable=True),
    sa.Column('role', sa.String(length=24), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_staff_users', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_staff_users_email'), ['email'], unique=True)
        batch_op.create_index(batch_op.f('ix_pc_staff_users_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_vehicles',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('plate', sa.String(length=16), nullable=False),
    sa.Column('ownership_status', sa.String(length=32), nullable=True),
    sa.Column('first_registration_date', sa.Date(), nullable=True),
    sa.Column('purchase_date', sa.Date(), nullable=True),
    sa.Column('make', sa.String(length=64), nullable=True),
    sa.Column('model', sa.String(length=120), nullable=True),
    sa.Column('trim', sa.String(length=160), nullable=True),
    sa.Column('fuel_type', sa.String(length=32), nullable=True),
    sa.Column('power_kw', sa.Integer(), nullable=True),
    sa.Column('primary_use', sa.String(length=48), nullable=True),
    sa.Column('annual_kilometres', sa.Integer(), nullable=True),
    sa.Column('overnight_parking', sa.String(length=48), nullable=True),
    sa.Column('anti_theft_system', sa.String(length=48), nullable=True),
    sa.Column('towing_hook', sa.Boolean(), nullable=True),
    sa.Column('field_sources', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_vehicles', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_vehicles_plate'), ['plate'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_vehicles_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_consent_records',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('customer_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('quote_request_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('consent_type', sa.String(length=48), nullable=False),
    sa.Column('granted', sa.Boolean(), nullable=False),
    sa.Column('scope_provider_ids', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('granted_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('recorded_by_user_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('policy_version', sa.String(length=32), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['customer_id'], ['pc_customers.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_consent_records', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_consent_records_customer_id'), ['customer_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_consent_records_quote_request_id'), ['quote_request_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_consent_records_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_customer_profiles',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('customer_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('owner_date_of_birth', sa.Date(), nullable=True),
    sa.Column('first_name', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('last_name', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('tax_code', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('gender', sa.String(length=16), nullable=True),
    sa.Column('mobile_number', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('address_street', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('municipality', sa.String(length=120), nullable=True),
    sa.Column('province', sa.String(length=8), nullable=True),
    sa.Column('postcode', sa.String(length=16), nullable=True),
    sa.Column('subject_type', sa.String(length=16), nullable=False),
    sa.Column('company_name', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('vat_number', policy_comparator.db.EncryptedString(), nullable=True),
    sa.Column('policyholder_same_as_owner', sa.Boolean(), nullable=False),
    sa.Column('field_sources', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['customer_id'], ['pc_customers.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_customer_profiles', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_customer_profiles_customer_id'), ['customer_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_customer_profiles_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_quote_requests',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('created_by_user_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('customer_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('customer_profile_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('vehicle_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('insurance_history_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('coverage_preference_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('policy_start_date', sa.Date(), nullable=False),
    sa.Column('selected_provider_ids', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('recommended_quote_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('demonstration_data', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['customer_id'], ['pc_customers.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_quote_requests', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_quote_requests_customer_id'), ['customer_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_requests_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_requests_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_provider_attempts',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('quote_request_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_id', sa.String(length=48), nullable=False),
    sa.Column('provider_type', sa.String(length=24), nullable=False),
    sa.Column('provider_mode', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('outcome', sa.String(length=32), nullable=True),
    sa.Column('error_category', sa.String(length=64), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('attempt_count', sa.Integer(), nullable=False),
    sa.Column('idempotency_key', sa.String(length=80), nullable=False),
    sa.Column('resume_token', policy_comparator.db.JSONColumn(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('duration_ms', sa.Integer(), nullable=True),
    sa.Column('diagnostic_artifact_path', sa.String(length=512), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['quote_request_id'], ['pc_quote_requests.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('quote_request_id', 'provider_id', name='uq_pc_attempt_request_provider')
    )
    with op.batch_alter_table('pc_provider_attempts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_provider_attempts_provider_id'), ['provider_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_attempts_quote_request_id'), ['quote_request_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_attempts_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_attempts_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_normalized_quotes',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('quote_request_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_attempt_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_id', sa.String(length=48), nullable=False),
    sa.Column('insurer_name', sa.String(length=120), nullable=False),
    sa.Column('source_channel', sa.String(length=24), nullable=False),
    sa.Column('product_name', sa.String(length=160), nullable=True),
    sa.Column('provider_quote_reference', sa.String(length=120), nullable=True),
    sa.Column('annual_total_premium', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('instalment_count', sa.Integer(), nullable=True),
    sa.Column('instalment_amount', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('instalment_total_cost', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('liability_limit_people', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('liability_limit_property', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('driving_formula', sa.String(length=24), nullable=True),
    sa.Column('deductible', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('percentage_excess', sa.String(length=16), nullable=True),
    sa.Column('requires_black_box', sa.Boolean(), nullable=True),
    sa.Column('requires_approved_repair_network', sa.Boolean(), nullable=True),
    sa.Column('important_exclusions', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('quote_expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('purchase_url', sa.String(length=1024), nullable=True),
    sa.Column('product_document_url', sa.String(length=1024), nullable=True),
    sa.Column('precontractual_document_url', sa.String(length=1024), nullable=True),
    sa.Column('raw_provider_status', sa.String(length=64), nullable=True),
    sa.Column('is_demonstration', sa.Boolean(), nullable=False),
    sa.Column('duplicate_of_quote_id', policy_comparator.db.GUID(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_attempt_id'], ['pc_provider_attempts.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['quote_request_id'], ['pc_quote_requests.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_normalized_quotes', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_normalized_quotes_provider_attempt_id'), ['provider_attempt_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_normalized_quotes_quote_request_id'), ['quote_request_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_normalized_quotes_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_provider_missing_fields',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_attempt_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('field_path', sa.String(length=120), nullable=False),
    sa.Column('label', sa.String(length=200), nullable=False),
    sa.Column('input_type', sa.String(length=24), nullable=False),
    sa.Column('choices', policy_comparator.db.JSONColumn(), nullable=True),
    sa.Column('required', sa.Boolean(), nullable=False),
    sa.Column('help_text', sa.Text(), nullable=True),
    sa.Column('resolved', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_attempt_id'], ['pc_provider_attempts.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_provider_missing_fields', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_provider_missing_fields_provider_attempt_id'), ['provider_attempt_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_missing_fields_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_provider_raw_responses',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_attempt_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('provider_id', sa.String(length=48), nullable=False),
    sa.Column('attempt_number', sa.Integer(), nullable=False),
    sa.Column('raw_status', sa.String(length=64), nullable=True),
    sa.Column('payload', policy_comparator.db.JSONColumn(), nullable=False),
    sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_attempt_id'], ['pc_provider_attempts.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_provider_raw_responses', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_provider_raw_responses_provider_attempt_id'), ['provider_attempt_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_provider_raw_responses_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('pc_quote_coverages',
    sa.Column('id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('tenant_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('quote_id', policy_comparator.db.GUID(length=36), nullable=False),
    sa.Column('code', sa.String(length=64), nullable=False),
    sa.Column('label', sa.String(length=200), nullable=False),
    sa.Column('included', sa.Boolean(), nullable=False),
    sa.Column('price', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('limit_amount', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('deductible', policy_comparator.db.Money(length=32), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['quote_id'], ['pc_normalized_quotes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('pc_quote_coverages', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pc_quote_coverages_quote_id'), ['quote_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pc_quote_coverages_tenant_id'), ['tenant_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('pc_quote_coverages', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_quote_coverages_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_coverages_quote_id'))

    op.drop_table('pc_quote_coverages')
    with op.batch_alter_table('pc_provider_raw_responses', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_provider_raw_responses_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_raw_responses_provider_attempt_id'))

    op.drop_table('pc_provider_raw_responses')
    with op.batch_alter_table('pc_provider_missing_fields', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_provider_missing_fields_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_missing_fields_provider_attempt_id'))

    op.drop_table('pc_provider_missing_fields')
    with op.batch_alter_table('pc_normalized_quotes', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_normalized_quotes_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_normalized_quotes_quote_request_id'))
        batch_op.drop_index(batch_op.f('ix_pc_normalized_quotes_provider_attempt_id'))

    op.drop_table('pc_normalized_quotes')
    with op.batch_alter_table('pc_provider_attempts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_provider_attempts_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_attempts_status'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_attempts_quote_request_id'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_attempts_provider_id'))

    op.drop_table('pc_provider_attempts')
    with op.batch_alter_table('pc_quote_requests', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_quote_requests_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_requests_status'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_requests_customer_id'))

    op.drop_table('pc_quote_requests')
    with op.batch_alter_table('pc_customer_profiles', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_customer_profiles_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_customer_profiles_customer_id'))

    op.drop_table('pc_customer_profiles')
    with op.batch_alter_table('pc_consent_records', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_consent_records_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_consent_records_quote_request_id'))
        batch_op.drop_index(batch_op.f('ix_pc_consent_records_customer_id'))

    op.drop_table('pc_consent_records')
    with op.batch_alter_table('pc_vehicles', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_vehicles_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_vehicles_plate'))

    op.drop_table('pc_vehicles')
    with op.batch_alter_table('pc_staff_users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_staff_users_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_staff_users_email'))

    op.drop_table('pc_staff_users')
    with op.batch_alter_table('pc_quote_jobs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_status'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_run_after'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_quote_request_id'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_provider_attempt_id'))
        batch_op.drop_index(batch_op.f('ix_pc_quote_jobs_lease_expires_at'))

    op.drop_table('pc_quote_jobs')
    with op.batch_alter_table('pc_provider_health', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_provider_health_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_provider_health_provider_id'))

    op.drop_table('pc_provider_health')
    with op.batch_alter_table('pc_insurance_histories', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_insurance_histories_tenant_id'))

    op.drop_table('pc_insurance_histories')
    with op.batch_alter_table('pc_customers', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_customers_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_customers_email_fingerprint'))

    op.drop_table('pc_customers')
    with op.batch_alter_table('pc_coverage_preferences', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_coverage_preferences_tenant_id'))

    op.drop_table('pc_coverage_preferences')
    with op.batch_alter_table('pc_audit_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pc_audit_events_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_pc_audit_events_entity_id'))
        batch_op.drop_index(batch_op.f('ix_pc_audit_events_created_at'))
        batch_op.drop_index(batch_op.f('ix_pc_audit_events_actor_user_id'))
        batch_op.drop_index(batch_op.f('ix_pc_audit_events_action'))

    op.drop_table('pc_audit_events')
