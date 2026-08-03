"""SQLAlchemy models for the policy comparator.

Importing this package registers every mapper, which is what
:func:`policy_comparator.db.create_all` and the Alembic migrations rely on.
"""

from .customer import (
    ConsentRecord,
    CoveragePreference,
    Customer,
    CustomerProfile,
    InsuranceHistory,
    Vehicle,
)
from .enums import (
    AttemptStatus,
    AuditAction,
    ConsentType,
    DrivingFormula,
    FieldSource,
    JobStatus,
    PaymentFrequency,
    ProviderMode,
    ProviderType,
    QuoteOutcome,
    RequestStatus,
)
from .jobs import AuditEvent, QuoteJob, StaffUser
from .quoting import (
    NormalizedQuote,
    ProviderAttempt,
    ProviderHealthState,
    ProviderMissingFieldRequest,
    ProviderRawResponse,
    QuoteCoverage,
    QuoteRequest,
)

__all__ = [
    "AttemptStatus",
    "AuditAction",
    "AuditEvent",
    "ConsentRecord",
    "ConsentType",
    "CoveragePreference",
    "Customer",
    "CustomerProfile",
    "DrivingFormula",
    "FieldSource",
    "InsuranceHistory",
    "JobStatus",
    "NormalizedQuote",
    "PaymentFrequency",
    "ProviderAttempt",
    "ProviderHealthState",
    "ProviderMissingFieldRequest",
    "ProviderMode",
    "ProviderRawResponse",
    "ProviderType",
    "QuoteCoverage",
    "QuoteJob",
    "QuoteOutcome",
    "QuoteRequest",
    "RequestStatus",
    "StaffUser",
    "Vehicle",
]
