"""Shared vocabulary for provider outcomes, request lifecycle and field origin.

These are ``str`` enums so they serialize straight to JSON and compare equal to
their stored database value.
"""

from __future__ import annotations

from enum import StrEnum


class ProviderType(StrEnum):
    """What sits behind a provider id.

    An ``aggregator`` is not an insurer: it returns quotes issued by other
    companies, which may duplicate what a direct insurer adapter returns.
    """

    INSURER = "insurer"
    AGGREGATOR = "aggregator"


class ProviderMode(StrEnum):
    MOCK = "mock"
    API = "api"
    BROWSER = "browser"


class QuoteOutcome(StrEnum):
    """Result of a single provider attempt."""

    QUOTED = "quoted"
    MISSING_INFORMATION = "missing_information"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    MANUAL_ACTION_REQUIRED = "manual_action_required"
    AUTHENTICATION_REQUIRED = "authentication_required"
    CONFIGURATION_ERROR = "configuration_error"
    FAILED = "failed"

    @property
    def is_terminal_success(self) -> bool:
        return self is QuoteOutcome.QUOTED

    @property
    def is_retryable(self) -> bool:
        """Whether an automatic retry could plausibly change the result.

        Human-gated outcomes (missing information, CAPTCHA/MFA, bad
        credentials, broken configuration) are never retried automatically —
        retrying would just replay the same failure.
        """
        return self in {
            QuoteOutcome.UNAVAILABLE,
            QuoteOutcome.TIMED_OUT,
            QuoteOutcome.FAILED,
        }

    @property
    def counts_against_circuit(self) -> bool:
        """Whether this outcome indicates the provider itself is unhealthy."""
        return self in {
            QuoteOutcome.UNAVAILABLE,
            QuoteOutcome.TIMED_OUT,
            QuoteOutcome.FAILED,
        }


class AttemptStatus(StrEnum):
    """Lifecycle of one provider attempt, as shown on the progress screen."""

    WAITING = "waiting"
    RUNNING = "running"
    RETRYING = "retrying"
    QUOTED = "quoted"
    MISSING_INFORMATION = "missing_information"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    MANUAL_ACTION_REQUIRED = "manual_action_required"
    AUTHENTICATION_REQUIRED = "authentication_required"
    CONFIGURATION_ERROR = "configuration_error"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: The circuit breaker was open, so the provider was not contacted at all.
    SKIPPED_CIRCUIT_OPEN = "skipped_circuit_open"

    @property
    def is_pending(self) -> bool:
        return self in {AttemptStatus.WAITING, AttemptStatus.RUNNING, AttemptStatus.RETRYING}

    @property
    def is_finished(self) -> bool:
        return not self.is_pending

    @classmethod
    def from_outcome(cls, outcome: QuoteOutcome) -> "AttemptStatus":
        return cls(outcome.value)


class RequestStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    AWAITING_INFORMATION = "awaiting_information"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FieldSource(StrEnum):
    """Where a profile value came from.

    A value confirmed by staff is authoritative: a provider response may fill a
    blank, but must never overwrite ``STAFF``.
    """

    STAFF = "staff"
    PROVIDER = "provider"
    DERIVED = "derived"


class DrivingFormula(StrEnum):
    """Who is allowed to drive under the policy."""

    FREE = "free"                 # guida libera
    EXPERT = "expert"             # guida esperta
    EXCLUSIVE = "exclusive"       # guida esclusiva


class PaymentFrequency(StrEnum):
    ANNUAL = "annual"
    INSTALMENTS = "instalments"


class ConsentType(StrEnum):
    """Processing consent is mandatory; marketing consent is always separate."""

    PRIVACY_PROCESSING = "privacy_processing"
    PROVIDER_DATA_TRANSFER = "provider_data_transfer"
    MARKETING = "marketing"


class AuditAction(StrEnum):
    REQUEST_CREATED = "request_created"
    CONSENT_RECORDED = "consent_recorded"
    PROFILE_UPDATED = "profile_updated"
    PROVIDERS_STARTED = "providers_started"
    PROVIDER_SUBMITTED = "provider_submitted"
    PROVIDER_RESULT = "provider_result"
    PROVIDER_RETRIED = "provider_retried"
    REQUEST_CANCELLED = "request_cancelled"
    RESULTS_VIEWED = "results_viewed"
