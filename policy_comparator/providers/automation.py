"""Optional Playwright driver for authorized portal automation.

Nothing in this module is imported at application start: Playwright is an extra,
and the API and worker both run without it. It is only reached when a provider
is explicitly configured with ``mode=browser`` *and* every live-submission
precondition holds.

Two rules are absolute here:

* **Anti-bot protection is never bypassed.** A CAPTCHA, an MFA prompt or an
  unexpected login wall ends the attempt with ``manual_action_required``. There
  is no solver, no stealth patching and no retry-until-it-passes loop.
* **Diagnostics are sanitized.** Failure screenshots are only captured when
  diagnostics are enabled, and full page HTML is only written when
  ``PC_STORE_RAW_PAGES`` is explicitly turned on, because a rendered quote page
  contains the customer's personal data.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..config import Settings
from ..models.enums import QuoteOutcome
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import ProviderResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.async_api import Page

logger = logging.getLogger(__name__)


def playwright_available() -> bool:
    """Whether the optional Playwright extra is installed."""
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


#: Markers that mean "a human has to take over". Matched against page content
#: and frame URLs. Deliberately broad — a false positive costs one manual
#: review, a false negative means automating past a protection we must respect.
CAPTCHA_MARKERS: tuple[str, ...] = (
    "recaptcha",
    "g-recaptcha",
    "hcaptcha",
    "turnstile",
    "px-captcha",
    "datadome",
    "captcha",
    "verifica di sicurezza",
    "non sono un robot",
    "sei un robot",
    "attendi qualche istante",
    "checking your browser",
)

MFA_MARKERS: tuple[str, ...] = (
    "codice otp",
    "one-time password",
    "verifica in due passaggi",
    "two-factor",
    "autenticazione a due fattori",
    "inserisci il codice inviato",
)

LOGIN_MARKERS: tuple[str, ...] = (
    "area riservata",
    "accedi al tuo account",
    "sessione scaduta",
    "credenziali non valide",
)


@dataclass(frozen=True)
class Step:
    """One interaction in a portal flow.

    ``value_path`` reads from the standardized profile; ``value`` is a literal.
    Steps marked ``optional`` are skipped when their selector is absent, which
    keeps a cosmetic layout change from failing the whole flow.
    """

    action: str  # "fill" | "click" | "select" | "check" | "wait_for"
    selector: str
    value_path: str | None = None
    value: str | None = None
    optional: bool = False
    timeout_ms: int = 15_000


@dataclass(frozen=True)
class BrowserFlow:
    """A provider's portal flow. Lives in that provider's own module."""

    url: str
    steps: tuple[Step, ...] = ()
    #: Appears once quotes have rendered.
    result_selector: str | None = None
    #: Appears when the portal wants more information before quoting.
    missing_info_selector: str | None = None
    #: Pulls quote payloads off the rendered page.
    extract: Callable[["Page"], Awaitable[list[dict[str, Any]]]] | None = None
    navigation_timeout_ms: int = 45_000
    extra_http_headers: dict[str, str] = field(default_factory=dict)


class ManualActionRequired(RuntimeError):
    """A protection was encountered that must not be automated around."""


async def _page_text(page: "Page") -> str:
    try:
        return (await page.content()).lower()
    except Exception:  # pragma: no cover - page may be closing
        return ""


async def detect_manual_action(page: "Page") -> str | None:
    """Return a reason string when a human must take over, else ``None``."""
    content = await _page_text(page)
    for marker in CAPTCHA_MARKERS:
        if marker in content:
            return f"anti-bot protection detected ({marker})"
    for marker in MFA_MARKERS:
        if marker in content:
            return "multi-factor authentication prompt detected"
    for marker in LOGIN_MARKERS:
        if marker in content:
            return "unexpected login or expired session"
    return None


async def _capture_diagnostics(
    page: "Page", settings: Settings, provider_id: str, attempt_id: str
) -> str | None:
    """Write a failure screenshot. Returns its path relative to the artifact dir."""
    if not settings.store_diagnostics:
        return None
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        directory = settings.diagnostics_dir / provider_id
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{stamp}-{attempt_id[:8]}-{uuid.uuid4().hex[:6]}.png"
        await page.screenshot(path=str(directory / name), full_page=False)

        if settings.store_raw_pages:
            # Off by default: a rendered quote page contains customer PII.
            html_path = directory / name.replace(".png", ".html")
            html_path.write_text(await page.content(), encoding="utf-8")

        return str(Path(provider_id) / name)
    except Exception:  # pragma: no cover - diagnostics must never mask the error
        logger.warning("Could not capture diagnostics for %s", provider_id, exc_info=False)
        return None


def _resolve(step: Step, profile: QuotationProfile) -> str:
    if step.value is not None:
        return step.value
    if step.value_path is None:
        return ""
    raw = profile.get_path(step.value_path)
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw)


async def run_browser_flow(
    *,
    flow: BrowserFlow,
    profile: QuotationProfile,
    provider_id: str,
    settings: Settings,
    attempt_id: str,
    timeout_seconds: int,
) -> ProviderResult:
    """Drive one portal flow in an isolated browser context.

    A fresh context per attempt means no cookie, storage or session state is
    shared between two customers' quotations.
    """
    if not playwright_available():
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="playwright_not_installed",
            error_message=(
                "Browser automation is configured for this provider but Playwright "
                "is not installed. Install the extra with `pip install -e .[browser]` "
                "and run `playwright install chromium`."
            ),
        )

    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeout
    from playwright.async_api import async_playwright

    diagnostic_path: str | None = None

    async with async_playwright() as pw:
        browser = None
        context = None
        try:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                locale="it-IT",
                timezone_id="Europe/Rome",
                extra_http_headers=flow.extra_http_headers or {},
            )
            context.set_default_timeout(timeout_seconds * 1000)
            context.set_default_navigation_timeout(flow.navigation_timeout_ms)
            page = await context.new_page()

            await page.goto(flow.url, wait_until="domcontentloaded")

            reason = await detect_manual_action(page)
            if reason:
                diagnostic_path = await _capture_diagnostics(
                    page, settings, provider_id, attempt_id
                )
                return ProviderResult(
                    provider_id=provider_id,
                    outcome=QuoteOutcome.MANUAL_ACTION_REQUIRED,
                    error_category="manual_action_required",
                    error_message=f"{provider_id}: {reason} on the entry page",
                    raw_payload={"diagnostic_artifact": diagnostic_path} if diagnostic_path else {},
                )

            for step in flow.steps:
                locator = page.locator(step.selector).first
                try:
                    if step.action == "wait_for":
                        await locator.wait_for(timeout=step.timeout_ms)
                        continue
                    if step.optional and await locator.count() == 0:
                        continue
                    if step.action == "fill":
                        await locator.fill(_resolve(step, profile), timeout=step.timeout_ms)
                    elif step.action == "select":
                        await locator.select_option(
                            _resolve(step, profile), timeout=step.timeout_ms
                        )
                    elif step.action == "check":
                        await locator.check(timeout=step.timeout_ms)
                    elif step.action == "click":
                        await locator.click(timeout=step.timeout_ms)
                    else:  # pragma: no cover - guarded by the dataclass contract
                        raise ValueError(f"Unknown browser step action: {step.action}")
                except PlaywrightTimeout:
                    if step.optional:
                        continue
                    diagnostic_path = await _capture_diagnostics(
                        page, settings, provider_id, attempt_id
                    )
                    return ProviderResult(
                        provider_id=provider_id,
                        outcome=QuoteOutcome.FAILED,
                        error_category="selector_not_found",
                        error_message=(
                            f"{provider_id}: the portal element for step "
                            f"'{step.action}' was not found. The site layout has "
                            "probably changed and the selectors need re-verifying."
                        ),
                        raw_payload=(
                            {"diagnostic_artifact": diagnostic_path} if diagnostic_path else {}
                        ),
                    )

                # Protections often appear only after the first interaction.
                reason = await detect_manual_action(page)
                if reason:
                    diagnostic_path = await _capture_diagnostics(
                        page, settings, provider_id, attempt_id
                    )
                    return ProviderResult(
                        provider_id=provider_id,
                        outcome=QuoteOutcome.MANUAL_ACTION_REQUIRED,
                        error_category="manual_action_required",
                        error_message=f"{provider_id}: {reason}",
                        raw_payload=(
                            {"diagnostic_artifact": diagnostic_path} if diagnostic_path else {}
                        ),
                    )

            if flow.missing_info_selector:
                if await page.locator(flow.missing_info_selector).count() > 0:
                    return ProviderResult(
                        provider_id=provider_id,
                        outcome=QuoteOutcome.MISSING_INFORMATION,
                        error_category="provider_requires_more_data",
                        error_message=f"{provider_id}: the portal asked for additional data",
                        resume_token={"url": page.url},
                    )

            if flow.result_selector:
                await page.locator(flow.result_selector).first.wait_for(
                    timeout=timeout_seconds * 1000
                )

            raw_quotes = await flow.extract(page) if flow.extract else []
            if not raw_quotes:
                diagnostic_path = await _capture_diagnostics(
                    page, settings, provider_id, attempt_id
                )
                return ProviderResult(
                    provider_id=provider_id,
                    outcome=QuoteOutcome.UNAVAILABLE,
                    error_category="no_quotes_returned",
                    error_message=f"{provider_id}: the portal returned no quotes",
                    raw_payload=(
                        {"diagnostic_artifact": diagnostic_path} if diagnostic_path else {}
                    ),
                )

            return ProviderResult(
                provider_id=provider_id,
                outcome=QuoteOutcome.QUOTED,
                raw_quotes=raw_quotes,
                raw_status="portal_quoted",
            )

        except PlaywrightTimeout:
            return ProviderResult(
                provider_id=provider_id,
                outcome=QuoteOutcome.TIMED_OUT,
                error_category="portal_timeout",
                error_message=f"{provider_id}: the portal did not respond in time",
            )
        except PlaywrightError as exc:
            # Message only — never the exception's page context, which can echo
            # back the data that was typed into the form.
            return ProviderResult(
                provider_id=provider_id,
                outcome=QuoteOutcome.FAILED,
                error_category="browser_error",
                error_message=f"{provider_id}: browser automation failed ({type(exc).__name__})",
            )
        finally:
            # Always tear the context and browser down, even on the error paths.
            if context is not None:
                try:
                    await context.close()
                except Exception:  # pragma: no cover
                    pass
            if browser is not None:
                try:
                    await browser.close()
                except Exception:  # pragma: no cover
                    pass
