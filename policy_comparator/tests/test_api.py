"""HTTP API: authentication, tenant isolation, and the full staff workflow."""

from __future__ import annotations

import uuid

import pytest

from policy_comparator.api import deps
from policy_comparator.models import (
    AuditEvent,
    ConsentRecord,
    CoveragePreference,
    QuoteRequest,
)
from policy_comparator.models.enums import ConsentType, RequestStatus
from policy_comparator.tests.conftest import (
    FULL_ANSWERS,
    TENANT_A,
    TENANT_B,
    auth_headers,
    make_identity,
)
from policy_comparator.worker.runner import Worker


def drain(**overrides) -> None:
    """Run the worker to quiescence.

    Retry backoff is zeroed so a test never sleeps; the backoff schedule itself
    is covered in test_worker_resilience.py.
    """
    from dataclasses import replace

    from policy_comparator.config import get_settings

    settings = replace(get_settings(), provider_retry_backoff_seconds=0.0, **overrides)
    worker = Worker(settings=settings)
    for _ in range(20):
        if worker.run_once() == 0:
            break


def create_request(client, headers, body) -> str:
    response = client.post("/api/quotes", json=body, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["request_id"]


class TestAuthentication:
    def test_endpoints_reject_an_anonymous_caller(self, client, new_request_body):
        for method, path, body in (
            ("get", "/api/quotes", None),
            # A well-formed body, so a 422 cannot mask a missing auth check.
            ("post", "/api/quotes", new_request_body),
            ("get", "/api/providers", None),
            ("get", "/api/providers/health", None),
            ("get", f"/api/quotes/{uuid.uuid4()}", None),
            ("post", f"/api/quotes/{uuid.uuid4()}/start", None),
        ):
            response = (
                client.get(path) if method == "get" else client.post(path, json=body)
            )
            assert response.status_code == 401, f"{method} {path} was not protected"

    def test_a_malformed_token_is_rejected(self, client):
        response = client.get("/api/quotes", headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401

    def test_a_token_without_a_tenant_is_rejected(self, client):
        from jose import jwt

        from policy_comparator.config import get_settings

        settings = get_settings()
        token = jwt.encode(
            {"sub": str(uuid.uuid4()), "email": "x@y.it"},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        response = client.get("/api/quotes", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401
        assert "tenant" in response.json()["detail"].lower()

    def test_login_issues_a_working_token(self, client, staff_user):
        response = client.post(
            "/api/auth/login",
            json={"email": "staff@example.com", "password": "correct-horse"},
        )
        assert response.status_code == 200
        token = response.json()["access_token"]

        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["tenant_id"] == str(TENANT_A)

    def test_a_wrong_password_is_rejected(self, client, staff_user):
        response = client.post(
            "/api/auth/login",
            json={"email": "staff@example.com", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_an_unknown_email_gives_the_same_error(self, client, staff_user):
        """The message must not reveal whether the account exists."""
        unknown = client.post(
            "/api/auth/login", json={"email": "nobody@example.com", "password": "x"}
        )
        wrong = client.post(
            "/api/auth/login", json={"email": "staff@example.com", "password": "x"}
        )
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json()["detail"] == wrong.json()["detail"]


class TestTenantIsolation:
    def test_another_tenant_cannot_read_a_request(self, client, new_request_body):
        request_id = create_request(client, auth_headers(), new_request_body)

        other = auth_headers(make_identity(TENANT_B))
        # 404 rather than 403: confirming the id exists would itself leak.
        assert client.get(f"/api/quotes/{request_id}", headers=other).status_code == 404

    def test_another_tenant_cannot_act_on_a_request(self, client, new_request_body):
        request_id = create_request(client, auth_headers(), new_request_body)
        other = auth_headers(make_identity(TENANT_B))

        for method, path, body in (
            ("post", f"/api/quotes/{request_id}/start", None),
            ("get", f"/api/quotes/{request_id}/progress", None),
            ("get", f"/api/quotes/{request_id}/results", None),
            ("get", f"/api/quotes/{request_id}/missing-fields", None),
            ("post", f"/api/quotes/{request_id}/cancel", None),
            ("post", f"/api/quotes/{request_id}/retry", {"provider_id": "zurich"}),
            ("post", f"/api/quotes/{request_id}/missing-fields", {"updates": {"vehicle.make": "X"}}),
        ):
            response = (
                client.get(path, headers=other)
                if method == "get"
                else client.post(path, json=body, headers=other)
            )
            assert response.status_code == 404, f"{method} {path} leaked across tenants"

    def test_listing_only_returns_the_callers_tenant(self, client, new_request_body):
        create_request(client, auth_headers(), new_request_body)
        create_request(client, auth_headers(make_identity(TENANT_B)), new_request_body)

        mine = client.get("/api/quotes", headers=auth_headers()).json()["requests"]
        theirs = client.get(
            "/api/quotes", headers=auth_headers(make_identity(TENANT_B))
        ).json()["requests"]

        assert len(mine) == 1 and len(theirs) == 1
        assert mine[0]["request_id"] != theirs[0]["request_id"]


class TestConsentGate:
    def test_creation_requires_privacy_consent(self, client, new_request_body):
        new_request_body["privacy_accepted"] = False
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 400
        assert "consent" in response.json()["detail"].lower()

    def test_creation_requires_transfer_consent(self, client, new_request_body):
        new_request_body["provider_data_transfer_accepted"] = False
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 400

    def test_marketing_consent_is_separate_and_optional(self, client, db, new_request_body):
        request_id = create_request(client, auth_headers(), new_request_body)

        records = db.query(ConsentRecord).filter_by(quote_request_id=uuid.UUID(request_id)).all()
        types = {r.consent_type for r in records}
        assert ConsentType.PRIVACY_PROCESSING.value in types
        assert ConsentType.PROVIDER_DATA_TRANSFER.value in types
        # Not granted, so no marketing record exists at all.
        assert ConsentType.MARKETING.value not in types

    def test_transfer_consent_is_scoped_to_the_selected_providers(
        self, client, db, new_request_body
    ):
        new_request_body["selected_provider_ids"] = ["zurich"]
        request_id = create_request(client, auth_headers(), new_request_body)

        record = (
            db.query(ConsentRecord)
            .filter_by(
                quote_request_id=uuid.UUID(request_id),
                consent_type=ConsentType.PROVIDER_DATA_TRANSFER.value,
            )
            .one()
        )
        assert record.scope_provider_ids == ["zurich"]

        # A provider outside that scope cannot be retried into the request.
        response = client.post(
            f"/api/quotes/{request_id}/retry",
            json={"provider_id": "allianz"},
            headers=auth_headers(),
        )
        assert response.status_code == 404

    def test_starting_without_a_consent_record_is_refused(self, client, db, new_request_body):
        request_id = create_request(client, auth_headers(), new_request_body)

        # Simulate a withdrawn consent.
        db.query(ConsentRecord).filter_by(
            quote_request_id=uuid.UUID(request_id),
            consent_type=ConsentType.PROVIDER_DATA_TRANSFER.value,
        ).delete()
        db.commit()

        response = client.post(f"/api/quotes/{request_id}/start", headers=auth_headers())
        assert response.status_code == 403


class TestValidation:
    def test_an_unknown_provider_is_rejected(self, client, new_request_body):
        new_request_body["selected_provider_ids"] = ["zurich", "not-a-provider"]
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 400

    def test_no_provider_selected_is_rejected(self, client, new_request_body):
        new_request_body["selected_provider_ids"] = []
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 422

    def test_a_bad_email_is_rejected(self, client, new_request_body):
        new_request_body["customer_email"] = "not-an-email"
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 422

    def test_unknown_body_fields_are_rejected(self, client, new_request_body):
        new_request_body["injected"] = "surprise"
        response = client.post("/api/quotes", json=new_request_body, headers=auth_headers())
        assert response.status_code == 422


class TestRateLimit:
    def test_submissions_are_capped_per_tenant(self, client, new_request_body, monkeypatch):
        from dataclasses import replace

        from policy_comparator import config

        limited = replace(config.get_settings(), quote_rate_limit_per_hour=2)
        monkeypatch.setattr(deps, "get_settings", lambda: limited)

        headers = auth_headers()
        assert client.post("/api/quotes", json=new_request_body, headers=headers).status_code == 201
        assert client.post("/api/quotes", json=new_request_body, headers=headers).status_code == 201

        third = client.post("/api/quotes", json=new_request_body, headers=headers)
        assert third.status_code == 429


class TestProviderEndpoints:
    def test_the_catalogue_lists_every_adapter(self, client):
        payload = client.get("/api/providers", headers=auth_headers()).json()
        ids = {p["provider_id"] for p in payload["providers"]}
        assert ids == {"zurich", "allianz", "generali", "cercassicurazioni"}
        assert payload["live_provider_automation"] is False

    def test_the_aggregator_is_labelled_as_one(self, client):
        payload = client.get("/api/providers", headers=auth_headers()).json()
        by_id = {p["provider_id"]: p for p in payload["providers"]}
        assert by_id["cercassicurazioni"]["provider_type"] == "aggregator"
        assert by_id["zurich"]["provider_type"] == "insurer"

    def test_health_reports_mock_mode_and_no_live_link(self, client):
        payload = client.get("/api/providers/health", headers=auth_headers()).json()
        assert len(payload["providers"]) == 4
        assert all(p["mode"] == "mock" for p in payload["providers"])
        assert all(p["live_enabled"] is False for p in payload["providers"])


class TestFullWorkflow:
    """The whole staff journey, end to end, through the HTTP API."""

    def test_create_start_answer_resume_and_recommend(self, client, db, new_request_body):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)

        # 1. Nothing is transmitted until the staff user starts it.
        assert db.get(QuoteRequest, uuid.UUID(request_id)).status == RequestStatus.DRAFT.value

        started = client.post(f"/api/quotes/{request_id}/start", headers=headers)
        assert started.status_code == 200
        assert set(started.json()["queued_providers"]) == set(
            new_request_body["selected_provider_ids"]
        )

        # 2. Every provider asks for more information.
        drain()
        progress = client.get(f"/api/quotes/{request_id}/progress", headers=headers).json()
        assert progress["pending"] == 0
        assert {p["status"] for p in progress["providers"]} == {"missing_information"}
        assert progress["demonstration_data"] is True

        # 3. The same question from several providers is asked once.
        missing = client.get(f"/api/quotes/{request_id}/missing-fields", headers=headers).json()
        paths = [f["field_path"] for g in missing["groups"] for f in g["fields"]]
        assert len(paths) == len(set(paths)), "a question was duplicated across providers"

        merit = next(
            f for g in missing["groups"] for f in g["fields"]
            if f["field_path"] == "history.universal_merit_class"
        )
        assert len(merit["requested_by"]) > 1, "shared questions must attribute every asker"

        # 4. Answering resumes only the providers that were waiting.
        answered = client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        assert answered.status_code == 200
        assert set(answered.json()["resumed_providers"]) == set(
            new_request_body["selected_provider_ids"]
        )

        # 5. Results: a recommendation, a comparison, and nothing hidden.
        drain()
        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()

        assert results["recommended_quote_id"] is not None
        assert "più economico" in results["recommendation_explanation"].lower()
        assert results["eligible_quotes"]
        assert results["eligible_quotes"][0]["recommended"] is True
        assert results["demonstration_data"] is True

        # The recommendation really is the cheapest eligible quote.
        premiums = [float(q["annual_total_premium"]) for q in results["eligible_quotes"]]
        assert premiums == sorted(premiums)

        # The aggregator's copies of Zurich and Allianz are marked duplicate,
        # not silently dropped.
        duplicates = [
            q for q in results["ineligible_quotes"]
            if any(r["code"] == "duplicate" for r in q["ineligible_reasons"])
        ]
        assert duplicates, "aggregator duplicates should be visible and flagged"
        assert all(q["duplicate_of_quote_id"] for q in duplicates)

    def test_a_failed_provider_stays_visible_and_can_be_retried(
        self, client, db, new_request_body, monkeypatch
    ):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        headers = auth_headers()
        new_request_body["selected_provider_ids"] = ["zurich", "allianz"]
        request_id = create_request(client, headers, new_request_body)

        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()

        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        unavailable = {p["provider_id"] for p in results["unavailable_providers"]}
        assert "zurich" in unavailable, "a failed provider must never be omitted"
        assert results["eligible_quotes"], "the healthy provider still produced a quote"

        # Zurich recovers and is retried on its own.
        monkeypatch.delenv("PC_MOCK_FORCE_OUTCOME_ZURICH")
        retry = client.post(
            f"/api/quotes/{request_id}/retry",
            json={"provider_id": "zurich"},
            headers=headers,
        )
        assert retry.status_code == 200
        drain()

        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        assert not results["unavailable_providers"]
        insurers = {q["insurer_name"] for q in results["eligible_quotes"]}
        assert "Zurich" in insurers

    def test_editing_requirements_changes_the_recommendation(
        self, client, db, new_request_body
    ):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()

        before = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        cheapest = before["eligible_quotes"][0]

        # Refuse whatever the cheapest quote requires, and it must lose.
        if cheapest["requires_black_box"]:
            body = {"accepts_black_box": False}
        elif cheapest["requires_approved_repair_network"]:
            body = {"accepts_approved_repair_network": False}
        else:
            pytest.skip("the cheapest demonstration quote carries no restriction to refuse")

        assert client.put(
            f"/api/quotes/{request_id}/preferences", json=body, headers=headers
        ).status_code == 200

        after = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        assert after["recommended_quote_id"] != before["recommended_quote_id"]
        excluded = [
            q for q in after["ineligible_quotes"] if q["quote_id"] == cheapest["quote_id"]
        ]
        assert excluded, "the refused quote must appear with its reason, not vanish"
        assert excluded[0]["ineligible_reasons"]

    def test_cancelling_stops_pending_work(self, client, db, new_request_body):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)

        response = client.post(f"/api/quotes/{request_id}/cancel", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == RequestStatus.CANCELLED.value
        assert response.json()["cancelled_jobs"] == 4


class TestRequirementsUpdateWorkflow:
    """The PUT the 'Aggiorna i requisiti' button sends, end to end."""

    def _quoted_request(self, client, headers, body) -> str:
        request_id = create_request(client, headers, body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()
        return request_id

    def test_refusing_the_black_box_changes_eligibility_and_recommendation(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = self._quoted_request(client, headers, new_request_body)

        before = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        cheapest = before["eligible_quotes"][0]
        assert cheapest["requires_black_box"] is True, (
            "the demonstration set should have a cheapest quote that needs a box"
        )

        response = client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": False},
            headers=headers,
        )
        assert response.status_code == 200
        assert "preferences.accepts_black_box" in response.json()["updated_fields"]

        after = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        assert after["recommended_quote_id"] != before["recommended_quote_id"]

        excluded = next(
            q for q in after["ineligible_quotes"] if q["quote_id"] == cheapest["quote_id"]
        )
        assert any(r["code"] == "black_box_refused" for r in excluded["ineligible_reasons"])

    def test_a_false_boolean_is_stored_as_false_not_dropped(
        self, client, db, new_request_body
    ):
        """"No" is a requirement; treating it as "unset" would be a silent bug."""
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)

        client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": False, "accepts_approved_repair_network": False},
            headers=headers,
        )

        request = db.get(QuoteRequest, uuid.UUID(request_id))
        db.refresh(request)
        preferences = db.get(CoveragePreference, request.coverage_preference_id)
        db.refresh(preferences)
        assert preferences.accepts_black_box is False
        assert preferences.accepts_approved_repair_network is False

    def test_clearing_a_requirement_restores_the_cheaper_quote(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = self._quoted_request(client, headers, new_request_body)

        original = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": False},
            headers=headers,
        )
        restricted = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        assert restricted["recommended_quote_id"] != original["recommended_quote_id"]

        # Back to "Indifferente".
        client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": None},
            headers=headers,
        )
        cleared = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        assert cleared["recommended_quote_id"] == original["recommended_quote_id"]

    def test_the_updated_requirements_come_back_in_the_results(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = self._quoted_request(client, headers, new_request_body)

        client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"max_acceptable_deductible": "350", "driving_formula": "expert"},
            headers=headers,
        )
        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()

        assert results["requirements"]["max_acceptable_deductible"] == "350"
        assert results["requirements"]["driving_formula"] == "expert"

    def test_another_tenant_cannot_update_requirements(self, client, new_request_body):
        request_id = create_request(client, auth_headers(), new_request_body)
        other = auth_headers(make_identity(TENANT_B))
        response = client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": False},
            headers=other,
        )
        assert response.status_code == 404


class TestResultsCalculationPayload:
    def test_every_demonstration_quote_carries_its_breakdown(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()

        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        for quote in results["eligible_quotes"]:
            assert quote["calculation_source"] == "demonstration_formula"
            breakdown = quote["calculation_breakdown"]
            assert breakdown is not None
            assert breakdown["annual_total"] == quote["annual_total_premium"]
            assert breakdown["rounding"]
            codes = {s["code"] for s in breakdown["steps"]}
            assert {"base_premium", "merit_class", "annual_total"} <= codes

    def test_the_purchase_link_is_flagged_as_a_demonstration(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()

        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        for quote in results["eligible_quotes"]:
            assert quote["purchase_url_is_demonstration"] is True
            # Still https, so the UI never has to sanitize it itself.
            assert quote["purchase_url"].startswith("https://")

    def test_satisfied_requirements_are_listed_for_the_recommendation(
        self, client, new_request_body
    ):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )
        drain()
        client.put(
            f"/api/quotes/{request_id}/preferences",
            json={"accepts_black_box": False},
            headers=headers,
        )

        results = client.get(f"/api/quotes/{request_id}/results", headers=headers).json()
        best = next(q for q in results["eligible_quotes"] if q["recommended"])
        assert any("scatola nera" in r.lower() for r in best["satisfied_requirements"])


class TestAudit:
    def test_the_trail_records_the_workflow(self, client, db, new_request_body):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.get(f"/api/quotes/{request_id}/results", headers=headers)

        payload = client.get(f"/api/quotes/{request_id}/audit", headers=headers).json()
        actions = {e["action"] for e in payload["events"]}
        assert {"request_created", "consent_recorded", "providers_started"} <= actions

    def test_non_admins_cannot_read_the_trail(self, client, new_request_body):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)

        member = auth_headers(make_identity(TENANT_A, role="staff"))
        response = client.get(f"/api/quotes/{request_id}/audit", headers=member)
        assert response.status_code == 403

    def test_audit_metadata_never_contains_customer_values(
        self, client, db, new_request_body
    ):
        headers = auth_headers()
        request_id = create_request(client, headers, new_request_body)
        client.post(f"/api/quotes/{request_id}/start", headers=headers)
        drain()
        client.post(
            f"/api/quotes/{request_id}/missing-fields",
            json={"updates": dict(FULL_ANSWERS)},
            headers=headers,
        )

        blob = " ".join(
            str(e.metadata_json) for e in db.query(AuditEvent).filter_by(tenant_id=TENANT_A)
        )
        # Field names are recorded; the values the staff member typed are not.
        assert "customer.tax_code" in blob
        assert "RSSMRA85C04H501Z" not in blob
        assert "cliente@esempio.it" not in blob
        assert "AB123CD" not in blob
