"""Regression tests for the document-request / calculation-request boundary.

A drafting request and a calculation request about the same subject share
almost all of their vocabulary: "contratto di locazione ... 400 euro al mese"
is a lease to be WRITTEN, "imposta di registro ... 400 euro al mese" is a tax
to be COMPUTED. The lexical calculator matcher scores topic overlap only, so it
cannot tell them apart (see the tripwire in
calculation_platform/tests/test_matcher_corpus.py). Three mechanisms keep that
blindness away from the user:

  1. /api/chat classifies generation intent BEFORE the RAG graph runs, so a
     drafting request never reaches the calculation gate on the chat path.
  2. The generation branch then calls the RAG graph itself, for the sources the
     draft will cite. That internal call runs with `skip_calculation=True`,
     because the gate would otherwise intercept it and the draft would come
     back with no sources and no citations.
  3. Document comparison (/api/generate) calls the graph the same way, and for
     a sharper reason: two calculators are themselves comparators and share the
     flow's opening verb.

Layers, from outermost in:

  section 0  the real FastAPI /api/chat route, with only external services
             stubbed — the one place the whole request is exercised;
  sections 1-2  the routing PREDICATE alone, in isolation from the endpoint;
  sections 3-3b  the two internal graph calls;
  section 4  the graph's entry router.
"""

import os
from types import SimpleNamespace

import pytest
import requests

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import src.rag.calculation as calculation


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._body


# The four messages the boundary has to get right. The first three are
# documents to be drafted; the fourth is a genuine calculation about the very
# same lease, and must stay on the calculation path.
IT_LEASE_DRAFT = "Redigi un contratto di locazione con un canone di 400 euro al mese."
EN_LEASE_DRAFT = "Create a rental contract with rent of $400 per month."
INVOICE_DRAFT = "Preparami una fattura con importo netto di 1000 euro e IVA al 22%."
GENUINE_CALCULATION = (
    "Calcola l'imposta di registro per un canone di 400 euro al mese."
)

DOCUMENT_REQUESTS = [
    pytest.param(IT_LEASE_DRAFT, id="it-lease-draft"),
    pytest.param(EN_LEASE_DRAFT, id="en-lease-draft"),
    pytest.param(INVOICE_DRAFT, id="invoice-draft"),
]


@pytest.fixture
def chat_api(monkeypatch):
    """The real /api/chat module, imported with inert LLM credentials."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    import src.chatbot.api as api

    # Every classifier in this module falls back to an LLM call for messages
    # its fast paths cannot settle. Stubbing it to abstain ("rag", the module's
    # own safe default) keeps these tests offline AND proves the deterministic
    # paths carry the decision on their own — an assertion that would be
    # vacuous if a live model were quietly supplying the answer.
    monkeypatch.setattr(api, "_call_chat", lambda *args, **kwargs: "rag")
    return api


def _enters_generation_branch(api, message: str) -> bool:
    """Evaluate the real /api/chat generation condition for `message`.

    Mirrors the `if` in the chat endpoint exactly: both operands, same order.
    """
    top_intent = api._classify_top_level_intent(message, "it")
    return top_intent == "generate" or api.is_generation_request(message)


def _retrieval_rows():
    """Neo4j rows as the graph really returns them.

    Three shapes in one result set, because the two consumers read different
    ones: `_raw_result_to_sections` (sources) wants a node with `properties`,
    and `_extract_citations` wants the `d`/`s` document and section rows.
    """
    return [
        {
            "n": {
                "labels": ["DocumentSection"],
                "properties": {
                    "heading": "Art. 1571 c.c.",
                    "text_it": "Nozione della locazione.",
                    "document_title": "Codice Civile",
                },
            }
        },
        {"d": {"id": "LEGAL_DOC::codice-civile", "name": "Codice Civile"}},
        {
            "s": {
                "id": "DOCUMENT_SECTION::codice-civile::art-1571",
                "name": "art-1571",
                "title": "Art. 1571",
                "text": "La locazione e il contratto col quale una parte si obbliga.",
            }
        },
    ]


# --- 0. Through the real /api/chat endpoint --------------------------------
# The tests in section 1 evaluate the routing PREDICATE in isolation; they say
# nothing about the endpoint around it. This one drives the actual FastAPI
# route: real request validation, real branch, real response model. Only the
# genuinely external services are replaced (Neo4j + the LLM behind rag_run,
# the LLM behind document generation, the session file, the SQL session).


@pytest.fixture
def chat_client(chat_api, monkeypatch, tmp_path):
    """A TestClient over the real app, with external services stubbed out.

    Anonymous on purpose: `current_user` being None skips the document-aware
    branch, which is the part that needs a populated database. The generation
    branch under test sits after it and does not.
    """
    from fastapi.testclient import TestClient
    from src.db.base import get_db

    # Sessions must not touch the real /opt/chatbot data file.
    monkeypatch.setattr(chat_api.chatbot, "_sessions", {})
    monkeypatch.setattr(
        chat_api.chatbot, "_sessions_file", str(tmp_path / "sessions.json")
    )

    chat_api.app.dependency_overrides[get_db] = lambda: None
    try:
        yield TestClient(chat_api.app)
    finally:
        chat_api.app.dependency_overrides.pop(get_db, None)


def test_chat_endpoint_routes_a_calculator_flavoured_draft_to_generation(
    chat_api, chat_client, monkeypatch
):
    """POST /api/chat with a drafting request full of calculator vocabulary.

    "contratto di locazione" + "400 euro al mese" scores 3 against
    legal_it.registration_tax_leases — auto-route strength. The reply must be
    the generated document, and no calculator may be consulted anywhere in the
    request.
    """
    captured = {}

    def fake_rag_run(query, **kwargs):
        captured["rag_kwargs"] = kwargs
        return {"raw_result": _retrieval_rows()}

    monkeypatch.setattr(chat_api, "rag_run", fake_rag_run)
    monkeypatch.setattr(
        chat_api,
        "generate_document",
        lambda *args, **kwargs: {
            "draft": "CONTRATTO DI LOCAZIONE\n\nTra le parti...",
            "case_details": {"canone": "400 euro al mese"},
            "doc_type": "rental_standard",
        },
    )
    # The empty local template catalog would otherwise resolve to "unknown" and
    # short-circuit into the clarification reply, never reaching generation.
    monkeypatch.setattr(
        chat_api, "classify_system_template", lambda *args, **kwargs: "rental_standard"
    )
    # Nothing in a generation turn may reach the calculation platform.
    monkeypatch.setattr(
        calculation.requests,
        "post",
        lambda url, **kwargs: pytest.fail(f"generation turn called the platform: {url}"),
    )

    response = chat_client.post("/api/chat", json={"message": IT_LEASE_DRAFT})

    assert response.status_code == 200
    body = response.json()
    # Chose generation...
    assert body["status_messages"] == ["generation_mode"]
    assert body["draft"].startswith("CONTRATTO DI LOCAZIONE")
    # ...and returned a document, not a computed figure.
    assert "imposta di registro" not in body["answer"].lower()
    assert "768" not in body["answer"]
    assert "calculation_result" not in body
    # The supporting retrieval inside that branch ran in retrieval-only mode.
    assert captured["rag_kwargs"]["skip_calculation"] is True


def test_chat_endpoint_keeps_an_ordinary_question_out_of_the_generation_branch(
    chat_api, chat_client, monkeypatch
):
    """The counterpart: a non-drafting message must reach the normal chat path.

    Guards against a fix that routes everything to generation and passes the
    test above for the wrong reason.
    """
    monkeypatch.setattr(
        chat_api,
        "generate_document",
        lambda *args, **kwargs: pytest.fail("a question must not be drafted"),
    )
    monkeypatch.setattr(
        chat_api.chatbot,
        "chat",
        lambda session_id, message, **kwargs: {
            "session_id": session_id,
            "answer": "L'imposta di registro sulle locazioni e il 2%.",
            "original_query": message,
            "resolved_query": message,
            "session_language": "it",
            "status_messages": [],
        },
    )

    response = chat_client.post(
        "/api/chat", json={"message": "Come funziona l'imposta di registro?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status_messages"] == []
    assert body["draft"] == ""
    assert body["answer"].startswith("L'imposta di registro")


# --- 1. Document requests reach the generation branch ----------------------
# Predicate-level: these exercise the routing decision itself, not the endpoint
# around it. The endpoint is covered in section 0.

@pytest.mark.parametrize("message", DOCUMENT_REQUESTS)
def test_document_requests_enter_the_generation_branch(chat_api, message):
    assert _enters_generation_branch(chat_api, message) is True


def test_english_draft_relies_on_the_generation_request_predicate(chat_api):
    """The two operands of the chat condition are not redundant.

    `_classify_top_level_intent` has no fast path for "create a ..." and its
    LLM fallback is abstaining here, so this message routes to generation only
    because `is_generation_request` recognises it. Dropping either operand — or
    "simplifying" the condition to just the classifier — silently breaks
    English drafting requests.
    """
    assert chat_api._classify_top_level_intent(EN_LEASE_DRAFT, "it") == "rag"
    assert chat_api.is_generation_request(EN_LEASE_DRAFT) is True


# --- 2. A genuine calculation is NOT diverted into generation --------------

def test_genuine_calculation_does_not_enter_the_generation_branch(chat_api):
    assert _enters_generation_branch(chat_api, GENUINE_CALCULATION) is False


def test_genuine_calculation_still_auto_routes_to_the_calculator(monkeypatch):
    """The calculation path is untouched: same gate, same threshold."""
    candidate = {
        "calculator_id": "legal_it.registration_tax_leases",
        "score": 3,
        "required_inputs": [{"name": "annual_rent", "type": "decimal"}],
    }
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _Response({"status": "matched", "candidates": [candidate]})

    monkeypatch.setattr(calculation.requests, "post", fake_post)

    update = calculation.calculation_gate({"query": GENUINE_CALCULATION})

    assert update == {"calc_route": "calculate", "calculation_match": candidate}
    assert calls[0]["url"].endswith("/match")


# --- 3. Retrieval-only mode bypasses the gate ------------------------------

def test_retrieval_only_gate_never_consults_the_platform(monkeypatch):
    """`skip_calculation` short-circuits the gate node itself.

    No /match call at all: not merely "matched but ignored". The platform is
    the component that would name a calculator, so not asking it is what makes
    the bypass total — and it saves the round trip on every generation turn.
    """
    def fake_post(url, **kwargs):  # pragma: no cover - must never run
        pytest.fail(f"retrieval-only mode called the platform: {url}")

    monkeypatch.setattr(calculation.requests, "post", fake_post)

    assert calculation.calculation_gate(
        {"query": IT_LEASE_DRAFT, "skip_calculation": True}
    ) == {"calc_route": "normal"}


@pytest.mark.parametrize("message", DOCUMENT_REQUESTS)
def test_generation_retrieval_runs_in_retrieval_only_mode(chat_api, monkeypatch, message):
    """_run_generation_sync must ask the graph for retrieval, not for an answer,
    and must come back with the sources and citations the draft needs."""
    captured = {}

    def fake_rag_run(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return {"raw_result": _retrieval_rows()}

    def fake_generate_document(message_, doc_type, lang, citations, *args, **kwargs):
        captured["citations"] = citations
        return {"draft": "BOZZA", "case_details": {}, "doc_type": doc_type}

    monkeypatch.setattr(chat_api, "rag_run", fake_rag_run)
    monkeypatch.setattr(chat_api, "generate_document", fake_generate_document)

    result = chat_api._run_generation_sync(message, "it", "rental_standard")

    assert captured["skip_calculation"] is True
    assert captured["query"] == message
    # The point of the bypass: retrieval actually produced citable material,
    # and it reached both the reply's source list and the drafting call.
    assert result["sources"] == ["Codice Civile"]
    assert [c["document_name"] for c in captured["citations"]] == ["Codice Civile"]


def test_generation_citations_are_extracted_from_rows_not_from_state(chat_api, monkeypatch):
    """Regression: `_extract_citations` takes result ROWS, not the graph state.

    Passing the state made this raise on its first key, and the handler that
    caught it reset `sources` to [] — so every uncached generation silently lost
    both its sources and its citations, whatever the calculation gate did.
    """
    warnings = []
    monkeypatch.setattr(
        chat_api.logger, "warning", lambda msg, *a, **k: warnings.append(msg % a)
    )
    monkeypatch.setattr(
        chat_api, "rag_run", lambda query, **kwargs: {"raw_result": _retrieval_rows()}
    )
    monkeypatch.setattr(
        chat_api,
        "generate_document",
        lambda *args, **kwargs: {
            "draft": "BOZZA",
            "case_details": {},
            "doc_type": "rental_standard",
        },
    )

    result = chat_api._run_generation_sync(IT_LEASE_DRAFT, "it", "rental_standard")

    assert result["sources"] == ["Codice Civile"]
    assert warnings == []  # retrieval must not have thrown at all


def test_generation_never_returns_a_calculator_result(chat_api, monkeypatch):
    """A drafting turn returns a draft, never a computed figure.

    The stub stands in for a calculation-shaped graph return (what an
    intercepted turn produces: an `answer`, no `raw_result`). Whatever the
    graph says, the generation branch's contract is a draft plus sources —
    there is no path by which a calculator's answer becomes the reply.
    """
    monkeypatch.setattr(
        chat_api,
        "rag_run",
        lambda query, **kwargs: {
            "answer": "Risultato: tax_due: 768.00",
            "calculation_result": {"tax_due": "768.00"},
            "raw_result": [],
        },
    )
    monkeypatch.setattr(
        chat_api,
        "generate_document",
        lambda *args, **kwargs: {
            "draft": "CONTRATTO DI LOCAZIONE",
            "case_details": {},
            "doc_type": "rental_standard",
        },
    )

    result = chat_api._run_generation_sync(IT_LEASE_DRAFT, "it", "rental_standard")

    assert result["draft"] == "CONTRATTO DI LOCAZIONE"
    assert "768.00" not in result["draft"]
    assert "calculation_result" not in result
    assert result["sources"] == []


def test_cached_sections_skip_retrieval_entirely(chat_api, monkeypatch):
    """The pre-existing shortcut still short-circuits before any graph call."""
    monkeypatch.setattr(
        chat_api,
        "rag_run",
        lambda *args, **kwargs: pytest.fail("cached sections must not re-run RAG"),
    )
    monkeypatch.setattr(
        chat_api,
        "generate_document",
        lambda *args, **kwargs: {
            "draft": "BOZZA",
            "case_details": {},
            "doc_type": "rental_standard",
        },
    )

    result = chat_api._run_generation_sync(
        IT_LEASE_DRAFT,
        "it",
        "rental_standard",
        cached_sections=[{"document_title": "Codice Civile"}],
    )

    assert result["sources"] == ["Codice Civile"]


# --- 3b. Document comparison is not a calculator comparison ----------------
# /api/generate routes a message containing a comparison verb to
# _run_comparison_sync, which asks the RAG graph to compare two UPLOADED
# DOCUMENTS and returns its answer as the draft, with its citations as the
# sources. One of the platform's calculators is itself a comparator
# ("confronto gas e luce") and its vocabulary starts with the same verb,
# so a document comparison can auto-route into a
# calculator: "Confronta 'offerta-gas-enel.pdf' e 'offerta-gas-eni.pdf'"
# scores 3 against business.confronto_gas_luce. That calculator takes an
# object_list, so the intercepted turn returns its candidate-collection prompt
# ("Descrivimi la prima offerta...") as the comparison draft, with no
# citations at all.
#
# Comparing two offers numerically IS a calculation; comparing two documents'
# text is not. The verb cannot separate them, so the endpoint's own routing
# decides, and this call has to hold it.
DOCUMENT_COMPARISON = "Confronta 'offerta-gas-enel.pdf' e 'offerta-gas-eni.pdf'"


def test_document_comparison_runs_in_retrieval_only_mode(chat_api, monkeypatch):
    captured = {}

    def fake_rag_run(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return {
            "answer": "Le due offerte differiscono nel corrispettivo energia.",
            "citations": [{"document_name": "offerta-gas-enel.pdf", "sections": []}],
            "is_comparison": True,
        }

    monkeypatch.setattr(chat_api, "rag_run", fake_rag_run)

    result = chat_api._run_comparison_sync(DOCUMENT_COMPARISON, "it")

    assert captured["skip_calculation"] is True
    # The comparison flow still runs and still produces its own answer and
    # citations — the bypass must not cost the feature anything.
    assert result["answer"].startswith("Le due offerte differiscono")
    assert result["citations"] == [
        {"document_name": "offerta-gas-enel.pdf", "sections": []}
    ]
    assert result["is_comparison"] is True


def test_document_comparison_gate_never_consults_the_platform(monkeypatch):
    def fake_post(url, **kwargs):  # pragma: no cover - must never run
        pytest.fail(f"retrieval-only comparison called the platform: {url}")

    monkeypatch.setattr(calculation.requests, "post", fake_post)

    assert calculation.calculation_gate(
        {"query": DOCUMENT_COMPARISON, "skip_calculation": True}
    ) == {"calc_route": "normal"}


def test_unguarded_comparison_would_be_intercepted(monkeypatch):
    """Why the flag is needed, not just that it is set.

    With the real /match reply this message reaches the gate's auto-route
    threshold, so an unguarded comparison turn is handed to a comparator
    calculator and never reaches comparison_retrieval.
    """
    candidate = {
        "calculator_id": "business.confronto_gas_luce",
        "score": 3,
        "required_inputs": [{"name": "offerte", "type": "object_list"}],
    }
    monkeypatch.setattr(
        calculation.requests,
        "post",
        lambda url, **kwargs: _Response(
            {"status": "matched", "candidates": [candidate]}
        ),
    )

    assert calculation.calculation_gate({"query": DOCUMENT_COMPARISON}) == {
        "calc_route": "calculate",
        "calculation_match": candidate,
    }


# --- 4. Entry routing: skip_calculation is narrow -------------------------

def test_entry_router_sends_retrieval_only_turns_straight_to_retrieval():
    from src.rag.main import route_entry

    assert route_entry({"query": IT_LEASE_DRAFT, "skip_calculation": True}) == (
        "decompose_query"
    )


def test_entry_router_preserves_existing_precedence():
    from src.rag.main import route_entry

    pending = {"calculator_id": "legal_it.registration_tax_leases", "round": 1}
    # Ordinary chat turns are unaffected by the new flag's existence.
    assert route_entry({"query": GENUINE_CALCULATION}) == "calculation_gate"
    assert route_entry({"pending_calculation": pending}) == "calculation_node"
    assert route_entry({"awaiting_clarification": True}) == "rerank_from_clarification"
    assert route_entry({"skip_calculation": False}) == "calculation_gate"


def test_retrieval_only_does_not_consume_an_open_calculation():
    """A generation turn must not be read as the answer to a pending slot.

    Without this, "Redigi un contratto ... 400 euro al mese" arriving while a
    calculation waits for `annual_rent` would be mined for that value: the user
    asked for a document and would get a tax figure computed from a number they
    never offered as an answer. The pending calculation is left untouched for
    whichever later turn genuinely answers it.
    """
    from src.rag.main import route_entry

    assert route_entry(
        {
            "pending_calculation": {"calculator_id": "legal_it.registration_tax_leases"},
            "skip_calculation": True,
        }
    ) == "decompose_query"


def test_retrieval_only_still_allows_the_clarification_rerank():
    """The flag suppresses calculation, not the legal-RAG clarification path."""
    from src.rag.main import route_entry

    assert route_entry(
        {"awaiting_clarification": True, "skip_calculation": True}
    ) == "rerank_from_clarification"


def test_entry_router_is_wired_to_every_branch_it_can_return():
    """A branch the router can name but the graph has not mapped is a crash at
    runtime, on whichever turn first reaches it."""
    from src.rag.main import build_graph

    graph = build_graph(compile_graph=False)
    branches = graph.branches["__start__"]
    mapped = set(next(iter(branches.values())).ends)

    assert {
        "calculation_node",
        "rerank_from_clarification",
        "calculation_gate",
        "decompose_query",
    } <= mapped


def test_run_defaults_to_calculation_enabled(monkeypatch):
    """Only explicit retrieval-only callers get the bypass; chat keeps the gate."""
    import src.rag.main as rag_main

    captured = {}
    monkeypatch.setattr(
        rag_main,
        "_get_compiled_graph",
        lambda: SimpleNamespace(invoke=lambda state: captured.update(state) or state),
    )

    rag_main.run("Calcola l'IRPEF su 42000 euro")
    assert captured["skip_calculation"] is False

    captured.clear()
    rag_main.run("Redigi un contratto", skip_calculation=True)
    assert captured["skip_calculation"] is True
