"""The overflow fixes of 23 Oct. Prod logs, 20-23 Oct: the applicability check failed on 72 of
about 260 strategy answers (every candidate in one call: 87-89 articles, 19,580 answer tokens
against a 20,000-token context), and 11 strategy drafts ended in "Si è verificato un errore
nella generazione della strategia" (about 48 protected articles still over budget after the
last trim step, and a retry sized on the server's "at least N" lower bound).

  - the draft prompt is never sent over budget (_fit_parere_prompt's last resort);
  - _call_chat's retry sizes the answer on the real prompt and reads every wording;
  - the applicability check is capped, batched, and asks for the long verdict only on drops.
"""

import json
import os
import re
import threading

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from langchain_core.messages import HumanMessage  # noqa: E402

import src.rag.ai_chat as ai_chat  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
from src.rag.nodes.cypher import CROSSREF_SOURCE  # noqa: E402


# --- _call_chat's retry ----------------------------------------------------------------------

class _Model:
    """A chat model whose first call raises `error`; records every max_tokens asked for."""

    def __init__(self, error, calls, kw=None):
        self.error, self.calls, self.kw = error, calls, kw or {}

    def bind(self, **kw):
        return _Model(self.error, self.calls, kw)

    def invoke(self, messages):
        self.calls.append(self.kw.get("max_tokens"))
        if len(self.calls) == 1:
            raise RuntimeError(self.error)
        return type("Reply", (), {"content": " ok "})()


def _retry(monkeypatch, error, max_tokens, prompt_chars=0):
    calls = []
    monkeypatch.setattr(ai_chat, "chat_model", _Model(error, calls))
    messages = [HumanMessage(content="x" * prompt_chars)] if prompt_chars else []
    return ai_chat._call_chat(messages, max_tokens=max_tokens), calls


def test_retry_sizes_on_its_own_estimate_when_the_server_gives_a_lower_bound(monkeypatch):
    # Prod, 23 Oct: "at least 861" for a 19,140-token request; the retry asked 18,839 and failed.
    error = ("This model's maximum context length is 20000 tokens. However, you requested 19140 output tokens "
             "and your prompt contains at least 861 input tokens")
    reply, calls = _retry(monkeypatch, error, 19140, prompt_chars=40000)       # ~16,000 tokens
    assert reply == "ok" and calls == [19140, 20000 - 16000 - 300]


def test_retry_reads_the_characters_wording(monkeypatch):
    error = ("This model's maximum context length is 20000 tokens. However, you requested 19580 output tokens "
             "and your prompt contains 30000 characters (more than 53760 characters, which is the upper bound "
             "for 420 input tokens).")
    reply, calls = _retry(monkeypatch, error, 19580)
    assert reply == "ok" and calls == [19580, 20000 - 12000 - 300]


def test_retry_reads_an_answer_length_over_the_context(monkeypatch):
    error = "max_completion_tokens=19580 cannot be greater than max_model_len=20000. Please request fewer tokens."
    reply, calls = _retry(monkeypatch, error, 19580, prompt_chars=10000)        # ~4,000 tokens
    assert reply == "ok" and calls == [19580, 20000 - 4000 - 300]


def test_retry_never_asks_for_more_than_the_caller(monkeypatch):
    error = ("This model's maximum context length is 20000 tokens. However, your request has too many tokens: "
             "the prompt contains at least 2000 input tokens")
    reply, calls = _retry(monkeypatch, error, 1500)
    assert calls == [1500, 1500]


def test_retry_gives_up_when_the_prompt_itself_is_too_big(monkeypatch):
    error = ("This model's maximum context length is 20000 tokens. However, you requested 19580 output tokens "
             "and your prompt contains 65396 characters (more than 53760 characters, which is the upper bound "
             "for 420 input tokens).")
    with pytest.raises(RuntimeError):
        _retry(monkeypatch, error, 19580)            # ~26,000 tokens of prompt: no answer can fit


# --- the draft prompt's last resort -------------------------------------------------------------

SYSTEM = "Istruzioni per il parere. " * 200


def _article(number, text_char, **tags):
    return {"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"}, "_article": (number, "c.p."),
            "s": {"name": f"{number}.0.0", "plain_text": (text_char * 1600)}, **tags}


def test_protected_articles_are_dropped_least_protected_first_until_it_fits(monkeypatch):
    logged = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    rows = ([_article("584", "N")]                                                    # named in the case
            + [_article(str(40 + i), "H", _heading_match=True) for i in range(3)]
            + [_article(str(500 + i), "C", _source=CROSSREF_SOURCE) for i in range(10)]
            + [_article(str(700 + i), "M", _multi_signal=True) for i in range(40)])   # ~54 protected rows
    case = "Tizio colpisce Caio, che muore; il PM contesta l'art. 584 c.p."
    articles, _, human = dg._fit_parere_prompt(SYSTEM, case, {"questioni": []}, rows, [], budget=9100)
    assert dg._estimate_tokens(SYSTEM) + dg._estimate_tokens(human) <= 9100
    assert "N" * 500 in articles and "H" * 500 in articles                            # named, headings kept
    kept_numbers = set(re.findall(r"\[art\. (\d+)", articles))
    cross_refs, multi_signal = {str(500 + i) for i in range(10)}, {str(700 + i) for i in range(40)}
    assert "584" in kept_numbers and {"40", "41", "42"} <= kept_numbers
    assert len(kept_numbers & multi_signal) < 40                    # some were dropped ...
    if kept_numbers & multi_signal:                                 # ... and before any cross-reference
        assert cross_refs <= kept_numbers
    sizes = [data for event, data in logged if event == "case_analysis_prompt_tokens"][-1]
    assert sizes["dropped_for_size"] > 0 and sizes["step"] == len(dg._PROMPT_TRIM_STEPS)


def test_a_prompt_that_fits_at_the_last_step_drops_nothing(monkeypatch):
    logged = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    rows = [_article("584", "N")] + [_article(str(700 + i), "M", _multi_signal=True) for i in range(3)]
    dg._fit_parere_prompt(SYSTEM, "Caso: art. 584 c.p.", {"questioni": []}, rows, [], budget=9100)
    assert logged[-1][1]["dropped_for_size"] == 0


# --- the applicability check: capped, batched, short verdicts ------------------------------------

FACTS = "Sempronio, agendo da solo, schiaffeggia Caio, che cade e muore. Il PM contesta l'art. 584 c.p."
INCOMPATIBLE = {"116 c.p.", "605 c.p.", "620 c.p."}


def _row(name, text="Testo dell'articolo."):
    return {"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
            "s": {"id": name, "name": f"{name}.0.0", "plain_text": text}, "_article": (name, "c.p.")}


@pytest.fixture
def checker(monkeypatch):
    """A fake model answering one verdict per article it is shown; INCOMPATIBLE ones dropped.
    Records each call's labels and max_tokens; `fail_on` makes the batch holding it fail."""
    seen, lock, state = [], threading.Lock(), {"fail_on": None, "system": ""}

    def fake_call_chat(messages, max_tokens=None, **kwargs):
        labels = re.findall(r"^\[(\d+(?:-\w+)? c\.p\.)\]", messages[-1].content, re.MULTILINE)
        with lock:
            seen.append({"labels": labels, "max_tokens": max_tokens})
            state["system"] = messages[0].content
        if state["fail_on"] in labels:
            raise RuntimeError("This model's maximum context length is 20000 tokens.")
        return json.dumps([
            {"article": label, "verdict": "incompatibile", "requisito": "più concorrenti nel reato",
             "fatto": "Sempronio ha agito da solo", "reason": "Richiede più agenti."}
            if label in INCOMPATIBLE else {"article": label, "verdict": "applicabile"} for label in labels])

    logged = []
    monkeypatch.setattr(dg, "_call_chat", fake_call_chat)
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    return seen, state, logged


def test_check_is_capped_and_batched(checker):
    seen, _, logged = checker
    rows = [_row("116")] + [_row(str(600 + i)) for i in range(49)]            # 50 candidates
    kept = [r["_article"][0] for r in dg._filter_applicable_articles(FACTS, "", rows)]
    sent = [label for call in seen for label in call["labels"]]
    assert len(seen) == 3 and all(len(call["labels"]) == 12 for call in seen)  # 36 checked, in 3 batches
    assert len(sent) == dg._APPLICABILITY_MAX_CHECKED and "116 c.p." in sent and "648 c.p." not in sent
    assert "116" not in kept and "605" not in kept and "620" not in kept      # dropped where checked
    assert "648" in kept and len(kept) == 50 - 3                                # unchecked ones kept
    batch_log = [data for event, data in logged if event == "case_analysis_applicability" and "batches" in data]
    assert batch_log[-1] == {"checked": 36, "batches": 3, "failed_batches": 0}
    assert logged[-1][1]["unchecked"] == 14


def test_a_failed_batch_keeps_only_its_own_articles(checker):
    seen, state, _ = checker
    state["fail_on"] = "620 c.p."                                               # in the second batch
    rows = [_row("116")] + [_row(str(600 + i)) for i in range(29)]
    kept = [r["_article"][0] for r in dg._filter_applicable_articles(FACTS, "", rows)]
    assert "116" not in kept and "605" not in kept        # first batch still filtered
    assert "620" in kept                                   # its batch failed: kept, as before
    state["fail_on"] = None


def test_short_verdicts_and_answer_length_per_batch(checker):
    seen, state, _ = checker
    dg._filter_applicable_articles(FACTS, "", [_row("116")] + [_row(str(600 + i)) for i in range(4)])
    assert seen[0]["max_tokens"] == 400 + 120 * 5
    assert 'solo per un articolo "incompatibile"' in state["system"]
    assert "requisito" in state["system"] and "alternativa" in state["system"]


def test_every_batch_failing_keeps_everything(checker):
    seen, state, logged = checker
    rows = [_row("116")] + [_row(str(600 + i)) for i in range(5)]
    state["fail_on"] = "116 c.p."
    kept = dg._filter_applicable_articles(FACTS, "", rows)
    assert kept == rows
    assert logged[-1][1]["failed_batches"] == 1 and "error" in logged[-1][1]
    state["fail_on"] = None
