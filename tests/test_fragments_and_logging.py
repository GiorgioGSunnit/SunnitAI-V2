"""Cut comma openings reaching the answer, from their exact strings, and the trace reaching the logs of every path."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing, article_text, fact_signals, text_tidy, verbose_logger  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMMA_1062 = ("La destinazione del padre di famiglia ha luogo quando consta, mediante qualunque genere di prova, che due fondi, "
              "attualmente divisi, sono stati posseduti dallo stesso proprietario.")
COPIED_TITLE = "La destinazione del padre di famiglia ha luogo quando consta mediante"


def section(**fields):
    return {"plain_text": COMMA_1062, **fields}


def test_a_title_copying_the_comma_is_no_rubrica():
    assert article_text.heading(section(title=COPIED_TITLE)) == ""


@pytest.mark.parametrize("title, wanted", [("Destinazione del padre di famiglia", "Destinazione del padre di famiglia"),
                                           ("Servitù apparenti", "Servitù apparenti")])
def test_a_real_title_stays(title, wanted):
    assert article_text.heading(section(title=title)) == wanted


def test_a_cut_title_starting_mid_text_or_the_same_length_as_the_opening():
    assert article_text.heading(section(title="ha luogo quando consta mediante qualunque genere di prova")) == ""
    assert article_text.heading({"plain_text": "Il giudice decide la causa secondo diritto.", "title": "Il giudice decide la causa"}) == ""


def test_the_rubrica_always_wins():
    assert article_text.heading(section(titolo_articolo="Destinazione del padre di famiglia.", title=COPIED_TITLE)) \
        == "Destinazione del padre di famiglia."
    assert article_text.heading(None) == ""


def test_every_reader_of_a_rubrica_uses_it():
    row = {"_article": ("1062", "c.c."), "s": section(title=COPIED_TITLE)}
    assert ptw._headings([row]) == {} and dg._rows_heading([row]) == ""
    for path in ("plan_then_write", "defensive_generation", "anchored_writing", "fact_signals", "answer_invariants"):
        source = (ROOT / "src" / "rag" / f"{path}.py").read_text(encoding="utf-8")
        assert 'or s.get("title")' not in source and '.get("titolo_articolo") or ((row or {})' not in source


LONG = ("Il proprietario del fondo servente non può compiere innovazioni che rendano meno agevole l'esercizio della servitù, "
        "né può mutare lo stato dei luoghi. ") * 8


def test_the_fixed_cuts_end_on_a_clause():
    assert article_text.clause_cut(LONG, 300).endswith("luoghi. […]") or article_text.clause_cut(LONG, 300).endswith("servitù […]")
    catalogue = anchored_writing.Catalogue()
    pid = catalogue.add(("article", ("1067", "c.c."), 0), "article", "art. 1067 c.c.", LONG)
    assert catalogue.passages[pid]["text"].endswith("[…]") and "rendano […]" not in catalogue.passages[pid]["text"]


def test_no_fixed_length_cut_is_left_on_an_article_s_text():
    for path, pattern in (("plan_then_write", "text[:1200]"), ("anchored_writing", "[:PASSAGE_CHARS]"), ("defensive_generation", "[:600]")):
        assert pattern not in (ROOT / "src" / "rag" / f"{path}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("sentence", ["La servitù ai sensi dell'art. 1062 c.c., co. 1, ha luogo quando consta mediante. o all'art. 1064 c.c. si applica.",
                                      "Il proprietario non può compiere innovazioni che rendano. vieta al proprietario di agire."])
def test_the_exact_fragments_mid_sentence(sentence):
    assert text_tidy.unreadable(sentence) == "cut clause inside a sentence"
    out, _ = text_tidy.tidy(f"La strada è antica. {sentence} Il fondo è intercluso.")
    assert "mediante" not in out and "rendano" not in out and "intercluso" in out


@pytest.mark.parametrize("sentence", ["Ai sensi dell'art. 1062 c.c. il giudice decide.", "Si veda Cass. civ., sez. II, sent. n. 12 del 2020.",
                                      "Il termine (lett. a) decorre.", "La sentenza del Trib. di Roma è chiara.", "Il cod. proc. civ. lo prevede."])
def test_abbreviations_are_not_a_cut_clause(sentence):
    assert text_tidy.unreadable(sentence) == ""


def test_a_new_sentence_is_not_a_fragment():
    assert text_tidy.unreadable("Il fondo è diviso. Il giudice decide.") == ""


# --- the logs --------------------------------------------------------------------------------------------------------

def _dryrun():
    spec = importlib.util.spec_from_file_location("dryrun_strategy", ROOT / "scripts" / "dryrun_strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dry_run_captures_and_still_prints_when_verbose(monkeypatch, capsys):
    monkeypatch.setattr(verbose_logger, "VERBOSE", True)
    events = []
    fake = SimpleNamespace(vlog=None)
    for module, name in ((fact_signals, "vlog"), (ptw.dg, "vlog")):
        monkeypatch.setattr(module, name, getattr(module, name))
    from src.rag import ai_chat
    monkeypatch.setattr(ai_chat, "vlog", ai_chat.vlog)
    _dryrun().capture_vlog(fake, events)
    fake.vlog("case_analysis_plan_final", {"issues": [{"titolo": "Acquisto"}]})
    fact_signals.vlog("fact_rules", {"rules": ["FIRED regola"]})
    ai_chat.vlog("llm_call", {"call": "x"})
    out = capsys.readouterr().out
    assert "[VERBOSE:case_analysis_plan_final]" in out and "Acquisto" in out and "FIRED regola" in out and "llm_call" in out
    assert [e[0] for e in events] == ["case_analysis_plan_final", "fact_rules", "llm_call"]


def test_the_dry_run_prints_no_case_text_unless_verbose(monkeypatch, capsys):
    monkeypatch.setattr(verbose_logger, "VERBOSE", False)
    monkeypatch.setattr(fact_signals, "vlog", fact_signals.vlog)
    from src.rag import ai_chat
    monkeypatch.setattr(ai_chat, "vlog", ai_chat.vlog)
    events, fake = [], SimpleNamespace(vlog=None)
    _dryrun().capture_vlog(fake, events)
    fake.vlog("fact_rules", {"rules": ["testo del caso"]})
    assert capsys.readouterr().out == "" and len(events) == 1


def test_the_case_logs_carry_the_trace_whatever_verbose_says(monkeypatch, capsys):
    monkeypatch.setattr(verbose_logger, "VERBOSE", False)
    for event in ("fact_rules", "case_analysis_plan_final", "case_analysis_plan_call"):
        verbose_logger.trace(event, {"x": 1})
    verbose_logger.trace("case_analysis_draft_llm", {"x": 1})
    out = capsys.readouterr().out
    assert out.count("[VERBOSE:") == 3 and "draft_llm" not in out
    assert "verbose_logger.trace(event, data, duration_ms)" in (ROOT / "scripts" / "diagnose_long.py").read_text(encoding="utf-8")


def test_the_rules_print_whole():
    assert {"rules", "issues"} <= verbose_logger.FULL_KEYS
