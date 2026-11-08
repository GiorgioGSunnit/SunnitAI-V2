"""v46 speed: PLAN_PATCH_REDO, WRITER_LOOP_GUARD, ANCHOR_REWRITE_CAP (off by default), GRADER_VOTE_WORKERS; and the
article lists every citation reader now reads (flag-free). c1's and p1's own strings where they matter."""
import json
import os
import re

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing as aw  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

FLAGS = ("PLAN_PATCH_REDO", "WRITER_LOOP_GUARD", "ANCHOR_REWRITE_CAP", "REVIEW_FIXES_ENABLED", "WRITER_RULES_ENABLED",
         "PLAN_RULE_INCLUDE", "GRADER_VOTE_WORKERS", "ANCHORED_WRITING_ENABLED")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    return seen


def row(number, grade=3):
    return {"d": {"id": "doc-c.c.", "name": "c.c."}, "_article": (number, "c.c."), "_grade": grade,
            "s": {"name": f"{number}.0.0", "plain_text": f"Testo dell'articolo {number}.", "titolo_articolo": "R"}}


# --- 1. PLAN_PATCH_REDO ----------------------------------------------------------------------------------------------

# c1: the plan answered one explicit question of two; the review's only problem was the missing answer.
REQUEST = "Il paziente ha diritto al risarcimento del danno? Contro chi può agire, a quale titolo e con quali oneri probatori?"
FIRST = {"parte_assistita": "paziente", "questione_centrale": "Risarcimento del danno da infezione",
         "risposte": [{"quesito": "Il paziente ha diritto al risarcimento del danno?", "risposta": "Sì, per l'art. 1218 c.c."}],
         "questioni": [{"titolo": "Risarcimento del danno e titolo dell'azione contro la struttura",
                        "argomento": "Il paziente può agire contro la struttura a titolo contrattuale, con l'onere "
                                     "probatorio dell'art. 1218 c.c.; ha diritto al risarcimento del danno.",
                        "norme": ["art. 1218 c.c."]}]}


def _make_plan(monkeypatch, first, patch_reply):
    monkeypatch.setenv("REVIEW_FIXES_ENABLED", "true")
    plans = []
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: plans.append(messages) or first)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": "art. 1218 c.c.", "pertinente": True, "contraddice": False}])
    patches = []

    def chat_json(messages, **kwargs):
        if messages[0].content == ptw._PATCH_SYSTEM:
            patches.append(kwargs)
            return patch_reply
        return []
    monkeypatch.setattr(dg, "_chat_json", chat_json)
    plan = ptw.make_plan("Il paziente contrae un'infezione in ospedale.", REQUEST, {}, "[art. 1218 c.c.]", "", "",
                         [row("1218")], {})
    return plan, plans, patches


def test_only_missing_answers_are_patched_without_a_full_redo(monkeypatch, events):
    monkeypatch.setenv("PLAN_PATCH_REDO", "true")
    answer = [{"quesito": "Contro chi può agire, a quale titolo e con quali oneri probatori?",
               "risposta": "Contro la struttura, a titolo contrattuale (art. 1218 c.c.): prova il contratto e il danno."}]
    plan, plans, patches = _make_plan(monkeypatch, FIRST, answer)
    assert len(plans) == 1 and len(patches) == 1 and patches[0]["max_tokens"] == 600
    assert any("Contro la struttura" in a["risposta"] for a in plan["risposte"])
    log = next(d for n, d in events if n == "case_analysis_plan_patch")
    assert log["complete"] and log["then"] == "kept"


def test_a_patch_that_does_not_answer_falls_back_to_the_full_redo(monkeypatch, events):
    monkeypatch.setenv("PLAN_PATCH_REDO", "true")
    plan, plans, patches = _make_plan(monkeypatch, FIRST, [])
    assert len(patches) == 1 and len(plans) == 2
    assert next(d for n, d in events if n == "case_analysis_plan_patch")["then"] == "full redo"


def test_a_structural_problem_gets_the_full_redo(monkeypatch, events):
    monkeypatch.setenv("PLAN_PATCH_REDO", "true")
    # both questions answered, but the second treated by no issue: structural
    first = {**FIRST, "risposte": FIRST["risposte"] + [{"quesito": "Contro chi può agire?", "risposta": "Contro "
             "la struttura, a quale titolo contrattuale e con quali oneri probatori: quelli dell'art. 1218 c.c."}],
             "questioni": [{**FIRST["questioni"][0], "titolo": "Danno", "argomento": "Il danno da infezione."}]}
    plan, plans, patches = _make_plan(monkeypatch, first, [])
    assert patches == [] and len(plans) == 2


def test_without_the_flag_the_full_redo_as_before(monkeypatch, events):
    plan, plans, patches = _make_plan(monkeypatch, FIRST, [])
    assert patches == [] and len(plans) == 2


# --- 2. WRITER_LOOP_GUARD --------------------------------------------------------------------------------------------

P1_LOOP_SENTENCE = ("L'art. 52 c.p., comma 2, riconosce inoltre che l'uso di un'arma legittimamente detenuta è considerato "
                    "proporzionato quando si difende la propria o altrui incolumità, o beni propri o altrui, in assenza di "
                    "desistenza e in presenza di un pericolo d'aggressione.")
P1_HEAD = ("L'art. 575 c.p. disciplina l'omicidio doloso, che qui non ricorre. L'art. 589 c.p. punisce l'omicidio colposo. ")
P1_LOOPED = P1_HEAD + " ".join([P1_LOOP_SENTENCE] * 4) + " L'art. 52 c.p., comma 2, riconosce inoltre che l'uso di un'arma"
# p1's "Applicazione al caso" (v46): one formula per article, other words in between -- no loop
FORMULA = " ".join(f"L'art. {n} c.p., in base al passo [A{k}], disciplina {what}. Questa norma, applicabile al caso di "
                   f"Tizio, potrebbe escludere la responsabilità penale." for k, (n, what) in
                   enumerate([("52", "la difesa legittima"), ("55", "l'eccesso colposo"), ("42", "la responsabilità per "
                              "dolo"), ("43", "l'elemento psicologico del reato")], 1))


def test_a_looping_reply_is_cut_where_it_first_repeats(monkeypatch):
    monkeypatch.setenv("WRITER_LOOP_GUARD", "true")
    text, loop = ptw.unloop(P1_LOOPED)
    assert text == P1_HEAD + P1_LOOP_SENTENCE
    assert loop["count"] >= 3 and loop["ngram"].startswith("l art 52 c p comma 2")
    assert loop["chars_before"] == len(P1_LOOPED) and loop["chars_after"] == len(text)


def test_a_formula_repeated_with_other_words_between_is_no_loop(monkeypatch):
    monkeypatch.setenv("WRITER_LOOP_GUARD", "true")
    assert re.findall(r"Questa norma, applicabile al caso di Tizio, potrebbe", FORMULA).__len__() == 4
    assert ptw.find_loop(FORMULA) is None and ptw.unloop(FORMULA) == (FORMULA, None)


def test_the_guard_is_off_by_default():
    assert ptw.unloop(P1_LOOPED) == (P1_LOOPED, None)


def test_a_looped_section_is_not_written_again(monkeypatch):
    monkeypatch.setenv("WRITER_LOOP_GUARD", "true")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("a looped section rewritten"))
    cut = P1_HEAD + P1_LOOP_SENTENCE + " L'art. 52 c.p., comma 2, riconosce inoltre che l'uso di un'arma"
    out, log = ptw._complete_sections([("Norme e giurisprudenza", cut)], [("Norme e giurisprudenza", "i", "m", 100)],
                                      "s", looped={"Norme e giurisprudenza"})
    assert out[0][1] == P1_HEAD + P1_LOOP_SENTENCE
    assert log == [{"section": "Norme e giurisprudenza", "kind": "cut", "how": "looped: trimmed, not rewritten",
                    "sentence": cut[-160:]}]


def test_a_rewrite_that_loops_is_cut_and_logged(monkeypatch):
    monkeypatch.setenv("WRITER_LOOP_GUARD", "true")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: P1_LOOPED)
    cut = P1_HEAD + "L'art. 52 c.p. esclude"
    loops = []
    out, log = ptw._complete_sections([("Norme e giurisprudenza", cut)], [("Norme e giurisprudenza", "i", "m", 100)],
                                      "s", loops=loops)
    assert out[0][1] == P1_HEAD + P1_LOOP_SENTENCE and log[0]["how"] == "rewritten with more tokens"
    assert [(e["section"], e["call"]) for e in loops] == [("Norme e giurisprudenza", "rewrite (cut)")]


# --- 3. ANCHOR_REWRITE_CAP -------------------------------------------------------------------------------------------

class _Cat:
    passages = {f"A{k}": {"label": f"art. {k} c.c.", "text": f"testo {k}"} for k in range(1, 31)}


def _rewriter(calls):
    def chat_json(messages, max_tokens=None, **kwargs):
        lines = re.findall(r"^(\d+)\. (.+?) \[passi", messages[-1].content, re.MULTILINE)
        calls.append((max_tokens, [s for _, s in lines], re.findall(r"^\[(A\d+)\]", messages[-1].content, re.MULTILINE)))
        return [{"i": int(n), "tipo": "legge", "frase": f"nuova {s}"} for n, s in lines]
    return chat_json


def test_a_large_rewrite_is_split_under_the_cap(monkeypatch, events):
    monkeypatch.setenv("ANCHOR_REWRITE_CAP", "true")
    monkeypatch.setattr(aw, "vlog", lambda name, data=None, **k: events.append((name, data)))
    calls = []
    monkeypatch.setattr(dg, "_chat_json", _rewriter(calls))
    items = [(f"frase {k}", [f"A{k}"], "law") for k in range(1, 21)]
    out = aw._anchor_rewrite(_Cat(), items)
    assert sorted(len(s) for _, s, _ in calls) == [2, 9, 9] and all(t <= 2500 for t, _, _ in calls)
    assert {k: v["frase"] for k, v in out.items()} == {k - 1: f"nuova frase {k}" for k in range(1, 21)}
    for _, sentences, ids in calls:        # each call carries only its own sentences' passages
        assert ids == [f"A{s.split()[-1]}" for s in sentences]
    assert next(d for n, d in events if n == "anchored_writing_rewrite_split")["calls"] == 3


def test_without_the_cap_one_call_as_before(monkeypatch):
    calls = []
    monkeypatch.setattr(dg, "_chat_json", _rewriter(calls))
    aw._anchor_rewrite(_Cat(), [(f"frase {k}", [f"A{k}"], "law") for k in range(1, 21)])
    assert [(t, len(s)) for t, s, _ in calls] == [(300 + 230 * 20, 20)]


# --- GRADER_VOTE_WORKERS ---------------------------------------------------------------------------------------------

def test_the_vote_s_concurrency_is_a_setting(monkeypatch):
    assert dg._grader_vote_workers() == 8
    monkeypatch.setenv("GRADER_VOTE_WORKERS", "24")
    assert dg._grader_vote_workers() == 24
    monkeypatch.setenv("GRADER_VOTE_WORKERS", "many")
    assert dg._grader_vote_workers() == 8


# --- article lists (flag-free) ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, refs", [
    ("artt. 55 e 52 c.p.", [("55", "c.p."), ("52", "c.p.")]),
    ("artt. 55, 52 e 59 c.p.", [("55", "c.p."), ("52", "c.p."), ("59", "c.p.")]),
    ("degli artt. 55 c.p. e 52 c.p.", [("55", "c.p."), ("52", "c.p.")]),
    ("artt. 1176, comma 2, e 1218 c.c.", [("1176", "c.c."), ("1218", "c.c.")]),
    ("gli articoli 2043, 2048 e 2059 del codice civile", [("2043", "c.c."), ("2048", "c.c."), ("2059", "c.c.")]),
    ("artt. 2947 c.c. e 157 c.p.", [("2947", "c.c."), ("157", "c.p.")]),
])
def test_every_article_of_a_list_counts(text, refs):
    assert dg._refs_in(text) == refs
    assert [r for _, _, _, rs, _ in ptw.read_citations(text, [], None) for r in rs] == refs


@pytest.mark.parametrize("text, refs", [
    ("art. 52 c.p.", [("52", "c.p.")]),
    ("art. 52, comma 2, c.p.", [("52", "c.p.")]),
    ("ai sensi dell'art. 2043 c.c. e dell'art. 2059 c.c.", [("2043", "c.c."), ("2059", "c.c.")]),
    ("art. 2043 c.c., 5 anni dopo", [("2043", "c.c.")]),
    ("art. 614-bis c.p.c.", [("614-bis", "c.p.c.")]),
    ("l'art. 1218 c.c. (Responsabilità del debitore)", [("1218", "c.c.")]),
])
def test_single_citations_read_as_before(monkeypatch, text, refs):
    assert dg._refs_in(text) == refs
    new = ptw.read_citations(text, [], None)
    monkeypatch.setattr(dg, "_article_lists", lambda t: [])        # the readers without lists: as before
    assert dg._refs_in(text) == refs and ptw.read_citations(text, [], None) == new


def test_a_list_without_any_code_is_left_to_the_nearest_code():
    assert dg._article_lists("artt. 55 e 52") == []
    assert [r for _, _, _, rs, _ in ptw.read_citations("Secondo il c.p., artt. 55 e 52.", [], None) for r in rs] \
        == [("55", "c.p."), ("52", "c.p.")]


def test_a_list_goes_from_a_sentence_only_when_all_its_articles_go():
    sentence = "La difesa è legittima (artt. 55 c.p. e 52 c.p.) nel caso di Tizio."
    assert aw.drop_citations(sentence, [("55", "c.p.")], [], None) == sentence
    assert aw.drop_citations(sentence, [("55", "c.p."), ("52", "c.p.")], [], None) == "La difesa è legittima nel caso di Tizio."
