"""v46: the citation ledger (citation_removed, always on); plan cover's insertion and the review's flags; GROUNDED_EXCLUSION
(off by default). p1's own strings where they matter; invented rows otherwise."""
import importlib.util
import os
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import citation_ledger, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

A575 = "Chiunque cagiona la morte di un uomo è punito con la reclusione non inferiore ad anni ventuno."
A55 = ("Quando, nel commettere alcuno dei delitti preveduti dagli articoli 51, 52, 53 e 54, si eccedono colposamente i "
       "limiti stabiliti dalla legge o dall'ordine dell'Autorità ovvero imposti dalla necessità, si applicano le "
       "disposizioni concernenti i delitti colposi, se il fatto è preveduto dalla legge come delitto colposo.")
A52 = "Non è punibile chi ha commesso il fatto, per esservi stato costretto dalla necessità di difendere un diritto proprio od altrui."
A1218 = ("Il debitore che non esegue esattamente la prestazione dovuta è tenuto al risarcimento del danno, se non prova "
         "che l'inadempimento o il ritardo è stato determinato da impossibilità della prestazione derivante da causa a lui "
         "non imputabile.")
# p1, v45: plan cover's reply for 575 c.p., refused for citing 55 c.p.
P1_575 = ("L'art. 575 c.p. (Omicidio) prevede la punizione per chi cagiona la morte di un uomo, ma non si applica nel caso "
          "in cui l'azione rientri nella legittima difesa, come potrebbe essere qui riconosciuta ai sensi dell'art. 55 c.p.")
P1_COVER = ("L'art. 575 c.p. (Omicidio) prevede la pena per chi cagiona la morte di un uomo, ma non distingue tra forma "
            "dolosa e colposa, richiedendo una qualificazione giuridica che tenga conto dell'elemento soggettivo, come nel "
            "caso di una reazione impulsiva e non volontaria.")
P1_FLAG = ("la frase assume implicitamente che l'azione di Tizio sia stata impulsiva e non volontaria, un elemento "
           "soggettivo non stabilito dal caso")


def row(number, code, text, grade=None, heading=""):
    return {"_article": (number, code), **({"_grade": grade} if grade is not None else {}),
            "s": {"plain_text": text, "titolo_articolo": heading or f"Rubrica {number}"}}


ROWS = [row("575", "c.p.", A575, 3, "Omicidio"), row("55", "c.p.", A55, 2, "Eccesso colposo"),
        row("52", "c.p.", A52, 3, "Difesa legittima")]


def plan_of(*refs):
    return {"parte_assistita": "Tizio", "questioni": [
        {"titolo": "Elemento soggettivo e responsabilità di Tizio", "argomento": "Dolo o colpa.", "norme": list(refs),
         "pronunce": [], "obiezione": "", "replica": ""}]}


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    for name in ("WRITER_RULES_ENABLED", "REVIEW_FIXES_ENABLED", "GROUNDED_REVIEW_ENABLED", "GROUNDED_EXCLUSION",
                 "FACT_MARK_INLINE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ptw, "SECTION_TITLES", ptw.SECTION_TITLES[:3] + ("Applicazione al caso",) + ptw.SECTION_TITLES[4:])
    return seen


def ledger(seen):
    return [d for n, d in seen if n == citation_ledger.EVENT]


# --- 1. the citation ledger ------------------------------------------------------------------------------------------

def test_a_dropped_sentence_is_one_event_per_article_it_cites(events):
    sentence = "L'azione contrattuale ex art. 1218 c.c. concorre con quella dell'art. 2043 c.c. per i fatti del caso concreto."
    citation_ledger.record("closure", sentence, "incompatible")
    got = ledger(events)
    assert [(e["pass"], e["article"], e["dropped"]) for e in got] == [("closure", "1218 c.c.", True),
                                                                      ("closure", "2043 c.c.", True)]
    assert got[0]["sentence"] == sentence[:80] and got[0]["reason"] == "incompatible"


def test_a_rewrite_keeping_the_citation_is_not_a_drop(events):
    citation_ledger.record_entries("rewrite_unsupported", [
        {"how": "rewritten", "sentence": "Il danno è prescritto ai sensi dell'art. 2947 c.c.",
         "now": "Il danno sarebbe prescritto, se il termine dell'art. 2947 c.c. risultasse decorso."},
        {"how": "kept", "sentence": "L'art. 2043 c.c. si applica."}])
    assert [(e["article"], e["dropped"]) for e in ledger(events)] == [("2947 c.c.", False)]


def test_a_sentence_citing_nothing_is_not_in_the_ledger(events):
    citation_ledger.record_entries("answer_tidy", [{"kind": "repeated sentence", "sentence": "Il fatto è pacifico."}])
    assert ledger(events) == []


def test_an_article_taken_out_by_name_is_dropped_even_if_the_sentence_stays(events):
    # citation discipline and the qualification check log {"removed": [...], "now": ...}
    citation_ledger.record_entries("discipline", [
        {"how": "rephrased", "removed": ["2947 c.c."], "sentence": "Per gli artt. 2043 c.c. e 2947 c.c. il danno va risarcito.",
         "now": "Per l'art. 2043 c.c. il danno va risarcito."}])
    assert {e["article"]: e["dropped"] for e in ledger(events)} == {"2043 c.c.": False, "2947 c.c.": True}


def test_final_checks_records_the_copy_it_drops(events):
    text = ("Nel caso concreto, poiché la struttura ha agito ex art. 1218 c.c., il debitore che non esegue esattamente la "
            "prestazione dovuta è tenuto al risarcimento del danno e per questo Caio va risarcito.")
    out = ptw.final_checks([("Applicazione al caso", text)], [row("1218", "c.c.", A1218, 3)], {})
    assert "1218" not in out[0][1]
    [event] = [e for e in ledger(events) if e["article"] == "1218 c.c."]
    assert event["pass"] == "final_checks" and event["dropped"] and event["sentence"].startswith("Nel caso concreto")
    assert "a copy inside a sentence: sentence dropped" in event["reason"]


def test_the_answer_tidy_repeat_pass_is_recorded(events):
    sentence = "L'art. 1218 c.c. pone a carico del debitore la prova della causa non imputabile nel caso in esame."
    text, log = text_tidy.tidy(f"{sentence} Altro. {sentence}")
    citation_ledger.record_entries("answer_tidy", log)
    assert [(e["article"], e["reason"]) for e in ledger(events)] == [("1218 c.c.", "repeated sentence")]


def test_a_cut_section_rewritten_without_an_article_is_recorded(monkeypatch, events):
    monkeypatch.setenv("WRITER_RULES_ENABLED", "true")
    old = ("L'art. 575 c.p. disciplina l'omicidio doloso, che qui non ricorre. L'art. 52 c.p., comma 2, riconosce inoltre "
           "che l'uso di un'arma legittimamente detenuta è")
    new = "L'art. 52 c.p. esclude la punibilità di chi agisce per difendere un diritto proprio dal pericolo attuale."
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: new)
    out, _ = ptw._complete_sections([("Norme e giurisprudenza", old)], [("Norme e giurisprudenza", "i", "m", 100)], "s")
    assert out[0][1] == new
    assert [(e["pass"], e["article"], e["dropped"]) for e in ledger(events)] == [("complete_sections", "575 c.p.", True)]


def test_the_ledger_never_raises(monkeypatch, events):
    monkeypatch.setattr(dg, "_refs_in", lambda text: 1 / 0)
    assert citation_ledger.record_entries("closure", [{"how": "sentence dropped", "sentence": "art. 1 c.c."}]) == []
    assert [n for n, _ in events] == ["pipeline_exception"]


def test_diagnose_long_puts_the_last_pass_that_dropped_it_in_the_slot():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("diagnose_long", here.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    spec = importlib.util.spec_from_file_location("eval_narrative", here / "eval_narrative.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    spec = importlib.util.spec_from_file_location("test_diagnose_long", here / "test_diagnose_long.py")
    tdl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdl)
    rec = tdl._record()
    rec["citation_removed"] = [
        {"pass": "final_checks", "article": "2043 c.c.", "reason": "a", "sentence": "s1", "dropped": True},
        {"pass": "answer_tidy", "article": "2043 c.c.", "reason": "repeated sentence", "sentence": "s2", "dropped": True},
        {"pass": "rewrite_unsupported", "article": "2043 c.c.", "reason": "r", "sentence": "s3", "dropped": False}]
    slots = {s["label"]: s for s in dl.analyse_case(rec, ev)["slots"]}
    assert slots["art. 2043 cc"]["removed_by"] == {"pass": "answer_tidy", "reason": "repeated sentence", "sentence": "s2"}
    assert slots["art. 584 cp"]["removed_by"] is None


# --- 2. a review flag acts on plan cover's insertion -------------------------------------------------------------------

def _chat_json(flags):
    def chat(messages, **kwargs):
        if messages[0].content == ptw._REVIEW_SYSTEM:
            return flags
        return [{"articolo": "A0", "compatibile": True}]       # the qualification judge
    return chat


def _cover(monkeypatch, reply, flags):
    monkeypatch.setenv("WRITER_RULES_ENABLED", "true")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: reply)
    monkeypatch.setattr(dg, "_chat_json", _chat_json(flags))
    plan = plan_of(("575", "c.p."))
    return ptw._cover_one(("575", "c.p."), plan["questioni"][0], ROWS[0], plan, [], ROWS, "Tizio spara al ladro.", {})


def test_p1_s_cover_sentence_flagged_as_a_fact_not_in_the_case_is_refused(monkeypatch, events):
    entry = _cover(monkeypatch, P1_COVER, [{"i": 1, "tipo": "fatto_non_nel_caso", "perche": P1_FLAG}])
    assert entry["how"] == "refused" and entry["why"] == f"review flag: fatto_non_nel_caso: {P1_FLAG}"
    assert entry["sentence"] == P1_COVER


def test_with_inline_marks_the_flagged_cover_sentence_goes_in_marked(monkeypatch, events):
    monkeypatch.setenv("FACT_MARK_INLINE_ENABLED", "true")
    entry = _cover(monkeypatch, P1_COVER, [{"i": 1, "tipo": "fatto_non_nel_caso", "perche": P1_FLAG}])
    assert entry["how"] == "inserted" and entry["sentence"] != P1_COVER and ptw._FACT_MARK.strip() in entry["sentence"]


def test_a_flag_that_drops_a_writer_sentence_refuses_the_insertion_with_its_reason(monkeypatch, events):
    entry = _cover(monkeypatch, P1_COVER, [{"i": 1, "tipo": "contro_assistito", "perche": "propone un'azione contro Tizio"}])
    assert entry["how"] == "refused"
    assert entry["why"] == "dropped by the review: against_client: propone un'azione contro Tizio"
    assert entry["sentence"] == P1_COVER          # what the cover wrote, logged even though the review emptied it


def test_an_unflagged_cover_sentence_still_goes_in(monkeypatch, events):
    entry = _cover(monkeypatch, P1_COVER, [])
    assert entry["how"] == "inserted" and entry["sentence"] == P1_COVER


# --- 3. GROUNDED_EXCLUSION ---------------------------------------------------------------------------------------------

def _refusal(sentence, plan, context=None, rows=ROWS):
    return ptw._cover_refusal(sentence, ("575", "c.p."), [], rows, plan, context or {})


def test_p1_s_575_excluded_by_55_is_accepted(monkeypatch, events):
    monkeypatch.setenv("GROUNDED_EXCLUSION", "true")
    assert _refusal(P1_575, plan_of(("575", "c.p."), ("55", "c.p."))) == ""
    [log] = [d for n, d in events if n == "grounded_exclusion_accepted"]
    assert (log["A"], log["B"], log["rule"], log["grade"], log["where"]) == ("575 c.p.", "55 c.p.", None, 3, "plan cover")


def test_575_excluded_by_an_article_outside_the_plan_is_refused(monkeypatch, events):
    monkeypatch.setenv("GROUNDED_EXCLUSION", "true")
    sentence = P1_575.replace("art. 55 c.p.", "art. 52 c.p.")
    assert _refusal(sentence, plan_of(("575", "c.p."), ("55", "c.p."))) == "cites another article, or not this one"
    assert not [n for n, _ in events if n == "grounded_exclusion_accepted"]


def test_the_flag_is_off_by_default(events):
    # v49: 55 c.p. is in the plan, so the second citation is allowed; the exclusion itself still refuses
    assert _refusal(P1_575, plan_of(("575", "c.p."), ("55", "c.p."))) == "says it does not apply"


@pytest.mark.parametrize("sentence, plan, rows, why", [
    # A graded 1: not a plan article worth excluding on a ground (v49: 55 in the plan, refused as an exclusion)
    (P1_575, plan_of(("575", "c.p."), ("55", "c.p.")),
     [row("575", "c.p.", A575, 1), ROWS[1]], "says it does not apply"),
    # a third article, all three in the plan: refused as an exclusion
    (P1_575.replace("dell'art. 55 c.p.", "degli artt. 55 c.p. e 52 c.p."),
     plan_of(("575", "c.p."), ("55", "c.p."), ("52", "c.p.")), ROWS, "says it does not apply"),
    # a third article outside the plan: another article
    (P1_575.replace("dell'art. 55 c.p.", "degli artt. 55 c.p. e 52 c.p."),
     plan_of(("575", "c.p."), ("55", "c.p.")), ROWS, "cites another article, or not this one"),
    # no ground at all: still a remark on relevance
    ("L'art. 575 c.p. non è rilevante per i fatti di Tizio nel caso concreto.", plan_of(("575", "c.p.")), ROWS,
     "says it does not apply"),
])
def test_every_other_does_not_apply_sentence_is_refused(monkeypatch, events, sentence, plan, rows, why):
    monkeypatch.setenv("GROUNDED_EXCLUSION", "true")
    assert _refusal(sentence, plan, rows=rows) == why


def test_the_strict_reader_sees_every_article_of_a_list_and_no_comma_number():
    # a supplement to read_citations, which reads the article before "comma 2" itself
    assert set(ptw._strict_refs("degli artt. 55 c.p. e 52 c.p.")) == {("55", "c.p."), ("52", "c.p.")}
    assert set(ptw._strict_refs("degli artt. 55 e 52 c.p.")) == {("55", "c.p."), ("52", "c.p.")}
    assert ptw._strict_refs("ai sensi dell'art. 52, comma 2, c.p.") == []
    assert ptw._strict_refs("(art. 2653, n. 1, c.c.)") == []


def test_a_fired_rule_excluding_the_article_is_a_ground(monkeypatch, events):
    monkeypatch.setenv("GROUNDED_EXCLUSION", "true")
    sentence = "L'art. 575 c.p. non si applica, perché Tizio ha reagito a un'aggressione in atto nella propria abitazione."
    context = {"excluded_by": [{"rule": "legittima_difesa_domiciliare", "article": ("575", "c.p.")}]}
    assert _refusal(sentence, plan_of(("575", "c.p.")), context) == ""
    [log] = [d for n, d in events if n == "grounded_exclusion_accepted"]
    assert (log["A"], log["B"], log["rule"]) == ("575 c.p.", None, "legittima_difesa_domiciliare")
    assert _refusal(sentence, plan_of(("575", "c.p.")), {}) == "says it does not apply"     # no rule fired


def test_the_writer_review_keeps_a_grounded_exclusion_and_drops_the_rest(monkeypatch, events):
    monkeypatch.setenv("GROUNDED_EXCLUSION", "true")
    other = "L'art. 52 c.p. non è applicabile perché il caso concreto riguarda altro."
    text = f"{P1_575} {other}"
    flags = [{"i": 1, "tipo": "non_applicabile", "perche": "p"}, {"i": 2, "tipo": "non_applicabile", "perche": "p"}]
    monkeypatch.setattr(dg, "_chat_json", _chat_json(flags))
    out, log = ptw.review_section("Norme e giurisprudenza", text, "Il caso.", ROWS, [], "Tizio", {},
                                  plan=plan_of(("575", "c.p."), ("55", "c.p.")))
    assert P1_575 in out and "52 c.p." not in out
    assert [(e["kind"], e["sentence"][:12]) for e in log] == [("non_application", "L'art. 52 c.")]
    [accepted] = [d for n, d in events if n == "grounded_exclusion_accepted"]
    assert accepted["where"] == "Norme e giurisprudenza"


# --- v49: another article of the plan does not refuse a cover sentence ---------------------------------------------------

H7_320 = ("L'art. 320 c.p. (Corruzione di persona incaricata di un pubblico servizio) indica che le disposizioni degli "
          "artt. 318 e 319 si applicano anche all'incaricato di un pubblico servizio, ribadendo la rilevanza del contesto "
          "normativo.")


def test_h7_320_explained_through_318_and_319_of_the_plan_goes_in(events):
    rows = [row(n, "c.p.", f"Testo dell'articolo {n}.", 2) for n in ("318", "319", "320")]
    plan = plan_of(("318", "c.p."), ("319", "c.p."), ("320", "c.p."))
    assert ptw._cover_refusal(H7_320, ("320", "c.p."), [], rows, plan, {}) == ""
    # 318 and 319 outside the plan: still another article
    assert ptw._cover_refusal(H7_320, ("320", "c.p."), [], rows, plan_of(("320", "c.p.")), {}) == \
        "cites another article, or not this one"
