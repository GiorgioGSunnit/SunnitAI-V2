"""The v37 32-case report: wrong code on a citation, citations outside the material, plan articles never written,
articles past the grader's limit. General rules, on invented cases."""
import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import answer_invariants as ai, citation_closure as cc, feature_flags  # noqa: E402
from src.rag.issue_spotting import names_document, resolve_code  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

DOCUMENTS = [{"id": str(i), "name": name} for i, name in enumerate((
    "Sicurezza Urbana 2026", "Legge sul divorzio 2024", "Codice antimafia", "Codice del processo amministrativo 2025",
    "Codice del Consumo 2025", "L. 241/1990 Procedimento amministrativo", "Costituzione"))]


def refs(text, known=None):
    return [r for _, _, rs in ptw._citations(text, DOCUMENTS, known) for r in rs]


# --- 1: the code of a citation --------------------------------------------------------------------------------------

@pytest.mark.parametrize("tail", ["L. 241/1990", "cod. civ.", "si applica al caso", "del codice", "e la pena", "c.p.a"])
def test_a_word_after_the_number_is_not_a_document_it_starts_like(tail):
    code = dg._code_after(f" {tail}", DOCUMENTS)
    assert code not in ("Sicurezza Urbana 2026", "Legge sul divorzio 2024", "Codice antimafia")


def test_only_the_words_that_name_a_document_resolve_it():
    assert dg._code_after(" della legge sul divorzio e la", DOCUMENTS) == "Legge sul divorzio 2024"
    assert dg._code_after(" del codice antimafia", DOCUMENTS) == "Codice antimafia"
    assert dg._code_after(" del codice del consumo, che", DOCUMENTS) == "Codice del Consumo 2025"
    assert dg._code_after(" l. 241/1990", DOCUMENTS) == "L. 241/1990 Procedimento amministrativo"
    assert dg._code_after(" della legge sul procedimento amministrativo", DOCUMENTS) == "L. 241/1990 Procedimento amministrativo"
    assert dg._code_after(" Cost.", DOCUMENTS) == "Costituzione"


def test_a_short_phrase_matches_whole_words_only():
    assert resolve_code("si", DOCUMENTS) is None and resolve_code("cod", DOCUMENTS) is None
    assert not names_document("l", DOCUMENTS[5])
    assert not names_document("codice", {"name": "Codice antimafia"}) and not names_document("legge sul", DOCUMENTS[1])
    assert not names_document("procedimento amministrativo", {"name": "Codice del processo amministrativo 2025"})


def test_a_bare_number_in_a_quote_takes_the_code_the_quote_comes_from():
    text = ("L'art. 644 c.p. punisce «chiunque, fuori dei casi previsti dall'articolo 643, si fa dare interessi usurari».")
    assert refs(text) == [("644", "c.p."), ("643", "c.p.")]


def test_a_quote_of_several_sentences_keeps_the_code_it_opened_with():
    text = ("Dispone l'art. 644 c.p.: «È punito chiunque si fa dare interessi. Resta salvo quanto previsto dall'articolo 643.»")
    assert ("643", "c.p.") in refs(text)


def test_the_nearest_explicit_reference_wins_in_the_sentence():
    assert refs("L'art. 2043 c.c. e l'art. 185 c.p., in relazione all'articolo 186, fondano la pretesa.")[-1] == ("186", "c.p.")


def test_a_code_named_without_an_article_counts_as_an_explicit_reference():
    assert refs("Il codice penale, all'articolo 643, punisce la circonvenzione.") == [("643", "c.p.")]


def test_with_no_reference_near_it_takes_the_one_material_article_with_that_number():
    assert refs("La norma dell'art. 927 regola il caso.", {("927", "c.c.")}) == [("927", "c.c.")]


def test_several_or_no_material_articles_resolve_nothing():
    assert refs("La norma dell'art. 3 regola il caso.", {("3", "c.c."), ("3", "c.p.")}) == []
    assert refs("La norma dell'art. 927 regola il caso.", {("1", "c.c.")}) == []


def test_a_nearby_code_never_goes_to_a_citation_of_another_act():
    assert refs("L'art. 3 Cost. e l'art. 2059 c.c. fondano il danno.") == [("3", "Costituzione"), ("2059", "c.c.")]
    assert refs("L'art. 5 del contratto viola l'art. 1341 c.c.") == [("1341", "c.c.")]
    assert refs("L'art. 7 d.lgs. 999/2031 e l'art. 1341 c.c.") == [("1341", "c.c.")]


def test_the_material_corrects_a_near_code_it_does_not_hold():
    assert refs("L'art. 2043 c.c., in relazione all'art. 185, fonda il danno.", {("2043", "c.c."), ("185", "c.p.")})[-1] == ("185", "c.p.")


# --- 2: citations outside the material ------------------------------------------------------------------------------

class Session:
    """The corpus' texts by (number, code), through the real _fetch_articles query shapes: a code's documents by its
    name prefix (one document per code, its id the code), then their Sections by id."""

    def __init__(self, texts):
        self.texts = texts
        self.asked = []

    def run(self, query, prefixes=None, items=None, **_):
        if prefixes is not None:
            rows = [{"prefix": p, "id": c} for p in prefixes for c, name in dg._CODE_PREFIXES.items() if name == p]
        else:
            rows = []
            for it in items:
                code, n = it["doc"], it["num"]
                self.asked.append(n)
                text = self.texts.get((n, code))
                if text:
                    rows.append({"doc": code, "num": n, "d": {"id": code, "name": dg._CODE_PREFIXES[code],
                                                              "document_type": "primary"},
                                 "s": {"name": f"{n}.0.0", "plain_text": text, "titolo_articolo": f"Rubrica {n}"},
                                 "next": None})

        class Result:
            def data(self):
                return rows
        return Result()


ANALYSIS = {"frame": {"areas": ["responsabilita_civile"]}, "area": "civile"}
ROW = {"_article": ("2043", "c.c."), "_grade": 3, "s": {"name": "2043.0.0", "plain_text": "Qualunque fatto doloso."}}


def judge(incompatible=()):
    def chat(messages, **_):
        items = [line.split("]")[0][1:] for line in messages[1].content.split("\n") if line.startswith("[A")]
        names = [line.split("art. ")[1].split()[0] for line in messages[1].content.split("\n") if line.startswith("[A")]
        return [{"articolo": a, "compatibile": n not in incompatible, "perche": "altra materia", "tipo": "materia"}
                for a, n in zip(items, names)]
    return chat


def close(draft, session, chat=None, rows=(ROW,), case="Il caso."):
    return cc.close(draft, analysis=ANALYSIS, article_rows=list(rows), documents=DOCUMENTS, session=session,
                    case_text=case, chat_json=chat or judge())


def test_a_compatible_citation_becomes_material_and_a_source():
    text, rows, log = close("Il danno non patrimoniale spetta ex art. 2059 c.c. nei casi previsti dalla legge.",
                            Session({("2059", "c.c."): "Il danno non patrimoniale..."}))
    assert "2059" in text and rows[0]["_article"] == ("2059", "c.c.") and rows[0]["_closure"]
    assert log["verified"] == ["2059 c.c."] and log["remaining"] == 0
    found = dg._mark_unverified(text, [("2043", "c.c."), ("2059", "c.c.")], "", wrong_code=True)
    assert "DA VERIFICARE" not in found


def test_a_mark_the_writer_left_on_a_verified_citation_goes():
    text, _, _ = close("Spetta il danno ex art. 2059 c.c. [DA VERIFICARE] nei casi di legge.", Session({("2059", "c.c."): "testo"}))
    assert "DA VERIFICARE" not in text


def test_an_article_not_found_loses_the_citation_and_the_sentence_stands():
    text, rows, log = close("Il danno si risarcisce per intero nei limiti di legge (art. 9999 c.c.). Altro.", Session({}))
    assert "9999" not in text and "Il danno si risarcisce per intero" in text and rows == []
    assert log["counts"]["not found"] == 1 and log["citations_removed"] == 1


def test_a_sentence_that_cannot_stand_without_the_citation_goes():
    text, _, log = close("Si veda l'art. 9999 c.c. La domanda è fondata nel merito del caso.", Session({}))
    assert text == "La domanda è fondata nel merito del caso." and log["sentences_dropped"] == 1


def test_an_incompatible_citation_is_removed_with_the_check_s_reason():
    session = Session({("1917", "c.c."): "Nell'assicurazione della responsabilità civile..."})
    text, rows, log = close("Il venditore risponde del danno al compratore ai sensi dell'art. 1917 c.c. in ogni caso.",
                            session, judge({"1917"}))
    assert "1917" not in text and rows == [] and log["counts"]["incompatible"] == 1
    assert log["entries"][0]["reason"] == "altra materia"


def test_a_failed_check_verifies_nothing():
    def broken(*_, **__):
        raise ValueError("no reply")
    text, rows, log = close("Il danno spetta alla parte lesa dall'art. 2059 c.c. come regola generale.",
                            Session({("2059", "c.c."): "testo"}), broken)
    assert "2059" not in text and rows == [] and log["counts"]["no verdict"] == 1


def test_compatible_in_one_sentence_and_not_in_another_stays_in_the_first():
    def chat(messages, **_):
        lines = messages[1].content.split("\n\n")
        return [{"articolo": f"A{k}", "compatibile": "contratto" not in line, "tipo": "materia"} for k, line in enumerate(lines)]
    text, rows, _ = close("Il danno morale spetta alla vittima per l'art. 2059 c.c. secondo i fatti.\n\n"
                          "Il contratto si risolve di diritto per effetto dell'art. 2059 c.c. secondo i fatti.",
                          Session({("2059", "c.c."): "testo"}), chat)
    assert text.count("2059") == 1 and rows


def test_an_article_of_an_abrogated_text_is_not_found():
    _, rows, log = close("Il termine decorre dalla scoperta del fatto per l'art. 1000 c.c. in generale.",
                         Session({("1000", "c.c."): "(abrogato)"}))
    assert rows == [] and log["counts"]["not found"] == 1


def test_an_act_outside_the_material_and_areas_is_not_even_read():
    session = Session({})
    text, _, log = close("La tutela è completa nel caso concreto ai sensi dell'art. 5 Legge sul divorzio in generale.", session)
    assert session.asked == [] and "divorzio" not in text and log["counts"]["outside the case's material and areas"] == 1


def test_a_criminal_article_in_a_civil_case_outside_its_material_is_outside_its_scope():
    text, _, log = close("Il termine per agire è di tre mesi per l'art. 124 c.p. come regola generale.", Session({("124", "c.p."): "x"}))
    assert "124" not in text and log["remaining"] == 0


def test_a_bare_number_found_in_one_code_of_the_areas_is_that_article():
    text, rows, log = close("Spetta il danno non patrimoniale previsto dall'art. 2059 secondo i fatti del caso.",
                            Session({("2059", "c.c."): "testo"}))
    assert rows[0]["_article"] == ("2059", "c.c.") and log["by_area"] == {"2059": "2059 c.c."} and "2059" in text


def test_a_bare_number_found_in_several_codes_of_the_areas_goes():
    text, rows, log = close("Spetta il danno previsto dall'art. 700 secondo i fatti del caso concreto.",
                            Session({("700", "c.c."): "uno", ("700", "c.p.c."): "due"}))
    assert "700" not in text and rows == [] and log["counts"]["unresolved"] == 1


def test_the_case_s_own_bare_article_is_left_alone():
    text, _, log = close("La clausola dell'art. 5 impone il pagamento entro il mese corrente.", Session({}),
                         case="L'art. 5 impone il pagamento.")
    assert "art. 5" in text and log["remaining"] == 0


def test_a_run_with_a_good_and_a_bad_article_never_leaves_the_bad_one():
    text, _, log = close("La garanzia opera ai sensi degli artt. 2043 e 9999 c.c. per il danno subito.", Session({}))
    assert "9999" not in text and log["remaining"] == 0


def test_a_verified_article_passes_the_invariants_and_the_second_reading():
    row = {"_article": ("2059", "c.c."), "_closure": True, "s": {"plain_text": "testo", "titolo_articolo": "Danni"}}
    assert ("2059", "c.c.") in ai.allowed_refs([ROW, row], [], {}, [])
    text, counts = ai.i2_consistency("Spetta il danno per l'art. 2059 c.c. come stabilito dai fatti.", {("2059", "c.c."): row},
                                     set(), [], {("2059", "c.c.")}, lambda *a, **k: pytest.fail("judged twice"))
    assert counts["found"] == 0 and "2059" in text


def test_closure_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CITATION_CLOSURE_ENABLED", raising=False)
    assert not cc.enabled() and not feature_flags.enabled("PLAN_ARTICLES_COVER_ENABLED")


# --- 3: plan articles the writer omitted ----------------------------------------------------------------------------

def row(number, grade=3, **extra):
    tags = {"_grade": grade} if grade is not None else {}
    return {"_article": (number, "c.c."), **tags, **extra,
            "s": {"plain_text": f"Testo dell'articolo {number}.", "titolo_articolo": f"Rubrica {number}"}}


def plan_of(*norms):
    return {"parte_assistita": "Tizio", "questioni": [{"titolo": "Clausola penale", "argomento": "La penale è dovuta.",
                                                       "norme": [(n, "c.c.") for n in norms]}]}


SECTIONS = [("Risposte ai quesiti", "Sì, per l'art. 1218 c.c."),
            ("Applicazione al caso", "**Clausola penale**\n\nIl debitore è inadempiente ex art. 1218 c.c.\n\n"
                                     "**Altra questione**\n\nAltro testo del parere.")]


@pytest.fixture
def quiet(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    monkeypatch.setattr(ptw, "SECTION_TITLES", ptw.SECTION_TITLES[:3] + ("Applicazione al caso",) + ptw.SECTION_TITLES[4:])
    for name in ("WRITER_RULES_ENABLED", "REVIEW_FIXES_ENABLED", "GROUNDED_REVIEW_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dg, "_chat_json", judge())
    return events


def reply(monkeypatch, text):
    monkeypatch.setattr(dg, "_call_chat", lambda messages, **k: text)


def test_a_graded_plan_article_no_section_cites_is_written_under_its_issue(monkeypatch, quiet):
    reply(monkeypatch, "La penale pattuita è dovuta ai sensi dell'art. 1382 c.c., che limita il risarcimento.")
    out = ptw.cover_plan_articles(plan_of("1218", "1382"), SECTIONS, [row("1218"), row("1382")], [], "Il caso.")
    body = out[1][1]
    assert body.index("1382") < body.index("**Altra questione**") and body.index("1218") < body.index("1382")
    log = next(d for n, d in quiet if n == "case_analysis_plan_articles")
    assert log["inserted"] == 1 and log["entries"][0]["article"] == "1382 c.c." and "sentence" in log["entries"][0]
    assert log["entries"][0]["section"] == "Applicazione al caso"


def test_an_article_already_cited_or_graded_low_gets_no_call(monkeypatch, quiet):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("called"))
    rows = [row("1218"), row("1382", 1), row("1383", None, _applicability="non valutato")]
    assert ptw.cover_plan_articles(plan_of("1218", "1382", "1383"), SECTIONS, rows, [], "Il caso.") == SECTIONS


def test_an_article_never_sent_to_the_grader_is_not_covered(monkeypatch, quiet):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("called"))
    assert ptw.cover_plan_articles(plan_of("1079"), SECTIONS, [row("1079", None)], [], "Il caso.") == SECTIONS


@pytest.mark.parametrize("text, why", [
    ("L'art. 1382 c.c. e l'art. 1384 c.c. regolano la penale dovuta dal debitore.", "another article"),
    ("La penale è dovuta come regola del contratto stipulato tra le parti.", "another article"),
    ("La penale è dovuta per l'art. 1382 c.c. (Cass. n. 123/2020), come detto.", "a ruling"),
    ("L'art. 1382 c.c. non si applica al caso qui in esame per i fatti.", "does not apply"),
    ("La penale è dovuta ai sensi dell'art. 1382 c.c. e il", "empty or cut"),
])
def test_a_reply_that_breaks_a_rule_is_refused_and_logged(monkeypatch, quiet, text, why):
    reply(monkeypatch, text)
    out = ptw.cover_plan_articles(plan_of("1382"), SECTIONS, [row("1382")], [], "Il caso.")
    assert "1382" not in out[1][1]
    assert why in next(d for n, d in quiet if n == "case_analysis_plan_articles")["entries"][0]["why"]


def test_the_insertion_goes_through_the_qualification_check(monkeypatch, quiet):
    reply(monkeypatch, "La penale è dovuta ai sensi dell'art. 1382 c.c. come stabilito tra le parti.")
    monkeypatch.setattr(dg, "_chat_json", judge({"1382"}))
    assert "1382" not in ptw.cover_plan_articles(plan_of("1382"), SECTIONS, [row("1382")], [], "Il caso.")[1][1]


def test_the_insertion_goes_through_the_grounded_judge(monkeypatch, quiet):
    reply(monkeypatch, "La penale è dovuta ai sensi dell'art. 1382 c.c. come stabilito tra le parti.")
    monkeypatch.setenv("GROUNDED_REVIEW_ENABLED", "true")
    monkeypatch.setattr(ptw.grounded_review, "_grounded_judge", lambda *a: {0: {"tipo": "regola", "esito": "assente"}})
    assert "1382" not in ptw.cover_plan_articles(plan_of("1382"), SECTIONS, [row("1382")], [], "Il caso.")[1][1]


def test_at_most_two_sentences_go_in(monkeypatch, quiet):
    reply(monkeypatch, "Prima frase sull'art. 1382 c.c. del caso. Seconda frase della stessa norma. Terza frase di troppo.")
    assert "Terza" not in ptw.cover_plan_articles(plan_of("1382"), SECTIONS, [row("1382")], [], "Il caso.")[1][1]


def test_with_no_subheading_for_the_issue_it_closes_the_section():
    assert ptw._insert_for_issue("Testo.\n\nAltro.", "Clausola penale", "Nuova.") == "Testo.\n\nAltro.\n\nNuova."


# --- 4: every candidate graded --------------------------------------------------------------------------------------

def test_every_article_is_graded_whatever_its_rank(monkeypatch):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("APPLICABILITY_CHECK_ENABLED", "true")
    monkeypatch.delenv("GRADED_MAX_CHECKED", raising=False)
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    seen = []

    def grade(case, texts, *a, **k):
        seen.extend(texts)
        return {dg._applicability_key(label): {"grade": 2, "issue": "x"} for label in texts}, []
    monkeypatch.setattr(dg, "_graded_verdicts", grade)
    rows = [{"_article": (str(n), "c.c."), "s": {"plain_text": f"Testo {n}"}} for n in range(1, 101)]
    kept = dg._filter_applicable_articles("Fatti.", "", rows)
    assert len(seen) == 100 and all(r.get("_grade") == 2 for r in kept)
    log = next(d for n, d in events if n == "case_analysis_applicability" and "verdicts" in d)
    assert log["unchecked"] == 0 and log["unchecked_labels"] == []


def test_the_limit_stays_an_a_b_setting(monkeypatch):
    monkeypatch.setenv("GRADED_MAX_CHECKED", "72")
    assert dg._graded_max_checked() == 72
