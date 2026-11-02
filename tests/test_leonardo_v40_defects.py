"""Leonardo v40: the defects, from their exact strings, on invented cases."""
import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing as aw, citation_closure as cc, fact_signals as fs, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

TEXTS = {
    ("2043", "c.c."): "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.",
    ("700", "c.p.c."): "Fuori dei casi regolati, chi ha fondato motivo di temere un pregiudizio imminente e irreparabile può chiedere con ricorso al giudice i provvedimenti d'urgenza, che appaiono più idonei.",
    ("614-bis", "c.p.c."): "Con il provvedimento di condanna all'adempimento di obblighi diversi dal pagamento di una somma di denaro il giudice fissa la somma dovuta.",
    ("1067", "c.c."): "Il proprietario del fondo dominante non può fare innovazioni che rendano più gravosa la condizione del fondo servente.",
}
ROWS = [{"_article": k, "_grade": 3, "s": {"plain_text": t}} for k, t in TEXTS.items()]


def facts(**kw):
    base = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    base.update(kw)
    return base


# --- 1 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, è fondata in quanto l'opera è abusiva.",
    "In caso di pericolo grave si può chiedere con ricorso al giudice i provvedimenti d'urgenza, permettono a Leonardo di agire.",
    "Con il provvedimento di condanna all'adempimento di obblighi diversi dal., permettono a Leonardo di agire.",
    "Il proprietario del fondo dominante non può fare innovazioni che rendano, il proprietario del fondo servente subisce il danno.",
])
def test_an_article_s_words_with_no_citation_left_go(sentence):
    out, log = text_tidy.against_articles(f"Leonardo agisce. {sentence} La domanda è fondata.", ROWS)
    assert out == "Leonardo agisce. La domanda è fondata."
    assert log[0]["kind"] == "comma words with a citation"


def test_a_whole_comma_becomes_a_quote_and_one_broken_by_its_citation_goes():
    whole = TEXTS[("2043", "c.c.")]
    assert text_tidy.against_articles(whole, ROWS)[0] == f"«{whole[:-1]}» (art. 2043 c.c.)."
    broken = "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, per l'art. 2043 c.c., obbliga al risarcimento."
    assert text_tidy.against_articles(broken, ROWS)[0] == ""


@pytest.mark.parametrize("sentence", [
    "Secondo «Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto», è fondata la domanda.",
    "Leonardo ha subito un danno ingiusto, che va risarcito per intero dal vicino.",
])
def test_a_comma_quoted_cited_or_merely_echoed_stays(sentence):
    assert sentence in text_tidy.against_articles(sentence, ROWS)[0]


def test_a_section_title_cut_off_counts_as_the_article_s_words():
    rows = [{"_article": ("9", "c.c."), "s": {"plain_text": "Testo.", "title": "La tutela del possesso legittimo contro le turbative e le molestie di"}}]
    text = "La tutela del possesso legittimo contro le turbative e le molestie di, consente a Leonardo di agire."
    assert text_tidy.against_articles(text, rows)[0] == ""


@pytest.mark.parametrize("old, drop", [
    ("Gli artt. 700 e 614-bis c.p.c. permettono a Leonardo di agire subito in giudizio.", [("700", "c.p.c."), ("614-bis", "c.p.c.")]),
    ("Con il provvedimento di condanna diversi dal, ai sensi dell'art. 614-bis c.p.c., co. 1., permettono a Leonardo di agire.", [("614-bis", "c.p.c.")]),
])
def test_a_removal_that_breaks_the_sentence_is_seen(old, drop):
    assert text_tidy.breaks_without_citation(old, aw.drop_citations(old, drop, [], set(drop)))


def test_a_removal_that_leaves_a_sentence_is_seen_too():
    old = "Leonardo può agire in giudizio contro il vicino, ai sensi dell'art. 700 c.p.c., senza attendere."
    assert not text_tidy.breaks_without_citation(old, aw.drop_citations(old, [("700", "c.p.c.")], [], {("700", "c.p.c.")}))


def test_closure_drops_the_sentence_a_removal_would_break():
    text, log = cc._apply("Gli artt. 9998 e 9999 c.c. permettono a Leonardo di agire subito in giudizio.",
                          {(0, 76): [("9998", "c.c."), ("9999", "c.c.")]}, [])
    assert "Permettono" not in text and log[0]["how"] == "sentence dropped"


def test_the_writer_is_told_to_name_a_norm_by_its_number():
    assert "mai le parole del suo testo" in ptw._WRITER_RULES


def test_the_trace_logs_the_step_the_words_appear_in(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    ptw._trace_comma_words("writer_review", [("S", "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, è fondata la pretesa.")], ROWS)
    assert events[0][0] == "comma_words_trace" and events[0][1]["step"] == "writer_review" and events[0][1]["count"] == 1


def test_the_final_checks_run_before_cover_can_write_the_article_back(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    out = ptw.final_checks([("Applicazione al caso", "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, è fondata. Il resto del testo è corretto.")], ROWS, {})
    assert out[0][1] == "Il resto del testo è corretto."


# --- 2 ---------------------------------------------------------------------------------------------------------------

def test_a_servitude_case_makes_mediation_a_condition_of_the_action():
    m = fs.mediation(facts(), [{"id": "destinazione_padre_famiglia", "area": "diritti_reali"}])
    assert m["required"] and m["subject"] == "diritti reali" and "condizione di procedibilità" in m["text"] and "art. 5 d.lgs. 28/2010" in m["text"]
    block = fs.conclusions_block({"mediation": m})
    assert "Mediazione obbligatoria" in block


@pytest.mark.parametrize("contract, subject", [("comodato d'uso", "comodato"), ("locazione abitativa", "locazione"), ("polizza vita", "contratti assicurativi, bancari e finanziari")])
def test_the_subjects_of_the_article_by_the_contracts(contract, subject):
    assert fs.mediation(facts(contracts=[{"type": contract}]), [])["subject"] == subject


def test_outside_the_list_it_is_not_required():
    assert fs.mediation(facts(contracts=[{"type": "compravendita"}]), [{"id": "x", "area": "contratti"}])["required"] is False


def test_the_article_s_own_text_rules_the_list():
    fired = [{"id": "x", "area": "diritti_reali"}]
    assert fs.mediation(facts(), fired, "In materia di condominio, locazione e comodato ...")["required"] is False
    assert fs.mediation(facts(), fired, "In materia di condominio, diritti reali, divisione ...")["required"] is True


def test_a_sentence_denying_it_goes():
    assert "art. 5" in ptw._wrong_statement("La mediazione obbligatoria non è necessaria.", {"mediation": True})
    assert ptw._wrong_statement("La mediazione è condizione di procedibilità della domanda.", {"mediation": True}) == ""


def test_outside_the_list_nothing_on_it_being_mandatory():
    assert ptw._wrong_statement("La mediazione è obbligatoria.", {"mediation": False}) != ""
    assert ptw._wrong_statement("Le parti possono tentare una mediazione.", {"mediation": False}) == ""


# --- 3 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", ["Lo sfratto non è pertinente al caso.", "Il decreto ingiuntivo non è praticabile.",
                                      "Il sequestro non è necessario.", "La diffida non è necessaria.", "- Sequestro: non necessario."])
def test_no_line_on_a_tool_that_does_not_apply(sentence):
    assert ptw._wrong_statement(sentence, {}) == "a line on a tool that does not apply"


def test_a_tool_that_applies_stays():
    assert ptw._wrong_statement("Leonardo può chiedere il sequestro giudiziario del passaggio.", {}) == ""


TERMS = [{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c.", "length": "1 anno"}]


def test_a_time_barred_action_appears_once_in_the_terms_never_among_the_tools(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    text = ("**Termini**\n\nL'azione possessoria è preclusa: il termine annuale è decorso.\n\n"
            "**Strumenti processuali**\n\nLeonardo può chiedere il provvedimento d'urgenza. L'azione di reintegrazione resta un rimedio utile.\n\n"
            "**Voci di danno**\n\nIl danno da mancato uso.")
    out = ptw.final_checks([("Profili", text)], [], {"terms": TERMS})[0][1]
    assert "preclusa" in out and "reintegrazione" not in out and "provvedimento d'urgenza" in out and "Voci di danno" in out


# --- 4 ---------------------------------------------------------------------------------------------------------------

SERVITU = [{"id": "destinazione_padre_famiglia", "area": "diritti_reali", "acquired_right": True, "institutions": ["servitù per destinazione del padre di famiglia"]}]


def test_who_claims_a_servitude_has_the_confessoria_alone():
    assert [t["id"] for t in fs.no_prescription_terms(facts(), SERVITU)] == ["azione_confessoria"]


def test_the_owner_has_the_negatoria():
    ids = [t["id"] for t in fs.no_prescription_terms(facts(), [{"id": "x", "area": "diritti_reali"}])]
    assert "azione_negatoria" in ids and "azione_confessoria" not in ids


def plan_of(*issues):
    return {"parte_assistita": "Leonardo", "questioni": [{"titolo": t, "argomento": "", "norme": list(n)} for t, n in issues]}


def test_only_the_actions_the_plan_uses_are_listed():
    analysis = {"fact_signals": {"facts": {"parties": []}, "fired": [], "terms": [
        {"id": "azione_negatoria", "status": fs.NO_PRESCRIPTION_STATUS, "ref": "art. 949 c.c."},
        {"id": "azione_rivendicazione", "status": fs.NO_PRESCRIPTION_STATUS, "ref": "art. 948 c.c."},
        {"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."}]}}
    _, out = ptw.for_the_plan(plan_of(("Azione negatoria", [("949", "c.c.")])), analysis)
    assert [t["id"] for t in out["fact_signals"]["terms"]] == ["azione_negatoria", "possessoria"]


def test_art_949_leaves_the_plan_of_who_claims_a_servitude():
    analysis = {"fact_signals": {"facts": {"parties": []}, "fired": [{"acquired_right": True, "institutions": ["servitù per destinazione"], "why": "x"}], "terms": []}}
    plan, _ = ptw.for_the_plan(plan_of(("Tutela", [("949", "c.c."), ("1079", "c.c.")])), analysis)
    assert plan["questioni"][0]["norme"] == [("1079", "c.c.")]


def test_the_negatoria_sentence_goes_at_the_final_checks(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    text = "L'azione negatoria, volta a far dichiarare l'inesistenza del diritto di Domitilla al cancello, è proponibile. La confessoria è il rimedio."
    assert ptw.final_checks([("S", text)], [], {"servitude": True})[0][1] == "La confessoria è il rimedio."


# --- 5 ---------------------------------------------------------------------------------------------------------------

NON_USE = {"terms": [{"id": "non_uso_servitu", "from": {"date": "marzo 2016"}, "ref": "art. 1073 c.c.", "length": "20 anni", "status": "NON DECORSO"}]}


def test_the_non_use_term_says_what_it_is():
    use = [{"who": "Leonardo", "asset": "stradella", "until": "marzo 2016", "use": "passaggio"}]
    term = next(t for t in fs.compute_terms(facts(possession=use)) if t["id"] == "non_uso_servitu")
    assert term["text"] == "estinzione per non uso: 2036 (20 anni dall'ultimo uso, marzo 2016); non è il termine dell'usucapione"
    assert term["label"].startswith("estinzione della servitù per non uso")


def test_its_date_is_never_the_start_of_usucapione():
    assert "non-use" in ptw._wrong_statement("Il termine ventennale, calcolato a partire dal 2016, non è stato soddisfatto.", NON_USE)


def test_stated_as_extinction_it_stays():
    assert ptw._wrong_statement("La servitù si estingue per non uso dopo 20 anni dal 2016 (art. 1073 c.c.).", NON_USE) == ""


# --- 6 ---------------------------------------------------------------------------------------------------------------

MEVIO = [{"asset": "fondo grande", "owner": "Mevio", "ended_by": "divisione"}, {"asset": "fondo nord", "owner": "Sempronio"},
         {"asset": "fondo sud", "owner": "Tizio"}, {"asset": "fondo nord", "owner": "Domitilla"}, {"asset": "fondo sud", "owner": "Leonardo"}]


def mevio():
    return facts(property_history=MEVIO, split_into_parts=[{"asset": "fondo grande", "owner": "Mevio", "parts": "Sempronio, Tizio"}],
                 parties=[{"name": "Leonardo", "role": "cliente"}])


def test_the_other_contractor_is_of_the_split_of_the_land():
    out = fs.other_contractor(mevio())
    assert out == "l'altro contraente è Sempronio, condividente nella divisione del fondo di Mevio"


def test_a_sale_that_split_the_land_names_the_seller():
    f = mevio()
    f["property_history"] = [{**MEVIO[0], "ended_by": "compravendita"}] + MEVIO[1:]
    assert fs.other_contractor(f).startswith("l'altro contraente è Mevio, che ha alienato")


def test_art_1054_on_another_sale_goes():
    context = {"split_owner": "Mevio"}
    assert "Mevio" in ptw._wrong_statement("L'art. 1054 c.c. opera per l'alienazione del fondo di Sempronio a Domitilla.", context)
    assert ptw._wrong_statement("L'art. 1054 c.c. opera per l'alienazione del fondo di Mevio.", context) == ""
