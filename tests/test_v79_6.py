"""v79.6 -- synthetic sentences and the nightly cells only:

1. the questions answered first («Risposte ai quesiti» before §1), each grounded or «Non risulta …»
2. more than one client: per-client charge, candidates and §1; sections shared when the positions are the same
3. repealed articles: a repeal reference alone, a repeal note at the head, a status property; never a charge
4. alternative elements: one element with branches, graded on its branches
5. role unknown: derived from the text, else a reading said in one line; never the other side's wording by default
6. a keyword hit outside the case's matters is dropped before grading, every book
7. the nightly: questions_answered, per-client counts, alternative_elements, a two-client cell
"""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato", **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}


def _ctx(recs, posture=IMPUTATO, case="Alfa prende la bicicletta di Beta dal cortile.", client="Alfa"):
    from src.rag import lawyer_method as M
    return M.prepare({"records": recs, "sentences": [{"id": "S1", "text": case, "source": "caso",
                                                      "certainty": "accertato"}]}, posture, [], case, client, "",
                     list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v796")


def _theft(*statuses):
    texts = ["si impossessa della cosa mobile altrui", "sottraendola a chi la detiene"]
    return {"624 c.p.": {"rubrica": "Furto", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st) for i, st in enumerate(statuses, 1)]}}


# --- 1. questions ----------------------------------------------------------------------------------------------------------

def test_every_explicit_question_is_extracted_in_order():
    from src.rag import intake
    text = ("Alfa ha venduto un'auto a Beta. Chiedo se il contratto sia valido. Vorrei sapere entro quando si può agire. "
            "Quale sia il giudice competente non lo so. È possibile chiedere i danni? E la caparra?")
    assert intake.questions(text) == ["Chiedo se il contratto sia valido.", "Vorrei sapere entro quando si può agire.",
                                      "Quale sia il giudice competente non lo so.", "È possibile chiedere i danni?",
                                      "E la caparra?"]
    assert intake.questions("Assisto Alfa: quali azioni può intraprendere?") == ["quali azioni può intraprendere?"]
    assert intake.questions("Alfa ha venduto un'auto a Beta.") == []
    assert "questions" in intake.detect("Alfa ha venduto un'auto a Beta. È possibile chiedere i danni?", model=False)


def _plan(ctx):
    from src.rag import lawyer_method as M
    ctx["candidates"] = M.thesis_candidates(ctx)
    return M.clean_method({}, ctx["candidates"])


def test_the_parere_opens_with_the_answers_before_section_1():
    from src.rag import lawyer_method as M
    ctx = _ctx(_theft("provato", "da provare"))
    ctx["asked"] = ["Quale strategia difensiva?", "Entro quando si può proporre l'appello?"]
    out = M.compose([("Profili processuali", "Il termine per l'appello è di quindici giorni dal deposito della "
                                             "sentenza (art. 585 c.p.p.).")], ctx, _plan(ctx), (), {})
    assert out.index(M.ANSWERS_TITLE) < out.index(f"**1. {M.title('sintesi')}**")
    got = M.answers_of(out)
    assert [q for q, _ in got] == ctx["asked"]
    assert got[0][1].startswith("La linea principale è:")                     # strategy: from the selection
    assert "quindici giorni" in got[1][1]                                     # procedure: from §8
    assert all(2 <= len([x for x in M._SENT_SPLIT.split(a) if x.strip()]) <= 4 for _, a in got)
    assert "se vuoi" not in out.lower()


def test_an_answer_with_no_source_says_so_never_a_guess():
    from src.rag import lawyer_method as M
    ctx = _ctx(_theft("provato", "da provare"))
    q = "La norma si applica ai minori?"
    ctx["asked"] = [q]
    out = M.compose([("", "Testa."), (q, "La norma richiede la violenza sulla persona.")], ctx, _plan(ctx), (), {})
    assert M.answers_of(out) == [(q, M.NO_SOURCE_ANSWER)]


def test_section_9_no_longer_appends_the_questions():
    from src.rag import lawyer_method as M
    ctx = _ctx(_theft("provato", "da provare"))
    q = "È possibile la messa alla prova?"
    out = M.compose([(q, "Si veda il rito.")], ctx, _plan(ctx), (), {})
    assert f"**{q}**" not in dict(M.answer_sections(out))[M.title("conclusioni")]
    assert [x for x, _ in M.answers_of(out)] == [q]


def test_the_nightly_checks_every_question_is_answered_and_grounded():
    from src.rag import lawyer_method as M
    nightly = _script("nightly_sim")
    ctx = _ctx(_theft("provato", "da provare"))
    ctx["asked"] = ["Quale strategia difensiva?"]
    out = M.compose([("", "Testa.")], ctx, _plan(ctx), (), {})
    state = {"intake": {"questions": ctx["asked"]}, "output_sources": ctx["output_sources"]}
    assert nightly.questions_check(out, state)["ok"] is True
    missing = nightly.questions_check(out, {**state, "intake": {"questions": ctx["asked"] + ["E poi?"]}})
    assert missing["ok"] is False and "2 questions, 1 answers" in missing["why"]


# --- 2. more than one client -------------------------------------------------------------------------------------------------

CO_DEFENDANTS = ("Tizio e Caio sono imputati nello stesso processo: Tizio del reato di furto (art. 624 c.p.) per aver "
                 "sottratto una bicicletta dal cortile di Sempronio; Caio del reato di ricettazione (art. 648 c.p.) per "
                 "averla acquistata sapendo che era rubata. Sono il difensore di Tizio e di Caio.")


def test_another_clients_offence_never_enters_a_clients_charge():
    from src.rag import posture
    tizio = posture.client_text(CO_DEFENDANTS, {"name": "Tizio", "others": ["Caio"]})
    caio = posture.client_text(CO_DEFENDANTS, {"name": "Caio", "others": ["Tizio"]})
    assert posture.charge_labels(tizio) == ["624 c.p."] and posture.charge_labels(caio) == ["648 c.p."]
    assert posture.client_text(CO_DEFENDANTS, {"name": "Tizio", "others": []}) == CO_DEFENDANTS


def test_the_clients_are_read_from_the_first_read_or_the_users_words(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    assert [c["name"] for c in flow.clients_of({"case": CO_DEFENDANTS})] == ["Tizio", "Caio"]
    read = {"clients": [{"name": "Alfa S.r.l.", "role": "attore"}, {"name": "Beta S.r.l.", "role": "attore"}]}
    assert [c["name"] for c in flow.clients_of({"read": read, "case": ""})] == ["Alfa S.r.l.", "Beta S.r.l."]
    assert flow.clients_of({"case": "Assisto Tizio."}) == []
    from src.rag import case_read
    got = case_read.clean({"clienti": [{"nome": "Tizio", "ruolo": "imputato"}, {"nome": "Caio"}],
                           "ipotesi": [{"linea": "principale", "tesi": "x", "istituti": ["furto"]}]})
    assert [c["name"] for c in got["clients"]] == ["Tizio", "Caio"]
    single = case_read.clean({"cliente": {"nome": "Tizio"}, "ipotesi": [{"linea": "principale", "tesi": "x"}]})
    assert [c["name"] for c in single["clients"]] == ["Tizio"]


def _co_records():
    return {"624 c.p.": {"rubrica": "Furto", "requisites": [_req(1, "si impossessa della cosa mobile altrui"),
                                                           _req(2, "sottraendola a chi la detiene")]},
            "648 c.p.": {"rubrica": "Ricettazione", "requisites": [_req(3, "acquista cose provenienti da un delitto"),
                                                                  _req(4, "al fine di procurare a sé un profitto")]}}


def _client_ctx(monkeypatch, name, others):
    import copy
    from src.rag import interactive_flow as flow, posture
    monkeypatch.setattr(flow, "case_text", lambda s: CO_DEFENDANTS)
    state = {"records": copy.deepcopy(_co_records()), "client_scope": {"name": name, "others": others}}
    posture.mark_sides(state, IMPUTATO)
    return _ctx(state["records"], IMPUTATO, CO_DEFENDANTS, name), state


def test_each_co_defendant_gets_its_own_charge_and_candidates(monkeypatch):
    from src.rag import lawyer_method as M
    tizio, st = _client_ctx(monkeypatch, "Tizio", ["Caio"])
    caio, sc = _client_ctx(monkeypatch, "Caio", ["Tizio"])
    assert st["records"]["624 c.p."]["side"] == "threat" and st["records"]["648 c.p."]["side"] == "weapon"
    assert sc["records"]["648 c.p."]["side"] == "threat" and sc["records"]["624 c.p."]["side"] == "weapon"
    assert tizio["assessment"]["article"] == "624 c.p." and caio["assessment"]["article"] == "648 c.p."
    cands = M.thesis_candidates(tizio)
    # no line for Tizio admits Caio's conduct: the containment line, if any, is on Tizio's own charge
    assert not any(c["kind"] == "contenimento" and "648" in " ".join(c["poggia_su"]) for c in cands)
    assert not any(c.get("article") == "648 c.p." and c["kind"] in ("contenimento", "claim_offence") for c in cands)


def test_the_merged_parere_has_one_section_1_per_client_and_the_different_sections_per_client(monkeypatch):
    from src.rag import lawyer_method as M
    tizio, _ = _client_ctx(monkeypatch, "Tizio", ["Caio"])
    caio, _ = _client_ctx(monkeypatch, "Caio", ["Tizio"])
    items = [("Tizio", tizio, _plan(tizio)), ("Caio", caio, _plan(caio))]
    log = {}
    out = M.compose_clients([("", "Testa.")], items, (), log)
    by = dict(M.answer_sections(out))
    s1 = by[M.title("sintesi")]
    assert "**Nell'interesse di Tizio**" in s1 and "**Nell'interesse di Caio**" in s1
    s6 = by[M.title("soluzioni")]
    assert "Nell'interesse di Tizio" in s6 and M.title("soluzioni") not in log["shared_sections"]
    assert "**Nell'interesse di Caio**" in by[M.title("conclusioni")]
    assert [c["ok"] for c in M.structure_checks(out) if c["check"] in ("sections", "summary_max_lines")] == [True, True]
    assert tizio["output_sources"]["element_ids"]


def test_two_claimants_with_one_claim_share_the_sections_and_say_so():
    from src.rag import lawyer_method as M
    recs = {"1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon",
                          "requisites": [_req(1, "l'inadempimento del debitore")]}}
    case = "Alfa e Beta hanno prestato insieme 10.000 euro a Gamma, che non li restituisce."
    a = _ctx(recs, ATTORE, case, "Alfa")
    b = _ctx(recs, ATTORE, case, "Beta")
    log = {}
    out = M.compose_clients([("", "Testa.")], [("Alfa", a, _plan(a)), ("Beta", b, _plan(b))], (), log)
    by = dict(M.answer_sections(out))
    assert "**Nell'interesse di Alfa**" in by[M.title("sintesi")]            # §1 always per client
    shared = M.SHARED_NOTE.format(names="Alfa e Beta")
    assert M.title("norme") in log["shared_sections"] and shared in by[M.title("norme")]


# --- 3. repealed articles ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7.", "reference"),
    ("(abrogato)", "reference"),
    ("Soppresso.", "reference"),
    ("Articolo abrogato dalla L. 24 febbraio 2006, n. 85.", "reference"),
    ("[Abrogato dall'art. 3 L. 1/1999] Il testo previgente puniva chi offendeva l'onore.", "head_note"),
    ("Chiunque cagiona ad altri una lesione personale è punito.", ""),
    ("Le disposizioni del secondo comma sono abrogate dalla legge.", ""),
])
def test_the_repeal_is_read_from_the_text(text, kind):
    from src.rag import article_text
    assert article_text.repeal_kind(text) == kind
    assert article_text.is_abrogated(text) is bool(kind)


def test_a_status_property_marks_the_article_repealed_and_the_note_names_the_law():
    from src.rag import article_text
    assert article_text.row_abrogated({"s": {"plain_text": "Chiunque …", "abrogato": True}})
    assert article_text.row_abrogated({"s": {"plain_text": "Chiunque …", "status": "Abrogato"}})
    assert not article_text.row_abrogated({"s": {"plain_text": "Chiunque …", "status": "vigente"}})
    assert article_text.repeal_note("Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7.") == \
        "Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7"


def test_a_repealed_article_is_never_the_charge(monkeypatch):
    from src.rag import interactive_flow as flow, posture
    monkeypatch.setattr(flow, "case_text", lambda s: "Il PM contesta ad Alfa l'art. 594 c.p. e l'art. 612 c.p.")
    state = {"records": {"594 c.p.": {"rubrica": "Ingiuria", "requisites": [_req(1, "offende l'onore")]},
                         "612 c.p.": {"rubrica": "Minaccia", "requisites": [_req(2, "minaccia ad altri un danno")]}},
             "abrogated": ["594 c.p."]}
    out = posture.mark_sides(state, IMPUTATO)
    assert out["charged"] == ["612 c.p."] and state["records"]["594 c.p."]["side"] == "weapon"
    rows = [{"_article": ("594", "c.p."), "s": {"plain_text": "Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7."}},
            {"_article": ("612", "c.p."), "s": {"plain_text": "Chiunque minaccia ad altri un ingiusto danno …"}}]
    assert flow.abrogated_labels(rows) == ["594 c.p."]


def test_a_fetched_repealed_charge_gets_no_record(monkeypatch):
    from src.rag import interactive_flow as flow, issue_records, posture
    state = {"records": {}, "analysis": {}, "read": {}}
    monkeypatch.setattr(flow, "case_text", lambda s: "Il PM contesta ad Alfa l'art. 594 c.p.")
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a: pytest.fail("a repealed row decomposed"))
    row = {"_article": ("594", "c.p."), "s": {"plain_text": "Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7."}}
    link, _ = flow.decompose_records(state, IMPUTATO, [], set(), fetch=lambda keys: [row])
    assert link == [] and state["records"] == {} and state["abrogated"] == ["594 c.p."]


def test_the_general_parts_reentry_skips_a_repealed_article():
    from src.rag import interactive_flow as flow
    row = {"_article": ("62", "c.p."), "s": {"plain_text": "(abrogato)"}}
    rows, refs = flow.general_applications({"records": {}}, {"cenni": [row]}, [], [], set())
    assert not any(r.get("_article") == ("62", "c.p.") for r in rows)


def test_the_audit_script_selftest():
    audit = _script("repeal_audit")
    audit.selftest()


# --- 4. alternative elements ------------------------------------------------------------------------------------------------

def test_an_or_conduct_is_one_element():
    from src.rag import rule_quote
    text = "Chiunque, con violenza o minaccia, costringe altri a fare, tollerare od omettere qualche cosa è punito."
    assert rule_quote.offence_elements(text) == ["con violenza o minaccia",
                                                 "costringe altri a fare, tollerare od omettere qualche cosa"]
    assert rule_quote.penal_clauses(text) == rule_quote.offence_elements(text)


def test_a_category_of_victims_is_one_element_with_branches():
    from src.rag import rule_quote
    text = ("Chiunque maltratta una persona della famiglia o comunque convivente, o una persona sottoposta alla sua "
            "autorità per ragioni di educazione, istruzione, cura, vigilanza o custodia, è punito con la reclusione.")
    got = rule_quote.penal_clauses(text)
    assert len(got) == 1 and "istruzione, cura, vigilanza o custodia" in got[0]
    assert rule_quote.branches(got[0])[:2] == ["maltratta una persona della famiglia", "comunque convivente"]


def test_a_two_branch_remedy_and_a_lettered_alternative_list_are_one_element():
    from src.rag import rule_quote
    assert rule_quote.branches("chiedere l'adempimento o la risoluzione del contratto") == \
        ["chiedere l'adempimento", "la risoluzione del contratto"]
    text = ("Il creditore può agire quando il debitore: a) non adempie entro il termine pattuito; ovvero b) dichiara per "
            "iscritto di non voler adempiere.")
    got = rule_quote.requisites(text)
    assert len(got) == 1 and "non adempie" in got[0] and "dichiara per iscritto" in got[0]


def test_the_splitters_never_split_inside_a_branch_list_and_keep_a_means_clause_apart():
    from src.rag import rule_quote
    text = ("Chiunque, senza l'autorizzazione di cui all'articolo 17, coltiva, produce o fabbrica sostanze "
            "stupefacenti è punito con la reclusione.")
    assert rule_quote.penal_clauses(text) == ["senza l'autorizzazione di cui all'articolo 17",
                                              "coltiva, produce o fabbrica sostanze stupefacenti"]


def test_branch_grading_any_integrated_and_excluded_only_if_all():
    from src.rag import issue_records
    assert issue_records.branch_combine(["provato", "non risulta"]) == "provato"
    assert issue_records.branch_combine(["da provare", "non risulta"]) == "da provare"
    assert issue_records.branch_combine(["non risulta", "non risulta"]) == "non risulta"


def _sent(i, text):
    return {"id": f"S{i}", "text": text, "source": "caso", "certainty": "accertato", "verbatim": True}


def test_settle_one_branch_contradicted_another_supported():
    from src.rag import issue_records
    by = {"S1": _sent(1, "Alfa non ha usato alcuna violenza."), "S2": _sent(2, "Alfa ha rivolto gravi minacce a Beta.")}
    req = {"id": "R1", "kind": "requisito", "text": "con violenza o minaccia", "facts_status": "da provare",
           "branches": ["con violenza", "minaccia"], "link": {"supporting": ["S2"], "contradicting": ["S1"]}}
    issue_records.settle(req, by)
    assert req["status"] == "provato" and req["branch"] == "minaccia"
    alone = {**req, "link": {"supporting": [], "contradicting": ["S1"]}}
    issue_records.settle(alone, by)
    assert alone["status"] == "da provare" and alone["branch_status"]["con violenza"] == "non risulta"


def test_demotion_never_for_one_contradicted_branch_while_another_holds(monkeypatch):
    from src.rag import issue_records
    rows = [{"_article": ("610", "c.p."), "_grade": 2, "s": {}}]
    monkeypatch.setattr(issue_records, "distinctive", lambda rows, analysis: {
        "610 c.p.": {"comma": 1, "text": "con violenza o minaccia"}})
    monkeypatch.setattr(issue_records, "sentences", lambda case: [_sent(1, "Alfa non ha usato alcuna violenza."),
                                                                  _sent(2, "Alfa ha parlato con Beta.")])

    def call(topic, reqs, sents):
        return {r["id"]: {"sostiene": [], "contraddice": ["S1"], "manca": ""} for _, r in reqs}
    log = issue_records.demote_contradicted(rows, "x", {}, call=call)
    assert log["demoted"] == [] and rows[0]["_grade"] == 2


def test_the_sussunzione_prints_the_branch_that_applies():
    from src.rag import lawyer_method as M
    rec = {"requisites": [_req(1, "con violenza o minaccia", branch="minaccia", supporting=["S1"])]}
    rows = M.subsumption_rows(rec, {"S1": {"id": "S1", "text": "Alfa minaccia Beta."}})
    assert "alternativa che ricorre: «minaccia»" in rows[0][0]


def test_the_nightly_alternative_elements_check():
    nightly = _script("nightly_sim")
    good = {"records": {"610 c.p.": {"requisites": [{"id": "R1", "status": "provato",
                                                    "branch_status": {"a": "provato", "b": "non risulta"}}]}}}
    bad = {"records": {"610 c.p.": {"requisites": [{"id": "R1", "status": "non risulta",
                                                   "branch_status": {"a": "da provare", "b": "non risulta"}}]}}}
    assert nightly.alternatives_check(good)["ok"] is True and nightly.alternatives_check(bad)["ok"] is False


# --- 5. role unknown ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, rito, role", [
    ("La società ha ricevuto una diffida dal fornitore Beta, che chiede il pagamento di 9.000 euro.", "civile", "convenuto"),
    ("Alla ditta di Alfa è stata notificata una multa del Comune per l'insegna.", "civile", "convenuto"),
    ("Tizio è accusato di aver danneggiato l'auto del vicino.", "penale", "imputato"),
    ("Alfa vuole recuperare il prezzo della merce consegnata a Beta.", "civile", "attore"),
    ("Beta non ha mai pagato le fatture che Alfa gli ha emesso.", "civile", "attore"),
    ("Tizio è stato aggredito all'uscita dal bar da uno sconosciuto.", "penale", "persona_offesa"),
])
def test_six_phrasings_three_resisting_three_claiming(text, rito, role):
    from src.rag import intake
    got, why, reading = intake.derive_role(text, rito)
    assert got == role and why and reading == ""


def test_an_entity_takes_the_side_the_text_gives_it():
    from src.rag import intake
    text = ("Sono il difensore della società. Alla società è stato notificato un decreto ingiuntivo da Beta. "
            "Beta vuole recuperare il credito.")
    assert intake.derive_role(text, "civile")[0] == "convenuto"
    text = "Assisto l'impresa edile. L'impresa vuole recuperare il compenso dei lavori. Il committente è stato citato."
    assert intake.derive_role(text, "civile")[0] == "attore"


def test_neither_side_is_a_reading_said_in_one_line(monkeypatch):
    from src.rag import interaction_copy as C, interactive_flow as flow, intake
    role, why, reading = intake.derive_role("Alfa e Beta hanno firmato un contratto di fornitura.", "civile")
    assert role == "attore" and reading
    resolved = intake.resolve({"role": intake.UNKNOWN, "rito": "civile", "confidence": {}},
                              {flow.CASE_KEY: "Alfa e Beta hanno firmato un contratto di fornitura."})
    assert resolved["role"] == "attore" and resolved["role_reading"] == reading
    line = flow.role_reading_line({"intake": resolved})
    assert line == C.fill(C.ROLE_READING, role="attore", why=reading) and line.count(".") == 1


def test_an_unknown_role_gets_neither_claimant_nor_defendant_wording():
    from src.rag import lawyer_method as M
    recs = {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(1, "il fatto doloso o colposo")]}}
    unknown = {"rito": "civile", "family": "civile", "role": "sconosciuto", "stage": "pre_filing"}
    cands = M.thesis_candidates(_ctx(recs, unknown))
    assert not any(c["tesi"].startswith(("Fondare la domanda", "Domanda riconvenzionale")) or c["kind"] == "contenimento"
                   for c in cands)


def test_the_models_convenuto_rule_c5_still_holds(monkeypatch):
    from src.rag import intake
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {"role": "sconosciuto", "rito": "civile",
                                                               "deliverable": "parere", "stage": "pre_filing"})
    monkeypatch.setattr(intake, "client_rule", lambda *a, **k: {"role": "", "why": "", "name": ""})
    text = ("Il 3 marzo 2026 Alfa ha consegnato a Beta un quadro d'autore pattuendo un prezzo di 9.000 euro. Beta "
            "ha ritirato il quadro e da allora non ha pagato nulla. Quali azioni può intraprendere Alfa contro Beta?")
    got = intake.detect(text)
    assert got["role"] == "attore" and "agisce" in got["why"]["role"]


# --- 6. keyword hits across institutions ------------------------------------------------------------------------------------

def _row(n, code, rubrica, **tags):
    return {"_article": (n, code), "s": {"titolo_articolo": rubrica}, **tags}


def test_a_word_naming_two_institutions_in_two_books_keeps_only_the_cases():
    from src.rag import topic_groups
    rows = [_row("624", "c.p.", "Furto", _multi_signal=True),
            _row("62", "c.p.", "Circostanze attenuanti comuni", _first_read="riduzione della pena"),
            _row("2872", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione della pena"),
            _row("625", "c.p.", "Circostanze aggravanti", _first_read="furto aggravato")]
    log = {}
    kept = topic_groups.matter_gate(rows, [], log)
    assert [r["_article"][0] for r in kept] == ["624", "62", "625"]
    assert log["dropped"] == [{"article": "2872 c.c.", "matter": "c.c. VI responsabilità patrimoniale, privilegi e "
                                                                 "ipoteche", "institution": "riduzione della pena"}]


def test_a_whole_name_hit_and_a_named_article_stay_and_no_matter_drops_nothing():
    from src.rag import topic_groups
    rows = [_row("2872", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione delle ipoteche"),
            _row("624", "c.p.", "Furto", _multi_signal=True)]
    assert len(topic_groups.matter_gate(rows, [])) == 2                     # the institution's whole name
    hit = [_row("2872", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione della pena"),
           _row("624", "c.p.", "Furto", _multi_signal=True)]
    assert len(topic_groups.matter_gate(hit, ["2872 c.c."])) == 2            # named by the user
    alone = [_row("2872", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione della pena")]
    assert len(topic_groups.matter_gate(alone, [])) == 1                    # no matter known: nothing dropped
    assert topic_groups.matter_of(("2872", "c.c.")).startswith("c.c. VI") and topic_groups.matter_of(("405", "c.p.p.")) == ""


# --- 7. the nightly ---------------------------------------------------------------------------------------------------------

def test_the_nightly_has_a_two_client_cell_and_prints_per_client_counts():
    nightly = _script("nightly_sim")
    cells = nightly.cells()
    two = [c for c in cells if c.get("clients")]
    assert len(two) == 1 and two[0]["clients"] == ["Tizio", "Caio"] and two[0]["cell"]["role"] == "imputato"
    rows = [{"id": "x", "checks": [], "thesis_counts": {"offered": 3},
             "thesis_counts_by_client": {"Tizio": {"offered": 2, "chosen": "T1"}, "Caio": {"offered": 4}}}]
    text = "\n".join(nightly.counts_lines(rows))
    assert "| x / Tizio | 2 |" in text and "| x / Caio | 4 |" in text
    data = json.loads((HERE / "data" / "nightly_cells.json").read_text())
    assert "Sempronio" in two[0]["text"] and len(data["cells"]) == 6
