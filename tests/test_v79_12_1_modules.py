"""v79.12.1 -- module activation, unrelated code articles, mediation; synthetic cases, reads and rows only:

1. a module switches on only from the first read's institution (its name and rubricated name), a module article the
   request names, or the user's words -- never a claim sentence or a damage word alone (no «responsabilità sanitaria»
   for a slap taken to the hospital, no «successioni» for a homicide pleaded as legittima difesa with the heirs' claim);
   each active module carries the signal that switched it on (counter modules_active); the retrieval stage and the
   answer stage call active_modules with the same input (the read's institutions, not names only); the «disciplina
   speciale non è nell'archivio» line once per answer (§1, never §9) and only for a module that fits
2. the Capo-scope gate on every book of the codes: an article of a Titolo / Capo outside the case's institutions enters
   grading only through an institution match or the first read (2873 c.c. for a penalty clause, 890 / 892 c.c. in an
   administrative case, 699 / 518-bis / 610 c.p. in a homicide); counter code_articles_out_of_scope
3. mediation «obbligatoria» only for an art. 5, co. 1 d.lgs. 28/2010 matter (data/mediation_matters.json); appalto,
   preliminare di vendita and the like «facoltativa»
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _inst(name, rubricated="", claim=""):
    return {"name": name, "rubricated": rubricated, "claim": claim}


def _read(*hyps):
    return {"hypotheses": [{"line": line, "institutions": list(insts), "rubricated": list(rub), "claim": claim}
                           for line, insts, rub, claim in hyps]}


def _ids(case, institutions=()):
    from src.rag import lawyer_method as M
    return [m["id"] for m in M.active_modules(M.module_input(case, institutions))]


SLAP = ("Durante una lite al bar Tizio ha dato uno schiaffo a Caio. Caio è stato portato in ospedale, dove il medico del "
        "pronto soccorso gli ha dato cinque giorni di prognosi; ora chiede il risarcimento del danno e delle spese "
        "della clinica privata in cui ha fatto i controlli.")
SLAP_READ = [_inst("percosse", "Percosse", "risarcimento del danno per le cure in ospedale e in clinica"),
             _inst("lesioni personali", "Lesione personale", "responsabilità per le cure della struttura sanitaria")]
HOMICIDE = ("Tizio ha ucciso Caio con un coltello durante una rissa; sostiene di aver agito per legittima difesa. I figli "
            "di Caio, suoi eredi, chiedono il risarcimento del danno iure successionis e iure proprio.")
HOMICIDE_READ = [_inst("omicidio", "Omicidio", "gli eredi chiedono il danno iure hereditatis per la successione"),
                 _inst("legittima difesa", "Difesa legittima", "la difesa legittima esclude la punibilità")]


# --- 1. module activation ---------------------------------------------------------------------------------------------------

def test_a_slap_taken_to_the_hospital_never_switches_on_responsabilita_sanitaria():
    assert "responsabilita_sanitaria" not in _ids(SLAP, SLAP_READ)       # claim words of the read: never
    assert "responsabilita_sanitaria" not in _ids(SLAP)                  # the user's damage words alone: never
    assert "responsabilita_sanitaria" not in _ids(SLAP, ["percosse"])    # a string institution: a name


def test_a_homicide_never_switches_on_successioni():
    assert "successioni" not in _ids(HOMICIDE, HOMICIDE_READ)            # «legittima difesa» is no legittima
    assert "successioni" not in _ids(HOMICIDE)                           # «iure successionis», «eredi»: damage words
    assert "successioni" not in _ids("Tizio ha agito per difesa legittima.")


@pytest.mark.parametrize("case, institutions, module, signal", [
    ("Tizia è stata operata al ginocchio; il dolore non passa.", [_inst("responsabilità medica")],
     "responsabilita_sanitaria", "istituto"),
    ("Tizia si è rivolta alla clinica Alfa.", [_inst("responsabilità della struttura sanitaria")],
     "responsabilita_sanitaria", "istituto"),
    ("Dopo un errore medico durante l'intervento chirurgico Tizia ha perso la funzionalità della mano.", [],
     "responsabilita_sanitaria", "parole dell'utente"),
    ("Il ricovero in ospedale è seguito a una diagnosi tardiva del tumore.", [],
     "responsabilita_sanitaria", "parole dell'utente"),
    ("Si chiede se Tizia possa agire ai sensi dell'art. 7 della legge n. 24/2017.", [_inst("inadempimento")],
     "responsabilita_sanitaria", "articolo"),
    ("Tizio chiede la restituzione dell'auto ex art. 1809 c.c.", [_inst("inadempimento")], "comodato", "articolo"),
    ("Tizio, legittimario, chiede la riduzione delle donazioni che ledono la sua quota di legittima.", [],
     "successioni", "parole dell'utente"),
    ("Tizio, coerede, chiede la divisione dell'eredità.", [], "successioni", "parole dell'utente"),
])
def test_a_module_switches_on_from_the_institution_a_named_article_or_the_users_words(case, institutions, module,
                                                                                    signal):
    from src.rag import lawyer_method as M
    mods = {m["id"]: m for m in M.active_modules(M.module_input(case, institutions))}
    assert module in mods and mods[module]["signal"] == signal and mods[module]["signal_words"]
    assert M.modules_active(list(mods.values()))[module].startswith(f"{signal}: «")


def test_a_general_article_named_switches_on_no_module():
    from src.rag import lawyer_method as M
    case = "Tizio chiede i danni ex art. 2043 c.c. e art. 1218 c.c."
    assert _ids(case) == []
    mod = next(m for m in M.institution_modules()["modules"] if m["id"] == "responsabilita_sanitaria")
    assert M.named_module_articles(mod, case) == []


def test_the_retrieval_and_the_answer_stage_read_the_same_input():
    from src.rag import family_gate, interactive_flow as flow, lawyer_method as M
    read = _read(("principale", ["percosse"], ["Percosse"], "le cure in ospedale e la clinica"),
                 ("subordinata", ["responsabilità medica"], [], "l'errore del medico del pronto soccorso"))
    assert M.read_institutions(read) == flow.read_institutions(read)     # dicts, not names only
    assert all(isinstance(x, dict) and "rubricated" in x for x in M.read_institutions(read))
    retrieval = family_gate.context(SLAP, "", flow.read_institutions(read))
    answer = [m["id"] for m in M.active_modules({"case": SLAP, "institutions": M.read_institutions(read),
                                                 "records": {}, "posture": {}})]
    assert retrieval["modules"] == answer == ["responsabilita_sanitaria"]    # the subordinate's institution
    assert retrieval["modules_active"] == {"responsabilita_sanitaria": "istituto: «responsabilità medica»"}
    quiet = _read(("principale", ["percosse"], ["Percosse"], "le cure in ospedale e la clinica"))
    assert family_gate.context(SLAP, "", flow.read_institutions(quiet))["modules"] == [] == \
        [m["id"] for m in M.active_modules({"case": SLAP, "institutions": M.read_institutions(quiet)})]
    ctx = {"institutions": M.read_institutions(read)}
    assert M.action_subject(ctx) == "percosse"                            # the no-action line's subject: the name


def test_the_missing_laws_line_only_for_a_module_that_fits():
    from src.rag import lawyer_method as M
    known = ["Codice Civile 2026"]
    civil = {"case": SLAP, "institutions": SLAP_READ, "posture": {"rito": "civile", "family": "civile"}}
    assert M.missing_laws(civil, known=known) == {}                        # no module: no line
    medical = {"case": "Dopo un errore medico durante l'intervento Tizia chiede i danni.", "institutions": [],
               "posture": {"rito": "civile", "family": "civile"}}
    got = M.missing_laws(medical, known=known)
    assert got["module"] == "responsabilita_sanitaria" and got["leggi"] == ["L. 24/2017"]
    assert got["signal"] == "parole dell'utente"
    penal = {**medical, "posture": {"rito": "penale", "family": "penale", "role": "imputato"}}
    assert M.criminal(penal["posture"])
    assert M.missing_laws(penal, known=known) == {}                        # the user's words in a penal matter: no fit
    named = {**penal, "institutions": [_inst("responsabilità medica")]}
    assert M.missing_laws(named, known=known)["module"] == "responsabilita_sanitaria"   # the read's institution: fits
    other = {**medical, "institutions": [_inst("comodato")]}               # the read's institution is another family
    assert M.module_fits({"id": "responsabilita_sanitaria", "signal": "parole dell'utente"}, other)


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "dibattimento"}


def test_the_missing_laws_line_is_printed_once_per_answer():
    from src.rag import lawyer_method as M
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [{"id": "R1", "comma": 1, "kind": "requisito", "status": "provato",
                                         "text": "atti diretti a percuotere o ledere"},
                                        {"id": "R2", "comma": 1, "kind": "requisito", "status": "da provare",
                                         "text": "la morte come conseguenza degli atti"}]}}
    sents = [{"id": "S1", "text": "Sempronio colpisce Caio.", "certainty": "accertato"}]
    ctx = M.prepare({"records": recs, "sentences": sents}, IMPUTATO, [], "Nel 2025 Sempronio colpisce Caio.",
                    "Sempronio", "", list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="t79121")
    ctx["missing_laws"] = {"istituto": "responsabilità sanitaria", "leggi": ["L. 24/2017"],
                           "module": "responsabilita_sanitaria"}
    cands = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "medio", "argomento": "a"} for c in cands}}, cands)
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    line = M.missing_laws_line(ctx)
    assert answer.count(line.rstrip(".")) == 1
    by = dict(M.answer_sections(answer))
    assert line.rstrip(".") in by["Sintesi della soluzione"]
    assert line.rstrip(".") not in by["Conclusioni e valutazione del rischio"]


def test_the_counter_modules_active_from_the_answer_else_the_retrieval():
    from src.rag import interactive_flow as flow
    retrieval = {"family_gate": {"modules_active": {"comodato": "istituto: «comodato»"}}}
    assert flow.case_counters(retrieval)["modules_active"] == {"comodato": "istituto: «comodato»"}
    answer = {**retrieval, "block_counters": {"modules_active": {"comodato": "articolo: «1809 c.c.»"}}}
    assert flow.case_counters(answer)["modules_active"] == {"comodato": "articolo: «1809 c.c.»"}
    assert flow.case_counters({})["modules_active"] == {} and flow.case_counters({})["code_articles_out_of_scope"] == 0


# --- 2. unrelated code articles -----------------------------------------------------------------------------------------

def _row(number, code, rubrica, **extra):
    return {"_article": (number, code), "s": {"titolo_articolo": rubrica}, **extra}


def _labels(rows):
    return [f"{r['_article'][0]} {r['_article'][1]}" for r in rows]


def test_code_units_every_book():
    from src.rag import topic_groups as T
    assert T.code_unit(("518", "c.p.")).endswith("Titolo VIII – delitti contro l'economia pubblica, l'industria e il "
                                                 "commercio")
    assert "VIII-bis" in T.code_unit(("518-bis", "c.p.")) and "VI-bis" in T.code_unit(("452-quater", "c.p."))
    assert "Capo I" in T.code_unit(("575", "c.p.")) and "Capo III" in T.code_unit(("610", "c.p."))
    assert T.code_unit(("575", "c.p.")) == T.code_unit(("584", "c.p.")) == T.code_unit(("593-bis", "c.p."))
    assert "Libro III" in T.code_unit(("699", "c.p.")) and "Libro VI, Titolo III" in T.code_unit(("2873", "c.c."))
    assert T.code_unit(("890", "c.c.")) == T.code_unit(("892", "c.c.")) and "proprietà" in T.code_unit(("890", "c.c."))
    for general in (("52", "c.p."), ("62-bis", "c.p."), ("1218", "c.c."), ("1384", "c.c."), ("2043", "c.c."),
                    ("2697", "c.c."), ("2932", "c.c."), ("2946", "c.c."), ("1490", "c.c."),   # 1490: its Capo gate
                    ("444", "c.p.p."), ("21-nonies", "Legge sul Procedimento Amministrativo 2026")):
        assert T.code_unit(general) is None, general


PENALTY_ROWS = [
    _row("1382", "c.c.", "Effetti della clausola penale", _first_read="clausola penale"),
    _row("1384", "c.c.", "Riduzione della penale", _first_read="riduzione della penale"),
    _row("2873", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione della penale", _multi_signal=True),
    _row("1218", "c.c.", "Responsabilità del debitore"),
]


def test_a_penalty_clause_case_keeps_2873_out():
    from src.rag import topic_groups as T
    log = {}
    kept = T.code_scope_gate(PENALTY_ROWS, ["clausola penale", "riduzione della penale"], log=log)
    assert _labels(kept) == ["1382 c.c.", "1384 c.c.", "1218 c.c."]
    assert log["dropped"] == [{"article": "2873 c.c.", "unit": T.code_unit(("2873", "c.c.")),
                               "hit": "prima lettura, per parola"}]


def test_an_administrative_case_keeps_the_distances_out():
    from src.rag import topic_groups as T
    rows = [_row("21-nonies", "Legge sul Procedimento Amministrativo 2026", "Annullamento d'ufficio",
                 _first_read="annullamento d'ufficio"),
            _row("890", "c.c.", "Distanze per fabbriche e depositi nocivi o pericolosi", _multi_signal=True),
            _row("892", "c.c.", "Distanze per gli alberi", _first_read="permesso di costruire"),
            _row("2043", "c.c.", "Risarcimento per fatto illecito")]
    log = {}
    kept = T.code_scope_gate(rows, ["permesso di costruire", "annullamento d'ufficio"], log=log)
    assert _labels(kept) == ["21-nonies Legge sul Procedimento Amministrativo 2026", "2043 c.c."]
    assert [x["article"] for x in log["dropped"]] == ["890 c.c.", "892 c.c."] and log["units"] == []
    neighbours = T.code_scope_gate(rows, ["distanze nelle costruzioni", "distanze per gli alberi"])   # the distances'
    assert "890 c.c." in _labels(neighbours) and "892 c.c." in _labels(neighbours)                  # own case: kept


HOMICIDE_ROWS = [
    _row("575", "c.p.", "Omicidio", _first_read="omicidio"),
    _row("52", "c.p.", "Difesa legittima", _first_read="legittima difesa"),
    _row("584", "c.p.", "Omicidio preterintenzionale", _multi_signal=True),
    _row("699", "c.p.", "Porto abusivo di armi", _first_read="omicidio con arma", _multi_signal=True),
    _row("518-bis", "c.p.", "Furto di beni culturali", _multi_signal=True),
    _row("610", "c.p.", "Violenza privata"),
]


def test_a_homicide_keeps_699_518_bis_and_610_out():
    from src.rag import topic_groups as T
    log = {}
    kept = T.code_scope_gate(HOMICIDE_ROWS, ["omicidio", "Omicidio", "legittima difesa", "Difesa legittima"], log=log)
    assert _labels(kept) == ["575 c.p.", "52 c.p.", "584 c.p."]
    assert [x["article"] for x in log["dropped"]] == ["699 c.p.", "518-bis c.p.", "610 c.p."]
    assert log["units"] == [T.code_unit(("575", "c.p."))]
    # through an institution match or the user's naming it: in
    with_610 = T.code_scope_gate(HOMICIDE_ROWS, ["omicidio", "violenza privata"])
    assert "610 c.p." in _labels(with_610) and "699 c.p." not in _labels(with_610)
    assert "699 c.p." in _labels(T.code_scope_gate(HOMICIDE_ROWS, ["omicidio"], keep=["699 c.p."]))
    # a rule's anchor stays, never widening the scope
    log = {}
    anchored = T.code_scope_gate(HOMICIDE_ROWS, ["omicidio"], pass_only=["699 c.p."], log=log)
    assert "699 c.p." in _labels(anchored) and "610 c.p." not in _labels(anchored)
    assert log["kept_by"]["699 c.p."] == "ancora di una regola"


def test_an_institution_without_a_matching_row_falls_back_on_its_read_rows_sharing_a_word():
    from src.rag import topic_groups as T
    rows = [_row("575", "c.p.", "Omicidio", _first_read="omicidio volontario aggravato"),
            _row("577", "c.p.", "Altre circostanze aggravanti. Ergastolo"),
            _row("610", "c.p.", "Violenza privata")]
    log = {}
    kept = T.code_scope_gate(rows, ["omicidio volontario aggravato"], log=log)
    assert _labels(kept) == ["575 c.p.", "577 c.p."] and [x["article"] for x in log["dropped"]] == ["610 c.p."]


def test_no_first_read_institution_no_gate():
    from src.rag import topic_groups as T
    log = {}
    assert T.code_scope_gate(HOMICIDE_ROWS, [], log=log) == HOMICIDE_ROWS and log["off"]


def test_the_flow_reads_the_first_read_the_users_articles_and_the_modules_actions():
    from src.rag import interactive_flow as flow
    read = _read(("principale", ["omicidio"], ["Omicidio"], "il porto dell'arma e la violenza privata"))
    got = flow.code_scope(HOMICIDE_ROWS, "Tizio ha ucciso Caio. Si valuti anche l'art. 699 c.p.", read)
    assert _labels(got["_rows"]) == ["575 c.p.", "52 c.p.", "584 c.p.", "699 c.p."]
    assert [x["article"] for x in got["dropped"]] == ["518-bis c.p.", "610 c.p."]
    separation = _read(("principale", ["separazione personale"], ["Separazione giudiziale"], "l'assegno"))
    rows = [_row("151", "c.c.", "Separazione giudiziale", _first_read="separazione personale"),
            _row("433", "c.c.", "Persone obbligate"),                      # the famiglia module's action
            _row("2873", "c.c.", "Riduzione delle ipoteche")]
    got = flow.code_scope(rows, "Tizia chiede la separazione e il mantenimento.", separation)
    assert _labels(got["_rows"]) == ["151 c.c.", "433 c.c."] and got["modules"]["famiglia"].startswith("istituto")
    assert flow.code_scope(rows, "x", None)["off"]                         # the read failed: off


def test_the_counter_code_articles_out_of_scope():
    from src.rag import interactive_flow as flow
    state = {"code_scope": {"dropped": [{"article": "2873 c.c."}, {"article": "699 c.p."}]}}
    assert flow.case_counters(state)["code_articles_out_of_scope"] == 2


# --- 3. mediation -----------------------------------------------------------------------------------------------------------

def _analysis(contracts, areas=(), store_only=True, **facts):
    return {**({"store_offences_only": True} if store_only else {}), "frame": {"areas": list(areas)},
            "fact_signals": {"facts": {"contracts": list(contracts), **facts}}}


APPALTO = {"type": "contratto di appalto per la ristrutturazione del bagno", "parties": ["Alfa s.r.l.", "Tizio"],
           "clauses": ["pagamento con bonifico bancario a stati di avanzamento", "polizza fideiussoria dell'impresa"]}
PRELIMINARE = {"type": "contratto preliminare di compravendita immobiliare", "parties": ["Tizio", "Caio"],
               "clauses": ["caparra confirmatoria versata con assegno bancario", "rogito subordinato al mutuo"]}


@pytest.mark.parametrize("contract, areas, matter", [
    (APPALTO, ["contratti"], "appalto"),
    (APPALTO, ["contratti", "diritti_reali"], "appalto"),
    (PRELIMINARE, ["contratti", "diritti_reali"], "preliminare di vendita"),
    ({"type": "compravendita di un'auto usata"}, ["contratti"], "compravendita"),
])
def test_appalto_and_preliminare_print_facoltativa(contract, areas, matter):
    from src.rag import render_computed as R
    for store_only in (True, False):
        analysis = _analysis([contract], areas, store_only)
        got = R.mediation_of(analysis)
        assert not got["required"] and got["optional"] and got["subject"] == matter
        assert got["text"].startswith("Mediazione facoltativa") and "obbligatoria" not in got["text"]
        blocks = dict(R.blocks(analysis))
        assert blocks["Mediazione"].startswith("Mediazione facoltativa")


def test_the_legacy_facts_reading_never_makes_a_preliminare_mandatory():
    from src.rag import fact_signals, render_computed as R
    facts = {"contracts": [PRELIMINARE]}
    got = fact_signals.mediation(facts, [{"id": "x", "area": "diritti_reali"}])
    assert not got["required"] and got["optional"] and got["subject"] == "preliminare di vendita"
    legacy = {"fact_signals": {"facts": facts, "mediation": {"required": True, "subject": "diritti reali",
                                                             "text": "Mediazione obbligatoria (x)."}},
              "topic_refs": [["1158", "c.c."]]}
    assert R.mediation_of(legacy)["optional"]


@pytest.mark.parametrize("contracts, areas, matter, facts", [
    ([{"type": "contratto di locazione ad uso abitativo"}], ["locazione"], "locazione", {}),
    ([{"type": "comodato d'uso gratuito dell'appartamento"}], ["contratti"], "comodato", {}),
    ([{"type": "polizza assicurativa sulla casa"}], ["contratti"], "contratti assicurativi", {}),
    ([], ["diritti_reali"], "condominio", {"requests": ["annullamento della delibera condominiale"]}),
    ([APPALTO, {"type": "contratto di locazione del capannone"}], ["contratti"], "locazione", {}),   # listed wins
])
def test_an_article_5_matter_prints_obbligatoria(contracts, areas, matter, facts):
    from src.rag import render_computed as R
    analysis = _analysis(contracts, areas, **facts)
    got = R.mediation_of(analysis)
    assert got["required"] and not got.get("optional") and got["subject"] == matter
    assert got["text"].startswith("Mediazione obbligatoria (art. 5, co. 1, d.lgs. 28/2010)")
    assert dict(R.blocks(analysis))["Mediazione"].startswith("Mediazione obbligatoria")


def test_a_clause_word_never_makes_a_typed_contract_mandatory():
    from src.rag import render_computed as R
    seen = R.case_matter(_analysis([APPALTO], ["contratti"]))
    assert seen["matter"] == "" and seen["optional"] == "appalto" and seen["by"].startswith("contratto «contratto di "
                                                                                             "appalto")
    untyped = R.case_matter(_analysis([{"what": "contratto", "clauses": ["mutuo bancario"]}], ["contratti"]))
    assert untyped["matter"] == "contratti bancari e finanziari"        # no type known: the old reading, unchanged
