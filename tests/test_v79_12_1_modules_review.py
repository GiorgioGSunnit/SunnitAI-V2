"""v79.12.1 (review) -- module activation, unrelated code articles, mediation; synthetic cases, reads and rows only:

1. a criminal narrative's damage words never switch a civil module on from the user's words: the heirs' claim
   («a titolo ereditario», «i coeredi … parte civile», «si trasmette agli eredi per successione», «erede legittima»,
   «la difesa era legittima») no successioni, a fall after a slap no custodia, a pedestrian crossing no servitù -- those
   words are the modules' «context_patterns» (from the first read's institution only); the module's own words still
   switch it on (a testamento, the coeredi's divisione, a vase fallen from a balcony, the diritto di passaggio)
2. a module article named in its abbreviated form («art. 7 l. 241/1990», «art. 2 d.lgs. 23/2015», «art. 31 d.P.R.
   380/2001») switches its module on
3. an incidental article the user cites never replaces the module the user's words name (the brief's order: the
   institution, the user's words, a named article)
4. a surgical error or a hospital infection described plainly switches responsabilità sanitaria on from the user's
   words; a slap taken to the hospital still does not
5. the code-scope fallback: a first-read row only when its rubrica names nothing the institution does not -- one shared
   word is a keyword hit (2873 «Riduzione delle ipoteche» for «riduzione equitativa della penale», 699 «Porto abusivo
   di armi» for an «omicidio aggravato dall'uso di armi»)
6. mediation: an art. 5 matter the client's request names stays «obbligatoria» beside an appalto / compravendita /
   trasporto contract; a professional's contract is a contratto d'opera (in the list); a «never» matter (reati, rito
   del lavoro) prints no mediazione, facoltativa neither
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


def _inst(name, rubricated="", claim=""):
    return {"name": name, "rubricated": rubricated, "claim": claim}


def _active(case, institutions=()):
    from src.rag import lawyer_method as M
    return {m["id"]: m["signal"] for m in M.active_modules(M.module_input(case, list(institutions)))}


OMICIDIO = [_inst("omicidio", "Omicidio")]
PERCOSSE = [_inst("percosse", "Percosse")]
STRADALE = [_inst("omicidio stradale", "Omicidio stradale")]


# --- 1. a criminal narrative's damage words -------------------------------------------------------------------------------

@pytest.mark.parametrize("case, read, module", [
    ("Tizio ha ucciso Caio. Tizio sostiene la legittima difesa. Gli eredi chiedono il risarcimento del danno a titolo "
     "ereditario.", OMICIDIO, "successioni"),
    ("Tizio ha ucciso Caio. I coeredi della vittima si costituiscono parte civile.", OMICIDIO, "successioni"),
    ("Tizio ha ucciso Caio. Il diritto al risarcimento si trasmette agli eredi per successione.", OMICIDIO,
     "successioni"),
    ("Tizio ha ucciso Caio. Secondo Tizio la difesa era legittima.", OMICIDIO, "successioni"),
    ("Tizio ha ucciso Caio. La vedova, erede legittima, chiede il danno.", OMICIDIO, "successioni"),
    ("Tizio ha dato uno schiaffo a Caio, che è caduto dalla sedia.", PERCOSSE, "custodia"),
    ("Tizio ha investito Caio sul passaggio pedonale.", STRADALE, "servitu"),
])
def test_a_criminal_narratives_damage_words_never_switch_on_a_civil_module(case, read, module):
    assert module not in _active(case, read)
    assert module not in _active(case)                                   # the read failed: the user's words alone


@pytest.mark.parametrize("case, module", [
    ("Mio padre è morto lasciando un testamento olografo che mi esclude.", "successioni"),
    ("I coeredi non trovano un accordo sulla divisione della casa al mare.", "successioni"),
    ("Mia sorella ha ricevuto in donazione la casa: chiedo la reintegrazione della quota di legittima.", "successioni"),
    ("Si è aperta la successione del nonno Sempronio e i beni ereditari sono contesi.", "successioni"),
    ("Un vaso è caduto dal balcone del vicino e ha colpito la mia auto.", "custodia"),
    ("Sono scivolata su una buca del marciapiede comunale e mi sono rotta il polso.", "custodia"),
    ("Il vicino mi nega il diritto di passaggio sul suo fondo per raggiungere la strada.", "servitu"),
])
def test_the_modules_own_words_still_switch_it_on(case, module):
    assert _active(case).get(module) == "parole dell'utente"


def test_a_context_word_still_counts_from_the_first_reads_institution():
    assert _active("Tizio è morto.", [_inst("successione legittima")]).get("successioni") == "istituto"
    assert _active("Il vicino non mi lascia passare.", [_inst("passaggio coattivo")]).get("servitu") == "istituto"


# --- 2. a module article in its abbreviated form ---------------------------------------------------------------------------

@pytest.mark.parametrize("case, module, words", [
    ("Chiedo tutela ai sensi dell'art. 7 l. 24/2017.", "responsabilita_sanitaria", "7 l. 24/2017"),
    ("Agisco ai sensi dell'art. 7 della l. 241/1990.", "amministrativo", "7 l. 241/1990"),
    ("Agisco ai sensi dell'art. 7 l. 241/1990.", "amministrativo", "7 l. 241/1990"),
    ("Agisco ai sensi dell'art. 2 d.lgs. 23/2015.", "licenziamento", "2 d.lgs. 23/2015"),
    ("Agisco ai sensi dell'art. 31 d.P.R. 380/2001.", "amministrativo", "31 d.P.R. 380/2001"),
    ("Agisco ai sensi dell'art. 31 del d.P.R. n. 380/2001.", "amministrativo", "31 d.P.R. 380/2001"),
])
def test_an_abbreviated_special_law_citation_switches_its_module_on(case, module, words):
    from src.rag import lawyer_method as M
    got = M.modules_active(M.active_modules(M.module_input(case, [])))
    assert got == {module: f"articolo: «{words}»"}


def test_a_citation_never_crosses_the_end_of_a_sentence():
    from src.rag import lawyer_method as M
    module = next(m for m in M.institution_modules()["modules"] if m["id"] == "amministrativo")
    assert M.named_module_articles(module, "Vale l'art. 7 del regolamento. Il fondo 241/1990 è altro.") == []


# --- 3. the user's words before an incidental article ----------------------------------------------------------------------

def test_an_incidental_article_never_replaces_the_module_the_users_words_name():
    from src.rag import lawyer_method as M
    read = M.read_institutions({"hypotheses": [{"line": "principale", "institutions": ["sfratto per morosità"],
                                                "rubricated": [], "claim": "convalida di sfratto"}]})
    case = ("Il conduttore non paga i canoni di locazione da sei mesi; inoltre usa il cortile comune in violazione "
            "dell'art. 1102 c.c. Chiedo lo sfratto per morosità.")
    assert _active(case, read) == {"locazione": "parole dell'utente"}
    case = ("Sono stato licenziato senza giusta causa dopo aver segnalato che l'azienda violava l'art. 2087 c.c. sulla "
            "sicurezza; chiedo la reintegrazione.")
    assert _active(case) == {"licenziamento": "parole dell'utente"}
    assert M.MODULE_SIGNALS[:3] == ("istituto", "parole dell'utente", "articolo")


# --- 4. a plainly described medical error -----------------------------------------------------------------------------------

SURGERY = ("Mia madre è stata operata in clinica: durante l'intervento chirurgico il chirurgo le ha reciso un nervo e ora "
           "ha una paralisi permanente. Vogliamo chiedere il risarcimento alla clinica e al chirurgo.")


@pytest.mark.parametrize("case, institutions", [
    (SURGERY, [_inst("responsabilità contrattuale", "", "risarcimento del danno per l'intervento chirurgico in clinica")]),
    (SURGERY, []),
    ("Dopo il ricovero in ospedale ho contratto un'infezione nosocomiale; chiedo i danni all'ASL.", []),
    ("Durante il parto è stata lesionata la spalla del neonato.", []),
    ("Ho subito danni per i quali intendo agire contro l'ospedale.", []),
])
def test_a_plain_medical_error_switches_on_responsabilita_sanitaria(case, institutions):
    assert _active(case, institutions).get("responsabilita_sanitaria") == "parole dell'utente"


def test_a_slap_taken_to_the_hospital_still_never_switches_it_on():
    case = ("Tizio ha dato uno schiaffo a Caio. Caio è stato portato in ospedale, dove il medico del pronto soccorso gli "
            "ha dato cinque giorni di prognosi; ora chiede il risarcimento del danno e delle spese della clinica.")
    assert "responsabilita_sanitaria" not in _active(case, PERCOSSE)
    assert "responsabilita_sanitaria" not in _active(case)


# --- 5. the code-scope fallback ----------------------------------------------------------------------------------------------

def _row(number, code, rubrica, **extra):
    return {"_article": (number, code), "s": {"titolo_articolo": rubrica}, **extra}


def _labels(rows):
    return [f"{r['_article'][0]} {r['_article'][1]}" for r in rows]


def test_one_shared_word_never_brings_a_unit_in():
    from src.rag import topic_groups as T
    rows = [_row("1382", "c.c.", "Effetti della clausola penale", _first_read="clausola penale"),
            _row("1384", "c.c.", "Riduzione della penale", _first_read="riduzione equitativa della penale"),
            _row("2873", "c.c.", "Riduzione delle ipoteche", _first_read="riduzione equitativa della penale")]
    log = {}
    kept = T.code_scope_gate(rows, ["clausola penale", "riduzione equitativa della penale"], log=log)
    assert _labels(kept) == ["1382 c.c.", "1384 c.c."]
    assert [x["article"] for x in log["dropped"]] == ["2873 c.c."]
    rows = [_row("575", "c.p.", "Omicidio", _first_read="omicidio"),
            _row("699", "c.p.", "Porto abusivo di armi", _first_read="omicidio aggravato dall'uso di armi")]
    log = {}
    kept = T.code_scope_gate(rows, ["omicidio", "eccesso di legittima difesa", "omicidio aggravato dall'uso di armi"],
                             log=log)
    assert _labels(kept) == ["575 c.p."] and log["units"] == [T.code_unit(("575", "c.p."))]


def test_the_fallback_keeps_a_rubrica_the_institution_covers():
    from src.rag import topic_groups as T
    rows = [_row("640", "c.p.", "Truffa", _first_read="truffa aggravata ai danni dello Stato"),
            _row("646", "c.p.", "Appropriazione indebita"),
            _row("610", "c.p.", "Violenza privata")]
    log = {}
    kept = T.code_scope_gate(rows, ["truffa aggravata ai danni dello Stato"], log=log)
    assert _labels(kept) == ["640 c.p.", "646 c.p."] and [x["article"] for x in log["dropped"]] == ["610 c.p."]


# --- 6. mediation ------------------------------------------------------------------------------------------------------------

def _analysis(types, areas=(), requests=(), store_only=True):
    return {**({"store_offences_only": True} if store_only else {}), "frame": {"areas": list(areas)},
            "fact_signals": {"facts": {"contracts": [{"type": t} for t in types], "requests": list(requests)}}}


@pytest.mark.parametrize("types, areas, requests, matter", [
    (["compravendita dell'immobile dal padre al fratello"], ["successioni"],
     ["azione di riduzione per lesione della quota di legittima"], "successioni ereditarie"),
    (["contratto di appalto per il rifacimento della facciata condominiale"], ["diritti_reali"],
     ["impugnazione della delibera condominiale che ha approvato l'appalto"], "condominio"),
    (["compravendita della quota ereditaria"], ["successioni"],
     ["divisione dell'immobile ereditato e scioglimento della comunione"], "divisione"),
    (["compravendita del fondo"], ["diritti_reali"],
     ["accertamento dell'usucapione della striscia di terreno confinante"], "diritti reali"),
    (["trasporto in ambulanza"], ["responsabilita_civile"], ["risarcimento per l'errore medico in ospedale"],
     "risarcimento del danno da responsabilità medica e sanitaria"),
])
def test_a_matter_the_request_names_stays_obbligatoria_beside_a_named_contract(types, areas, requests, matter):
    from src.rag import fact_signals, render_computed as R
    for store_only in (True, False):
        analysis = _analysis(types, areas, requests, store_only)
        got = R.mediation_of(analysis)
        assert got["required"] and not got.get("optional") and got["subject"] == matter, store_only
        assert dict(R.blocks(analysis))["Mediazione"].startswith("Mediazione obbligatoria")
    facts = _analysis(types, areas, requests)["fact_signals"]["facts"]
    assert fact_signals.mediation(facts, [])["subject"] == matter
    assert not R.mediation_of(_analysis(types, areas, (), True))["required"]       # no request: facoltativa


def test_the_legacy_reading_keeps_a_fired_matter_the_request_names():
    from src.rag import fact_signals
    facts = {"contracts": [{"type": "contratto di appalto per il rifacimento della facciata"}],
             "requests": ["impugnazione della delibera condominiale"]}
    got = fact_signals.mediation(facts, [{"id": "x", "area": "condominio_parti_comuni"}])
    assert got["required"] and got["subject"] == "condominio"
    got = fact_signals.mediation({"contracts": facts["contracts"]}, [{"id": "x", "area": "condominio_parti_comuni"}])
    assert not got["required"] and got["optional"]                       # the fired rule's area alone: facoltativa


@pytest.mark.parametrize("contract", ["prestazione di servizi professionali di consulenza contabile",
                                      "mandato professionale all'avvocato",
                                      "incarico al commercialista per la dichiarazione dei redditi"])
def test_a_professionals_contract_is_a_contratto_dopera(contract):
    from src.rag import fact_signals, render_computed as R
    got = R.mediation_of(_analysis([contract], ["contratti"]))
    assert got["required"] and got["subject"] == "contratto d'opera"
    assert fact_signals.mediation({"contracts": [{"type": contract}]}, [])["subject"] == "contratto d'opera"


@pytest.mark.parametrize("types, areas, asked", [
    (["compravendita di un'auto usata"], ["penale"], "condanna per la truffa sul contachilometri"),
    (["contratto di agenzia (agente di commercio)"], ["lavoro"], "pagamento delle provvigioni"),
])
def test_a_never_matter_prints_no_mediation_at_all(types, areas, asked):
    from src.rag import render_computed as R
    for store_only in (True, False):
        analysis = _analysis(types, areas, [asked], store_only)
        got = R.mediation_of(analysis)
        assert not got.get("required") and not got.get("optional") and not got.get("text")
        assert "Mediazione" not in dict(R.blocks(analysis))
