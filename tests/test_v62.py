"""v62: the posture consumers -- intake's rito, role, stage, prior decisions and requested blocks used downstream: the
main line's act and goal from data/posture_acts.json (the default cell: today's plan exactly); each record a weapon or a
threat for the client, a threat's status read from the defence; the requisites counted only for fattispecie; the
procedural section by stage (data/stage_procedure.json), «Parti da convenire» only for civile/attore/pre_filing, the
competenza by art. 5 c.p.p. first; the requested blocks as sections in the asked order. Synthetic records per cell; the
model and the graph are stubbed."""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, intake, issue_records, posture, procedural_facts, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA
NORMS = flow.ANSWER_TITLES[1]


def _rec(rubrica, *statuses, side=None):
    reqs = [{"kind": "requisito", "comma": 1, "text": f"elemento {i}", "status": st} for i, st in enumerate(statuses, 1)]
    rec = {"rubrica": rubrica, "requisites": reqs}
    rec["verdict"] = issue_records.verdict(rec)
    if side:
        rec["side"] = side
    return rec


def _posture(rito, role, stage, **extra):
    return {"rito": rito, "family": posture.family(rito), "role": role, "stage": stage,
            "entry": posture.entry_for(rito, role, stage), "prior_decisions": [], "querela_stated": False, **extra}


# --- the act and the goal by posture ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("cell, act", [
    (("penale", "parte_civile", "impugnazione"), "appello ai soli effetti civili (art. 576 c.p.p.)"),
    (("penale", "imputato", "indagini"), "linea difensiva"),
    (("penale", "imputato", "primo_grado"), "linea difensiva"),
    (("civile", "convenuto", "esecuzione"), "opposizione all'esecuzione o agli atti esecutivi (artt. 615 e 617 c.p.c.)"),
    (("esecuzione", "terzo", "esecuzione"), "opposizione di terzo all'esecuzione (art. 619 c.p.c.)"),
])
def test_the_act_of_a_posture_cell(cell, act):
    entry = posture.entry_for(*cell)
    assert entry["act"] == act and entry["goal"]


@pytest.mark.parametrize("cell", [("civile", "attore", "pre_filing"), ("penale", "sconosciuto", "indagini"),
                                  ("sconosciuto", "attore", "pre_filing"), ("penale", "imputato", "sconosciuto")])
def test_the_default_cell_and_an_unknown_field_have_no_entry(cell):
    assert posture.entry_for(*cell) is None


def test_the_plan_takes_the_act_and_the_goal_and_the_default_cell_keeps_todays_plan():
    plan = {"parte_assistita": "Tizio", "posizione": "", "obiettivo": "vincere", "questione_centrale": "",
            "risposte": [], "subordinate": [], "punti_deboli": [], "questioni": []}
    assert posture.apply_plan(plan, _posture("civile", "attore", "pre_filing")) is plan
    moved = posture.apply_plan(plan, _posture("penale", "parte_civile", "impugnazione"))
    assert moved["atto"].startswith("appello ai soli effetti civili") and moved["obiettivo"].startswith("l'accertamento")
    text = ptw._plan_text(moved)
    assert "Linea principale: appello ai soli effetti civili (art. 576 c.p.p.)" in text
    assert "Linea principale" not in ptw._plan_text(plan)


def test_the_posture_reads_intake_and_falls_back_to_the_client_logic():
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"client_side": "Tizio (imputato)"}},
                flow.CASE_KEY: "Caio presenta querela contro Tizio."}
    read = intake.detect("Tizio è stato condannato in primo grado per truffa. Il candidato, assunte le vesti del "
                         "difensore di Tizio, prospetti il mezzo di impugnazione.", model=False)
    got = posture.of({"intake": read}, analysis)
    assert (got["rito"], got["role"], got["stage"]) == ("penale", "imputato", "impugnazione")
    assert got["entry"]["act"].startswith("appello contro la sentenza di condanna") and got["querela_stated"]
    old = posture.of({}, analysis)                      # no intake (an old state): role from the client logic, no stage
    assert old["role"] == "imputato" and old["stage"] == posture.UNKNOWN and old["entry"] is None


# --- the records' sides, the threats' statuses -----------------------------------------------------------------------------

def _sided(role, records, case="Il PM contesta a Tizio l'art. 584 c.p.", alternatives=()):
    state = {"records": records, "text": case, "read": {"alternatives": list(alternatives)},
             "analysis": {flow.CASE_KEY: case}}
    posture.mark_sides(state, {"role": role})
    return {k: r["side"] for k, r in records.items()}


def test_the_charge_sentences_name_their_articles():
    assert posture.charge_labels("Il PM contesta a Tizio l'art. 584 c.p. Tizio nega.") == ["584 c.p."]
    listed = posture.charge_labels("Si costituisce parte civile nel processo in cui Sempronio è imputato di lesione "
                                   "personale aggravata, a norma degli articoli 582 - 583, comma 1, numero 1 – 585 – "
                                   "577, comma 1, numero 4 – 61, numero 4 – del codice penale.")
    assert listed == ["582 c.p.", "583 c.p.", "585 c.p.", "577 c.p.", "61 c.p."]


def test_an_imputato_faces_the_charge_and_keeps_the_alternatives():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO),
               "589 c.p.": _rec("Omicidio colposo", DA_PROVARE),
               "62 c.p.": _rec("Circostanze attenuanti comuni", DA_PROVARE)}
    sides = _sided("imputato", records, alternatives=[{"name": "omicidio colposo", "rubricated": "Omicidio colposo"}])
    assert sides == {"584 c.p.": "threat", "589 c.p.": "weapon", "62 c.p.": "weapon"}
    assert records["584 c.p."]["side_why"] == "nominata nell'addebito o nella domanda del caso"


def test_a_victim_wields_every_article_and_a_convenuto_faces_the_claim():
    records = {"582 c.p.": _rec("Lesione personale", PROVATO)}
    assert _sided("parte_civile", records) == {"582 c.p.": "weapon"}
    claim = {"2043 c.c.": {**_rec("Risarcimento per fatto illecito", PROVATO), "relied_by": "domanda di Caio"},
             "1227 c.c.": _rec("Concorso del fatto colposo del creditore", DA_PROVARE)}
    assert _sided("convenuto", claim, case="Caio cita in giudizio Tizio.") == {"2043 c.c.": "threat",
                                                                               "1227 c.c.": "weapon"}


def test_a_threats_status_reads_from_the_defence_never_the_count():
    rec = _rec("Omicidio preterintenzionale", PROVATO, DA_PROVARE, side="threat")
    state = {"posture": {"family": "penale", "role": "imputato"}}
    line = flow.status_sentence("584 c.p.", rec, state)
    assert line == ("Per la difesa: elemento da contestare: «elemento 1»; elemento non provato dall'accusa: "
                    "«elemento 2».")
    assert "requisiti provati" not in line and "applicabile" not in line
    civil = flow.status_sentence("2043 c.c.", rec, {"posture": {"family": "civile", "role": "convenuto"}})
    assert "non provato dalla controparte" in civil
    weapon = flow.status_sentence("582 c.p.", _rec("Lesione personale", PROVATO), {})
    assert weapon == C.fill(C.ARTICLE_STATUS, verdict="applicabile", proved="1", total="1")


@pytest.mark.parametrize("label, rubrica, counted", [
    ("61 c.p.", "Circostanze aggravanti comuni", False), ("43 c.p.", "Elemento psicologico del reato", False),
    ("576 c.p.p.", "Impugnazione della parte civile", False), ("167 c.p.c.", "Comparsa di risposta", False),
    ("29 c.p.a.", "Azione di annullamento", False), ("1470 c.c.", "Nozione", False),
    ("584 c.p.", "Omicidio preterintenzionale", True), ("659 c.p.", "Disturbo delle occupazioni", True),
    ("2043 c.c.", "Risarcimento per fatto illecito", True),
])
def test_requisites_are_counted_only_for_fattispecie(label, rubrica, counted):
    assert issue_records.counted(label, rubrica) is counted
    line = flow.status_sentence(label, _rec(rubrica, PROVATO), {})
    assert (line is not None) is counted


def test_the_body_and_the_citations_count_only_fattispecie_and_read_threats_from_the_defence():
    draft = (f"**2. {NORMS}**\n\n**Art. 584 c.p. — Omicidio preterintenzionale**\n\nL'accusa ricostruisce il fatto.\n\n"
             "**Art. 61 c.p. — Circostanze aggravanti comuni**\n\nL'aggravante riguarda la condotta di Sempronio.\n\n"
             "**Art. 2043 c.c.**\n\nIl danno è risarcibile.\n\n**5. Conclusioni**\n\nX.")
    state = {"posture": {"family": "penale", "role": "imputato"}, "client": {"name": "Sempronio"},
             "records": {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO, DA_PROVARE, side="threat"),
                         "61 c.p.": _rec("Circostanze aggravanti comuni", PROVATO, side="threat"),
                         "2043 c.c.": _rec("Risarcimento per fatto illecito", PROVATO)}}
    text, rows = flow.ledger_out(draft, state)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "**Art. 584 c.p. — Omicidio preterintenzionale**\nPer la difesa: elemento da contestare" in norms
    assert "**Art. 61 c.p. — Circostanze aggravanti comuni**\n\nL'aggravante riguarda la condotta di Sempronio." \
        in norms                                                                                       # no status
    assert "**Art. 2043 c.c.**\nEsito sui fatti del caso: applicabile (requisiti provati: 1 su 1)." in norms
    assert norms.count("requisiti provati") == 1
    heads = {r["article"]: r for r in rows if r["requisite"] is None and r["comma"] is None}
    assert heads["61 c.p."]["status"] == issue_records.APPLICATION
    assert not [r for r in rows if r["article"] == "61 c.p." and r["requisite"]]
    assert heads["584 c.p."]["side"] == "threat"


def test_the_writers_schede_mark_applications_and_threats_with_their_rules():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO, DA_PROVARE, side="threat"),
               "576 c.p.p.": _rec("Impugnazione della parte civile", PROVATO)}
    block = issue_records.writer_block(records, [("584", "c.p."), ("576", "c.p.p.")], [], penal=True)
    from src.rag import writer_rules
    assert "[1] art. 584 c.p. — Omicidio preterintenzionale — " + writer_rules.THREAT_MARK["penale"] in block
    assert "  - elemento da contestare: «elemento 1»" in block and "elemento non provato dall'accusa" in block
    assert "[2] art. 576 c.p.p. — Impugnazione della parte civile — " + writer_rules.APPLICATION_MARK in block
    assert writer_rules.APPLICATION_RULE in block and writer_rules.THREAT_RULE["penale"] in block
    assert "esito: applicabile" not in block


# --- the procedural section by stage -----------------------------------------------------------------------------------------

def _analysis(stage, rito="penale", role="imputato", **posture_extra):
    facts = {"584": {"querela": False, "procedibilita": "d'ufficio", "pena": "reclusione da dieci a diciotto anni",
                     "competenza": "tribunale in composizione collegiale (art. 33-bis c.p.p.)"},
             "582": {"querela": True, "procedibilita": "a querela della persona offesa (co. 2)",
                     "querela_termine": "tre mesi (art. 124 c.p.)", "remissione": "remissibile (art. 152 c.p.)",
                     "pena": "reclusione da sei mesi a tre anni",
                     "competenza": "tribunale in composizione monocratica (art. 33-ter c.p.p.)"}}
    return {"frame": {"areas": [rito]}, "store_offences_only": True,
            "offences": [("584", "Omicidio preterintenzionale"), ("582", "Lesione personale")], "offence_facts": facts,
            "fact_signals": {"facts": {}, "terms": [], "tools": [],
                             "defendants": [{"name": "Caio", "why": "autore del fatto"}]},
            "posture": {**_posture(rito, role, stage), **posture_extra}}


def test_pre_filing_and_indagini_keep_todays_procedibilita_and_competenza_by_art_5():
    blocks = dict(render_computed.blocks(_analysis("indagini")))
    procedure = blocks[render_computed.PROCEDURE_TITLE]
    assert "Omicidio preterintenzionale (art. 584 c.p.): procedibile d'ufficio; competenza: Corte d'assise (art. 5 " \
           "c.p.p.)" in procedure
    assert "procedibile a querela della persona offesa (co. 2); querela entro tre mesi" in procedure
    assert "Parti da convenire" not in blocks                    # never outside civile / attore / pre_filing


def test_the_trial_drops_procedibilita_when_a_querela_is_stated_and_adds_its_blocks():
    blocks = render_computed.blocks(_analysis("primo_grado", querela_stated=True))
    titles = [t for t, _ in blocks]
    procedure = dict(blocks)[render_computed.PROCEDURE_TITLE]
    assert "procedibile" not in procedure and "querela entro" not in procedure
    assert "competenza: Corte d'assise (art. 5 c.p.p.)" in procedure
    assert titles[-2:] == ["Riti alternativi", "Costituzione di parte civile"]
    assert "art. 74 c.p.p." in dict(blocks)["Costituzione di parte civile"]
    kept = dict(render_computed.blocks(_analysis("primo_grado")))[render_computed.PROCEDURE_TITLE]
    assert "procedibile a querela" in kept                       # no querela stated: today's line


def test_the_appeal_is_its_closed_list_never_procedibilita_nor_parties():
    blocks = render_computed.blocks(_analysis("impugnazione", role="parte_civile"))
    titles = [t for t, _ in blocks]
    assert titles == ["Legittimazione", "Atto", "Termini", "Rinnovazione dell'istruzione dibattimentale",
                      "Effetti sul giudizio civile"]
    text = "\n".join(b for _, b in blocks)
    assert "art. 576 c.p.p." in text and "art. 593 c.p.p." not in text
    assert dict(blocks)["Atto"].startswith("Appello ai soli effetti civili (art. 576 c.p.p.)")
    # v66: the «Termini» block's terms are the store's -- here not read: no number of the list's own
    assert "Art. 585 c.p.p.: testo non letto" in text and "Da verificare: il caso non dice la data della lettura" in text
    assert "quindici" not in dict(blocks)["Termini"]
    assert "art. 603, co. 1, c.p.p." in text and "art. 651 c.p.p." in text and "art. 652 c.p.p." in text
    assert "procedibil" not in text.lower() and "Parti da convenire" not in titles
    dated = dict(render_computed.blocks(_analysis("impugnazione", role="imputato", prior_decisions=[
        {"court": "Tribunale di Milano", "outcome": "condanna", "date": "3 marzo 2024"}])))
    assert "art. 593 c.p.p." in dated["Legittimazione"] and "3 marzo 2024" in dated["Termini"]


def test_the_execution_block_and_the_civil_appeal():
    execution = render_computed.blocks(_analysis("esecuzione", rito="esecuzione", role="convenuto"))
    assert [t for t, _ in execution] == ["Opposizioni all'esecuzione"]
    assert "art. 615 c.p.c." in execution[0][1] and "art. 617 c.p.c." in execution[0][1]
    appeal = dict(render_computed.blocks(_analysis("impugnazione", rito="civile", role="attore")))
    assert "Art. 325 c.p.c." in appeal["Termini"] and appeal["Atto"].startswith("Appello (artt. 339 ss. c.p.c.)")


@pytest.mark.parametrize("cell, allowed", [
    (("civile", "attore", "pre_filing"), True), (("lavoro", "attore", "pre_filing"), True),
    (("civile", "convenuto", "pre_filing"), False), (("civile", "attore", "primo_grado"), False),
    (("penale", "persona_offesa", "pre_filing"), False), (("amministrativo", "attore", "pre_filing"), False),
    (("civile", "sconosciuto", "pre_filing"), True), (("sconosciuto", "sconosciuto", "sconosciuto"), True),
])
def test_parties_to_sue_only_for_civile_attore_pre_filing(cell, allowed):
    assert render_computed.parties_allowed(_posture(*cell)) is allowed
    blocks = dict(render_computed.blocks({**_analysis("pre_filing"), "posture": _posture(*cell), "offences": []}))
    assert ("Parti da convenire" in blocks) is allowed


def test_without_a_posture_the_section_is_todays():
    analysis = _analysis("indagini")
    analysis.pop("posture")
    assert "Parti da convenire" in dict(render_computed.blocks(analysis))


@pytest.mark.parametrize("number, pena, court", [
    ("584", "reclusione da dieci a diciotto anni", procedural_facts.ASSISE),
    ("579", "reclusione da sei a quindici anni", procedural_facts.ASSISE),
    ("575", "reclusione non inferiore ad anni ventuno", procedural_facts.ASSISE),
    ("628", "reclusione da quattro a dieci anni", "STORE"),
    ("582", "reclusione da sei mesi a tre anni", "STORE"),
])
def test_the_competenza_reads_art_5_cpp_before_33_bis_and_33_ter(number, pena, court):
    got = procedural_facts.court(number, {"pena": pena, "competenza": "STORE"})
    assert got == court


# --- the requested blocks ---------------------------------------------------------------------------------------------------

BODY = (f"**1. {flow.ANSWER_TITLES[0]}**\n\nLa difesa.\n\n**2. {NORMS}**\n\nLe norme.\n\n"
        f"**3. {flow.ANSWER_TITLES[4]}**\n\nL'esito.")
SEMPRONIO_BLOCKS = [{"kind": "cenni", "subject": "omicidio preterintenzionale"},
                    {"kind": "cenni", "subject": "nesso causale"}, {"kind": "parere", "subject": "strategia difensiva"},
                    {"kind": "alternative", "subject": "possibili reati alternativi"}]


def _rows():
    return [{"_article": ("584", "c.p."), "s": {"titolo_articolo": "Omicidio preterintenzionale",
                                                 "plain_text": "Chiunque, con atti diretti a commettere uno dei "
                                                               "delitti preveduti dagli articoli 581 e 582, cagiona "
                                                               "la morte di un uomo, è punito con la reclusione."}}]


def test_the_requested_blocks_are_sections_in_the_asked_order(monkeypatch):
    rows = _rows() + [{"_article": ("589", "c.p."), "s": {"titolo_articolo": "Omicidio colposo", "plain_text": "…"}}]
    state = {"rows": rows, "read": {"alternatives": [{"name": "omicidio colposo", "rubricated": "Omicidio colposo",
                                                      "why": "la morte non voluta"}]}}
    jobs_seen, fetched = [], []

    def write(jobs):
        jobs_seen.extend(jobs)
        return [(t, "[1] La morte non era voluta." if C.ALTERNATIVES_SECTION in t else f"Testo di {t.split(':', 1)[1]}.")
                for t, *_ in jobs]

    def fetch(refs):            # v63: the general concepts' articles read by number
        fetched.extend(refs)
        names = {"40": "Rapporto di causalità", "41": "Concorso di cause", "43": "Elemento psicologico del reato"}
        return [{"_article": ref, "s": {"titolo_articolo": names[ref[0]], "plain_text": "…"}} for ref in refs]
    out = flow.requested_sections(state, BODY, SEMPRONIO_BLOCKS, write, {"frame": {"areas": ["penale"]}}, "Il caso.",
                                  search=lambda names: [], fetch=fetch, index=lambda subject, code: [])
    titles = [t for t, _ in answer_ledger.sections_of(out)]
    assert titles == ["Cenni: Omicidio preterintenzionale", "Cenni: Nesso causale", C.ALTERNATIVES_SECTION,
                      flow.ANSWER_TITLES[0], NORMS, flow.ANSWER_TITLES[4]]           # v68: the requested blocks first
    assert out.startswith("**1. Cenni: Omicidio preterintenzionale**") and f"**3. {C.ALTERNATIVES_SECTION}**" in out
    assert fetched == [("43", "c.p."), ("40", "c.p."), ("41", "c.p.")]      # 584 from the case's rows, by its rubrica
    cenni = {t: m for t, _, m, _ in jobs_seen}
    assert "NORMA: art. 43 c.p. — Elemento psicologico del reato" in cenni["0:Cenni: Omicidio preterintenzionale"]
    assert "NORMA: art. 584 c.p. — Omicidio preterintenzionale" in cenni["0:Cenni: Omicidio preterintenzionale"]
    assert "NORMA: art. 40 c.p. — Rapporto di causalità" in cenni["1:Cenni: Nesso causale"]
    assert "[1] art. 589 c.p. — Omicidio colposo (omicidio colposo): la morte non voluta" in \
        cenni[f"3:{C.ALTERNATIVES_SECTION}"]
    assert "**Art. 589 c.p. — Omicidio colposo**\nLa morte non era voluta." in out


def test_an_ungrounded_cenni_cites_nothing_and_no_alternative_is_said_so():
    from src.rag import writer_rules
    jobs_seen = []

    def write(jobs):
        jobs_seen.extend(jobs)
        return [(t, "Paragrafo.") for t, *_ in jobs]
    out = flow.requested_sections({"read": {}}, BODY, [{"kind": "cenni", "subject": "ruolo della vittima"},
                                                       {"kind": "alternative", "subject": "alternative"}],
                                  write, {}, "Il caso.", search=lambda names: [], fetch=lambda refs: [],
                                  index=lambda subject, code: [])
    assert writer_rules.CENNI_UNGROUNDED in jobs_seen[0][1] and jobs_seen[0][2] == ""
    sections = answer_ledger.sections_of(out)
    assert [t for t, _ in sections][0] == "Cenni: Ruolo della vittima"
    assert [t for t, _ in sections][1] == C.ALTERNATIVES_SECTION          # v68: the requested blocks, then the body
    assert [t for t, _ in sections][2] == flow.ANSWER_TITLES[0]
    assert dict(sections)[C.ALTERNATIVES_SECTION].strip() == C.ALTERNATIVES_NONE


def test_no_cenni_nor_alternative_leaves_the_answer_as_it_is():
    out = flow.requested_sections({}, BODY, [{"kind": "parere", "subject": "x"}, {"kind": "atto", "subject": "y"}],
                                  lambda jobs: pytest.fail("no section to write"), {}, "Il caso.")
    assert out == BODY


def test_the_single_shot_writes_the_blocks_around_the_body_and_the_posture_moves_the_plan(monkeypatch):
    import test_v60k
    state, seen = test_v60k._single(monkeypatch)
    plans = []
    real = ptw.write_sections

    def capture(plan, jobs, *a, **k):
        plans.append(plan)
        if any(str(t).split(":", 1)[0].isdigit() for t, *_ in jobs):
            return [(t, "Cenni scritti.") for t, *_ in jobs]
        return real(plan, jobs, *a, **k)
    monkeypatch.setattr(ptw, "write_sections", capture)
    monkeypatch.setattr(flow, "first_read_search", lambda *a, **k: ([], {}))
    read = intake.detect("Sempronio, imputato di lesioni ai danni di Tizio, è stato assolto in primo grado. Tizio si "
                         "era costituito parte civile. Premessi brevi cenni sulla servitù, il candidato, assunte le "
                         "vesti del legale di Tizio, rediga parere.", model=False)
    assert (read["rito"], read["role"], read["stage"]) == ("penale", "parte_civile", "impugnazione")
    text = flow.single_shot("Il caso.", intake=read)["answer"]
    assert plans[0]["atto"].startswith("appello ai soli effetti civili") and "risarcimento" in plans[0]["obiettivo"]
    titles = [t for t, _ in answer_ledger.sections_of(text) if t]
    assert titles[0] == "Cenni: Servitù" and titles[1] == flow.ANSWER_TITLES[0]
    assert state["posture"]["entry"]["act"].startswith("appello ai soli effetti civili")


def test_the_single_shot_default_cell_keeps_todays_plan(monkeypatch):
    import test_v60k
    state, seen = test_v60k._single(monkeypatch)
    plans = []
    real = ptw.write_sections

    def capture(plan, jobs, *a, **k):
        plans.append(plan)
        return real(plan, jobs, *a, **k)
    monkeypatch.setattr(ptw, "write_sections", capture)
    read = intake.detect("Sono l'avvocato di Leonardo, che vuole recuperare il passaggio sul fondo di Domitilla. "
                         "Quali azioni?", model=False)
    assert (read["role"], read["stage"]) == ("attore", "pre_filing")
    flow.single_shot("Il caso.", intake=read)
    assert "atto" not in plans[0] and plans[0]["obiettivo"] == ""
    assert "Linea principale" not in ptw._plan_text(plans[0])


# --- the intake's parte civile -----------------------------------------------------------------------------------------------

def test_the_persona_offesa_who_joined_the_trial_is_the_parte_civile():
    read = intake.detect("Tizio presenta denuncia e querela nei confronti di Sempronio. Si costituisce successivamente "
                         "parte civile nel processo penale in cui Sempronio è imputato. Il candidato, assunte le vesti "
                         "del legale di Tizio, rediga parere.", model=False)
    assert read["role"] == "parte_civile" and read["stage"] == "primo_grado"
