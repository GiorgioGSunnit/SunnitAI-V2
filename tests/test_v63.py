"""v63: the review of the v62 sim (sempronio) -- the «cenni» anchored on the codes' general part (never a special-part
article for a general concept, nothing when nothing resolves) and each of their articles an «applicazione» paragraph
in the norms section; the alternatives each «art. N — rubrica» with one sentence, a fragment without its article
dropped; the client's own alternative a weapon whatever the charge sentence names, the threats only the charge and
what the other side relies on; §1 and the conclusions contest only the threat's recorded elements; the procedural
section only for the charge and the alternatives argued, no querela term at the trial; no «(da valutare)» on a
defending client's alternative. Synthetic records; the model and the graph are stubbed."""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, cenni, intake, issue_records, posture, render_computed, writer_rules  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA
NORMS = flow.ANSWER_TITLES[1]
PENAL = {"frame": {"areas": ["penale"]}}
RUBRICHE = {"40": "Rapporto di causalità", "41": "Concorso di cause", "42": "Responsabilità per dolo o per colpa o per "
            "delitto preterintenzionale. Responsabilità obiettiva", "43": "Elemento psicologico del reato",
            "584": "Omicidio preterintenzionale", "609-nonies": "Pene accessorie e altri effetti penali",
            "582": "Lesione personale", "575": "Omicidio", "586": "Morte o lesioni come conseguenza di altro delitto",
            "589": "Omicidio colposo", "581": "Percosse", "572": "Maltrattamenti contro familiari e conviventi"}


def _row(number, code="c.p.", text="…", **tags):
    return {"_article": (number, code), "s": {"titolo_articolo": RUBRICHE.get(number, ""), "plain_text": text}, **tags}


def _fetch(seen=None):
    def fetch(refs):
        if seen is not None:
            seen.extend(refs)
        return [_row(n, c) for n, c in refs]
    return fetch


def _resolve(subjects, rows=(), codes=("c.p.", "c.p.p."), index=None, search=None, fetched=None):
    return cenni.resolve(subjects, list(rows), list(codes), fetch=_fetch(fetched),
                         index=index or (lambda subject, code: []), search=search or (lambda names: []))


def _labels(grounds, subject):
    return [flow._label(flow._key(r["_article"])) for r in grounds[subject]]


def _rec(rubrica, *statuses, **extra):
    reqs = [{"kind": "requisito", "comma": 1, "text": f"elemento {i}", "status": st} for i, st in enumerate(statuses, 1)]
    rec = {"rubrica": rubrica, "requisites": reqs, **extra}
    rec["verdict"] = issue_records.verdict(rec)
    return rec


# --- 1. the cenni anchored on the general part --------------------------------------------------------------------------

def test_a_general_concept_rests_on_its_general_part_articles_never_on_a_special_part_hit():
    fetched = []
    grounds, log = _resolve(["nesso causale", "elemento soggettivo", "omicidio preterintenzionale"],
                            rows=[_row("584"), _row("609-nonies", _first_read="elemento soggettivo")],
                            search=lambda names: [_row("609-nonies")], fetched=fetched)
    assert _labels(grounds, "nesso causale") == ["40 c.p.", "41 c.p."]
    assert _labels(grounds, "elemento soggettivo") == ["42 c.p.", "43 c.p."]          # never 609-nonies, a rubrica hit
    assert _labels(grounds, "omicidio preterintenzionale") == ["43 c.p.", "584 c.p."]   # the subject names 584 itself
    assert log["omicidio preterintenzionale"]["how"] == f"{cenni.GENERAL} + {cenni.NAMED}"
    assert ("584", "c.p.") not in fetched                                                 # the case's row, not re-read


def test_nothing_resolved_with_confidence_cites_nothing():
    grounds, log = _resolve(["ruolo della vittima"], rows=[_row("582")],
                            search=lambda names: [_row("609-nonies"), _row("586")])         # hits that share no name
    assert grounds["ruolo della vittima"] == [] and log["ruolo della vittima"]["how"] == cenni.NONE
    civil, _ = _resolve(["nesso causale"], codes=("c.c.", "c.p.c."))                      # c.p.'s concept, not in scope
    assert civil["nesso causale"] == []


def test_the_index_answers_only_with_general_articles_and_a_name_with_its_rubrica():
    def index(subject, code):
        return [_row("27", "c.p.p."), _row("575")] if code == "c.p.p." else []
    grounds, log = _resolve(["presunzione di non colpevolezza"], index=index)
    assert _labels(grounds, "presunzione di non colpevolezza") == ["27 c.p.p."]           # 575, special part: dropped
    assert log["presunzione di non colpevolezza"]["how"] == cenni.INDEX
    named, _ = _resolve(["percosse"], search=lambda names: [_row("581")])
    assert _labels(named, "percosse") == ["581 c.p."]                                     # not general: its own article


def test_the_cenni_articles_get_an_application_paragraph_in_the_norms_section():
    state = {"records": {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO, side="threat")}}
    grounds = {"cenni: nesso causale": [_row("40"), _row("41")],
               "cenni: omicidio preterintenzionale": [_row("43"), _row("584")]}
    rows, refs = flow.general_applications(state, grounds, [_row("584")], [("584", "c.p.")], set())
    assert refs == [("584", "c.p."), ("40", "c.p."), ("41", "c.p."), ("43", "c.p.")]
    assert {flow._label(flow._key(r["_article"])) for r in rows} == {"584 c.p.", "40 c.p.", "41 c.p.", "43 c.p."}
    records = state["records"]
    assert records["41 c.p."] == {"rubrica": "Concorso di cause", "requisites": [], "applicazione": "cenni: nesso causale"}
    assert "applicazione" not in records["584 c.p."]                              # its own record, untouched
    assert issue_records.block_due(records["41 c.p."]) and issue_records.applied("41 c.p.", records["41 c.p."])
    block = issue_records.writer_block(records, refs, [])
    assert f"[3] art. 41 c.p. — Concorso di cause — {writer_rules.APPLICATION_MARK}" in block
    assert flow.status_sentence("41 c.p.", records["41 c.p."], state) is None
    named = {"records": {"582 c.p.": _rec("Lesione personale", PROVATO)}}
    flow.general_applications(named, {"cenni: percosse": [_row("581")]}, [], [], set())
    assert issue_records.applied("581 c.p.", named["records"]["581 c.p."])     # a special-part cenni: no count either


def _every_paragraph(monkeypatch):
    """The writer stub writes one paragraph per norm of the norms job's order (NORME NELL'ORDINE DEI PARAGRAFI)."""
    import re
    real = ptw.write_sections

    def write(plan, jobs, *a, **k):
        out = []
        for (t, x), job in zip(real(plan, jobs, *a, **k), jobs):
            if t == NORMS:
                order = re.findall(r"(?m)^(\d+)\. art\. (\S+ \S+)", job[2].split("NORME NELL'ORDINE DEI PARAGRAFI:")[-1])
                x = "\n\n".join(f"[{n}] Come opera l'art. {label} per Leonardo." for n, label in order)
            out.append((t, x))
        return out
    monkeypatch.setattr(ptw, "write_sections", write)


def test_the_single_shot_resolves_the_cenni_once_and_writes_their_applications(monkeypatch):
    import test_v60k
    state, seen = test_v60k._single(monkeypatch)
    _every_paragraph(monkeypatch)
    fetched = []
    monkeypatch.setattr(cenni, "graph_tools", lambda documents: (_fetch(fetched), lambda subject, code: []))
    monkeypatch.setattr(flow, "first_read_search", lambda *a, **k: ([], {}))
    monkeypatch.setattr(flow, "area_codes", lambda analysis: ["c.c.", "c.p."])
    read = intake.detect("Tizio è imputato di lesioni ai danni di Caio. Il candidato, assunte le vesti del difensore "
                         "di Tizio, premessi brevi cenni sul nesso causale, rediga parere.", model=False)
    out = flow.single_shot("Il caso.", intake=read)
    assert fetched == [("40", "c.p."), ("41", "c.p.")]                            # once, for the norms and the cenni
    norms_job = next(j for jobs in seen["jobs"] for j in jobs if j[0] == NORMS)
    mark = writer_rules.APPLICATION_MARK_DEFENCE           # v65: the client is the imputato -- the defence's reading
    assert f"art. 40 c.p. — Rapporto di causalità — {mark}" in norms_job[2]
    assert f"art. 41 c.p. — Concorso di cause — {mark}" in norms_job[2]
    cenni_job = next(j for jobs in seen["jobs"] for j in jobs if str(j[0]).endswith("Cenni: Nesso causale"))
    assert "NORMA: art. 41 c.p. — Concorso di cause" in cenni_job[2]
    sections = answer_ledger.sections_of(out["answer"])
    assert [t for t, _ in sections if t][0] == "Cenni: Nesso causale"
    norms = dict(sections)[NORMS]
    block = norms.split("**Art. 41 c.p. — Concorso di cause**")[1]
    assert block.strip().startswith("Come opera l'art. 41 c.p.") and "requisiti provati" not in block
    assert out["state"]["records"]["41 c.p."]["applicazione"] == "cenni: nesso causale"
    rows = {r["article"]: r for r in out["state"]["ledger_rows"] if r.get("requisite") is None}
    assert rows["41 c.p."]["status"] == issue_records.APPLICATION and rows["41 c.p."]["in_body"]


# --- 2. the alternatives, each with its article ------------------------------------------------------------------------

ALTERNATIVES = [{"name": "lesioni personali", "rubricated": "Lesione personale", "why": "la morte non era voluta"},
                {"name": "percosse", "rubricated": "", "why": ""},
                {"name": "eccesso di difesa", "rubricated": "", "why": "reazione"}]


def test_each_alternative_is_its_article_and_one_sentence_and_a_fragment_is_dropped():
    state = {"rows": [_row("582"), _row("581")], "read": {"alternatives": ALTERNATIVES}}
    alternatives = posture.alternative_rows(state)
    assert [flow._label(flow._key(r["_article"])) if r else None for _, r in alternatives] == ["582 c.p.", "581 c.p.",
                                                                                                None]
    written = ("Tuttavia, la condotta non era diretta a uccidere.\n\n[1] La lesione è voluta, la morte no. Inoltre "
               "il soccorso prestato lo conferma.\n\nTuttavia resta il nesso.\n[2] ")
    log = {}
    text = flow.alternatives_text(written, alternatives, log)
    assert text == "**Art. 582 c.p. — Lesione personale**\nLa lesione è voluta, la morte no."
    assert "Tuttavia" not in text and "Inoltre" not in text
    assert log["dropped"] == ["percosse", "eccesso di difesa"] and log["orphan"] is True
    assert flow.alternatives_text("Un paragrafo senza numero.", alternatives) == C.ALTERNATIVES_NONE


def test_the_alternatives_job_lists_only_the_alternatives_with_an_article():
    state = {"rows": [_row("582")], "read": {"alternatives": ALTERNATIVES}}
    jobs, fixed = flow.requested_jobs(state, [{"kind": "alternative", "subject": "x"}], {}, "Il caso.",
                                      posture.alternative_rows(state))
    assert not fixed and "[1] art. 582 c.p. — Lesione personale (lesioni personali): la morte non era voluta" in jobs[0][2]
    assert "percosse" not in jobs[0][2] and "[1] SÌ: ..." in jobs[0][1]
    none, fixed = flow.requested_jobs({"read": {"alternatives": ALTERNATIVES}}, [{"kind": "alternative"}], {}, "x",
                                      posture.alternative_rows({"read": {"alternatives": ALTERNATIVES}}))
    assert none == [] and fixed == {0: C.ALTERNATIVES_NONE}


# --- 3. the client's alternative is a weapon -----------------------------------------------------------------------------

def _sided(role, records, case, alternatives=(), rows=()):
    state = {"records": records, "text": case, "read": {"alternatives": list(alternatives)}, "rows": list(rows),
             "analysis": {flow.CASE_KEY: case}}
    posture.mark_sides(state, {"role": role})
    return {k: r["side"] for k, r in records.items()}


def test_the_clients_alternative_is_a_weapon_even_when_the_charge_names_it():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO), "582 c.p.": _rec("Lesione personale", PROVATO),
               "575 c.p.": _rec("Omicidio", DA_PROVARE), "586 c.p.": _rec(RUBRICHE["586"], DA_PROVARE)}
    sides = _sided("imputato", records, "Il PM contesta a Sempronio l'art. 584 c.p. e l'art. 582 c.p.",
                   alternatives=ALTERNATIVES[:1], rows=[_row("582")])
    assert sides == {"584 c.p.": "threat", "582 c.p.": "weapon", "575 c.p.": "weapon", "586 c.p.": "weapon"}
    assert records["582 c.p."]["alternative"] is True and records["582 c.p."]["charged"] == 2
    assert records["584 c.p."]["charged"] == 1 and "alternative" not in records["584 c.p."]


def test_a_charge_named_by_its_name_is_the_threat_and_the_less_specific_name_is_not():
    records = {"572 c.p.": _rec(RUBRICHE["572"], PROVATO), "612-bis c.p.": _rec("Atti persecutori", DA_PROVARE)}
    rows = [_row("572", _first_read="maltrattamenti in famiglia")]
    case = ("Caia sporge denuncia-querela. Viene avviato a carico di Tizio un procedimento penale per il delitto di "
            "maltrattamenti in famiglia.")
    assert _sided("imputato", records, case, rows=rows) == {"572 c.p.": "threat", "612-bis c.p.": "weapon"}
    assert posture.charged_by_name("Tizio è imputato di omicidio preterintenzionale.",
                                   {"575 c.p.": _rec("Omicidio"), "584 c.p.": _rec("Omicidio preterintenzionale")}) \
        == ["584 c.p."]


# --- 4. §1 and the conclusions contest only the recorded elements ---------------------------------------------------------

def test_the_threats_elements_are_all_the_lead_and_the_conclusions_may_contest():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO, DA_PROVARE, side="threat"),
               "586 c.p.": _rec(RUBRICHE["586"], side="threat")}
    rule = flow.threat_elements_rule(records, ["584 c.p.", "586 c.p."])
    assert "art. 584 c.p.: «elemento 1», «elemento 2»" in rule
    assert "art. 586 c.p.: nessun elemento nel registro" in rule and "Nessun altro elemento" in rule
    assert issue_records.threat_elements(records["584 c.p."]) == ["elemento 1", "elemento 2"]


def test_the_lead_and_the_conclusions_jobs_carry_the_element_list(monkeypatch):
    import test_v60k
    state, seen = test_v60k._single(monkeypatch)
    real = posture.mark_sides

    def sides(state_, posture_):
        out = real(state_, posture_)
        rec = state_["records"]["1062 c.c."]
        rec["side"] = "threat"
        return out
    monkeypatch.setattr(posture, "mark_sides", sides)
    flow.single_shot("Il caso.")
    jobs = {j[0]: j for j in seen["jobs"][0]}
    for title in (flow.ANSWER_TITLES[0], flow.ANSWER_TITLES[4]):
        assert "Di queste norme contesta solo gli elementi del registro" in jobs[title][1]
        assert "art. 1062 c.c.:" in jobs[title][1]


# --- 5. the procedural section: the charge and the alternatives argued ----------------------------------------------------

def test_the_procedure_covers_the_charge_and_the_alternatives_that_fit():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", charged=1),
               "582 c.p.": _rec("Lesione personale", PROVATO, alternative=True),
               "581 c.p.": _rec("Percosse", DA_PROVARE, alternative=True),
               "575 c.p.": _rec("Omicidio"), "586 c.p.": _rec(RUBRICHE["586"]), "589 c.p.": _rec("Omicidio colposo")}
    offences = [("575", "Omicidio"), ("584", "Omicidio preterintenzionale"), ("586", RUBRICHE["586"]),
                ("589", "Omicidio colposo"), ("582", "Lesione personale"), ("581", "Percosse")]
    kept, log = posture.procedure_offences(records, offences)          # no alternatives section: by the record
    assert kept == [("584", "Omicidio preterintenzionale"), ("582", "Lesione personale")]   # 581 proves nothing
    assert log["dropped"] == ["575 c.p.", "586 c.p.", "589 c.p.", "581 c.p."] and log["fits_by"] == "record"
    uncharged = {k: {kk: v for kk, v in r.items() if kk != "charged"} for k, r in records.items()}
    assert posture.procedure_offences(uncharged, offences) == (offences, {})                 # no charge: as today
    alone = {"584 c.p.": _rec("Omicidio preterintenzionale", charged=1), "61 c.p.": _rec("Aggravanti", charged=2)}
    assert posture.procedure_offences(alone, [("575", "Omicidio")])[0] == [("584", "Omicidio preterintenzionale")]


def _criminal(stage, querela_stated=False):
    facts = {"582": {"querela": True, "ufficio": [], "competenza": "tribunale monocratico", "pena_max": 3}}
    return {"store_offences_only": True, "offences": [("582", "Lesione personale")], "offence_facts": facts,
            "posture": {"stage": stage, "querela_stated": querela_stated, "family": "penale", "role": "imputato"}}


@pytest.mark.parametrize("stage, term", [("indagini", True), ("pre_filing", True), ("primo_grado", False)])
def test_at_the_trial_the_date_warning_keeps_prescrizione_and_drops_the_querela_term(stage, term):
    text = render_computed.criminal_procedure(_criminal(stage))
    assert "prescrizione del reato (art. 157 c.p.)" in text
    assert ("per i reati a querela, il termine per proporla" in text) is term


def test_the_table_offences_found_by_words_stay_within_the_charge():
    analysis = {"frame": {"areas": ["penale"]}, "offences": [("584", "Omicidio preterintenzionale")],
                "procedure_focus": ["584"], flow.CASE_KEY: "Caio muore. Omicidio, lesioni e percosse."}
    text = render_computed.criminal_procedure(analysis)
    assert "art. 584 c.p." in text and "art. 575" not in text and "art. 582" not in text


# --- 6. no «(da valutare)» on a defending client's alternative ------------------------------------------------------------

@pytest.mark.parametrize("role, marked", [("imputato", False), ("convenuto", False), ("parte_civile", True)])
def test_no_marker_on_the_defences_own_alternative(role, marked):
    records = {"582 c.p.": _rec("Lesione personale", DA_PROVARE, alternative=True),
               "40 c.p.": {"rubrica": "Rapporto di causalità", "requisites": [], "applicazione": "cenni: x"}}
    state = {"records": records, "posture": {"role": role}}
    draft = (f"**1. {flow.ANSWER_TITLES[0]}**\n\nIn subordine il fatto integra le lesioni (art. 582 c.p.), con il nesso "
             f"causale dell'art. 40 c.p.\n\n**2. {NORMS}**\n\nX.")
    out = flow.asserted_only(draft, state)
    assert ("(da valutare" in out) is marked
    if marked:
        assert "art. 582 c.p.)" in out.split("(da valutare")[1] and "40 c.p." not in out.split("(da valutare")[1]
