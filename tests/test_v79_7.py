"""v79.7 -- the four defects of the v79.6 nightly, on synthetic records, rows and passages only:

1. valued: every candidate is valued by the model, the code-argued ones too (their text stays the code's, the realismo
   is the model's); the reply's keys read in every shape (T1, «Tesi 1», «1», the candidate's text, a list without ids
   in the candidates' order, a list of one-key objects, top-level keys); the main line is the best realismo
2. the imputato's enumeration is never empty -- the per-element contests (minoritaria on an «integrato» element), the
   riqualificazione into the read's alternatives, the scriminanti and attenuanti, the containment line last; a charge
   record nobody marked «threat» still gives its contests, and the counts say why it was unmarked
3. Block D: a defence article enters by the concept the facts name, by the retrieval's rank, or (40, 41, 43) with any
   charge; a concept defence is a line with its elements still to prove; the retrieval's top-ranked articles not in
   the record are graded as candidates (a dropped article's stub too); each one's fate traced for the nightly
4. the passage picker: the rubrica's head noun by stem, more rule verbs, the element's own nouns (not a generic one)
   for relevance, per-article counts of what each filter dropped
small: --baseline takes a file or a folder and fails before any cell; the signature block is stripped; the custody
rule never fires on works entrusted to a contractor
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
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _posture(family, role, stage="primo_grado"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


IMPUTATO = _posture("penale", "imputato")
ATTORE = _posture("civile", "attore", "pre_filing")
CASE = "Alfa prende la bicicletta di Beta dal cortile e la porta a casa."
SENTS = [{"id": "S1", "text": CASE, "source": "caso", "certainty": "accertato"}]


def _ctx(recs, posture=IMPUTATO, case=CASE):
    from src.rag import lawyer_method as M
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], case, "Alfa", "", list(recs), [], [], [], [],
                     link_call=lambda *a: {}, case_id_="v79_7")


def _theft(*statuses, side="threat"):
    texts = ["s'impossessa della cosa mobile altrui", "sottraendola a chi la detiene",
             "al fine di trarne profitto per sé o per altri"]
    rec = {"rubrica": "Furto", "charged": 1, "requisites": [_req(i, texts[i - 1], st) for i, st in enumerate(statuses, 1)]}
    if side:
        rec["side"] = side
    return {"624 c.p.": rec}


def _claims():
    return {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "weapon",
                          "requisites": [_req(1, "un contratto a prestazioni corrispettive"),
                                         _req(2, "l'inadempimento di non scarsa importanza")]},
            "1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon",
                          "requisites": [_req(3, "l'inadempimento della prestazione dovuta")]},
            "2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(4, "un fatto doloso o colposo"), _req(5, "un danno ingiusto")]}}


# --- 1. valued ---------------------------------------------------------------------------------------------------------------

def test_a_reply_keyed_t1_to_tn_values_every_candidate_and_the_best_realismo_leads():
    from src.rag import lawyer_method as M
    ctx = _ctx(_claims(), ATTORE)
    cands = M.thesis_candidates(ctx)
    claims = [c for c in cands if c["kind"] == "claim"]
    assert len(claims) == 3 and all(c["argued"] == "code" for c in cands)     # before v79.7: valued 0 by construction
    real = {c["id"]: "basso" for c in cands}
    middle = claims[1]["id"]                    # neither the first nor the last
    real[middle] = "alto"
    real[claims[0]["id"]] = "medio"
    reply = {"valutazioni": {i: {"realismo": r} for i, r in real.items()}}
    plan = M.method_plan(ctx, call=lambda m: [reply])
    counts = ctx["selection_counts"]
    assert counts["valued"] == counts["offered"] == len(cands)
    sols, main, _ = M.selection(ctx, plan)
    assert sols[main]["candidate"] == middle and counts["chosen"] == middle
    assert sols[main]["argomento"] == claims[1]["code_argument"]["argomento"]       # the text stays the code's


def test_a_code_argued_line_with_nothing_proven_is_never_alto():
    from src.rag import lawyer_method as M
    cands = [{"id": "T1", "kind": "eccezione", "tesi": "Eccepire x", "contesta": [], "poggia_su": [],
              "label": M.SUSTAINABLE, "argued": "code", "code_argument": {"realismo": "basso", "argomento": "a"}}]
    sols = M.apply_valuations({"valutazioni": {"T1": {"realismo": "alto"}}}, cands)
    assert sols[0]["realismo"] == "medio" and sols[0]["valued"]
    assert M.apply_valuations({}, cands)[0]["realismo"] == "basso"          # no valuation: the code's


CANDS = [{"id": f"T{i}", "kind": "contest_open", "tesi": f"L'elemento numero {i} non è provato", "contesta": [f"R{i}"],
          "poggia_su": [], "label": "sostenibile", "argued": "model"} for i in (1, 2, 3)]


@pytest.mark.parametrize("raw", [
    {"valutazioni": {"T1": {"realismo": "basso"}, "T2": {"realismo": "alto"}, "T3": {"realismo": "medio"}}},
    {"valutazioni": {"Tesi 1": {"realismo": "basso"}, "tesi_2": {"realismo": "alto"}, "T 3": {"realismo": "medio"}}},
    {"valutazioni": {"1": {"realismo": "basso"}, "T02": {"realismo": "alto"}, "[T3]": {"realismo": "medio"}}},
    {"valutazioni": {"L'elemento numero 1 non è provato": {"realismo": "basso"},
                     "L'elemento numero 2 non è provato.": {"realismo": "alto"},
                     "candidata 3": {"realismo": "medio"}}},
    {"valutazioni": [{"realismo": "basso"}, {"realismo": "alto"}, {"realismo": "medio"}]},
    {"valutazioni": [{"T1": {"realismo": "basso"}}, {"T2": {"realismo": "alto"}}, {"T3": {"realismo": "medio"}}]},
    {"T1": {"realismo": "basso"}, "T2": {"realismo": "alto"}, "T3": {"realismo": "medio"}, "questione_giuridica": "q"},
])
def test_the_valuations_are_matched_in_every_key_shape(raw):
    from src.rag import lawyer_method as M
    vals = M.valuations(raw, CANDS)
    assert {k: v["realismo"] for k, v in vals.items()} == {"T1": "basso", "T2": "alto", "T3": "medio"}
    assert M.score_sample(raw, CANDS, {})[0] == 10.0
    sols, main = M.coherent_solutions({"soluzioni": M.apply_valuations(raw, CANDS)}, {}, {})
    assert sols[main]["candidate"] == "T2"


def test_a_list_with_some_ids_is_not_read_by_position():
    from src.rag import lawyer_method as M
    vals = M.valuations({"valutazioni": [{"id": "T3", "realismo": "alto"}, {"realismo": "basso"}]}, CANDS)
    assert set(vals) == {"T3"}


# --- 2. the imputato's enumeration ---------------------------------------------------------------------------------------------

def test_a_charge_with_every_element_integrato_yields_three_candidates_and_containment_last():
    from src.rag import lawyer_method as M
    ctx = _ctx(_theft("provato", "provato", "provato"))
    cands = M.thesis_candidates(ctx)
    assert len(cands) >= 3 and cands[-1]["kind"] == "contenimento"
    assert [c["kind"] for c in cands[:3]] == ["contest_integrated"] * 3
    assert all(c["label"] == M.MINORITY for c in cands[:3]) and [c["contesta"] for c in cands[:3]] == \
        [["R1"], ["R2"], ["R3"]]
    assert ctx["candidates_trace"]["threats_from"] == "side"


def test_the_order_is_contests_requalification_scriminante_attenuante_containment():
    from src.rag import lawyer_method as M
    recs = {**_theft("provato", "da provare", "provato"),
            "646 c.p.": {"rubrica": "Appropriazione indebita", "side": "weapon", "alternative": True,
                         "requisites": [_req(7, "si appropria della cosa altrui di cui ha il possesso", "da provare")]},
            "54 c.p.": {"rubrica": "Stato di necessità", "side": "weapon",
                        "requisites": [_req(8, "il pericolo attuale di un danno grave alla persona")]}}
    cands = M.thesis_candidates(_ctx(recs))
    assert [c["kind"] for c in cands] == ["contest_integrated", "contest_open", "contest_integrated", "requalify",
                                          "scriminante", "contenimento"]
    assert cands[3]["article"] == "646 c.p."          # listed by the read: offered with its element still open


def test_a_charge_nobody_marked_threat_still_gives_its_contests_and_says_why():
    from src.rag import lawyer_method as M
    ctx = _ctx(_theft("provato", "provato", "provato", side="weapon"))
    ctx["candidates"] = M.thesis_candidates(ctx)
    kinds = [c["kind"] for c in ctx["candidates"]]
    assert kinds.count("contest_integrated") == 3 and kinds[-1] == "contenimento"
    assert ctx["candidates_trace"] == {**ctx["candidates_trace"], "threats_from": "charged", "no_threat": "unmarked"}
    counts = M.selection_counts(ctx, M.clean_method({}, ctx["candidates"]), {})
    assert counts["offered"] == 4 and counts["threats_from"] == "charged" and counts["no_threat"] == "unmarked"


def test_with_no_charge_at_all_the_imputato_still_gets_the_containment_line():
    from src.rag import lawyer_method as M
    ctx = _ctx({})
    cands = M.thesis_candidates(ctx)
    assert [c["kind"] for c in cands] == ["contenimento"] and cands[0]["poggia_su"] == []
    assert ctx["candidates_trace"]["no_threat"] == "no_charge"
    arg = cands[0]["code_argument"]
    assert arg["realismo"] == "medio" and "l'addebito" in arg["contro"]


def test_past_the_cap_the_minority_contests_go_and_the_containment_stays():
    from src.rag import lawyer_method as M
    recs = _theft("provato", "provato", "provato")
    recs["624 c.p."]["requisites"] += [_req(i, f"elemento aperto {i}", "da provare") for i in range(4, 12)]
    cands = M.thesis_candidates(_ctx(recs))
    kinds = [c["kind"] for c in cands]
    assert len(cands) == M.CANDIDATES_MAX and kinds[-1] == "contenimento" and "contest_integrated" not in kinds


# --- 3. Block D -----------------------------------------------------------------------------------------------------------------

def _row(number, code, rubrica="", rank=None):
    row = {"_article": (number, code), "s": {"name": number, "titolo_articolo": rubrica}, "d": {"id": code}}
    if rank is not None:
        row["_fused_rank"] = rank
    return row


def _decompose(monkeypatch, state, posture, rows, reqs_by_label, charges=(), case=""):
    from src.rag import interactive_flow as flow, issue_records, posture as posture_mod
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture_mod, "intake_charges", lambda p: [])
    monkeypatch.setattr(posture_mod, "charge_labels", lambda c: list(charges))
    monkeypatch.setattr(posture_mod, "claim_from_read", lambda *a, **k: [])
    monkeypatch.setattr(posture_mod, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a, *_: [dict(x) for x in reqs_by_label.get(
        flow._label(flow._key(r["_article"])), [])])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    monkeypatch.setattr(issue_records, "seed", lambda rec, row: None)
    return flow.decompose_records(state, posture, rows, set(), fetch=lambda keys: [])


def _el(text, status="da provare"):
    return {"id": "", "comma": 1, "kind": "requisito", "text": text, "status": status}


def test_legittima_difesa_in_the_facts_puts_52_in_the_record_and_gives_a_defence_line(monkeypatch):
    from src.rag import lawyer_method as M
    case = ("Beta aggredisce Alfa con un coltello; Alfa lo spinge via per legittima difesa e Beta cade e si ferisce. "
            "Alfa è imputato di lesioni personali (art. 582 c.p.).")
    state = {"records": {}, "analysis": {}, "read": {}}
    rows = [_row("52", "c.p.", "Difesa legittima"), _row("582", "c.p.", "Lesione personale")]
    reqs = {"52 c.p.": [_el("costretto dalla necessità di difendere un diritto contro il pericolo attuale di un'offesa "
                            "ingiusta"), _el("la difesa sia proporzionata all'offesa")],
            "582 c.p.": [_el("cagiona una lesione personale", "provato"), _el("dalla quale deriva una malattia",
                                                                                "provato")]}
    link, _ = _decompose(monkeypatch, state, IMPUTATO, rows, reqs, charges=["582 c.p."], case=case)
    rec = state["records"]["52 c.p."]
    assert "52 c.p." in link and rec["defence_by"] == "concetto" and rec["decomposed_for"] == "difesa di parte generale"
    for label, r in state["records"].items():           # the ids as number() gives them
        for i, x in enumerate(r.get("requisites") or [], 1):
            x["id"] = x["id"] or f"{label.split()[0]}_{i}"
    state["records"]["582 c.p."].update(side="threat", charged=1)
    ctx = _ctx(state["records"], case=case)
    assert "52 c.p." in ctx["records"]                  # a concept defence is never «no fact» noise
    cands = M.thesis_candidates(ctx)
    lines = [c for c in cands if c["kind"] == "scriminante"]
    assert len(lines) == 1 and lines[0]["article"] == "52 c.p." and cands[-1]["kind"] == "contenimento"
    ctx["candidates"] = cands
    assert M.selection_counts(ctx, M.clean_method({}, cands), {})["defence_lines"] == 1


def test_causation_and_fault_enter_with_any_charge_and_never_without_one(monkeypatch):
    from src.rag import interactive_flow as flow
    state = {"records": {}, "analysis": {}, "read": {}}
    assert flow.defence_wanted(state, charged=True) == {"40 c.p.": "addebito", "41 c.p.": "addebito",
                                                        "43 c.p.": "addebito"}
    assert flow.defence_wanted(state) == {}
    rows = [_row(n, "c.p.") for n in ("40", "41", "43", "624")]
    reqs = {"41 c.p.": [_el("cause sopravvenute da sole sufficienti a determinare l'evento")]}
    state["defence_gate"] = {"ranked": [], "dropped": ["40 c.p.", "47 c.p.", "49 c.p."]}
    link, _ = _decompose(monkeypatch, state, IMPUTATO, rows, reqs, charges=["624 c.p."])
    assert state["records"]["41 c.p."]["defence_by"] == "addebito"
    assert state["defence_gate"]["dropped"] == ["47 c.p.", "49 c.p."]      # 40 entered by the charge: not dropped
    assert state["decompose_why"]["40 c.p."] == "undecomposed"       # no element read: traced, no record


def test_a_civil_case_grades_a_ranked_article_outside_the_record_as_a_candidate(monkeypatch):
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = {"records": {"1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon",
                                       "requisites": [_req(1, "l'inadempimento della prestazione")]},
                         "1455 c.c.": {"rubrica": "Importanza dell'inadempimento", "requisites": [],
                                       "dropped": "grado 1"}},
             "analysis": {}, "read": {}, "ranked": ["1218 c.c.", "1453 c.c.", "1455 c.c."],
             "retrieval_top": [{"article": "1218 c.c.", "rank": 1}, {"article": "1453 c.c.", "rank": 2},
                               {"article": "1455 c.c.", "rank": 3}, {"article": "405 c.p.p.", "rank": 4,
                                                                      "gate": "area_gate"}]}
    rows = [_row("1453", "c.c.", "Risolubilità del contratto per inadempimento"),
            _row("1455", "c.c.", "Importanza dell'inadempimento")]
    reqs = {"1453 c.c.": [_el("un contratto a prestazioni corrispettive", "provato")],
            "1455 c.c.": [_el("l'inadempimento non ha scarsa importanza", "non risulta")]}
    link, _ = _decompose(monkeypatch, state, ATTORE, rows, reqs)
    recs = state["records"]
    assert link == ["1453 c.c.", "1455 c.c."] and recs["1453 c.c."]["ranked_candidate"]
    assert recs["1455 c.c."]["ranked_candidate"]            # a dropped article's stub is no record: graded too
    recs["1453 c.c."]["requisites"][0].update(id="R2", status="provato")
    recs["1455 c.c."]["requisites"][0].update(id="R3", status="non risulta")
    ctx = _ctx(recs, ATTORE)
    cands = M.thesis_candidates(ctx)
    assert any(c["kind"] == "claim" and c["article"] == "1453 c.c." for c in cands)
    state["candidate_articles"] = M.candidate_articles({"candidates": cands})
    trace = {t["article"]: t for t in flow.ranked_trace(state, ctx)}
    assert trace["1453 c.c."]["status"] == "candidate" and trace["1218 c.c."]["status"] == "in record"
    assert trace["1455 c.c."]["status"] == "dropped" and "nessun elemento provato" in trace["1455 c.c."]["why"]
    assert trace["405 c.p.p."]["status"] == "dropped" and "area_gate" in trace["405 c.p.p."]["why"]
    selection = M.selection_counts({**ctx, "candidates": cands}, M.clean_method({}, cands), {})
    assert selection["ranked_candidates"] == 1


def test_the_nightly_prints_the_ranked_trace_per_cell():
    nightly = _script("nightly_sim")
    lines = nightly.ranked_lines([{"id": "c1", "ranked_trace": [
        {"rank": 1, "article": "1453 c.c.", "status": "candidate", "why": "valutato e ammesso come candidato",
         "line": True}]}])
    assert "| c1 | 1 | 1453 c.c. | candidate (linea) | valutato e ammesso come candidato |" in lines


# --- 4. the passage picker ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rubrica, stem", [("Lesione personale", "lesion"),
                                           ("Maltrattamenti contro familiari e conviventi", "maltratt"),
                                           ("Furto", "furt"), ("Ricettazione", "ricettaz"), ("", "")])
def test_the_rubricas_head_noun_by_stem(rubrica, stem):
    from src.rag import rulings
    assert rulings.head_stem(rubrica) == stem


@pytest.mark.parametrize("abstract, rubrica, ok", [
    ("Maltrattamenti in famiglia: abitualità della condotta", "Maltrattamenti contro familiari e conviventi", True),
    ("Lesioni: malattia e durata", "Lesione personale", True),
    ("Furti in abitazione e destrezza", "Furto", True),
    ("Truffa contrattuale e artifizi", "Furto", False),
    ("Qualsiasi cosa", "", True),
])
def test_one_noun_by_stem_puts_the_abstract_on_the_article(abstract, rubrica, ok):
    from src.rag import rulings
    assert rulings.on_rubrica(abstract, rubrica) is ok


RECEIVE = {"id": "R2", "text": "acquista riceve od occulta denaro o cose provenienti da un qualsiasi delitto"}
PROFIT = {"id": "R1", "text": "al fine di procurare a sé o ad altri un profitto"}
PRINCIPLE = "La ricettazione sussiste quando la cosa ricevuta ha una provenienza delittuosa nota all'agente."
MENTION = "Si richiama l'art. 648 c.p. come indicato nella sentenza impugnata."
OFF_ELEMENT = ("Il profitto del reato tributario consiste nel risparmio di imposta conseguito con l'evasione fiscale.")


def test_a_principle_sentence_passes_a_mention_passes_nothing_an_off_element_passage_is_dropped():
    from src.rag import rulings
    counts = {}
    got = rulings.candidate_sentences(" ".join([MENTION, PRINCIPLE, OFF_ELEMENT]), [PROFIT, RECEIVE], "648 c.p.",
                                      counts)
    assert got == [PRINCIPLE]
    assert counts == {"rule_verb": 1, "element_nouns": 1}
    assert rulings.sentence_filter(OFF_ELEMENT, [PROFIT]) == "element_nouns"       # «profitto» alone: generic
    assert rulings.quotable(PRINCIPLE, RECEIVE) and not rulings.quotable(OFF_ELEMENT, PROFIT)


@pytest.mark.parametrize("sentence", [
    "Il delitto di furto è configurabile anche quando la cosa sottratta resta nella sfera di vigilanza del detentore.",
    "Ai fini della ricettazione è sufficiente il dubbio sulla provenienza della cosa ricevuta.",
    "L'abitualità della condotta deve essere ritenuta anche per atti intervallati nel tempo.",
    "Non basta un singolo episodio di percosse a integrare i maltrattamenti.",
])
def test_the_rule_verbs_cover_the_massimas_forms(sentence):
    from src.rag import rulings
    assert rulings._RULE_RE.search(sentence)


def test_the_build_counts_what_each_filter_dropped_per_article():
    from src.rag import rulings

    class Session:
        def __init__(self, by_article, texts):
            self.by_article, self.text_of = by_article, texts

        def run(self, query, **kw):
            rows = self.by_article.get((kw["doc"], kw["num"]), []) if query == rulings.FETCH_QUERY else \
                [{"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

            class R:
                def data(self_inner):
                    return rows
            return R()

    def ruling(i, abstract, section="20"):
        return {"ruling": f"X{i}", "file_name": f"snpen@s{section}@a2025@n{i:05d}@tS.clean.pdf", "name": "",
                "document_number": "1", "date_enacted": "2025-03-01", "linked": True, "section": f"S{i}",
                "abstract": abstract}
    ctx = {"records": {"648 c.p.": {"side": "threat", "rubrica": "Ricettazione", "requisites": [
        {**RECEIVE, "status": "da provare", "kind": "requisito"}]}}, "case": "Caio compra una bicicletta rubata.",
        "candidates": [{"id": "T1", "kind": "contest_open", "tesi": "x", "contesta": ["R2"], "poggia_su": [],
                        "label": "sostenibile"}], "assessment": {"article": "648 c.p."}}
    rows = [{"_article": ("648", "c.p."), "d": {"id": "CP"}, "s": {"name": "648"}}]
    session = Session({("CP", "648"): [ruling(1, "Ricettazione: provenienza delittuosa"),
                                       ruling(2, "Truffa e artifizi")]},
                      {"S1": " ".join([MENTION, PRINCIPLE, OFF_ELEMENT])})
    record = rulings.build(ctx, rows, session, call=lambda m, t: {"P1": {"frase": "P1.1", "elemento": "R2",
                                                                         "pertinenza": 2, "verso": "a favore"}})
    art = record["log"]["articles"]["648 c.p."]
    assert art["filters"]["rubrica"] == 1 and art["filters"]["rule_verb"] == 1 and art["passages"] == 1
    assert record["kept"][0]["passage"] == PRINCIPLE
    nightly = _script("nightly_sim")
    text = "\n".join(nightly.rulings_lines([{"id": "c1", "rulings": record["log"]}]))
    assert "| c1 | 648 c.p. | 2 | 2 | 1 | 1 | 1 | 1 |" in text


# --- small ----------------------------------------------------------------------------------------------------------------------

def test_the_baseline_is_a_file_or_a_folder_and_an_unreadable_one_fails_before_any_cell(tmp_path, monkeypatch):
    nightly = _script("nightly_sim")
    run = {"version": "79.6.0", "cells": [{"id": "c1", "checks": []}]}
    (tmp_path / "nightly-v79.6").mkdir()
    (tmp_path / "nightly-v79.6" / "other.json").write_text(json.dumps(run))
    assert nightly.read_baseline(str(tmp_path / "nightly-v79.6"))["version"] == "79.6.0"
    (tmp_path / "nightly-v79.6" / "nightly.json").write_text(json.dumps({**run, "version": "79.6.1"}))
    assert nightly.read_baseline(str(tmp_path / "nightly-v79.6"))["version"] == "79.6.1"
    (tmp_path / "flat.json").write_text(json.dumps(run))
    assert nightly.read_baseline(str(tmp_path / "flat"))["cells"]          # «flat» for «flat.json»
    assert nightly.read_baseline(None) is None
    ran = []
    monkeypatch.setattr(nightly, "run_cell", lambda spec: ran.append(spec) or {})
    for bad in (tmp_path / "missing", tmp_path / "empty"):
        if bad.name == "empty":
            bad.mkdir()
        with pytest.raises(SystemExit):
            nightly.main(["--env", "/nonexistent", "--baseline", str(bad), "--no-stages"])
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(SystemExit):
        nightly.main(["--env", "/nonexistent", "--baseline", str(tmp_path / "broken.json"), "--no-stages"])
    assert ran == []


def test_the_signature_block_leaves_the_passage_and_the_check_finds_it():
    from src.rag import rulings
    text = ("La condotta richiede un atto di dominio. Firmato Da: ROSSI MARIO Emesso Da: ARUBAPEC S.P.A. NG CA 3 "
            "Serial#: 7c5b1a9e Il fatto è punito.")
    out = rulings.strip_stamp(text)
    assert out == "La condotta richiede un atto di dominio. Il fatto è punito."
    assert rulings.strip_stamp("Firmato digitalmente da Mario Rossi La Corte rigetta.") == "La Corte rigetta."
    assert any("firma" in x for x in rulings.printed_problems("«Il fatto è punito. Firmato Da: ROSSI MARIO»"))
    assert rulings.printed_problems("La condotta è firmata da chi la compie.") == []


def test_the_custody_rule_never_fires_on_works_entrusted_to_a_contractor():
    from src.rag import fact_signals as fs

    def fired(**kw):
        base = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
        base.update(kw)
        return {r["id"] for r in fs.fire_rules(base, fs.compute_terms(base))}

    def ev(what):
        return {"date": "", "what": what, "who": ""}
    entrusted = fired(events=[ev("Tizio affida all'impresa Alfa il rifacimento del tetto"),
                              ev("compaiono infiltrazioni d'acqua dal tetto")])
    assert "danno_cosa_in_custodia" not in entrusted
    assert "danno_cosa_in_custodia" not in fired(contracts=[{"type": "appalto", "parties": "", "clauses": ""}],
                                                 events=[ev("infiltrazioni dal tetto")])
    assert "danno_cosa_in_custodia" in fired(events=[ev("infiltrazioni d'acqua dal tetto condominiale nel bagno")])


def test_the_filters_measurement_runs_offline():
    _script("rulings_filters").selftest()
