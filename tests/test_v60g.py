"""v60g: the writer applies the per-article record under one de-conflicted rule set (d); one check against the record,
run last, is the only one that removes, every removal with its reason in the ledger (e) -- with its three invariants;
then the round-1 demotion by the distinctive element (1), the read-back's acts of a party (2), the institution/rubrica
check on every section (3), the first read's parties and the verified re-read (4), terms only with a graded topic of
their matter (5), relative periods as date ranges (6), the alternative qualifications (7). The model and the graph are
stubbed."""

import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import (answer_ledger, case_read, citation_ledger, fact_signals, issue_records, readback,  # noqa: E402
                     record_check, render_computed, topic_step, writer_rules)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402
import test_v60e  # noqa: E402
from test_v60f import WRITER, _graded_state  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]


def _answer(monkeypatch, state, writer, link=None, seen=None):
    """The answer step with the plan, the writer and the link stubbed (link: {requisite id: its answer})."""
    seen = seen if seen is not None else {}
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])
    monkeypatch.setattr(ptw, "make_plan", lambda *a, **k: {
        "parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
        "subordinate": [], "punti_deboli": [], "questioni": []})
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    monkeypatch.setattr(issue_records, "_link_call",
                        lambda topic, reqs, sents: {r["id"]: v for _, r in reqs for k, v in (link or {}).items()
                                                    if r["id"] == k})

    def fake_write(plan, jobs, lang, named, documents, rows, case, **kwargs):
        seen["jobs"], seen["writer_rules"] = jobs, kwargs.get("writer_rules")
        return [(t, writer.get(t, f"Testo di {t}.")) for t, *_ in jobs]
    monkeypatch.setattr(ptw, "write_sections", fake_write)
    return flow.write_answer(state, [1])


def _status_lines(text):
    """{requisite text (60 chars): status} of §2's status lines."""
    body = answer_ledger.sections_of(text)
    norms = dict(body).get(NORMS, "")
    out = {}
    for m in re.finditer(r"(?m)^- [^\n]*?, requisito «(.+?)»: ([a-zà-ü ]+)\.$", norms):
        out[m.group(1)[:60]] = m.group(2)
    return out


# --- d. the writer applies the record -----------------------------------------------------------------------------------

def test_the_records_status_settles_the_facts_and_the_link():
    by_id = {"S1": {"id": "S1", "certainty": "accertato", "text": "Domitilla ha chiuso il cancello del fondo nel 2016."},
             "S2": {"id": "S2", "certainty": "allegato", "text": "Secondo Leonardo il cancello del fondo era aperto."}}
    req = lambda **kw: {"kind": "requisito", "facts_status": C.STATUS_DA_PROVARE, "supporting": [], "contradicting": [],
                        "text": "la chiusura del cancello del fondo", **kw}
    cases = [(req(supporting=["S1"]), C.STATUS_PROVATO),
             (req(supporting=["S2"]), C.STATUS_PROVATO),            # v60t (1): grounded -- an allegation too
             (req(supporting=["S9"]), C.STATUS_DA_PROVARE),         # a sentence the case does not hold
             (req(contradicting=["S1"], supporting=["S1"]), C.STATUS_NON_RISULTA),
             (req(facts_status=C.STATUS_PROVATO, contradicting=["S1"]), C.STATUS_PROVATO),
             (req(kind="comma", supporting=["S1"]), C.STATUS_DA_PROVARE), (req(), C.STATUS_DA_PROVARE)]
    for r, status in cases:
        issue_records.settle(r, by_id)
        assert r["status"] == status and r["why"], r
    rec = {"requisites": [{"kind": "requisito", "comma": 1, "status": C.STATUS_NON_RISULTA, "text": "a", "why": "x"},
                          {"kind": "requisito", "comma": 1, "status": C.STATUS_PROVATO, "text": "b"}]}
    assert issue_records.verdict(rec) == issue_records.NOT_APPLICABLE and "«a» non risulta" in \
        issue_records.verdict_reason(rec)
    rec["requisites"].append({"kind": "requisito", "comma": 2, "status": C.STATUS_PROVATO, "text": "c"})
    assert issue_records.verdict(rec) == issue_records.NOT_APPLICABLE  # v60r (3): comma 1, the anchor comma, decides
    rec["anchor_comma"] = 2                                              # v60s (2): its topic names co. 2
    assert issue_records.verdict(rec) == issue_records.APPLICABLE        # comma 1 «non pertinente», comma 2 proven
    assert issue_records.verdict({"requisites": [rec["requisites"][1]]}) == issue_records.APPLICABLE


def test_while_the_answer_is_written_every_status_is_the_records(monkeypatch):
    state = _graded_state(monkeypatch)
    text = state["records"]["1062 c.c."]["requisites"][0]["text"]
    analysis = flow._restored_analysis(state["analysis"])
    state["records"]["1062 c.c."]["requisites"][0]["status"] = C.STATUS_PROVATO
    assert flow.requisite_status(text, analysis) == C.STATUS_DA_PROVARE              # outside: the facts'
    with issue_records.statuses_from(state["records"]):
        assert flow.requisite_status(text, analysis) == C.STATUS_PROVATO               # inside: the record's
        assert flow.requisite_status("un requisito che nessun record ha", analysis) == C.STATUS_DA_PROVARE
    assert issue_records.recorded_status(text) is None


def test_the_writer_reads_the_schede_under_the_deconflicted_rules(monkeypatch):
    state = _graded_state(monkeypatch)
    seen = {}
    _answer(monkeypatch, state, WRITER, {"R1": {"sostiene": ["S2"], "manca": "l'atto di divisione"}}, seen)
    norms = next(j for j in seen["jobs"] if j[0] == NORMS)
    assert norms[1].startswith(writer_rules.NORMS_INSTRUCTION)
    assert issue_records.WRITER_TITLE in norms[2] and "«Domitilla ha chiuso il cancello nel 2016.» (accertato)" in norms[2]
    assert issue_records.FACTS_TITLE in norms[2] and "riporta quello stato" not in norms[2]
    assert "esito: " in norms[2] and "S2" not in norms[2].split(issue_records.WRITER_TITLE)[1].split("\n\n")[0]
    rules = seen["writer_rules"]
    assert rules.startswith(writer_rules.RECORD_RULES) and "una norma che non serve si omette" not in rules
    assert "Onere della prova" in rules                                     # the other writer rules stand
    assert all(writer_rules.SCOPE_NOTE in j[1] for j in seen["jobs"])     # P4


def test_p1_to_p6_are_resolved_and_written_down(monkeypatch):
    doc = writer_rules.__doc__
    for item in ("P1", "P2", "P3", "P4", "P5", "P6"):
        assert f"  {item}  " in doc, item
    monkeypatch.setattr(ptw, "_writer_rules", lambda: True)
    system = ptw._section_system({"parte_assistita": "Leonardo", "posizione": ""}, "it", flow.sections_note(),
                                 writer_rules.system_rules("Leonardo"))
    assert "Non scrivere mai che una norma è «non pertinente»" not in system \
        and "Nelle SCHEDE ci sono solo le norme con esito «applicabile», «argomento del cliente» o «da provare»" \
        in system                                                                            # v60h (5), v60i (1, 2)
    assert "Non scrivere mai che una norma è «non pertinente»" in ptw._section_system(
        {"parte_assistita": "Leonardo", "posizione": ""}, "it")                       # the parere unchanged
    analysis = {"questioni": [{"titolo": "Usucapione del passaggio"}], "fact_signals": {"fired": []}}
    assert ptw.acquired_rights(analysis) and not ptw.acquired_rights(analysis, stated_only=True)       # P6
    monkeypatch.setattr(ptw, "_review_fixes", lambda: True)
    monkeypatch.setattr(ptw, "_writer_rules", lambda: False)
    fixed = ptw._plan_system(1).replace(ptw._ACQUIRED_RULE.split("\n- ")[0] + "\n", ptw.FIXED_ACQUIRED_RULE)
    assert ptw.FIXED_ACQUIRED_RULE in fixed and "la prima questione è il riconoscimento" not in fixed


def test_an_article_the_record_supports_is_applied_even_when_not_the_topics_anchor(monkeypatch):
    state = _graded_state(monkeypatch)
    planned = [("1062", "c.c."), ("1061", "c.c."), ("1159", "c.c.")]
    analysis = flow._restored_analysis(state["analysis"])
    topics = [{"anchor": "1062 c.c.", "matched": ["1062 c.c."]}]
    before = flow.glossed_refs(state, topics, analysis, "", planned)
    # v60j: 1062's missing facts are open round-1 questions (atto_separazione): a block
    assert ("1062", "c.c.") in before and ("1159", "c.c.") not in before
    state["records"]["1159 c.c."]["requisites"] = [{"status": C.STATUS_DA_PROVARE, "supporting": ["S1"]}]
    assert flow.glossed_refs(state, topics, analysis, "", planned) == before + [("1159", "c.c.")]
    # v60i (1): the questions answered and no sentence of the case for it, the topic's anchor alone earns none
    state["answered"] = {q["id"]: "risposta" for q in state["questions"]}
    assert ("1062", "c.c.") not in flow.glossed_refs(state, topics, analysis, "", planned)


# --- e. one check, last, against the record: the three invariants --------------------------------------------------------

def test_a_step_proposes_and_never_deletes_with_a_deferring_ledger():
    ledger = answer_ledger.Ledger(defer=True)
    with answer_ledger.active(ledger):
        out = ptw._step("cut", [("A", "Prima frase del testo. Seconda frase da togliere. Terza frase che resta.")],
                        lambda secs: [("A", "Prima frase del testo. Terza frase che resta (da verificare).")])
    assert out[0][1] == "Prima frase del testo. Seconda frase da togliere. Terza frase che resta (da verificare)."
    assert [(e["kind"], e["step"]) for e in ledger.entries] == [("proposed", "cut")]
    assert ledger.unreasoned() == []
    writing = answer_ledger.Ledger(defer=True)       # the writer's own rewrite of a cut section is writing
    writing.record("complete_sections", [("A", "Una frase che il redattore riscrive.")], [("A", "Altro testo nuovo.")])
    assert [(e["kind"], e["reason"]) for e in writing.entries] == [
        ("removed", answer_ledger.WRITING_STEPS["complete_sections"])]


def test_invariant_no_deletion_without_a_recorded_reason(monkeypatch):
    state = _graded_state(monkeypatch)
    writer = dict(WRITER)
    writer["Risposta e strategia"] += " L'art. 1061 c.c. non si applica al caso."
    text, _ = _answer(monkeypatch, state, writer)
    ledger = state["ledger"]
    assert ledger["defer"] and ledger["unaccounted"] == [] and ledger["unreasoned"] == []
    removed = [e for e in ledger["entries"] if e["kind"] == "removed"]
    assert removed and all(e["step"] == "record_check" and e["reason"] and e["rule"] in record_check.REMOVE_RULES
                           for e in removed)
    assert any(e["rule"] == "esito_contraddetto" and "1061" in e["sentence"] for e in removed)
    assert "non si applica al caso" not in text


def test_invariant_the_answers_statuses_are_the_records(monkeypatch):
    state = _graded_state(monkeypatch)
    state["sentences"].append({"id": "S3", "text": "I due fondi, ora divisi, erano posseduti dallo stesso proprietario.",
                               "source": "precisazione", "certainty": "accertato"})
    # v60h (5): R1 proved by a sentence holding its own nouns (fondi, posseduti, proprietario); R2's sentence holds none
    text, _ = _answer(monkeypatch, state, WRITER, {"R1": {"sostiene": ["S3"]}, "R2": {"sostiene": ["S2"]}})
    lines = _status_lines(text)
    assert lines, text
    records = {r["text"][:60]: r["status"] for rec in state["records"].values() for r in rec.get("requisites") or []}
    assert all(records[k] == v for k, v in lines.items())
    assert C.STATUS_PROVATO in lines.values()                     # R1, linked to an established sentence


def test_invariant_every_planned_article_in_exactly_one_bucket_the_inapplicable_by_the_record(monkeypatch):
    state = _graded_state(monkeypatch)
    writer = dict(WRITER)
    writer[NORMS] += " L'art. 1062 c.c. si applica al caso."
    text, _ = _answer(monkeypatch, state, writer, {"R1": {"sostiene": ["S2"]}, "R2": {"contraddice": ["S2"]}})
    buckets = {label: rec["answer"] for label, rec in state["records"].items() if rec.get("answer")}
    assert buckets["1062 c.c."]["bucket"] == flow.BUCKET_NOT_APPLICABLE and "non risulta" in buckets["1062 c.c."]["reason"]
    assert all(b["bucket"] in (flow.BUCKET_WRITTEN, flow.BUCKET_NOT_APPLICABLE, flow.BUCKET_NOT_TREATED, flow.BUCKET_CAP)
               for b in buckets.values())
    assert all(b["bucket"] != flow.BUCKET_NOT_TREATED or b["reason"] for b in buckets.values())
    assert "si applica al caso" not in text                              # contradicts the record's verdict


def _ctx(**kw):
    base = {"records": {"1062 c.c.": {"requisites": [{"kind": "requisito", "comma": 1, "status": "da provare",
                                                      "text": "fondi divisi"}], "verdict": "da provare"},
                        "1061 c.c.": {"requisites": [], "verdict": "da provare"}},
            "allowed": {"1062 c.c.", "1061 c.c."}, "glossed": {"1062 c.c."},
            "texts": {"1062 c.c.": "Quando consta che due fondi, attualmente divisi, sono stati posseduti dallo stesso "
                                   "proprietario, la servitù s'intende stabilita."},
            "rubriche": {"1062 c.c.": "Destinazione del padre di famiglia"}, "commi": {"1062 c.c.": {1}},
            "documents": [], "client": "Leonardo Rossi", "skip": {"Profili processuali"}, "norms_title": NORMS,
            "numbering": [("1062", "c.c.")]}
    base.update(kw)
    return base


def test_the_checks_rules_each_against_the_record():
    sections = [("Risposta e strategia",
                 "Leonardo agisce in confessoria contro Domitilla per il passaggio. L'art. 2043 c.c. regola il danno. "
                 "Si consiglia di querelare Rossi per il danno. , e quindi il resto. Questo requisito è provato. "
                 "Questo requisito è provato. Leonardo agisce in confessoria contro Domitilla per il passaggio."),
                (NORMS, "**Art. 1062 c.c. — Destinazione del padre di famiglia**\n\n[1] L'art. 1062 c.c. dice «la servitù "
                        "nasce sempre dal solo uso ventennale del passaggio» e vale qui. Il comma 3 dell'art. 1062 c.c. "
                        "non serve.\n\n**Art. 1061 c.c.**\n\nL'art. 1061 c.c. riguarda le opere visibili sul fondo."),
                ("Profili processuali", "L'art. 9999 c.c. resta: è del codice.")]
    out, log = record_check.check(sections, _ctx())
    rules = {e["rule"] for e in log["removed"]}
    assert {"fuori_dal_record", "contro_assistito", "frammento", "ripetizione", "virgolette_fuori_testo",
            "comma_inesistente", "norma_senza_testo"} <= rules, log["removed"]
    assert out[2][1] == sections[2][1]                                       # the code's section is not read
    assert out[0][1].count("confessoria") == 1 and out[0][1].count("Questo requisito è provato.") == 2
    dashed, _ = record_check.check([(NORMS, "**Art. 1062 c.c.**\n\n— Il venditore è tenuto a garantire la cosa.")], _ctx())
    assert "— Il venditore è tenuto a garantire la cosa." in dashed[0][1]          # a list's dash is no fragment
    quoted, _ = record_check.check([(NORMS, "**Art. 1062 c.c. — Destinazione del padre di famiglia**\n\nL'art. 1062 c.c. "
                                            "(«Destinazione del padre di famiglia») dice che «due fondi, attualmente "
                                            "divisi […] la servitù s'intende stabilita».")], _ctx())
    assert "«Destinazione del padre di famiglia»" in quoted[0][1]             # the rubrica, and a cut quotation, hold


def test_a_proposal_no_rule_confirms_is_rejected_and_the_sentence_stays():
    ledger = answer_ledger.Ledger(defer=True)
    before = [("Conclusioni", "L'esito è favorevole al cliente nella causa. Il giudice valuterà le prove raccolte.")]
    kept = ledger.record("unanswered_out", before, [("Conclusioni", "L'esito è favorevole al cliente nella causa.")])
    out, log = record_check.check(kept, _ctx(), ledger)
    assert "Il giudice valuterà le prove raccolte." in out[0][1]
    assert log["rejected"] and log["rejected"][0]["proposed_by"] == ["unanswered_out"] and not log["removed"]


def test_the_record_decides_non_applicable_never_the_gloss():
    text = ("**Art. 1062 c.c.**\n- testo: da provare.\n\nLa norma non si applica al caso.\n\n**Art. 1051 c.c.**\n"
            "- testo: da provare.\n\n**Art. 1054 c.c.**")
    out = flow.norms_final(text, [], {"1062 c.c.": "da provare", "1054 c.c.": issue_records.NOT_APPLICABLE})
    assert "La norma non si applica al caso." in out and "**Art. 1051 c.c.**" in out
    assert C.NOT_APPLICABLE_LINE.split("{")[0] + "art. 1054 c.c." in out
    assert "Norme considerate e non applicabili" not in flow.norms_final(text.replace("**Art. 1054 c.c.**", ""), [], {})


# --- 1. round 1: the distinctive element ----------------------------------------------------------------------------------

def _graded_rows(monkeypatch):
    state = _graded_state(monkeypatch)
    rows = flow._restored_rows(state["rows"])
    for r in rows:
        r["_grade"] = 2
    return rows


def test_an_article_whose_distinctive_element_the_case_contradicts_is_graded_one(monkeypatch):
    rows = _graded_rows(monkeypatch)
    elements = issue_records.distinctive(rows, {})
    assert "1062 c.c." in elements and elements["1062 c.c."]["text"]
    supported = elements["1062 c.c."]["text"]
    calls = []

    def call(topic, reqs, sents):       # v60m: only a sentence narrated as a fact contradicting it demotes
        calls.append(len(reqs))
        return {r["id"]: {"sostiene": ["S2"] if r["text"] == supported else [],
                          "contraddice": [] if r["text"] == supported else ["S2"], "manca": "x"} for _, r in reqs}
    log = issue_records.demote_contradicted(rows, "Leonardo passa sul fondo. Domitilla ha chiuso il cancello nel 2016.",
                                            {}, keep=["1158 c.c."], call=call)
    by = {issue_records._label(r["_article"]): r for r in rows}
    assert by["1062 c.c."]["_grade"] == 2 and by["1062 c.c."]["_distinctive"]["supporting"] == ["S2"]
    demoted = {d["article"] for d in log["demoted"]}
    assert demoted and "1062 c.c." not in demoted and "1158 c.c." not in demoted and "1158 c.c." in log["kept_named"]
    assert by["1158 c.c."]["_grade"] == 2
    assert all(by[a]["_grade"] == 1 and by[a]["_grade_before"] == 2 and "elemento distintivo" in by[a]["_grade_issue"]
               for a in demoted)
    assert sum(calls) == len(elements) and log["ms"] >= 0


def test_a_failed_or_late_link_demotes_nothing(monkeypatch):
    rows = _graded_rows(monkeypatch)
    log = issue_records.demote_contradicted(rows, "Leonardo passa sul fondo.", {},
                                            call=lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert log["errors"] and not log["demoted"] and all(r["_grade"] == 2 for r in rows)
    log = issue_records.demote_contradicted(rows, "Leonardo passa sul fondo.", {},
                                            call=lambda *a: time.sleep(0.5) or {}, seconds=0.1)
    assert log["errors"][0].startswith("timeout") and all(r["_grade"] == 2 for r in rows)


# --- 2. the read-back: a party's act is the case's or «da verificare» --------------------------------------------------

def test_a_partys_act_maps_to_a_case_sentence_or_is_marked():
    sents = issue_records.sentences("Il conduttore non ha pagato i canoni da marzo. Il locatore ha inviato una "
                                    "diffida il 3 aprile.")
    sections = [("Risposta e strategia", "**1. Risposta e strategia**\nIl conduttore ha omesso di pagare i canoni da "
                                         "marzo. Il locatore ha distrutto il cancello del cortile. Il conduttore deve "
                                         "pagare i canoni.")]
    log = []
    out = readback.party_acts(sections, sents, ["Rossi"], log)
    text = out[0][1]
    assert "Il conduttore ha omesso di pagare i canoni da marzo." in text
    assert "Il locatore ha distrutto il cancello del cortile (da verificare)." in text
    assert "Il conduttore deve pagare i canoni." in text                     # a rule, no act
    assert [e.get("mapped") for e in log if e.get("mapped")] == ["S1"] and sum(1 for e in log if e.get("marked")) == 1


# --- 3. the institution named for an article, on every section ------------------------------------------------------------

def test_an_institution_sharing_no_stem_with_the_articles_rubrica_is_marked_in_any_section():
    ctx = _ctx(rubriche={"1062 c.c.": "Destinazione del padre di famiglia", "2051 c.c.": "Danno cagionato da cosa in "
                                                                                       "custodia"},
               texts={"2051 c.c.": "Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia."},
               allowed={"1062 c.c.", "2051 c.c."})
    sections = [("Conclusioni", "Resta la responsabilità del vettore marittimo (art. 2051 c.c.) per il carico perso."),
                ("Risposta e strategia", "Vale la destinazione del padre di famiglia (art. 1062 c.c.) per la strada.")]
    out, log = record_check.check(sections, ctx)
    assert "(da verificare: art. 2051 c.c., rubrica «Danno cagionato da cosa in custodia»)" in out[0][1]
    assert "(da verificare" not in out[1][1]
    assert [m["rule"] for m in log["marked"]] == ["istituto_rubrica"]


# --- 4. the first read: every named party, the persona offesa, the verified re-read ---------------------------------------

def test_the_first_read_names_every_party_and_tops_up_the_ones_without_lines():
    raw = {"cliente": {"nome": ""}, "fatti": ["x"],
           "parti": [{"nome": "Rossi", "ruolo": "imputato"}, {"nome": "la persona offesa", "ruolo": "vittima"}],
           "ipotesi": [{"parte": "Rossi", "linea": "principale", "tesi": "Legittima difesa", "istituti": ["legittima difesa"]}],
           "qualificazioni_alternative": [{"istituto": "eccesso colposo", "rubricato": "Eccesso colposo",
                                           "perche": "se la reazione fu sproporzionata"},
                                          {"istituto": "art. 55 c.p.", "rubricato": ""}]}
    read = case_read.clean(raw)
    assert [p["name"] for p in read["parties"]] == ["Rossi", "la persona offesa"]
    assert [a["name"] for a in read["alternatives"]] == ["eccesso colposo"]         # never an article number
    assert case_read.missing_parties(read) == ["la persona offesa"]
    asked = {}

    def reader(text, request, lang):
        asked["n"] = asked.get("n", 0) + 1
        return case_read.clean({"fatti": ["x"], "ipotesi": [{"parte": "la persona offesa", "linea": "principale",
                                                            "tesi": "Costituzione di parte civile",
                                                            "istituti": ["risarcimento del danno"]}]})
    more = case_read.topup_read("t", "", "it", read, 15.0, reader=reader)
    assert more["topped_up"] == {"parties": ["la persona offesa"], "lines": 1} and asked["n"] == 1
    assert case_read.missing_parties(more) == []
    late = case_read.topup_read("t", "", "it", read, 2.0, reader=reader)
    assert late["topped_up"]["skipped"] and asked["n"] == 1
    assert "persona offesa" in case_read._SYSTEM and "qualificazioni_alternative" in case_read._SYSTEM


def test_a_late_top_up_leaves_the_first_read(monkeypatch):
    monkeypatch.setattr(case_read, "READ_SECONDS", 0.3)
    first = {"client": {"name": ""}, "parties": [{"name": "A"}, {"name": "B"}],
             "hypotheses": [{"party": "A", "line": "principale", "claim": "x"}], "facts": ["x"]}
    pending = case_read.Pending("t", reader=lambda t, r, l: first, topup=lambda *a: time.sleep(1) or {})
    out = pending.result()
    assert pending.timed_out and out["hypotheses"] == first["hypotheses"] and out["topped_up"]["skipped"] == "timeout"


def test_the_reside_line_verifies_the_reread(monkeypatch):
    state = test_v60e._sided_state(monkeypatch)
    state["read"]["hypotheses"] = [dict(h, party="Rossi") for h in state["read"]["hypotheses"][:2]]
    lines = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: lines.append((event, data)))
    monkeypatch.setattr(case_read, "read", lambda t, r="", l="it", client="", parties=(): case_read.clean(
        {"fatti": ["x"], "ipotesi": [{"linea": "principale", "tesi": "Eccezione", "istituti": ["eccezione"]}]}))
    entry = flow.reside(state, "Bianchi")
    assert entry["reread"] and entry["verified"] and entry["parties_read"] == ["Rossi"] and entry["attached"] >= 0
    flow.reside(state, "Bianchi")
    assert [d.get("skipped") for e, d in lines if e == "reside"][-1] and lines[-1][1]["verified"]
    state2 = test_v60e._sided_state(monkeypatch)
    state2["read"]["hypotheses"] = [dict(h, party="Rossi") for h in state2["read"]["hypotheses"][:2]]
    monkeypatch.setattr(case_read, "read", lambda *a, **k: {})
    entry = flow.reside(state2, "Bianchi")
    assert entry["reread"] and not entry["verified"] and entry["why"] == "la rilettura non ha dato linee della parte"


# --- 5. a term only with a topic graded 2+ of its matter ---------------------------------------------------------------

def test_a_term_needs_a_listed_topic_graded_two_or_more_of_its_matter():
    term = {"id": "prescrizione_extracontrattuale", "ref": "2947 c.c."}
    weak = {"title": "Danno", "articles": ["2043 c.c."], "grades": {"2043 c.c.": 1}}
    strong = {"title": "Danno", "articles": ["2043 c.c.", "2051 c.c."], "grades": {"2043 c.c.": 1, "2051 c.c.": 3}}
    own = {"title": "Prescrizione", "articles": ["2947 c.c."], "source": "termine prescrizione"}
    contract = {"title": "Vendita", "articles": ["1490 c.c."], "grades": {"1490 c.c.": 3}}
    assert not topic_step.carries_term(term, [weak, contract])
    assert not topic_step.carries_term(term, [weak, own])                 # its own topic, a grade-1 article: never
    assert topic_step.carries_term(term, [strong, contract])
    assert topic_step.carries_term(term, [{"title": "Danno", "articles": ["2043 c.c."]}])     # no grades: as before
    assert flow.carried_terms([term], [weak]) == []


# --- 6. relative periods as date ranges for terms ------------------------------------------------------------------------

def test_relative_periods_become_ranges():
    today = dt.date(2026, 11, 10)
    cases = {"circa tre mesi fa": (dt.date(2026, 7, 10), dt.date(2026, 9, 10)),
             "qualche settimana fa": (dt.date(2026, 9, 29), dt.date(2026, 10, 27)),
             "negli ultimi sei mesi": (dt.date(2026, 5, 10), today),
             "l'estate scorsa": (dt.date(2026, 6, 1), dt.date(2026, 8, 31)),
             "a fine dell'anno scorso": (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
             "tra marzo e aprile": (dt.date(2026, 3, 1), dt.date(2026, 4, 30))}
    for words, span in cases.items():
        assert fact_signals.parse_date(fact_signals.relative_date(words, today))[:2] == span, words
    far = fact_signals.parse_date(fact_signals.relative_date("da oltre vent'anni", today))
    assert far[1] == dt.date(2006, 11, 10) and far[0] <= dt.date(1996, 11, 10)
    assert fact_signals.relative_date("tre mesi fa", today) == "agosto 2026"          # an exact one stays exact


def test_the_cases_relative_dates_are_resolved_against_its_present_and_terms_run_on_the_range():
    facts = {"events": [{"date": "circa tre mesi fa", "what": "lesioni"}], "present_date": "10 novembre 2026"}
    done = fact_signals.resolve_relative(facts)
    assert done == [{"field": "events.date", "written": "circa tre mesi fa",
                     "now": "periodo dal 10 luglio 2026 al 10 settembre 2026"}]
    assert facts["events"][0]["date_relative"] == "circa tre mesi fa"
    start, end, _ = fact_signals.parse_date(facts["events"][0]["date"])
    present = fact_signals.parse_date(facts["present_date"])[:2]
    assert fact_signals.term_status((start, end), present, months=3) == "INCERTO"
    assert fact_signals.term_status((start, end), present, months=1) == "DECORSO"
    assert fact_signals.term_status((start, end), present, months=6) == "NON DECORSO"
    none = {"events": [{"date": "dopo", "what": "x"}]}
    assert fact_signals.resolve_relative(none) == [] and "present_date" not in none


# --- 7. the alternative qualifications are searched -------------------------------------------------------------------------

def test_the_alternative_qualifications_are_searched_beyond_the_lines_cap():
    read = {"hypotheses": [{"line": "principale", "claim": "c", "institutions": [f"istituto {i}" for i in range(4)]},
                           {"line": "subordinata", "claim": "c", "institutions": [f"altro {i}" for i in range(4)]}],
            "alternatives": [{"name": "eccesso colposo", "rubricated": "Eccesso colposo", "why": "sproporzione"},
                             {"name": "istituto 0", "rubricated": "", "why": "x"}]}
    found = flow.read_institutions(read)
    assert len(found) == flow.FIRST_READ_MAX + 1 and found[-1] == {
        "name": "eccesso colposo", "rubricated": "Eccesso colposo", "claim": "sproporzione", "alternative": True}
    rows, log = flow.first_read_search(None, "db", [], found, ["c.p."], session=object(),
                                       resolve=lambda own, docs, q: [])
    assert log["alternatives"] == ["eccesso colposo"]
