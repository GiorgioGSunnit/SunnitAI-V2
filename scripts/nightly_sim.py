#!/usr/bin/env python3
"""v70: the nightly -- a five-cell sim with structural checks, not one score. The lesson of the v69 batch: the rubric
moved +0.8 while two sections went to zero; a mean hides a section that vanished. Each night, per posture cell
(tests/data/nightly_cells.json: civile/attore/pre_filing, civile/convenuto/primo_grado, penale/imputato/primo_grado,
penale/persona_offesa/pre_filing, lavoro/attore/pre_filing):

  1. intake: the cell read from the case as the API reads it (intake.request_intake, logged; v73: a cell without the
     intake line fails) -- a wrong rito, role or stage fails the cell
  2. the parere (interactive_flow.request_case_first: the flow with that intake, as the API runs it), timed: the thesis selection's and the writer's
     seconds (LLM_THINKING=per_call)
  3. its structure (lawyer_method.structure_checks): his ten sections present and non-empty, §4's tables, §6's table
     with the stragiudiziale option, three objections, §9's esito / grado / obiettivo, no «da verificare» outside §10;
     by the cell: the claimant's pre-filing mediazione / ATP / 700, the imputato's riti alternativi, the ratione
     temporis line
  4. once: the stage lists against v59 for the default cell (scripts/stage_diff.py)

The report (--out, markdown) has one row per cell and one column per check -- a check that passed yesterday and fails
today is in the «Regressions» list (--baseline: yesterday's --json-out). Exit 1 on any failure.

v77: the acceptance is this table for structure and regressions, and the lawyer's scores on the held-out set for
substance (scripts/heldout_export.py, scripts/heldout_score.py) -- the 31-case sim (diagnose_long --interactive-sim) is no
measure any more. Each run records the VERSION it ran (the report's first line, the JSON's "version"); a live cell's
answer must end with that version's footer (check version_footer); the report ends with the lawyer's latest mean per
version (the held-out folder's scores.jsonl: --heldout-dir, else HELDOUT_DIR, else /opt/chatbot/heldout).

v78 (RULINGS_RECORD on): every penal cell, and every civil cell whose charge is a c.c. fattispecie of the record, has at
least one ruling printed with its quoted passage in §5 (check ruling_passage; a cell with none fails); the report adds
the rulings record per cell (per article found / linked / kept, estremi parsed / unparsed, passages quoted / none,
orientations computed / to verify) and the penal cells' §5 verbatim.

v79 (B3): check grounded_output -- in the delivered parere, no rule statement without a source (a record element, a
store field, a registry passage: src/rag/rule_statements.py) and no article outside the answer's scope.

v79.6: checks questions_answered (every question intake extracted answered in «Risposte ai quesiti», each answer
grounded) and alternative_elements (an element with branches graded as its branches say); a sixth cell, two
co-defendants, whose per-client thesis counts get a row each.

v79 (F3): check rulings_printed -- no registry label («G1»), no «copia non ufficiale» stamp, no ruling cited
without its head («Sez. 5, n. 12/2025» with no «Cass.»).

v79 (E2, E3): check reader_wording -- no element code outside the tables, no «registro», no «l'accusa» or
«aggravante» outside a criminal matter; check ledger_removed_absent -- every sentence the ledger logs as removed is
absent from the parere. v79 (D4): the selection table adds the defence lines, the defence articles the general-part gate
dropped (criminal cells), the ranked candidates.

v79.8.6: check no_registro -- zero occurrences of «registro» in any answer, any section (the E2 rule, absolute).

v79.8: per cell, every block's counters as the interactive_single_shot_timings line carries them (A: graded_articles,
samples_used, llm_calls, total_s, the grading funnel; B-E: theirs), «before → now» against the baseline.

v79.7: per cell, the retrieval's top-ranked articles and what became of each (in record / candidate / dropped, with the
reason: interactive_flow.ranked_trace); the rulings table adds, per article, how many rulings each filter dropped (the
rubrica's noun, the rule verb, the procedural marker, the element's nouns, the Sez. 7 / ordinanza rank); the counts add
where the charge's contests came from (threats_from) and, when no record was marked the charge, why (no_threat).
--baseline takes a file or a folder (its nightly.json, else its newest .json) and fails before any cell runs when it
cannot be read.

v79 (A2, A3): per cell, the thesis selection's counts (candidates offered / parsed / valued / barred / unverifiable /
minoritaria, the main line chosen, whether it is the fallback) in their own table, compared with the baseline's;
--deterministic runs the thesis call once at temperature 0 (THESIS_DETERMINISTIC) for a version-to-version comparison
-- the default stays best-of-3.

v79.11: --element-schemas runs the cells with ELEMENT_SCHEMAS on (the per-article element schemas,
src/rag/element_schemas.py; the mode says so), to be compared with a flag-off run passed as --baseline. Per cell, the
records by element path (schema / reviewed / text, with why the text path ran: state["element_paths"]) and the shadow
admission (agree, false exclusions with their reasons, extra, candidate recall: state["schema_shadow"]); the «v79.11
headline» table, «before → now» side by side: elements_per_article, fragments_in_tables, mental_integrated_by_conduct,
duty_polarity_errors, unadmitted_candidates, main_lines_by_rule, candidate_recall, false_exclusions. With the flag on,
the schemas' coverage by code (usable / unusable / carded without one: the generated file against the cards file) and
their quality against the lawyer's reviews (element_schemas.quality_report: «nessuna revisione» when none). A baseline
without these keys renders («—»).

v79.10 (A5): the family gate's alarm, FIRST in the report (right after the header): per cell, every document the gate
dropped that held FAMILY_ALARM_MIN (5) or more graded-relevant articles -- «we just cut the governing law»; read it before
any promote. The run sets FAMILY_GATE_SHADOW_GRADE=true (the dropped rows graded in shadow, recorded only;
--no-family-shadow-grade skips it: a proxy count, «stima (non graduati)»). The alarm alone never fails the run; the
header line says «ALLARME presente». The family table lists the override doors (door and signal).

v79.10.2 (B): per cell, questions_extracted (one question at least: a «?», a request phrased as one, a task -- «illustra»,
«valutando se» …; the run's counter, else the cell's text) and clients_sections_once (no section of a several-client
parere printed twice, lawyer_method.printed_twice), both offline too; the counters role_resisting, reductions_offered,
questions_extracted, clients_detected, clients_shared_sections, clients_sections_twice in the counters table.

v79.10.3 (C): per cell, rites_and_alternatives_fit -- the counters rites_unavailable_offered (a sentence offering a rito
the charge's pena from the store does not admit, or one whose pena is unknown without «da verificare sulla pena
edittale») and requalify_event_mismatch (a riqualificazione the certain facts' event does not fit: a death and 590,
injuries and no death and 575/589) both 0; the counters, with rites_precluded, rites_to_verify and
requalify_refused_event, in the counters table (the run's state only: offline cells have no counters).

v79.10.4 (D): the counters termini_graded and termine_excluded_inside (must be 0: a term «escluso» while a fact gives a
duration inside it), cap_anchors_kept (the similar rulings' articles and the anchor's own section the rank alone would
have cut), counterparty_elements, counterparty_breaches and counterparty_contradicted (must be 0: the counterparty's
duty «non risulta» on its breach, on the client's record) in the counters table; the fragments list reads D4's kinds
(a category, a confiscation or medicinal clause, a modal permission).

v79.10.5 (E): check names_redacted -- the names left in the quoted passages (the capitalised tokens after «sig.»,
«dott.», «avv.», the pairs of capitalised tokens outside the gazetteer: rulings.unredacted_names), counted, must be 0,
offline too; ledger_removed_absent reads answer_ledger.removed_present, the flow's settle's own test, and must be 0. The
counters charge_rule (the rule that chose the charge: user_article / read_hypothesis / rubrica_match / none),
charge_thesis_refused (the thesis call's legal question naming another charge, replaced), names_redacted,
names_unredacted (0), ledger_struck, ledger_kept and ledger_removed_present (0) in the counters table.

v79.10.6 (E review): names_redacted counts the passages' names with its own literal reading (rulings.names_left: a
title and the name after it, particles included; two adjacent capitalised words outside the gazetteer), never the
redactor's name_spans; ledger_removed_absent counts a removed sentence only where its rule says it is absent
(answer_ledger.misplaced: the one check's section, a repetition's later occurrences, the client's own text) -- the
entries are never relabelled to read 0; charge_thesis_refused counts §1's refused summary too.

v79.12.1 (modules, code scope): the counters modules_active (per cell, one row per module active: «signal: words» --
the signal that switched it on: istituto / articolo / parole dell'utente / schema, the answer stage's, else the
retrieval's; never a claim sentence or a damage word alone) and code_articles_out_of_scope (the code articles of a
Titolo / Capo outside the case's institutions kept from grading: topic_groups.code_scope_gate) in the counters table
(interactive_flow.case_counters; the run's state only).

v79.12.1 (defence records, negated elements, pertinence): the counters negated_integrated_by_presence (must be 0: an
absence-phrased element «provato» with no supporting fatto certo stating the absence) and negated_elements
(issue_records.negated_counters), promoted_by_pertinence (the main line promoted over realismo's choice by pertinence:
lawyer_method.promote_by_pertinence) and main_outside_institution (a main line on the client's own norm outside the
first read's institution -- must be 0) in the counters table (interactive_flow.case_counters).

  python scripts/nightly_sim.py --env .env --out nightly.md --json-out nightly.json --baseline last.json
  python scripts/nightly_sim.py --env .env --deterministic --out det.md --json-out det.json --baseline det.prev.json
  python scripts/nightly_sim.py --env .env --element-schemas --out es.md --json-out es.json --baseline nightly.json
  python scripts/nightly_sim.py --answers DIR      # offline: DIR/<cell id>.md, the checks only (no model, no DB)

Needs the server's environment (Neo4j, the model, LAWYER_METHOD=true, CASE_FIRST_SINGLE_SHOT=true, ...) except with
--answers. Proposed schedule (prod, UTC; outside the cards' GPU window, 21:00-22:30):
  15 2 * * * cd /opt/chatbot && cp -f /var/log/maint/nightly.json /var/log/maint/nightly.prev.json; venv/bin/python scripts/nightly_sim.py --env .env --out /var/log/maint/nightly.md \
      --json-out /var/log/maint/nightly.json --baseline /var/log/maint/nightly.prev.json >> /var/log/maint/nightly.log 2>&1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELLS_FILE = ROOT / "tests" / "data" / "nightly_cells.json"
CELL_KEYS = ("rito", "role", "stage")


def cells(path: Path = CELLS_FILE) -> List[Dict[str, Any]]:
    return [c for c in json.loads(path.read_text(encoding="utf-8")).get("cells") or [] if c.get("id")]


def cell_check(detected: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    wrong = {k: detected.get(k) for k in CELL_KEYS if expected.get(k) and detected.get(k) != expected[k]}
    return {"check": "intake_cell", "ok": not wrong, "why": "" if not wrong else f"read {wrong}"}


def intake_lines():
    """v73: the [VERBOSE:intake] lines logged while the block runs (intake.vlog wrapped, then restored)."""
    from contextlib import contextmanager
    from src.rag import intake

    @contextmanager
    def capture():
        seen: List[Dict[str, Any]] = []
        original = intake.vlog

        def vlog(event, data=None, duration_ms=None, **kwargs):
            if event == "intake":
                seen.append(data or {})
            return original(event, data, duration_ms, **kwargs)
        intake.vlog = vlog
        try:
            yield seen
        finally:
            intake.vlog = original
    return capture()


def intake_line_check(seen: List[Dict[str, Any]], state: Dict[str, Any], route: str) -> Dict[str, Any]:
    """v73: the cell ran as the API runs it -- the intake line logged once, before the flow, the flow given it (not
    reading its own), the API's route the case-first one. A cell without it fails."""
    from src.rag import intake
    why = ("no [VERBOSE:intake] line" if not seen else
           f"the flow read the intake itself ({state.get('intake_source')!r})" if state.get("intake_source") != "request"
           else f"the API would route it {route!r}, not case-first" if route != intake.CASE_FIRST else "")
    return {"check": "intake_line", "ok": not why, "why": why}


def run_cell(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The cell live: intake, the parere, the checks; the error, when a step fails, as a failed check. v73: the intake
    and the flow as the API runs them (interactive_flow.request_case_first: intake.request_intake, then the flow)."""
    from src.rag import interactive_flow, lawyer_method
    out: Dict[str, Any] = {"id": spec["id"], "cell": spec["cell"], "checks": []}
    t0 = time.time()
    try:
        with intake_lines() as seen:
            result = interactive_flow.request_case_first(spec["text"], "it", case_id=spec["id"])
        detected = result["intake"]
        out["intake"] = {k: detected.get(k) for k in CELL_KEYS}
        print(f"{spec['id']}: intake {detected}", file=sys.stderr)
        out["checks"].append(intake_line_check(seen, result.get("state") or {}, result.get("api_route") or ""))
        out["checks"].append(cell_check(detected, spec["cell"]))
        out["answer"] = result.get("answer") or ""
        out["thesis_counts"] = dict((result.get("state") or {}).get("thesis_counts") or {})      # v79 (A2)
        if spec["cell"].get("family") == "penale":      # v79 (D4): the defence articles the general-part gate dropped
            gate = (result.get("state") or {}).get("defence_gate") or {}
            out["thesis_counts"]["defence_gate_dropped"] = len(gate.get("dropped") or [])
            out["defence_gate"] = gate
        out["ranked_trace"] = list((result.get("state") or {}).get("ranked_trace") or [])        # v79.7 (item 3)
        out["family_gate"] = dict((result.get("state") or {}).get("family_gate") or {})          # v79.9 (C2)
        out["fragments_surviving"] = interactive_flow.surviving_fragments(result.get("state") or {})   # v79.9 (D2)
        st = result.get("state") or {}                  # v79.8: every block's counters, the timings line's
        out["element_paths"] = dict(st.get("element_paths") or {})                              # v79.11
        out.update(shadow_of(st))                                                               # v79.11
        out["counters"] = dict((st.get("single_shot_timings") or {}).get("counters") or
                               interactive_flow.case_counters(st, round(time.time() - t0, 1)))
        timings = (result.get("state") or {}).get("timings") or {}
        out["timings"] = {"total_s": round(time.time() - t0, 1),
                          **{k: timings.get(k) for k in ("thesis_selection_s", "writer_s", "answer_s")}}
        out["checks"] += lawyer_method.structure_checks(out["answer"], spec["cell"])
        out["checks"].append(footer_check(out["answer"]))
        grounded = grounded_check(out["answer"], result.get("state") or {})      # v79 (B3)
        out["checks"].append(grounded)
        out["output_check"] = grounded.get("found") or {}
        out["checks"].append(wording_check(out["output_check"], result.get("state") or {}))     # v79 (E2)
        out["checks"].append(ledger_check(out["answer"], result.get("state") or {}))            # v79 (E3)
        out["checks"].append(printed_check(out["answer"]))                                      # v79 (F3)
        out["checks"].append(names_check(out["answer"]))                                        # v79.10.5 (E2)
        out["checks"].append(registro_check(out["answer"]))                                     # v79.8.6
        out["checks"].append(no_action_check(out["answer"]))                                    # v79.10.1 (A1)
        out["checks"].append(questions_check(out["answer"], result.get("state") or {}))         # v79.6 (1)
        out["checks"].append(questions_extracted_check(spec["text"], out["counters"]))          # v79.10.2 (B2)
        out["checks"].append(clients_once_check(out["answer"]))                                 # v79.10.2 (B3)
        out["checks"].append(rites_fit_check(out["counters"]))                                  # v79.10.3 (C3)
        out["checks"].append(alternatives_check(result.get("state") or {}))                     # v79.6 (4)
        by_client = (result.get("state") or {}).get("thesis_counts_by_client") or {}
        if by_client:                                                                          # v79.6 (2)
            out["thesis_counts_by_client"] = by_client
        from src.rag import rulings
        if rulings.enabled():
            record = (result.get("state") or {}).get("rulings_record")
            out["rulings"] = (record or {}).get("log") or {}
            out["rulings"].update({k: (record or {}).get(k) for k in ("charge", "charge_cc_fattispecie")})
            check = ruling_check(out["answer"], spec["cell"], record)
            if check:
                out["checks"].append(check)
    except Exception as exc:          # noqa: BLE001 -- a failed cell is a failed check, the others still run
        out["checks"].append({"check": "run", "ok": False, "why": f"{type(exc).__name__}: {exc}"[:300]})
    return out


def footer_check(answer: str) -> Dict[str, Any]:
    """v77: the answer ends with «SunnitAI <version>», this checkout's VERSION."""
    from src import version
    ok = version.has_footer(answer, version.current())
    last = (answer or "").rstrip().rsplit("\n", 1)[-1].strip()[:80]
    return {"check": "version_footer", "ok": ok, "why": "" if ok else f"last line «{last}», not «{version.footer()}»"}


def grounded_check(answer: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """v79 (B3): no rule statement without a source, no article outside the scope, in the delivered parere
    (lawyer_method.output_check, on the sources compose kept in the state). A parere the method did not write has no
    sources: the check fails, saying so."""
    from src.rag import lawyer_method
    out = (state or {}).get("output_sources")
    if not out:
        return {"check": "grounded_output", "ok": False, "why": "no sources: the method was not applied", "found": {}}
    found = lawyer_method.output_check(answer, out)
    why = "; ".join([f"{len(found['unsourced'])} rule statements without a source: "
                     + " | ".join(x[:100] for x in found["unsourced"][:3])] * bool(found["unsourced"])
                    + [f"articles outside the scope: {', '.join(found['out_of_scope'][:6])}"] * bool(found["out_of_scope"]))
    return {"check": "grounded_output", "ok": not why, "why": why, "found": found}


def wording_check(found: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """v79 (E2): no internal wording in the delivered parere -- no element code outside the tables, no «registro»,
    no «l'accusa» or «aggravante» outside a criminal matter (lawyer_method.output_check's "wording")."""
    if not (state or {}).get("output_sources"):
        return {"check": "reader_wording", "ok": False, "why": "no sources: the method was not applied"}
    hits = list((found or {}).get("wording") or [])
    return {"check": "reader_wording", "ok": not hits, "why": "; ".join(hits[:6])}


def ledger_check(answer: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """v79 (E3): every sentence the removal ledger logs as «removed» is absent from the delivered parere (a sentence
    under 25 characters is not checked: too short to tell). v79.10.5 (E3): the test is answer_ledger.removed_present,
    the one the flow's settle reads (the count must be 0). v79.10.6 (E3): read where each removal holds
    (answer_ledger.misplaced)."""
    from src.rag import answer_ledger
    entries = ((state or {}).get("ledger") or {}).get("entries") or []
    present = [f"{e.get('step')}: {str(e.get('sentence'))[:80]}"
               for e in answer_ledger.removed_present(answer or "", entries)]
    return {"check": "ledger_removed_absent", "ok": not present, "count": len(present),
            "why": f"{len(present)}: " + "; ".join(present[:3]) if present else ""}


def names_check(answer: str) -> Dict[str, Any]:
    """v79.10.5 (E2): no party's name left in a quoted passage -- the name after a title (its particles too), the pairs
    of capitalised tokens outside the gazetteer; counted, must be 0. v79.10.6 (E2): rulings.unredacted_names reads them
    literally (rulings.names_left), never through the redactor's own name_spans, so a redactor's miss is counted.
    Needs no state."""
    from src.rag import rulings
    found = rulings.unredacted_names(answer or "")
    return {"check": "names_redacted", "ok": not found, "count": len(found),
            "why": f"{len(found)}: " + "; ".join(found[:5]) if found else ""}


def printed_check(answer: str) -> Dict[str, Any]:
    """v79 (F3): the rulings as printed -- no registry label («G1»), no PDF stamp, no citation without its head."""
    from src.rag import rulings
    found = rulings.printed_problems(answer)
    return {"check": "rulings_printed", "ok": not found, "why": "; ".join(found[:4])}


def registro_check(answer: str) -> Dict[str, Any]:
    """v79.8.6: zero occurrences of «registro» in the answer -- any form, any section (the E2 wording rule, absolute)."""
    from src.rag import lawyer_method
    found = lawyer_method.registro_hits(answer)
    return {"check": "no_registro", "ok": not found,
            "why": f"{len(found)}: " + " | ".join(f"…{x}…" for x in found[:3]) if found else ""}


def no_action_check(answer: str) -> Dict[str, Any]:
    """v79.10.1 (A1): the no-action sentence at most once, never as «Linea principale», never with a forecast
    (lawyer_method.no_action_problems)."""
    from src.rag import lawyer_method
    found = lawyer_method.no_action_problems(answer)
    return {"check": "no_action_prose", "ok": not found, "why": "; ".join(found)}


def questions_check(answer: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """v79.6 (item 1): every question intake extracted has a non-empty answer in «Risposte ai quesiti», and every
    answer passes the grounded output check (no rule statement without a source, no article outside the scope)."""
    from src.rag import lawyer_method
    asked = list(((state or {}).get("intake") or {}).get("questions") or [])
    got = lawyer_method.answers_of(answer)
    why = []
    if len(got) < len(asked):
        why.append(f"{len(asked)} questions, {len(got)} answers")
    why += [f"empty answer to «{q[:60]}»" for q, a in got if not a.strip()]
    out = (state or {}).get("output_sources")
    if out and got:
        found = lawyer_method.output_check(f"{lawyer_method.ANSWERS_TITLE}\n\n" + "\n".join(
            f"{i}. «{q}» — {a}" for i, (q, a) in enumerate(got, 1)), out)
        why += [f"answer without a source: {x[:80]}" for x in found["unsourced"]]
        why += [f"answer cites outside the scope: {x}" for x in found["out_of_scope"]]
    return {"check": "questions_answered", "ok": not why, "why": "; ".join(why[:4])}


def questions_extracted_check(text: str, counters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v79.10.2 (B2): the cell yields one question at least -- the run's counter questions_extracted, else the cell's
    text read as the intake reads it (intake.questions: «?», a request phrased as a question, a task)."""
    from src.rag import intake
    got = (counters or {}).get("questions_extracted")
    n = int(got) if isinstance(got, int) else len(intake.questions(text or ""))
    return {"check": "questions_extracted", "ok": n >= 1, "why": "" if n >= 1 else "no question extracted"}


def clients_once_check(answer: str) -> Dict[str, Any]:
    """v79.10.2 (B3): no section of a several-client parere prints the same text twice -- identical for the clients, it
    prints once with the note (lawyer_method.printed_twice); a single-client parere passes."""
    from src.rag import lawyer_method
    found = lawyer_method.printed_twice(answer)
    return {"check": "clients_sections_once", "ok": not found, "why": "; ".join(found[:4])}


def rites_fit_check(counters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v79.10.3 (C3): no rito the charge does not admit offered (rites_unavailable_offered) and no riqualificazione the
    event does not fit (requalify_event_mismatch) -- both 0; a counter the run did not set reads 0."""
    counters = counters or {}
    why = [f"{k}={counters.get(k)}" for k in ("rites_unavailable_offered", "requalify_event_mismatch")
           if int(counters.get(k) or 0)]
    return {"check": "rites_and_alternatives_fit", "ok": not why, "why": "; ".join(why)}


def alternatives_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """v79.6 (item 4): every element with branches is graded as its branches say (issue_records.branch_combine) --
    «integrato» if a branch is, «escluso» only if every branch is."""
    from src.rag import issue_records
    bad = []
    for label, rec in ((state or {}).get("records") or {}).items():
        for r in rec.get("requisites") or []:
            got = r.get("branch_status") or {}
            if not got or all(v == "da provare" for v in got.values()):
                continue
            want = issue_records.branch_combine(got.values())
            if want == "provato" and r.get("status") != "provato" or want == "non risulta" and r.get("status") != want \
                    or want != "non risulta" and r.get("status") == "non risulta":
                bad.append(f"{label} {r.get('id')}: {r.get('status')} (alternative: {want})")
    return {"check": "alternative_elements", "ok": not bad, "why": "; ".join(bad[:4])}


def section5(answer: str) -> str:
    from src.rag import lawyer_method
    return dict(lawyer_method.answer_sections(answer)).get(lawyer_method.title("orientamenti")) or ""


def ruling_check(answer: str, cell: Dict[str, Any], record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v78: a penal cell, or a civil cell charged with a c.c. fattispecie, needs a ruling quoted in §5; None when the
    cell is neither (no check)."""
    from src.rag import rulings
    penal = cell.get("family") == "penale"
    if not penal and not (record or {}).get("charge_cc_fattispecie"):
        return None
    n = rulings.quoted(section5(answer))
    why = "" if n else ("no rulings record" if record is None else
                        f"§5 quotes no passage (kept {len(record.get('kept') or [])}, charge {record.get('charge')})")
    return {"check": "ruling_passage", "ok": bool(n), "why": why}


def rulings_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v78: per cell, the rulings record's counts; then the penal cells' §5 verbatim."""
    rows = [c for c in results if c.get("rulings") is not None]
    if not rows:
        return []
    from src.rag import rulings
    lines = ["", "## Rulings in the record (per cell, per article)", "",
             "Filters (v79.7): how many rulings (rubrica, rank, shortlist, link) or sentences (length, reference, verb, "
             "procedural, nouns) each one dropped.", "",
             "| cell | article | found | linked | graded | kept | passages | " + " | ".join(rulings.FILTERS) + " |",
             "|---" * (7 + len(rulings.FILTERS)) + "|"]
    for c in rows:
        for label, a in (c["rulings"].get("articles") or {}).items():
            f = a.get("filters") or {}
            lines.append(f"| {c['id']} | {label} | {a.get('found', 0)} | {a.get('linked', 0)} | {a.get('graded', 0)} | "
                         f"{a.get('kept', 0)} | {a.get('passages', '—')} | "
                         + " | ".join(str(f.get(k, '—')) for k in rulings.FILTERS) + " |")
    lines += ["", "| cell | estremi parsed | unparsed | section unmapped | passages quoted | none | orientations "
                  "computed | to verify | signatures stripped |", "|---|---|---|---|---|---|---|---|---|"]
    for c in rows:
        r = c["rulings"]
        e, p, o = r.get("estremi") or {}, r.get("passages") or {}, r.get("orientation") or {}
        lines.append(f"| {c['id']} | {e.get('parsed', 0)} | {e.get('unparsed', 0)} | {e.get('section_unmapped', 0)} | "
                     f"{p.get('quoted', 0)} | {p.get('none', 0)} | {o.get('computed', 0)} | {o.get('da_verificare', 0)} | "
                     f"{r.get('signatures_stripped', '—')} |")
    for c in results:
        if (c.get("cell") or {}).get("family") == "penale" and c.get("answer"):
            lines += ["", f"### §5 of {c['id']}", "", section5(c["answer"]) or "(no §5)"]
    return lines


COUNT_COLUMNS = ("offered", "parsed", "valued", "barred", "unverifiable", "minoritaria", "chosen", "main_is_fallback",
                 "defence_lines", "defence_gate_dropped", "ranked_candidates",          # v79 (D4)
                 "threats_from", "no_threat")                                           # v79.7 (item 2)


def counts_lines(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]] = None) -> List[str]:
    """v79 (A2): per cell, the thesis selection's counts -- «before → now» where the baseline's differ."""
    rows = [c for c in results if c.get("thesis_counts") or c.get("thesis_counts_by_client")]
    if not rows:
        return []
    before = {c["id"]: c.get("thesis_counts") or {} for c in (baseline or {}).get("cells") or []}

    def cell(cid: str, key: str, now: Any) -> str:
        was = before.get(cid, {}).get(key)
        now_s = ("yes" if now else "no") if isinstance(now, bool) else str(now if now is not None else "—")
        if cid in before and key in before[cid] and was != now:
            was_s = ("yes" if was else "no") if isinstance(was, bool) else str(was)
            return f"{was_s} → {now_s}"
        return now_s
    lines = ["", "## Thesis selection (per cell)", "", "| cell | " + " | ".join(COUNT_COLUMNS) + " |",
             "|---" * (len(COUNT_COLUMNS) + 1) + "|"]
    for c in rows:
        t = c.get("thesis_counts") or {}
        lines.append(f"| {c['id']} | " + " | ".join(cell(c["id"], k, t.get(k)) for k in COUNT_COLUMNS) + " |")
        for name, tc in (c.get("thesis_counts_by_client") or {}).items():     # v79.6 (item 2): per client
            key = f"{c['id']} / {name}"
            before.setdefault(key, next((x.get("thesis_counts_by_client", {}).get(name) or {}
                                         for x in (baseline or {}).get("cells") or [] if x.get("id") == c["id"]), {}))
            lines.append(f"| {key} | " + " | ".join(cell(key, k, tc.get(k)) for k in COUNT_COLUMNS) + " |")
    gates = [c for c in rows if (c.get("defence_gate") or {}).get("dropped")]
    if gates:                   # v79 (D4): which defence articles the gate dropped (expected: the irrelevant ones)
        lines += ["", "Defence articles the general-part gate dropped: " + "; ".join(
            f"{c['id']}: {', '.join(c['defence_gate']['dropped'])}" for c in gates)]
    return lines


def _flat(d: Dict[str, Any], head: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{head}{k}."))
        else:
            out[f"{head}{k}"] = v
    return out


def counter_lines(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]] = None) -> List[str]:
    """v79.8: per cell, every block's counters (interactive_flow.case_counters, the timings line's) -- «before → now»
    where the baseline had the counter and it changed."""
    rows = [c for c in results if c.get("counters")]
    if not rows:
        return []
    before = {c["id"]: _flat(c.get("counters") or {}) for c in (baseline or {}).get("cells") or []}
    now = {c["id"]: _flat(c["counters"]) for c in rows}
    keys = list(dict.fromkeys(k for c in rows for k in now[c["id"]]))
    lines = ["", "## Counters (per cell; the timings line's)", "", "| counter | " + " | ".join(c["id"] for c in rows)
             + " |", "|---" * (len(rows) + 1) + "|"]
    for k in keys:
        cells = []
        for c in rows:
            v, was = now[c["id"]].get(k), before.get(c["id"], {}).get(k)
            cells.append(f"{was} → {v}" if k in before.get(c["id"], {}) and was != v else str(v if v is not None
                                                                                              else "—"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return lines


def family_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v79.9 (C2): per cell, every document admitted outside the codes with what admitted it (a default, a module, the
    signal's words, the user naming it), and the rows the family gate dropped."""
    rows = [c for c in results if c.get("family_gate")]
    if not rows:
        return []
    lines = ["", "## Documents outside the codes (per cell; the family gate)", "", "| cell | document | family | admitted by |",
             "|---|---|---|---|"]
    for c in rows:
        for x in c["family_gate"].get("admitted") or []:
            lines.append(f"| {c['id']} | {x.get('doc')} | {x.get('family')} | {x.get('by')} |")
    gone = [(c["id"], x) for c in rows for x in c["family_gate"].get("dropped") or []]
    if gone:
        lines += ["", "Dropped by the family gate: " + "; ".join(
            f"{cid}: {x.get('article')} ({x.get('family')}, {x.get('source')})" for cid, x in gone)]
    doors = [(c["id"], x) for c in rows for x in c["family_gate"].get("overrides") or []]      # v79.10 (A5)
    if doors:
        lines += ["", "Admitted by an override door (family_override): " + "; ".join(
            f"{cid}: {x.get('article')} — {x.get('doc')} (porta {x.get('door')}, segnale {x.get('signal')})"
            for cid, x in doors)]
    return lines


ALARM_TITLE = "ALLARME: legge di disciplina tagliata — da leggere prima di ogni promote"
FAMILY_ALARM_MIN = 5                # src/rag/family_gate.FAMILY_ALARM_MIN, kept here so the report reads a JSON alone


def family_alarm(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.10 (A5): a cell's alarmed documents -- its family_gate «alarm», else computed from «dropped_graded» (graded:
    graded_relevant >= 5; proxy: dropped_articles >= 5); [] for a cell (or a baseline) without them."""
    fam = c.get("family_gate") or {}
    if isinstance(fam.get("alarm"), list):
        return [x for x in fam["alarm"] if isinstance(x, dict)]
    out = []
    for doc, x in (fam.get("dropped_graded") or {}).items():
        if not isinstance(x, dict):
            continue
        proxy = bool(x.get("proxy"))
        n = int((x.get("dropped_articles") if proxy else x.get("graded_relevant")) or 0)
        if n >= FAMILY_ALARM_MIN:
            out.append({"doc": doc, "n": n, "articles": list(x.get("articles") or []), "proxy": proxy})
    return sorted(out, key=lambda a: -a["n"])


def alarm_state(results: List[Dict[str, Any]]) -> str:
    """v79.10 (A5): «ALLARME presente» when a cell has a graded alarm; «ALLARME presente (stima, non graduati)» when only
    proxy ones; "" when none."""
    found = [a for c in results for a in family_alarm(c)]
    if any(not a.get("proxy") for a in found):
        return "ALLARME presente"
    return "ALLARME presente (stima, non graduati)" if found else ""


def family_alarm_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v79.10 (A5): the report's first section -- per cell, each document the family gate dropped holding 5 or more
    graded-relevant articles (the articles listed), or «nessuna»; a proxy (the shadow grading off) labelled «stima (non
    graduati)»; a cell without the family gate's record «nessun dato»."""
    lines = ["", f"## {ALARM_TITLE}", ""]
    if not results:
        return lines + ["- nessuna"]
    for c in results:
        fam = c.get("family_gate")
        alarms = family_alarm(c)
        if not fam:
            lines.append(f"- {c.get('id')}: nessun dato (nessun family gate registrato)")
        elif not alarms:
            graded = fam.get("dropped_graded") or {}
            proxy = any(isinstance(x, dict) and x.get("proxy") for x in graded.values())
            lines.append(f"- {c.get('id')}: nessuna" + (" (stima (non graduati))" if proxy else ""))
        else:
            for a in alarms:
                what = "articoli scartati, stima (non graduati)" if a.get("proxy") else "articoli rilevanti (grado ≥ 2)"
                lines.append(f"- {c.get('id')}: {a.get('doc')} — {a.get('n')} {what}: "
                             + (", ".join(a.get("articles") or []) or "—"))
    return lines


def fragment_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v79.9 (D2): per cell, any fragment (a modal permission, a bare object, a scope or saving clause, a bare subject;
    v79.10.4 D4: a category, a confiscation or medicinal clause) that survived decomposition into a record §4's tables
    print -- must be none."""
    rows = [(c["id"], x) for c in results for x in c.get("fragments_surviving") or []]
    return ["", "## Fragments that survived into a table", ""] + [f"- {cid}: {x}" for cid, x in rows] if rows else []


def ranked_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v79.7 (item 3): per cell, the retrieval's top-ranked articles -- in record / candidate / dropped, with why."""
    rows = [c for c in results if c.get("ranked_trace")]
    if not rows:
        return []
    lines = ["", "## Retrieval top-ranked articles (per cell)", "", "| cell | rank | article | status | why |",
             "|---|---|---|---|---|"]
    for c in rows:
        for t in c["ranked_trace"]:
            lines.append(f"| {c['id']} | {t.get('rank', '—')} | {t.get('article')} | {t.get('status')}"
                         f"{' (linea)' if t.get('line') else ''} | {t.get('why') or ''} |")
    return lines


# --- v79.11: element schemas ----------------------------------------------------------------------------------------------

HEADLINE = ("elements_per_article", "fragments_in_tables", "mental_integrated_by_conduct", "duty_polarity_errors",
            "unadmitted_candidates", "main_lines_by_rule", "candidate_recall", "false_exclusions")


def shadow_of(state: Dict[str, Any]) -> Dict[str, Any]:
    """v79.11: the shadow admission the run kept (lawyer_method's ctx["schema_shadow"]: state["schema_shadow"] whole,
    else the thesis counts' numbers -- selection_counts keeps counts, not lists), and each client's in a multi-client
    cell (state["schema_shadow_by_client"], else the clients' thesis counts) -- {} when there is none (flag off)."""
    st = state or {}
    out: Dict[str, Any] = {}
    shadow = st.get("schema_shadow") or (st.get("thesis_counts") or {}).get("schema_shadow")
    if isinstance(shadow, dict) and shadow:
        out["schema_shadow"] = shadow
    by_client = {name: sh for name, sh in (st.get("schema_shadow_by_client") or {}).items()
                 if isinstance(sh, dict) and sh} or \
        {name: tc.get("schema_shadow") for name, tc in (st.get("thesis_counts_by_client") or {}).items()
         if isinstance(tc, dict) and isinstance(tc.get("schema_shadow"), dict) and tc.get("schema_shadow")}
    if by_client:
        out["schema_shadow_by_client"] = by_client
    return out


def _count(v: Any) -> Any:
    """A count from a number, a list (its length) or a bool; None stays None."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple, set, dict)):
        return len(v)
    return v


def _pairs(items: Any) -> List[Tuple[str, str]]:
    """(article, category) pairs from a shadow's current/schema maps ({cat: [articles]}) or a list of pairs; [] from
    a count (selection_counts' numbers)."""
    if isinstance(items, dict):
        return [(str(a), str(cat)) for cat, arts in items.items() if isinstance(arts, (list, tuple)) for a in arts]
    if not isinstance(items, (list, tuple)):
        return []
    return [(str(x[0]), str(x[1])) for x in items or [] if isinstance(x, (list, tuple)) and len(x) >= 2]


def shadow_recall(shadow: Dict[str, Any]) -> Optional[float]:
    """v79.11 (§6): the current admission's article-categories the schema admits too, over the current ones whose
    article has a schema -- the shadow's own figure when it gives one; None when no current one has a schema."""
    shadow = shadow or {}
    if "candidate_recall" in shadow:
        return shadow.get("candidate_recall")
    if isinstance(shadow.get("recall_of"), int) and not isinstance(shadow.get("recall_of"), bool):  # the counts' shape
        of = shadow["recall_of"]
        return round(int(shadow.get("recall_hit") or 0) / of, 3) if of else None
    if not isinstance(shadow.get("current"), dict) or not shadow.get("current"):
        return None
    no_schema = {str(a) for a in shadow.get("no_schema") or []}
    current = [p for p in _pairs(shadow.get("current")) if p[0] not in no_schema]
    admitted = set(_pairs(shadow.get("schema")))
    return round(sum(1 for p in current if p in admitted) / len(current), 3) if current else None


def _why(why: Any) -> str:
    return "; ".join(str(x) for x in why) if isinstance(why, (list, tuple)) else str(why or "")


def false_exclusions(shadow: Dict[str, Any]) -> List[str]:
    """«article (category): why» for each article-category the current admission admits and the schema refuses; []
    when the shadow kept only their count."""
    out = []
    items = (shadow or {}).get("false_exclusions")
    for x in items if isinstance(items, (list, tuple)) else []:
        if isinstance(x, dict):
            out.append(f"{x.get('article')} ({x.get('cat') or x.get('category')}): {_why(x.get('why'))}")
        elif isinstance(x, (list, tuple)) and x:
            out.append(f"{x[0]}" + (f" ({x[1]})" if len(x) > 1 else "") + (f": {_why(x[2])}" if len(x) > 2 else ""))
        else:
            out.append(str(x))
    return out


def headline_value(c: Dict[str, Any], key: str) -> Any:
    """v79.11: one headline counter of a cell's result -- the counters' (the timings line's), else what the cell kept
    (the shadow admission, the thesis counts' main_admitted_by); None when the run did not have it."""
    counters = c.get("counters") or {}
    shadow_counts = counters.get("schema_shadow") if isinstance(counters.get("schema_shadow"), dict) else {}
    if key in ("candidate_recall", "false_exclusions"):
        if key in shadow_counts:
            return _count(shadow_counts.get(key))
        shadow = c.get("schema_shadow")
        if not isinstance(shadow, dict):
            return None
        return shadow_recall(shadow) if key == "candidate_recall" else (_count(shadow.get("false_exclusions")) or 0)
    if key == "main_lines_by_rule" and key not in counters:
        tcs = list((c.get("thesis_counts_by_client") or {}).values()) or [c.get("thesis_counts") or {}]
        rules = [str(t.get("main_admitted_by") or "—") for t in tcs if isinstance(t, dict) and "main_admitted_by" in t]
        if not rules:
            return None
        return {r: rules.count(r) for r in dict.fromkeys(rules)}
    return counters.get(key)


def _show(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {x}" for k, x in v.items()) if v else "none"
    if isinstance(v, (list, tuple)):
        return str(len(v))
    return str(v)


def headline_lines(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]] = None) -> List[str]:
    """v79.11: per cell, the counters the element schemas are judged on, «before → now» side by side wherever the
    baseline ran the cell (a baseline without a counter shows «—» before the arrow)."""
    rows = [c for c in results if any(headline_value(c, k) is not None for k in HEADLINE)]
    if not rows:
        return []
    before = {c.get("id"): c for c in (baseline or {}).get("cells") or [] if isinstance(c, dict)}
    lines = ["", "## v79.11 headline (per cell; before → now)", "", "| cell | " + " | ".join(HEADLINE) + " |",
             "|---" * (len(HEADLINE) + 1) + "|"]
    for c in rows:
        cells = []
        for k in HEADLINE:
            now = _show(headline_value(c, k))
            cells.append(f"{_show(headline_value(before[c['id']], k))} → {now}" if c["id"] in before else now)
        lines.append(f"| {c['id']} | " + " | ".join(cells) + " |")
    return lines


def _path(entry: Any) -> Tuple[str, str]:
    if isinstance(entry, dict):
        return str(entry.get("path") or "text"), str(entry.get("why") or "")
    return str(entry or "text"), ""


def schema_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v79.11: per cell, the records by element path (schema / reviewed / text, the text path's reasons) and the shadow
    admission (agree, false exclusions with their reasons, extra, candidate recall, articles with no schema)."""
    paths = [c for c in results if c.get("element_paths") or (c.get("counters") or {}).get("element_paths")]
    shadows = [(c["id"], s) for c in results for s in [c.get("schema_shadow")] if isinstance(s, dict) and s] + [
        (f"{c['id']} / {name}", s) for c in results for name, s in (c.get("schema_shadow_by_client") or {}).items()
        if isinstance(s, dict) and s]
    if not paths and not shadows:
        return []
    lines: List[str] = []
    if paths:
        lines += ["", "## Element paths (per cell; v79.11)", "",
                  "| cell | schema | reviewed | text | elements per article (schema · reviewed · text) | text path: why |",
                  "|---|---|---|---|---|---|"]
        detail = []
        for c in paths:
            got = c.get("element_paths") or {}
            if got:
                by: Dict[str, List[str]] = {}
                whys: Dict[str, int] = {}
                for label, entry in got.items():
                    path, why = _path(entry)
                    by.setdefault(path, []).append(f"{label}" + (f" ({why})" if path == "text" and why else ""))
                    if path == "text":
                        whys[why or "—"] = whys.get(why or "—", 0) + 1
                n = {p: len(by.get(p) or []) for p in ("schema", "reviewed", "text")}
                detail.append(f"- {c['id']}: " + "; ".join(f"{p}: {', '.join(v)}" for p, v in by.items()))
            else:               # only the counters' {path: n}
                counted = (c.get("counters") or {}).get("element_paths") or {}
                n = {p: counted.get(p, 0) for p in ("schema", "reviewed", "text")}
                whys = dict((c.get("counters") or {}).get("schema_rejected") or {})
            epa = (c.get("counters") or {}).get("elements_per_article_by_path") or {}
            lines.append(f"| {c['id']} | {n['schema']} | {n['reviewed']} | {n['text']} | "
                         + " · ".join(_show(epa.get(p)) for p in ("schema", "reviewed", "text")) + " | "
                         + (", ".join(f"{w}: {k}" for w, k in whys.items()) or "—") + " |")
        lines += [""] + detail if detail else []
    if shadows:
        lines += ["", "## Shadow admission by the schemas (per cell; v79.11, never authoritative)", "",
                  "| cell | agree | false exclusions | extra | candidate recall | no schema |", "|---|---|---|---|---|---|"]
        for cid, s in shadows:
            lines.append(f"| {cid} | {_show(_count(s.get('agree')))} | {_count(s.get('false_exclusions')) or 0} | "
                         f"{_count(s.get('extra')) or 0} | {_show(shadow_recall(s))} | "
                         f"{_count(s.get('no_schema')) or 0} |")
        gone = [(cid, x) for cid, s in shadows for x in false_exclusions(s)]
        if gone:
            lines += ["", "False exclusions (admitted now, refused by the schema): "
                      + "; ".join(f"{cid}: {x}" for cid, x in gone)]
        extra = [(cid, f"{a} ({cat})") for cid, s in shadows for a, cat in _pairs(s.get("extra"))]
        if extra:
            lines += ["", "Extra (admitted by the schema only): " + "; ".join(f"{cid}: {x}" for cid, x in extra)]
    return lines


def _card_keys(path: Optional[str] = None) -> List[Tuple[str, str]]:
    """The cards file's (doc, article) keys, in its order, each once (ARTICLE_CARDS_FILE, else article_cards'
    default); [] when the file is missing."""
    if path is None:
        from src.rag import article_cards
        path = os.getenv(article_cards.FILE_ENV) or article_cards.DEFAULT_FILE
    if not path or not os.path.isfile(path):
        return []
    keys: Dict[Tuple[str, str], None] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("article") is not None:
                keys[(str(record.get("doc") or ""), str(record["article"]).strip().lower())] = None
    return list(keys)


def schema_coverage_lines(cards_path: Optional[str] = None) -> List[str]:
    """v79.11: the schemas' coverage by code -- carded or generated articles with a usable schema, with an unusable one
    (and which blocking checks raised), carded without one; the lawyer's reviewed overlays beside them
    (element_schemas.generated() against the cards file)."""
    from src.rag import element_schemas as es
    gen = es.generated()
    cards = _card_keys(cards_path)
    table: Dict[str, Dict[str, int]] = {}
    checks: Dict[str, int] = {}

    def row(code: str) -> Dict[str, int]:
        return table.setdefault(code or "—", {"usable": 0, "unusable": 0, "without": 0, "reviewed": 0})
    for key in list(dict.fromkeys(cards + list(gen))):
        line = gen.get(key)
        code = es._code((line or {}).get("code") or key[0])
        if line is None:
            row(code)["without"] += 1
        elif es.usable(line):
            row(code)["usable"] += 1
        else:
            row(code)["unusable"] += 1
            for k in es.blocking(line):
                checks[k] = checks.get(k, 0) + 1
    for label, line in es.reviewed().items():
        if line.get("reviewed") is True and isinstance(line.get("schema"), dict):
            row(label.split(" ", 1)[1] if " " in label else "")["reviewed"] += 1
    lines = ["", "## Element schemas: coverage (by code; v79.11)", "",
             f"Generated {es.schema_file()}, cards {cards_path or 'ARTICLE_CARDS_FILE / default'}.", "",
             "| code | usable | unusable | carded without | reviewed |", "|---|---|---|---|---|"]
    if not table:
        return lines + ["| — | 0 | 0 | 0 | 0 |"]
    total = {k: sum(r[k] for r in table.values()) for k in ("usable", "unusable", "without", "reviewed")}
    for code, r in sorted(table.items()):
        lines.append(f"| {code} | {r['usable']} | {r['unusable']} | {r['without']} | {r['reviewed']} |")
    lines.append(f"| all | {total['usable']} | {total['unusable']} | {total['without']} | {total['reviewed']} |")
    if checks:
        lines += ["", "Unusable by blocking check: " + ", ".join(f"{k} {n}" for k, n in
                                                                  sorted(checks.items(), key=lambda kv: -kv[1]))]
    return lines


QUALITY_MEASURES = ("n", "element_precision", "element_recall", "logic_accuracy", "exception_recall", "comma_accuracy",
                    "grounding_accuracy", "correction_rate", "grounding_rate_auto")


def quality_lines(lines: Optional[List[Dict[str, Any]]] = None,
                  generated_lines: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """v79.11 (§9): the generated schemas against the lawyer's reviews (element_schemas.quality_report over the
    reviewed overlay) -- the nine measures; «nessuna revisione» when none is reviewed."""
    from src.rag import element_schemas as es
    q = es.quality_report(lines, generated_lines)
    out = ["", "## Element schemas: quality against the lawyer's reviews (v79.11)", ""]
    if q.get("note"):
        out += [f"- {q['note']} ({q.get('reviewed', 0)} reviewed lines)", ""]
    out += ["| measure | value |", "|---|---|"]
    out += [f"| {k} | {_show(q.get(k))}" + (f" (of {q.get('explicit_elements', 0)} explicit elements)"
                                            if k == "grounding_rate_auto" else "") + " |" for k in QUALITY_MEASURES]
    return out


def schema_sections() -> List[str]:
    """v79.11: coverage and quality, for a run with ELEMENT_SCHEMAS on; a failure is a line, never the run's end."""
    out: List[str] = []
    for part in (schema_coverage_lines, quality_lines):
        try:
            out += part()
        except Exception as exc:          # noqa: BLE001 -- a measurement, the cells' results stand
            out += ["", f"- {part.__name__}: not computed ({type(exc).__name__}: {exc})"[:300]]
    return out


def lawyer_lines(folder: Optional[Path]) -> List[str]:
    """v77: the lawyer's latest mean per version on the held-out set (src/chatbot/heldout.py)."""
    from src.chatbot import heldout
    folder = folder or heldout.folder()
    table = heldout.means_table(heldout.scores(folder)) if folder and Path(folder).is_dir() else []
    return ["", "## The lawyer's scores per version (held-out set, 0-5)", ""] + (
        table or [f"- none recorded ({folder or 'no held-out folder'})"])


def offline_cell(spec: Dict[str, Any], folder: Path) -> Dict[str, Any]:
    from src.rag import lawyer_method
    path = folder / f"{spec['id']}.md"
    out: Dict[str, Any] = {"id": spec["id"], "cell": spec["cell"], "checks": []}
    if not path.is_file():
        out["checks"].append({"check": "run", "ok": False, "why": f"no {path.name}"})
        return out
    out["answer"] = path.read_text(encoding="utf-8")
    out["checks"] = lawyer_method.structure_checks(out["answer"], spec["cell"])
    out["checks"].append(printed_check(out["answer"]))         # v79 (F3): needs no state
    out["checks"].append(names_check(out["answer"]))           # v79.10.5 (E2): needs no state
    out["checks"].append(registro_check(out["answer"]))        # v79.8.6
    out["checks"].append(no_action_check(out["answer"]))       # v79.10.1 (A1)
    out["checks"].append(questions_extracted_check(spec["text"]))       # v79.10.2 (B2): from the cell's text
    out["checks"].append(clients_once_check(out["answer"]))             # v79.10.2 (B3): needs no state
    from src.rag import rulings
    check = ruling_check(out["answer"], spec["cell"], None) if rulings.enabled() else None
    if check:           # v78: offline, the penal cells only (the record is not in a file)
        out["checks"].append(check)
    return out


def read_baseline(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """v79.7: the baseline run -- a --json-out file, or a folder holding one (its nightly.json, else its newest
    .json); None without --baseline. Anything unreadable raises SystemExit, before any cell runs."""
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        found = p / "nightly.json" if (p / "nightly.json").is_file() else max(
            (x for x in p.glob("*.json") if x.is_file()), key=lambda x: x.stat().st_mtime, default=None)
        if found is None:
            raise SystemExit(f"--baseline {path}: a folder with no .json in it")
        p = found
    if not p.is_file():
        sibling = Path(f"{path}.json")          # «nightly-v79.6» for «nightly-v79.6.json»
        if not sibling.is_file():
            raise SystemExit(f"--baseline {path}: no such file or folder")
        p = sibling
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--baseline {p}: not readable ({type(exc).__name__}: {exc})")
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise SystemExit(f"--baseline {p}: not a nightly --json-out (no «cells»)")
    print(f"nightly: baseline {p} (version {data.get('version') or '—'})", file=sys.stderr)
    return data


def stage_check() -> Dict[str, Any]:
    """The stage lists against v59 (stage_diff.py, a git archive of the v59 commit): skipped -- not failed -- outside a
    git checkout (the server's copy): run it where the repository is."""
    if not (ROOT / ".git").exists():
        return {"ok": None, "report": "skipped: not a git checkout (run scripts/stage_diff.py in the repository)"}
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "stage_diff.py")], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=900)
    return {"ok": done.returncode == 0, "report": done.stdout.strip() or done.stderr.strip()[-2000:]}


def regressions(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> List[str]:
    before = {c["id"]: {k["check"]: k["ok"] for k in c.get("checks") or []} for c in (baseline or {}).get("cells") or []}
    out = []
    for c in results:
        for k in c["checks"]:
            if not k["ok"] and before.get(c["id"], {}).get(k["check"]):
                out.append(f"{c['id']}: {k['check']} ({k['why']})")
    return out


def report(results: List[Dict[str, Any]], stages: Optional[Dict[str, Any]], regressed: List[str], run_version: str = "",
           lawyer: Optional[List[str]] = None, baseline: Optional[Dict[str, Any]] = None, mode: str = "",
           schemas: Optional[List[str]] = None) -> str:
    names = list(dict.fromkeys(k["check"] for c in results for k in c["checks"]))
    alarm = alarm_state(results)                    # v79.10 (A5): the header says it, the alarm section comes first
    lines = [f"# Nightly: {len(results)} cells, structural checks" + (f" — {alarm}" if alarm else ""), "",
             f"Version: {run_version or '—'}"
             + (f" (baseline {baseline.get('version') or '—'})" if baseline else "") + (f"; {mode}" if mode else "")]
    lines += family_alarm_lines(results)
    lines += ["", "| cell | " + " | ".join(names) + " | thesis s | writer s |", "|---" * (len(names) + 3) + "|"]
    for c in results:
        by = {k["check"]: k["ok"] for k in c["checks"]}
        t = c.get("timings") or {}
        lines.append(f"| {c['id']} | " + " | ".join("✓" if by.get(n) else ("✗" if n in by else "·") for n in names)
                     + f" | {t.get('thesis_selection_s') or '—'} | {t.get('writer_s') or '—'} |")
    lines += ["", "## Failures", ""]
    fails = [f"- {c['id']}: {k['check']} — {k['why']}" for c in results for k in c["checks"] if not k["ok"]]
    lines += fails or ["- none"]
    lines += ["", "## Regressions (passed in the baseline, fail now)", ""] + ([f"- {x}" for x in regressed] or ["- none"])
    if stages is not None:
        lines += ["", "## Stage lists against v59 (default cell)", "", stages["report"]]
    lines += counts_lines(results, baseline)
    lines += counter_lines(results, baseline)
    lines += headline_lines(results, baseline)      # v79.11
    lines += ranked_lines(results)
    lines += family_lines(results)                  # v79.9 (C2)
    lines += fragment_lines(results)                # v79.9 (D2)
    lines += schema_lines(results)                  # v79.11
    lines += schemas or []                          # v79.11: coverage and quality (flag on)
    lines += rulings_lines(results)
    lines += lawyer or []
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env")
    ap.add_argument("--answers", type=Path, help="offline: a folder of <cell id>.md answers, the checks only")
    ap.add_argument("--only", nargs="*", help="cell ids")
    ap.add_argument("--out")
    ap.add_argument("--json-out")
    ap.add_argument("--baseline")
    ap.add_argument("--no-stages", action="store_true", help="skip the stage lists' diff against v59")
    ap.add_argument("--heldout-dir", type=Path, help="the held-out folder whose scores the report ends with")
    ap.add_argument("--reasoning-effort", help="v78.1.1: LLM_REASONING_EFFORT for this run (the calls with thinking on)")
    ap.add_argument("--deterministic", action="store_true",
                    help="v79: the thesis call once at temperature 0 (THESIS_DETERMINISTIC), for version-to-version runs")
    ap.add_argument("--element-schemas", action="store_true",
                    help="v79.11: ELEMENT_SCHEMAS on for this run (compare with a flag-off run as --baseline)")
    ap.add_argument("--no-family-shadow-grade", action="store_true",
                    help="v79.10 (A5): skip the family gate's shadow grading (the alarm on a proxy count)")
    args = ap.parse_args(argv)
    if not args.answers and os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    if args.reasoning_effort is not None:       # v78.1.1: over the env file's, for this run
        os.environ["LLM_REASONING_EFFORT"] = args.reasoning_effort
    if args.deterministic:                      # v79 (A3): over the env file's, for this run
        os.environ["THESIS_DETERMINISTIC"] = "1"
    if args.element_schemas:                    # v79.11: over the env file's, for this run
        os.environ["ELEMENT_SCHEMAS"] = "true"
    # v79.10 (A5): the family gate's dropped rows graded in shadow, for the alarm -- over the env file's, for this run
    os.environ["FAMILY_GATE_SHADOW_GRADE"] = "false" if args.no_family_shadow_grade else "true"
    baseline = read_baseline(args.baseline)             # v79.7: before any cell runs
    specs = [c for c in cells() if not args.only or c["id"] in args.only]
    results = [offline_cell(s, args.answers) if args.answers else run_cell(s) for s in specs]
    stages = None if args.no_stages else stage_check()
    regressed = regressions(results, baseline)
    from src import version
    run_version = version.current()
    print(f"nightly: version {run_version}", file=sys.stderr)
    mode = "deterministic (one thesis sample, temperature 0)" if os.getenv("THESIS_DETERMINISTIC") else "best-of-3"
    from src.rag import element_schemas
    schemas = None
    if element_schemas.enabled():               # v79.11: the flag on, by --element-schemas or the env file
        mode += "; element schemas on (ELEMENT_SCHEMAS)"
        schemas = schema_sections()
    text = report(results, stages, regressed, run_version, lawyer_lines(args.heldout_dir), baseline, mode, schemas)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"version": run_version, "mode": mode, "cells": results,
                                                   "stages_ok": None if stages is None else stages["ok"]},
                                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(text)
    failed = any(not k["ok"] for c in results for k in c["checks"]) or (stages is not None and stages["ok"] is False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
