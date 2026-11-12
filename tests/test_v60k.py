"""v60k: single shot on the case-first pipeline (CASE_FIRST_SINGLE_SHOT) -- one message in, one complete parere out:
round 1 inside and never shown; the client stated or assumed (the first line); every graded topic covered, at most 15
articles, the rest in the «non trattate» line with their topic; open points instead of questions, «Punti da precisare»
(at most three, filtered); follow-ups updating the parere with the changed parts marked; the two question defects
fixed in both modes; the timings and the funnel logged; the sim's single-shot mode. The model and the graph are
stubbed."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records, record_check, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402
import test_interactive_flow as base  # noqa: E402
from test_interactive_flow import chat  # noqa: E402,F401  (the /api/chat fixture)
from test_v60f import _graded_state  # noqa: E402
from test_v60g import _ctx  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
NT_PREFIX = C.OTHER_NORMS_LINE.split("{")[0]
WRITER = {flow.ANSWER_TITLES[0]: "Leonardo può far valere la servitù per destinazione del padre di famiglia.",
          NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù quando i fondi erano di un solo proprietario.",
          flow.ANSWER_TITLES[4]: "L'esito dipende dalla prova della separazione dei fondi."}


def _single(monkeypatch, writer=None, link=None, seen=None):
    """flow.analyse_case stubbed with the graded state, the plan, the writer and the link stubbed; the events kept."""
    seen = seen if seen is not None else {}
    seen.setdefault("events", [])
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    state = _graded_state(monkeypatch)
    monkeypatch.setattr(flow, "analyse_case", lambda text, request="", lang="it": state)
    monkeypatch.setattr(flow, "_angle_topic", lambda *a: None)
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])
    monkeypatch.setattr(ptw, "make_plan", lambda *a, **k: {
        "parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
        "subordinate": [], "punti_deboli": [], "questioni": []})
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    monkeypatch.setattr(issue_records, "_link_call",
                        lambda topic, reqs, sents: {r["id"]: v for _, r in reqs for k, v in (link or {}).items()
                                                    if r["id"] == k})

    def fake_write(plan, jobs, lang, named, documents, rows, case, **kwargs):
        seen.setdefault("jobs", []).append(jobs)
        texts = seen.get("writer") or writer or WRITER
        return [(t, texts.get(t, f"Testo di {t}.")) for t, *_ in jobs]
    monkeypatch.setattr(ptw, "write_sections", fake_write)
    real_log = flow._log
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: (seen["events"].append((event, data)),
                                                                    real_log(event, data, ms)))
    return state, seen


def _event(seen, name):
    return next((d for e, d in reversed(seen["events"]) if e == name), None)


# --- the flag ------------------------------------------------------------------------------------------------------------

def test_the_flag_turns_the_case_first_pipeline_on_and_leaves_the_rounds_off(monkeypatch):
    assert not flow.single_shot_enabled() and not flow.turns_on()
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    assert flow.single_shot_enabled() and flow.case_first_enabled() and flow.turns_on() and not flow.enabled()


# --- one message in, one complete parere out -------------------------------------------------------------------------------

def test_one_message_gives_the_complete_parere_with_no_round_one_text(monkeypatch):
    state, seen = _single(monkeypatch)
    out = flow.single_shot("Leonardo, proprietario di un terreno, ... Domitilla ha chiuso il cancello nel 2016.")
    text = out["answer"]
    assert out["kind"] == "answer" and out["state"]["mode"] == "single_shot" and out["state"]["state"] == flow.ANSWERED
    assert "Ho letto il caso" not in text and C.TOPICS_TITLE not in text and "Riassumo come imposterò" not in text
    for title in (flow.ANSWER_TITLES[0], NORMS, flow.ANSWER_TITLES[3], flow.ANSWER_TITLES[4]):
        assert f". {title}**" in text, title
    assert out["state"]["chosen"] == [1, 3, 2]          # v60q: grade 3 first, then the rest -- every one, up to five
    assert "**Altri profili**" not in text                                                 # nothing left uncovered
    ledger = out["state"]["ledger"]
    assert ledger["unaccounted"] == [] and ledger["unreasoned"] == []          # the invariants: no silent deletion
    removed = [e for e in ledger["entries"] if e["kind"] == "removed"]
    assert all(e["step"] == "record_check" and e["rule"] in record_check.REMOVE_RULES for e in removed)
    buckets = [r["answer"]["bucket"] for r in out["state"]["records"].values() if r.get("answer")]
    assert buckets and all(b in (flow.BUCKET_WRITTEN, flow.BUCKET_NOT_APPLICABLE, flow.BUCKET_NOT_TREATED,
                                 flow.BUCKET_CAP) for b in buckets)                # one bucket per article
    timings = _event(seen, "single_shot_timings")
    assert timings["target_s"] == 240 and set(timings) >= {"round1_s", "answer_s", "total_s", "round1_stages",
                                                            "answer_stages", "over_target"}
    funnel = _event(seen, "single_shot_funnel")
    assert funnel["covered"] == [1, 3, 2] and funnel["topics"] == 3 and funnel["asked"] <= 3      # v60q: up to five


def test_the_assumed_client_is_said_in_the_first_line(monkeypatch):
    facts = {**base.FACTS, "parties": [{"name": "Leonardo", "role": "proprietario"},
                                       {"name": "Domitilla", "role": "convenuta, proprietaria"}]}
    state = base._state(monkeypatch, facts)
    assert any(q["id"] == "parte_assistita" for q in state["questions"])
    state["mode"] = "single_shot"
    client, line = flow.settle_client(state, flow._restored_analysis(state["analysis"]))
    assert client == "Domitilla" and line == ("Il parere è reso nell'interesse di Domitilla (convenuta); se assisti "
                                              "un'altra parte, dimmelo.")      # v60r (5): the frame's label
    state["answered_in_reply"] = True                                   # said every time, follow-ups too
    assert flow.settle_client(state, flow._restored_analysis(state["analysis"]))[1] == line


# --- scope: every graded topic, the cap, the rest with its topic ----------------------------------------------------------

def test_the_articles_beyond_the_cap_are_named_with_their_topic(monkeypatch):
    state, seen = _single(monkeypatch)
    monkeypatch.setattr(flow, "SINGLE_SHOT_ARTICLES", 3)
    text = flow.single_shot("Il caso.")["answer"]
    cap = _event(seen, "single_shot_cap")
    assert cap["cap"] == 3 and len(cap["planned"]) == 3 and cap["beyond"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert not any(x.startswith(NT_PREFIX) for x in norms.split("\n"))      # v60r (1): a row of the citations only
    heads = {r["article"]: r for r in state["ledger_rows"] if r["requisite"] is None and r["comma"] is None}
    for item in cap["beyond"]:
        row = heads[item["article"]]
        assert row["status"] == "non trattata" and row["in_body"] is False
        assert row["reason"] == f"tema «{item['topic']}», oltre il limite di 3 articoli"


def test_the_writer_budget_scales_with_the_planned_articles():
    small, large = flow.writer_budget(2, 1), flow.writer_budget(15, 4)
    assert small == {"lead": 900, "norms": 1700, "conclusions": flow.CONCLUSIONS_TOKENS}       # never below the rounds'
    assert large["norms"] == min(flow.NORMS_TOKENS_MAX, 400 + 220 * 15) and large["lead"] > 900
    assert flow.writer_budget(40, 20)["norms"] == flow.NORMS_TOKENS_MAX


# --- open points instead of questions --------------------------------------------------------------------------------------

def test_a_fact_the_case_does_not_state_is_an_open_point_in_its_status_line(monkeypatch):
    state, seen = _single(monkeypatch)
    text = flow.single_shot("Il caso.")["answer"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "con quale atto e in quale anno" not in norms and "- testo, requisito" not in norms   # v60r (1): no row
    row = next(r for r in state["ledger_rows"] if r["article"] == "1062 c.c." and r["requisite"])
    assert row["status"] == C.STATUS_DA_PROVARE                     # v60r (6): the payload's, not «da verificare»
    writer_material = " ".join(str(j) for j in seen["jobs"][-1])
    assert C.OPEN_POINTS_WRITER_TITLE in writer_material and "non fare mai domande nel testo" in writer_material


def test_the_points_block_asks_at_most_three_that_change_something(monkeypatch):
    monkeypatch.setenv(flow.OPEN_POINTS_ENV, "3")          # v60q: no block by default (OPEN_POINTS_MAX 0)
    state, seen = _single(monkeypatch)
    text = flow.single_shot("Il caso.")["answer"]
    assert f"**{C.OPEN_POINTS_TITLE}**" in text
    block = text.split(f"**{C.OPEN_POINTS_TITLE}**", 1)[1]
    assert block.count("\n1. ") == 1 and "\n4. " not in block
    points = _event(seen, "open_points")
    asked = [a["question"] for a in points["asked"]]
    assert asked == ["opera_separazione_0", "possesso_inizio_0", "accesso_alternativo"]        # a term and a verdict first
    assert {"esito", "termine"} <= set(points["asked"][0]["changes"]) and points["asked"][2]["changes"] == ["esito"]
    assert points["dropped"] == [{"question": "atto_separazione", "reason": "oltre i 3 punti"}]
    assert len(points["points"]) == 4 and all(p["point"].startswith("da verificare: il caso non dice") for p in points["points"])


def test_a_point_is_dropped_when_it_changes_nothing_the_case_says_it_or_it_is_generic(monkeypatch):
    monkeypatch.setenv(flow.OPEN_POINTS_ENV, "3")          # v60q: no block by default (OPEN_POINTS_MAX 0)
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    term = "Querela: ora DECORSO (prima DA VERIFICARE)"
    state["questions"] = [
        {"id": "data_fatto", "family": "data_fatto", "template": "data_fatto", "kind": "date",
         "text": C.QUESTIONS["data_fatto"], "unlocks": [term]},
        {"id": "read_1", "kind": "free", "text": "Il caso non dice quando è stata notificata l'ordinanza cautelare: "
         "puoi precisarlo?", "point": "quando è stata notificata l'ordinanza cautelare", "unlocks": [term]},
        {"id": "read_2", "kind": "free", "text": "Il caso non dice se c'erano testimoni: puoi precisarlo?",
         "point": "se c'erano testimoni", "unlocks": []}]
    state["text"] = "L'ordinanza cautelare è stata notificata il 3 marzo 2026."
    got = flow.open_points(state, [])
    reasons = {d["question"]: d["reason"] for d in got["dropped"]}
    assert reasons["read_1"].startswith("nel caso: «L'ordinanza cautelare è stata notificata il 3 marzo 2026")
    assert reasons["read_2"] == "la risposta non cambia termini, procedibilità né esiti"
    assert reasons["data_fatto"] == "generica: il caso indica una data"
    state["text"] = "Tizio è stato aggredito sotto casa."                  # no date: the generic one stands
    assert [a["question"] for a in flow.open_points(state, [])["asked"]] == ["read_1", "data_fatto"]


def test_a_question_in_the_body_is_removed_in_single_shot_only():
    sections = [("Conclusioni", "Il cancello è stato chiuso nel 2016. Quando è iniziato il possesso?")]
    out, log = record_check.check(sections, _ctx(single_shot=True))
    assert out[0][1].strip() == "Il cancello è stato chiuso nel 2016." and log["rules"] == {"domanda_nel_parere": 1}
    out, _ = record_check.check(sections, _ctx())
    assert "Quando è iniziato il possesso?" in out[0][1]


# --- the two question defects, both modes -----------------------------------------------------------------------------------

def test_a_question_the_case_answers_is_never_asked_in_either_mode():
    case = "Mario è stato arrestato. L'ordinanza cautelare è stata eseguita il 2 marzo 2026."
    asked = [{"id": "data_misura", "family": "data_misura", "template": "data_misura", "kind": "date",
              "text": C.QUESTIONS["data_misura"]}]
    log = []
    out = flow.with_read_questions(asked, ["se Mario aveva precedenti"], {}, log, case=case)
    assert [q["id"] for q in out] == ["read_1"]
    assert log[0]["dropped"] == "nel caso" and log[0]["sentence"].startswith("L'ordinanza cautelare è stata eseguita")
    other = "Mario è stato raggiunto da un'ordinanza cautelare."               # the order named, its date not stated
    assert [q["id"] for q in flow.with_read_questions(asked, [], {}, case=other)] == ["data_misura"]
    yes_no = {"id": "accesso_alternativo", "template": "accesso_alternativo", "kind": "yesno",
              "text": "Il fondo di Leonardo ha un altro accesso alla via pubblica, anche se scomodo o insufficiente?"}
    stated = "Il fondo di Leonardo non ha un altro accesso alla via pubblica."
    assert flow.asked_in_case(yes_no, [stated], ["Leonardo"]) == stated
    assert flow.asked_in_case(yes_no, ["Leonardo raggiunge il fondo dalla via pubblica."], ["Leonardo"]) == ""


def test_generic_rule_questions_rank_below_the_specific_ones():
    assert flow.generic_question({"id": "data_fatto", "template": "data_fatto"})
    assert not flow.generic_question({"id": "accesso_alternativo", "template": "accesso_alternativo"})
    assert not flow.generic_question({"id": "read_1", "kind": "free"})
    rules = [{"id": "data_fatto", "template": "data_fatto", "text": C.QUESTIONS["data_fatto"]},
             {"id": "accesso_alternativo", "template": "accesso_alternativo", "text": "Il fondo ha un altro accesso?"}]
    out = flow.with_read_questions(rules, ["se l'arma era detenuta legalmente", "chi ha chiamato i soccorsi"], {})
    assert [q["id"] for q in out] == ["accesso_alternativo", "read_1", "read_2", "data_fatto"]


# --- follow-ups: the parere again, the changed parts marked ---------------------------------------------------------------

def test_a_follow_up_returns_the_full_parere_with_the_changed_parts_marked(monkeypatch):
    state, seen = _single(monkeypatch)
    first = flow.single_shot("Il caso.")
    state = first["state"]
    assert C.UPDATED_MARK not in first["answer"]
    monkeypatch.setattr(flow, "parse_reply", lambda message, st: {
        **flow._clean_reply({"answers": {"atto_separazione": "con un atto di divisione del 1990"}}, st), "how": "model"})
    seen["writer"] = {**WRITER, NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù: i fondi furono divisi nel 1990 con "
                                       "un atto di divisione."}
    turn = flow.follow_up("I fondi furono separati con un atto di divisione del 1990.", state)
    text = turn["answer"]
    assert turn["kind"] == "regenerate" and f". {flow.ANSWER_TITLES[4]}**" in text           # the full parere again
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    changed = next(x for x in norms.split("\n\n") if "nel 1990 con un atto di divisione" in x)
    assert changed.endswith(C.UPDATED_MARK)                                                   # the changed paragraph
    assert "(da verificare: il caso non dice se alla separazione dei fondi esistevano opere visibili" in changed
    assert "Esito sui fatti del caso: da provare (requisiti provati: 1 su 2)." + C.UPDATED_MARK in norms  # v60r
    lead = dict(answer_ledger.sections_of(text))[flow.ANSWER_TITLES[0]]
    assert C.UPDATED_MARK not in lead                                                        # unchanged: not marked
    assert "con quale atto e in quale anno i fondi furono separati" not in norms               # the point answered
    update = _event(seen, "single_shot_update")
    assert update["gave"] and update["marked"] >= 1 and "1062 c.c." in update["labels"]


def test_a_message_the_interpreter_reads_nothing_in_is_an_angle_and_updates_the_parere(monkeypatch):
    state, seen = _single(monkeypatch)
    state = flow.single_shot("Il caso.")["state"]
    monkeypatch.setattr(flow, "parse_reply", lambda message, st: {**flow._clean_reply({}, st), "how": "model"})
    turn = flow.follow_up("E se Domitilla avesse riaperto il cancello?", state)
    assert turn["kind"] == "regenerate" and turn["state"]["angle"] == "E se Domitilla avesse riaperto il cancello?"
    assert flow.follow_up("come funziona?", state)["kind"] in ("regenerate", "flow_help")


def test_mark_updated_marks_only_what_changed_and_what_the_update_touches():
    old = ("**1. Risposta e strategia**\n\nPrimo paragrafo invariato.\n\n**2. Norme applicate ai fatti**\n\n"
           "**Art. 1062 c.c. — Destinazione**\n- co. 1: da provare.\n\nVecchia applicazione.\n\n"
           "**4. Profili processuali**\n\n- Termine: in corso.")
    new = ("**1. Risposta e strategia**\n\nPrimo paragrafo riscritto senza legami.\n\n**2. Norme applicate ai fatti**\n\n"
           "**Art. 1062 c.c. — Destinazione**\n- co. 1: provato.\n\nNuova applicazione dell'art. 1062 c.c.\n\n"
           "**4. Profili processuali**\n\n- Termine: decorso.\n\n**Punti da precisare**\n\n1. Una domanda?")
    log = {}
    out = flow.mark_updated(new, old, {"1062 c.c."}, "", log)
    assert "Primo paragrafo riscritto senza legami." + C.UPDATED_MARK not in out
    assert "**Art. 1062 c.c. — Destinazione**" + C.UPDATED_MARK in out and "- co. 1: provato." + C.UPDATED_MARK in out
    assert "Nuova applicazione dell'art. 1062 c.c." + C.UPDATED_MARK in out
    assert "- Termine: decorso." + C.UPDATED_MARK in out and "1. Una domanda?" + C.UPDATED_MARK not in out
    assert flow.mark_updated(out, out, {"1062 c.c."}, "", {}) == out                         # nothing new: no mark


# --- through /api/chat ----------------------------------------------------------------------------------------------------

def test_the_api_answers_a_case_in_one_parere_and_routes_its_follow_up(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    monkeypatch.setattr(flow, "start", lambda *a, **k: pytest.fail("round 1 shown in single shot"))
    state = base._state(monkeypatch)
    state.update(mode="single_shot", state=flow.ANSWERED)
    monkeypatch.setattr(flow, "single_shot", lambda text, lang="it", request="": {
        "answer": "IL PARERE COMPLETO", "interaction": flow.interaction(state, 3, ("deepen", "add_fact")),
        "state": state, "citations": [], "kind": "answer"})
    body = client.post("/api/chat", json={"message": base.CASE}).json()
    assert calls["single_shot"] == 0 and body["status_messages"] == ["interactive_single_shot"]
    assert body["answer"] == "IL PARERE COMPLETO"
    monkeypatch.setattr(flow, "follow_up", lambda message, st: {
        "answer": "IL PARERE AGGIORNATO", "interaction": flow.interaction(st, 3, ("deepen",)), "state": st,
        "citations": [], "kind": "regenerate"})
    again = client.post("/api/chat", json={"message": "approfondisci il possesso",
                                           "session_id": body["session_id"]}).json()
    assert again["answer"] == "IL PARERE AGGIORNATO" and again["status_messages"] == ["interactive_follow_up"]


# --- the sim, single shot ---------------------------------------------------------------------------------------------------

def test_the_sim_runs_one_call_per_case_and_records_the_timings_and_the_funnel(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    state, seen = _single(monkeypatch)
    monkeypatch.setattr(dl, "answer_extras", lambda *a, **k: {})
    monkeypatch.setattr(dl, "_eval_case", lambda cid: None)

    class Ev:
        @staticmethod
        def score(case, answer, citations):
            return {"expected": [], "failures": []}
    out = tmp_path / "case.json"
    dl.run_interactive_sim({"id": "c1", "out": str(out), "single_shot": True}, "Il caso.", "", Ev(),
                           {"calls": 0, "seconds": 0, "errors": 0}, lambda message: None)
    import json
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["status"] == "ok" and record["mode"] == "single_shot" and record["oracle"] is None
    assert record["single_shot"]["timings"]["target_s"] == 240 and record["single_shot"]["funnel"]["covered"] == [1, 3, 2]
    assert set(record["seconds_by_round"]) == {"round1", "answer"} and record["answer"].count("**") > 4


def test_tidy_never_cuts_the_codes_status_lines():
    from src.rag import text_tidy
    for line in ("- co. 1: da verificare: il caso non dice se il fondo ha un altro accesso alla via pubblica.",
                 "- co. 1: da provare.", "- co. 2, requisito «fondi divisi»: provato.", "- testo: da provare."):
        out, log = text_tidy.tidy(f"**Art. 1051 c.c.**\n{line}")
        assert out.endswith(line) and log == [], line
    out, log = text_tidy.tidy("co. 2 del codice, il fatto è provato e la norma si applica.")     # a cut citation still is
    assert out.startswith("Del codice") and log[0]["kind"] == "fragment repaired"


def test_the_procedural_sections_open_points_read_the_single_shot_way(monkeypatch):
    rendered = ("- Querela: termine di 3 mesi — DA VERIFICARE: il caso non indica le date da cui decorre (art. 124 c.p.).\n"
                "- Usucapione: ventennio — DA VERIFICARE, inizio del possesso da accertare: il caso non indica quando il "
                "possesso è iniziato.\n- Data del fatto: non nota. Da essa decorre la prescrizione del reato (art. 157 c.p.).")
    out = flow.open_points_wording(rendered)
    assert "— da verificare: il caso non dice le date da cui decorre" in out
    assert "— da verificare: il caso non dice quando il possesso è iniziato" in out
    assert "- Da verificare: il caso non dice la data del fatto. Da essa decorre" in out
    state, seen = _single(monkeypatch)
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": rendered)
    text = flow.single_shot("Il caso.")["answer"]
    procedural = dict(answer_ledger.sections_of(text))[flow.ANSWER_TITLES[3]]
    assert "il caso non dice le date da cui decorre" in procedural and "non nota" not in procedural
