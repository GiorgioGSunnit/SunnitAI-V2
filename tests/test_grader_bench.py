"""scripts/grader_bench.py offline: a synthetic run folder, a fake grader, no database."""

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("grader_bench", HERE.parent / "scripts" / "grader_bench.py")
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)
dl = gb._diagnose_long()
_ev_spec = importlib.util.spec_from_file_location("eval_narrative", HERE / "eval_narrative.py")
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)

# The 1206-1415 dump format: the label itself in brackets, the multi-signal mark of T1.
DUMP = ("stage: applicability\nmax_tokens: 840\nstatus: ok\n\n===== SYSTEM =====\nun voto di pertinenza\n\n"
        "===== USER =====\nFatti del caso:\nTizio spinge Caio, che muore.\n\nArticoli candidati (testo reale dalla "
        "banca dati), 3: un oggetto per ciascuno degli identificativi A1, A2, A3.\n\n"
        "[A1] 584 c.p.\nRubrica: Omicidio preterintenzionale\nTrovato da: più fonti\nTesto: Chiunque, con atti "
        "diretti a commettere...\n\n[A2] 133 c.p.\nRubrica: Gravità del reato\nTesto: Nell'esercizio del potere...\n\n"
        "[A3] 2043 c.c.\nRubrica: -\nTesto: Qualunque fatto doloso...\n\n===== REPLY =====\n[]\n")
RECORD = {"id": "sempronio", "kind": "eval", "expected": [{"code": "cp", "article": "584"}],
          "frame": {"tasks": ["Quale reato?"], "central_issue_candidates": [{"nome": "omicidio preterintenzionale"}],
                    "areas": ["penale"]},
          "questioni": [{"titolo": "Questione centrale: omicidio preterintenzionale", "voce": "questione_centrale",
                         "istituti": [{"nome": "omicidio preterintenzionale"}]}],
          "area": "penale",
          "pool": [{"label": "584 c.p.", "sources": [], "multi_signal": True}, {"label": "133 c.p.", "sources": []},
                   {"label": "2043 c.c.", "sources": []}],
          "fusion": {"signals": {"current": ["584 c.p."], "baseline": ["584 c.p.", "2043 c.c."], "distill": ["133 c.p."]}},
          "institutions": [{"phase": "v2", "nome": "omicidio preterintenzionale", "articles": [{"label": "584 c.p."}]}],
          "check_result": {"grades": {"584 c.p.": 1, "133 c.p.": 1, "2043 c.c.": 0}}}


class Grader:
    """3 for 584 when it sees the evidence, else 1; 133 always 1; 2043 always 0."""

    def __init__(self):
        self.humans = []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        human = messages[-1].content
        self.humans.append(human)
        out = []
        for key, label in re.findall(r"^\[(A\d+)\] (.+)$", human, re.MULTILINE):
            block = human.split(f"[{key}] {label}", 1)[1]
            grade = {"133 c.p.": 1, "2043 c.c.": 0}.get(label, 3 if "Sentenze simili" in block or "(più fonti)" in block
                                                         and "istituto di una questione centrale" in block else 1)
            out.append({"a": key, "g": grade, "q": "Q1", "r": "x"})
        return json.dumps(out)


def test_inputs_come_from_the_dumps_in_both_formats():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    old = DUMP.replace("[A1] 584 c.p.\n", "[584 c.p.]\n").replace("[A2] 133 c.p.\n", "[133 c.p.]\n") \
        .replace("[A3] 2043 c.c.\n", "[2043 c.c.]\n")
    with open(os.path.join(d, "09_applicability_retry.txt"), "w", encoding="utf-8") as fh:
        fh.write(old.replace("584 c.p.", "585 c.p."))
    inputs = gb.graded_inputs(d, dl._dump_parts)
    assert list(inputs) == ["584 c.p.", "133 c.p.", "2043 c.c.", "585 c.p."]
    assert inputs["584 c.p."] == ("Omicidio preterintenzionale", "Chiunque, con atti diretti a commettere...")
    assert inputs["2043 c.c."][0] == ""


def test_pool_tags_rebuild_the_evidence():
    tags = gb.pool_tags(RECORD, dg)
    assert tags["584 c.p."]["_signals"] == ["current", "baseline"] and tags["584 c.p."]["_central"]
    assert tags["584 c.p."]["_multi_signal"] and not tags["2043 c.c."].get("_central")


def test_the_bench_grades_every_version_and_reports(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", Grader())
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    inputs = gb.graded_inputs(d, dl._dump_parts)
    cases = {"sempronio": {"question": "Tizio spinge Caio, che cade e muore."}}
    results = gb.bench_case(RECORD, inputs, ["v0", "v3"], dg, dl._ekey_fn(ev), cases, vote=False)
    by = {r["label"]: r for r in results}
    assert by["584 c.p."]["expected"] and not by["133 c.p."]["expected"]
    assert by["584 c.p."]["grades"]["saved"]["grade"] == 1 and by["584 c.p."]["grades"]["v3"]["grade"] == 3
    assert by["584 c.p."]["grades"]["v3"]["issue"] == "Quale reato?"            # Q1 read as the first issue
    summary = gb.summarise(results)
    assert summary["v3"]["exp_pct"] == 100 and summary["saved"]["exp_pct"] == 0 and summary["v3"]["other_hi"] == 0
    text = gb.render("/runs/20261027-1415", results, ["v0", "v3"], 60, [])
    assert "| v3 | 1 | 1 (100%) | 2 | 0 (0%) | 0/0/0/1/0 | 1/1/0/0/0 |" in text
    assert "## saved: expected articles graded 1 or less (1)" in text and "| sempronio | 584 c.p. | 1 |" in text


class Voter:
    """One article per call expected; its votes for each label in turn (cycled), recording every call."""

    def __init__(self, votes):
        self.votes, self.calls = {k: list(v) for k, v in votes.items()}, []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        ids = re.findall(r"^\[(A\d+)\] (.+)$", messages[-1].content, re.MULTILINE)
        assert len(ids) == 1                       # one article per call
        key, label = ids[0]
        self.calls.append((label, temperature))
        grade = self.votes[label].pop(0)
        return "[]" if grade is None else json.dumps([{"a": key, "g": grade, "q": "q", "r": f"r{grade}"}])


def test_the_vote_keeps_the_second_highest_and_logs_every_vote(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    voter = Voter({"584 c.p.": [3, 1, 2], "133 c.p.": [2, 1, 1], "2043 c.c.": [0, 0, 0],
                   "52 c.p.": [None, 3, None], "55 c.p.": [None, None, None], "575 c.p.": [None, 2, 3]})
    monkeypatch.setattr(dg, "_call_chat", voter)
    texts = {label: "Rubrica: r\nTesto: t" for label in voter.votes}
    verdicts, ungraded = dg._voted_verdicts("Il caso.", texts)
    grade = {label: dg._grade(verdicts.get(dg._applicability_key(label))) for label in texts}
    # applicable (>= 2) exactly when two or more votes give >= 2
    assert grade == {"584 c.p.": 2, "133 c.p.": 1, "2043 c.c.": 0, "52 c.p.": 1, "55 c.p.": None, "575 c.p.": 2}
    assert ungraded == ["55 c.p."]
    assert verdicts[dg._applicability_key("584 c.p.")]["votes"] == [3, 1, 2]
    assert len(voter.calls) == 18 and {t for _, t in voter.calls} == {0}
    log = next(d for n, d in seen if n == "case_analysis_applicability")
    assert log["votes"]["133 c.p."] == [2, 1, 1] and log["batches"] == 18 and "584 c.p." in log["split"]


def test_the_vote_is_in_a_fixed_order_and_at_most_eight_at_once(monkeypatch):
    import threading
    import time as _time
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    live, peak, lock = [0], [0], threading.Lock()

    def chat(messages, **kwargs):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        _time.sleep(0.02)
        with lock:
            live[0] -= 1
        key = re.search(r"^\[(A\d+)\]", messages[-1].content, re.MULTILINE).group(1)
        return json.dumps([{"a": key, "g": 2}])
    monkeypatch.setattr(dg, "_call_chat", chat)
    texts = {f"{n} c.c.": "t" for n in (2043, 1218, 1176, 2059, 1223, 1226)}
    seen = []
    real = dg._graded_batch
    monkeypatch.setattr(dg, "_graded_batch", lambda case, t, *a, **k: (seen.append(list(t)[0]), real(case, t, *a, **k))[1])
    monkeypatch.setattr(dg, "_GRADER_VOTE_WORKERS", 8)
    dg._voted_verdicts("Il caso.", dict(reversed(list(texts.items()))))
    assert peak[0] <= 8
    assert list(dict.fromkeys(seen)) == sorted(texts, key=dg._label_key)       # the order, whatever the input's


def test_grader_vote_routes_the_graded_check(monkeypatch):
    monkeypatch.setenv("GRADER_VOTE", "true")
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    voter = Voter({"584 c.p.": [3, 3, 0]})
    monkeypatch.setattr(dg, "_call_chat", voter)
    verdicts, _ = dg._graded_verdicts("Il caso.", {"584 c.p.": "t"})
    assert dg._grade(verdicts[dg._applicability_key("584 c.p.")]) == 3 and len(voter.calls) == 3


def test_the_bench_s_v_vote_reports_flips_against_the_saved_grades(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    monkeypatch.setattr(dg, "_call_chat", Voter({"584 c.p.": [3, 2, 1], "133 c.p.": [1, 1, 1], "2043 c.c.": [0, 2, 0]}))
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    inputs = gb.graded_inputs(d, dl._dump_parts)
    cases = {"sempronio": {"question": "Tizio spinge Caio, che cade e muore."}}
    results = gb.bench_case(RECORD, inputs, ["v_vote"], dg, dl._ekey_fn(ev), cases, vote=False)
    by = {r["label"]: r["grades"]["v_vote"] for r in results}
    assert by["584 c.p."]["grade"] == 2 and by["584 c.p."]["votes"] == [3, 2, 1]
    flips = gb.flip_counts(results, "v_vote")
    # saved 584: 1 -> 2 (crossed, expected); 133: 1 -> 1; 2043: 0 -> 0; two articles' votes split
    assert (flips["n"], flips["changed"], flips["crossed"], flips["crossed_expected"], flips["split"]) == (3, 1, 1, 1, 2)
    text = gb.render("/runs/v46", results, ["v_vote"], 60, [])
    assert "| v_vote | 3 | 1 (33%) | 1 (33%; 1 / 0) | 1 | 2 | - |" in text


class Binary:
    """v_bin's model: 584 applies with a quote from its text; 133 "applies" with a quote the text does not hold; 2043 no."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        self.calls.append((messages[0].content, max_tokens))
        ids = dict((label, key) for key, label in re.findall(r"^\[(A\d+)\] (.+)$", messages[-1].content, re.MULTILINE))
        return json.dumps([
            {"a": ids["584 c.p."], "applica": "sì", "clausola": "con atti diretti a commettere"},
            {"a": ids["133 c.p."], "applica": "sì", "clausola": "il giudice deve tenere conto della gravità"},
            {"a": ids["2043 c.c."], "applica": "no", "clausola": ""}])


def test_v_bin_counts_a_quote_not_in_the_text_as_no_and_flips_on_the_line_of_2(monkeypatch):
    model = Binary()
    monkeypatch.setattr(dg, "_call_chat", model)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    inputs = gb.graded_inputs(d, dl._dump_parts)
    cases = {"sempronio": {"question": "Tizio spinge Caio, che cade e muore."}}
    results = gb.bench_case(RECORD, inputs, ["v_bin"], dg, dl._ekey_fn(ev), cases, vote=False)
    by = {r["label"]: r["grades"]["v_bin"] for r in results}
    assert (by["584 c.p."]["grade"], by["584 c.p."]["applies"], by["584 c.p."]["quote_in_text"]) == (2, True, True)
    assert (by["133 c.p."]["grade"], by["133 c.p."]["said"], by["133 c.p."]["quote_in_text"]) == (0, "sì", False)
    assert by["2043 c.c."]["grade"] == 0 and not by["2043 c.c."]["applies"]
    assert len(model.calls) == 1 and model.calls[0][0] == gb.BINARY_SYSTEM and model.calls[0][1] == 200 + 160 * 3
    flips = gb.flip_counts(results, "v_bin")
    # saved 584: 1 -> yes (crossed, expected); 133: 1 -> no (unchanged); 2043: 0 -> no
    assert (flips["changed"], flips["crossed"], flips["to_yes"], flips["to_no"], flips["crossed_expected"],
            flips["quote_missing"]) == (1, 1, 1, 0, 1, 1)
    assert "| v_bin | 3 | 1 (33%) | 1 (33%; 1 / 0) | 1 | - | 1 |" in gb.render("/runs/v46", results, ["v_bin"], 60, [])


def test_the_quote_check_is_a_substring_check():
    text = "Chiunque cagiona la morte di un uomo è punito con la reclusione non inferiore ad anni ventuno."
    assert gb.quote_in_text("«cagiona la  morte di un uomo»", text)
    assert gb.quote_in_text("Chiunque cagiona la morte di un uomo è punito.", text)
    assert not gb.quote_in_text("cagiona la morte di una persona", text) and not gb.quote_in_text("", text)
