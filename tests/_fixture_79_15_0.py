"""v79.15.0 build B -- the end-to-end harness the two synthetic fixtures share (tests/test_v79_15_0_fixture_civil.py,
tests/test_v79_15_0_fixture_penal.py): the real entry, intake.request_intake -> interactive_flow.single_shot ->
write_answer -> _write_answer, with ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT and LAWYER_METHOD on, as
tests/test_v79_14_3_fixture.py runs it. Not a test module (no test_ prefix).

Faked at their boundary only: the frame (case_frame._default_call), the fit (applicability._default_call), the analysis
step (analysis_step._default_call -- every call recorded, the prompt parsed back), the thesis call (ai_chat.call_chat_n),
the element link (issue_records._link_call); the graph, round 1, the plan, the section writer, the documents' names
and §8's renderer. Every other model path refused at its source (ai_chat.chat_model, ai_chat.ChatOpenAI, every
module's _call_chat / _call_chat_with_tools / embedding_model / _embed_query_with_prefix -- defensive_generation's
included) and, below them, every outbound socket. Synthetic cases only: invented names and facts, statutory article
texts.
"""

import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_14_2_fixture as F  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import verbose_logger  # noqa: E402

FLAGS = ("ANALYSIS_STEP", "CASE_FRAME", "APPLICABILITY_FIT", "LAWYER_METHOD")
EVENTS = ("analysis_step", "applicability_fit", "legacy_plan", "lawyer_method_prose_review")     # v79.15.1
NEW_COUNTERS = tuple(A.COUNTERS) + tuple(A.RENDER_COUNTERS) + ("fit_unverified_ids", "fit_no_unverified_ids",
                                                               "fit_protected_kept")


def _no_network(*a, **k):
    raise RuntimeError("no network in the fixture")


def parse_prompt(human: str) -> Dict[str, Any]:
    """The analysis step's human message read back: {"facts": {S-id: text}, "norms": {article: {"elements": [...],
    "eids": [the elements' E-ids, v79.15.1], "periods": [A-ids], "remedy": bool}}}."""
    facts = {i: t for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", human, re.MULTILINE)}
    norms: Dict[str, Dict[str, Any]] = {}
    current, mode = None, ""
    for line in human.splitlines():
        m = re.match(r"^ARTICOLO: (.+?)( \(rimedio o parte generale: mai una linea\))?$", line)
        if m:
            current = m.group(1)
            norms[current] = {"elements": [], "eids": [], "periods": [], "remedy": bool(m.group(2))}
            mode = ""
            continue
        if current is None:
            continue
        if line.startswith("Elementi dello schema:"):
            mode = "el"
        elif line.startswith("Testo (periodi numerati):"):
            mode = "per"
        elif mode == "el" and line.startswith("- ") and line != "- (nessuno)":
            e = re.match(r"^- \[(A[^\]]+:E\d+)\] (.*)$", line)          # v79.15.1: «- [A<art>:E<n>] <testo>»
            norms[current]["eids"].append(e.group(1) if e else "")
            norms[current]["elements"].append(e.group(2) if e else line[2:])
        elif mode == "per":
            p = re.match(r"^\[(A[^\]]+)\]", line)
            if p:
                norms[current]["periods"].append(p.group(1))
            elif not line.strip():
                mode = ""
    return {"facts": facts, "norms": norms}


class AnalysisModel:
    """The analysis step's model faked: reply(parsed prompt, attempt) -> the reply object; every call recorded."""

    def __init__(self, reply: Callable[[Dict[str, Any], int], Dict[str, Any]]):
        self.reply, self.calls = reply, []

    def __call__(self, messages, kwargs):
        parsed = parse_prompt(messages[1].content)
        self.calls.append({"parsed": parsed, "n_messages": len(messages), "last": messages[-1].content,
                           "kwargs": kwargs})
        return json.dumps(self.reply(parsed, len(self.calls)), ensure_ascii=False)


def id_form(reply: Callable[[Dict[str, Any], int], Dict[str, Any]]) -> Callable[[Dict[str, Any], int], Dict[str, Any]]:
    """v79.15.2: a faked analysis reply with every «articolo» in the id form the prompt sends, as the server's model
    returned it -- a line's as the bare token («1385cc», «646cp», «62biscp»), an element verdict's with its «A»
    («A1385cc»); the E-ids as given."""
    def convert(parsed: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        got = reply(parsed, attempt)
        for kind in ("linea_principale", "linea_subordinata"):
            x = got.get(kind)
            if isinstance(x, dict) and x.get("articolo"):
                x["articolo"] = ap.article_token(x["articolo"])
        for it in got.get("elementi") or []:
            if isinstance(it, dict) and it.get("articolo"):
                it["articolo"] = "A" + ap.article_token(it["articolo"])
        return got
    convert.__name__ = f"id_form_{getattr(reply, '__name__', 'reply')}"
    return convert


def eid(parsed: Dict[str, Any], article: str, text: str) -> str:
    """v79.15.1: the E-id the prompt gave the element of that text."""
    norm = parsed["norms"][article]
    return norm["eids"][norm["elements"].index(text)]


def rows_of(articles):
    return [{"_article": [n, c], "_grade": 3, "s": {"name": f"{n}.0.0", "plain_text": text, "titolo_articolo": rubrica},
             "d": {"id": doc, "name": name}} for n, c, text, rubrica, doc, name in articles]


def run(monkeypatch, tmp_path, case) -> Dict[str, Any]:
    """The case through the real entry. case: TEXT, ARTICLES, FRAME_REPLY, fit_reply, link, analysis_reply, title,
    anchor, rubrica, documents, client, writer_line (label -> the writer's paragraph), writer_question (optional: a
    question section's title -> its text), OLDER_FRAME (optional, v79.15.1: the older frame's fields in round 1's
    analysis, e.g. central_issue_candidates), objections (optional, v79.15.1: the thesis call's «obiezioni»), ENV
    (optional, v79.15.1: more environment flags) and REVIEWED_SCHEMAS (optional, v79.15.1: reviewed element-schema
    lines, written to the run's ARTICLE_SCHEMAS_REVIEWED_FILE)."""
    for flag in FLAGS:
        monkeypatch.setenv(flag, "true")
    for key, value in (getattr(case, "ENV", None) or {}).items():      # v79.15.1: a case's own flags
        monkeypatch.setenv(key, value)
    reviewed = getattr(case, "REVIEWED_SCHEMAS", None)                   # v79.15.1: the lawyer's schema overlay
    if reviewed:
        with open(tmp_path / "reviewed.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in reviewed)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    from src.rag import ai_chat, defensive_generation, issue_records, main, plan_then_write as ptw, render_computed
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(ai_chat, "chat_model", F._NoModel())
    monkeypatch.setattr(ai_chat, "ChatOpenAI", F._refused)
    for mod in [m for name, m in list(sys.modules.items()) if name.startswith("src.rag.") and m is not None]:
        for attr in ("_call_chat", "_call_chat_with_tools", "embedding_model", "_embed_query_with_prefix"):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, F._refused)
    assert defensive_generation._call_chat is F._refused and ai_chat._call_chat is F._refused
    labels = [f"{n} {c}" for n, c, *_ in case.ARTICLES]
    model = AnalysisModel(case.analysis_reply)
    monkeypatch.setattr(cf, "_default_call", lambda m, k: json.dumps(case.FRAME_REPLY, ensure_ascii=False))
    monkeypatch.setattr(ap, "_default_call", case.fit_reply)
    monkeypatch.setattr(A, "_default_call", model)
    objections = getattr(case, "objections", None)
    if objections is None:
        monkeypatch.setattr(ai_chat, "call_chat_n", F._thesis_replies)
    else:                       # v79.15.1: the thesis call's objections as the case gives them
        def thesis(messages, n, **kw):
            ids = re.findall(r"^- \[(T\d+)\]", messages[-1].content, re.MULTILINE)
            valued = {i: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"} for i in ids}
            reply = {"valutazioni": valued, "obiezioni": list(objections)}
            return [json.dumps(reply, ensure_ascii=False)] * max(1, int(n or 1))
        monkeypatch.setattr(ai_chat, "call_chat_n", thesis)
    monkeypatch.setattr(ptw, "negation_review", lambda items, known, call=None: [])
    monkeypatch.setattr(issue_records, "_link_call", case.link)
    monkeypatch.setattr(main, "driver", F._NoGraph())

    def round1(text, request="", lang="it", frame=None):
        topic = {"title": case.TITLE, "anchor": case.ANCHOR, "how": "exact", "rubrica": case.RUBRICA,
                 "articles": list(labels), "matched": list(labels), "planned": len(labels),
                 "grades": {x: 3 for x in labels}, "graded": len(labels), "facts": [], "source": "frame", "merged": []}
        # v79.15.1: an optional OLDER_FRAME -- the older frame's fields (central_issue_candidates) in round 1's analysis
        analysis = {"parte_assistita": "", "frame": {"areas": [], **dict(getattr(case, "OLDER_FRAME", None) or {})},
                    flow.CASE_KEY: text}
        state = {"schema": flow.SCHEMA, "version": 0, "state": "", "created": "2026-11-20T00:00:00+00:00",
                 "lang": lang, "text": text, "request": request, "additions": [], "analysis": flow._jsonable(analysis),
                 "topics": [topic], "others": [], "units": {},
                 "rows": [flow._compact_row(r) for r in rows_of(case.ARTICLES)], "ranked": [], "retrieval_top": [],
                 "defence_gate": {}, "family_gate": {}, "code_scope": {}, "tool_rows": [], "pool": [], "rendered": "",
                 "answered": {}, "chosen": [], "angle": "", "changes": [], "answers": {}, "timings": {},
                 "grade_funnel": {}}
        state["questions"] = []
        flow.build_records(state, rows_of(case.ARTICLES))
        writer.state = state
        return state

    def writer(plan, jobs, lang, named, documents, rows, case_text, **kwargs):
        out = []
        for title, *_ in jobs:
            have = [x for x in labels if x in (writer.state.get("records") or {})]
            special = getattr(case, "writer_question", None)
            text = special(title) if special and str(title).endswith("?") else None
            out.append((title, text if text is not None else "\n\n".join(case.writer_line(x) for x in have)))
        return out
    writer.state = {}
    monkeypatch.setattr(flow, "analyse_case", round1)
    monkeypatch.setattr(flow, "_documents", lambda: list(case.DOCUMENTS))
    monkeypatch.setattr(ptw, "make_plan", lambda *a, **k: {
        "parte_assistita": case.CLIENT, "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
        "subordinate": [], "punti_deboli": [], "questioni": []})
    monkeypatch.setattr(ptw, "write_sections", writer)
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    events: List[Any] = []
    failed: List[str] = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append(
        (event, data, duration_ms)))
    real_isolated = flow._isolated

    def isolated(name, fn, default=None):
        def step():
            try:
                return fn()
            except Exception as e:
                failed.append(f"{name}: {type(e).__name__}: {e}")
                raise
        return real_isolated(name, step, default)
    monkeypatch.setattr(flow, "_isolated", isolated)
    got = intake.request_intake(case.TEXT)
    out = flow.single_shot(case.TEXT, intake=got, case_id=case.CASE_ID)
    by: Dict[str, List[Any]] = {}
    for e, d, ms in events:
        by.setdefault(e, []).append((d, ms))
    return {"out": out, "state": out["state"], "answer": out["answer"], "events": events, "by": by, "failed": failed,
            "model": model}


def section(answer: str, key: str) -> str:
    return dict(M.answer_sections(answer)).get(M.title(key)) or ""


def esito_column(answer: str) -> List[str]:
    """§4's element tables (Elemento | Fatti citati | Esito) as «Art. X | elemento | esito» lines."""
    out, art = [], ""
    for line in section(answer, "norme").splitlines():
        m = re.match(r"^\*\*Art\. (.+?)\*\*", line)
        if m:
            art = m.group(1)
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) == 3 and cells[0] not in ("Elemento", "---"):
            out.append(f"{art} | {cells[0]} | {cells[2]}")
    return out


def write_report(path: str, title: str, run_: Dict[str, Any], mode: str = "w") -> None:
    """The fixture's log lines (production format), the new counters, §1 as delivered and §4's Esito column."""
    c = run_["state"].get("block_counters") or {}
    lines = [e for e in run_["events"] if e[0] in EVENTS]
    with open(path, mode, encoding="utf-8") as f:
        f.write(f"# {title}\n\n## log lines (production format: verbose_logger, each value cut at 100 characters)\n\n")
        f.writelines(F._line(e, d, ms) + "\n" for e, d, ms in lines)
        f.write("\n## the same lines' payloads, whole\n\n")
        for e, d, _ in lines:
            f.write(f"{e}: {json.dumps(d, ensure_ascii=False, default=str)}\n")
        f.write("\n## new counters (block_counters)\n\n")
        f.write(json.dumps({k: c.get(k) for k in NEW_COUNTERS}, ensure_ascii=False, indent=1) + "\n")
        f.write("\n## analysis calls (the faked model)\n\n")
        for i, call in enumerate(run_["model"].calls, 1):
            f.write(f"call {i}: {call['n_messages']} messages; last message: {call['last'][:600]!r}\n")
        f.write("\n## §1 as delivered\n\n" + section(run_["answer"], "sintesi") + "\n")
        f.write("\n## §4 Esito column (Art. | Elemento | Esito)\n\n" + "\n".join(esito_column(run_["answer"])) + "\n")
        f.write("\n## §7 objections as delivered\n\n" + section(run_["answer"], "obiezioni") + "\n")      # v79.15.1
        qa = run_["answer"].split("**1. ")[0].strip()
        f.write("\n## the Q&A block as delivered\n\n" + (qa if M.ANSWERS_TITLE in qa else "(none)") + "\n")
        f.write("\n## steps the pipeline isolated as failed\n\n" + "".join(f"{x[:200]}\n" for x in run_["failed"]))
        f.write("\n## the answer\n\n" + run_["answer"] + "\n\n")
