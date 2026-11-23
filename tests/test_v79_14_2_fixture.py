"""v79.14.2 -- ONE synthetic penal fixture end to end (invented names and facts; statutory text of 640 and 646 c.p.,
1439 and 2043 c.c.) through the real pipeline entry, interactive_flow.single_shot -> write_answer -> _write_answer,
CASE_FRAME, APPLICABILITY_FIT and LAWYER_METHOD on. Faked at their boundary only: the models (case_frame._default_call,
applicability._default_call, ai_chat.call_chat_n for the thesis call, every other ai_chat call refused), the graph (the
driver refused), round 1's retrieval (interactive_flow.analyse_case: a synthetic round-1 state -- topics, rows and the
records build_records makes of them), the plan and the section writer (plan_then_write.make_plan / write_sections),
the documents' names, §4's renderer and the element link.

  intake.request_intake (the frame read first: «case_frame_read») -> single_shot: take_intake, the stated client,
  covered_topics -> write_answer: posture.of -> decompose_records with the fit hook (applicability_fit: the judgment,
  «applicability_fit»; the verified «no» on 1439 c.c. removed: «applicability_fit_discard» «removed: fit no
  (verified)») -> link_records, mark_sides, surviving_issues, anchor_blocks, charge_rites -> method_inputs (prepare --
  the frame's binding: «lawyer_method_frame_binding» -- and the thesis call: «lawyer_method_plan») -> the writers, the
  record check -> case_frame_audit.

The client is indagato for truffa; the candidates are 640 c.p. (charged), 646 c.p., 1439 c.c. (a civil reading of the
deceit) and 2043 c.c. (the persona offesa's damages, named in the text: with the frame off the method offers its
elements' contests). 1439 c.c. never reaches the plan nor «Norme» (the verified «no»); 2043 c.c. stays in §4 as context,
never the charge, «Linea principale» nor «Esito probabile» (the binding). With FIXTURE_79_14_2_OUT set, the three log lines (production format,
verbose_logger's) and the counters are written there.
"""

import json
import os
import re
import sys
from pathlib import Path


for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import verbose_logger  # noqa: E402

TEXT = ("Assisto Bruno Gelso, indagato per truffa. Bruno Gelso ha venduto a Carla Tasso un orologio dichiarandolo "
        "d'oro. L'orologio era placcato e Carla Tasso ha pagato 4.000 euro con un bonifico. Il pubblico ministero "
        "procede per truffa ai sensi dell'art. 640 c.p. e le indagini preliminari sono in corso. Carla Tasso chiede a "
        "Bruno Gelso il risarcimento del danno ai sensi dell'art. 2043 c.c. per 4.000 euro. Quali rischi corre Bruno Gelso?")

ART_640 = ("Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un ingiusto profitto "
           "con altrui danno, è punito con la reclusione da sei mesi a tre anni e con la multa da euro 51 a euro 1.032.")
ART_646 = ("Chiunque, per procurare a sé o ad altri un ingiusto profitto, si appropria il denaro o la cosa mobile altrui "
           "di cui abbia, a qualsiasi titolo, il possesso, è punito, a querela della persona offesa, con la reclusione "
           "da due a cinque anni e con la multa da euro 1.000 a euro 3.000.")
ART_1439 = ("Il dolo è causa di annullamento del contratto quando i raggiri usati da uno dei contraenti sono stati tali "
            "che, senza di essi, l'altra parte non avrebbe contrattato. Quando i raggiri sono stati usati da un terzo, "
            "il contratto è annullabile se essi erano noti al contraente che ne ha tratto vantaggio.")
ART_2043 = ("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il "
            "fatto a risarcire il danno.")
ARTICLES = (("640", "c.p.", ART_640, "Truffa", "LEGAL_DOC::cp", "Codice Penale 2026"),
            ("646", "c.p.", ART_646, "Appropriazione indebita", "LEGAL_DOC::cp", "Codice Penale 2026"),
            ("1439", "c.c.", ART_1439, "Dolo", "LEGAL_DOC::cc", "Codice Civile 2026"),
            ("2043", "c.c.", ART_2043, "Risarcimento per fatto illecito", "LEGAL_DOC::cc", "Codice Civile 2026"))
LABELS = [f"{n} {c}" for n, c, *_ in ARTICLES]

FRAME_REPLY = {
    "parti": [{"nome": "Bruno Gelso", "ruolo_nel_fatto": "venditore", "lato": "cliente"},
              {"nome": "Carla Tasso", "ruolo_nel_fatto": "acquirente", "lato": "controparte"}],
    "cliente": {"nome": "Bruno Gelso", "perche": "«Assisto Bruno Gelso, indagato per truffa»"},
    "ruolo": {"valore": "indagato", "perche": "«Assisto Bruno Gelso, indagato per truffa»"},
    "rito": {"valore": "penale", "perche": "«Il pubblico ministero procede per truffa»"},
    "stadio": {"valore": "indagini", "perche": "«le indagini preliminari sono in corso»"},
    "addebito_o_domanda": {"articolo": "640 c.p.", "descrizione": "truffa",
                           "frase": "«procede per truffa ai sensi dell'art. 640 c.p.»"},
    "istituto": None,
    "quesiti": ["Quali rischi corre Bruno Gelso?"]}


def _fit_item(article, fit, perche, ambito="fattispecie", cf=None, cn=None):
    """v79.15.0 (item 3): the judgment cites ids -- cf the facts («cita_fatti»), cn the article's numbered periods
    («cita_norma»; default: a «si» cites the article's first period, applicability.article_ids)."""
    if cn is None and fit == "si":
        cn = [f"A{ap.article_token(article)}.1"]
    return {"articolo": article, "fit": fit, "perche": perche, "condizione": "", "ambito": ambito,
            "cita_fatti": list(cf or []), "cita_norma": list(cn or [])}


def _fit_reply(messages, kwargs):
    """The fit model faked: each norm the call carries judged -- 1439 c.c. «no» quoting the fact it rests on."""
    human = messages[-1].content
    sid = {t: i for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", human, re.MULTILINE)}
    pm = next(i for t, i in sid.items() if t.startswith("Il pubblico ministero"))
    sale = next(i for t, i in sid.items() if t.startswith("Bruno Gelso ha venduto"))
    harm = next(i for t, i in sid.items() if t.startswith("Carla Tasso chiede"))
    items = {"640 c.p.": _fit_item("640 c.p.", "si", f"Bruno Gelso ha venduto a Carla Tasso un orologio ({sale}): «con "
                                                     "artifizi o raggiri».", cf=[sale]),       # v79.16.1 (item 2)
             "646 c.p.": _fit_item("646 c.p.", "si", f"Bruno Gelso ha venduto a Carla Tasso un orologio ({sale}): "
                                                     "«si appropria il denaro o la cosa mobile altrui».", cf=[sale]),
             "1439 c.c.": _fit_item("1439 c.c.", "no", f"«Il pubblico ministero procede per truffa» ({pm}).",
                                    ambito="rimedio", cf=[pm]),        # v79.15.0 (item 3): the fact by its id
             "2043 c.c.": _fit_item("2043 c.c.", "si", f"Carla Tasso chiede a Bruno Gelso il risarcimento del danno "
                                                       f"({harm}): "
                                                       "«obbliga colui che ha commesso il fatto a risarcire il danno».",
                                    ambito="rimedio", cf=[harm])}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def _thesis_replies(messages, n, **kw):
    """The thesis model faked: every candidate valued (n samples, the same reply)."""
    ids = re.findall(r"^- \[(T\d+)\]", messages[-1].content, re.MULTILINE)
    reply = {"valutazioni": {i: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"} for i in ids},
             "obiezioni": [{"obiezione": "La persona offesa insisterà sul raggiro.", "risposta": "Va provato."}]}
    return [json.dumps(reply, ensure_ascii=False)] * max(1, int(n or 1))


def _refused(*a, **k):
    raise RuntimeError("no model or graph in the fixture")


class _NoModel:
    """The chat model every _call_chat goes through (ai_chat.chat_model, whoever imported _call_chat): refused."""
    def bind(self, *a, **k):
        return self

    def invoke(self, *a, **k):
        raise RuntimeError("no model in the fixture")


class _NoGraph:
    def session(self, *a, **k):
        raise RuntimeError("no graph in the fixture")

    def __getattr__(self, name):
        raise RuntimeError("no graph in the fixture")


def _rows():
    return [{"_article": [n, c], "_grade": 3, "s": {"name": f"{n}.0.0", "plain_text": text, "titolo_articolo": rubrica},
             "d": {"id": doc, "name": name}} for n, c, text, rubrica, doc, name in ARTICLES]


def _round1(text, request="", lang="it", frame=None):
    """Round 1 faked (interactive_flow.analyse_case's state shape): one topic on the four synthetic articles, the rows,
    the records build_records makes of them."""
    topic = {"title": "Truffa nella vendita dell'orologio", "anchor": "640 c.p.", "how": "exact", "rubrica": "Truffa.",
             "articles": list(LABELS), "matched": list(LABELS), "planned": len(LABELS),
             "grades": {x: 3 for x in LABELS}, "graded": len(LABELS), "facts": [], "source": "frame", "merged": []}
    analysis = {"parte_assistita": "", "frame": {"areas": []}, flow.CASE_KEY: text}
    state = {"schema": flow.SCHEMA, "version": 0, "state": "", "created": "2026-11-20T00:00:00+00:00", "lang": lang,
             "text": text, "request": request, "additions": [], "analysis": flow._jsonable(analysis),
             "topics": [topic], "others": [], "units": {}, "rows": [flow._compact_row(r) for r in _rows()],
             "ranked": [], "retrieval_top": [], "defence_gate": {}, "family_gate": {}, "code_scope": {},
             "tool_rows": [], "pool": [], "rendered": "", "answered": {}, "chosen": [], "angle": "", "changes": [],
             "answers": {}, "timings": {}, "grade_funnel": {}}
    state["questions"] = []
    flow.build_records(state, _rows())
    return state


def _writer(plan, jobs, lang, named, documents, rows, case, **kwargs):
    """The section writer faked: each job's section names the articles the record still holds (never one removed)."""
    out = []
    for title, *_ in jobs:
        labels = [x for x in LABELS if x in (_writer.state.get("records") or {})]
        out.append((title, "\n\n".join(f"[{i}] L'art. {x} riguarda i fatti narrati." for i, x in enumerate(labels, 1))))
    return out


def _line(event, data, duration_ms=None):
    """A log line as production prints it (verbose_logger's format: each value cut at 100 characters)."""
    parts = [f"[VERBOSE:{event}]"]
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.1f}ms")
    for k, v in (data or {}).items():
        parts.append(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" if k in verbose_logger.FULL_KEYS
                     else f"{k}={repr(v)[:100]}")
    return " | ".join(parts)


def run_fixture(monkeypatch, tmp_path):
    for flag in ("CASE_FRAME", "APPLICABILITY_FIT", "LAWYER_METHOD"):
        monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    from src.rag import ai_chat, issue_records, main, plan_then_write as ptw, render_computed
    # the models
    monkeypatch.setattr(cf, "_default_call", lambda m, k: json.dumps(FRAME_REPLY, ensure_ascii=False))
    monkeypatch.setattr(ap, "_default_call", _fit_reply)
    monkeypatch.setattr(ai_chat, "call_chat_n", _thesis_replies)
    monkeypatch.setattr(ai_chat, "chat_model", _NoModel())             # every other call refused, never sent
    monkeypatch.setattr(ai_chat, "ChatOpenAI", _refused)
    monkeypatch.setattr(ptw, "negation_review", lambda items, known, call=None: [])
    monkeypatch.setattr(issue_records, "_link_call", lambda topic, reqs, sents: {})
    # the graph, round 1, the plan, the writer, the documents, §4
    monkeypatch.setattr(main, "driver", _NoGraph())
    monkeypatch.setattr(flow, "analyse_case", _round1)
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
                                                     {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])
    monkeypatch.setattr(ptw, "make_plan", lambda *a, **k: {
        "parte_assistita": "Bruno Gelso", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
        "subordinate": [], "punti_deboli": [], "questioni": []})
    monkeypatch.setattr(ptw, "write_sections", _writer)
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    events, failed = [], []
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append(
        (event, data, duration_ms)))
    real_isolated = flow._isolated

    def isolated(name, fn, default=None):       # the pipeline's own isolation, each failed step kept for the report
        def run():
            try:
                return fn()
            except Exception as e:
                failed.append(f"{name}: {type(e).__name__}: {e}")
                raise
        return real_isolated(name, run, default)
    monkeypatch.setattr(flow, "_isolated", isolated)
    got = intake.request_intake(TEXT)           # the intake: the frame read before the rules
    seen = {}
    real_round1 = flow.analyse_case

    def round1(*a, **k):
        state = real_round1(*a, **k)
        seen["state"] = _writer.state = state
        return state
    monkeypatch.setattr(flow, "analyse_case", round1)
    out = flow.single_shot(TEXT, intake=got, case_id="v79142_fixture")
    return {"out": out, "state": out["state"], "answer": out["answer"], "events": events, "failed": failed,
            "intake": got}


def test_the_penal_fixture_end_to_end(monkeypatch, tmp_path):
    run = run_fixture(monkeypatch, tmp_path)
    state, answer, events = run["state"], run["answer"], run["events"]
    by = {}
    for e, d, ms in events:
        by.setdefault(e, []).append((d, ms))
    # the method wrote the answer (never the step-aside prose), every step ran
    assert "**1. Sintesi della soluzione**" in answer and not state.get("method_refused")
    assert [x for x in run["failed"] if x.split(":")[0] in ("posture", "decompose_records", "applicability_fit",
                                                             "record_sides", "lawyer_method_plan")] == []
    # the frame: every field but the absent institution verified, the client indagato in a penal rito
    frame = by["case_frame_read"][0][0]
    assert frame["status"] == "ok" and frame["counters"]["frame_fields_verified"] == 7
    assert state["posture"]["role"] == "imputato" and (state.get("client") or {}).get("name") == "Bruno Gelso"
    # A: the verified «no» on 1439 c.c. -- removed, logged, counted; out of «Norme» and of the whole answer
    fit = by["applicability_fit"][0][0]["counters"]
    assert fit["fit_no_verified"] == 1 and fit["fit_no_unverified_ids"] == 0 and fit["fit_unverified"] == 0
    removal = [d for d, _ in by["applicability_fit_discard"]]
    assert removal == [{"article": "1439 c.c.", "reason": "removed: fit no (verified)", "fit": "no",
                        "perche": removal[0]["perche"]}]
    assert "1439 c.c." not in state["records"] and state["block_counters"]["fit_removed"] == 1
    assert "1439" not in answer
    # B: the binding -- 2043 c.c. kept as context, never the charge, «Linea principale» nor «Esito probabile»
    binding = by["lawyer_method_frame_binding"][0][0]
    assert binding["mode"] == "penale" and binding["charge_set"] == ["640 c.p.", "2043 c.c."]
    assert binding["kept"] == ["640 c.p."] and binding["excluded"] == ["2043 c.c."] and not binding["empty"]
    # v79.14.3 (item 4): excluded means excluded -- marked in the records, never printed (no §4 block, no table)
    assert state["records"]["2043 c.c."]["side"] == "threat" and "**Art. 2043 c.c." not in answer
    plan_log = by["lawyer_method_plan"][0][0]
    offered = [x for x in plan_log["candidate_articles"] if x]
    assert offered and all(x.endswith("c.p.") for x in offered), offered
    assert "1439 c.c." not in plan_log["candidate_articles"] and "2043 c.c." not in plan_log["candidate_articles"]
    esito = next(x for x in answer.splitlines() if "**Esito probabile**" in x)
    assert "1439" not in esito and "2043" not in esito
    principale = [x for x in answer.splitlines() if "Linea principale" in x]
    assert principale and not any("2043" in x or "1439" in x for x in principale)
    c = state["block_counters"]
    assert c["frame_binding_applied"] == 1 and c["frame_binding_excluded"] == ["2043 c.c."]
    assert c["frame_binding_empty"] == 0
    target = os.environ.get("FIXTURE_79_14_2_OUT")
    if target:
        three = [(e, d, ms) for e, d, ms in events if e in ("case_frame_read", "applicability_fit",
                                                             "lawyer_method_plan")]
        extra = [(e, d, ms) for e, d, ms in events if e in ("applicability_fit_discard",
                                                             "lawyer_method_frame_binding")]
        keys = [k for k in c if k.startswith(("fit_", "frame_", "norms_removed", "corpus_line"))]
        with open(target, "w", encoding="utf-8") as f:
            f.write("# v79.14.2 synthetic penal fixture (tests/test_v79_14_2_fixture.py) through "
                    "interactive_flow.single_shot -> write_answer -- the three log lines, production format\n\n")
            f.writelines(_line(e, d, ms) + "\n" for e, d, ms in three)
            f.write("\n# the removal and the binding lines\n\n")
            f.writelines(_line(e, d, ms) + "\n" for e, d, ms in extra)
            f.write("\n# the three lines' payloads, whole\n\n")
            for e, d, ms in three + extra:
                f.write(f"{e}: {json.dumps(d, ensure_ascii=False, default=str)}\n")
            f.write("\n# block_counters (fit_*, frame_*, norms_removed_by_fit, corpus_line*)\n\n")
            f.write(json.dumps({k: c[k] for k in sorted(keys)}, ensure_ascii=False, indent=1) + "\n")
            f.write("\n# steps the pipeline isolated as failed (the faked boundaries)\n\n")
            f.writelines(f"{x[:200]}\n" for x in run["failed"])
            f.write(f"\n# Linea principale / Esito probabile\n\n" + "\n".join(principale) + f"\n{esito}\n")
            f.write("\n# the answer\n\n" + answer + "\n")
