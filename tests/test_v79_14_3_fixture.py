"""v79.14.3 -- ONE synthetic penal fixture end to end (invented names and facts; statutory text of 646 c.p., 1439 and
2043 c.c.) through the real pipeline entry, intake.request_intake -> interactive_flow.single_shot -> write_answer ->
_write_answer, CASE_FRAME, APPLICABILITY_FIT and LAWYER_METHOD on. Faked at their boundary only, as
tests/test_v79_14_2_fixture.py does: the models (case_frame._default_call, applicability._default_call,
ai_chat.call_chat_n for the thesis call, the element link issue_records._link_call), every other model path refused at
its source (ai_chat.chat_model, ai_chat.ChatOpenAI, ai_chat._call_chat, defensive_generation._call_chat and every module
that imported _call_chat or an embedding model) and, below them, any outbound socket; the graph, round 1, the plan, the
section writer, the documents' names and §4's renderer.

The client, Bruno Gelso, is indagato for appropriazione indebita (646 c.p.) and denies; the company he worked for alleges
the appropriation («La società sostiene che …»), and claims damages under 2043 c.c. (a civil article the penal binding
excludes); the fit judges 1439 c.c. «no» with its «fatto_citato» (verified: removed). Expected: 646's elements resting
on the company's allegation «contestato (allegazione della società)», the defence's theses on them («L'accusa deve
provare «…»»), never «Ammettere la condotta», corpus_line_code 0, 2043 c.c. and 1439 c.c. absent outside «Altri
profili». With FIXTURE_79_14_3_OUT set, the three log lines (production format) and the counters are written there.
"""

import json
import os
import re
import socket
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_14_2_fixture as F  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import verbose_logger  # noqa: E402

ALLEGATION = "La società sostiene che Bruno Gelso si è appropriato di 9.000 euro degli incassi"
PM = "Il pubblico ministero procede per appropriazione indebita"
TEXT = ("Assisto Bruno Gelso, indagato per appropriazione indebita. Bruno Gelso lavorava come cassiere presso la "
        f"società Orafi Riuniti e custodiva la cassa del negozio. {ALLEGATION}. Bruno Gelso nega di aver preso il "
        f"denaro. {PM} ai sensi dell'art. 646 c.p. e le indagini preliminari sono in corso. La società chiede a Bruno "
        "Gelso il risarcimento del danno ai sensi dell'art. 2043 c.c. per 9.000 euro. Quali rischi corre Bruno Gelso?")

ARTICLES = (("646", "c.p.", F.ART_646, "Appropriazione indebita", "LEGAL_DOC::cp", "Codice Penale 2026"),
            ("1439", "c.c.", F.ART_1439, "Dolo", "LEGAL_DOC::cc", "Codice Civile 2026"),
            ("2043", "c.c.", F.ART_2043, "Risarcimento per fatto illecito", "LEGAL_DOC::cc", "Codice Civile 2026"))
LABELS = [f"{n} {c}" for n, c, *_ in ARTICLES]

FRAME_REPLY = {
    "parti": [{"nome": "Bruno Gelso", "ruolo_nel_fatto": "cassiere", "lato": "cliente"},
              {"nome": "Orafi Riuniti", "ruolo_nel_fatto": "società", "lato": "controparte"}],
    "cliente": {"nome": "Bruno Gelso", "perche": "«Assisto Bruno Gelso, indagato per appropriazione indebita»"},
    "ruolo": {"valore": "indagato", "perche": "«Assisto Bruno Gelso, indagato per appropriazione indebita»"},
    "rito": {"valore": "penale", "perche": f"«{PM}»"},
    "stadio": {"valore": "indagini", "perche": "«le indagini preliminari sono in corso»"},
    "addebito_o_domanda": {"articolo": "646 c.p.", "descrizione": "appropriazione indebita",
                           "frase": "«procede per appropriazione indebita ai sensi dell'art. 646 c.p.»"},
    "istituto": None,
    "quesiti": ["Quali rischi corre Bruno Gelso?"]}


def _fit_reply(messages, kwargs):
    """The fit model faked: 646 c.p. and 2043 c.c. «si» on their quoted text; 1439 c.c. «no» with its «fatto_citato»."""
    human = messages[-1].content
    sid = {t: i for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", human, re.MULTILINE)}
    pm = next(i for t, i in sid.items() if t.startswith("Il pubblico ministero"))
    cash = next(i for t, i in sid.items() if t.startswith("Bruno Gelso lavorava"))
    harm = next(i for t, i in sid.items() if t.startswith("La società chiede"))
    items = {"646 c.p.": F._fit_item("646 c.p.", "si", f"Bruno Gelso custodiva la cassa del negozio ({cash}): «di cui "
                                                     "abbia, a qualsiasi titolo, il possesso»."),
             "1439 c.c.": dict(F._fit_item("1439 c.c.", "no", f"Il fatto ({pm}) non riguarda un «annullamento del "
                                                              "contratto».", ambito="rimedio"), fatto_citato=PM),
             "2043 c.c.": F._fit_item("2043 c.c.", "si", f"La società chiede a Bruno Gelso il risarcimento del danno "
                                                         f"({harm}): «obbliga colui che ha commesso il fatto a "
                                                         "risarcire il danno».", ambito="rimedio")}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def _link(topic, reqs, sents):
    """The element link faked: 646's appropriation and profit on the company's allegation only, its possession on the
    certain fact of the cash desk; 2043's damage on the company's claim."""
    sid = {s["text"]: s["id"] for s in sents}
    allegation = next(i for t, i in sid.items() if t.startswith("La società sostiene"))
    cash = next(i for t, i in sid.items() if t.startswith("Bruno Gelso lavorava"))
    claim = next(i for t, i in sid.items() if t.startswith("La società chiede"))
    out = {}
    for label, r in reqs:
        text = str(r.get("text") or "").lower()
        if label.startswith("646") and ("appropria" in text or "profitto" in text):
            out[r["id"]] = {"sostiene": [allegation], "contraddice": [], "manca": ""}
        elif label.startswith("646") and "possesso" in text:
            out[r["id"]] = {"sostiene": [cash], "contraddice": [], "manca": ""}
        elif label.startswith("2043") and "danno" in text:
            out[r["id"]] = {"sostiene": [claim], "contraddice": [], "manca": ""}
    return out


def _rows():
    return [{"_article": [n, c], "_grade": 3, "s": {"name": f"{n}.0.0", "plain_text": text, "titolo_articolo": rubrica},
             "d": {"id": doc, "name": name}} for n, c, text, rubrica, doc, name in ARTICLES]


def _round1(text, request="", lang="it", frame=None):
    topic = {"title": "Appropriazione degli incassi", "anchor": "646 c.p.", "how": "exact",
             "rubrica": "Appropriazione indebita.", "articles": list(LABELS), "matched": list(LABELS),
             "planned": len(LABELS), "grades": {x: 3 for x in LABELS}, "graded": len(LABELS), "facts": [],
             "source": "frame", "merged": []}
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
    out = []
    for title, *_ in jobs:
        labels = [x for x in LABELS if x in (_writer.state.get("records") or {})]
        out.append((title, "\n\n".join(f"[{i}] L'art. {x} riguarda i fatti narrati." for i, x in enumerate(labels, 1))))
    return out


def _no_network(*a, **k):
    raise RuntimeError("no network in the fixture")


def run_fixture(monkeypatch, tmp_path):
    for flag in ("CASE_FRAME", "APPLICABILITY_FIT", "LAWYER_METHOD"):
        monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    from src.rag import ai_chat, defensive_generation, issue_records, main, plan_then_write as ptw, render_computed
    # every model path refused at its source -- and below them, any outbound connection
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(ai_chat, "chat_model", F._NoModel())
    monkeypatch.setattr(ai_chat, "ChatOpenAI", F._refused)
    for mod in [m for name, m in list(sys.modules.items()) if name.startswith("src.rag.") and m is not None]:
        for attr in ("_call_chat", "_call_chat_with_tools", "embedding_model", "_embed_query_with_prefix"):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, F._refused)
    assert defensive_generation._call_chat is F._refused and ai_chat._call_chat is F._refused
    # the faked model steps
    monkeypatch.setattr(cf, "_default_call", lambda m, k: json.dumps(FRAME_REPLY, ensure_ascii=False))
    monkeypatch.setattr(ap, "_default_call", _fit_reply)
    monkeypatch.setattr(ai_chat, "call_chat_n", F._thesis_replies)
    monkeypatch.setattr(ptw, "negation_review", lambda items, known, call=None: [])
    monkeypatch.setattr(issue_records, "_link_call", _link)
    # the graph, round 1, the plan, the writer, the documents, §4
    monkeypatch.setattr(main, "driver", F._NoGraph())
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

    def isolated(name, fn, default=None):
        def run():
            try:
                return fn()
            except Exception as e:
                failed.append(f"{name}: {type(e).__name__}: {e}")
                raise
        return real_isolated(name, run, default)
    monkeypatch.setattr(flow, "_isolated", isolated)
    got = intake.request_intake(TEXT)
    real_round1 = flow.analyse_case

    def round1(*a, **k):
        state = real_round1(*a, **k)
        _writer.state = state
        return state
    monkeypatch.setattr(flow, "analyse_case", round1)
    out = flow.single_shot(TEXT, intake=got, case_id="v79143_fixture")
    return {"out": out, "state": out["state"], "answer": out["answer"], "events": events, "failed": failed}


def test_the_penal_fixture_end_to_end(monkeypatch, tmp_path):
    run = run_fixture(monkeypatch, tmp_path)
    state, answer, events = run["state"], run["answer"], run["events"]
    by = {}
    for e, d, ms in events:
        by.setdefault(e, []).append((d, ms))
    assert "**1. Sintesi della soluzione**" in answer and not state.get("method_refused")
    assert [x for x in run["failed"] if x.split(":")[0] in ("posture", "decompose_records", "applicability_fit",
                                                             "record_sides", "lawyer_method_plan")] == []
    frame = by["case_frame_read"][0][0]
    assert frame["status"] == "ok" and state["posture"]["role"] == "imputato"
    assert (state.get("client") or {}).get("name") == "Bruno Gelso"
    c = state["block_counters"]
    # 2: the «no» on 1439 c.c. verified on its «fatto_citato», removed
    fit = by["applicability_fit"][0][0]
    assert fit["prompt_hash"] == ap.prompt_hash() and fit["counters"]["fit_no_verified"] == 1
    assert c["fit_removed"] == 1 and "1439" not in answer
    # 1: 646's appropriation rests only on the company's allegation -- contestato, never integrato
    sections = dict(M.answer_sections(answer))
    norme = sections[M.title("norme")]
    assert "contestato (allegazione della società)" in norme
    assert c["elements_contestato"] >= 1 and c["theses_from_contestato"] >= 1
    plan = by["lawyer_method_plan"][0][0]
    assert plan["contestato"]["elements_contestato"] >= 1
    # the defence's theses on the penal charge: «L'accusa deve provare «…»» on 646 c.p.
    offered = [x for x in plan["candidate_articles"] if x]
    assert offered and all(x == "646 c.p." for x in offered), offered
    soluzioni = sections[M.title("soluzioni")]
    assert "L'accusa deve provare «" in soluzioni and "646 c.p." in soluzioni
    principale = [x for x in answer.splitlines() if "Linea principale" in x]
    assert principale and any("646 c.p." in x for x in principale)
    # 3: never «Ammettere la condotta», never the corpus line on a code article
    assert "Ammettere la condotta" not in answer and "mmettere la condotta" not in answer
    assert c["corpus_line_code"] == 0 and "corpus non contiene" not in answer
    # 4: 2043 c.c. excluded by the penal binding -- absent outside «Altri profili»
    binding = by["lawyer_method_frame_binding"][0][0]
    assert binding["mode"] == "penale" and binding["excluded"] == ["2043 c.c."]
    before, _, profili = norme.partition("**Altri profili**")       # §4: schede and tables, «Altri profili» aside
    assert "2043" not in before, [x for x in before.splitlines() if "2043" in x]
    mancanti = sections[M.title("fatti")].partition("**Fatti mancanti**")[2]
    for text in (mancanti, sections[M.title("orientamenti")], sections[M.title("obiezioni")], soluzioni):
        assert "2043" not in text, text
    assert "2043" not in next(x for x in answer.splitlines() if "**Esito probabile**" in x)
    assert "2043 c.c." in c["excluded_dropped"] and c["frame_binding_excluded"] == ["2043 c.c."]
    target = os.environ.get("FIXTURE_79_14_3_OUT")
    if target:
        three = [(e, d, ms) for e, d, ms in events if e in ("case_frame_read", "applicability_fit",
                                                             "lawyer_method_plan")]
        extra = [(e, d, ms) for e, d, ms in events if e in ("applicability_fit_discard", "lawyer_method_frame_binding",
                                                             "lawyer_method_frame_binding_rescue")]
        keys = [k for k in c if k.startswith(("fit_", "frame_", "norms_removed", "corpus_line", "elements_contestato",
                                              "theses_from_contestato", "excluded_dropped"))]
        with open(target, "w", encoding="utf-8") as f:
            f.write("# v79.14.3 synthetic penal fixture (tests/test_v79_14_3_fixture.py) through intake.request_intake "
                    "-> interactive_flow.single_shot -> write_answer -- the three log lines, production format\n\n")
            f.writelines(F._line(e, d, ms) + "\n" for e, d, ms in three)
            f.write("\n# the removal and the binding lines\n\n")
            f.writelines(F._line(e, d, ms) + "\n" for e, d, ms in extra)
            f.write("\n# the three lines' payloads, whole\n\n")
            for e, d, ms in three + extra:
                f.write(f"{e}: {json.dumps(d, ensure_ascii=False, default=str)}\n")
            f.write("\n# block_counters\n\n")
            f.write(json.dumps({k: c[k] for k in sorted(keys)}, ensure_ascii=False, indent=1) + "\n")
            f.write("\n# steps the pipeline isolated as failed (the faked boundaries)\n\n")
            f.writelines(f"{x[:200]}\n" for x in run["failed"])
            f.write("\n# Linea principale\n\n" + "\n".join(principale) + "\n")
            f.write("\n# the answer\n\n" + answer + "\n")
