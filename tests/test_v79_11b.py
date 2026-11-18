"""v79.11 block B -- the element schemas' generator and review surface, on synthetic article texts only (no model, no
Neo4j, no network):

B1. scripts/article_cards.py: call() takes system= and usage= (backward compatible: the card's SYSTEM by default, usage
    filled from the reply's "usage", a cut reply's too)
B2. scripts/article_cards_bulk.py --schemas: the endpoint (port 8003, qwen38-27b, thinking off, json mode, 6000 tokens),
    the order (carded first, then the codes by number, "(abrogato)" left out), resume (no line, source_hash,
    generation_hash, failed; --force; NEVER a reviewed label, notes-only lines not counted as reviewed), the retries on
    a cut reply and on invalid JSON, the accounting (made, failed, unusable by check, tokens, calls), the coverage and
    the summary; the card mode untouched
B3. scripts/schema_show.py: rendering (tables, the logic outline, notes), --edit (reviewer, date, source_hash,
    generated_schema, corrections, origin, checks with warnings, never refused for a check), re-edit of a reviewed line,
    --import-notes (idempotent, a reviewed schema kept, 62 n. 2 / n. 6 under one label), --stale, --coverage, --top,
    --quality
"""

import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    monkeypatch.setenv("ARTICLE_CARDS_FILE", str(tmp_path / "cards.jsonl"))
    for name in ("SCHEMA_LLM_BASE_URL", "SCHEMA_LLM_MODEL", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bulk():
    return _load("article_cards_bulk")


@pytest.fixture
def show():
    return _load("schema_show")


# synthetic article texts
PENAL = ("Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente, "
         "è punito con la reclusione da sei mesi a tre anni.")
OTHER = "1. Il custode che lascia deteriorare il bene affidatogli risponde del danno.\n2. Il danno si presume."
DOC = "Codice Penale 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", fonte=None, **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding,
           "fonte": {"comma": comma, "testo": testo if fonte is None else fonte}, "spiegazione": None}
    out.update(extra)
    return out


def _schema(**over):
    schema = {"tipo": ["sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
              "legittimato": [{"ruolo": "persona offesa", "condizione": None},
                              {"ruolo": "PA", "condizione": "se il fatto è commesso in servizio"}],
              "soggetto_passivo": [{"ruolo": "persona offesa", "condizione": None}], "rapporto": "lesioni personali",
              "elementi": [_el("E1", "cagiona ad alcuno una lesione personale"),
                           _el("E2", "dalla quale deriva una malattia nel corpo o nella mente", kind="evento",
                               branches=["malattia nel corpo", "malattia nella mente"], connective="o"),
                           _el("E3", "coscienza e volontà di cagionare la lesione", kind="elemento soggettivo",
                               grounding="inferred", spiegazione="delitto punito solo a titolo di dolo")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
              "eccezioni": [], "conseguenza": {"testo": "è punito con la reclusione da sei mesi a tre anni",
                                               "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema["elementi"][2]["fonte"] = {"comma": "1", "testo": None}
    schema.update(over)
    return schema


def _article(number="901", text=PENAL, code="c.p.", rubrica="Lesione sintetica"):
    return {"number": number, "code": code, "rubrica": rubrica, "text": text}


def _gen_line(number="901", text=PENAL, schema=None, model="qwen38-27b", **kw):
    from src.rag import element_schemas as es
    return es.make_line(_article(number, text), model, _schema() if schema is None else schema, doc=DOC, **kw)


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for x in lines:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def _gen_file():
    return os.environ["ARTICLE_SCHEMAS_FILE"]


def _rev_file():
    return os.environ["ARTICLE_SCHEMAS_REVIEWED_FILE"]


# --- B1 ---------------------------------------------------------------------------------------------------------------

class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(monkeypatch, ac, finish="stop", content="{\"tipo\": []}"):
    seen = []

    def urlopen(request, timeout=None):
        seen.append(json.loads(request.data.decode("utf-8")))
        return _Reply(json.dumps({"choices": [{"finish_reason": finish, "message": {"content": content}}],
                                  "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                                            "total_tokens": 150}}).encode("utf-8"))
    monkeypatch.setattr(ac.urllib.request, "urlopen", urlopen)
    return seen


def test_b1_call_takes_system_and_usage_and_stays_backward_compatible(monkeypatch, bulk):
    ac = bulk.ac
    seen = _fake_urlopen(monkeypatch, ac)
    end = {"url": "http://model.invalid:8003/v1", "key": "k", "model": "m", "max_tokens": 10}
    assert ac.call(end, "u") == '{"tipo": []}'
    assert seen[-1]["messages"][0] == {"role": "system", "content": ac.SYSTEM}          # the card's prompt by default
    usage = {}
    ac.call(end, "u", system="SCHEMA", usage=usage)
    assert seen[-1]["messages"] == [{"role": "system", "content": "SCHEMA"}, {"role": "user", "content": "u"}]
    assert usage["prompt_tokens"] == 120 and usage["completion_tokens"] == 30
    seen = _fake_urlopen(monkeypatch, ac, finish="length")
    usage = {}
    with pytest.raises(ValueError, match="cut at max_tokens"):
        ac.call(end, "u", usage=usage)
    assert usage["completion_tokens"] == 30                                             # a cut reply is counted too


# --- B2 ---------------------------------------------------------------------------------------------------------------

def test_b2_the_schema_endpoint(monkeypatch, bulk):
    from types import SimpleNamespace
    args = SimpleNamespace(url=None, model=None, json_mode=True)
    monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.5:8000/v1")
    end = bulk.schema_endpoint(args)
    assert end["url"] == "http://10.0.0.5:8003/v1" and end["model"] == "qwen38-27b" and end["max_tokens"] == 6000
    assert end["extra"] == {"chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}}
    assert "response_format" not in bulk.schema_endpoint(SimpleNamespace(url=None, model=None, json_mode=False))["extra"]
    monkeypatch.setenv("SCHEMA_LLM_BASE_URL", "http://schemas:9000/v1")
    monkeypatch.setenv("SCHEMA_LLM_MODEL", "other-model")
    end = bulk.schema_endpoint(args)
    assert end["url"] == "http://schemas:9000/v1" and end["model"] == "other-model"
    end = bulk.schema_endpoint(SimpleNamespace(url="http://given/v1", model="given", json_mode=True))
    assert end["url"] == "http://given/v1" and end["model"] == "given"
    monkeypatch.delenv("SCHEMA_LLM_BASE_URL")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.invalid/v1")
    with pytest.raises(SystemExit, match="8003"):
        bulk.schema_endpoint(args)


def test_b2_the_order_carded_first_then_the_codes_by_number(bulk):
    docs = {"CP": {n: {"number": n, "code": "c.p.", "rubrica": "", "text": f"Testo dell'articolo {n}."}
                   for n in ("10", "2", "9-bis", "9")},
            "CC": {"1": {"number": "1", "code": "c.c.", "rubrica": "", "text": "(abrogato)"},
                   "3": {"number": "3", "code": "c.c.", "rubrica": "", "text": "Testo tre."}},
            "Legge X": {"5": {"number": "5", "code": "Legge X", "rubrica": "", "text": "Testo cinque."}}}
    cards = [("Legge X", "5"), ("CP", "9"), ("CP", "77"), ("CC", "3")]       # 77 is no longer in its document
    items, bare, carded = bulk.schema_items(cards, docs, ["CP", "CC"])
    assert [(d, a["number"]) for d, a in items] == \
        [("Legge X", "5"), ("CP", "9"), ("CC", "3"), ("CP", "2"), ("CP", "9-bis"), ("CP", "10")]
    assert [(d, a["number"]) for d, a in bare] == [("CC", "1")] and len(carded) == 3


def _fake_call(replies, seen):
    """A model: each call pops the next reply -- a string, or an exception to raise."""
    def call(endpoint, user, system=None, usage=None):
        seen.append({"user": user, "system": system, "model": endpoint["model"]})
        reply = replies.pop(0) if replies else json.dumps(_schema())
        if usage is not None:
            usage.update({"prompt_tokens": 100, "completion_tokens": 50})
        if isinstance(reply, Exception):
            raise reply
        return reply
    return call


END = {"url": "http://model.invalid:8003/v1", "key": "k", "model": "qwen38-27b", "max_tokens": 6000, "extra": {}}


def test_b2_a_schema_line_from_the_model(monkeypatch, bulk):
    from src.rag import element_schemas as es
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call(["<think>x</think>```json\n" + json.dumps(_schema()) + "\n```"],
                                                     seen))
    stats = {}
    line = bulk.make_schema(END, DOC, _article(), stats)
    assert seen[0]["system"] == es.SCHEMA_SYSTEM and "TESTO UFFICIALE" in seen[0]["user"]
    assert es.usable(line) and line["label"] == "901 c.p." and line["doc"] == DOC and line["text"] == PENAL
    assert line["source_hash"] == es.text_hash(PENAL) and line["generation_hash"] == es.generation_hash(
        line["source_hash"], "qwen38-27b")
    assert line["usage"] == {"prompt_tokens": 100, "completion_tokens": 50} and stats["calls"] == 1
    assert line["schema"]["elementi"][0]["fonte"]["start"] == PENAL.index("cagiona")       # computed by the code


def test_b2_a_cut_reply_once_more_shorter_then_failed(monkeypatch, bulk):
    from src.rag import element_schemas as es
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call([ValueError("reply cut at max_tokens 6000")], seen))
    line = bulk.make_schema(END, DOC, _article())
    assert len(seen) == 2 and es.SHORTER_NOTE in seen[1]["user"] and es.SHORTER_NOTE not in seen[0]["user"]
    assert es.usable(line) and not line["checks"]["cut"] and line["usage"]["prompt_tokens"] == 200
    seen.clear()
    cut = ValueError("reply cut at max_tokens 6000")
    monkeypatch.setattr(bulk.ac, "call", _fake_call([cut, cut, json.dumps(_schema())], seen))
    line = bulk.make_schema(END, DOC, _article())
    assert len(seen) == 2 and line["checks"]["cut"] is True and "cut" in line["checks"]["failed"]
    assert line["schema"] is None and not es.usable(line)


def test_b2_invalid_json_once_more_with_the_json_note_then_failed(monkeypatch, bulk):
    from src.rag import element_schemas as es
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call(["non è JSON", json.dumps(_schema())], seen))
    line = bulk.make_schema(END, DOC, _article())
    assert len(seen) == 2 and es.JSON_NOTE in seen[1]["user"] and es.usable(line)
    seen.clear()
    monkeypatch.setattr(bulk.ac, "call", _fake_call(["non è JSON", "{rotto", json.dumps(_schema())], seen))
    line = bulk.make_schema(END, DOC, _article())
    assert len(seen) == 2 and line["checks"]["failed"].startswith("invalid JSON") and not line["checks"]["cut"]
    seen.clear()
    monkeypatch.setattr(bulk.ac, "call", _fake_call([ValueError("HTTP 500: down")], seen))
    line = bulk.make_schema(END, DOC, _article())                        # any other failure: no retry here
    assert len(seen) == 1 and line["checks"]["failed"] == "HTTP 500: down"


def _items(*numbers, text=PENAL):
    return [(DOC, _article(n, text)) for n in numbers]


def test_b2_resume_hash_generation_failed_force_and_never_a_reviewed_label(monkeypatch, tmp_path, bulk):
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call([], seen))
    path = _gen_file()
    items = _items("901", "902", "903")
    have = {}
    out = bulk.run_schemas(items, END, path, have, 2, bulk.Throttle(0))
    assert out["due"] == 3 and out["made"] == 3 and len(seen) == 3 and out["why"] == {"new": 3}
    have = bulk.read_file(path)
    assert bulk.run_schemas(items, END, path, have, 2, bulk.Throttle(0))["due"] == 0 and len(seen) == 3
    changed = [(DOC, _article("901", PENAL + " Testo cambiato."))] + items[1:]
    out = bulk.run_schemas(changed, END, path, have, 2, bulk.Throttle(0))
    assert out["why"] == {"source": 1} and len(seen) == 4
    out = bulk.run_schemas(changed, {**END, "model": "another"}, path, have, 2, bulk.Throttle(0))
    assert out["why"] == {"generation": 3} and seen[-1]["model"] == "another"
    have[(DOC, "902")]["checks"]["failed"] = "HTTP 500"
    out = bulk.run_schemas(changed, {**END, "model": "another"}, path, have, 2, bulk.Throttle(0))
    assert out["why"] == {"failed": 1}
    # the lawyer reviewed 903; a notes-only line for 902 is no review
    _write(_rev_file(), [{"label": "903 c.p.", "reviewed": True, "schema": _schema(), "source_hash": "x"},
                         {"label": "902 c.p.", "reviewed": False, "note_di_inquadramento": [{"testo": "n"}]}])
    reviewed = bulk.reviewed_labels()
    assert reviewed == {"903 c.p."}
    calls = len(seen)
    out = bulk.run_schemas(changed, END, path, have, 2, bulk.Throttle(0), force=True, reviewed=reviewed)
    assert out["due"] == 2 and out["reviewed_kept"] == 1 and len(seen) == calls + 2
    assert all("903" not in s["user"].split("\n")[0] for s in seen[calls:])
    assert bulk.schema_due(_article("903"), DOC, {}, "m", reviewed) == "reviewed"
    lines = open(path, encoding="utf-8").read().splitlines()
    bulk.compact(path, bulk.read_file(path))
    assert len(open(path, encoding="utf-8").read().splitlines()) == 3 < len(lines)


def test_b2_the_accounting(monkeypatch, capsys, bulk):
    ungrounded = _schema()
    ungrounded["elementi"][0] = _el("E1", "cagiona ad alcuno una frattura del femore", fonte="frattura del femore")
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call([json.dumps(_schema()), json.dumps(ungrounded),
                                                     ValueError("HTTP 400: bad")], seen))
    out = bulk.run_schemas(_items("901", "902", "903"), END, _gen_file(), {}, 1, bulk.Throttle(0), every=1)
    assert out["made"] == 2 and out["failed"] == 1 and out["unusable"] == {"ungrounded": 1}
    assert out["prompt_tokens"] == 300 and out["completion_tokens"] == 150 and out["calls"] == 3
    assert out["per_hour"] > 0 and out["mean_call_seconds"] >= 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3 and "3/3: 2 made, 1 failed, unusable by check: ungrounded 1" in printed[-1]
    assert "articles an hour" in printed[-1] and "s a call (3 calls)" in printed[-1]
    have = {(x["doc"], x["article"]): x for x in map(json.loads, open(_gen_file(), encoding="utf-8"))}
    cov = bulk.schema_coverage(_items("901", "902", "903", "904"), have)
    assert cov == {"c.p.": {"usable": 1, "unusable": 1, "without": 2}}
    summary = "\n".join(bulk.schema_summary(list(have.values()), cov))
    assert "c.p." in summary and "ungrounded" in summary and "300 prompt + 150 completion tokens" in summary


def test_b2_the_card_mode_is_untouched(monkeypatch, tmp_path, bulk):
    card = "\n".join(f"**{h}:** voce" for h in bulk.ac.HEADINGS)
    calls = []

    def call(endpoint, user):                        # the card mode's two-argument call, as before v79.11
        calls.append(user)
        return card
    monkeypatch.setattr(bulk.ac, "call", call)
    record = bulk.make({"model": "m"}, "D", _article())
    assert set(record) == {"doc", "article", "card", "model", "text_hash", "created_at", "checks"}
    assert record["card"] == card and calls == [bulk.ac.user_message(_article())]
    out = bulk.run([("D", _article("1")), ("D", _article("2"))], {"model": "m"}, str(tmp_path / "c.jsonl"), {}, 2,
                   bulk.Throttle(0))
    assert out == {"due": 2, "made": 2}
    assert bulk.due(_article("1"), "D", bulk.read_file(str(tmp_path / "c.jsonl"))) is False
    assert not os.path.exists(_gen_file())                        # the card mode never writes the schema file


def test_b2_main_dispatches_schemas_only_with_the_flag(monkeypatch, bulk):
    called = []
    monkeypatch.setattr(bulk, "schemas_main", lambda args: called.append(args))
    monkeypatch.setattr("sys.argv", ["article_cards_bulk.py", "--schemas", "--dry-run", "--no-json-mode"])
    bulk.main()
    assert len(called) == 1 and called[0].schemas and called[0].dry_run and called[0].json_mode is False


# --- B3 ---------------------------------------------------------------------------------------------------------------

def test_b3_render_tables_logic_outline_and_notes(show):
    from src.rag import element_schemas as es
    tree = {"op": "all_of", "args": [{"ref": "E1"}, {"op": "any_of", "args": [{"ref": "E2"}, {"op": "not", "arg":
            {"ref": "E3"}}]}, {"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"}, "else": {"ref": "E3"}}]}
    _write(_gen_file(), [_gen_line(schema=_schema(logic=tree))])
    _write(_rev_file(), [{"label": "901 c.p.", "reviewed": False, "note_di_inquadramento": [
        {"testo": "Attenzione al dolo.", "numero": None, "autore": "avvocato"}]}])
    v = show.view("901 c.p.")
    assert v["source"] == "generated" and not v["stale"]
    plain = show.render(v)
    for needed in ("ELEMENTI", "LOGICA", "ECCEZIONI", "CONSEGUENZA", "COMMI SPECIALI", "CONTROLLI",
                   "NOTE DELL'AVVOCATO", "TUTTI:", "  ALMENO UNO:", "    NON:", "  SE:", "  ALLORA:", "  ALTRIMENTI:",
                   "PA (se il fatto è commesso in servizio)", "generato dal modello", "qwen38-27b",
                   "«[nota dell'avvocato]» Attenzione al dolo.", "implicito: delitto punito solo a titolo di dolo",
                   "malattia nel corpo / malattia nella mente", "Lesione sintetica"):
        assert needed in plain, needed
    assert "logic_errors" not in plain                                          # every element is cited
    md = show.render(v, markdown=True)
    assert "| ID | Testo | Tipo di elemento | Comma | Fondamento | Alternative | Connettivo/parti |" in md
    assert "### Logica" in md and "- TUTTI:" in md and "  - ALMENO UNO:" in md
    assert show.view("999 c.p.") is None
    assert es.reviewed_file() == _rev_file()


def _yaml(tmp_path, text, name="corr.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_b3_edit_writes_the_reviewed_line(monkeypatch, tmp_path, show):
    from src.rag import element_schemas as es
    gen = _gen_line()
    _write(_gen_file(), [gen])
    path = _yaml(tmp_path, """
article: "901 c.p."
reviewer: Avv. Rossi
tipo: [sanction]
rapporto: lesioni
elementi_patch:
  E1: {kind: condotta, testo: "cagiona ad alcuno una lesione personale"}
  E3: {spiegazione: "solo dolo: nessuna ipotesi colposa"}
add_elementi:
  - {testo: "la malattia è nel corpo o nella mente", kind: evento, comma: "1", grounding: inferred,
     fonte: {comma: "1", testo: null}, spiegazione: "dal testo"}
note: "rivisto in udienza"
""")
    line, warnings = show.edit_file(path, today="2026-11-18")
    assert line["reviewed"] is True and line["reviewer"] == "Avv. Rossi" and line["review_date"] == "2026-11-18"
    assert line["source_hash"] == gen["source_hash"] and line["generated_schema"] == gen["schema"]
    assert line["schema"]["rapporto"] == "lesioni" and line["note"] == "rivisto in udienza"
    corr = line["corrections"]
    assert {"field": "rapporto", "from": "lesioni personali", "to": "lesioni", "by": "Avv. Rossi"} in corr
    assert not any(c.get("field") == "tipo" for c in corr)                     # unchanged: no correction
    assert any(c.get("element") == "E3" and c.get("field") == "spiegazione" for c in corr)
    assert any(c.get("element") == "E4" and c["from"] is None for c in corr)
    assert line["origin"] == {"E1": "modello", "E2": "modello", "E3": "avvocato", "E4": "avvocato"}
    assert line["text"] == PENAL and line["checks"]["logic_errors"] == ["E4: mai citato nella logic"]
    assert any(w.startswith("controllo logic_errors") for w in warnings)        # warned, saved all the same
    saved = [json.loads(x) for x in open(_rev_file(), encoding="utf-8")]
    assert len(saved) == 1 and saved[0]["label"] == "901 c.p."
    # a structural problem keeps the reviewed schema off the pipeline (the text path), never a crash
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    assert es.for_label("901 c.p.", PENAL) == (None, "unusable:logic_errors")
    # a clean re-edit: from the reviewed line, the model's schema kept verbatim, corrections accumulated
    path = _yaml(tmp_path, """
article: 901 c.p.
reviewer: Avv. Bianchi
date: 2026-11-20
logic: {op: all_of, args: [{ref: E1}, {ref: E2}, {ref: E3}, {ref: E4}]}
""", "again.yaml")
    again, warnings = show.edit_file(path)
    assert again["reviewer"] == "Avv. Bianchi" and again["review_date"] == "2026-11-20"
    assert again["generated_schema"] == gen["schema"] and again["source_hash"] == gen["source_hash"]
    assert again["schema"]["rapporto"] == "lesioni" and len(again["corrections"]) == len(corr) + 1
    assert again["corrections"][-1]["by"] == "Avv. Bianchi" and again["origin"]["E4"] == "avvocato"
    assert not any(w.startswith("controllo ") and "bloccante" in w for w in warnings)
    assert len(open(_rev_file(), encoding="utf-8").read().splitlines()) == 1          # one line per label
    got, why = es.for_label("901 c.p.", PENAL)
    assert why == "reviewed" and got["schema"]["rapporto"] == "lesioni"


def test_b3_edit_refuses_only_what_it_cannot_save(tmp_path, show):
    _write(_gen_file(), [_gen_line()])
    with pytest.raises(ValueError, match="reviewer"):
        show.edit_file(_yaml(tmp_path, "article: 901 c.p.\ntipo: [sanction]\n"))
    with pytest.raises(ValueError, match="nessuno schema"):
        show.edit_file(_yaml(tmp_path, "article: 950 c.p.\nreviewer: X\n"))
    assert not os.path.exists(_rev_file())
    line, warnings = show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\nelementi_patch: {E9: null}\n"
                                                    "colore: blu\n"), today="2026-11-18")
    assert line["corrections"] == [] and any("E9" in w for w in warnings) and any("colore" in w for w in warnings)
    assert show.main(["--edit", _yaml(tmp_path, "article: 950 c.p.\nreviewer: X\n", "bad.yaml")]) == 2


NOTES = """# Note

<!-- commento -->

## art. 901 c.p.
Errori da evitare:
- Non confondere dolo e colpa.

## art. 62 n. 2 c.p.
- Nota sul numero due.

## art. 62 n. 6 c.p.
- Nota sul numero sei.
"""


def test_b3_import_notes_idempotent_and_keeping_a_reviewed_schema(tmp_path, show):
    from src.rag import element_schemas as es
    _write(_gen_file(), [_gen_line()])
    show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\nrapporto: lesioni\n"), today="2026-11-18")
    before = es.reviewed()["901 c.p."]
    notes = tmp_path / "note.md"
    notes.write_text(NOTES, encoding="utf-8")
    got = show.import_notes(str(notes), today="2026-11-18")
    assert got["articles"] == ["901 c.p.", "62 c.p."] and sorted(got["changed"]) == ["62 c.p.", "901 c.p."]
    rev = es.reviewed()
    assert rev["901 c.p."]["reviewed"] is True and rev["901 c.p."]["schema"] == before["schema"]
    assert rev["901 c.p."]["note_di_inquadramento"] == [
        {"testo": "Errori da evitare: Non confondere dolo e colpa.", "numero": None, "autore": "avvocato",
         "fonte": str(notes), "importato": "2026-11-18"}]
    sixty = rev["62 c.p."]
    assert sixty["reviewed"] is False and "schema" not in sixty
    assert [(n["numero"], n["testo"]) for n in sixty["note_di_inquadramento"]] == \
        [("2", "Nota sul numero due."), ("6", "Nota sul numero sei.")]
    content = open(_rev_file(), encoding="utf-8").read()
    again = show.import_notes(str(notes), today="2026-12-01")              # the same notes: nothing changes
    assert again["changed"] == [] and open(_rev_file(), encoding="utf-8").read() == content
    # an edited note replaces the earlier import of the same file; a note from another source is kept
    line = dict(es.reviewed()["901 c.p."])
    line["note_di_inquadramento"] = line["note_di_inquadramento"] + [{"testo": "altra", "fonte": "a mano"}]
    show.write_overlay({"901 c.p.": line})
    notes.write_text(NOTES.replace("dolo e colpa", "dolo e preterintenzione").replace("## art. 62 n. 6 c.p.\n- Nota sul "
                                                                                       "numero sei.\n", ""),
                     encoding="utf-8")
    show.import_notes(str(notes), today="2026-12-01")
    rev = es.reviewed()
    assert [n["testo"] for n in rev["901 c.p."]["note_di_inquadramento"]] == \
        ["altra", "Errori da evitare: Non confondere dolo e preterintenzione."]
    assert [n["numero"] for n in rev["62 c.p."]["note_di_inquadramento"]] == ["2"]
    # an article gone from the file: its imported notes go; a notes-only line left empty is dropped
    notes.write_text("## art. 901 c.p.\n- Non confondere dolo e preterintenzione.\n", encoding="utf-8")
    show.import_notes(str(notes))
    assert "62 c.p." not in es.reviewed() and es.reviewed()["901 c.p."]["reviewed"] is True
    # a later edit keeps the notes
    show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: Y\n", "y.yaml"))
    assert any(n.get("fonte") == "a mano" for n in es.reviewed()["901 c.p."]["note_di_inquadramento"])


def test_b3_import_of_the_lawyer_notes_file_gives_its_six_articles(show):
    from src.rag import element_schemas as es
    got = show.import_notes(str(ROOT / "docs" / "lawyer" / "note_inquadramento.md"), today="2026-11-18")
    assert sorted(got["articles"]) == sorted(["584 c.p.", "41 c.p.", "43 c.p.", "62 c.p.", "589 c.p.", "590 c.p."])
    rev = es.reviewed()
    assert [n["numero"] for n in rev["62 c.p."]["note_di_inquadramento"]] == ["2", "6"]
    assert rev["584 c.p."]["note_di_inquadramento"][0]["fonte"] == "docs/lawyer/note_inquadramento.md"
    assert got["path"] == _rev_file()                                          # the tmp overlay, never the repo's


def test_b3_stale_coverage_and_quality(monkeypatch, tmp_path, show, capsys):
    from src.rag import element_schemas as es
    bad = _schema()
    bad["elementi"][0] = _el("E1", "cagiona ad alcuno una frattura", fonte="frattura")
    failed = es.make_line(_article("903"), "qwen38-27b", None, failed="HTTP 500", doc=DOC)
    _write(_gen_file(), [_gen_line("901"), _gen_line("902", schema=bad), failed])
    _write(os.environ["ARTICLE_CARDS_FILE"], [{"doc": DOC, "article": "901", "card": "x"},
                                              {"doc": DOC, "article": "904", "card": "x"},
                                              {"doc": "Codice Civile 2026", "article": "7", "card": "x"}])
    assert show.quality_lines()[-1] == "nessuna revisione"
    show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\nrapporto: lesioni\n"), today="2026-11-18")
    show.edit_file(_yaml(tmp_path, "article: 902 c.p.\nreviewer: X\n", "b.yaml"), today="2026-11-18")
    assert show.stale() == []
    _write(_gen_file(), [_gen_line("901", PENAL + " Comma aggiunto."), _gen_line("902", schema=bad), failed])
    found = show.stale()
    assert [x["label"] for x in found] == ["901 c.p."] and found[0]["reviewer"] == "X"
    cov = show.coverage()
    assert cov["c.p."] == {"usable": 1, "unusable": 1, "without": 2, "reviewed": 2}
    assert cov["c.c."] == {"without": 1}
    lines = show.coverage_lines(cov)
    assert lines[0].startswith("Codice") and lines[-1].startswith("totale")
    q = {row.rsplit("  ", 1)[0].strip(): row.rsplit("  ", 1)[1].strip() for row in show.quality_lines()[2:] if row}
    assert q["schemi corretti dall'avvocato"] == "0.5" and q["schemi rivisti confrontati"] == "2"
    assert q["precisione degli elementi"] == "1.0" and q["logica esatta"] == "1.0"
    assert show.main(["--stale"]) == 0 and "901 c.p." in capsys.readouterr().out
    assert show.main(["--quality", "--markdown"]) == 0 and "| Misura | Valore |" in capsys.readouterr().out


def test_b3_a_review_goes_stale_when_the_text_changes(monkeypatch, tmp_path, bulk, show):
    """v79.11 review: the generator never makes a reviewed label again, but counts it «reviewed_stale» when its text
    changed and records the text now beside the generated line -- --stale, view and the runtime see it; the overlay
    is never written by the generator; --edit adopts the text now only with «testo_attuale: true»."""
    from src.rag import element_schemas as es
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    _write(_gen_file(), [_gen_line("901")])
    show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\nrapporto: lesioni\n"), today="2026-11-18")
    old, new_text = es.text_hash(PENAL), PENAL + " Comma aggiunto."
    assert es.reviewed()["901 c.p."]["source_hash"] == old and show.stale() == []
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call([], seen))
    reviewed = bulk.reviewed_hashes()
    assert reviewed == {"901 c.p.": old} and bulk.reviewed_labels() == {"901 c.p."}
    items = [(DOC, _article("901", new_text))]
    assert bulk.schema_due(items[0][1], DOC, {}, "m", reviewed, force=True) == "reviewed_stale"
    assert bulk.schema_due(_article("901"), DOC, {}, "m", reviewed) == "reviewed"
    assert bulk.schema_due(items[0][1], DOC, {}, "m", {"901 c.p."}) == "reviewed"       # a set of labels: as before
    overlay_before = open(_rev_file(), encoding="utf-8").read()
    have = bulk.read_file(_gen_file())
    out = bulk.run_schemas(items, END, _gen_file(), have, 1, bulk.Throttle(0), force=True, reviewed=reviewed)
    assert seen == [] and out["due"] == 0 and out["reviewed_kept"] == 1 and out["reviewed_stale"] == 1
    assert out["why"] == {} and open(_rev_file(), encoding="utf-8").read() == overlay_before
    line = bulk.read_file(_gen_file())[(DOC, "901")]
    assert line["source_hash"] == old and line["current_source_hash"] == es.text_hash(new_text)
    assert line["current_text"] == new_text and line["text"] == PENAL and line["schema"] == _gen_line("901")["schema"]
    n = len(open(_gen_file(), encoding="utf-8").read().splitlines())
    bulk.run_schemas(items, END, _gen_file(), bulk.read_file(_gen_file()), 1, bulk.Throttle(0), reviewed=reviewed)
    assert len(open(_gen_file(), encoding="utf-8").read().splitlines()) == n          # recorded once
    assert [x["label"] for x in show.stale()] == ["901 c.p."]
    seen_view = show.view("901 c.p.")
    assert seen_view["stale"] is True and seen_view["text"] == new_text
    assert es.for_label("901 c.p.")[1] == "reviewed_stale" and es.for_label("901 c.p.", new_text)[1] == "reviewed_stale"
    _, warnings = show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\nnote: ritocco\n", "b.yaml"),
                                 today="2026-11-19")
    assert any("testo_attuale" in w for w in warnings) and es.reviewed()["901 c.p."]["source_hash"] == old
    assert es.for_label("901 c.p.", new_text)[1] == "reviewed_stale"
    _, warnings = show.edit_file(_yaml(tmp_path, "article: 901 c.p.\nreviewer: X\ntesto_attuale: true\n", "c.yaml"),
                                 today="2026-11-20")
    rev = es.reviewed()["901 c.p."]
    assert rev["source_hash"] == es.text_hash(new_text) and rev["text"] == new_text and rev["schema"]["rapporto"] == "lesioni"
    assert show.stale() == [] and es.for_label("901 c.p.", new_text)[1] == "reviewed"
    assert bulk.schema_due(items[0][1], DOC, {}, "m", bulk.reviewed_hashes()) == "reviewed"


def test_b3_top_pack_by_frequency_then_the_file_order(tmp_path, show):
    _write(_gen_file(), [_gen_line("901"), _gen_line("902"), _gen_line("903")])
    runs = tmp_path / "runs" / "r1" / "cases"
    runs.mkdir(parents=True)
    (runs / "c1.json").write_text(json.dumps({"pool": [{"label": "903 c.p.", "document": DOC}],
                                              "plan": {"norme": ["903 c.p.", "902 c.p."]}}), encoding="utf-8")
    out = tmp_path / "pack.md"
    text = show.top_pack(2, str(out), str(tmp_path / "runs" / "*" / "cases" / "*.json"))
    assert out.read_text(encoding="utf-8") == text
    assert text.index("## Art. 903 c.p.") < text.index("## Art. 902 c.p.") and "Art. 901" not in text
    assert text.index("### Testo ufficiale") < text.index("### Elementi") and "> Chiunque cagiona" in text
    text = show.top_pack(2, None, str(tmp_path / "none" / "*.json"))
    assert "ordine del file generato" in text and text.index("Art. 901") < text.index("Art. 902")


def test_b3_main_shows_an_article(capsys, show):
    _write(_gen_file(), [_gen_line()])
    assert show.main(["--article", "901 c.p.", "--doc", DOC, "--markdown"]) == 0
    assert "## Art. 901 c.p. — Lesione sintetica" in capsys.readouterr().out
    assert show.main(["--article", "950 c.p."]) == 1
    assert show.main(["--coverage"]) == 0 and "totale" in capsys.readouterr().out
