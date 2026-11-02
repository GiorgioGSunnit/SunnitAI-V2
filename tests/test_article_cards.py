"""scripts/article_cards.py: no model, no database."""

import importlib.util
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ac = _module("article_cards", ROOT / "scripts" / "article_cards.py")
ev = _module("eval_narrative_for_cards", ROOT / "tests" / "eval_narrative.py")


def _audit():
    sentences = [{"sentence": "L'art. 2043 c.c. richiede sempre il dolo.", "verdict": "contraddetta", "kind": "regola"},
                 {"sentence": "L'art. 2043 c.c. richiede sempre il dolo.", "verdict": "contraddetta", "kind": "regola"},
                 {"sentence": "L'art. 1218 c.c. regola altro.", "verdict": "assente", "kind": "regola"},
                 {"sentence": "L'art. 589 c.p. si applica a Tizio.", "verdict": "assente", "kind": "applicazione"},
                 {"sentence": "L'art. 1490 c.c. è citato.", "verdict": "supportata", "kind": "regola"},
                 {"sentence": "La Corte (art. 95 d.lgs. 36/2023) dice altro.", "verdict": "assente", "kind": "regola"}]
    return {"cases": [{"id": "h1@r1", "provenance": {"sentences": sentences},
                       "understanding": {"lost": [{"article": "art. 582/583 cp"}, {"article": "art. 1 tuel"}]}},
                      {"id": "h1@r2", "provenance": {"sentences": sentences[:1]}, "understanding": {"lost": []}}]}


def test_tally_counts_the_audits_and_the_eval_keys():
    table = ac.tally([_audit()], [{"expected": [{"code": "cc", "article": "2043"},
                                                {"code": "cp", "any_of": ["582", "583"]},
                                                {"code": "consumo", "article": "128"}]}], ev.mentions)
    assert table[("2043", "cc")]["counts"] == Counter(contraddetta=1, expected=1)     # one sentence, once per case
    assert table[("1218", "cc")]["counts"]["assente"] == 1
    assert ("589", "cp") not in table and ("1490", "cc") not in table                 # an application, a supported one
    assert table[("582", "cp")]["counts"] == Counter(lost=1, expected=1)
    assert table[("583", "cp")]["counts"]["lost"] == 1 and ("128", "consumo") not in table
    assert "dolo" in table[("2043", "cc")]["example"]
    assert "frasi contraddette: 1" in ac.reason(table[("2043", "cc")])


def test_choose_alternates_the_lists_and_moves_unfilled_slots():
    def entry(**counts):
        return {"counts": Counter(counts), "example": ""}
    table = {("1", "cc"): entry(contraddetta=5), ("2", "cc"): entry(expected=9), ("3", "cc"): entry(assente=4),
             ("4", "cc"): entry(expected=1), ("5", "cp"): entry(lost=1), ("6", "cpc"): entry(assente=1),
             ("7", "cc"): entry(assente=1)}
    picked, notes = ac.choose(table, {"c.c.": 3, "c.p.": 2, "c.p.c.": 1, "c.p.p.": 1},
                              lambda ref: ref != ("3", "c.c."))
    assert picked[:3] == [("1", "cc"), ("2", "cc"), ("4", "cc")]     # 3 has no text
    assert ("7", "cc") in picked and len(picked) == 6
    assert any("art. 3 c.c." in n for n in notes) and any("c.p.:" in n for n in notes) and any("c.p.p.:" in n for n in notes)


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    def __init__(self, sections):
        self.sections = sections

    def run(self, query, **p):
        if "RETURN d.id AS id" in query:
            return _Rows([{"id": "old", "name": "Codice Civile vecchio"}, {"id": "cc", "name": "Codice Civile"}])
        doc = p.get("doc") or p["items"][0]["doc"]
        num = p.get("n") or p["items"][0]["num"]
        found = {k: v for k, v in self.sections.get(doc, {}).items() if k == num or k.startswith(num + ".")}
        if "items" in p:
            return _Rows([{"doc": doc, "num": num, "d": {"id": doc},
                           "s": {"name": k, "plain_text": v, "titolo_articolo": "Rubrica" if doc == "cc" else ""}}
                          for k, v in found.items()])
        return _Rows([{"name": k, "text": v} for k, v in found.items()])


def test_the_article_is_read_with_the_shared_function_and_its_date():
    text = "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga al risarcimento."
    session = _Session({"old": {"2043.0.0": text + " ma più lungo della copia corrente, senza rubrica."},
                        "cc": {"2043.0.0": "In vigore dal 11 dicembre 2021", "2043": text, "2044": "x" * 70}})
    article = ac.read_article(session, "2043", "c.c.")
    assert article["document"] == "Codice Civile" and article["text"] == text       # the current copy, not the longer
    assert article["date"] == "11 dicembre 2021" and not article["fragment"]
    assert "in vigore dal 11 dicembre 2021 (Codice Civile)" in ac.version_line(article)
    assert ac.read_article(session, "9", "c.c.") == {}
    assert "non presente nel DB" in ac.version_line({**article, "date": ""})
    assert "VICINI" not in ac.user_message(article) and "vicini" not in ac.SYSTEM


def test_reasoning_is_left_out_and_the_code_flags_what_it_can_see():
    assert ac.clean_reply("<think>penso\n</think>\n**Regola principale:** x") == "**Regola principale:** x"
    assert ac.clean_reply("ragiono\n</think>\ncarta") == "carta"
    card = "\n".join(f"**{h}:** nessuno" for h in ac.HEADINGS[:5]) + " entro 30 giorni e art. 2044"
    article = {"number": "2043", "code": "c.c.", "rubrica": "", "text": "entro 8 giorni"}
    assert ac.checks(card, article) == ["voci mancanti: Articoli collegati", "numeri non presenti nel testo: 30, 2044"]


def test_a_card_is_saved_once_and_b_has_thinking_off(tmp_path, monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            finish = "length" if "cut" in seen[-1][1]["messages"][1]["content"] else "stop"
            body = json.dumps({"choices": [{"finish_reason": finish,
                                            "message": {"content": "<think>x</think>**Regola principale:** r"}}]})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "qwen25")
    assert list(ac.endpoints(SimpleNamespace(with_b=False))) == ["A"]     # by default Qwen2.5 alone, no thinking
    ends = ac.endpoints(SimpleNamespace(with_b=True, b_url=None, b_model="qwen3-32b-awq", b_thinking=False))
    assert ends["B"]["url"].endswith(":8010/v1") and ends["A"]["model"] == "qwen25"
    ends["B"]["url"] = ends["A"]["url"]
    article = {"number": "2043", "code": "c.c.", "rubrica": "R", "text": "testo"}
    for _ in range(2):
        got = ac.card(ends["B"], "B", article, str(tmp_path))
    assert got["text"] == "**Regola principale:** r" and len(seen) == 1
    assert seen[0][0] == "/v1/chat/completions" and seen[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0][1]["temperature"] == 0 and "chat_template_kwargs" not in {**ends["A"]}
    failed = ac.card(ends["A"], "A", {**article, "text": "cut"}, str(tmp_path))
    server.shutdown()
    assert "cut at max_tokens" in failed["error"] and len(list(tmp_path.iterdir())) == 1


def test_endpoints_refuse_a_base_url_without_a_port(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(SystemExit, match="--b-url"):
        ac.endpoints(SimpleNamespace(with_b=True, b_url=None, b_model="b", b_thinking=False))


def test_the_export_holds_text_and_both_cards_in_order():
    article = {"number": "2043", "code": "c.c.", "rubrica": "Risarcimento", "text": "Riga uno\n\nRiga due",
               "document": "Codice Civile", "date": "", "fragment": True}
    item = {"key": ("2043", "cc"), "article": article, "why": "casi in cui è atteso: 2", "example": "Es",
            "cards": {"A": {"text": "**Regola principale:** a\n**Termini:** b"}, "B": {"error": "HTTP 500: no"}}}
    text = ac.render([item], ["c.p.: solo 1"], {"A": "qwen25", "B": "qwen3"}, ["a/audit.json"])
    assert text.index("### Testo ufficiale") < text.index("### Scheda A (qwen25)") < text.index("### Scheda B (qwen3)")
    assert "> Riga uno\n>\n> Riga due" in text and "Scheda non generata: HTTP 500" in text
    only_a = ac.render([{**item, "cards": {"A": item["cards"]["A"]}}], [], {"A": "qwen25"}, [])
    assert "Scheda B" not in only_a
    assert "a\n\n**Termini:** b" in text and "solo in frammenti" in text and "Da controllare: voci mancanti" in text and "c.p.: solo 1" in text


def test_the_card_asks_for_penalties_every_comma_and_every_item():
    assert ac.HEADINGS.index("Pene") < ac.HEADINGS.index("Termini")
    assert "OGNI comma" in ac.SYSTEM and "ogni voce di un elenco" in ac.SYSTEM and "mai quelli di una nota" in ac.SYSTEM
    assert "**Pene:**" in ac.SYSTEM


def test_the_same_articles_come_back_from_an_earlier_export(tmp_path):
    export = tmp_path / "article_cards.md"
    export.write_text("| # | articolo | rubrica | perché |\n|---:|---|---|---|\n| 1 | art. 61 c.p. | R | x |\n"
                      "| 2 | art. 337-ter c.c. | - | y |\n| 3 | art. 113 c.p.c. | - | z |\n", encoding="utf-8")
    assert ac.articles_from(str(export)) == "61 c.p.,337-ter c.c.,113 c.p.c."
    assert ac.articles_from(None) == ""
