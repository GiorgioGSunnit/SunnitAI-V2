"""scripts/article_cards_bulk.py and ARTICLE_CARDS_ENABLED: no model, no database."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import article_cards, grounded_review  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("article_cards_bulk", ROOT / "scripts" / "article_cards_bulk.py")
bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk)
ac = bulk.ac

CARD = "\n".join(f"**{h}:** voce di {h}" for h in ac.HEADINGS)
ARTICLE = {"number": "589", "code": "c.p.", "rubrica": "Omicidio colposo",
           "text": "1) Chiunque cagiona per colpa la morte. [Soggetto sotto effetto di sostanze stupefacenti o psicotrope nel fatto.] (abrogato)"}


def test_the_prompt_keeps_abrogated_text_dead_and_official_numbers():
    assert "«abrogato»" in ac.SYSTEM and "non rinumerare" in ac.SYSTEM and ac.A_TOKENS == 4000


def test_frequency_counts_pools_and_plans_a_case_at_a_time(tmp_path):
    case = {"pool": [{"label": "2043 c.c.", "document": "Codice Civile 2026"}, {"label": "128 Codice del Consumo", "document": "Consumo"}],
            "plan": {"norme": ["2043 c.c.", "61 c.p."]}}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    counts = bulk.frequency([str(path), str(path), str(tmp_path / "gone.json")])
    assert counts[("Codice Civile 2026", "2043", "c.c.")] == 4 and counts[(None, "61", "c.p.")] == 2
    assert counts[("Consumo", "128", "Codice del Consumo")] == 2


def test_the_order_is_top_then_codes_then_laws_each_once():
    assert bulk.order_refs([("cc", "2043")], {"cc": ["1", "2043"], "cp": ["5"]}, {"consumo": ["9"]}) == \
        [("cc", "2043"), ("cc", "1"), ("cp", "5"), ("consumo", "9")]
    assert sorted(["10", "9-bis", "9"], key=bulk.number_key) == ["9", "9-bis", "10"]


def test_abrogated_text_used_as_live_is_seen_unless_called_abrogated():
    live = "**Presupposti:** soggetto sotto effetto di sostanze stupefacenti o psicotrope"
    assert bulk.abrogated_as_live(live, ARTICLE["text"]) == [live]
    assert bulk.abrogated_as_live("- Soggetto sotto effetto di sostanze stupefacenti o psicotrope: abrogato", ARTICLE["text"]) == []


def test_renumbered_items_and_commi():
    text = "1. Il giudice decide: a) prima; b) seconda. 3-bis. Altro comma; 11-quater) voce."
    assert bulk.renumbered("co. 3-bis; lett. a) e 11-quater)", text) == []
    assert bulk.renumbered("3) voce", text) == ["3"] and bulk.renumbered("comma 2", text) == ["2"]
    assert bulk.renumbered("comma 2", "Il giudice decide. Altro comma senza numero.") == []          # the text numbers none


def test_empty_fields():
    assert bulk.empty_fields(CARD) == []
    assert bulk.empty_fields(CARD.replace("voce di Pene", "")) == ["Pene"]
    assert bulk.empty_fields(CARD.replace("**Termini:** voce di Termini\n", "")) == ["Termini"]
    assert bulk.empty_fields(CARD.replace("voce di Termini", "Nessuno nel testo.")) == []


def test_checks_put_together():
    found = bulk.checks("**Regola principale:** entro 30 giorni", ARTICLE)
    assert found["numbers_not_in_text"] == ["30"] and "Pene" in found["empty_fields"]


def test_a_cut_reply_is_tried_once_shorter_then_failed(monkeypatch):
    seen = []

    def call(endpoint, user):
        seen.append(user)
        if len(seen) == 1 or "second-fails" in user:
            raise ValueError("reply cut at max_tokens 4000")
        return CARD
    monkeypatch.setattr(ac, "call", call)
    end = {"model": "m"}
    record = bulk.make(end, "Codice Penale", ARTICLE)
    assert record["card"] == CARD and bulk.SHORTER in seen[1] and not record["checks"]["cut"]
    seen.clear()
    record = bulk.make(end, "Codice Penale", {**ARTICLE, "text": "second-fails"})
    assert record["card"] == "" and record["checks"]["cut"] and "cut" in record["checks"]["failed"] and len(seen) == 2
    assert set(record) == {"doc", "article", "card", "model", "text_hash", "created_at", "checks"}


def test_resuming_only_the_changed_or_failed(tmp_path, monkeypatch):
    path = str(tmp_path / "sub" / "cards.jsonl")
    calls = []
    monkeypatch.setattr(ac, "call", lambda endpoint, user: calls.append(user) or CARD)
    first = {**ARTICLE, "number": "1"}
    second = {**ARTICLE, "number": "2"}
    end = {"model": "m"}
    items = [("D", first), ("D", second)]
    bulk.run(items, end, path, {}, 2, bulk.Throttle(0))
    assert len(calls) == 2
    have = bulk.read_file(path)
    assert bulk.run(items, end, path, have, 2, bulk.Throttle(0)) == {"due": 0} and len(calls) == 2
    changed = [("D", {**first, "text": "Testo cambiato di proposito."}), ("D", second)]
    bulk.run(changed, end, path, have, 2, bulk.Throttle(0))
    assert len(calls) == 3
    have[("D", "2")]["checks"]["failed"] = "HTTP 500"
    assert bulk.run(changed, end, path, have, 2, bulk.Throttle(0))["due"] == 1
    assert len(open(path, encoding="utf-8").readlines()) == 4
    bulk.compact(path, bulk.read_file(path))
    assert len(open(path, encoding="utf-8").readlines()) == 2
    assert bulk.run(changed, end, path, bulk.read_file(path), 2, bulk.Throttle(0), force=True)["due"] == 2


def test_the_throttle_spaces_the_starts():
    now, slept = [0.0], []
    throttle = bulk.Throttle(3600, clock=lambda: now[0], sleep=lambda s: slept.append(round(s, 3)))
    for _ in range(3):
        throttle.wait()
    assert slept == [1.0, 2.0]
    assert bulk.Throttle(0).gap == 0


def test_the_summary_and_the_stratified_sample():
    records = [{"doc": "A", "article": str(i), "card": CARD, "model": "m", "text_hash": "x", "checks": {}} for i in range(90)]
    records += [{"doc": "B", "article": str(i), "card": CARD, "model": "m", "text_hash": "x",
                 "checks": {"renumbered": ["3"]}} for i in range(10)]
    text = "\n".join(bulk.summary(records))
    assert "100 cards" in text and "90 with no check raised" in text and "renumbered" in text
    chosen = bulk.sample(records, 10, 1)
    assert len(chosen) == 10 and sum(r["doc"] == "B" for r in chosen) == 1 and chosen == bulk.sample(records, 10, 1)
    out = bulk.render_sample(chosen[:2], {})
    assert "### Testo ufficiale" in out and "### Scheda (m)" in out and "(non trovato)" in out


def test_a_text_that_is_only_abrogato_gets_no_card():
    assert not bulk.has_body({"text": "(abrogato)"}) and not bulk.has_body({"text": "Abrogato."}) and bulk.has_body(ARTICLE)


# --- ARTICLE_CARDS_ENABLED -----------------------------------------------------------------------------------------

LONG = " ".join(f"Comma numero {i} del lungo articolo sul risarcimento del danno." for i in range(60))
SHORT = "Il debitore risponde del danno."


def _row(text=LONG, **extra):
    return {"d": {"name": "Codice Civile"}, "s": {"plain_text": text}, "_article": ("1218", "c.c."), **extra}


@pytest.fixture
def cards(tmp_path, monkeypatch):
    path = tmp_path / "cards.jsonl"

    def write(text=LONG, **record):
        line = {"doc": "Codice Civile", "article": "1218", "card": "**Regola principale:** il debitore risponde.",
                "model": "m", "text_hash": article_cards.text_hash(text), "checks": {"failed": "", "renumbered": []}, **record}
        path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    write()
    monkeypatch.setenv(article_cards.FILE_ENV, str(path))
    monkeypatch.setenv(article_cards.FLAG, "true")
    monkeypatch.setattr(article_cards, "_signature", None)
    return write


def test_off_the_block_is_the_text(cards, monkeypatch):
    monkeypatch.setenv(article_cards.FLAG, "false")
    assert article_cards.HEAD not in dg._source_block([_row()], 1500)


def test_over_budget_the_card_instead_of_the_text(cards):
    block = dg._source_block([_row()], 1500)
    assert block.startswith("[art. 1218 c.c.]\n" + article_cards.HEAD) and "il debitore risponde" in block
    assert "Comma numero 3" not in block


def test_the_cards_given_are_logged(cards, monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    dg._source_block([_row()], 1500)
    assert ("article_cards_given", {"articles": ["art. 1218 c.c."], "of": 1}) in events


def test_within_budget_a_short_text_and_the_card(cards):
    cards(SHORT)
    block = dg._source_block([_row(SHORT)], 1500)
    assert block.index(SHORT) < block.index(article_cards.HEAD) < block.index("il debitore risponde")


@pytest.mark.parametrize("record", [{"text_hash": "altro"}, {"checks": {"renumbered": ["3"]}}, {"checks": {"failed": "x"}}, {"card": ""}])
def test_a_stale_or_doubtful_card_leaves_the_official_text(cards, record):
    cards(**record)
    assert article_cards.HEAD not in dg._source_block([_row()], 1500)


def test_the_reviews_read_the_official_text(cards):
    assert article_cards.HEAD not in dg._source_block([_row()], 1500, official=True)
    source = (ROOT / "src" / "rag" / "grounded_review.py").read_text(encoding="utf-8")
    assert "dg._source_block(mine, FETCH_CHARS, official=True)" in source and grounded_review


def test_a_new_file_is_read_without_a_restart(cards):
    assert "risponde" in dg._source_block([_row()], 1500)
    cards(card="**Regola principale:** nuova scheda più lunga di prima.")
    assert "nuova scheda" in dg._source_block([_row()], 1500)


def test_no_file_no_cards(monkeypatch, tmp_path):
    monkeypatch.setenv(article_cards.FLAG, "true")
    monkeypatch.setenv(article_cards.FILE_ENV, str(tmp_path / "none.jsonl"))
    assert article_cards.material(_row(), 1500) is None


def test_a_document_is_read_whole_and_each_article_by_the_shared_rule():
    sections = [("61", "Articolo", "Aggravano il reato le circostanze seguenti: 1) i motivi abietti. (1) Numero aggiunto dall'art. 1."),
                ("61.0.0", "Sottocomma", "Aggravano il reato le circostanze seguenti:"), ("61.1.0", "Sottocomma", "i motivi abietti."),
                ("559", "Articolo", "(1) Articolo abrogato dall'art. 2."), ("att-5", "Articolo", "Altra numerazione.")]

    class Session:
        def run(self, query, **params):
            assert "$name" in query and params["name"] == "Codice Penale 2026"
            rows = [{"s": {"name": n, "type": t, "plain_text": x, "titolo_articolo": "Rubrica (1)"}, "next": None, "doc": "cp"}
                    for n, t, x in sections]
            return type("R", (), {"data": lambda s: rows})()
    read = bulk.read_document(Session(), "Codice Penale 2026", "c.p.")
    assert set(read) == {"61", "559"} and read["61"]["text"].endswith("1) i motivi abietti.")
    assert read["61"]["rubrica"] == "Rubrica (1)" and not bulk.has_body(read["559"])
