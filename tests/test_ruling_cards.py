"""RULING_CARDS_ENABLED: rulings chosen from the precomputed ruling cards (src/rag/ruling_cards.py).
Synthetic cards only."""

import json

import pytest

from src.rag import ruling_cards as rc


def _card(cid, applied, cited=(), keywords=(), side="", quote="", branch="penale", favoured="", **extra):
    return {"id": cid, "articoli_applicati": list(applied), "articoli_citati": list(cited or applied),
            "parole_chiave_fatti": list(keywords), "a_favore_di": side, "parte_favorita": favoured,
            "esito": "annullamento senza rinvio", "ricorrente": "imputato", "citazione": quote,
            "citazione_verificata": bool(quote), "riferimento": f"Cass. pen., Sez. V, sent. n. {cid}/2026",
            "riferimento_completo": True, "branch": branch, **extra}


CARDS = ([_card("1", ["584 c.p."], ["584 c.p.", "43 c.p."], ["schiaffo al volto", "caduta mortale"], "imputato",
                "la morte non era prevedibile"),
          _card("2", ["584 c.p."], keywords=["rissa tra vicini"], side="accusa"),
          _card("3", ["81 c.p."], cited=["81 c.p.", "584 c.p."], side="imputato"),
          {"id": "4", "errore": "json"},
          _card("5", ["2051 c.c."], branch="civile", favoured="danneggiato"),
          _card("1", ["640 c.p."])]                                         # a repeated id: the first one counts
         + [_card(str(100 + i), ["81 c.p."]) for i in range(10)])


@pytest.fixture
def cards_file(tmp_path, monkeypatch):
    path = tmp_path / "cards.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CARDS) + "\nnot json\n", encoding="utf-8")
    monkeypatch.setenv(rc.FILES_ENV, f"{path}, {tmp_path / 'missing.jsonl'}")
    monkeypatch.setattr(rc, "_store", None)
    monkeypatch.setattr(rc, "_signature", None)
    return path


def test_cards_are_loaded_once_and_again_when_a_file_grows(cards_file):
    cards = rc.store()
    assert len(cards.cards) == 14 and rc.store() is cards                  # the error, repeat and bad line skipped
    assert sorted(cards.cards[i]["id"] for i in cards.by_article["584 c.p."]) == ["1", "2", "3"]
    assert cards.article_rarity["584 c.p."] > cards.article_rarity["81 c.p."]
    with open(cards_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_card("6", ["609-bis c.p."])) + "\n")
    assert len(rc.store().cards) == 15


def test_no_file_means_no_cards(monkeypatch):
    monkeypatch.setenv(rc.FILES_ENV, "")
    assert rc.store() is None
    assert rc.card_rulings(None, None, "neo4j", "caso", {}, {}) == ([], {})


def test_central_articles_leave_out_the_method_candidates():
    picked = {"chosen": [("584", "c.p."), ("81", "c.p."), ("612 bis", "c.p.")], "named": [("43", "c.p.")],
              "institutions_resolved": [("584", "c.p.")], "method_candidates": [("81", "c.p.")]}
    assert rc.central_weights(picked, [("41", "c.p.")]) == {"584 c.p.": 1.0, "612-bis c.p.": 0.5,
                                                           "43 c.p.": 1.0, "41 c.p.": 1.0}


def test_rank_by_central_articles_keywords_and_the_clients_side(cards_file):
    cards = rc.store()
    central = {"584 c.p.": 1.0}
    case = "Sempronio dà uno schiaffo al volto a Caio, che cade: caduta mortale."
    best = rc.rank(cards, central, case, side="imputato", party="Sempronio, imputato")
    assert [c["id"] for c, _ in best] == ["1", "2"]          # applied + keywords + side first; 3 only cites 584
    assert best[0][1] > best[1][1] > 0
    assert rc.rank(cards, {"999 c.p.": 1.0}, case, "imputato", "") == []


def test_a_card_needs_a_specific_central_anchor(cards_file, monkeypatch):
    cards = rc.store()
    assert rc.anchors(cards.cards[0], {"584 c.p.": 1.0}, "", cards) == ["584 c.p."]      # named or resolved
    assert rc.anchors(cards.cards[0], {"584 c.p.": 0.5}, "nulla in comune", cards) == []  # pool only, no keyword
    assert rc.anchors(cards.cards[0], {"584 c.p.": 0.5}, "uno schiaffo al volto", cards) == ["584 c.p."]
    monkeypatch.setattr(cards, "generic", {"584 c.p."})                                   # too common to anchor
    assert rc.anchors(cards.cards[0], {"584 c.p.": 1.0}, "schiaffo al volto", cards) == []
    assert rc.rank(cards, {"584 c.p.": 1.0}, "schiaffo al volto", "imputato", "") == []


def test_generic_articles_are_the_very_common_ones():
    many = [rc._trim(_card(str(i), ["606 c.p.p."] + (["584 c.p."] if i < 60 else []))) for i in range(3000)]
    store = rc.CardStore(many)
    assert store.generic == {"606 c.p.p."}               # 3000 of 3000; 584 in 60 = 2%, not above max(60, 50)


def test_side_of_a_criminal_and_of_a_civil_card(cards_file):
    cards = {c["id"]: c for c in rc.store().cards}
    assert rc.client_side("imputato") == "imputato" and rc.client_side("persona offesa") == "accusa"
    assert rc.client_side("") == "" and rc.client_side("imputato e persona offesa") == ""
    assert rc.favours_client(cards["1"], "imputato", "") and not rc.favours_client(cards["2"], "imputato", "")
    assert rc.favours_client(cards["5"], "", "i genitori di Tizio, danneggiati")
    assert not rc.favours_client(cards["5"], "", "l'istituto scolastico")


class _Session:
    def __init__(self, ids):
        self.ids, self.queries = ids, []

    def run(self, query, **params):
        self.queries.append(params)
        rows = [{"d": {"id": i, "name": f"ruling {i}"}} for i in params["ids"] if i in self.ids]
        return type("Result", (), {"data": lambda _self: rows})()


def test_card_rulings_lead_the_reading_with_their_summary(cards_file, monkeypatch):
    logged = []
    monkeypatch.setattr(rc, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    monkeypatch.setattr(rc, "_resolved_institutions", lambda *a: [("584", "c.p.")])
    analysis = {"posizione": "imputato", "questioni": [{"titolo": "Omicidio preterintenzionale",
                                                         "parole_chiave": ["schiaffo al volto"]}]}
    rows, carded = rc.card_rulings(_Session({"1", "2"}), None, "neo4j", "Uno schiaffo.", analysis, {})
    # 1 favours the client (the defendant); 2 the prosecution: the adverse slot; 3 favours the client but
    # only cites the article, under half the best score: not on topic enough for a favourable slot.
    assert [r["d"]["id"] for r in rows] == ["1", "2"] and set(carded) == {"1", "2"}
    log = logged[-1][1]
    assert [c["role"] for c in log["chosen"]] == ["favorevole", "sfavorevole"] and log["client_side"] == "imputato"
    assert log["missing_in_db"] == []
    reading = rc.lead_with(rows, [{"d": {"id": "9"}}, {"d": {"id": "1"}}])
    assert [r["d"]["id"] for r in reading] == ["1", "2", "9"]
    annotated = rc.annotate([{**rows[0], "_passage": "Il passo."}, {**rows[1], "_passage": "Altro passo."},
                             {"d": {"id": "9"}, "_passage": "Altro."}], carded)
    assert annotated[0]["_passage"] == ("Orientamento favorevole alla parte assistita. Esito (dalla scheda): "
                                        "annullamento senza rinvio; ricorso di: imputato; favorevole a: imputato. "
                                        "Principio decisivo: «la morte non era prevedibile». Passo: Il passo.")
    assert annotated[1]["_passage"].startswith("ORIENTAMENTO SFAVOREVOLE alla parte assistita: è la "
                                               "giurisprudenza della controparte")
    assert annotated[2]["_passage"] == "Altro."


def _ranked(*items):
    """[(trimmed card, score)] from (id, side, score, branch)."""
    return [(rc._trim(_card(cid, ["584 c.p."], side=side, branch=branch)), score) for cid, side, score, branch in items]


def test_choose_by_side_favourable_first_then_one_adverse():
    ranked = _ranked(("a", "imputato", 10, "penale"), ("b", "accusa", 8, "penale"), ("c", "non chiaro", 7, "penale"),
                     ("d", "imputato", 6, "penale"), ("e", "imputato", 4, "penale"))
    chosen = rc.choose(ranked, "imputato", "")
    assert [(c["id"], role) for c, _, role in chosen] == [("a", "favorevole"), ("d", "favorevole"), ("b", "sfavorevole")]
    # no favourable ruling on topic: one adverse, the rest only unclear ones, never more adverse rulings
    ranked = _ranked(("b", "accusa", 9, "penale"), ("c", "non chiaro", 7, "penale"), ("f", "accusa", 6, "penale"),
                     ("e", "imputato", 3, "penale"))
    assert [(c["id"], role) for c, _, role in rc.choose(ranked, "imputato", "")] == [("b", "sfavorevole"), ("c", "neutro")]
    # only favourable rulings when the adverse slot is 0
    ranked = _ranked(("a", "imputato", 10, "penale"), ("b", "accusa", 9, "penale"))
    assert [c["id"] for c, _, _ in rc.choose(ranked, "imputato", "", unfavourable=0)] == ["a"]
    # the client's side unknown: the best three, all unclear
    ranked = _ranked(("a", "imputato", 10, "penale"), ("b", "accusa", 9, "penale"), ("c", "accusa", 8, "penale"),
                     ("d", "imputato", 7, "penale"))
    assert [(c["id"], role) for c, _, role in rc.choose(ranked, "", "")] == [("a", "neutro"), ("b", "neutro"), ("c", "neutro")]
    assert rc.choose([], "imputato", "") == []


def test_execution_stage_rulings_only_for_a_case_at_that_stage(tmp_path, monkeypatch):
    # Round 6: supervision-court and suspension-of-execution rulings cited, as favourable, in a stalking defence.
    path = tmp_path / "cards.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in [
        _card("trial", ["572 c.p."], side="imputato"),
        _card("execution", ["572 c.p.", "656 c.p.p."], keywords=["maltrattamenti"], side="imputato")]), encoding="utf-8")
    monkeypatch.setenv(rc.FILES_ENV, str(path))
    monkeypatch.setattr(rc, "_store", None)
    monkeypatch.setattr(rc, "_signature", None)
    cards = rc.store()
    assert [c["id"] for c, _ in rc.rank(cards, {"572 c.p.": 1.0}, "maltrattamenti", "imputato", "")] == ["trial"]
    assert {c["id"] for c, _ in rc.rank(cards, {"572 c.p.": 1.0}, "maltrattamenti", "imputato", "",
                                         execution=True)} == {"trial", "execution"}
    logged = []
    monkeypatch.setattr(rc, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    monkeypatch.setattr(rc, "_resolved_institutions", lambda *a: [("572", "c.p.")])
    analysis = {"posizione": "condannato", "frame": {"posture_type": "", "procedural_posture":
                                                     "Istanza al tribunale di sorveglianza dopo la condanna"}}
    rows, _ = rc.card_rulings(_Session({"trial", "execution"}), None, "neo4j", "caso", analysis, {})
    assert logged[-1][1]["execution_stage_allowed"] is True and {r["d"]["id"] for r in rows} == {"trial", "execution"}


def test_a_civil_ruling_for_another_party_is_not_adverse():
    card = rc._trim(_card("x", ["2051 c.c."], branch="civile", favoured="agenzia delle entrate"))
    assert rc.role_of(card, "", "i genitori di Tizio, danneggiati") == "neutro"
    card = rc._trim(_card("y", ["2051 c.c."], branch="civile", favoured="danneggiato"))
    assert rc.role_of(card, "", "i genitori di Tizio, danneggiati") == "favorevole"


def test_a_failure_leaves_the_rulings_alone(cards_file, monkeypatch):
    def broken(*a):
        raise RuntimeError("database down")
    monkeypatch.setattr(rc, "_resolved_institutions", broken)
    assert rc.card_rulings(None, None, "neo4j", "caso", {}, {}) == ([], {})
