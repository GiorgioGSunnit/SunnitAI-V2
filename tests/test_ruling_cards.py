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
    assert [c["id"] for c, _ in best] == ["1", "2", "3"]                  # applied + keywords + side first
    scores = [s for _, s in best]
    assert scores[0] > scores[1] > scores[2] > 0                          # cited only: the weakest
    assert rc.rank(cards, {"999 c.p.": 1.0}, case, "imputato", "") == []


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
    rows, carded = rc.card_rulings(_Session({"1", "3"}), None, "neo4j", "Uno schiaffo.", analysis, {})
    assert [r["d"]["id"] for r in rows] == ["1", "3"] and set(carded) == {"1", "3"}
    assert logged[-1][1]["missing_in_db"] == ["2"] and logged[-1][1]["chosen"][0]["a_favore_di"] == "imputato"
    reading = rc.lead_with(rows, [{"d": {"id": "9"}}, {"d": {"id": "1"}}])
    assert [r["d"]["id"] for r in reading] == ["1", "3", "9"]
    annotated = rc.annotate([{**rows[0], "_passage": "Il passo."}, {"d": {"id": "9"}, "_passage": "Altro."}], carded)
    assert annotated[0]["_passage"] == ("Esito (dalla scheda): annullamento senza rinvio; ricorso di: imputato; "
                                        "favorevole a: imputato. Principio decisivo: «la morte non era "
                                        "prevedibile». Passo: Il passo.")
    assert annotated[1]["_passage"] == "Altro."


def test_a_failure_leaves_the_rulings_alone(cards_file, monkeypatch):
    def broken(*a):
        raise RuntimeError("database down")
    monkeypatch.setattr(rc, "_resolved_institutions", broken)
    assert rc.card_rulings(None, None, "neo4j", "caso", {}, {}) == ([], {})
