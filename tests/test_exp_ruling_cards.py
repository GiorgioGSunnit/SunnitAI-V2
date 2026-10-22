"""scripts/exp_ruling_cards.py's pure helpers, offline, on invented snippets. The
model call and the Neo4j reads need prod and are only exercised by running the
script there."""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp_ruling_cards", HERE.parent / "scripts" / "exp_ruling_cards.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

TEXT = ("LA CORTE SUPREMA DI CASSAZIONE SEZIONE QUINTA PENALE. Ritenuto in fatto: lo schiaffo provocava la "
        "caduta. Considerato in diritto: ai fini dell'omicidio preterintenzionale è sufficiente che l'evento "
        "morte sia conseguenza causale degli atti diretti a percuotere, senza che rilevi la sua "
        "prevedibilità in concreto. Gli artt. 581 e 584 c.p. P.Q.M. Rigetta il ricorso.")
ARTICLES = ["581 c.p.", "584 c.p."]


def test_long_ruling_is_cut_to_opening_and_end():
    text = "A" * rc.HEAD_CHARS + "B" * 50000 + "FINE"
    cut, in_part = rc.ruling_text_for_prompt(text)
    assert in_part and cut.startswith("A" * rc.HEAD_CHARS) and cut.endswith("FINE")
    assert len(cut) <= rc.MAX_CHARS + 10
    assert rc.ruling_text_for_prompt("breve") == ("breve", False)


def test_quote_in_text():
    assert rc.quote_in_text("«È sufficiente che l'evento morte sia conseguenza causale degli atti diretti a "
                            "percuotere»", TEXT)
    assert rc.quote_in_text("è sufficiente che l'evento morte sia conseguenza causale ... senza che rilevi la "
                            "sua prevedibilità in concreto", TEXT)
    assert not rc.quote_in_text("la morte deve essere prevedibile in concreto per l'agente", TEXT)
    assert not rc.quote_in_text("schiaffo", TEXT)   # too short to prove anything


def test_check_card_keeps_only_allowed_values():
    raw = {"articoli_applicati": ["art. 584 c.p.", "41 c.p.", "581 cod. pen.", "584 c.p."],
           "ricorrente": "Imputato", "esito": "Rigetto.", "a_favore_di": "parte civile",
           "argomento_decisivo": ["nesso causale", "nesso causale", ""],
           "parole_chiave_fatti": ["Schiaffo", "caduta"] + [f"f{i}" for i in range(10)],
           "parole_chiave_questioni": "not a list",
           "citazione": "è sufficiente che l'evento morte sia conseguenza causale degli atti diretti a percuotere"}
    card = rc.check_card(raw, TEXT, ARTICLES)
    assert card["articoli_applicati"] == ["584 c.p.", "581 c.p."]
    assert card["articoli_scartati"] == 1                        # 41 c.p. is not in the ruling's list
    assert card["ricorrente"] == "imputato" and card["esito"] == "rigetto"
    assert card["a_favore_di"] == "non chiaro"                   # not an allowed value
    assert card["argomento_decisivo"] == ["nesso causale"]
    assert card["parole_chiave_fatti"][:2] == ["schiaffo", "caduta"] and len(card["parole_chiave_fatti"]) == 8
    assert card["parole_chiave_questioni"] == []
    assert card["citazione_verificata"] and card["citazione"]


def test_check_card_drops_an_invented_quote_and_survives_garbage():
    card = rc.check_card({"citazione": "la morte deve essere prevedibile in concreto per l'agente"}, TEXT, ARTICLES)
    assert card["citazione"] == "" and not card["citazione_verificata"]
    card = rc.check_card(["not", "a", "dict"], TEXT, ARTICLES)
    assert card["articoli_applicati"] == [] and card["esito"] == "altro"


def test_sezione_and_reference():
    assert rc.sezione_of(TEXT) == "Sez. V"
    assert rc.sezione_of("SEZIONI UNITE PENALI") == "Sez. U."
    assert rc.sezione_of("Sezione 4 penale") == "Sez. IV"
    assert rc.sezione_of("nessuna") == ""
    link = {"court": "Cassazione", "number": 12345, "date": "12/03/2019"}
    assert rc.reference_of(link, "Sez. IV") == "Cass. pen., Sez. IV, n. 12345, 12/03/2019"
    assert rc.reference_of({"court": None, "number": None, "date": "2019"}, "") == "2019"


def _link(i, branch="penale", articles=("584 c.p.",), chars=1000, favours="?"):
    return {"id": f"d{i}", "branch": branch, "articles": list(articles), "chars": chars, "favours": favours,
            "court": "Cassazione", "number": i, "date": "2020"}


def test_pick_criminal_c_p_first_skipping_done():
    links = [_link(1, articles=("606 c.p.p.",)), _link(2), _link(3, branch="civile"), _link(4, chars=0), _link(5)]
    assert [r["id"] for r in rc.pick(links, "penale", {"d5"})] == ["d2", "d1"]


def test_latest_keeps_one_card_per_ruling(tmp_path):
    path = tmp_path / "cards.jsonl"
    rows = [{"id": "a", "errore": "json"}, {"id": "a", "esito": "rigetto"}, {"id": "b", "esito": "rigetto"},
            {"id": "b", "errore": "chiamata: Timeout"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    latest = {c["id"]: c for c in rc._latest(str(path))}
    assert latest["a"] == {"id": "a", "esito": "rigetto"}
    assert latest["b"] == {"id": "b", "esito": "rigetto"}       # a later failure does not undo a card


def test_grounded_keywords():
    assert rc.grounded(["schiaffo", "prevedibilità dell'evento", "nesso causale", "pistola"], TEXT) == 2
    assert rc.grounded(["", "di"], TEXT) == 0                   # no word of 4+ letters: not counted


def test_agreement_with_step_one():
    cards = [{"esito": "rigetto", "esito_regole": "rigettato"},
             {"esito": "annullamento senza rinvio", "esito_regole": "accolto"},
             {"esito": "inammissibilità", "esito_regole": "rigettato"},
             {"esito": "condanna", "esito_regole": "rigettato"},     # not comparable: skipped
             {"esito": "rigetto", "esito_regole": ""}]                # step 1 gave no answer: skipped
    assert rc.agreement(cards, "esito", "esito_regole", rc._OUTCOME_CLASS) == (2, 3)
    assert rc.rules_appellant("imputato+pm") == "più parti"
    assert rc.rules_appellant("pm") == "pubblico ministero"
    assert rc.rules_appellant("?") == ""


def test_card_report():
    base = {"secondi": 12.0, "letta_in_parte": False, "citazione_verificata": True, "articoli_applicati": ["584 c.p."],
            "articoli_scartati": 0, "riferimento_completo": True, "ricorrente": "imputato", "esito": "rigetto",
            "parole_chiave_fatti": ["schiaffo"], "parole_chiave_questioni": ["nesso causale"]}
    cards = [dict(base, id="a", a_favore_di="accusa", a_favore_di_regole="accusa"),
             dict(base, id="b", a_favore_di="imputato", a_favore_di_regole="accusa"),
             {"id": "c", "errore": "json"}]
    lines = rc.card_report(cards, remaining=100, workers=2)
    assert lines[0].startswith("Cards: 2  (failed: 1")
    assert any("same 1 / 2" in line for line in lines)
    assert any("Still to card: 100" in line and "0.2 h" in line for line in lines)
