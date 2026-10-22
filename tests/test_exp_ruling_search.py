"""scripts/exp_ruling_search.py's pure helpers, offline, on invented cards. The live
part (the pipeline's analysis and ruling search) needs prod and is only exercised
by running the script there."""

import importlib.util
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
_spec = importlib.util.spec_from_file_location("exp_ruling_search", HERE.parent / "scripts" / "exp_ruling_search.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

CASES = [
    {"id": "p1", "expected": [{"code": "cp", "article": "52"}, {"code": "cp", "article": "575"}]},
    {"id": "h5", "heldout": True, "expected": [{"code": "cp", "article": "56"}, {"code": "cpp", "article": "581"}]},
    {"id": "c1", "expected": [{"code": "cc", "article": "2043"}]},
    {"id": "mixed", "expected": [{"code": "cp", "article": "584"}, {"code": "cc", "article": "2043"}]},
    {"id": "p2", "expected": [{"code": "cp", "article": "609-bis"}]},
]


def test_criminal_cases_only_cp_cpp_and_heldout_on_request():
    assert [c["id"] for c in rs.criminal_cases(CASES, False)] == ["p1", "p2"]
    assert [c["id"] for c in rs.criminal_cases(CASES, True)] == ["p1", "h5", "p2"]
    assert [c["id"] for c in rs.criminal_cases(CASES, True, {"h5"})] == ["h5"]


def test_expected_keys():
    assert rs.expected_keys(CASES[1]) == {"56 c.p.", "581 c.p.p."}
    assert rs.expected_keys(CASES[4]) == {"609-bis c.p."}


def test_client_side():
    assert rs.client_side("imputato in primo grado") == "imputato"
    assert rs.client_side("indagato per tentato omicidio") == "imputato"
    assert rs.client_side("persona offesa, costituita parte civile") == "accusa"
    assert rs.client_side("non indicata") == ""
    assert rs.client_side("imputato e persona offesa") == ""      # both: unclear


def _card(i, articles, side="accusa", facts=(), issues=(), full=True):
    return {"id": f"d{i}", "articoli_applicati": list(articles), "a_favore_di": side,
            "parole_chiave_fatti": list(facts), "parole_chiave_questioni": list(issues),
            "argomento_decisivo": [], "riferimento_completo": full}


def test_central_weights_leave_out_the_method_candidates():
    picked = {"chosen": [("81", "c.p."), ("575", "c.p."), ("640", "c.p.")],
              "named": [("52", "c.p.")], "institutions_resolved": [("575", "c.p.")],
              "method_candidates": [("81", "c.p."), ("133", "c.p.")]}
    assert rs.central_weights(picked) == {"575 c.p.": 1.0, "640 c.p.": 0.5, "52 c.p.": 1.0}
    assert rs.ref_key(("609 bis", "c.p.")) == "609-bis c.p." and rs.ref_key("584 c.p.") == "584 c.p."


def test_rare_articles_and_keywords_weigh_more():
    cards = [_card(i, ["81 c.p."], issues=["motivazione"]) for i in range(8)] + \
            [_card(8, ["584 c.p.", "81 c.p."], issues=["motivazione"]), _card(9, ["640 c.p."])]
    articles, keywords = rs.card_rarities(cards)
    assert articles["584 c.p."] > articles["640 c.p."] - 1e-9 > articles["81 c.p."] > 0
    assert 0 < keywords["motivazione"] < 0.1


def test_rank_cards_by_central_rare_articles_keywords_and_side():
    generic = [_card(10 + i, ["81 c.p.", "133 c.p.", "575 c.p."], facts=["pena"]) for i in range(6)]
    cards = generic + [
        _card(1, ["575 c.p."]),
        _card(2, ["575 c.p.", "52 c.p."], side="imputato"),
        _card(3, ["575 c.p."], facts=["coltello", "lite in famiglia"], issues=["legittima difesa"]),
        _card(4, ["640 c.p."], side="imputato", facts=["coltello"]),          # not a central article: never
        _card(5, ["52 c.p."], full=False)]
    central = {"575 c.p.": 1.0, "52 c.p.": 1.0}
    terms = "Tizio colpisce con un coltello durante una lite in famiglia; invoca la legittima difesa"
    ranked = rs.rank_cards(cards, central, terms, "imputato", top=3)
    # 52 is rarer than 575 (on 2 cards vs 10): d2 (both + side) first, then d5 (52 only) and d3 (575 +
    # three keywords); the generic sentencing cards share only 575 among the central articles
    assert ranked[0] == "d2" and set(ranked[1:]) == {"d3", "d5"}
    assert "d4" not in rs.rank_cards(cards, central, terms, "imputato", top=20)
    assert rs.rank_cards(cards, {}, terms, "imputato") == []            # no central article: nothing


def test_available_on_topic():
    links = {"a": {"articles": ["584 c.p."]}, "b": {"articles": ["81 c.p."]}, "c": {"articles": ["584 c.p."]}}
    assert rs.available_on_topic({"a": {}, "b": {}}, links, {"584 c.p."}) == 1


def test_measure_and_reference():
    links = {"d1": {"id": "d1", "branch": "penale", "articles": ["575 c.p.", "81 c.p."], "court": "Cassazione",
                    "number": None, "date": "01/02/2026",
                    "italgiure": {"kind": "snpen", "section": "I", "year": "2026", "number": "190", "type": "S"}},
             "d2": {"id": "d2", "branch": "civile", "articles": ["2043 c.c."], "court": "Cassazione",
                    "number": "77", "date": "", "italgiure": None}}
    cards = {"d1": dict(_card(1, ["575 c.p."], side="imputato"), riferimento="Cass. pen., Sez. I, sent. n. 190/2026")}
    m = rs.measure(["d1", "d2", "special-doc"], links, cards, {"575 c.p."}, "imputato")
    assert m == {"rulings": 3, "criminal": 1, "on_topic": 1, "full_reference": 1, "client_side": 1,
                 "side_known": 1, "not_a_ruling": 1}
    assert rs.reference(links["d1"], cards["d1"]) == "Cass. pen., Sez. I, sent. n. 190/2026"
    assert rs.reference(links["d1"], None) == "Cass. pen., Sez. I, n. 190/2026"
    assert rs.reference(None, None).startswith("(not a ruling")
    lines = rs.show_lines("AFTER", ["d1"], links, cards, {"575 c.p."})
    assert "favours: imputato" in lines[1] and "applies: 575 c.p." in lines[1]


def test_report_lines_totals():
    before = {"rulings": 5, "criminal": 3, "on_topic": 1, "full_reference": 2, "client_side": 0,
              "side_known": 2, "not_a_ruling": 1}
    after = {"rulings": 5, "criminal": 5, "on_topic": 4, "full_reference": 5, "client_side": 3,
             "side_known": 5, "not_a_ruling": 0}
    lines = rs.report_lines([("p1", "imputato", before, after, 12)])
    assert any(line.startswith("p1") and line.endswith("(available 12)") for line in lines)
    assert any(line.startswith("BEFORE total rulings 5: criminal 3 (60%)") for line in lines)
    assert any(line.startswith("AFTER  total rulings 5: criminal 5 (100%), on topic 4 (80%)") for line in lines)
