"""scripts/exp_relations.py: the relations agent's prototype (27 Oct) -- article keys, references in an article's
text, co-occurrence relations, and the two tests."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp_relations", HERE.parent / "scripts" / "exp_relations.py")
rel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rel)


def test_keys():
    assert rel.key("1490") == "1490" and rel.key("1490.0.0") == "1490" and rel.key("1117.3.0_2") == "1117"
    assert rel.key("337 ter") == "337-ter" and rel.key("337-TER") == "337-ter" and rel.key("2051") == "2051"
    assert rel.key("corte_cassazione.8") is None and rel.key("") is None and rel.key("0") is None


def test_card_articles_keeps_the_code_asked():
    card = {"articoli_applicati": ["1490 c.c.", "1495 c.c.", "360 c.p.c.", "337 ter c.c.", "x"]}
    assert rel.card_articles(card) == {"1490", "1495", "337-ter"}
    assert rel.card_articles(card, "c.p.c.") == {"360"}


def test_text_refs():
    text = ("Nei casi indicati dall'articolo 1490 il compratore può domandare la risoluzione. Si applicano gli "
            "articoli 1491 e 1492, ma non l'articolo 360 del codice di procedura civile né l'art. 2 della legge "
            "n. 392. Restano salvi gli artt. da 1494 a 1496 e l'articolo 2 a 5 anni.")
    assert rel.text_refs(text, "1493") == {"1490", "1491", "1492", "1494", "1495", "1496", "2"}
    assert rel.text_refs("Vedi l'articolo 1490.", "1490") == set()             # never itself
    assert rel.text_refs("ai sensi dell'articolo 117-bis", "1") == {"117-bis"}


def test_co_relation_rankings():
    groups = [{"1490", "1495"}, {"1490", "1495", "1492"}, {"1490", "1492"}, {"1490", "2697"}, {"1218", "1176"},
              {"1218", "1176"}, {"1495"}]
    co = rel.CoRelation(groups, exclude={"2697"})
    assert co.n == 7 and co.single["1490"] == 4 and "2697" not in co.single
    assert [a for a, _ in co.neighbours("1490", "count")] == ["1492", "1495"]      # 2 each: by number
    assert [a for a, _ in co.neighbours("1490", "cosine")] == ["1492", "1495"]     # 1492 rarer: higher cosine
    assert co.neighbours("1490", "count", min_support=3) == []
    assert [a for a, _ in co.neighbours("1218", "lift")] == ["1176"]


def test_refers_to_both_directions():
    r = rel.RefersTo({"1492": {"1490"}, "1495": {"1490", "9999"}, "1490": set()}, {"1490", "1492", "1495"})
    assert r.edges() == 3 - 1                                                       # 9999 is not an article
    assert [a for a, _ in r.neighbours("1492")] == ["1490"]
    assert [a for a, _ in r.neighbours("1490")] == ["1492", "1495"]                 # named by both: nearer first


def test_connection_test_leave_one_out():
    cases = {"c": ["1490", "1495", "1492"]}
    perfect = lambda a: [(b, 1.0) for b in ["1490", "1495", "1492"] if b != a]       # noqa: E731
    nothing = lambda a: []                                                         # noqa: E731
    report = rel.connection_test(cases, {"perfect": perfect, "nothing": nothing})
    assert report["perfect"]["hit@5"] == 100.0 and report["perfect"]["seeds_with_a_hit@10"] == 100.0
    assert report["nothing"]["hit@10"] == 0.0 and report["nothing"]["seeds_with_neighbours"] == 0.0
    assert rel.connection_by_case(cases, perfect) == {"c": "6/6"}
    assert [a for a, _ in rel.near("1490", ["1488", "1490", "1491", "1500"])][:2] == ["1491", "1488"]


def test_profiles_and_retrieval_test():
    profiles = rel.Profiles({"1490": Counter(rel.stems("vizi della cosa venduta garanzia compratore")),
                             "2051": Counter(rel.stems("custodia della cosa danno cagionato caso fortuito"))})
    ranked = profiles.rank("La barca venduta presenta vizi: il compratore chiede la garanzia")
    assert ranked[0][0] == "1490"
    report = rel.retrieval_test({"c2": "vizi della barca venduta"}, {"c2": ["1490", "1495"]}, {"p": profiles.rank})
    assert report["p"]["recall@10"] == 50.0 and report["p"]["ranks"]["c2"] == [1, "-"]
    assert [a for a, _ in rel.fuse([[("1490", 1), ("1495", 1)], [("1495", 1)]])] == ["1495", "1490"]


def test_read_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"c2": {"id": "c2", "question": "barca", "expected": [
        {"code": "consumo", "any_of": ["128"]}, {"code": "cc", "article": "1490"}, {"code": "cc", "article": "1495"}]},
        "cc_short": {"id": "cc_short", "question": "nullità", "expected": [{"code": "cc", "article": "1418"}]}}),
        encoding="utf-8")
    cases, questions = rel.read_cases(str(path))
    assert cases == {"c2": ["1490", "1495"]} and questions == {"c2": "barca", "cc_short": "nullità"}
    assert rel._single_expected(str(path)) == {"cc_short": "1418"}


class _Session:
    """Fake read session: the Codice Civile's sections and two rulings' CITES."""

    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)
        if "d.name STARTS WITH $p" in query:
            data = [{"id": "cc", "name": "Codice Civile 2026"}, {"id": "cc-old", "name": "Codice Civile 2026 (vecchio)"}]
        elif "CONTAINS]->(s:Section) RETURN s.name" in query:
            data = [{"name": "1490.0.0", "text": "Il venditore è tenuto a garantire.", "heading": "Garanzia per i vizi"},
                    {"name": "128.0.0", "text": "In vigore dal 10 dicembre 2021", "heading": "Ambito"},
                    {"name": "128", "text": "In vigore dal 10 dicembre 2021 1. Il presente capo disciplina la vendita.",
                     "heading": ""},
                    {"name": "105.0.0", "text": "In vigore dal 23 ottobre 2005", "heading": ""},
                    {"name": "105.2.0", "text": "Secondo comma.", "heading": ""},
                    {"name": "105.1.0", "text": "Primo comma.", "heading": ""},
                    {"name": "corte_cassazione.1", "text": "x", "heading": ""}]
        elif "RETURN d.id AS id\"" in query or query.endswith("RETURN d.id AS id"):
            data = [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}]
        else:
            data = [{"id": i, "arts": ["1490", "1495"] if i == "r1" else ["2051"]} for i in params["ids"] if i != "r3"]
        return type("Result", (), {"data": lambda self_: data})()


def test_code_articles_reads_the_text_as_the_app_does():
    texts, headings = rel.code_articles(_Session())
    assert texts["1490"] == "Il venditore è tenuto a garantire." and headings["1490"] == "Garanzia per i vizi"
    assert texts["128"] == "1. Il presente capo disciplina la vendita."
    assert texts["105"] == "Primo comma.\nSecondo comma."
    assert "corte_cassazione" not in " ".join(texts)


def test_cites_groups_in_batches():
    assert rel.cites_groups(_Session(), batch=2, pause=0) == {"r1": {"1490", "1495"}, "r2": {"2051"}}
