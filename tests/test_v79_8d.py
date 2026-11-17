"""v79.8 block D -- commi are fattispecie, on a synthetic article only:

D1. a comma stating a special case, a presumption or an aggravation is its own record, decomposed on its own and
    labelled with its comma; it never adds elements to the base rule's record, nor its open elements to the base
    rule's missing facts
D2. a base rule and two special commi -> three records
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


COMMI = [(1, "Il custode risponde del danno cagionato dalla cosa che ha in custodia."),
         (2, "Si presume la colpa del custode quando la cosa è pericolosa per sua natura."),
         (3, "Nei casi previsti dal primo comma il danno è risarcito anche al terzo trasportato.")]
REQS = {1: ["il custode ha la cosa in custodia", "il danno è cagionato dalla cosa"],
        2: ["la cosa è pericolosa per sua natura"],
        3: ["il danneggiato è un terzo trasportato"]}


def test_a_base_rule_and_two_special_commi_are_three_records_and_the_base_lists_only_its_own(monkeypatch):
    from src.rag import interactive_flow as flow, issue_records, lawyer_method as M, rule_quote
    rule = {"how": "civil", "commi": COMMI, "selected": [1, 2, 3], "requisites": REQS}
    monkeypatch.setattr(rule_quote, "article_rules", lambda *a, **k: [rule])
    monkeypatch.setattr(flow, "facts_status", lambda text, analysis: "da provare")
    reqs = issue_records._requisites({"_article": ("2099", "c.c."), "s": {"plain_text": "x"}}, {})
    assert [r.get("special", "") for r in reqs] == ["", "", "presunzione", "caso speciale"]
    for i, r in enumerate(reqs, 1):
        r["id"] = f"R{i}"
    reqs[0]["status"] = "provato"
    for r in (reqs[0], reqs[2], reqs[3]):       # a fact touches them (a fact-free record is v78's noise)
        r["supporting"] = ["S1"]
    recs = {"2099 c.c.": {"rubrica": "Danno da cosa in custodia", "side": "weapon", "requisites": reqs}}
    log = {}
    view = M.method_records(recs, log)
    assert list(view) == ["2099 c.c.", "2099 c.c. co. 2", "2099 c.c. co. 3"]
    assert [r["id"] for r in view["2099 c.c."]["requisites"]] == ["R1", "R2"]
    assert [r["id"] for r in view["2099 c.c. co. 2"]["requisites"]] == ["R3"]
    assert view["2099 c.c. co. 2"]["comma_of"] == "2099 c.c." and view["2099 c.c. co. 3"]["comma"] == 3
    assert "presunzione" in view["2099 c.c. co. 2"]["rubrica"]
    assert log["special_commi"] == ["2099 c.c. co. 2", "2099 c.c. co. 3"]
    missing = M.facts_split([], view)["mancanti"]
    base = [m["elemento"] for m in missing if m["article"] == "2099 c.c."]
    assert base == ["R2"] and {m["article"] for m in missing if m["elemento"] in ("R3", "R4")} == \
        {"2099 c.c. co. 2", "2099 c.c. co. 3"}


@pytest.mark.parametrize("text, kind", [
    ("Nei casi previsti dal primo comma la pena è aumentata.", "aggravamento"),
    ("Si considera cosa mobile anche l'energia elettrica.", "equiparazione"),
    ("Si presume la colpa del conducente fino a prova contraria.", "presunzione"),
    ("Il proprietario risponde dei danni cagionati dall'animale.", ""),
])
def test_the_kind_of_a_special_comma(text, kind):
    from src.rag import issue_records
    assert issue_records.special_comma(text) == kind


def test_the_first_comma_is_never_special_and_a_record_without_one_is_unchanged():
    from src.rag import issue_records
    rec = {"requisites": [{"id": "R1", "comma": 1, "text": "x"}]}
    assert issue_records.split_special("2051 c.c.", rec) == {"2051 c.c.": rec}
