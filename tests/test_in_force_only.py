"""One article text for every path (src/rag/article_text.py): "N.0.0" only if it holds more than its date
in force, else the bare "N" without its date, else the commi "N.k.0" in order (the boat case, 26 Oct)."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")

from src.rag import article_text  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import issue_spotting  # noqa: E402

SECTIONS = {
    "consumo": [
        ("135-octiesdecies", "In vigore dal 11 dicembre 2021 1. In caso di difetto di conformità del bene, il "
                             "consumatore ha diritto al ripristino della conformità o a ricevere una riduzione."),
        ("135-octiesdecies.0.0", "In vigore dal 11 dicembre 2021"),
        ("135-octiesdecies.1.0", "In caso di difetto di conformità del bene, il consumatore ha diritto."),
        ("105.0.0", "In vigore dal 23 ottobre 2005"),
        ("105.2.0", "Si presume che un prodotto sia sicuro quando è conforme alle norme."),
        ("105.1.0", "In mancanza di specifiche disposizioni comunitarie, un prodotto è sicuro se."),
        ("105.1", "un frammento di lettera"),
        ("9.0.0", "In vigore dal 1 gennaio 2006"),
        ("12.1", "primo frammento"),
        ("12.2", "secondo frammento"),
        ("1490.0.0", "Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi."),
        ("1490.1.0", "Il patto con cui si esclude o si limita la garanzia non ha effetto."),
        ("14", "In vigore dal 2 maggio 2020 Il bare text."),
        ("140", "un altro articolo"),
    ],
}


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        assert "UNWIND $items AS it" in query
        self.calls.append(params)
        rows = []
        for item in params["items"]:
            for name, text in SECTIONS.get(item["doc"], []):
                if name == item["num"] or name.startswith(item["num"] + "."):
                    rows.append({"doc": item["doc"], "num": item["num"],
                                 "d": {"id": item["doc"], "name": "Codice del Consumo 2026"},
                                 "s": {"name": name, "plain_text": text}})
        return type("Result", (), {"data": lambda self_: rows})()


def _row(name, text, doc="consumo"):
    return {"d": {"id": doc, "name": "Codice del Consumo 2026"}, "s": {"name": name, "plain_text": text}}


def test_in_force_only():
    assert article_text.in_force_only("In vigore dal 10 dicembre 2021")
    assert article_text.in_force_only("  in vigore dal 3 settembre 2015. ")
    assert not article_text.in_force_only("In vigore dal 10 dicembre 2021 1. Il presente capo disciplina")
    assert not article_text.in_force_only("Il venditore è tenuto a garantire")
    assert not article_text.in_force_only("")


def test_choose_text_follows_the_order_zero_zero_bare_commi():
    assert article_text.choose_text("7", {"7.0.0": "Testo vero.", "7": "Altro"}) == "Testo vero."
    assert article_text.choose_text("7", {"7.0.0": "In vigore dal 1 gennaio 2020", "7": "In vigore dal 1 "
                                          "gennaio 2020 Testo nudo."}) == "Testo nudo."
    assert article_text.choose_text("7", {"7.0.0": "In vigore dal 1 gennaio 2020", "7.2.0": "b", "7.10.0": "c",
                                          "7.1.0": "a", "7.1": "frammento", "70.1.0": "altro"}) == "a\nb\nc"
    assert article_text.choose_text("7", {"7.0.0": "In vigore dal 1 gennaio 2020"}) == ""


def test_fetch_article_returns_the_canonical_row_with_the_bare_text_without_its_date():
    rows = article_text.fetch_article(FakeSession(), "consumo", "135-octiesdecies")
    assert len(rows) == 1
    assert rows[0]["s"]["name"] == "135-octiesdecies.0.0"
    assert rows[0]["s"]["plain_text"].startswith("1. In caso di difetto di conformità del bene")


def test_fetch_article_without_a_bare_section_joins_the_commi_in_order():
    rows = article_text.fetch_article(FakeSession(), "consumo", "105")
    assert rows[0]["s"]["plain_text"] == ("In mancanza di specifiche disposizioni comunitarie, un prodotto è sicuro "
                                          "se.\nSi presume che un prodotto sia sicuro quando è conforme alle norme.")


def test_fetch_article_keeps_real_zero_zero_text_and_never_another_article():
    rows = article_text.fetch_article(FakeSession(), "consumo", "1490")
    assert rows[0]["s"]["plain_text"] == "Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi."
    rows = article_text.fetch_article(FakeSession(), "consumo", "14")
    assert [r["s"]["name"] for r in rows] == ["14"] and rows[0]["s"]["plain_text"] == "Il bare text."


def test_fetch_article_without_a_canonical_section_returns_its_fragments():
    rows = article_text.fetch_article(FakeSession(), "consumo", "12")
    assert [r["s"]["name"] for r in rows] == ["12.1", "12.2"]
    assert article_text.fetch_article(FakeSession(), "consumo", "99") == []


def test_issue_spotting_and_the_strategy_lookups_share_it():
    assert issue_spotting.fetch_article(FakeSession(), "consumo", "105")[0]["s"]["plain_text"].startswith(
        "In mancanza")
    assert dg._section_quality is article_text.section_quality


def test_fill_texts_gives_a_date_only_row_the_bare_section_without_its_date():
    rows = [_row("135-octiesdecies.0.0", "In vigore dal 11 dicembre 2021")]
    filled = article_text.fill_texts(FakeSession(), rows)
    assert rows[0]["s"]["plain_text"].startswith("1. In caso di difetto di conformità del bene")
    assert rows[0]["s"]["name"] == "135-octiesdecies.0.0"
    assert filled == ["Codice del Consumo 2026 135-octiesdecies.0.0"]


def test_fill_texts_strips_a_leading_date_without_a_query():
    session = FakeSession()
    rows = [_row("14", "In vigore dal 2 maggio 2020 Il bare text.")]
    assert article_text.fill_texts(session, rows) == ["Codice del Consumo 2026 14"]
    assert rows[0]["s"]["plain_text"] == "Il bare text." and session.calls == []


def test_fill_texts_leaves_real_text_and_articles_with_nothing_else():
    session = FakeSession()
    rows = [_row("1490.0.0", "Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi"),
            _row("9.0.0", "In vigore dal 1 gennaio 2006")]
    assert article_text.fill_texts(session, rows) == []
    assert rows[1]["s"]["plain_text"] == "In vigore dal 1 gennaio 2006"
    assert session.calls[0]["items"] == [{"doc": "consumo", "num": "9"}]          # one query, the dated row only


def test_fill_from_rows_uses_the_rows_already_read():
    rows = [_row("105.0.0", "In vigore dal 23 ottobre 2005"), _row("105.1.0", "Primo comma."),
            _row("105.2.0", "Secondo comma.")]
    assert article_text.fill_from_rows("105", rows) == ["Codice del Consumo 2026 105.0.0"]
    assert rows[0]["s"]["plain_text"] == "Primo comma.\nSecondo comma."
    assert rows[1]["s"]["plain_text"] == "Primo comma."


def test_fetch_articles_by_code_prefix_takes_the_shared_rule():
    class PrefixSession:
        def run(self, query, **params):
            assert "d.name STARTS WITH $prefix" in query and "s.name = $n OR" in query
            data = [{"d": {"id": "cc", "name": "Codice Civile"}, "s": {"name": n, "plain_text": t}}
                    for n, t in [("1495.0.0", "In vigore dal 1 gennaio 2022"), ("1495.1.0", "Il compratore decade."),
                                 ("1495.2.0", "La denunzia non è necessaria.")]
                    if n == params["n"] or n.startswith(params["n"] + ".")]
            return type("Result", (), {"data": lambda self_: data})()

    rows = dg._fetch_articles(PrefixSession(), [("1495", "c.c.")])
    assert len(rows) == 1 and rows[0]["_article"] == ("1495", "c.c.")
    assert rows[0]["s"]["plain_text"] == "Il compratore decade.\nLa denunzia non è necessaria."
