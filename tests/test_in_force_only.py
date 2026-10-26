"""Articles whose "N.0.0" Section holds only their date of entry into force get the article's text
(defensive_generation._fill_in_force_only; the boat case, 26 Oct)."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

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
    ],
}


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append(params)
        rows = []
        for item in params["items"]:
            for name, text in SECTIONS.get(item["doc"], []):
                if name == item["num"] or (name.startswith(item["num"] + ".") and name.endswith(".0")):
                    rows.append({"doc": item["doc"], "num": item["num"], "name": name, "text": text})
        return type("Result", (), {"data": lambda self_: rows})()


def _row(name, text, doc="consumo"):
    return {"d": {"id": doc, "name": "Codice del Consumo 2026"}, "s": {"name": name, "plain_text": text}}


def test_in_force_only():
    assert dg._in_force_only("In vigore dal 10 dicembre 2021")
    assert dg._in_force_only("  in vigore dal 3 settembre 2015. ")
    assert not dg._in_force_only("In vigore dal 10 dicembre 2021 1. Il presente capo disciplina")
    assert not dg._in_force_only("Il venditore è tenuto a garantire")
    assert not dg._in_force_only("")


def test_the_bare_section_fills_the_date_without_its_date():
    rows = [_row("135-octiesdecies.0.0", "In vigore dal 11 dicembre 2021")]
    filled = dg._fill_in_force_only(FakeSession(), rows)
    assert rows[0]["s"]["plain_text"].startswith("1. In caso di difetto di conformità del bene")
    assert rows[0]["s"]["name"] == "135-octiesdecies.0.0"
    assert filled == ["Codice del Consumo 2026 135-octiesdecies.0.0"]


def test_without_a_bare_section_the_commi_in_order():
    rows = [_row("105.0.0", "In vigore dal 23 ottobre 2005")]
    dg._fill_in_force_only(FakeSession(), rows)
    assert rows[0]["s"]["plain_text"] == ("In mancanza di specifiche disposizioni comunitarie, un prodotto è sicuro se.\n"
                                          "Si presume che un prodotto sia sicuro quando è conforme alle norme.")


def test_rows_with_text_and_articles_with_nothing_else_are_left_as_they_are():
    session = FakeSession()
    rows = [_row("1490.0.0", "Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi"),
            _row("9.0.0", "In vigore dal 1 gennaio 2006")]
    assert dg._fill_in_force_only(session, rows) == []
    assert rows[1]["s"]["plain_text"] == "In vigore dal 1 gennaio 2006"
    assert session.calls[0]["items"] == [{"doc": "consumo", "num": "9"}]          # one query, the dated row only
    assert dg._fill_in_force_only(FakeSession(), [_row("1490.0.0", "Il venditore è tenuto")]) == []
