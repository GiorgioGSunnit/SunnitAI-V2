"""scripts/article_sections.py: no database."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("article_sections", ROOT / "scripts" / "article_sections.py")
sections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sections)


def test_names_are_sorted_shaped_and_marked_as_today_s_rule_treats_them():
    names = ["640.10.0", "640_2", "640.2.0", "640.2_3", "640.0.0", "640"]
    assert sorted(names, key=sections.sort_key) == ["640", "640.0.0", "640_2", "640.2.0", "640.2_3", "640.10.0"]
    assert [sections.shape(n) for n in ("640", "640.0.0", "640.2.0", "640.2_3", "640_2", "337-ter.1.2")] == \
        ["N", "N.0.0", "N.k.0", "N.k_j", "N_j", "N.k.k"]
    assert sections.article_of("337-ter.1.0") == "337-ter" and sections.article_of("tutti.3") == ""
    assert [sections.status(n, "640") for n in ("640", "640.0.0", "640.2.0", "640.2_3", "640.0.1", "640_2")] == \
        ["", "", "", "UNUSED", "UNUSED", "HIDDEN"]


def test_a_note_is_an_editorial_one_never_a_recital_or_a_list_item():
    assert sections.note_at("Testo. (1) Articolo aggiunto dall'art. 9, D.Lgs. 6/2003.") == 7
    assert sections.note_at("salvo il caso (3).\n(3) Comma così modificato dall'art. 1.") is not None
    assert sections.note_at("(1) La Corte costituzionale con sentenza 63/1966 ha dichiarato") == 0
    assert sections.note_at("(1) Considerando che gli Stati membri devono cooperare") is None
    assert sections.note_at("1) l'avere agito per motivi abietti o futili;") is None
    assert sections.note_at("del reato (3).") is None


def test_the_census_finds_hidden_text_and_a_read_that_stops_at_a_colon():
    rows = [{"name": "61.0.0", "plain_text": "Aggravano il reato, quando non ne sono elementi costitutivi, le circostanze seguenti:"},
            {"name": "61_2", "plain_text": "1) l'avere agito per motivi abietti o futili;"},
            {"name": "61.2_3", "plain_text": "Testo. (1) Comma modificato dall'art. 1."}]
    text = "\n".join(sections.census(rows))
    assert "end at a colon: 0" in text                  # read whole now, from these pieces
    assert "HIDDEN  N_j" in text and "UNUSED  N.k_j" in text and "N_j 1" in text
    assert "comma modificato 1" in text and "text before the note 1" in text


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    """One document per code; the 61 c.p. of the review, its list under a name the lookups cannot see."""
    SECTIONS = {"61.0.0": "Aggravano il reato le circostanze seguenti:", "61_2": "1) l'avere agito per motivi abietti o futili;"}

    def run(self, query, **p):
        if "d.name STARTS WITH" in query:
            return _Rows([{"id": p["prefix"], "name": p["prefix"]}])
        if "type(r)" in query:
            return _Rows([{"type": "CONTAINS", "n": 2}])
        if "keys(s)" in query:
            rows = self.SECTIONS.items() if p["doc"] == "Codice Penale" else []
            return _Rows([{"s": {"id": k, "name": k, "type": "Comma", "plain_text": v}, "fields": ["id", "name"]}
                          for k, v in rows])
        found = self.SECTIONS if p["items"][0]["doc"] == "Codice Penale" and p["items"][0]["num"] == "61" else {}
        item = p["items"][0]
        return _Rows([{"doc": item["doc"], "num": item["num"], "d": {"id": item["doc"]},
                       "s": {"name": k, "plain_text": v}} for k, v in found.items() if k == "61" or k.startswith("61.")])


def test_the_report_shows_the_sections_the_probe_and_the_census():
    text = "\n".join(sections.report(_Session(), ["61 c.p.", "177 c.c."]))
    assert "le circostanze seguenti:" in text and "HIDDEN" in text
    assert 'PROBE "motivi abietti o futili": Codice Penale: 61_2' in text
    assert 'PROBE "Costituiscono oggetto della comunione": NOT IN ANY SECTION' in text
    assert "--- Codice Penale (c.p.)" in text and "Section fields: id 2, name 2" in text and "CONTAINS 2" in text
