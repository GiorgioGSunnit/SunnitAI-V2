"""scripts/unlinked_citations.py: what the CITES linker leaves unlinked, sorted by kind (28 Oct)."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("unlinked_citations", HERE.parent / "scripts" / "unlinked_citations.py")
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)


def test_special_laws_in_their_usual_spellings():
    cases = {
        " d.lgs. n. 231/2001, in tema di": "d.lgs. 231/2001",
        " del d.lgs. 8 giugno 2001, n. 231": "d.lgs. 231/2001",
        ", comma 1, lett. b), d.P.R. 9 ottobre 1990, n. 309": "d.P.R. 309/1990",
        " DPR 600/73": "d.P.R. 600/1973",
        " della legge n. 241 del 1990": "l. 241/1990",
        " l. 212/2000 (statuto)": "l. 212/2000",
        " del decreto legislativo 31 dicembre 1992, n. 546": "d.lgs. 546/1992",
        " d.l. n. 18 del 2020": "d.l. 18/2020",
        " r.d. 16 marzo 1942, n. 267": "r.d. 267/1942",
        " d.lgs. n. 74 dell'10 marzo 2000": "d.lgs. 74/2000",
    }
    for tail, law in cases.items():
        assert uc.classify(tail, "") == ("special law", law), tail


def test_the_other_kinds():
    assert uc.classify(" disp. att. c.p.p.", "") == ("implementing rules", "disp. att. c.p.p.")
    assert uc.classify(" delle disposizioni di attuazione del c.p.c.", "") == ("implementing rules", "disp. att. c.p.c.")
    assert uc.classify(" TUIR", "") == ("named act", "TUIR (d.P.R. 917/1986)")
    assert uc.classify(" c.d.s., per la guida", "") == ("named act", "Codice della strada (d.lgs. 285/1992)")
    assert uc.classify(" ord. pen.", "") == ("named act", "Ordinamento penitenziario (l. 354/1975)")
    assert uc.classify(" l. fall.", "") == ("named act", "Legge fallimentare (r.d. 267/1942)")
    assert uc.classify(" CEDU", "") == ("EU / international", "")
    assert uc.classify(" della Convenzione europea", "") == ("EU / international", "")
    assert uc.classify(" dello stesso decreto", "") == ("same act, named before", "")
    assert uc.classify(" cit., che", "") == ("same act, named before", "")
    assert uc.classify(" e 2059, il danno", "ai sensi degli artt. 2043 c.c. e ") == ("code named earlier in the sentence", "")
    assert uc.classify(" e 13", "ai sensi dell'art. 5 d.lgs. n. 231/2001 e ") == ("law named earlier in the sentence", "")
    kind, detail = uc.classify(" della convenzione di 12 anni", "Le parti. Il contratto ")
    assert kind == "unknown" and "##" in detail


def test_a_code_named_in_an_earlier_sentence_does_not_count():
    assert uc.classify(" e 13", "Si applica l'art. 2043 c.c. Il giudice ritiene che l'") == ("unknown", "e ##")


def test_precedents_by_number_and_year_not_the_own_docket():
    text = ("Come chiarito da Cass. Sez. U, n. 18672 del 2019 e da Cass. civ., sez. II, ord. n. 1234/2021, il termine... "
            "LA CORTE SUPREMA DI CASSAZIONE ORDINANZA sul ricorso iscritto al n. 15966/2025")
    assert uc.precedents(text) == [("18672", "2019"), ("1234", "2021")]


def test_find_document():
    docs = [{"name": "Responsabilità Amministrativa di Società ed Enti 2026"},
            {"name": "Codice dei contratti pubblici - Decreto Legislativo 31 marzo 2023, n. 36"},
            {"name": "Legge sui Reati Tributari 2024"}]
    assert uc.find_document("d.lgs. 231/2001", docs) == "Responsabilità Amministrativa di Società ed Enti 2026"
    assert uc.find_document("d.lgs. 36/2023", docs) == "Codice dei contratti pubblici - Decreto Legislativo 31 marzo 2023, n. 36"
    assert uc.find_document("d.P.R. 600/1973", docs) is None
    assert uc.find_document("d.P.R. 600/1973", docs, resolve=lambda law, d: {"name": "Accertamento"}) == \
        "Accertamento (by name: to check)"


def _code_after(tail):
    """A stand-in for the linker's rule: a code right at the start (after "del")."""
    import re
    m = re.match(r"\s*,?\s*(?:del\s+)?(c\.p\.p\.|cod\. proc\. pen|cod\. proc\. civ|c\.c\.)", tail, re.IGNORECASE)
    return {"c.p.p.": "c.p.p.", "cod. proc. pen": "c.p.p.", "cod. proc. civ": "c.p.c.", "c.c.": "c.c."}.get(
        m.group(1).lower()) if m else None


def test_main_code_citations_todays_rule_misses():
    assert uc.classify(" Corte di Cassazione - copia non ufficiale 3 cod. proc. pen.", "", _code_after) == \
        ("code missed by today's rule", "c.p.p.")
    assert uc.classify(", comma 1, lett. b) ed e), cod. proc. pen., in relazione", "", _code_after) == \
        ("code missed by today's rule", "c.p.p.")
    assert uc.classify(", comma 1, lett. b) e 125, comma 3, cod. proc. pen.", "", _code_after) == \
        ("code missed by today's rule", "c.p.p.")
    assert uc.classify(", 1° co., n. 3, cod. proc. civ.", "", _code_after) == ("code missed by today's rule", "c.p.c.")
    assert uc.classify("-ter, comma 1, c.p.p.", "", _code_after) == ("code missed by today's rule", "c.p.p.")
    assert uc.classify(" d.lgs. 231/2001", "", _code_after) == ("special law", "d.lgs. 231/2001")


def test_suffixed_commi_the_court_fee_and_the_same_article():
    assert uc.classify(", comma 1 quater, del d.P.R. n. 115 del 2002", "") == ("court-fee boilerplate", "d.P.R. 115/2002")
    assert uc.classify(", comma 17, della legge n. 228 del 2012", "") == ("court-fee boilerplate", "l. 228/2012")
    assert uc.classify(", comma 1-quater, d.P.R. 30 maggio 2002, n. 115", "") == ("court-fee boilerplate", "d.P.R. 115/2002")
    assert uc.classify(", se dovuto. Così deciso in Roma", "a norma del comma 1-bis dello stesso ") == \
        ("same act, named before", "")


def test_a_law_number_is_not_a_precedent():
    assert uc.precedents("come ritenuto da Cass. n. 1234/2021, ai sensi del d.P.R. n. 115/2002") == [("1234", "2021")]
    assert uc.precedents("Cass., sez. VI, in applicazione della legge n. 103/2017") == []


def test_other_courts_and_the_fund_are_not_cassation_precedents():
    text = ("in favore della Cassa delle ammende, non ravvisandosi assenza di colpa (Corte cost. n. 186 del 13 giugno 2000). "
            "Avverso la sentenza propone ricorso per cassazione, richiamando Corte cost. n. 120/2014; "
            "Corte di Cassazione - copia non ufficiale 4 n. 77/2019; come da Cass. pen. Sez. 6, n. 5501 del 2020")
    assert uc.precedents(text) == [("5501", "2020")]


def test_our_ruling_from_its_file_stem():
    assert uc.our_ruling("snpen@s10@a2026@n01234@tS.clean.pdf Sentenza") == ("penale", "1234", "2026")
    assert uc.our_ruling("snciv_s30_a2025_n27195_tO.pdf") == ("civile", "27195", "2025")
    assert uc.our_ruling("Ordinanza senza file") is None
