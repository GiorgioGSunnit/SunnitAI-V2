"""v79.19.6 -- the 21 special-law documents ingested on 27 Nov 2026 registered in the families (data only): the tables
load, each archive name resolves exactly as written to its law, its table entry and its family; the aliases by common
name and by number name the right law; the mediation matters route the new store names; the modules' laws are the same
in both files. Synthetic sentences; the archive names are the user's.
"""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from src.rag import corpus_laws, family_gate, render_computed, special_laws  # noqa: E402

# (archive name, corpus_expected law, table entry id, family)
DOCS = [
    ("Licenziamenti Individuali L. 604/1966 2026", "L. 604/1966", "legge_604_1966", "lavoro"),
    ("Statuto dei Lavoratori 2026", "L. 300/1970", "statuto_lavoratori", "lavoro"),
    ("Contratto a Tutele Crescenti 2026", "d.lgs. 23/2015", "tutele_crescenti", "lavoro"),
    ("Disciplina Organica dei Contratti di Lavoro 2026", "d.lgs. 81/2015", "contratti_lavoro_81_2015", "lavoro"),
    ("Licenziamenti Individuali L. 108/1990 2026", "L. 108/1990", "legge_108_1990", "lavoro"),
    ("Licenziamenti Collettivi e Cassa Integrazione 2026", "L. 223/1991", "legge_223_1991", "lavoro"),
    ("Testo Unico Pubblico Impiego 2026", "d.lgs. 165/2001", "tu_pubblico_impiego", "lavoro"),
    ("Locazioni di Immobili Urbani 2026", "L. 392/1978", "locazioni_urbane_392_1978", "locazioni"),
    ("Locazioni Abitative 2026", "L. 431/1998", "locazioni_abitative_431_1998", "locazioni"),
    ("Legge Gelli-Bianco Responsabilità Sanitaria 2026", "L. 24/2017", "gelli_bianco", "sanitario"),
    ("Mediazione Civile e Commerciale 2026", "d.lgs. 28/2010", "mediazione_civile", "mediazione"),
    ("Negoziazione Assistita 2026", "d.l. 132/2014", "negoziazione_assistita", "procedura civile"),
    ("Riti Civili Semplificati 2026", "d.lgs. 150/2011", "riti_semplificati_150_2011", "procedura civile"),
    ("Testo Unico Spese di Giustizia 2026", "d.P.R. 115/2002", "tu_spese_giustizia", "procedura civile"),
    ("Competenza Penale del Giudice di Pace 2026", "d.lgs. 274/2000", "giudice_pace_penale_274_2000", "penale"),
    ("Depenalizzazione e Sanzioni Amministrative 2026", "L. 689/1981", "depenalizzazione_689_1981", "penale"),
    ("Codice Rosso 2026", "L. 69/2019", "codice_rosso_69_2019", "penale"),
    ("Misure Urgenti Sicurezza Pubblica e Stalking 2026", "d.l. 11/2009", "sicurezza_pubblica_stalking_11_2009", "penale"),
    ("Disciplina delle Armi 2026", "L. 110/1975", "disciplina_armi_110_1975", "penale"),
    ("Unioni Civili e Convivenze 2026", "L. 76/2016", "unioni_civili_76_2016", "famiglia"),
    ("Servizi di Pagamento 2026", "d.lgs. 11/2010", "dlgs_11_2010", "bancario"),
]


def _data(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


# --- the families load --------------------------------------------------------------------------------------------------

def test_the_tables_load_and_every_document_has_its_entries():
    table = {x["id"]: x for x in special_laws.laws()}         # every expression compiled, none skipped
    assert len(table) == len(_data("special_laws.json")["laws"]) == 46
    exp = _data("corpus_expected.json")["laws"]
    for name, law, law_id, family in DOCS:
        assert name in exp[law]["archive_names"], name
        assert table[law_id]["family"] == family and table[law_id]["expected"] == law
        assert table[law_id]["archive_names"] == [name] and name in table[law_id]["_archive"]
        assert set(table[law_id]["_archive"]) == set(corpus_laws.archive_names(law))
        assert table[law_id]["_signals"], law_id                  # each has a domain signal of its own
    assert len({d[0] for d in DOCS}) == len({d[2] for d in DOCS}) == 21
    assert all(x["expected"] in exp for x in table.values() if "expected" in x)
    assert not [k for k, v in exp.items() if not v["archive_names"]]       # no unmapped law is left


def test_the_modules_laws_are_the_same_in_both_files():
    exp = _data("corpus_expected.json")["modules"]
    mods = {m["id"]: m.get("laws") or [] for m in _data("institution_modules.json")["modules"] if m.get("laws")}
    assert exp == mods
    assert exp["licenziamento"][-3:] == ["L. 108/1990", "L. 223/1991", "d.lgs. 165/2001"]
    assert "d.lgs. 165/2001" in exp["amministrativo"] and exp["famiglia"] == ["L. 898/1970", "L. 76/2016"]


# --- each name resolves exactly as written -------------------------------------------------------------------------------

@pytest.mark.parametrize("name, law, law_id, family", DOCS)
def test_each_name_resolves_to_its_law_entry_and_family(name, law, law_id, family):
    assert corpus_laws.law_of_name(name) == law and corpus_laws.citation_of(name) == law
    assert corpus_laws.present(law, [name]) and corpus_laws.status(law, [name]) == corpus_laws.PRESENT
    assert special_laws.law_of(name, name_only=True)["id"] == law_id
    assert family_gate.classify(name) == (family, law_id)
    assert family_gate.classify("", name) == (family, law_id)              # a row with its document's name
    # not a variant: case, a substring, a numbered name
    assert corpus_laws.law_of_name(name.lower()) == "" and corpus_laws.law_of_name(name.rsplit(" ", 1)[0]) == ""
    assert family_gate.classify(f"Legge n. {law.split()[-1].split('/')[0]}") == (family_gate.OTHER, "")


def test_an_article_label_of_a_new_document_is_cited_by_its_law():
    assert corpus_laws.cited_label("18 Statuto dei Lavoratori 2026") == "18 L. 300/1970"
    assert corpus_laws.cited_label("3 Legge Gelli-Bianco Responsabilità Sanitaria 2026 co. 1") == "3 L. 24/2017 co. 1"
    assert corpus_laws.cited_label("4 Codice Rosso 2026") == "4 L. 69/2019"


# --- aliases: by common name and by number -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, law, alias", [
    ("il lavoratore chiede la reintegra ai sensi dello Statuto dei lavoratori", "L. 300/1970", "statuto dei lavoratori"),
    ("assunto con contratto a tutele crescenti (Jobs Act)", "d.lgs. 23/2015", "tutele crescenti"),
    ("responsabilità del chirurgo secondo la legge Gelli", "L. 24/2017", "legge gelli"),
    ("un procedimento per codice rosso", "L. 69/2019", "codice rosso"),
    ("il T.U. pubblico impiego regola il rapporto", "d.lgs. 165/2001", "t.u. pubblico impiego"),
    ("la negoziazione assistita è condizione di procedibilità", "d.l. 132/2014", "negoziazione assistita"),
    ("il contributo unificato dovuto", "d.P.R. 115/2002", "contributo unificato"),
    ("una convivenza di fatto registrata (legge Cirinnà)", "L. 76/2016", "legge cirinnà"),
    ("i licenziamenti collettivi e la cassa integrazione", "L. 223/1991", "licenziamenti collettivi"),
    ("un canone di locazione ad equo canone", "L. 392/1978", "equo canone"),
    ("un bonifico non autorizzato: servizi di pagamento", "d.lgs. 11/2010", "servizi di pagamento"),
])
def test_an_alias_names_the_right_law(text, law, alias):
    got = family_gate.naming(text, "", law, "", aliases=True)
    exp = _data("corpus_expected.json")["laws"]
    assert got and got in exp[law]["aliases"] and alias in exp[law]["aliases"], (text, got)


@pytest.mark.parametrize("text, law_id, law, said", [
    ("licenziato ai sensi della L. 604/1966", "legge_604_1966", "L. 604/1966", "604/1966"),
    ("il d.lgs. 23/2015 sulle tutele crescenti", "tutele_crescenti", "d.lgs. 23/2015", "23/2015"),
    ("la legge 108/1990", "legge_108_1990", "L. 108/1990", "108/1990"),
    ("ex d.lgs. 274/2000", "giudice_pace_penale_274_2000", "d.lgs. 274/2000", "274/2000"),
    ("la L. 689/1981", "depenalizzazione_689_1981", "L. 689/1981", "689/1981"),
    ("il d.l. 11/2009 sullo stalking", "sicurezza_pubblica_stalking_11_2009", "d.l. 11/2009", "11/2009"),
])
def test_the_number_names_the_law(text, law_id, law, said):
    assert family_gate.naming(text, law_id, law, "") == said
    assert special_laws.law_named(text)["id"] == law_id


def test_the_bare_word_statuto_is_no_alias_and_lesioni_names_nothing():
    exp = _data("corpus_expected.json")["laws"]
    assert "statuto" not in exp["L. 300/1970"]["aliases"]
    assert family_gate.naming("modifica dello statuto sociale", "", "L. 300/1970", "", aliases=True) == ""
    assert special_laws.law_named("una lesione personale") is None


# --- the family gate admits a document by its signal, the user's naming or the module -----------------------------------------

def _row(number, doc):
    return {"_article": (number, doc), "d": {"name": doc}}


def test_a_new_document_passes_by_its_signal_and_is_dropped_without_it():
    gelli = "Legge Gelli-Bianco Responsabilità Sanitaria 2026"
    rows = [_row("7", gelli)]
    ctx = family_gate.context("Un errore chirurgico in ospedale ha causato un danno al paziente.")
    kept, dropped, outside = family_gate.gate(rows, ctx, set())
    assert [r["_article"][1] for r in kept] == [gelli] and outside[0]["family"] == "sanitario"
    assert outside[0]["by"].startswith("segnale")
    loan = family_gate.context("Tizio ha prestato un libro a Caio, che non lo restituisce.")
    kept, dropped, _ = family_gate.gate(rows, loan, set())
    assert kept == [] and dropped[0]["family"] == "sanitario"
    named = family_gate.context("Tizio chiede un parere sulla legge Gelli.")
    assert family_gate.decide("", gelli, named)["by"] == "nominata dall'utente"
    rosso = _row("1", "Codice Rosso 2026")
    stalk = family_gate.context("La ex compagna di Tizio lo accusa di maltrattamenti e di stalking.")
    assert family_gate.gate([rosso], stalk, set())[0] == [rosso]
    assert family_gate.gate([rosso], loan, set())[0] == []
    # the mediation act is a default family: admitted in every civil case
    mediazione = _row("5", "Mediazione Civile e Commerciale 2026")
    assert family_gate.decide("", "Mediazione Civile e Commerciale 2026", loan)["by"] == "default"
    assert family_gate.gate([mediazione], loan, set())[0] == [mediazione]


def test_special_laws_gate_reads_the_new_names():
    row = _row("18", "Statuto dei Lavoratori 2026")
    assert special_laws.gate([row], "Tizio ha prestato un libro a Caio.", set())[0] == []
    assert special_laws.gate([row], "Il lavoratore è stato licenziato senza contestazione disciplinare.", set())[0] == [row]


# --- the mediation matters take the new store names ---------------------------------------------------------------------------

@pytest.mark.parametrize("ref, matter, never", [
    (("18", "Statuto dei Lavoratori 2026"), "rito del lavoro", True),
    (("1", "Licenziamenti Individuali L. 604/1966 2026"), "rito del lavoro", True),
    (("55", "Testo Unico Pubblico Impiego 2026"), "rito del lavoro", True),
    (("1", "Codice Rosso 2026"), "reati", True),
    (("4", "Disciplina delle Armi 2026"), "reati", True),
    (("3", "Legge Gelli-Bianco Responsabilità Sanitaria 2026"), "risarcimento del danno da responsabilità medica e sanitaria",
     False),
    (("2", "Locazioni Abitative 2026"), "locazione", False),
    (("27", "Locazioni di Immobili Urbani 2026"), "locazione", False),
    (("10", "Servizi di Pagamento 2026"), "contratti bancari e finanziari", False),
    (("5", "Mediazione Civile e Commerciale 2026"), "", False),         # the act itself is no matter
])
def test_the_mediation_matters_route_the_new_store_names(ref, matter, never):
    assert render_computed.matter_of(ref) == (matter, never)


def test_mediation_matters_name_the_acts_they_come_from():
    mm = _data("mediation_matters.json")
    assert mm["mediazione"]["archive_names"][0] == "Mediazione Civile e Commerciale 2026"
    assert mm["negoziazione_assistita"]["archive_names"] == ["Negoziazione Assistita 2026"]
    assert mm["mediazione"]["ref"] == "art. 5, co. 1, d.lgs. 28/2010"
    assert "servizi di pagamento" in mm["mediazione"]["matters"]["contratti bancari e finanziari"]["laws"]


# --- the institutions' governing law --------------------------------------------------------------------------------------------

def test_institution_laws_name_the_new_laws():
    il = _data("institution_laws.json")["institutions"]
    laws = {x["name"]: x for x in il}
    assert laws["statuto dei lavoratori"]["law"] == "L. 300/1970" and laws["codice rosso"]["penal"] is True
    assert laws["pubblico impiego"]["law"] == "d.lgs. 165/2001" and "penal" not in laws["pubblico impiego"]
    exp = _data("corpus_expected.json")["laws"]
    assert all(x["law"] in exp or x["law"] in ("c.c.", "c.p.", "c.p.c.", "c.p.p.", "Reg. UE 2016/679", "d.lgs. 79/2011",
                                                 "d.lgs. 30/2005", "L. 633/1941", "d.lgs. 14/2019", "d.lgs. 36/2023",
                                                 "d.lgs. 231/2001", "d.lgs. 385/1993", "d.lgs. 209/2005")
               for x in il), [x["law"] for x in il]
    names = [x["name"] for x in il]
    assert names.index("codice rosso") < names.index("legittima difesa")      # the special laws before the codes
