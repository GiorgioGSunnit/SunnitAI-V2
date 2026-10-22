"""scripts/exp_ruling_links.py's pure helpers, offline, on invented snippets. The
live part (reading the rulings from Neo4j) needs prod and is only exercised by
running the script there."""

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
_spec = importlib.util.spec_from_file_location("exp_ruling_links", HERE.parent / "scripts" / "exp_ruling_links.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)


# --- Articles cited -------------------------------------------------------------

def test_cited_articles_runs_and_comma_forms():
    text = ("ai sensi degli artt. 581 e 584 cod. pen., nonché dell'art. 41, secondo comma, c.p. "
            "e dell'art. 62, n. 2, c.p.; di nuovo art. 584 c.p.")
    assert rl.cited_articles(text) == ["581 c.p.", "584 c.p.", "41 c.p.", "62 c.p."]


def test_cited_articles_codes_and_suffixes():
    text = ("art. 609 bis c.p.; art. 2043 cod. civ.; art. 360, primo comma, n. 3, c.p.c.; "
            "art. 606, comma 1, lett. e), c.p.p.; art. 3 Cost.; art. 41 cpv. c.p.")
    assert rl.cited_articles(text) == ["609-bis c.p.", "2043 c.c.", "360 c.p.c.", "606 c.p.p.", "3 Cost.", "41 c.p."]


def test_cited_articles_codes_in_full():
    text = "art. 52 del codice penale e art. 125 del codice di procedura penale"
    assert rl.cited_articles(text) == ["52 c.p.", "125 c.p.p."]


def test_cited_articles_skips_numbers_without_a_code():
    text = "art. 165 disp. att. c.p.p.; art. 5 del d.lgs. n. 231 del 2001; art. 2 della legge n. 689; art. 7"
    assert rl.cited_articles(text) == []


# --- Court, branch ----------------------------------------------------------------

def test_court_of_takes_the_court_named_first():
    header = "LA CORTE SUPREMA DI CASSAZIONE ... avverso la sentenza della Corte d'appello di XXX"
    assert rl.court_of(header) == "Cassazione"
    assert rl.court_of("Il Consiglio di Stato in sede giurisdizionale") == "Consiglio di Stato"
    assert rl.court_of("TRIBUNALE AMMINISTRATIVO REGIONALE PER IL LAZIO") == "TAR"
    assert rl.court_of("TRIBUNALE DI XXX, sezione civile") == "Tribunale"
    assert rl.court_of("In tema di custodia (Cass. civ., Sez. III)") == "Cassazione"
    assert rl.court_of("nessuna corte nominata") is None


def test_branch_of_by_section_then_by_codes():
    assert rl.branch_of("CORTE DI CASSAZIONE SEZIONE QUARTA PENALE", []) == "penale"
    assert rl.branch_of("SEZIONI UNITE CIVILI", ["584 c.p."]) == "civile"
    assert rl.branch_of("Cass., Sez. Lav., ordinanza", []) == "civile"
    assert rl.branch_of("", ["2043 c.c.", "2051 c.c.", "41 c.p."]) == "civile"
    assert rl.branch_of("", ["584 c.p.", "606 c.p.p.", "2043 c.c."]) == "penale"
    assert rl.branch_of("", ["2043 c.c.", "41 c.p."]) == "?"


# --- Outcome -------------------------------------------------------------------------

QUOTED_LOWER_COURT = ("Svolgimento del processo. Il Tribunale, P.Q.M., condannava l'imputato alla pena di "
                      "anni due. Motivi della decisione. Il ricorso è fondato. ")


def test_dispositivo_prefers_the_pqm_before_the_closing():
    text = (QUOTED_LOWER_COURT + "P.Q.M. Annulla senza rinvio la sentenza impugnata perché il fatto non "
            "sussiste. Così deciso in Roma, il 1 gennaio 2020.")
    part, anchored = rl.dispositivo(text)
    assert anchored and part.strip().startswith("Annulla senza rinvio")
    assert "Così deciso" not in part
    assert rl.outcome_of(part) == "accolto"


def test_dispositivo_without_closing_takes_the_last_pqm():
    part, anchored = rl.dispositivo("Motivi. P.Q.M. Rigetta il ricorso.")
    assert not anchored and rl.outcome_of(part) == "rigettato"
    assert rl.dispositivo("nessun dispositivo qui") == ("", False)


def test_outcome_of_cases():
    assert rl.outcome_of("Dichiara inammissibile il ricorso e condanna il ricorrente al pagamento delle spese "
                         "processuali e della somma di euro tremila in favore della Cassa delle ammende.") == "inammissibile"
    assert rl.outcome_of("Annulla la sentenza impugnata limitatamente al trattamento sanzionatorio e rinvia per "
                         "nuovo giudizio; rigetta nel resto il ricorso.") == "in parte accolto"
    assert rl.outcome_of("Accoglie il ricorso, cassa la sentenza impugnata e, decidendo nel merito, "
                         "rigetta la domanda.") == "accolto"
    assert rl.outcome_of("Dichiara estinto il giudizio.") == "estinto"
    assert rl.outcome_of("Visti gli atti.") == "?"


# --- Who appealed --------------------------------------------------------------------

def test_appellant_of():
    assert rl.appellant_of("SENTENZA sul ricorso proposto da: TIZIO nato a XXX il 01/01/1970 avverso la sentenza "
                           "della Corte d'appello di XXX; udito il Pubblico Ministero") == "imputato"
    assert rl.appellant_of("sul ricorso proposto dal Procuratore generale presso la Corte d'appello di XXX nel "
                           "procedimento a carico di CAIO nato a XXX") == "pm"
    assert rl.appellant_of("sui ricorsi proposti da: TIZIO nato a XXX; dalla parte civile avverso") == "imputato+parte civile"
    assert rl.appellant_of("ordinanza senza intestazione") == "?"


def test_favours_of():
    assert rl.favours_of("imputato", "accolto") == "imputato"
    assert rl.favours_of("imputato", "in parte accolto") == "imputato"
    assert rl.favours_of("imputato", "inammissibile") == "contro l'imputato"
    assert rl.favours_of("pm", "rigettato") == "imputato"
    assert rl.favours_of("pm", "accolto") == "contro l'imputato"
    assert rl.favours_of("imputato+pm", "accolto") == "?"
    assert rl.favours_of("imputato", "estinto") == "?"


# --- One ruling, reports ----------------------------------------------------------------

RULING = ("REPUBBLICA ITALIANA LA CORTE SUPREMA DI CASSAZIONE SEZIONE QUINTA PENALE SENTENZA sul ricorso "
          "proposto da: TIZIO nato a XXX avverso la sentenza della Corte d'appello di XXX. "
          "Ritenuto in fatto: lo schiaffo provocava la caduta e un trauma cranico (artt. 581 e 584 c.p.). "
          "Considerato in diritto: va esclusa la prevedibilità; art. 41, secondo comma, c.p. "
          "P.Q.M. Annulla senza rinvio la sentenza impugnata perché il fatto non sussiste. Così deciso in Roma.")


def test_read_ruling_labels_without_text():
    r = rl.read_ruling("doc-1", {"number": 12345, "date": "2020-01-01"}, RULING,
                       words=["schiaffo", "trauma cranico", "pugno"])
    assert r["court"] == "Cassazione" and r["branch"] == "penale"
    assert r["articles"] == ["581 c.p.", "584 c.p.", "41 c.p."]
    assert r["outcome"] == "accolto" and r["pqm_anchored"] and r["senza_rinvio"] and r["acquittal_formula"]
    assert r["appellant"] == "imputato" and r["favours"] == "imputato"
    assert r["words"] == ["schiaffo", "trauma cranico"]
    assert rl.full_reference(r)
    assert all(not isinstance(v, str) or len(v) < 40 for v in r.values())   # no text kept


def test_read_ruling_without_pqm_or_text():
    r = rl.read_ruling("doc-2", {}, "")
    assert r["outcome"] == "nessun P.Q.M." and r["articles"] == [] and r["chars"] == 0
    assert not rl.full_reference(r)


def test_shape_hides_content():
    assert rl.shape("12345/2019") == "99999/9999"
    assert rl.shape("Rossi n. 7") == "aaaaa a. 9"
    assert rl.shape(None) == "(empty)"


def _rec(i, articles, branch="penale", appellant="imputato", outcome="accolto", words=()):
    return {"id": f"d{i}", "number": i, "date": "2020", "court": "Cassazione", "branch": branch,
            "outcome": outcome, "pqm_anchored": True, "senza_rinvio": outcome == "accolto",
            "acquittal_formula": False, "prescription": False, "appellant": appellant,
            "favours": rl.favours_of(appellant, outcome) if branch == "penale" else "-",
            "chars": 1000, "articles": list(articles), "words": list(words)}


def test_focus_report_and_report_lines():
    records = [
        _rec(1, ["584 c.p.", "41 c.p."], words=["schiaffo", "cadut"]),
        _rec(2, ["584 c.p.", "589 c.p."], outcome="inammissibile", words=["schiaffo"]),
        _rec(3, ["584 c.p."], appellant="pm", outcome="rigettato", words=["cadut"]),
        _rec(4, ["2043 c.c."], branch="civile", appellant="-", outcome="rigettato"),
    ]
    f = rl.focus_report(records, "584 c.p.", ["schiaffo", "cadut"])
    assert f["n"] == 3 and f["pro"] == 2
    assert f["favours"] == {"imputato": 2, "contro l'imputato": 1}
    assert f["with"] == {"41 c.p.": 1, "589 c.p.": 1}
    assert f["words"] == {"schiaffo": (2, 1), "cadut": (2, 2)}
    assert f["pro_two_words"] == 1
    lines = rl.report(records, "584 c.p.", ["schiaffo", "cadut"])
    assert any(line.startswith("## Rulings citing 584 c.p.: 3") for line in lines)
    assert any("Full reference possible" in line for line in lines)
