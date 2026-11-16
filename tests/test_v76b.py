"""v76b: the decomposer fixed against the failure patterns of scripts/decompose_report.py -- one test per pattern, never
per article -- and the report's own reading of a code's text."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.rag import issue_records, rule_quote as R  # noqa: E402


def _elements(number, code, text):
    return [r["text"] for r in issue_records._requisites({"_article": (number, code), "s": {"plain_text": text}}, {})
            if r["kind"] == "requisito"]


def test_a_later_sentence_never_makes_a_comma_a_rinvio():
    text = ("Chiunque distrugge, o rende inservibili, in tutto o in parte, navi o altre opere militari è punito con la "
            "reclusione non inferiore a otto anni. Si applica la pena dell'ergastolo se il fatto è commesso nell'interesse "
            "di uno Stato in guerra.")
    assert not R.refers_penalties(text)
    assert R.refers_penalties("Le pene stabilite negli articoli precedenti sono diminuite se la simulazione concerne una "
                              "contravvenzione.")
    assert R.refers_penalties("Le disposizioni degli articoli 295 e 296 si applicano solo in quanto la legge straniera …")


def test_a_qualified_subject_is_its_own_subject_and_an_element():
    text = ("Il cittadino che porta le armi contro lo Stato o presta servizio nelle forze armate di uno Stato in guerra "
            "contro lo Stato italiano, è punito con l'ergastolo.")
    assert not R.refers_penalties(text)
    assert _elements("242", "c.p.", text)[0] == "il cittadino"
    assert len(_elements("242", "c.p.", text)) >= 2
    assert R.qualified_subject("Il proprietario, il cui fondo è circondato da fondi altrui, e che non ha uscita")[0] == ""


def test_the_subject_after_the_penalty_and_after_a_riserva():
    after = ("Salvo che il fatto costituisca più grave reato, è punito con la reclusione da tre a otto anni il pubblico "
             "ufficiale o l'incaricato di pubblico servizio che, al fine di impedire o ostacolare un'indagine, immuta "
             "artificiosamente il corpo del reato.")
    assert len(_elements("375", "c.p.", after)) >= 2
    consumer = ("Salvo che il fatto costituisca reato, l'operatore che contravviene alle norme di cui agli articoli 70 e "
                "71 è punito con la sanzione amministrativa pecuniaria da 516 a 25.823 euro.")
    assert not R.refers_penalties(consumer)
    got = _elements("81", "codice_del_consumo", consumer)
    assert got[0] == "l'operatore" and not any(x.lower().startswith("salvo che") for x in got)


def test_a_clausola_di_riserva_and_a_default_are_never_elements():
    for clause in ("Salvo che il fatto costituisca più grave reato", "Fuori dei casi previsti dall'articolo 314",
                   "il contratto non dispone diversamente", "salvo patto contrario", "salvo quanto previsto nel comma 2",
                   "fermo restando l'obbligo"):
        assert not R.element_kept(clause), clause
    assert R.element_kept("la cosa venduta sia immune da vizi")


def test_conditions_the_connector_missed_and_long_ones():
    got = _elements("1931", "c.c.", "In caso di liquidazione coatta dell'impresa del riassicuratore, i debiti e i "
                                    "crediti si compensano di diritto.")
    assert got and got[0] == "liquidazione coatta dell'impresa del riassicuratore"
    recesso = R.requisites("In caso di recesso il cliente ha diritto alla restituzione delle somme versate.")
    assert not any("ha diritto" in x for x in recesso)          # the condition ends where the main clause begins
    se_la = ("Chi ha ricevuto le cose non è liberato dall'obbligo di pagarne il prezzo, se la restituzione di esse nella "
             "loro integrità è divenuta impossibile per causa a lui non imputabile.")
    assert _elements("1557", "c.c.", se_la)[0] == "Chi ha ricevuto le cose"


def test_a_single_condition_is_cut_on_its_commas_never_on_an_adverbial():
    t1062 = ("Se i due fondi cessarono di appartenere allo stesso proprietario, senza alcuna disposizione relativa alla "
             "servitù, questa s'intende stabilita attivamente e passivamente.")
    assert R.civil_split(t1062) == ["i due fondi cessarono di appartenere allo stesso proprietario",
                                    "senza alcuna disposizione relativa alla servitù"]
    assert not any(x.startswith("senza eccessivo") for x in R.civil_elements(
        "Il proprietario, il cui fondo è circondato da fondi altrui, e che non ha uscita sulla via pubblica né può "
        "procurarsela senza eccessivo dispendio o disagio, ha diritto di ottenere il passaggio."))


def test_a_single_condition_is_an_element_of_the_record_only():
    text = "Se nell'atto costitutivo non si prevede il recesso del socio, la quota è intrasferibile per atto tra vivi."
    assert R.civil_split(text) == [] and R.civil_split(text, single=True) == [
        "nell'atto costitutivo non si prevede il recesso del socio"]
    assert _elements("2469", "c.c.", text) == ["nell'atto costitutivo non si prevede il recesso del socio"]


def test_a_special_laws_penal_provision_takes_the_offence_path():
    text = ("Chiunque, senza l'autorizzazione di cui all'articolo 17, coltiva, produce o fabbrica sostanze stupefacenti "
            "è punito con la reclusione da sei a venti anni.")
    assert R.penal_provision(text)
    got = _elements("73", "testo_unico_stupefacenti", text)
    assert len(got) >= 2 and not any("è punito" in x for x in got)


def test_rubriche_that_administer_other_articles_are_no_fattispecie():
    for rubrica in ("Circostanze aggravanti", "Pene accessorie", "Confisca", "Casi di non punibilità", "Giurisdizione",
                    "Competenza", "Disposizioni transitorie", "Entrata in vigore"):
        assert not issue_records.counted("999 c.p.", rubrica), rubrica


# --- the report ---------------------------------------------------------------------------------------------------------------

def test_the_report_reads_a_codes_text_as_the_store_keeps_it():
    import decompose_report as D
    text = ("TITOLO XII - DEI DELITTI CONTRO LA PERSONA\nArt. 584. Omicidio preterintenzionale\nChiunque, con atti "
            "diretti a commettere uno dei delitti preveduti dagli\narticoli 581 e 582, cagiona la morte di un uomo, è punito "
            "con la reclusione\nda dieci a diciotto anni.\nAltalex eBook | Collana Codici Altalex\n(1) Nota editoriale.\n"
            "Art. 636. Introduzione o abbandono di animali nel fondo altrui e pascolo\nabusivo\nChiunque introduce animali "
            "nel fondo altrui è punito con la multa.\nArt. 73 (Legge 26 giugno 1990, n. 162, art. 14, comma 1)\nProduzione e "
            "traffico illeciti [90]\nIn vigore dal 25 aprile 2026\n1.\nChiunque coltiva, produce sostanze è punito.\n")
    arts = {n: (r, b) for n, r, b in D.pdf_articles(text)}
    assert arts["584"][0] == "Omicidio preterintenzionale" and "Nota editoriale" not in arts["584"][1]
    assert "cagiona la morte di un uomo, è punito con la reclusione da dieci a diciotto anni." in arts["584"][1]
    assert arts["636"][0].endswith("pascolo abusivo") and arts["636"][1].startswith("Chiunque introduce")
    assert arts["73"][0] == "Produzione e traffico illeciti" and arts["73"][1].startswith("1. Chiunque coltiva")
    rows = [D.decompose("c.p.", "584", *arts["584"]), D.decompose("c.p.", "636", *arts["636"])]
    for r in rows:
        r["group"] = "c.p. Libro II"
    tot = next(t for t in D.totals(rows) if t["group"] == "c.p. Libro II")
    assert tot["articles"] == 2 and tot["two_or_more"] == 50.0 and tot["none"] == 0.0
    assert D.in_scope("c.p.", "56") is None and D.in_scope("c.c.", "1490") == "c.c. IV.III singoli contratti"
