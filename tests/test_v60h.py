"""v60h: procedural facts as closed lists and record strictness -- the d'ufficio conditions with their record statuses
(1), the mediazione's closed list and the negoziazione's own (2), terms only with their triggering fact (3), the first
read's second and third search channels and the channel per anchor (4), the record's strictness (5), sanction articles
as lines (6), accent-insensitive relative dates (7), restatements that name facts (8), roles that are not names (9), the
copy export (10), the analysis stage's calls (11), the ingestion report (12). The model and the graph are stubbed."""

import importlib.util
import os
import re
import sys
import time
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import (answer_ledger, fact_resolver, fact_signals, issue_records, party_names,  # noqa: E402
                     procedural_facts, record_check, render_computed, stage_meter)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402
from test_v60f import WRITER, _graded_state  # noqa: E402
from test_v60g import _answer  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
ART_582 = [(1, "Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella "
               "mente, è punito con la reclusione da sei mesi a tre anni."),
           (2, "Si procede tuttavia d'ufficio se ricorre taluna delle circostanze aggravanti previste negli articoli 61, "
               "numero 11-octies), 583 e 585. Si procede altresì d'ufficio se il fatto è commesso contro persona "
               "incapace, per età o per infermità."),
           (3, "Il delitto è punibile a querela della persona offesa.")]
ART_590 = [(1, "Chiunque cagiona ad altri per colpa una lesione personale è punito con la reclusione fino a tre mesi."),
           (5, "Il delitto è punibile a querela della persona offesa, salvo nei casi previsti nel primo e secondo "
               "capoverso, limitatamente ai fatti commessi con violazione delle norme per la prevenzione degli "
               "infortuni sul lavoro.")]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 1. procedibilità: the d'ufficio conditions, each with its record status ------------------------------------------

def test_the_dufficio_conditions_are_read_from_the_offences_text():
    facts = procedural_facts.extract("582", "c.p.", ART_582)
    assert facts["querela"] and facts["querela_comma"] == "co. 3" and procedural_facts.VERSION == 2
    assert [(c["comma"], c["opener"], c["text"][:22]) for c in facts["ufficio_condizioni"]] == [
        ("co. 2", "se", "ricorre taluna delle c"), ("co. 2", "se", "il fatto è commesso co")]
    salvo = procedural_facts.extract("590", "c.p.", ART_590)["ufficio_condizioni"]
    assert len(salvo) == 1 and salvo[0]["opener"] == "" and salvo[0]["text"].startswith("nei casi previsti")
    split = procedural_facts.conditions([(3, "Il delitto è punibile a querela della persona offesa. Si procede, "
                                             "tuttavia, d'ufficio se la persona offesa è incapace, ovvero se ricorre "
                                             "taluna delle circostanze di cui all'articolo 625.")])
    assert [c["text"][:12] for c in split] == ["la persona o", "ricorre talu"]       # «Si procede, tuttavia,», «ovvero se»
    che = procedural_facts.conditions([(3, "Il delitto è punibile a querela della persona offesa, salvo che ricorra "
                                             "un'altra circostanza aggravante.")])
    assert che and che[0]["opener"] == "se" and che[0]["text"].startswith("ricorra")


def test_each_condition_takes_its_record_status_and_the_line_follows_them():
    facts = procedural_facts.extract("582", "c.p.", ART_582)
    first, second = facts["ufficio_condizioni"]
    mixed = procedural_facts.with_statuses(facts, [{"text": second["text"], "status": "non risulta"}])
    line = procedural_facts.line("582", "Lesione personale", mixed)
    assert "procedibile a querela della persona offesa (co. 3); d'ufficio se ricorre" in line
    assert "(co. 2: da provare); oppure se il fatto è commesso" in line and "(co. 2: non risulta)" in line
    assert "esito: " + C.PROCEDURE_VERDICTS["da provare"] in line
    proved = procedural_facts.with_statuses(facts, [{"text": first["text"], "status": "provato"}])
    assert proved["ufficio_esito"] == "provato" and C.PROCEDURE_VERDICTS["provato"] in procedural_facts.line(
        "582", "Lesione personale", proved)
    none = procedural_facts.with_statuses(facts, [{"text": c["text"], "status": "non risulta"}
                                                  for c in facts["ufficio_condizioni"]])
    assert none["ufficio_esito"] == "non risulta"
    # the writer's claim checked against the verdict: proved -> «a querela» contradicts, «d'ufficio» stands
    log = []
    out = procedural_facts.consistency([("Risposta", "Il reato di cui all'art. 582 c.p. è procedibile a querela.")],
                                       {"582": proved}, {"582": "Lesione personale"}, log)
    assert log and "condizione provata" in log[0]["why"] and "procedibile a querela" not in out[0][1].split(":")[0]
    log = []
    procedural_facts.consistency([("Risposta", "Il reato di cui all'art. 582 c.p. è procedibile d'ufficio.")],
                                 {"582": proved}, {"582": "Lesione personale"}, log)
    assert log == []


def test_the_conditions_are_linked_and_settled_like_requisites_never_part_of_the_verdict():
    row = {"_article": ("590", "c.p."), "_proc": procedural_facts.extract("590", "c.p.", ART_590)}
    conds = issue_records._conditions(row)
    assert conds and conds[0]["kind"] == "procedibilita" and conds[0]["status"] == "da provare"
    records = {"590 c.p.": {"requisites": [{"id": "", "kind": "requisito", "comma": 1, "text": "lesione personale per colpa",
                                            "facts_status": "da provare", "status": "da provare", "supporting": [],
                                            "contradicting": [], "missing": ""}],
                            "procedibilita": conds, "verdict": "da provare"}}
    issue_records.number(records)
    assert records["590 c.p."]["procedibilita"][0]["id"] == "P1"
    sents = [{"id": "S1", "text": "Il lavoratore è caduto dal ponteggio: il datore non rispettava le norme per la "
                                  "prevenzione degli infortuni sul lavoro.", "certainty": "accertato"}]
    asked = []

    def call(topic, reqs, s):
        asked.extend(r["id"] for _, r in reqs)
        return {"P1": {"sostiene": ["S1"], "contraddice": [], "manca": ""}}
    issue_records.link(records, [{"title": "Lesioni"}], [["590 c.p."]], sents, call=call)
    assert "P1" in asked
    cond = records["590 c.p."]["procedibilita"][0]
    assert cond["status"] == "provato" and records["590 c.p."]["verdict"] == "da provare"     # the verdict: requisites only
    assert procedural_facts.with_statuses(row["_proc"], records["590 c.p."]["procedibilita"])["ufficio_esito"] == "provato"


# --- 2. the mediazione's closed list, the negoziazione's own --------------------------------------------------------------

def test_mediation_appears_only_for_a_chosen_topics_matter_in_the_closed_list():
    of = render_computed.matter_of
    assert of(("1117", "c.c.")) == ("condominio", False) and of(("1158", "c.c.")) == ("diritti reali", False)
    assert of(("2087", "c.c.")) == ("rito del lavoro", True) and of(("2043", "c.c.")) == ("fatto illecito", True)
    assert of(("640", "c.p.")) == ("reati", True) and of(("1168", "c.c."))[1]               # possessory: never
    assert of(("1", "D.Lgs. 9 aprile 2008, n. 81 (sicurezza sul lavoro)")) == ("rito del lavoro", True)
    assert of(("768-bis", "c.c.")) == ("patti di famiglia", False) and of(("1490", "c.c.")) == ("", False)
    case_first = {"store_offences_only": True, "fact_signals": {"mediation": {"required": True, "subject": "x",
                                                                              "text": "dai fatti"}}}
    for refs in ([["2087", "c.c."]], [["2043", "c.c."], ["640", "c.p."]], []):
        assert not render_computed.mediation_of({**case_first, "topic_refs": refs})["required"]   # the facts never
    listed = render_computed.mediation_of({**case_first, "topic_refs": [["2087", "c.c."], ["1117", "c.c."]]})
    assert listed["required"] and listed["subject"] == "condominio" and "art. 5, co. 1, d.lgs. 28/2010" in listed["text"]
    assert render_computed.mediation_of({"fact_signals": {"mediation": {"required": True, "text": "dai fatti"}}})[
        "text"] == "dai fatti"                                                                 # the legacy path as before


def test_the_negotiation_keeps_its_own_list():
    payment = {"store_offences_only": True, "topic_refs": [["1218", "c.c."]], "fact_signals": {"facts": {
        "requests": [{"what": "pagamento di 3.000 euro"}], "contracts": [{"what": "fornitura"}],
        "parties": [{"name": "Alfa S.r.l.", "role": "controparte"}]}}}
    assert render_computed.negotiation_of(payment)["required"]
    consumer = {**payment, "fact_signals": {"facts": {**payment["fact_signals"]["facts"],
                                                       "parties": [{"name": "Mario", "role": "cliente, consumatore"}]}}}
    assert render_computed.negotiation_of(consumer) == {"required": False, "text": "",
                                                        "why": "esclusa: contratto tra professionista e consumatore"}
    labour = {**payment, "topic_refs": [["2087", "c.c."]]}
    assert render_computed.negotiation_of(labour)["why"] == "esclusa: rito del lavoro"
    assert render_computed.negotiation_of({**payment, "topic_refs": [["640", "c.p."]]})["why"] == "esclusa: reati"


# --- 3. a term needs its triggering fact -------------------------------------------------------------------------------

FRAUD = ("Ho acquistato online uno smartphone a 400 euro pagando con bonifico. Il telefono non è mai arrivato e il "
         "venditore non risponde. Ho ricevuto solo una mail di conferma.")


def test_a_term_is_neither_computed_nor_rendered_without_its_triggering_fact():
    terms = [{"id": "denuncia_conformita_consumo"}, {"id": "azione_vizi_vendita"}, {"id": "prescrizione_ordinaria"},
             {"id": "ricorso_tar"}, {"id": "a_term_with_no_rule"}]
    kept, dropped = fact_signals.untriggered(terms, FRAUD)
    assert [t["id"] for t in kept] == ["prescrizione_ordinaria", "a_term_with_no_rule"]
    assert {d["id"]: d["fact"] for d in dropped}["denuncia_conformita_consumo"] == "un bene consegnato"   # never arrived
    kept, _ = fact_signals.untriggered(terms[:2], "Il telefono mi è stato consegnato il 3 marzo ma non si accende.")
    assert len(kept) == 2 and kept[0]["trigger"].startswith("Il telefono mi è stato consegnato")
    assert fact_signals.trigger_sentence("ricorso_tar", "Il Comune mi ha notificato il diniego il 3 marzo.")
    # possession: the use of a way or a thing over time, not only the word «possesso»
    for said in ("Leonardo passa sul fondo di Domitilla.", "La famiglia transita sulla strada.", "Uso il sentiero da anni.",
                 "Utilizza il magazzino da tempo.", "Coltiviamo quel campo da sempre."):
        assert fact_signals.trigger_sentence("usucapione_ordinaria", said), said
    for said in ("Il cliente non usa più la strada.", "L'anno passato ho venduto la casa.", "Mario è proprietario."):
        assert fact_signals.trigger_sentence("usucapione_ordinaria", said) == "", said


def test_signals_drop_an_untriggered_term_before_the_rules_and_the_flow_logs_it(monkeypatch):
    facts = {"contracts": [{"type": "compravendita di uno smartphone"}], "present_date": "10 novembre 2026",
             "events": [{"date": "3 marzo 2026", "what": "acquisto dello smartphone", "kind": "vendita"},
                        {"date": "10 marzo 2026", "what": "scoperta del difetto", "kind": "scoperta vizio"}],
             "parties": [{"name": "Mario", "role": "acquirente"}]}
    plain = fact_signals.signals_from(facts, areas=["civile"])
    assert "terms_untriggered" not in plain and any(t["id"] == "azione_vizi_vendita" for t in plain["terms"])
    strict = fact_signals.signals_from(facts, areas=["civile"], case=FRAUD)
    assert not any(t["id"] == "azione_vizi_vendita" for t in strict["terms"])
    assert "azione_vizi_vendita" in [d["id"] for d in strict["terms_untriggered"]]
    seen = []
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: seen.append((event, data)))
    monkeypatch.setattr(flow, "case_first_enabled", lambda: True)
    analysis = {"frame": {"areas": ["civile"]}, "fact_signals": {"facts": facts, "terms": plain["terms"]}}
    dropped = flow.trigger_terms(analysis, FRAUD)
    assert "azione_vizi_vendita" in [d["id"] for d in dropped]
    line = dict(seen)["terms_untriggered"]
    assert line["dropped"] == dropped and all("trigger" in k for k in line["kept"])


# --- 4. the first read's channels -------------------------------------------------------------------------------------

def test_the_text_and_index_channels_search_when_the_rubrica_misses():
    calls = []

    def resolve(session, documents, istituto):
        calls.append(("rubrica", istituto["nome"]))
        return [], {}

    def text_search(session, documents, query, code, nouns):
        calls.append(("testo", query))
        return ([{"_article": ("2087", "c.c."), "s": {"titolo_articolo": "Tutela delle condizioni di lavoro"}}]
                if "sicurezza" in query else []), {}

    def index_search(session, documents, query, code, nouns):
        calls.append(("indice", query))
        return [{"_article": ("1117", "c.c."), "s": {"titolo_articolo": "Parti comuni dell'edificio"}}], {}
    rows, log = flow.first_read_search(None, "db", [], [{"name": "obbligo di sicurezza", "rubricated": "", "claim": ""},
                                                         {"name": "parti comuni", "rubricated": "", "claim": ""}],
                                       ["c.c."], session=object(), resolve=resolve, text_search=text_search,
                                       index_search=index_search)
    assert {r["_first_read"]: r["_first_read_channel"] for r in rows} == {"obbligo di sicurezza": "testo",
                                                                           "parti comuni": "indice"}
    assert log["channels"] == {"obbligo di sicurezza": {"2087 c.c.": "testo"}, "parti comuni": {"1117 c.c.": "indice"}}
    assert ("indice", "obbligo di sicurezza") not in calls                     # the text found it: no third channel


def test_the_channels_keep_only_articles_holding_the_words(monkeypatch):
    docs = [{"id": "cc", "name": "Codice Civile 2026"}]
    texts = {"2087": "L'imprenditore è tenuto ad adottare le misure necessarie a tutelare l'integrità fisica dei "
                     "prestatori di lavoro.", "1": "Le leggi entrano in vigore."}
    monkeypatch.setattr(fact_resolver, "_rows_for", lambda session, doc, code, numbers: [
        {"_article": (n, code), "s": {"plain_text": texts.get(n, "")}} for n in numbers])

    class Session:
        def run(self, query, **kw):
            return type("R", (), {"data": lambda self: [{"name": "1"}, {"name": "2087.0.0"}]})()
    rows, log = fact_resolver.text_channel(Session(), docs, "tutela integrità fisica prestatori", "c.c.")
    assert [r["_article"][0] for r in rows] == ["2087"] and log["kept"] == ["2087 c.c."]
    index = [{"name": "Delle servitù prediali", "rubricated": "Delle servitù prediali", "level": "TITOLO",
              "articles": ["1027", "1028", "1051"], "rubriche": {"1051": "Passaggio coattivo"}},
             {"name": "Del passaggio coattivo", "rubricated": "Del passaggio coattivo", "level": "SEZIONE",
              "articles": ["1051", "1052"], "rubriche": {"1051": "Passaggio coattivo", "1052": "Passaggio coattivo a "
                                                                                              "favore di fondo non intercluso"}}]
    rows, log = fact_resolver.index_channel(None, docs, "passaggio coattivo", "c.c.", index=index)
    assert log["unit"] == "Del passaggio coattivo" and [r["_article"][0] for r in rows] == ["1051", "1052"]
    rows, log = fact_resolver.index_channel(None, docs, "passaggio coattivo", "c.c.", index=None)
    assert not rows and "no index" in log["why"]                     # not built: the channel is skipped, said so
    assert fact_resolver.code_slug("c.p.p.") == "cpp" and fact_resolver.index_path("c.c.").endswith("institutions_cc.json")


def test_the_index_is_built_from_the_stores_units():
    build = _script("build_institutions")
    from src.rag import article_text, topic_groups
    from src.rag.rule_quote import clean_rubrica
    helpers = (article_text.article_number, topic_groups.unit_title, topic_groups.generic, clean_rubrica)
    rows = [{"uid": "u1", "level": "CAPO", "unit": "CAPO II – Della vendita", "name": "1490.0.0", "rubrica": "Garanzia "
             "per i vizi della cosa venduta"},
            {"uid": "u1", "level": "CAPO", "unit": "CAPO II – Della vendita", "name": "1470", "rubrica": "Nozione"},
            {"uid": "u1", "level": "CAPO", "unit": "CAPO II – Della vendita", "name": "1490.2", "rubrica": "x"},
            {"uid": "u2", "level": "CAPO", "unit": "CAPO I – Disposizioni generali", "name": "1321", "rubrica": "Nozione"}]
    out = build.entries(rows, *helpers)
    assert [(e["name"], e["level"], e["articles"]) for e in out] == [("Vendita", "CAPO", ["1470", "1490"])]
    assert out[0]["rubricated"] == "Della vendita" and out[0]["rubriche"]["1490"].startswith("Garanzia")
    assert build.level_of("Chapter") == "CAPO" and build.level_of("titolo") == "TITOLO" and build.level_of("") == "?"


def test_a_code_without_units_is_built_from_what_it_has():
    build = _script("build_institutions")
    from src.rag import article_text, citations, topic_groups
    from src.rag.rule_quote import clean_rubrica
    helpers = (article_text.article_number, topic_groups.unit_title, topic_groups.generic, clean_rubrica)
    docs = [{"id": "a", "name": "Codice del Processo Amministrativo 2026"}, {"id": "b", "name": "Codice Civile 2026"}]
    matched, how = build.code_documents(docs, "c.p.a.", citations.code_key, "Codice del processo amministrativo")
    assert [d["id"] for d in matched] == ["a"]
    # a structural property on the articles, or a breadcrumb path
    values = [{"name": "29", "rubrica": "Azione di annullamento", "value": "Capo II – Azioni di cognizione"},
              {"name": "30", "rubrica": "Azione di condanna", "value": "Capo II – Azioni di cognizione"}]
    out = build.entries(build.property_rows(values, "capo"), *helpers)
    assert [(e["name"], e["level"], e["articles"]) for e in out] == [("Azioni di cognizione", "CAPO", ["29", "30"])]
    crumbs = [{"name": "29", "rubrica": "Azione di annullamento",
               "value": "Libro II > Titolo III – Ricorso giurisdizionale > Capo II – Azioni di cognizione"}]
    assert {e["level"] for e in build.entries(build.property_rows(crumbs, "breadcrumb"), *helpers)} == {"TITOLO", "CAPO"}
    # heading Sections in the NEXT chain: each article under the last heading of each level
    chain = [{"sid": "1", "name": "T3", "title": "Titolo III – Azioni di cognizione", "rubrica": None, "next": "2"},
             {"sid": "2", "name": "C1", "title": "Capo I – Azione avverso il silenzio", "rubrica": None, "next": "3"},
             {"sid": "3", "name": "29", "title": "", "rubrica": "Azione di annullamento", "next": "4"},
             {"sid": "4", "name": "C2", "title": "Capo II – Azione di condanna", "rubrica": None, "next": "5"},
             {"sid": "5", "name": "30", "title": "", "rubrica": "Azione di condanna", "next": None}]
    out = {(e["level"], e["name"]): e["articles"] for e in build.entries(build.chain_rows(chain), *helpers)}
    assert out == {("TITOLO", "Azioni di cognizione"): ["29", "30"], ("CAPO", "Azione avverso il silenzio"): ["29"],
                   ("CAPO", "Azione di condanna"): ["30"]}
    # nothing else: each article its own institution, by its rubrica
    single = build.entries(build.rubrica_rows([{"name": "29", "rubrica": "Azione di annullamento"}]),
                           article_text.article_number, lambda u: clean_rubrica(u), lambda n: False, clean_rubrica)
    assert single == [{"name": "Azione di annullamento", "rubricated": "Azione di annullamento", "level": "ARTICOLO",
                       "articles": ["29"], "rubriche": {"29": "Azione di annullamento"}}]


def test_the_cpa_units_article_links_writes_build_the_index_from_units():
    import test_article_links as tal
    build = _script("build_institutions")
    from src.rag import article_text, topic_groups
    from src.rag.rule_quote import clean_rubrica
    result = tal.al.analyse([tal._cpa_doc()])[0]
    rows = tal.al.graph_rows([result])
    units = {(u["level"], u["path"]): u for u in rows["units"]}
    parents = {(c["level"], c["path"]): (c["parent_level"], c["parent_path"]) for c in rows["child_of"]}
    found = []                     # what UNITS_QUERY returns once written: each article's unit and the units above it
    for h in rows["has_article"]:
        key = (h["level"], h["path"])
        article = next(a for a, ids in result["pieces"].items() if h["section"] in ids)
        while key:
            found.append({"uid": key[1], "level": key[0], "unit": units[key]["name"], "name": article,
                          "rubrica": f"Rubrica {article}"})
            key = parents.get(key)
    out = build.entries(found, article_text.article_number, topic_groups.unit_title, topic_groups.generic, clean_rubrica)
    by = {(e["level"], e["name"]): e["articles"] for e in out}
    assert by[("TITOLO", "Principi e organi della giurisdizione amministrativa")] == ["1", "2", "4"]
    assert by[("CAPO", "Organi della giurisdizione amministrativa")] == ["4"]
    assert by[("TITOLO", "Azioni di cognizione")] == ["29", "30"]
    assert ("LIBRO", "Disposizioni generali") not in by                       # a generic name: left out


def test_each_anchor_logs_the_channel_that_found_it(monkeypatch):
    seen = []
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: seen.append((event, data)))
    rows = [{"_article": ("2087", "c.c."), "_first_read": "sicurezza", "_first_read_channel": "testo"},
            {"_article": ("2043", "c.c.")}, {"_article": ("1117", "c.c."), "_first_read": "parti comuni"}]
    out = flow.log_anchor_channels([{"title": "A", "anchor": "2087 c.c."}, {"title": "B", "anchor": "2043 c.c."},
                                    {"title": "C", "anchor": "1117 c.c."}, {"title": "D", "anchor": "9 c.c."}], rows)
    assert [x["channel"] for x in out] == ["testo", "ricerca", "rubrica", "fuori dal pool"]
    assert dict(seen)["anchor_channels"]["by_channel"] == {"testo": 1, "ricerca": 1, "rubrica": 1, "fuori dal pool": 1}


# --- 5. the record's strictness --------------------------------------------------------------------------------------

def test_provato_needs_a_supporting_sentence_holding_the_elements_nouns():
    by_id = {"S1": {"id": "S1", "certainty": "accertato", "text": "Il datore non aveva fornito i dispositivi di "
                                                                   "protezione."},
             "S2": {"id": "S2", "certainty": "accertato", "text": "Il lavoratore è caduto dal ponteggio."}}
    req = {"kind": "requisito", "text": "il datore di lavoro ha omesso di fornire i dispositivi di protezione",
           "facts_status": "da provare", "supporting": ["S1"], "contradicting": []}
    issue_records.settle(req, by_id)
    assert req["status"] == "provato"
    req = {**req, "supporting": ["S2"], "status": None}
    issue_records.settle(req, by_id)
    assert req["status"] == "da provare" and "sostantivi dell'elemento" in req["why"]


def test_a_distinctive_element_that_does_not_hold_makes_the_article_not_applicable():
    rec = {"requisites": [{"kind": "requisito", "comma": 1, "status": "non risulta", "text": "violenza", "distinctive": True},
                          {"kind": "requisito", "comma": 2, "status": "provato", "text": "altro"}]}
    assert issue_records.verdict(rec) == issue_records.NOT_APPLICABLE
    assert issue_records.verdict_reason(rec).startswith("elemento distintivo «violenza» non risulta")
    rec["requisites"][0]["distinctive"] = False
    assert issue_records.verdict(rec) == issue_records.TO_PROVE           # comma 2 holds: as before
    row = {"_distinctive_text": "violenza"}
    seeded = {"requisites": [{"kind": "requisito", "text": "violenza"}, {"kind": "requisito", "text": "altro"}]}
    issue_records.seed(seeded, row)
    assert [r.get("distinctive") for r in seeded["requisites"]] == [True, None]


def test_the_demotion_runs_to_completion_and_logs_what_it_left_unassessed(monkeypatch):
    rows = []
    for n in range(1, 14):
        rows.append({"_article": (str(n), "c.c."), "_grade": 2,
                     "s": {"plain_text": f"{n}. Il possessore del bene numero {n} acquista la proprietà del fondo."}})
    monkeypatch.setattr(issue_records, "distinctive", lambda rows, analysis: {
        issue_records._label(r["_article"]): {"comma": 1, "text": f"elemento {i}", "kind": "requisito"}
        for i, r in enumerate(rows)})
    sizes = []

    def call(topic, reqs, sents):
        sizes.append(len(reqs))
        if len(sizes) == 2:
            time.sleep(0.4)
        return {r["id"]: {"sostiene": ["S1"]} for _, r in reqs[1:]}            # the first of each call unanswered
    log = issue_records.demote_unsupported(rows, "Il possessore usa il fondo da anni.", {}, call=call, seconds=0.2)
    assert issue_records.DISTINCTIVE_PER_CALL == 6 and sorted(sizes) == [1, 6, 6]
    assert not log["complete"] and len(log["unassessed"]) >= 6 and log["errors"]
    assert all(r.get("_distinctive_text") for r in rows)                       # every element known to its record


def test_a_non_applicable_article_is_one_line_never_a_block():
    text = ("**Art. 1062 c.c.**\n- co. 1, requisito «fondi divisi»: non risulta.\n\n**Art. 1061 c.c.**\n"
            "- testo: da provare.\n\nLe opere sono visibili.")
    out = flow.norms_final(text, [], {"1062 c.c.": issue_records.NOT_APPLICABLE, "1061 c.c.": "da provare"})
    assert "**Art. 1062" not in out and "fondi divisi" not in out and "**Art. 1061 c.c.**" in out
    assert out.endswith(C.fill(C.NOT_APPLICABLE_LINE, articles="art. 1062 c.c"))
    prose = text.replace("non risulta.\n\n", "non risulta.\n\nLa servitù sorge per destinazione.\n\n")
    assert "**Art. 1062 c.c.**" in flow.norms_final(prose, [], {"1062 c.c.": issue_records.NOT_APPLICABLE})   # check first
    ctx = {"records": {"1062 c.c.": {"requisites": [{"kind": "requisito", "status": "non risulta", "text": "x"}],
                                     "verdict": issue_records.NOT_APPLICABLE}},
           "allowed": {"1062 c.c."}, "glossed": set(), "texts": {}, "rubriche": {}, "commi": {}, "documents": [],
           "client": "", "skip": set(), "norms_title": NORMS, "numbering": []}
    sections, log = record_check.check([(NORMS, prose)], ctx)
    assert [x["rule"] for x in log["removed"]] == ["norma_non_applicabile"] and "per destinazione" not in sections[0][1]


def test_the_answer_lists_a_non_applicable_article_in_one_line_and_keeps_the_invariants(monkeypatch):
    state = _graded_state(monkeypatch)
    rec = state["records"]["1062 c.c."]
    rec["requisites"][0]["distinctive"] = True                               # round 1's element of 1062
    text, _ = _answer(monkeypatch, state, WRITER, {"R1": {"contraddice": ["S2"]}})
    assert state["records"]["1062 c.c."]["verdict"] == issue_records.NOT_APPLICABLE
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "**Art. 1062" not in norms and C.NOT_APPLICABLE_LINE.split("{")[0] + "art. 1062 c.c." in norms
    ledger = state["ledger"]
    assert ledger["unaccounted"] == [] and ledger["unreasoned"] == []                   # invariant 1
    removed = [e for e in ledger["entries"] if e["kind"] == "removed"]
    assert all(e["step"] == "record_check" and e["rule"] in record_check.REMOVE_RULES for e in removed)
    buckets = {label: r["answer"] for label, r in state["records"].items() if r.get("answer")}
    assert buckets["1062 c.c."]["bucket"] == flow.BUCKET_NOT_APPLICABLE                 # invariant 3
    assert buckets["1062 c.c."]["reason"].startswith("elemento distintivo")


# --- 6. a sanction article is a line under the obligation it sanctions ------------------------------------------------

LAW = "d.lgs. 81/2008"


def _law_rows():
    return [{"_article": ("18", LAW), "s": {"titolo_articolo": "Obblighi del datore di lavoro e del dirigente",
                                             "plain_text": "1. Il datore di lavoro deve fornire ai lavoratori i "
                                                           "necessari e idonei dispositivi di protezione individuale."}},
            {"_article": ("55", LAW), "s": {"titolo_articolo": "Sanzioni per il datore di lavoro e il dirigente",
                                             "plain_text": "1. È punito con l'arresto da tre a sei mesi il datore di "
                                                           "lavoro per la violazione dell'articolo 18, comma 1."}},
            {"_article": ("37", LAW), "s": {"titolo_articolo": "Formazione dei lavoratori",
                                             "plain_text": "1. Il datore di lavoro assicura che ciascun lavoratore "
                                                           "riceva una formazione sufficiente."}}]


def test_a_sanction_article_is_recognised_and_mapped_to_its_obligation():
    rows = _law_rows()
    assert flow.is_sanction(rows[1]) and not flow.is_sanction(rows[0])
    by_text = {"_article": ("59", LAW), "s": {"titolo_articolo": "Lavoratori", "plain_text":
               "1. I lavoratori sono puniti con l'arresto fino a un mese per la violazione dell'articolo 20, comma 2. "
               "2. È soggetto alla sanzione amministrativa il lavoratore che viola l'articolo 21."}}
    assert flow.is_sanction(by_text) and flow.sanctioned(by_text) == ["20", "21"]
    offence = {"_article": ("590", "c.p."), "s": {"titolo_articolo": "Lesioni personali colpose", "plain_text":
               "Chiunque cagiona ad altri per colpa una lesione è punito con la reclusione, ai sensi dell'articolo 43."}}
    assert not flow.is_sanction(offence)                          # an offence carries its own pena
    civil_offence = {"_article": ("2621", "c.c."), "s": {"titolo_articolo": "False comunicazioni sociali", "plain_text":
                     "Fuori dai casi previsti dall'art. 2622, gli amministratori sono puniti con la pena della "
                     "reclusione da uno a cinque anni."}}
    assert not flow.is_sanction(civil_offence)                    # cites an article, sanctions no obligation
    refs = [("18", LAW), ("55", LAW), ("37", LAW)]
    assert flow.sanction_map(refs, rows) == {("55", LAW): ("18", LAW)}
    assert flow.sanction_map([("55", LAW), ("37", LAW)], rows) == {("55", LAW): None}


def test_a_sanction_gets_a_line_under_its_obligations_block_or_the_untreated_line():
    rows = _law_rows()
    rules = [{"ref": r["_article"], "commi": [(1, r["s"]["plain_text"][3:])], "selected": [1], "requisites": {}}
             for r in rows]
    refs = [("18", LAW), ("55", LAW), ("37", LAW)]
    text = "[1] Il datore doveva fornire i dispositivi.\n\n[2] La formazione era insufficiente."
    log = []
    out = flow.assemble_norms(text, rules, {}, rows, log, refs, glossed=[("18", LAW), ("37", LAW)], keep=True,
                              sanctions=flow.sanction_map(refs, rows))
    block_18 = out.split("**Art. 37")[0]
    assert "**Art. 55" not in out and "- Sanzione: art. 55 " in block_18 and "Sanzioni per il datore" in block_18
    assert log[-1]["sanctions"] == {f"55 {LAW}": f"18 {LAW}"}
    alone = flow.assemble_norms(text, rules, {}, rows, [], [("55", LAW), ("37", LAW)], glossed=[("37", LAW)],
                                keep=True, sanctions={("55", LAW): None})
    assert "**Art. 55" not in alone and C.OTHER_NORMS_LINE.split("{")[0] in alone and "art. 55" in alone
    buckets = flow.bucket_articles(refs, [("18", LAW), ("37", LAW)], f"**2. {NORMS}**\n\n{out}", None,
                                   sanctions=flow.sanction_map(refs, rows))
    assert buckets[f"55 {LAW}"]["bucket"] == flow.BUCKET_WRITTEN and "riga sotto art. 18" in buckets[f"55 {LAW}"]["reason"]


# --- 7. relative dates, accent-insensitive ----------------------------------------------------------------------------

def test_relative_dates_are_read_whatever_the_accents_and_apostrophes():
    today = __import__("datetime").date(2026, 11, 10)
    assert fact_signals.relative_date("3 giorni fà", today) == "7 novembre 2026"
    assert fact_signals.relative_date("un’ ora fa", today) == "10 novembre 2026"
    assert fact_signals.relative_date("ieri sera", today) == "9 novembre 2026"
    assert fact_signals.relative_date("circa tre mesi fà", today).startswith("periodo dal")
    assert fact_signals.relative_date("più di due anni fa", today) == fact_signals.relative_date("piu di due anni fa", today)
    assert fact_signals.relative_date("l’ anno scorso", today) == "2025"
    assert fact_signals.relative_text("Un' ORA fà, lunedì") == "un ora fa, lunedi"


# --- 8. the restatement names facts --------------------------------------------------------------------------------------

def test_a_yes_no_answer_is_restated_as_the_fact_it_settles():
    read = lambda point: {"kind": "free", "template": "lettura", "point": point}
    assert flow.stated_fact(read("se il lavoratore abbia firmato il documento di formazione"), "sì") == \
        "il lavoratore ha firmato il documento di formazione"
    assert flow.stated_fact(read("se l'azienda abbia effettuato controlli periodici"), "no, mai") == \
        "l'azienda non ha effettuato controlli periodici"
    assert flow.stated_fact(read("se il datore si fosse accorto del pericolo"), "no") == \
        "il datore non si era accorto del pericolo"
    assert flow.stated_fact(read("la data dell'infortunio"), "sì") == "" and flow.stated_fact(read("se x"), "forse") == ""
    assert flow.stated_fact({"template": "accesso_alternativo"}, "no") == C.RESTATE_YES_NO["accesso_alternativo"][1]


def test_the_restatement_and_the_precisazione_carry_the_fact_never_the_question(monkeypatch):
    state = base._state(monkeypatch)
    q = {"id": "read_1", "family": "lettura", "template": "lettura", "kind": "free", "target": None,
         "text": "Il caso non dice se l'azienda abbia effettuato controlli periodici: puoi precisarlo?",
         "short": "controlli periodici", "point": "se l'azienda abbia effettuato controlli periodici"}
    state["questions"] = list(state["questions"]) + [q]
    flow.apply_reply(state, {"answers": {"read_1": "no"}})
    said = flow.restatement(state)
    assert "l'azienda non ha effettuato controlli periodici" in said and "Il caso non dice" not in said
    assert "L'azienda non ha effettuato controlli periodici." in state["additions"]


# --- 9. roles are not names ----------------------------------------------------------------------------------------------

def test_a_party_named_by_role_reads_as_a_role_in_every_string():
    assert party_names.label("Lavoratore") == "il lavoratore" and party_names.label("Acquirente") == "l'acquirente"
    assert party_names.label("Società") == "la società" and party_names.label("Datore di lavoro") == "il datore di lavoro"
    assert party_names.label("Società Alfa") == "Società Alfa" and party_names.label("Mario Rossi") == "Mario Rossi"
    assert party_names.contract("Ho letto il caso di il lavoratore, a l'acquirente.") == \
        "Ho letto il caso del lavoratore, all'acquirente."
    text = ("Lavoratore ha firmato. La colpa di Datore di lavoro è piena; il Lavoratore chiede a Società il danno. "
            "Società Alfa no.")
    roles = party_names.role_names([{"name": n} for n in ("Lavoratore", "Datore di lavoro", "Società", "Società Alfa")])
    assert party_names.lower_roles(text, roles) == ("Il lavoratore ha firmato. La colpa del datore di lavoro è piena; il "
                                                    "lavoratore chiede alla società il danno. Società Alfa no.")
    heading = "**Art. 2 d.lgs. 81/2008 — Lavoratore e datore di lavoro**\nLavoratore ha firmato."
    assert party_names.lower_roles(heading, roles) == ("**Art. 2 d.lgs. 81/2008 — Lavoratore e datore di lavoro**\n"
                                                       "Il lavoratore ha firmato.")
    facts = {"parties": [{"name": "Lavoratore", "role": ""}, {"name": "Società", "role": ""}]}
    q = flow.client_question(facts)
    assert q and q[0]["text"] == "Chi assisti in questo caso: il lavoratore o la società?" and q[0]["names"] == [
        "Lavoratore", "Società"]                                   # the answer still matched on the names
    analysis = {"fact_signals": {"facts": facts}}
    assert render_computed._named("Parti da convenire: Società (datore).", analysis) == \
        "Parti da convenire: la società (datore)."
    # a trade the role list does not hold: a role when the case writes it in lower case; a surname never is
    case = "La signora Bianchi chiama un idraulico. Il produttore della caldaia non risponde. Idraulico e Bianchi litigano."
    many = {"parties": [{"name": "Signora Bianchi", "role": ""}, {"name": "Idraulico", "role": ""},
                        {"name": "Produttore", "role": ""}]}
    assert flow.client_question(many, (), case)[0]["text"] == \
        "Chi assisti in questo caso: Signora Bianchi, l'idraulico o il produttore?"
    assert not party_names.is_role("Bianchi", case) and party_names.common_in("idraulico", case)


def test_round_one_and_the_restatement_open_with_the_role(monkeypatch):
    state = base._state(monkeypatch)
    state["analysis"]["fact_signals"]["facts"]["parties"] = [{"name": "Lavoratore", "role": "cliente"},
                                                             {"name": "Società", "role": "controparte"}]
    state["analysis"]["fact_signals"]["facts"]["client_side"] = "Lavoratore"
    state["analysis"]["parte_assistita"] = "Lavoratore"
    text, _ = flow.round1(state)
    assert text.startswith("Ho letto il caso del lavoratore.") and "il lavoratore (cliente)" in text
    assert flow.restatement(state).startswith("Imposto il parere per il lavoratore.")


# --- 10. the copy export ---------------------------------------------------------------------------------------------------

def test_the_copy_review_is_generated_and_in_sync(tmp_path):
    export = _script("copy_export")
    out = tmp_path / "copy.md"
    export.main(["--out", str(out)])
    assert out.read_text(encoding="utf-8") == (HERE.parent / "docs" / "copy_review.md").read_text(encoding="utf-8")
    rows = export.copy_rows()
    keys = {k for k, _, _ in rows}
    assert {"PROCEDURE_CONDITIONAL", "SANCTION_LINE", "RESTATE_YES_NO[accesso_alternativo]#2", "CLIENT_ASSUMED",
            "QUESTION_FROM_READ"} <= keys
    assert "ROUND1_OPENING" not in keys and "QUESTIONS[atto_notifica]" not in keys          # the PO's own texts
    assert all(w for _, _, w in rows) and "| key | text | where it appears |" in out.read_text(encoding="utf-8")


# --- 11. the analysis stage's calls ------------------------------------------------------------------------------------------

def test_each_chat_call_records_its_seconds_and_tokens_on_the_stage_meter(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    from src.rag import ai_chat

    class Model:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(content="ok", usage_metadata={"input_tokens": 1200, "output_tokens": 300,
                                                           "total_tokens": 1500})
    monkeypatch.setattr(ai_chat, "chat_model", Model())
    meter = stage_meter.Meter()
    with meter.active():
        meter.mark("analysis")

        def read_frame():
            return ai_chat._call_chat([HumanMessage(content="x")], max_tokens=10)
        read_frame()
        meter.mark("select")
        ai_chat._call_chat([HumanMessage(content="y")])
    meter.stop()
    log = meter.as_log()
    assert log["analysis_calls"] and log["analysis_calls"][0]["call"] == "read_frame"
    assert log["analysis_calls"][0]["in"] == 1200 and log["analysis_calls"][0]["out"] == 300 and log["analysis_llm"] == 1
    assert log["analysis_tokens_in"] == 1200 and len(meter.calls) == 2


# --- 12. the ingestion report --------------------------------------------------------------------------------------------

def test_the_ingestion_report_classifies_document_names():
    report = _script("ingestion_issues")
    for name in ("Codice Civile 2026", "D.Lgs. 9 aprile 2008, n. 81", "Legge sul Procedimento Amministrativo 2026",
                 "CCNL Commercio 2024", "Regolamento (UE) 2016/679"):
        assert report.NORMATIVE_RE.search(name), name
    for name in ("Cass. civ. sez. III n. 1234/2020", "appunti_cliente_rossi", "Tribunale di Milano 12/03/2021"):
        assert not report.NORMATIVE_RE.search(name), name
    assert report.RULING_RE.search("Cass. civ. sez. III n. 1234/2020") and not report.RULING_RE.search("appunti_rossi")
    from src.rag import article_text
    rows = [{"name": "12.0.0", "id": "a"}, {"name": "12", "id": "b"}, {"name": "12.1", "id": "c"}, {"name": "13", "id": "d"}]
    assert {k: v["id"] for k, v in report._canonical(rows, article_text.article_number).items()} == {"12": "a", "13": "d"}
