"""The parere written from facts described in a chat message (no document).

Oct 2026: a bar-exam style case (pensioner falls in a pothole, defend the Comune)
got an answer that never named art. 2051 c.c., quoted 2043 and 2044 from memory
(wrongly), invented facts and copied the instructions' examples. These tests pin
the pipeline that replaced it: articles read from the database by number, case
law from full-text search, every cited article or decision checked against them.
"""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import src.rag.defensive_generation as DG  # noqa: E402

CASE = ("Tizio, pensionato ottantenne, inciampa in una buca su Via Freguglia nella tarda mattinata e si "
        "frattura una caviglia. Caio dichiara di essere inciampato nella stessa buca la stessa mattina. "
        "Il candidato, legale del Comune, esponga parere sulle ragioni di difesa.")
ART_2051 = "Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, salvo che provi il caso fortuito."
RULING = ("La sentenza impugnata ha correttamente applicato l'art. 2051 cod. civ.: il caso fortuito può consistere "
          "nella condotta del danneggiato, tanto più rilevante quanto più la situazione era prevedibile.")
ANALYSIS = {
    "parte_assistita": "Comune di Milano", "posizione": "convenuto", "area": "civile",
    "richiesta": "ragioni di difesa del Comune", "ricerca_fatti": ["buca", "manto stradale", "caduta"],
    "questioni": [{"titolo": "Responsabilità da cosa in custodia", "norme": ["art. 50 c.c."],
                   "parole_chiave": ["custodia", "buca", "caso fortuito"]}],
    "fatti": [{"fatto": "caduta in tarda mattinata", "effetto": "favorevole", "perche": "piena visibilità"}],
}


# --- Reference checking -----------------------------------------------------

def test_articles_and_decisions_not_in_the_sources_are_flagged():
    sources = "[art. 2051 c.c.]\n" + ART_2051 + "\n[Ordinanza sul ricorso iscritto al n. 23908/2022 R.G.]\n" + RULING
    text = ("Si applica l'art. 2051 c.c. e non l'art. 2044 c.c. (Cass., ord. n. 23908/2022). "
            "Così anche Cass. sez. un. n. 13489/2022, e il d.lgs. 28/2010 e la legge n. 128/2001.")
    out = DG._mark_unverified(text, [("2051", "c.c.")], sources)
    assert "art. 2051 c.c. e" in out                                   # fetched: untouched
    assert "art. 2044 c.c. [DA VERIFICARE]" in out                     # never retrieved
    assert "23908/2022)" in out                                        # in the sources
    assert "13489/2022 [DA VERIFICARE]" in out                         # not in the sources
    assert "28/2010 e" in out and "128/2001." in out                   # laws, not decisions


def test_norms_are_read_with_the_right_code():
    analysis = {"questioni": [{"norme": ["art. 2051 c.c.", "art. 1227, co. 1, c.c.", "art. 2947 cod. civ.",
                                         "art. 163 c.p.c.", "art. 582 c.p.", "art. 415-bis c.p.p.",
                                         "art. 415bis cpp"]}]}
    assert DG._norm_refs(analysis) == [("2051", "c.c."), ("1227", "c.c."), ("2947", "c.c."),
                                       ("163", "c.p.c."), ("582", "c.p."), ("415-bis", "c.p.p.")]


def test_articles_the_rulings_agree_on_become_candidates():
    """The pothole case, Oct 2026: the model named art. 50, 1219, 2667 c.c.; the
    rulings found for the same question cite 2051 and 1227."""
    rulings = [{"d": {"id": "r1"}, "s": {"plain_text": "violazione dell'art. 2051 cod. civ. in relazione all'art. "
                                                       "360 cod. proc. civ.; l'art. 1227, comma 1, c.c.; art. 2052 c.c."}},
               {"d": {"id": "r2"}, "s": {"plain_text": "responsabilità ex art. 2051 cod. civ. e concorso ex "
                                                       "articolo 1227 del codice civile"}}]
    assert DG._cited_by_rulings(rulings) == [("2051", "c.c."), ("1227", "c.c.")]   # 2052: one ruling only

    analysis = {"area": "amministrativo", "questioni": [{"norme": ["art. 2667 c.c.", "art. 50 c.c."]}]}
    facts = "Tizio chiede i danni al Comune, che riceve l'avviso ex art. 415bis cpp."
    keep, candidates = DG._candidate_articles(facts, analysis, rulings, code_refs=[("1495", "c.c.")])
    assert keep == [("415-bis", "c.p.p.")]                             # named in the user's text: always used
    assert candidates[:3] == [("2051", "c.c."), ("1227", "c.c."), ("1495", "c.c.")]   # rulings first
    assert ("2043", "c.c.") in candidates and ("2947", "c.c.") in candidates   # damages backbone, any area but penal
    assert ("360", "c.p.c.") not in candidates                         # Cassazione procedure, not substance
    assert ("2667", "c.c.") in candidates                              # the model's numbers: candidates only


def test_whole_rulings_are_read_and_strong_agreement_is_kept():
    """Run 5, Oct 2026: the facts search matched the passage telling the facts;
    art. 2051 c.c. is cited in the reasoning, elsewhere in the same ruling."""
    class Session:
        def run(self, query, **params):
            texts = {"r1": ["Tizio cadeva in una buca.", "Ai sensi dell'art. 2051 c.c. ... art. 1227 c.c."],
                     "r2": ["responsabilità ex art. 2051 cod. civ.", "art. 360 cod. proc. civ."],
                     "r3": ["violazione dell'articolo 2051 del codice civile; art. 1227 c.c."]}
            return _Result([{"id": d, "text": t} for d in params["ids"] for t in texts.get(d, [])])

    counts = DG._articles_cited_in_rulings(Session(), ["r1", "r2", "r3", "r1", None])
    assert counts == {("2051", "c.c."): 3, ("1227", "c.c."): 2}        # c.p.c. procedure ignored

    keep, candidates = DG._candidate_articles("Il Comune chiede...", {"area": "civile"}, [], [], counts,
                                              similar_counts={("2051", "c.c."): 2})
    assert ("2051", "c.c.") in keep                                    # three rulings agree: always used
    assert ("1227", "c.c.") in candidates and ("1227", "c.c.") not in keep
    # ...but only from the case's own code: run 6 forced art. 2697 c.c. into a criminal case,
    # and only when rulings on similar facts agree: run 7 kept art. 416-bis c.p. for a race crash.
    keep, candidates = DG._candidate_articles(
        "Tizio riceve l'avviso", {"area": "penale"}, [], [],
        {("2697", "c.c."): 4, ("590", "c.p."): 3, ("416-bis", "c.p."): 5},
        similar_counts={("2697", "c.c."): 3, ("590", "c.p."): 2, ("416-bis", "c.p."): 1})
    assert keep == [("590", "c.p.")]
    assert ("2697", "c.c.") in candidates and ("416-bis", "c.p.") in candidates


def test_a_damages_claim_against_a_public_body_counts_as_civil(monkeypatch):
    monkeypatch.setattr(DG, "_call_chat", lambda *a, **k: json.dumps({"area": "amministrativo"}))
    assert DG._analyse_case("Tizio chiede i danni al Comune di Milano.", "it")["area"] == "civile"
    assert DG._analyse_case("Ricorso al TAR contro il diniego del permesso.", "it")["area"] == "amministrativo"


def test_the_model_chooses_among_candidates_with_their_text(monkeypatch):
    seen = {}

    def fake_chat(messages, max_tokens=None, stop=None):
        seen["human"] = messages[1].content
        return '["art. 2051 c.c.", "art. 9999 c.c.", "art. 1227 c.c."]'

    monkeypatch.setattr(DG, "_call_chat", fake_chat)
    rows = [{"_article": ("2051", "c.c."), "s": {"plain_text": ART_2051}},
            {"_article": ("2667", "c.c."), "s": {"plain_text": "Trascrizione delle domande giudiziali ..."}},
            {"_article": ("1227", "c.c."), "s": {"plain_text": "Se il fatto colposo del creditore ..."}}]
    assert DG._choose_articles(CASE, rows) == [("2051", "c.c."), ("1227", "c.c.")]   # 9999: not a candidate
    assert "[art. 2051 c.c.] Ciascuno è responsabile" in seen["human"]
    monkeypatch.setattr(DG, "_call_chat", lambda *a, **k: "non è un JSON")
    assert DG._choose_articles(CASE, rows) is None


def test_rulings_on_similar_facts_are_found_by_the_facts_words():
    """The model framed the pothole claim as "responsabilità amministrativa";
    rulings on falls in a pothole are found by the facts themselves."""
    class Session:
        params = None

        def run(self, query, **params):
            Session.params = params
            return _Result([{"d": {"id": f"r{i}"}, "s": {"plain_text": "..."}} for i in range(8)])

    rows = DG._search_similar_facts(Session(), {"area": "civile", "ricerca_fatti": ["buca", "manto stradale",
                                                                                 "caduta", "pedone"]},
                                    exclude_docs={"r0"})
    assert Session.params == {"t": "buca manto mant* stradale stradal* caduta cadut* pedone pedon*",
                              "types": ["interpretation"]}
    assert [r["d"]["id"] for r in rows] == ["r1", "r2", "r3", "r4"]
    assert DG._search_similar_facts(Session(), {"ricerca_fatti": []}, set()) == []


def test_code_articles_are_found_by_keywords():
    class Session:
        def run(self, query, **params):
            return _Result([{"doc": "Codice Civile 2026", "sec": "1495.0.0", "score": 9.0},
                            {"doc": "Codice di procedura civile Edizione 2026", "sec": "163.0.0", "score": 5.0},
                            {"doc": "Codice Civile 2026", "sec": "Titolo III", "score": 4.0}])

    refs = DG._search_code_articles(Session(), {"questioni": [{"titolo": "vizi della cosa venduta",
                                                                "parole_chiave": ["denunzia"]}]})
    assert refs == [("1495", "c.c."), ("163", "c.p.c.")]


def test_case_law_is_searched_by_keywords_not_by_the_models_numbers():
    class Session:
        terms = []

        def run(self, query, **params):
            Session.terms.append(params["t"])
            return _Result([])

    analysis = {"questioni": [{"titolo": "Custodia della strada", "norme": ["art. 2667 c.c."],
                               "parole_chiave": ["buca", "caso fortuito"]}]}
    DG._search_case_law(Session(), analysis)
    assert Session.terms == ["Custodia custodi* della strada strad* buca caso fortuito fortuit*"]


def test_searches_also_match_other_forms_of_each_word():
    """Run 6: "vizio nascosto" never matched art. 1495 c.c. ("denunzia i vizi")."""
    assert DG._lucene_query("vizio nascosto della cosa, art. 1495") == \
        "vizio vizi* nascosto nascost* della cosa art 1495"


def test_list_fields_given_as_one_string_are_split(monkeypatch):
    """Run 4, Oct 2026: the model returned "parole_chiave" as one comma-separated
    string; iterated as a list it searched the database letter by letter."""
    reply = {"area": "civile", "ricerca_fatti": "buca, manto stradale; caduta",
             "questioni": [{"titolo": "Custodia", "norme": "art. 2051 c.c., art. 1227 c.c.",
                            "parole_chiave": "custodia, caso fortuito"}], "fatti": []}
    monkeypatch.setattr(DG, "_call_chat", lambda *a, **k: json.dumps(reply))
    analysis = DG._analyse_case(CASE, "it")
    assert analysis["ricerca_fatti"] == ["buca", "manto stradale", "caduta"]
    assert analysis["questioni"][0]["parole_chiave"] == ["custodia", "caso fortuito"]
    assert DG._norm_refs(analysis) == [("2051", "c.c."), ("1227", "c.c.")]


def test_search_terms_are_safe_for_the_full_text_index():
    assert DG._lucene_terms('art. 2051 "c.c." (custodia) / caso: fortuito') == "art. 2051 c.c. custodia caso fortuito"


# --- What the writer reads from the rulings ---------------------------------

FACTS_PART = "Tizio conveniva in giudizio il Comune deducendo di essere caduto in una buca del manto stradale."
COSTS_PART = "Le spese seguono la soccombenza e si liquidano come in dispositivo."


def test_the_writer_reads_the_reasoning_not_the_passage_the_search_matched():
    """Oct 2026: the search on the facts landed on the passage telling the facts;
    the principles on art. 2051 c.c. are in the reasoning, further on."""
    terms = DG._passage_terms({"ricerca_fatti": ["buca"],
                               "questioni": [{"parole_chiave": ["custodia della strada", "caso fortuito"]}]})
    assert terms == [("buca",), ("custodi", "strad"), ("caso", "fortuit")]
    sections = [FACTS_PART, COSTS_PART, RULING, "Sulle strade in custodia dell'ente grava una responsabilità oggettiva."]
    passage = DG._best_passage(sections, [("2051", "c.c.")], terms, 2500)
    assert passage == " […] ".join([FACTS_PART, RULING, sections[3]])   # reading order, costs left out
    # Little room: the article citation and the concepts first.
    assert DG._best_passage(sections, [("2051", "c.c.")], terms, 400) == RULING
    assert DG._best_passage([COSTS_PART], [("2051", "c.c.")], terms, 2500) == ""


def test_a_long_section_is_cut_around_the_chosen_article():
    long_text = "Premessa sul giudizio. " * 200 + "Nel merito. " + RULING + " Altro ancora." * 200
    out = DG._window(long_text, {("2051", "c.c.")}, [], 1000)
    assert out.startswith("… ") and "art. 2051 cod. civ." in out and len(out) <= 1002


def test_sections_are_read_in_document_order():
    class Session:
        def run(self, query, **params):
            return _Result([{"id": "r1", "name": "10_1", "text": "dieci"}, {"id": "r1", "name": "2_3", "text": "due"},
                            {"id": "r2", "name": "1_1", "text": " "}, {"id": "r1", "name": "2_1", "text": "uno"}])

    assert DG._ruling_sections(Session(), ["r1", "r2", "r1"]) == {"r1": ["uno", "due", "dieci"]}


def test_every_legal_question_gets_a_ruling_among_the_first_read():
    def rows(*specs):
        return [{"d": {"id": doc}, **({"_question": q} if q is not None else {})} for doc, q in specs]

    similar = rows(("s1", None), ("s2", None), ("s3", None), ("s4", None))
    second = rows(("a", 0), ("b", 0), ("c", 1))
    first = rows(("d", 0), ("e", 2), ("a", 0))
    order = [r["d"]["id"] for r in DG._reading_order(similar, second, first)]
    assert order == ["s1", "s2", "a", "c", "e", "s3", "s4", "b", "d"]


def test_a_civil_claim_against_a_comune_is_not_called_administrative_liability():
    """Run 10: the analysis titled the pothole claim "Responsabilità amministrativa"
    and the parere used it as a heading."""
    analysis = {"area": "civile", "questioni": [{"titolo": "Responsabilità amministrativa per manutenzione strade"},
                                                {"titolo": "Danno erariale e responsabilità amministrativa"}]}
    assert DG._issue_titles(analysis) == ("1. Responsabilità civile della P.A. per manutenzione strade\n"
                                          "2. Danno erariale e responsabilità amministrativa")
    assert DG._civil_title("QUESTIONE 1: responsabilità amministrativa", "civile") == \
        "QUESTIONE 1: responsabilità civile della P.A."
    assert DG._civil_title("Responsabilità amministrativa", "amministrativo") == "Responsabilità amministrativa"


def test_only_rulings_of_the_cases_own_branch_are_read():
    """Run 9: a race crash (criminal) got a civil ruling on a pothole as its authority."""
    class Session:
        def run(self, query, **params):
            texts = {"civil": ["Violazione dell'art. 2051 c.c. e dell'art. 360 c.p.c.: la buca era visibile."],
                     "penal": ["Lesioni colpose ex art. 590 c.p.; ricorso ex art. 606 c.p.p. in gara sportiva."]}
            return _Result([{"id": d, "name": "1_1", "text": t} for d in params["ids"] for t in texts.get(d, [])])

    rows = [{"d": {"id": "civil"}, "s": {"plain_text": "buca"}}, {"d": {"id": "penal"}, "s": {"plain_text": "gara"}},
            {"d": {"id": "unread"}, "s": {"plain_text": "Massima senza articoli."}}]
    penal = DG._ruling_passages(Session(), rows, [("590", "c.p.")], {"area": "penale", "ricerca_fatti": ["gara"]})
    assert [r["d"]["id"] for r in penal] == ["penal", "unread"]
    assert penal[0]["_passage"].startswith("Lesioni colpose") and "_passage" not in penal[1]
    civil = DG._ruling_passages(Session(), rows, [("2051", "c.c.")], {"area": "civile", "ricerca_fatti": ["buca"]})
    assert [r["d"]["id"] for r in civil] == ["civil", "unread"]


def test_principles_are_listed_with_the_ruling_that_states_them(monkeypatch):
    seen = {}

    def fake_chat(messages, max_tokens=None, stop=None):
        seen["human"] = messages[1].content
        return ("**QUESTIONE 1: Responsabilità da cosa in custodia**\n"
                "- Il caso fortuito può consistere nella condotta del danneggiato [S1]\n"
                "- Un principio senza fonte\n"
                "- Un principio da un passo che non esiste [S7]\n"
                "2. Il custode risponde se il pericolo era prevedibile [S1, S2]\n"
                "QUESTIONE 2: Vizi della cosa\n"
                "- Il venditore risponde dei vizi nascosti della cosa venduta [S2]\n")

    monkeypatch.setattr(DG, "_call_chat", fake_chat)
    custodian = "Il  custode  risponde  dei danni quando il pericolo era prevedibile ed evitabile."
    rows = [{"d": {"name": "[ORD] Ordinanza n. 23908/2022"}, "s": {"plain_text": "matched"}, "_passage": RULING},
            {"d": {"name": "Ordinanza n. 2480/2018"}, "s": {"plain_text": custodian}}]
    out = DG._extract_principles(CASE, ANALYSIS, rows)
    # Each principle keeps only the rulings whose passage states it: run 9
    # credited rulings with principles they do not contain.
    assert out == ("QUESTIONE 1: Responsabilità da cosa in custodia\n"
                   "- Il caso fortuito può consistere nella condotta del danneggiato [Ordinanza n. 23908/2022]\n"
                   "- Il custode risponde se il pericolo era prevedibile [Ordinanza n. 2480/2018]")
    assert "[S1] Ordinanza n. 23908/2022\n" + RULING in seen["human"]      # the chosen passage
    assert "[S2] Ordinanza n. 2480/2018\n" + " ".join(custodian.split()) in seen["human"]
    assert DG._extract_principles(CASE, ANALYSIS, []) == ""

    def failing(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(DG, "_call_chat", failing)
    assert DG._extract_principles(CASE, ANALYSIS, rows) == ""


# --- Routing ----------------------------------------------------------------

def _capture_parere(monkeypatch):
    calls = []

    def fake(facts, lang, request=""):
        calls.append({"facts": facts, "request": request})
        return {"draft": "PARERE", "citations": [{"document_name": "Codice Civile 2026"}]}

    monkeypatch.setattr(DG, "run_case_analysis_pipeline", fake)
    return calls


def test_a_message_without_a_document_gets_the_parere(monkeypatch):
    calls = _capture_parere(monkeypatch)
    out = DG.run_defensive_pipeline(CASE, CASE, "it")
    assert out["draft"] == "PARERE"
    assert out["citations"] == [{"document_name": "Codice Civile 2026"}]
    assert out["proceeding"] == DG._safe_defaults()
    assert calls == [{"facts": CASE, "request": ""}]


def test_an_uploaded_document_gets_the_parere_with_the_request(monkeypatch):
    """Oct 2026: the PM's case was an uploaded file; the old flow wrote it up as
    a "documento giudiziario" analysis with invented facts."""
    calls = _capture_parere(monkeypatch)
    out = DG.run_defensive_pipeline(CASE, "Analizza il caso e indica la strategia del Comune", "it")
    assert out["draft"] == "PARERE"
    assert calls == [{"facts": CASE, "request": "Analizza il caso e indica la strategia del Comune"}]


def test_the_request_and_a_long_document_reach_the_model_visibly():
    text = DG._case_text("x" * (DG._FACTS_CHARS + 10), "Indica la strategia")
    assert text.startswith("RICHIESTA DELL'UTENTE: Indica la strategia\n\nDOCUMENTO CARICATO:\n")
    assert text.endswith("[... documento troncato ...]")
    assert DG._case_text(CASE) == CASE


# --- The whole pipeline, with a fake model and a fake database --------------

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.log.append((query, params))
        if "s.name AS name" in query:                       # every section of the rulings read
            return _Result([{"id": "LEGAL_DOC::o1", "name": "1_1", "text": COSTS_PART},
                            {"id": "LEGAL_DOC::o1", "name": "2_3", "text": RULING}])
        if "any(p IN $prefixes" in query:                   # keyword search in the codes
            return _Result([{"doc": "Codice Civile 2026", "sec": "2051.0.0", "score": 8.0}])
        if "STARTS WITH $prefix" in query:
            if params["prefix"] == "Codice Civile" and params["n"] == "2051":
                return _Result([{"d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026", "document_type": "primary"},
                                 "s": {"id": "DOCUMENT_SECTION::cc::2051", "name": "2051.0.0", "plain_text": ART_2051}}])
            return _Result([])
        if "section_fulltext" in query:
            return _Result([{"d": {"id": "LEGAL_DOC::o1", "name": "Ordinanza sul ricorso iscritto al n. 23908/2022 R.G.",
                                   "document_type": "interpretation"},
                             "s": {"id": "DOCUMENT_SECTION::o1::2_3", "name": "2_3", "plain_text": RULING},
                             "score": 29.5}])
        return _Result([])


class _Driver:
    def __init__(self):
        self.log, self.modes = [], []

    def session(self, **config):
        self.modes.append(config.get("default_access_mode"))
        return _Session(self.log)


@pytest.fixture
def pipeline(monkeypatch):
    import src.rag.main as main
    fake = _Driver()
    monkeypatch.setattr(main, "driver", fake)
    prompts = {}

    def fake_chat(messages, max_tokens=None, stop=None):
        system, human = messages[0].content, messages[1].content
        if "impostazione di un parere" in system:
            return json.dumps(ANALYSIS)
        if "articoli di legge candidati" in system:
            return '["art. 2051 c.c."]'
        if "principi di diritto" in system:
            prompts.update(principles_human=human)
            return "QUESTIONE 1: Custodia\n- Il caso fortuito può consistere nella condotta del danneggiato [S1]"
        if "PARERE MOTIVATO" in system:
            prompts.update(system=system, human=human, max_tokens=max_tokens)
            return ("**1. Inquadramento**\nLa pretesa rientra nell'art. 2051 c.c., non nell'art. 2044 c.c., "
                    "né nell'articolo 50 del Codice Civile.\n\n"
                    "**3. Argomenti a favore del Comune di Milano**\nIl caso fortuito (Cass., ordinanza sul ricorso "
                    "iscritto al n. 23908/2022 R.G.; Cass. 2480/2018).")
        return '["caso fortuito", "condotta del danneggiato"]'

    monkeypatch.setattr(DG, "_call_chat", fake_chat)
    return fake, prompts


def test_the_parere_is_written_from_the_retrieved_law(pipeline):
    fake, prompts = pipeline
    out = DG.run_case_analysis_pipeline(CASE, "it")

    import neo4j
    assert fake.modes == [neo4j.READ_ACCESS]                           # the database is only read
    assert ART_2051 in prompts["human"] and RULING in prompts["human"]
    assert COSTS_PART not in prompts["human"]                          # the ruling's reasoning, not its costs
    assert ("PRINCIPI DALLA GIURISPRUDENZA (per questione):\nQUESTIONE 1: Custodia\n- Il caso fortuito può "
            "consistere nella condotta del danneggiato [Ordinanza sul ricorso iscritto al n. 23908/2022 R.G.]"
            ) in prompts["human"]
    assert "[S1] Ordinanza sul ricorso iscritto al n. 23908/2022 R.G.\n" + RULING in prompts["principles_human"]
    assert out["principles"].startswith("QUESTIONE 1: Custodia")
    assert "art. 50" not in prompts["human"]                           # the analysis' guessed numbers stay out
    second_search = [p["t"] for q, p in fake.log if "section_fulltext" in q and "types" in p][-1]
    assert "2051" in second_search                                     # rulings searched with the verified article
    assert "piena visibilità" in prompts["human"]                      # every fact, with its effect
    assert "Comune di Milano" in prompts["system"]
    for copied in ("regole di gioco", "576 c.p.p.", "15 giorni dalla notifica"):
        assert copied not in prompts["system"]                         # no examples to copy
    assert "fase processuale in corso" in prompts["system"]            # e.g. what a 415-bis notice allows
    assert prompts["max_tokens"] == DG._PARERE_TOKENS

    draft = out["draft"]
    assert "art. 2051 c.c., non" in draft
    assert "art. 2044 c.c. [DA VERIFICARE]" in draft
    assert "articolo 50 del Codice Civile [DA VERIFICARE]" in draft
    assert "23908/2022 R.G.;" in draft and "2480/2018 [DA VERIFICARE]" in draft
    assert "Vuoi approfondire" in draft and "⚠️" in draft
    assert [c["document_name"] for c in out["citations"]] == [
        "Codice Civile 2026", "Ordinanza sul ricorso iscritto al n. 23908/2022 R.G."]


def test_civil_cases_do_not_search_the_criminal_commentary(pipeline):
    fake, _ = pipeline
    DG.run_case_analysis_pipeline(CASE, "it")
    ruling_searches = [params for query, params in fake.log if "section_fulltext" in query and "types" in params]
    assert ruling_searches and all(p["types"] == ["interpretation"] for p in ruling_searches)
