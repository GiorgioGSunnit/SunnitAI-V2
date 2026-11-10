"""v60e: live runs on unseen cases (civil-insurance, tenancy) -- 1 side-dependent reasoning recomputed when the client
is settled; 2 one citation reading, code names with spaces; 3 the rubrica/text mismatch at load; 4 the «non si applica»
detector widened; 5 institutions as the code rubricates them; 6 round 2's one behaviour; 7 the procedural rules with their
own facts. The model and the graph are stubbed."""

import copy
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import article_text, case_read, citations, interaction_copy as C, render_computed, topic_groups  # noqa
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402

# --- 1. the client's side ---------------------------------------------------------------------------------------------

TWO_SIDES = {"cliente": {"nome": "", "ruolo": ""}, "fatti": ["Il conduttore non paga"],
             "ipotesi": [{"parte": "Rossi", "linea": "principale", "tesi": "Sfratto per morosità",
                          "istituti": ["sfratto"], "rubricati": ["Intimazione di sfratto per morosità"]},
                         {"parte": "Rossi", "linea": "controparte", "tesi": "Eccezione di inadempimento",
                          "istituti": ["eccezione di inadempimento"]},
                         {"parte": "Bianchi", "linea": "principale", "tesi": "Eccezione di inadempimento del locatore",
                          "istituti": ["eccezione di inadempimento"],
                          "rubricati": ["Eccezione d'inadempimento"]},
                         {"parte": "Bianchi", "linea": "controparte", "tesi": "Sfratto per morosità",
                          "istituti": ["sfratto"]}]}


def test_the_first_read_gives_each_partys_lines_when_the_client_is_open():
    read = case_read.clean(TWO_SIDES)
    assert {h["party"] for h in read["hypotheses"]} == {"Rossi", "Bianchi"}
    assert [h["claim"] for h in case_read.side(read["hypotheses"], "Mario Bianchi")] == [
        "Eccezione di inadempimento del locatore", "Sfratto per morosità"]
    assert case_read.side(read["hypotheses"], "Verdi") == [] and len(case_read.side(read["hypotheses"], "")) == 4
    assert read["hypotheses"][0]["rubricated"] == ["Intimazione di sfratto per morosità"]
    assert case_read.same_party("la società Alfa", "Alfa S.r.l.") and not case_read.same_party("Rossi", "Bianchi")


def _sided_state(monkeypatch):
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "true")
    state = base._state(monkeypatch, base.NO_CLIENT)
    hyps = case_read.clean(TWO_SIDES)["hypotheses"]
    eviction = {"title": "Sfratto", "anchor": "658 c.p.c.", "articles": ["658 c.p.c."], "planned": 1, "graded": 1,
                "best_grade": 3, "primary": True, "code": "c.p.c.", "unit": "", "source": "rubrica",
                "grades": {"658 c.p.c.": 3}, "hypotheses": [], "offences": []}
    exception = {"title": "Eccezione d'inadempimento", "anchor": "1460 c.c.", "articles": ["1460 c.c."], "planned": 1,
                 "graded": 1, "best_grade": 3, "primary": True, "code": "c.c.", "unit": "", "source": "rubrica",
                 "grades": {"1460 c.c.": 3}, "hypotheses": [], "offences": []}
    state["topics"] = [eviction, exception]
    state["pool"] = [{"_article": ["658", "c.p.c."], "rubrica": "Intimazione di sfratto per morosità"},
                     {"_article": ["1460", "c.c."], "rubrica": "Eccezione d'inadempimento"}]
    state["read"] = {"client": {"name": ""}, "hypotheses": hyps, "unknown": []}
    state["read_side"] = ""
    monkeypatch.setattr(topic_groups, "_embedder", lambda: (lambda texts: [[1.0, 0.0] for _ in texts]))
    return state


def test_a_stated_client_drives_the_lines_the_order_and_the_immediate_answer(monkeypatch):
    state = _sided_state(monkeypatch)
    chosen = [1]
    entry = flow.reside(state, "Bianchi", [chosen])
    assert entry["client"] == "Bianchi" and entry["rerun_ms"] is None
    assert state["topics"][0]["title"] == "Eccezione d'inadempimento"          # its main line first
    assert [h["party"] for t in state["topics"] for h in t["hypotheses"]] == ["Bianchi", "Bianchi"]
    assert chosen == [2]                                                       # «1» was the eviction topic
    assert flow.default_choice(state)[0] == 1 and state["read_side"] == "Bianchi"
    assert flow.reside(state, "Bianchi") == {}                                 # once


def test_a_single_sided_read_is_read_again_for_the_settled_client(monkeypatch):
    state = _sided_state(monkeypatch)
    state["read"]["hypotheses"] = [dict(h, party="Rossi") for h in state["read"]["hypotheses"][:2]]
    asked = {}

    def reader(text, request="", lang="it", client=""):
        asked["client"] = client
        return case_read.clean({"fatti": ["x"], "ipotesi": [{"linea": "principale", "tesi": "Eccezione",
                                                             "istituti": ["eccezione d'inadempimento"]}]})
    monkeypatch.setattr(case_read, "read", reader)
    entry = flow.reside(state, "Bianchi")
    assert asked["client"] == "Bianchi" and entry["rerun_ms"] is not None and entry["lines"] == 1
    assert all(h["party"] == "Bianchi" for t in state["topics"] for h in t["hypotheses"])


def test_the_immediate_answer_follows_the_assumed_client_too(monkeypatch):
    state = _sided_state(monkeypatch)
    analysis = flow._restored_analysis(state["analysis"])
    analysis.pop("parte_assistita", None)
    flow._facts(analysis)["parties"] = [{"name": "Rossi", "role": "locatore"}, {"name": "Bianchi", "role": "convenuto"}]
    state["analysis"] = flow._jsonable(analysis)
    state["questions"] = [{"id": "parte_assistita", "family": "parte_assistita"}]
    flow.prepare_side(state)
    assert state["read_side"] == "Bianchi" and state["topics"][0]["title"] == "Eccezione d'inadempimento"


def test_round_one_and_the_writer_say_whose_line_while_the_client_is_open(monkeypatch):
    state = _sided_state(monkeypatch)
    state["topics"][0]["hypotheses"] = [{"line": "principale", "claim": "Sfratto", "party": "Rossi"}]
    state["topics"][1]["hypotheses"] = [{"line": "principale", "claim": "Eccezione", "party": "Bianchi"}]
    reply, _ = flow.round1(state)
    assert "linea principale per Rossi: Sfratto" in reply and "linea principale per Bianchi: Eccezione" in reply
    assert "- linea principale per Rossi: Sfratto" in flow.hypotheses_block(state["topics"])


# --- 2. one citation reading -----------------------------------------------------------------------------------------

STORE_NAMES = ["Codice Civile 2026", "Codice Penale 2026", "Codice di procedura civile 2026",
               "Codice di procedura penale 2026", "Codice del processo amministrativo 2026", "Codice della Strada 2026",
               "Codice delle Assicurazioni Private 2026", "Legge sul Procedimento Amministrativo 2026",
               "Codice degli Appalti 2026", "Codice del Consumo 2026", "Statuto dei Lavoratori 2026",
               "Testo Unico Edilizia 2026", "Legge sulle Locazioni Abitative 2026", "Codice della Privacy 2026"]
DOCS = [{"id": n, "name": n, "aliases": []} for n in STORE_NAMES]


@pytest.mark.parametrize("name", STORE_NAMES)
def test_every_code_name_is_read_in_every_citation_form(name):
    assert citations.self_check([d for d in DOCS if d["name"] == name] + DOCS) == []


def test_a_paragraph_citing_a_named_code_goes_under_its_heading():
    refs = [("148", "Codice delle Assicurazioni Private 2026"), ("2054", "c.c.")]
    paragraph = "Ai sensi dell'art. 148 del Codice delle Assicurazioni Private 2026, l'impresa formula l'offerta."
    assert flow._paragraph_article(paragraph, 2, refs, {}, flow.ref_documents(refs)) == 1
    assert flow._paragraph_article(paragraph, 2, refs, {}) == 1          # the names from the refs themselves


def test_the_headings_name_the_documents_when_no_store_is_at_hand():
    text = ("**Art. 3 Legge sul Procedimento Amministrativo 2026 — Motivazione**\n\nL'art. 3 Legge sul Procedimento "
            "Amministrativo 2026 impone la motivazione; l'art. 2054 c.c. no.")
    assert citations.refs_in(text) == [("3", "Legge sul Procedimento Amministrativo 2026"),
                                       ("3", "Legge sul Procedimento Amministrativo 2026"), ("2054", "c.c.")]
    assert citations.heading("**Art. 10-bis Legge sul Procedimento Amministrativo 2026 — Preavviso**") == (
        "10-bis", "Legge sul Procedimento Amministrativo 2026", "Preavviso")
    assert citations.code_key("Codice della Strada 2026") == "c.d.s." and citations.code_key("Codice Civile 2026") == "c.c."


def test_the_lint_finds_a_cited_article_listed_as_not_treated():
    import answer_lint
    norms = ("**Art. 148 Codice delle Assicurazioni Private 2026 — Procedura di risarcimento**\n\n(co. 1) …\n\n"
             "**Art. 2054 c.c. — Circolazione di veicoli**\n\nL'art. 148 Codice delle Assicurazioni Private 2026 "
             "fissa i termini dell'offerta.\n\nAltre norme del tema non trattate: art. 148 Codice delle Assicurazioni "
             "Private 2026.")
    assert answer_lint.lint_untreated_cited(norms) == [
        "art. 148 Codice delle Assicurazioni Private 2026: «non trattata» but cited in §2"]
    assert answer_lint.lint_marker_title("1. **[1025]** (art. 1025 c.c.)") == ["topic 1 «[1025]»"]


def test_the_read_back_reads_a_named_code():
    sections = [("A", "L'art. 3 Legge sul Procedimento Amministrativo 2026 si applica."),
                ("B", "L'art. 3 Legge sul Procedimento Amministrativo 2026 non si applica.")]
    from src.rag import readback
    out = readback.contradictions(sections, {"3 leggesulprocedimentoamministrativo2026": ["non risulta"]}, [],
                                  documents=DOCS)
    assert out[0][1].strip() == ""


def test_a_footnote_marker_is_never_a_topic_title():
    assert topic_groups.unit_title("[1025]") == "" and topic_groups.unit_title("CAPO II [12] – Delle servitù") == "Servitù"
    row = {"_article": ("1025", "c.c."), "_grade": 3, "s": {"name": "1025.0.0", "titolo_articolo": "[1025]"},
           "d": {"id": "cc"}}
    assert topic_groups._name({"unit": "[7]", "rows": [row]}) == "Art. 1025 c.c."


# --- 3. the rubrica / text mismatch ------------------------------------------------------------------------------------

def test_a_node_whose_rubrica_is_not_its_text_is_flagged():
    bad = article_text.misaligned("Furto in abitazione e furto con strappo",
                                  "Il proprietario del fondo servente non può compiere alcuna cosa che tenda a diminuire "
                                  "l'esercizio della servitù o a renderlo più incomodo.")
    assert bad and bad["rubrica"].startswith("Furto")
    assert article_text.misaligned("Omicidio", "Chiunque cagiona la morte di un uomo è punito con la reclusione.") is None
    assert article_text.misaligned("Danno cagionato da cosa in custodia",
                                   "Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, salvo il "
                                   "caso fortuito.") is None
    assert article_text.misaligned("Disposizioni generali", "Le norme del presente capo si applicano ai contratti.") is None


class _Neighbours:
    """A graph session answering the neighbours' query with the given {number: (rubrica, text)}."""
    def __init__(self, articles):
        self.articles = articles

    def run(self, query, items=()):
        recs = [{"key": it["key"], "name": n, "rubrica": self.articles[n.split(".")[0]][0],
                 "text": self.articles[n.split(".")[0]][1]}
                for it in items for n in it["names"] if not n.endswith(".0.0") and n in self.articles]
        return type("R", (), {"data": lambda _self: recs})()


def _row(number, code, rubrica, text, grade=3):
    return {"_article": (number, code), "_grade": grade, "s": {"name": f"{number}.0.0", "titolo_articolo": rubrica,
                                                              "plain_text": text}, "d": {"id": f"doc::{code}"}}


def test_a_suspect_is_excluded_only_when_a_neighbour_confirms_the_shift():
    next_text = ("Il proprietario del fondo dominante, nel fare le opere necessarie per conservare la servitù, deve "
                 "scegliere il tempo e il modo che siano per recare minore incomodo al proprietario.")
    shifted = _row("1068", "c.c.", "Trasferimento in luogo diverso", next_text)      # 1068's rubrica on 1069's text
    real = _row("378", "c.p.", "Favoreggiamento personale",
                "Chiunque, dopo che fu commesso un delitto, aiuta taluno a eludere le investigazioni dell'autorità, è "
                "punito con la reclusione fino a quattro anni.")
    session = _Neighbours({"1067": ("Divieto di aggravare o di diminuire l'esercizio della servitù", "Il proprietario "
                                    "del fondo dominante non può fare innovazioni che rendano più gravosa la condizione."),
                           "1069": ("Opere sul fondo servente", "Il proprietario del fondo servente non è tenuto a "
                                    "compiere alcun atto, salvo che la legge o il titolo dispongano diversamente."),
                           "377": ("Intralcio alla giustizia", "Chiunque offre denaro..."),
                           "379": ("Favoreggiamento reale", "Chiunque aiuta taluno ad assicurare il prodotto...")})
    assert article_text.misaligned(real["s"]["titolo_articolo"], real["s"]["plain_text"])       # the stem check alone
    confirmed = article_text.confirm_misaligned(session, [shifted, real])
    assert confirmed == [shifted] and shifted["_misaligned"]["neighbour"] in ("Divieto di aggravare o di diminuire l'esercizio della servitù", "Opere sul fondo servente")
    assert real["_misaligned"] is None                                  # a real article: never excluded
    assert [r["_article"] for r in topic_groups._best_rows([shifted, real], lambda g: g >= 2)] == [("378", "c.p.")]
    topics = [{"title": "T", "anchor": "1068 c.c.", "articles": ["1068 c.c.", "378 c.p."], "planned": 2,
               "matched": ["1068 c.c."]}]
    out = flow.prune_abrogated(topics, [shifted, real])
    assert out[0]["articles"] == ["378 c.p."] and out[0]["anchor"] == "378 c.p."
    logged = flow.log_misaligned([shifted, real])
    assert list(logged) == ["c.c."] and logged["c.c."][0]["article"] == "1068"
    assert article_text.neighbour_numbers("624-bis") == ["623", "625", "624"]
    aggravanti = _row("583", "c.p.", "Circostanze aggravanti", "La lesione personale è grave, e si applica la reclusione "
                      "da tre a sette anni, se dal fatto deriva una malattia che metta in pericolo la vita.")
    near = _Neighbours({"582": ("Lesione personale", "Chiunque cagiona ad alcuno una lesione personale è punito; il delitto "
                                "è punibile a querela salvo che ricorra taluna delle circostanze aggravanti."),
                        "584": ("Omicidio preterintenzionale", "Chiunque cagiona la morte di un uomo con atti diretti...")})
    assert article_text.confirm_misaligned(near, [aggravanti]) == [] and aggravanti["_misaligned"] is None


# --- 4. the «non si applica» detector ------------------------------------------------------------------------------

@pytest.mark.parametrize("gloss", [
    "La norma non rileva direttamente: il contratto è stato concluso prima.",
    "La questione non costituisce un problema nel caso, perché il termine è rispettato.",
    "Il provvedimento mantiene la sua efficacia: l'annullamento non è stato chiesto.",
    "La disposizione non trova applicazione: manca il presupposto della recidiva.",
    "La norma non è pertinente alla vicenda.",
    "Il problema della prescrizione non si pone.",
])
def test_the_wider_phrasings_declare_the_article_inapplicable(gloss):
    assert flow.declares_inapplicable(gloss)


# --- 5. institutions as the code rubricates them --------------------------------------------------------------------

def test_the_search_tries_the_rubricated_name_then_the_wording_then_the_claims_nouns():
    asked = []

    def resolve(session, documents, ist):
        asked.append((ist["nome"], ist["codice"]))
        if ist["nome"] == "morosità conduttore canoni":
            return [{"_article": ("658", "c.p.c."), "s": {}}], {}
        return [], {}
    items = [{"name": "sfratto", "rubricated": "Intimazione di sfratto", "claim": "La morosità del conduttore nei canoni"}]
    rows, log = flow.first_read_search(None, "db", [], items, ["c.p.c."], session=object(), resolve=resolve)
    assert asked == [("Intimazione di sfratto", "c.p.c."), ("sfratto", "c.p.c."), ("morosità conduttore canoni", "c.p.c.")]
    assert [r["_article"] for r in rows] == [("658", "c.p.c.")]
    assert [t[1] for t in log["tried"]["sfratto"]] == ["rubricato", "istituto", "tesi"]
    assert log["hits"]["Intimazione di sfratto"] == ["658 c.p.c."]


def test_a_hypothesis_anchors_by_its_rubricated_name():
    topic = {"title": "Sfratto", "grades": {"658 c.p.c.": 3}, "hypotheses": [], "unit": "", "source": "rubrica"}
    h = {"line": "principale", "claim": "x", "institutions": ["mandar via l'inquilino"],
         "rubricated": ["Intimazione di sfratto per morosità"]}
    unanchored, log = topic_groups.anchor_by_similarity([topic], {"topics": [[]]}, [h],
                                                        rubriche={"658 c.p.c.": "Intimazione di sfratto per morosità"})
    assert not unanchored and log[0]["path"] == "rubrica" and topic["hypotheses"][0]["anchor_institution"].startswith("Intim")


def test_an_unanchored_hypothesis_logs_what_was_searched(monkeypatch):
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    topic = {"title": "Sfratto", "grades": {"658 c.p.c.": 3}, "hypotheses": [], "unit": "", "source": "rubrica"}
    h = {"line": "subordinata", "claim": "x", "institutions": ["abuso del diritto"], "rubricated": ["Abuso del diritto"]}
    search = {"hits": {"abuso del diritto": []}, "tried": {"abuso del diritto": [["Abuso del diritto", "rubricato", "c.c.", 0]]}}
    flow.attach_side([topic], [h], {"topics": [[]]}, {"658 c.p.c.": "Intimazione di sfratto"}, search)
    entry = next(d for e, d in logged if e == "hypothesis_anchor")
    assert entry["attached"] is False and entry["searched"]["names"] == ["Abuso del diritto", "abuso del diritto"]
    assert entry["searched"]["tried"]["abuso del diritto"][0][1] == "rubricato"


# --- 6. round 2, one behaviour ---------------------------------------------------------------------------------------

def test_the_restatement_is_two_sentences_in_the_answers_words(monkeypatch):
    state = base._state(monkeypatch)
    state["said_items"] = ["data del fatto: 7 novembre 2026", "data del fatto: 7 novembre 2026", "il cancello è chiuso"]
    state["chosen"] = [1]
    text = flow.restatement(state)
    first, rest = text.split("\n\n")
    assert first == ("Imposto il parere per Leonardo. Tengo conto di quanto mi hai precisato: data del fatto: 7 novembre "
                     "2026; il cancello è chiuso.")
    assert rest.startswith("Tratterò: Servitù per destinazione")
    assert "→" not in text and "?" not in first


def test_a_reply_answering_a_question_gets_the_restatement_and_one_asking_the_answer_gets_it(monkeypatch):
    state = base._state(monkeypatch)
    q = next(q for q in state["questions"] if q["kind"] == "yesno")
    raw = {"topics": [], "confirm": False, "answers": {q["id"]: "sì"}, "angle": "", "corrections": [], "statements": [],
           "intents": ["rispondi"], "question": ""}
    monkeypatch.setattr(flow, "_reply_call", lambda *a: dict(raw))
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (f"RISPOSTA su {numbers}", []))
    out = flow.follow_up("sì, c'era", copy.deepcopy(state))
    assert out["kind"] == "round2" and out["answer"].startswith("Imposto il parere per Leonardo. Tengo conto")
    assert q["text"] not in out["answer"]
    raw.update(confirm=True, intents=["rispondi", "procedi"])
    assert flow.follow_up("sì, c'era; procedi", copy.deepcopy(state))["kind"] == "answer"


# --- 7. procedural rules with their own facts ------------------------------------------------------------------------

def _civil(facts, case="", **extra):
    return {"frame": {"areas": ["responsabilita_civile"]}, "store_offences_only": True, flow.CASE_KEY: case,
            "fact_signals": {"facts": facts, "tools": [], "terms": []}, **extra}


def test_negotiation_instead_of_mediation_for_a_road_accident():
    facts = {"events": [{"what": "sinistro stradale: tamponamento dell'auto di Rossi", "date": ""}],
             "injuries": [{"what": "colpo di frusta"}], "requests": ["risarcimento del danno"]}
    analysis = _civil(facts, topic_refs=[("1917", "c.c.")])          # an insurance-contract article: mediation matter
    assert render_computed.mediation_of(analysis)["required"]
    titles = [t for t, _ in render_computed.blocks(analysis)]
    assert render_computed.NEGOTIATION_TITLE in titles and "Mediazione" not in titles
    legacy = [t for t, _ in render_computed.blocks({**analysis, "store_offences_only": False})]
    assert "Mediazione" in legacy                                    # the old path unchanged


def test_negotiation_for_a_small_payment_outside_mediation_matters():
    facts = {"events": [], "requests": ["pagamento della fattura di euro 12.000"], "contracts": []}
    assert render_computed.negotiation_of(_civil(facts))["why"] == "pagamento"
    facts["requests"] = ["pagamento della fattura di euro 80.000"]
    assert not render_computed.negotiation_of(_civil(facts))["required"]


def test_a_strumento_only_with_its_triggering_fact_from_the_case():
    tool = {"id": "provvedimento_urgenza", "refs": [["700", "c.p.c."]], "text": "Provvedimento d'urgenza (art. 700 c.p.c.).",
            "matched": ["la polizza è stata stipulata"]}
    case = "La compagnia rifiuta il pagamento. Il muro rischia di crollare sulla strada: c'è pericolo per i passanti."
    out = render_computed._triggered_tools([tool], _civil({}, case))
    assert out[0]["matched"] == ["Il muro rischia di crollare sulla strada: c'è pericolo per i passanti."]
    assert render_computed._triggered_tools([tool], _civil({}, "La compagnia rifiuta il pagamento.")) == []
    ata = {"id": "accertamento_preventivo", "matched": ["danni"]}
    assert render_computed._triggered_tools([ata], _civil({}, "Ci sono danni all'auto.")) == []
    kept = render_computed._triggered_tools([ata], _civil({}, "Le infiltrazioni vanno riparate prima che peggiorino."))
    assert kept and "Nel caso: Le infiltrazioni" in render_computed._tool_line(kept[0])
