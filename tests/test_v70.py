"""v70: the lawyer's method as protocol -- synthetic records and texts, no case of the sim, no critique as a fixture.

1. the plan's Fasi: facts certi / ipotizzabili / mancanti (with what each would change); the sussunzione table, never
   «N su M»; orientamenti only from retrieved rulings; soluzioni a confronto with realismo, the main thesis never one the
   proven elements contradict; three objections; the coherence check; conclusions with the record's assessment; every
   «da verificare» collected in the last section -- his ten sections
2. the honesty rules and the domain modules by rito, in every writer call
3. LLM_THINKING=per_call: thinking on only where a call asks (the thesis selection, the writer)
4. the rubric's --critique mode
+ the outcome from the record (esito pieno / parziale / contenimento), circumstances as fattispecie with the aggravante
  check, the imputato's riti alternativi, ratione temporis, litisconsorzio by matter, the stragiudiziale option, the
  pre-filing civil list (mediazione / ATP / 700) and its diff against v59, the nightly's five cells
"""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rag import ai_chat, issue_records, lawyer_method as M, posture as posture_mod, render_computed  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}
SENTS = [{"id": "S1", "text": "Tizio e Caia hanno convissuto per anni.", "source": "caso", "certainty": "accertato"},
         {"id": "S2", "text": "Dopo la separazione Tizio ha pedinato e aggredito Caia.", "source": "caso",
          "certainty": "accertato"},
         {"id": "S3", "text": "Forse Tizio era geloso.", "source": "caso", "certainty": "ipotizzato"},
         {"id": "S4", "text": "Il sistema assume la data.", "source": "sistema", "certainty": "assunto"}]


def _req(i, text, status, **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _charge(*statuses, alternative=None):
    texts = ["condotte abituali di maltrattamento", "persona della famiglia o convivente"]
    recs = {"572 c.p.": {"rubrica": "Maltrattamenti", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st, supporting=["S2"] if st == "provato" else [])
                                        for i, st in enumerate(statuses, 1)]}}
    if alternative:
        recs["612-bis c.p."] = {"rubrica": "Atti persecutori", "side": "weapon", "alternative": True,
                                "requisites": [_req(9, "condotte reiterate di molestia", alternative)]}
    return recs


# --- 1. the facts, the sussunzione ----------------------------------------------------------------------------------------

def test_the_facts_are_split_by_certainty_and_the_missing_say_what_they_change():
    recs = _charge("provato", "da provare")
    recs["572 c.p."]["requisites"][1]["missing"] = "la convivenza al tempo dei fatti"
    split = M.facts_split(SENTS, recs, penal=True)
    assert split["certi"] == [SENTS[0]["text"], SENTS[1]["text"]]
    assert split["ipotizzabili"] == [{"text": "Forse Tizio era geloso.", "how": "ipotesi"}]       # never the system's
    [missing] = split["mancanti"]
    assert missing["fatto"] == "la convivenza al tempo dei fatti"
    assert "572 c.p." in missing["cambierebbe"] and "addebito" in missing["cambierebbe"]
    text = M.facts_section(split)
    assert "**Fatti certi**" in text and "**Fatti ipotizzabili**" in text and "cambierebbe:" in text


def test_the_sussunzione_is_a_table_per_article_element_fact_outcome():
    recs = _charge("provato", "da provare")
    table = M.subsumption_table("572 c.p.", recs["572 c.p."], {s["id"]: s for s in SENTS})
    assert "| Elemento | Fatto che lo integra | Esito |" in table
    assert "| condotte abituali di maltrattamento | Dopo la separazione Tizio ha pedinato e aggredito Caia. | integrato |" \
        in table
    assert "| da provare |" in table and "su" not in table.split("|")[-2]


def test_no_count_sentence_under_the_method(monkeypatch):
    rec = {"rubrica": "Furto", "requisites": [_req(1, "impossessamento", "provato")]}
    state = {"mode": "single_shot", "records": {"624 c.p.": rec}}
    assert flow.status_sentence("624 c.p.", rec, state)                   # today: «requisiti provati: 1 su 1»
    monkeypatch.setenv("LAWYER_METHOD", "true")
    assert flow.status_sentence("624 c.p.", rec, state) is None


# --- the probability from the record (point 7) ----------------------------------------------------------------------------

@pytest.mark.parametrize("statuses, alternative, esito, obiettivo", [
    (("provato", "provato"), None, M.SFAVOREVOLE, M.CONTENIMENTO),
    (("provato", "provato"), "provato", M.SFAVOREVOLE, M.PARZIALE),
    (("provato", "da provare"), None, M.SFAVOREVOLE, M.PARZIALE),
    (("provato", "non risulta"), None, M.FAVOREVOLE, M.PIENO),
    (("da provare", "da provare"), None, M.INCERTO, M.PIENO),
])
def test_the_defence_outcome_comes_from_the_charges_proven_elements(statuses, alternative, esito, obiettivo):
    a = M.assessment(_charge(*statuses, alternative=alternative), IMPUTATO, client="Tizio")
    assert (a["esito"], a["obiettivo"]) == (esito, obiettivo)
    assert a["article"] == "572 c.p."


def test_the_claimant_outcome_comes_from_the_main_articles_elements():
    rec = {"rubrica": "Servitù coattiva", "side": "weapon",
           "requisites": [_req(1, "fondo intercluso", "provato"), _req(2, "accesso alla via pubblica", "provato")]}
    assert M.assessment({"1051 c.c.": rec}, ATTORE, "1051 c.c.")["esito"] == M.FAVOREVOLE
    rec["requisites"][1]["status"] = "da provare"
    assert M.assessment({"1051 c.c.": rec}, ATTORE, "1051 c.c.")["esito"] == M.INCERTO
    text = M.assessment_text(M.assessment(_charge("provato", "provato"), IMPUTATO), "Tizio", "contenere la pena")
    assert "**Esito probabile**: sfavorevole a Tizio" in text and "**Obiettivo realistico**: contenimento" in text


# --- the coherence check --------------------------------------------------------------------------------------------------

def test_a_thesis_the_proven_elements_contradict_is_never_the_main_one():
    # v71: by the ids the thesis declares (contesta), never by its words
    recs = _charge("provato", "provato")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [
        {"tesi": "Assoluzione perché il fatto non sussiste", "pro": "-", "contro": "-", "realismo": "alto",
         "contesta": ["R1"]},
        {"tesi": "Tesi della convivenza mai esistita", "pro": "-", "contro": "-", "realismo": "medio", "minoritaria": True,
         "poggia_su": ["R1"]},
        {"tesi": "Riqualificazione in atti persecutori", "pro": "-", "contro": "-", "realismo": "medio",
         "poggia_su": ["R1"]}],
        "tesi_principale": 1})
    log = []
    sols, main = M.coherent_solutions(plan, recs, a, (), log)
    assert sols[main]["tesi"] == "Riqualificazione in atti persecutori"
    assert sols[0]["barred"] == ["R1"] and sols[0]["realismo"] == "basso"
    table = M.solutions_section(sols, main, plan, [], "accordo")
    assert "(tesi principale)" in table and "(tesi minoritaria)" in table and M.BARRED_LABEL in table


def test_no_standing_thesis_falls_back_to_containment():
    recs = _charge("provato", "provato")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [{"tesi": "Proscioglimento pieno", "realismo": "alto", "contesta": ["R2"]}]})
    sols, main = M.coherent_solutions(plan, recs, a, ["Giudizio abbreviato: x"])
    assert sols[main].get("fallback") and "Contenimento" in sols[main]["tesi"]


def test_the_coherence_check_drops_flattering_outcomes_and_matches_no_words():
    # v71: the elements are checked through the theses' declared ids -- a sentence is never matched against them
    recs = _charge("provato", "provato")
    a = M.assessment(recs, IMPUTATO)
    text = ("L'esito sarà favorevole a Tizio. Le condotte abituali di maltrattamento non sussistono. La difesa punta "
            "sulla pena.")
    log = []
    assert M.coherent_text(text, recs, a, log) == ("Le condotte abituali di maltrattamento non sussistono. La difesa "
                                                   "punta sulla pena.")
    assert len(log) == 1
    good = {**a, "esito": M.FAVOREVOLE}
    assert "favorevole" in M.coherent_text("L'esito sarà favorevole a Tizio.", recs, good)


# --- rulings, markers -----------------------------------------------------------------------------------------------------

def test_an_orientation_cites_only_a_retrieved_ruling():
    allowed = ["Cass. pen., sez. VI, n. 1234/2020"]
    assert M.ruling_ok("Cass. n. 1234/2020", allowed) and not M.ruling_ok("Cass. 9999/2019", allowed)
    out = []
    rulings = [{"id": "doc-1234", "label": allowed[0], "massima": "Il reato abituale richiede più condotte."}]
    text = M.orientations_section({"orientamenti": [
        {"punto": "convivenza", "pronuncia": "Cass. 9999/2019"},
        {"punto": "abitualità", "pronuncia": "Cass. n. 1234/2020"}]}, rulings, out)
    assert "- convivenza: orientamento da verificare" in text      # v71: code-rendered, the model's text never printed
    assert "- abitualità: Cass. pen., sez. VI, n. 1234/2020 (documento doc-1234): «Il reato abituale richiede più " \
           "condotte.»" in text
    assert out == ["Cass. 9999/2019"]
    assert "9999" not in M.unretrieved_rulings_out("Come afferma Cass. pen. n. 9999/2019, la norma vale.", allowed)


def test_every_marker_leaves_the_body_for_the_last_section():
    sections = [("A", "La querela è tempestiva (da verificare: la data della notizia del fatto)."),
                ("B", "- Termine: da verificare: il caso non dice la data.\n- Altro (art. 102 c.p.c.; orientamento da "
                      "verificare)."),
                ("C", "La regola vale — orientamento da verificare.")]
    body, points = M.collect_points(sections)
    joined = "\n".join(x for _, x in body)
    assert "da verificare" not in joined.lower()
    assert "(v. punto 1)" in body[0][1] and "(art. 102 c.p.c.; v. punto 2)" in body[1][1]
    assert "- Termine: v. punto 3." in body[1][1] and "(v. punto 4)." in body[2][1]
    # v74: each point the whole sentence that carried the marker, the marker out
    assert points == ["La querela è tempestiva (la data della notizia del fatto).", "Altro (art. 102 c.p.c.).",
                      "Termine: il caso non dice la data.", "La regola vale."]


# --- the layout: his ten sections -----------------------------------------------------------------------------------------

def _ctx(recs, posture=IMPUTATO, case="Nel gennaio 2025 Tizio ha aggredito Caia."):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], case, "Tizio", "", list(recs),
                     ["Cass. n. 1234/2020"], [], [], [], link_call=lambda *a: {})


def test_the_answer_is_laid_out_in_his_ten_sections():
    recs = _charge("provato", "provato")
    ctx = _ctx(recs)
    plan = M.clean_method({"questione_giuridica": "Se le condotte integrino il 572 c.p.",
                           "soluzioni": [{"tesi": "Riqualificazione", "pro": "a", "contro": "b", "realismo": "medio"}],
                           "obiezioni": [{"obiezione": "La convivenza c'era", "risposta": "Era cessata."}],
                           "conclusioni": "Va contenuta la pena.", "obiettivo": "contenere la pena"})
    sections = [("Risposta e strategia", "vecchio"), ("Norme applicate ai fatti", "**Art. 572 c.p.**\nregola"),
                ("Profili processuali", "- Procedibilità: d'ufficio.")]
    answer = M.compose(sections, ctx, plan)
    assert [t for t, _ in M.answer_sections(answer)] == M.titles()
    assert "vecchio" not in answer
    failed = [c for c in M.structure_checks(answer, IMPUTATO) if not c["ok"]]
    assert failed == []
    assert answer.rstrip().split("**10. Punti da verificare**")[1].strip()


def test_three_objections_even_when_the_model_gives_one():
    recs = _charge("provato", "da provare")
    ctx = _ctx(recs)
    text = M.objections_section({"obiezioni": [{"obiezione": "A", "risposta": "B"}]}, ctx["objections_fallback"])
    assert text.count("**Obiezione") == 3 and "è provato l'elemento" in text and "la prova spetta all'accusa" in text


# --- circumstances as fattispecie -----------------------------------------------------------------------------------------

def test_a_circumstance_is_a_record_with_its_requisites_and_the_aggravante_check_always_runs():
    recs = _charge("provato", "provato")
    case = "Tizio, in stato d'ira per un fatto ingiusto, ha colpito Caia."
    cands = M.circumstance_candidates(case, recs, IMPUTATO)
    labels = [c["label"] for c in cands]
    assert "62 n. 2 c.p." in labels and "62-bis c.p." in labels
    circ = M.circumstance_records(cands)
    texts = [r["text"] for r in circ["62 n. 2 c.p."]["requisites"]]
    assert any("fatto ingiusto" in t for t in texts) and len(texts) == 3

    def link(topic, reqs, sents):
        return {r["id"]: {"sostiene": ["S2"], "contraddice": [], "manca": ""} for _, r in reqs
                if r["text"].startswith("il reato è commesso in stato d'ira")}
    M.grade_circumstances(circ, SENTS, call=link)
    statuses = {r["text"][:20]: r["status"] for r in circ["62 n. 2 c.p."]["requisites"]}
    assert list(statuses.values()).count("provato") == 1          # the fatto ingiusto stays to prove
    assert M.aggravante_note(circ, recs) == M.NO_AGGRAVANTE
    aggr = M.circumstance_records([c for c in M.data()["circumstances"] if c["label"] == "61 n. 4 c.p."])
    for r in aggr["61 n. 4 c.p."]["requisites"]:
        r["status"] = "provato"
    assert "art. 61 n. 4 c.p.: integrata" in M.aggravante_note(aggr, recs)
    assert M.aggravante_note({}, {"1051 c.c.": {"requisites": []}}) == ""          # no charge, no check


# --- the imputato's subordinate line, ratione temporis, litisconsorzio, stragiudiziale --------------------------------------

def test_the_riti_alternativi_are_gated_on_the_charged_pena():
    # v79.10.3 (C1): a rito the pena precludes is no longer a line under the strategy -- riti_precluded, §8's
    lines = M.riti_lines(IMPUTATO, "è punito con la reclusione da due a sei anni")
    assert lines[0].startswith("Giudizio abbreviato") and not any("messa alla prova" in x for x in lines)
    assert "preclusa" in next(x for x in M.riti_precluded(IMPUTATO, "è punito con la reclusione da due a sei anni")
                              if "messa alla prova" in x)
    assert "preclusa" not in next(x for x in lines if "444" in x)
    heavy = "è punito con la reclusione da dodici a venti anni"     # (79.10.3 review) 12 x 2/3 (62-bis) x 2/3 > 5
    assert not any("444" in x for x in M.riti_lines(IMPUTATO, heavy))
    assert "preclusa" in next(x for x in M.riti_precluded(IMPUTATO, heavy) if "444" in x)
    assert M.riti_lines({**IMPUTATO, "stage": "impugnazione"}, "") == [] and M.riti_lines(ATTORE, "") == []


def test_the_imputato_posture_entry_carries_the_subordinate_line():
    entry = posture_mod.entry_for("penale", "imputato", "primo_grado")
    plan = posture_mod.apply_plan({"subordinate": ["altro"]}, {"entry": entry})
    assert plan["subordinate"][0] == "altro" and any("abbreviato" in x for x in plan["subordinate"])
    assert any("444" in x for x in plan["subordinate"]) and any("62-bis" in x for x in plan["subordinate"])


def test_ratione_temporis_by_the_fact_date():
    line, points = M.temporal("Nell'ottobre 2007 Sempronio ha colpito Tizio.", IMPUTATO)
    assert line.startswith("Disciplina applicabile alla data dei fatti (ottobre 2007)")
    assert "anteriori a riforma Cartabia penale" in line and "da verificare" not in line
    assert points and "Cartabia" in points[0]
    line, points = M.temporal("Tizio ha colpito Caio.", IMPUTATO)
    assert "non indica quando" in line and points[0].startswith("La data dei fatti")
    line, _ = M.temporal("Il 3 marzo 2024 Tizio ha affidato i lavori.", ATTORE)
    assert "si applicano riforma Cartabia civile" in line
    assert M.temporal("x", {"family": "amministrativo"}) == ("", [])


def test_litisconsorzio_by_matter_and_the_stragiudiziale_option():
    lines = M.litisconsorzio_lines("Tizio chiede la divisione della comunione ereditaria con i coeredi.", {}, ATTORE)
    assert lines == [next(m["line"] for m in M.data()["litisconsorzio"] if m["matter"] == "divisione")]
    assert M.litisconsorzio_lines("divisione", {}, IMPUTATO) == []
    assert "162-ter" in M.stragiudiziale_line(IMPUTATO, "querela")     # v79.10.3 (C1): only when the store says querela
    assert "162-ter" not in M.stragiudiziale_line(IMPUTATO)
    assert "1219" in M.stragiudiziale_line(ATTORE)
    assert "410" in M.stragiudiziale_line({"rito": "lavoro", "family": "civile", "role": "attore"})


# --- 2. the honesty rules and the modules, in every writer call -------------------------------------------------------------

def test_the_rules_follow_the_rito_and_reach_every_section_call(monkeypatch):
    penal = M.writer_rules_text("penale")
    assert M.HONESTY_TITLE in penal and "MODULO: DIRITTO PENALE SOSTANZIALE" in penal and "MODULO: PROCEDURA PENALE" in penal
    assert "FASE 7 - Confusioni frequenti da evitare" in penal and "Italiano giuridico chiaro" in penal
    labour = M.writer_rules_text("lavoro")
    assert "MODULO: DIRITTO DEL LAVORO" in labour and "MODULO: PROCEDURA CIVILE" in labour and "PENALE" not in labour
    assert M.current_rules() == ""
    with M.scope():
        M.begin("civile", ["diritti_reali"])
        assert "MODULO: DIRITTI REALI, SUCCESSIONI, FAMIGLIA" in M.current_rules()
        assert "MODULO: PROCEDURA CIVILE" in M.current_rules()
    assert M.current_rules() == ""

    from src.rag import plan_then_write as ptw, defensive_generation as dg
    seen = []

    def fake(messages, max_tokens=None, stop=None, temperature=None, thinking=None):
        seen.append((messages[0].content, thinking))
        return "Testo della sezione."
    monkeypatch.setattr(dg, "_call_chat", fake)
    plan = {"parte_assistita": "Tizio", "posizione": "imputato", "questioni": [], "pronunce_contrarie": []}
    with M.active("penale"):
        ptw.write_sections(plan, [("Norme applicate ai fatti", "SEZIONE", "materiale", 300)], "it", thinking=True)
    assert seen and M.HONESTY_TITLE in seen[0][0] and seen[0][1] is True


# --- 3. thinking per call -------------------------------------------------------------------------------------------------

def test_per_call_thinking_is_off_by_default_and_on_where_asked():
    assert ai_chat._thinking_kwargs("per_call") == {"extra_body": ai_chat._NO_THINKING}
    assert ai_chat.thinking_bind(True, 1000, "per_call") == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_tokens": 1000 + ai_chat.THINKING_EXTRA_TOKENS}
    assert ai_chat.thinking_bind(None, 1000, "per_call") == {}
    assert ai_chat.thinking_bind(True, 1000, "false") == {} and ai_chat.thinking_bind(True, 1000, None) == {}


def test_call_chat_binds_the_thinking_body_under_per_call(monkeypatch):
    bound = []

    class Reply:
        content, response_metadata, usage_metadata = "ok", {"finish_reason": "stop"}, {}

    class Model:
        def bind(self, **kw):
            bound.append(kw)
            return self

        def invoke(self, messages):
            return Reply()
    monkeypatch.setattr(ai_chat, "chat_model", Model())
    monkeypatch.setenv("LLM_THINKING", "per_call")
    assert ai_chat._call_chat([], max_tokens=500, thinking=True) == "ok"
    assert bound[-1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert bound[-1]["max_tokens"] == 500 + ai_chat.THINKING_EXTRA_TOKENS
    ai_chat._call_chat([], max_tokens=500)
    assert bound[-1] == {"max_tokens": 500}


def test_the_thesis_selection_is_timed_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))       # v71: the thesis log, never into the repository
    ctx = _ctx(_charge("provato", "da provare"))
    plan = M.method_plan(ctx, call=lambda messages: {"valutazioni": {"T2": {"realismo": "altissimo"}}})
    first = plan["soluzioni"][1]                        # v72: the code's candidates, the model's valuations (v79.7:
    assert first["candidate"] == "T2" and first["realismo"] == "medio" and first["contesta"] == ["R2"]   # elements' order
    assert "thesis_selection_s" in ctx["timings"]
    failed = M.method_plan(ctx, call=lambda m: (_ for _ in ()).throw(ValueError("x")))
    assert [t["candidate"] for t in failed["soluzioni"]] == [c["id"] for c in ctx["candidates"]]   # unargued, kept


# --- (10) the pre-filing civil list, against v59 --------------------------------------------------------------------------

def test_the_default_cell_lists_mediazione_atp_and_700_once():
    blocks = render_computed.stage_blocks(ATTORE)
    titles = [t for t, _ in blocks]
    assert titles == ["Mediazione", "Accertamento tecnico preventivo", "Provvedimento d'urgenza"]
    shown = "Mediazione\nMediazione obbligatoria (art. 5 d.lgs. 28/2010)\nStrumenti processuali\n- ATP (art. 696 c.p.c.)"
    assert [t for t, _ in render_computed.stage_blocks(ATTORE, rendered=shown)] == ["Provvedimento d'urgenza"]
    assert render_computed.stage_blocks({**ATTORE, "role": "convenuto"}) == []


def test_the_pre_filing_list_reaches_the_section_for_civil_claims_only():
    analysis = {"fact_signals": {}, "posture": ATTORE}
    titles = [t for t, _ in render_computed.blocks(analysis)]
    assert {"Mediazione", "Accertamento tecnico preventivo", "Provvedimento d'urgenza"} <= set(titles)
    labour = {"fact_signals": {}, "posture": {**ATTORE, "rito": "lavoro"}}
    assert "Accertamento tecnico preventivo" not in [t for t, _ in render_computed.blocks(labour)]


def test_the_stage_diff_counts_a_moved_block_as_no_loss():
    import stage_diff
    old = [["Strumenti processuali", "- ATP (art. 696 c.p.c.)\n- Urgenza (art. 700 c.p.c.)"], ["Termini", "art. 1168 c.c."]]
    new = [["Accertamento tecnico preventivo", "art. 696 c.p.c."], ["Provvedimento d'urgenza", "art. 700 c.p.c."]]
    d = stage_diff.diff(old, new)
    assert d["lost_blocks"] == ["Termini"] and d["lost_articles"] == ["1168 c.c."]


# --- 4. the rubric against the lawyer's critique --------------------------------------------------------------------------

def test_the_critique_mode_scores_points_and_finds_regressions(tmp_path):
    import rubric_score as R
    (tmp_path / "a.critique.md").write_text("La tesi è sbagliata.", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("x", encoding="utf-8")
    assert R.load_critiques(str(tmp_path)) == {"a": "La tesi è sbagliata."}
    raw = {"punti": [{"punto": "Circostanza 62 n. 6 senza requisiti", "categoria": "categoria", "esito": "ripetuto",
                      "why": "di nuovo"}, {"punto": "Esito favorevole", "esito": "boh"}],
           "tesi": {"score": 0.75, "why": "ok"}, "errori_di_categoria": {"score": 2}, "coerenza": {"score": "x"}}
    got = R.critique_score("crit", "caso", "parere", caller=lambda system, human: raw)
    assert [p["esito"] for p in got["points"]] == ["non risolto", "non risolto"]       # v71: the default
    assert got["model_scores"]["tesi"]["score"] == 0.75 and got["scores"]["errori_di_categoria"]["score"] == 0.0
    base = {"a": {"points": [{"punto": "Circostanza 62 n. 6 senza requisiti", "esito": "risolto"}],
                  "scores": {"errori_di_categoria": {"score": 0.5}}}}
    table = R.critique_table({"a": got}, base)
    assert "0 (-0.5)" in table and "(was good, now non risolto: di nuovo)" in table


# --- the nightly ----------------------------------------------------------------------------------------------------------

def test_the_nightly_has_five_cells_and_reports_regressions(tmp_path):
    import nightly_sim as N
    cells = N.cells()
    # v79.6: plus one two-client cell; v79.12.5 (item 10): plus one two-turn cell (a «followup»)
    single = [c for c in cells if not c.get("clients") and not c.get("followup")]
    assert len(single) == 5 and len({json.dumps(c["cell"], sort_keys=True) for c in single}) == 5
    assert len(cells) == 7 and [len(c["clients"]) for c in cells if c.get("clients")] == [2]
    assert len([c for c in cells if c.get("followup")]) == 1
    assert N.cell_check({"rito": "penale", "role": "imputato", "stage": "indagini"}, IMPUTATO)["ok"] is False
    spec = next(c for c in cells if c["cell"]["role"] == "imputato")
    recs = _charge("provato", "provato")
    plan = M.clean_method({"obiezioni": [{"obiezione": "x", "risposta": "y"}]})
    (tmp_path / f"{spec['id']}.md").write_text(M.compose([("Profili processuali", "- x")], _ctx(recs), plan),
                                               encoding="utf-8")
    result = N.offline_cell(spec, tmp_path)
    assert all(k["ok"] for k in result["checks"]), [k for k in result["checks"] if not k["ok"]]
    missing = N.offline_cell(cells[0], tmp_path)
    assert missing["checks"][0]["ok"] is False
    base = {"cells": [{"id": cells[0]["id"], "checks": [{"check": "run", "ok": True}]}]}
    assert N.regressions([missing], base) == [f"{cells[0]['id']}: run (no {cells[0]['id']}.md)"]


# --- the wiring: the single-shot answer under LAWYER_METHOD ---------------------------------------------------------------

def test_the_single_shot_answer_is_written_by_the_method_end_to_end(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    asked = {}

    def plan(ctx, call=None):
        asked["ctx"] = ctx
        ctx.setdefault("timings", {})["thesis_selection_s"] = 0.1
        cands = M.thesis_candidates(ctx)        # v75: the code's candidates (a thesis declaring nothing is a placeholder)
        ctx["candidates"] = cands
        return M.clean_method({"questione_giuridica": "Se spetti la servitù.", "obiezioni": [
            {"obiezione": "Il passaggio era tollerato", "risposta": "Era esercitato da decenni."}],
            "valutazioni": {c["id"]: {"pro": "opere visibili", "contro": "prova della separazione", "realismo": "medio"}
                            for c in cands}, "conclusioni": "Agire."}, cands)
    monkeypatch.setattr(M, "method_plan", plan)
    seen = {}
    link = {"R1": {"sostiene": ["S1"]}}         # v75: a proven element -- with none, the attore has no line (placeholder)
    text, _ = test_v60g._answer(monkeypatch, state, {}, link, seen)
    assert [j[0] for j in seen["jobs"]] == [flow.ANSWER_TITLES[1]]          # the writer writes the norms only
    assert [t for t, _ in M.answer_sections(text)] == M.titles()
    assert "**Esito probabile**" in text and "| Tesi | Pro | Contro | Realismo |" in text
    assert "requisiti provati" not in text and "Opzione stragiudiziale:" in text
    # v74: the method's view of the records -- the decomposed fattispecie, their read elements only
    assert set(asked["ctx"]["records"]) <= set(state["records"]) and state["timings"]["thesis_selection_s"] == 0.1
    assert all(r.get("kind") == "requisito" for rec in asked["ctx"]["records"].values() for r in rec["requisites"])
    assert "writer_s" in state["timings"]


def test_a_circumstance_claimed_without_its_requisites_gets_them_as_a_point():
    circ = M.circumstance_records([c for c in M.data()["circumstances"] if c["label"] == "62 n. 2 c.p."])
    circ["62 n. 2 c.p."]["requisites"][0]["status"] = "provato"
    log = []
    text = M.circumstance_claims("Va riconosciuta l'attenuante dell'art. 62, n. 6, c.p. Spetta anche la provocazione "
                                 "(art. 62 n. 2 c.p.). La pena va contenuta.", circ, log)
    assert "i requisiti dell'art. 62 n. 6 c.p.: prima del giudizio; la riparazione integrale del danno" in text
    assert "adoperato spontaneamente ed efficacemente" in M.data()["circumstances"][7]["elements"][1]
    assert "i requisiti dell'art. 62 n. 2 c.p.: lo stato d'ira è determinato da un fatto ingiusto altrui" in text
    assert text.endswith("La pena va contenuta.") and len(log) == 2
    for r in circ["62 n. 2 c.p."]["requisites"]:
        r["status"] = "provato"
    assert "da verificare" not in M.circumstance_claims("Spetta la provocazione (art. 62 n. 2 c.p.).", circ)
    assert M.circumstance_claims("L'art. 621 c.p. non rileva.", {}) == "L'art. 621 c.p. non rileva."



# --- his words, verbatim ----------------------------------------------------------------------------------------------------

def _his(text):
    """metodo.md as words: the file's word joiners and bullet marks out (the import's only change)."""
    return " ".join(text.replace("\u2060", "").replace("•", " ").split())


def test_the_data_file_holds_his_words_verbatim_and_in_sync():
    import lawyer_method_import as I
    source = _his((ROOT / "docs" / "lawyer" / "metodo.md").read_text(encoding="utf-8"))
    d = M.data()
    assert M.titles() == ["Sintesi della soluzione", "Fatti: certi / ipotizzabili / mancanti", "Questione giuridica",
                          "Norme e analisi per elementi", "Orientamenti rilevanti", "Soluzioni a confronto",
                          "Obiezioni e risposte", "Profili processuali", "Conclusioni e valutazione del rischio",
                          "Punti da verificare"]
    strings = [d["ruolo"], d["stile"]] + d["honesty_rules"] + d["controllo_finale"] + \
        [s["title"] for s in d["sections"]] + [s["note"] for s in d["sections"] if s.get("note")] + \
        [x for f in d["fasi"] for x in [f["title"]] + f["lines"]] + \
        [x for m in d["modules"].values() for x in [m["title"]] + m["lines"]]
    missing = [x for x in strings if _his(x) not in source]
    assert missing == []
    assert len(d["fasi"]) == 10 and len(d["modules"]) == 11 and len(d["controllo_finale"]) == 8
    assert I.main(["--check"]) == 0


def test_the_thesis_selection_reads_his_whole_method():
    text = M.method_text("penale")
    assert text.startswith("Sei un giurista italiano esperto di diritto penale.")
    for n in range(1, 11):
        assert f"FASE {n} - " in text
    assert "CONTROLLO FINALE (prima di rispondere)\n[ ] Ho distinto fatti certi" in text
    assert "1. Sintesi della soluzione (max 5 righe)" in text and "MODULO: PROCEDURA PENALE" in text
    assert "MODULO: DIRITTO TRIBUTARIO" in M.method_text("civile", ["tributario"])
    assert "MODULO: DIRITTO CIVILE" in M.method_text("civile")          # no area: the general civil module


def test_a_declared_minority_thesis_is_main_only_without_a_sustainable_non_minority_line():
    # v71 (the PO's rule 1): his Fase 6 lets a declared minority thesis lead -- only when no other line is sustainable
    recs = _charge("provato", "da provare")
    a = M.assessment(recs, IMPUTATO)
    minority = {"tesi": "Riqualificazione in omicidio colposo", "realismo": "basso", "minoritaria": True,
                "poggia_su": ["R1"]}
    plan = M.clean_method({"soluzioni": [{"tesi": "Pena vicina al minimo", "realismo": "alto", "poggia_su": ["R1"]},
                                         minority], "tesi_principale": 2})
    sols, main = M.coherent_solutions(plan, recs, a)
    assert sols[main]["tesi"] == "Pena vicina al minimo"
    plan = M.clean_method({"soluzioni": [{"tesi": "Pena vicina al minimo", "realismo": "alto"}, minority],
                           "tesi_principale": 2})
    sols, main = M.coherent_solutions(plan, recs, a)
    assert main == 1 and sols[0]["unverifiable"]
    assert "Riqualificazione in omicidio colposo (tesi principale) (tesi minoritaria)" in \
        M.solutions_section(sols, main, plan, [], "x")


def test_the_conclusions_carry_his_fase_10_and_the_final_check_is_answered():
    recs = _charge("provato", "provato")
    ctx = _ctx(recs)
    plan = M.clean_method({
        "sintesi": "Uno. Due. Tre. Quattro. Cinque. Sei. Sette.", "dipende_da": "la prova del diverbio",
        "rischio_residuo": "la condanna a una pena superiore al minimo",
        "passi": [{"passo": "Perizia medico-legale", "scopo": "verificare la dinamica della caduta"}],
        "controllo_finale": [{"voce": 1, "esito": "sì", "nota": ""}],
        "obiezioni": [{"obiezione": "o", "risposta": "r"}]})
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan, log=log)
    by = dict(M.answer_sections(answer))
    # v72: the summary, the conclusions and the next steps are the selection's -- never the model's plan
    assert by["Sintesi della soluzione"].startswith("Linea principale: Contenimento della pena")
    con = by["Conclusioni e valutazione del rischio"]
    assert "**Rischio residuo**: la condanna per l'art. 572 c.p." in con
    assert "Perizia medico-legale" not in con and "la prova del diverbio" not in con
    assert "**Linea principale**: Contenimento della pena" in con
    assert "**Analisi per elementi**" in by["Norme e analisi per elementi"]
    checks = log["controllo_finale"]
    assert [c["domanda"] for c in checks] == M.data()["controllo_finale"]
    assert checks[0]["codice"] is True and checks[0]["modello"] == "sì" and checks[1]["codice"] is True
    assert all(c["ok"] for c in M.structure_checks(answer, IMPUTATO)), \
        [c for c in M.structure_checks(answer, IMPUTATO) if not c["ok"]]


def test_the_critique_input_is_his_points_verbatim_scored_by_id():
    import critique_input as CI
    import rubric_score as R
    path = ROOT / "docs" / "lawyer" / "critiques" / "sempronio_no_hint.critique.json"
    crit = json.loads(path.read_text(encoding="utf-8"))
    source = _his((ROOT / "docs" / "lawyer" / "critica_penale_v2.md").read_text(encoding="utf-8"))
    assert all(_his(p["punto"]) in source for p in crit["points"]) and crit["giudizio"].startswith("Giudizio complessivo")
    assert crit == {**crit, **CI.parse((ROOT / "docs" / "lawyer" / "critica_penale_v2.md").read_text(encoding="utf-8"))}
    cats = {p["id"]: p["categoria"] for p in crit["points"]}
    assert cats["1.1"] == "tesi" and cats["2.1"] == "categoria" and cats["3.4"] == "coerenza" and cats["6.1"] == "pregio"
    assert cats["7.5"] == "attesa" and "Sempronio" in crit["case"]
    assert R.load_critiques(str(path.parent))["sempronio_no_hint"]["id"] == "sempronio_no_hint"
    seen = {}

    def model(system, human):
        seen["human"], seen["system"] = human, system
        return {"punti": {"1.1": {"esito": "risolto"}, "6.1": {"esito": "perso"}, "7.5": {"esito": "adottato"},
                          "2.1": {"esito": "mantenuto"}}, "tesi": {"score": 1}}
    got = R.critique_score(crit, "", "parere", caller=model)
    assert "[3.5] (coerenza) Art. 62 n. 2 (provocazione)" in seen["human"] and "Sempronio" in seen["human"]
    by = {p["id"]: p for p in got["points"]}
    assert len(by) == len(crit["points"]) and by["6.1"]["esito"] == "non risolto"   # no evidence: the default
    assert by["1.1"]["esito"] == "non risolto" and by["1.1"]["modello"]["esito"] == "risolto"     # v71: the table's
    assert by["2.1"]["esito"] == "non risolto"                      # not an outcome of its category: the default
    base = {"sempronio_no_hint": {"points": [{**by["6.1"], "esito": "mantenuto"}], "scores": {}}}
    table = R.critique_table({"sempronio_no_hint": got}, base)
    assert "6.1 La struttura segue bene la traccia" in table.split("Regressions")[1]
