"""v79.18.2 -- item 1 the risk thresholds are data by side; item 2 the strategy's side from the pretesa's tipo when the
frame's ruolo is unverified, and the re-ask stating the side when every proposed line missed it; item 3 the frame-named
article enters the set only with a verified «si» / «dipende»; item 4 the retrieval's sub-stages named (the fusion's
marks, the replay's row); item 5 the matter gate demotes, never drops, the top-k hits of every further area the pretesa
or a quesito touches; item 6 a fit «no» in the charge's Titolo / Capo kept as the «alternativa» tier (a riqualificare line
only, never the principale). Synthetic rows, texts and replies only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_17_3 as T173  # noqa: E402
import test_v79_18_0 as Z  # noqa: E402
import test_v79_18_1 as C  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import stage_meter  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402
from src.rag import topic_groups as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_182", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


# --- item 1 ---------------------------------------------------------------------------------------------------------------

def test_item1_the_risk_thresholds_are_data_by_side():
    risk = S.config().get("risk") or {}
    assert set(risk) >= {"attore", "difesa", "neutro"}
    for side, cfg in risk.items():
        assert cfg["misura"] in ("provato", "aperti") and 0.0 <= float(cfg["sfavorevole_max"]) <= float(cfg["favorevole_min"]) <= 1.0
    # the word follows the data: the defence measures the open elements, the claimant the proven ones
    inp = Z._inp(side="difesa")
    line = {"articolo": "1385 c.c.", "eid": "A1385cc:E2", "cita_fatti": [], "cita_pronunce": []}
    f = S.forza("1385 c.c.", "A1385cc:E2", inp, line)
    assert S.risk_word({"analysis": {"linee": [{"forza": f}], "side": "difesa"}}) in ("favorevole", "incerto", "sfavorevole")


# --- item 2 ---------------------------------------------------------------------------------------------------------------

def test_item2_the_side_from_the_pretesas_tipo_when_the_frame_gave_none():
    assert S.side_of("difesa", {"addebito_o_domanda": {"tipo": "domanda"}}) == ("difesa", "frame")       # the frame wins
    assert S.side_of("attore", None) == ("attore", "frame")
    assert S.side_of("neutro", {"addebito_o_domanda": {"tipo": "accusa"}}) == ("difesa", "pretesa_tipo:accusa")
    assert S.side_of("neutro", {"addebito_o_domanda": {"tipo": "opposizione"}}) == ("difesa", "pretesa_tipo:opposizione")
    assert S.side_of(None, {"addebito_o_domanda": {"tipo": "domanda"}}) == ("attore", "pretesa_tipo:domanda")
    assert S.side_of("", {"addebito_o_domanda": {"tipo": "ricorso"}}) == ("attore", "pretesa_tipo:ricorso")
    assert S.side_of("neutro", {"addebito_o_domanda": {"tipo": "impugnazione"}}) == ("attore", "pretesa_tipo:impugnazione")
    assert S.side_of("neutro", {"addebito_o_domanda": {"tipo": "istanza"}}) == ("neutro", "")        # not in the data
    assert S.side_of("neutro", {}) == ("neutro", "") and S.side_of("neutro", None) == ("neutro", "")
    assert set(S.config()["side_by_tipo"]) == {"accusa", "opposizione", "domanda", "ricorso", "impugnazione"}


def test_item2_every_line_rejected_for_the_side_brings_the_side_statement_into_the_reask():
    inp = Z._inp(side="attore")
    # the model argued the defence's case: «contestare» on the provato E1, «fondare» on the da_provare E3
    reply = Z._reply([Z._linea("A1385cc", "A1385cc:E1", "contestare", "principale", cf=["S2"]),
                      Z._linea("A1385cc", "A1385cc:E3", "fondare", "subordinata", cf=["S2"])])
    checked = S.verify(reply, inp)
    assert checked["linee"] == [] and checked["side_mismatch"] is True and len(checked["rejected"]) == 2
    statement = S.side_statement(inp)
    assert statement.startswith("PARTE E LATO: la parte assistita è Anna Corsi, lato attore.") and "«fondare»" in statement
    assert statement not in checked["reasons"]                 # the re-ask's own line, not a reason bullet
    msgs = S.reask_messages(S.build_messages(inp), json.dumps(reply), checked["reasons"], inp, side_mismatch=True)
    assert statement in msgs[-1].content and msgs[-1].content.count("PARTE E LATO") == 1
    assert "PARTE E LATO" not in S.reask_messages(S.build_messages(inp), json.dumps(reply), checked["reasons"], inp)[-1].content
    # one bad line among good ones is no side mismatch: the ordinary re-ask
    reply2 = Z._reply([Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", cf=["S2"]),
                       Z._linea("A1385cc", "A1385cc:E3", "fondare", "subordinata", cf=["S2"])])
    checked2 = S.verify(reply2, inp)
    assert checked2["side_mismatch"] is False and len(checked2["linee"]) == 1
    # no line at all is not a side mismatch either
    assert S.verify(Z._reply([]), inp)["side_mismatch"] is False
    assert "{lato}" in S.config()["reask"]


def test_item2_the_reask_counts_once_and_the_second_reply_stands(monkeypatch):
    inp = Z._inp(side="attore")
    seen = []

    def call(messages, kwargs):
        seen.append(messages[-1].content)
        if len(seen) == 1:
            return json.dumps(Z._reply([Z._linea("A1385cc", "A1385cc:E1", "contestare", "principale", cf=["S2"]),
                                        Z._linea("A1385cc", "A1385cc:E3", "fondare", "subordinata", cf=["S2"])]))
        return json.dumps(Z._reply([Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", cf=["S2"]),
                                    Z._linea("A1385cc", "A1385cc:E2", "contestare", "subordinata", cf=["S3"])]))
    res = S.analyse(inp, call=call)
    assert res["status"] == "ok" and len(seen) == 2 and "PARTE E LATO" in seen[1]
    c = res["counters"]
    assert c["strategy_reasked"] == 1 and c["strategy_side_reasked"] == 1 and c["strategy_lines"] == 2
    assert c["strategy_line_rejected"] == 2 and S.side_statement(inp) in res["reasons"]


def test_item2_the_state_input_carries_the_side_and_its_source(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    monkeypatch.setenv("CASE_FRAME", "true")
    state, verified = Z._state_for_input()
    verified["analysis"]["side"] = "neutro"
    state["case_frame"]["frame"]["addebito_o_domanda"] = {"tipo": "domanda", "contenuto": "esigere il doppio", "articolo": "1385 c.c."}
    inp = S.state_input(state, verified)
    assert inp["side"] == "attore" and inp["side_source"] == "pretesa_tipo:domanda"
    verified["analysis"]["side"] = "difesa"
    inp = S.state_input(state, verified)
    assert inp["side"] == "difesa" and inp["side_source"] == "frame"


# --- item 3 ---------------------------------------------------------------------------------------------------------------

def test_item3_the_frame_named_article_needs_a_verified_si_or_dipende(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    # the 640-for-lesioni shape: the frame names 640 c.p., the fit's verdict on it unverified -> no «pretesa» reason
    state = T173._state([T173._j("640 c.p.", verified=False, governs=None, has_elements=True), T173._j("590 c.p.", governs=["pretesa"])],
                        frame_article="640 c.p.", pretesa="accusa: «lesioni personali colpose» (art. 640 c.p.)")
    b = flow.binding_pretesa(state)
    assert b["reasons"]["640 c.p."] == ["fit_non_verificato"] and b["principale"] == ["590 c.p.", "640 c.p."]
    # no verdict at all on it (the fit never judged it): out of the set
    state = T173._state([T173._j("640 c.p.", fit=None, verified=False, governs=None, has_elements=True), T173._j("590 c.p.", governs=["pretesa"])],
                        frame_article="640 c.p.")
    b = flow.binding_pretesa(state)
    assert b["pretesa_article"] == "640 c.p." and "640 c.p." not in b["reasons"] and b["principale"] == ["590 c.p."]
    # a verified «dipende»: in, as the pretesa's article
    state = T173._state([T173._j("640 c.p.", fit="dipende", governs=[], has_elements=True), T173._j("590 c.p.", governs=["pretesa"])],
                        frame_article="640 c.p.")
    assert flow.binding_pretesa(state)["reasons"]["640 c.p."] == ["pretesa", "fit_dipende"]


# --- item 4 ---------------------------------------------------------------------------------------------------------------

def test_item4_the_fusions_sub_stages_are_marked_and_the_meter_names_them():
    src = (ROOT / "src" / "rag" / "defensive_generation.py").read_text(encoding="utf-8")
    for name in ("select:current", "select:searches", "select:vote", "select:fetch", "select:crossref", "select:spotted"):
        assert f'stage_meter.mark("{name}")' in src
    meter = stage_meter.Meter()
    with meter.active():
        meter.mark("select")
        meter.mark("select:current")
        meter.count("graph", 2)
        meter.mark("select:searches")
        meter.count("embed", 3)
        meter.count("llm")
        meter.mark("grading")
    meter.stop()
    log = meter.as_log()
    assert "select_current_s" in log and log["select_current_graph"] == 2
    assert log["select_searches_embed"] == 3 and log["select_searches_llm"] == 1 and "grading_s" in log


def test_item4_the_replay_row_names_the_slowest_retrieval_sub_stages():
    state = {"timings": {"round1_stages": {"route_s": 0.1, "select_searches_s": 480.2, "grading_s": 30.1, "select_fetch_s": 4.0,
                                           "total_s": 514.4},
                         "round1_counts": {"select_searches_llm": 3, "select_searches_embed": 12, "select_searches_graph": 40,
                                           "grading_llm": 8}},
             "block_counters": {}}
    got = R.retrieval_stages(state)
    assert [x["stage"] for x in got] == ["select_searches", "grading", "select_fetch", "route"]      # total left out
    assert got[0] == {"stage": "select_searches", "s": 480.2, "llm": 3, "embed": 12, "graph": 40}
    text = R.retrieval_stages_text(got)
    assert text.startswith("select_searches 480.2 (llm 3, embed 12, graph 40); grading 30.1 (llm 8); select_fetch 4.0; route 0.1")
    assert R.retrieval_stages({}) == [] and R.retrieval_stages_text([]) == "—"
    assert R.retrieval_stages(state, n=2) == got[:2]
    row = R.measure("c", state, "x", 1.0)
    assert row["retrieval_stages"] == got and "| retrieval stages (s) | select_searches 480.2 (llm 3, embed 12, graph 40);" in R.case_table(row)


def test_item4_the_flow_keeps_the_sub_stages_call_counts_beside_the_seconds():
    src = (ROOT / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert '"round1_counts": {k: v for k, v in round1_stages.items()' in src


# --- item 5 ---------------------------------------------------------------------------------------------------------------

def _row(num, code="c.c.", **more):
    return {"_article": (num, code), "s": {"titolo_articolo": f"Art. {num}"}, **more}


def test_item5_the_matter_gate_keeps_the_top_k_of_a_further_area_demoted_and_drops_the_rest():
    # the case's matter: contracts in general (1385 c.c., two signals). The pretesa (the frame's article 2043 c.c.)
    # touches the torts' matter: its top 3 hits stay, demoted; the fourth goes; a hit outside every area goes as before
    rows = [_row("1385", _multi_signal=True), _row("2043"), _row("2051"), _row("2052"), _row("2053"), _row("2054"),
            _row("844"), _row("1453")]
    log = {}
    out = TG.matter_gate(rows, keep=[], log=log, every=True, touched=["2043 c.c."])
    kept = [TG._label(TG._ref(r)) for r in out]
    assert kept == ["1385 c.c.", "2043 c.c.", "2051 c.c.", "2052 c.c.", "1453 c.c."]
    assert [x["article"] for x in log["demoted"]] == ["2043 c.c.", "2051 c.c.", "2052 c.c."]
    assert all(x["area"] == "c.c. IV altre fonti e fatti illeciti" for x in log["demoted"])
    assert [x["article"] for x in log["dropped"]] == ["2053 c.c.", "2054 c.c.", "844 c.c."]
    assert log["matters"] == ["c.c. IV contratti in generale"]
    assert log["areas"] == ["c.c. IV altre fonti e fatti illeciti", "c.c. IV contratti in generale"]
    assert all(r.get("_demoted") == "c.c. IV altre fonti e fatti illeciti" for r in out if r["_article"][0] in ("2043", "2051", "2052"))
    assert not any(r.get("_demoted") for r in out if r["_article"][0] in ("1385", "1453"))


def test_item5_a_quesitos_row_opens_its_area_and_a_single_area_case_behaves_as_before():
    # a quesito brought 2932 c.c. (esecuzione specifica, matter «c.c. VI tutela dei diritti»): its area's other hits stay, demoted
    rows = [_row("1351", _multi_signal=True), _row("2932", _quesito=[1]), _row("2930"), _row("2931"), _row("844")]
    log = {}
    out = TG.matter_gate(rows, keep=["2932 c.c."], log=log, every=True, touched=[""])
    assert [TG._label(TG._ref(r)) for r in out] == ["1351 c.c.", "2932 c.c.", "2930 c.c.", "2931 c.c."]
    assert [x["article"] for x in log["demoted"]] == ["2930 c.c.", "2931 c.c."]       # the quesito's own row is kept, not demoted
    assert not out[1].get("_demoted") and out[2]["_demoted"] == TG.matter_of(("2932", "c.c."))
    assert [x["article"] for x in log["dropped"]] == ["844 c.c."]
    # one area only: the 79.8 behaviour, nothing demoted, the log says one area
    rows = [_row("1385", _multi_signal=True), _row("2043"), _row("844")]
    log = {}
    out = TG.matter_gate(rows, keep=[], log=log, every=True)
    assert [TG._label(TG._ref(r)) for r in out] == ["1385 c.c."] and log["demoted"] == [] and len(log["areas"]) == 1
    # no matter known: nothing dropped, nothing demoted
    log = {}
    assert len(TG.matter_gate([_row("2043"), _row("844")], log=log, every=True, touched=["1385 c.c."])) == 2 and log["demoted"] == []
    assert TG.MATTER_AREA_TOP_K == 3


def test_item5_a_demoted_row_ranks_after_every_other_and_the_counters_say_how_many():
    rows = [{"_article": ("2043", "c.c."), "_fused_rank": 1, "_demoted": "c.c. IV altre fonti e fatti illeciti"},
            {"_article": ("1385", "c.c."), "_fused_rank": 7}, {"_article": ("1453", "c.c.")}]
    assert flow._rank_of(rows[0], 0) == (1, 1, 0) and flow._rank_of(rows[1], 1) == (0, 7, 1) and flow._rank_of(rows[2], 2)[0] == 0
    assert [t["article"] for t in flow.retrieval_top(rows)] == ["1385 c.c.", "1453 c.c.", "2043 c.c."]
    state = {"block_counters": {}, "analysis": {},
             "matter_gate": {"areas": ["a", "b"], "matters": ["a"], "demoted": [{"article": "2043 c.c.", "area": "b"}]}}
    flow.quesiti_counters(state)
    assert state["block_counters"]["topic_filter_areas"] == 2 and state["block_counters"]["topic_filter_demoted"] == 1
    state = {"block_counters": {}, "analysis": {}}
    flow.quesiti_counters(state)
    assert state["block_counters"]["topic_filter_areas"] == 0 and state["block_counters"]["topic_filter_demoted"] == 0
    assert "topic_filter_areas" in R.RETRIEVAL_COUNTERS and "topic_filter_demoted" in R.RETRIEVAL_COUNTERS


def test_item5_the_flow_passes_the_frames_charge_as_a_touched_area():
    src = (ROOT / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert 'analysis["frame_charge"] = (case_frame.charge_label(frame) if frame else "") or ""' in src
    assert 'touched=[str(analysis.get("frame_charge") or "")]' in src


# --- item 6 ---------------------------------------------------------------------------------------------------------------

def test_item6_the_shared_titolo_or_capo_of_two_articles():
    assert "Titolo XII, Capo I" in TG.same_code_unit(("581", "c.p."), ("584", "c.p."))
    assert "Titolo XIII, Capo II" in TG.same_code_unit(("640", "c.p."), ("646", "c.p."))
    assert TG.same_code_unit(("575", "c.p."), ("640", "c.p.")) == ""           # different Titoli
    assert TG.same_code_unit(("640", "c.p."), ("640", "c.p.p.")) == ""         # different codes
    assert "vendita" in TG.same_code_unit(("1470", "c.c."), ("1498", "c.c."))   # the contracts' Capo
    assert TG.same_code_unit(("1470", "c.c."), ("1571", "c.c.")) == ""          # vendita vs locazione
    assert TG.same_code_unit(("2043", "c.c."), ("1218", "c.c.")) == ""          # general parts: no unit
    assert TG.same_code_unit(("2043", "c.c."), ("2051", "c.c.")) == ""          # 2043-2059: a general part of the table
    assert TG.same_code_unit((), ("584", "c.p.")) == "" and TG.same_code_unit(("584", "c.p."), None) == ""


def test_item6_the_fits_no_in_the_charges_unit_is_kept_as_an_alternative_never_removed():
    res = {"judgments": [{"article": "581 c.p.", "fit": "no", "verified": True, "perche": "x"},
                         {"article": "2043 c.c.", "fit": "no", "verified": True, "perche": "y"},
                         {"article": "582 c.p.", "fit": "no", "verified": False},
                         {"article": "584 c.p.", "fit": "no", "verified": True}]}
    unit = lambda label: "c.p. Libro II, Titolo XII, Capo I – …" if label.split()[0] in ("581", "582", "584") else ""   # noqa: E731
    done = ap.effects(res, ["584 c.p."], alternative=unit)
    assert [e["article"] for e in done["kept_alternative"]] == ["581 c.p."] and done["kept_alternative"][0]["alternative"].startswith("c.p. ")
    assert [e["article"] for e in done["removed"]] == ["2043 c.c."] and [e["article"] for e in done["kept_named"]] == ["584 c.p."]
    assert res["judgments"][0]["alternative"].startswith("c.p. Libro II") and "alternative" not in res["judgments"][1]
    # without the hook: 79.18.1's effects
    done = ap.effects(res, [])
    assert done["kept_alternative"] == [] and [e["article"] for e in done["removed"]] == ["581 c.p.", "2043 c.c.", "584 c.p."]


def test_item6_the_binding_carries_the_alternativa_tier_outside_the_set(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = T173._state([T173._j("584 c.p.", governs=["pretesa"], has_elements=True),
                         T173._j("581 c.p.", fit="no", governs=[], has_elements=True, alternative="c.p. Libro II, Titolo XII, Capo I – …"),
                         T173._j("2043 c.c.", fit="no", governs=[])], frame_article="584 c.p.")
    b = flow.binding_pretesa(state)
    assert b["principale"] == ["584 c.p."] and b["alternative"] == ["581 c.p."]
    assert b["tiers"] == {"584 c.p.": "si_verificato", "581 c.p.": "alternativa"} and "581 c.p." not in b["reasons"]
    assert state["block_counters"]["binding_set_alternative"] == 1 and state["block_counters"]["binding_set_size"] == 1
    assert flow.TIER_ALTERNATIVA == "alternativa" and "binding_set_alternative" in R.FRAME_COUNTERS
    assert "fit_no_kept_alternative" in R.FIT_COUNTERS


def _alt_inp():
    inp = C._penal_inp()
    inp["pretesa"], inp["charge"] = "accusa: «omicidio preterintenzionale» (art. 584 c.p.) contro Paolo Sarti", "584 c.p."
    inp["articles"] = [
        {"article": "584 c.p.", "token": "584cp", "rubrica": "Omicidio preterintenzionale", "tier": "si_verificato", "in_set": True,
         "valutato": True, "elements": [Z._el("A584cp:E1", "con atti diretti a commettere percosse o lesioni", "da_provare", 1),
                                        Z._el("A584cp:E2", "cagiona la morte di un uomo", "provato", 1, ["S1"])]},
        {"article": "581 c.p.", "token": "581cp", "rubrica": "Percosse", "tier": "alternativa", "in_set": False, "alternative": True,
         "valutato": True, "elements": [Z._el("A581cp:E1", "percuote taluno", "contestato", 1, ["S5"])]}]
    inp["lines"] = [{"kind": "linea_principale", "articolo": "584 c.p.", "eid": "A584cp:E1", "stato": "verificata"}]
    inp["rulings"] = []
    return inp


def test_item6_an_alternativa_article_admits_a_riqualificare_line_only_and_never_leads():
    inp = _alt_inp()
    assert [a["article"] for a in S.alternative_articles(inp)] == ["581 c.p."]
    ob = {"testo": "L'accusa obietterà.", "risposta_cita_fatti": ["S5"], "risposta_cita_elementi": ["A584cp:E1"], "cita_pronunce": []}
    # «contestare» on the alternative: rejected, the reason names the tier
    checked = S.verify(Z._reply([C._l("A584cp", "A584cp:E1", "contestare", "principale", cf=["S5"], ob=ob),
                                 C._l("A581cp", "A581cp:E1", "contestare", "subordinata", cf=["S5"], ob=ob)]), inp)
    assert len(checked["linee"]) == 1 and "alternativa" in checked["rejected"][0]["why"] and "solo «riqualificare»" in checked["rejected"][0]["why"]
    assert any("riqualificare" in r and "A581cp" in r for r in checked["reasons"])        # C3: the riqualificazione still asked
    # «riqualificare» on it: accepted; the defence's forza would put it first (contestato by the client's own fact), the
    # code keeps the set's line as principale
    checked = S.verify(Z._reply([C._l("A584cp", "A584cp:E1", "contestare", "principale", cf=["S5"], ob=ob),
                                 C._l("A581cp", "A581cp:E1", "riqualificare", "subordinata", cf=["S5"], ob=ob)]), inp)
    assert len(checked["linee"]) == 2 and checked["linee"][1]["alternative"] is True and checked["reasons"] == []
    a = S.finish(checked, inp)
    assert a["principale"]["articolo"] == "584 c.p." and a["linee"][1]["articolo"] == "581 c.p."
    assert tuple(a["linee"][1]["forza"]["key"]) > tuple(a["linee"][0]["forza"]["key"])    # stronger, yet second
    assert checked["counters"]["strategy_alternativa_demoted"] == 1 and a["reordered"] is True
    assert a["riqualificazione"]["articolo"] == "581 c.p." and a["riqualificazione"]["n"] == 2
    assert a["alternative"] == ["581 c.p."] and a["side_source"] == ""
    summary = S.replay_summary({"status": "ok", "analysis": a, "counters": checked["counters"]})
    assert summary["alternative"] == ["581 c.p."] and summary["side"] == "difesa"
    assert R.strategy_fields({"strategy": summary})["alternative"] == ["581 c.p."]
    # the prompt and the allowed list say it
    human = S.build_messages(inp)[-1].content
    assert "A581cp: art. 581 c.p. (Percosse) [alternativa] — alternativa (fit «no»" in human
    assert "A581cp (art. 581 c.p.) — alternativa (fit «no»" in S.allowed_text(inp)
    assert any("alternativa" in line and "riqualificare" in line for line in S.config()["prompt_system"])
    assert "strategy_alternativa_demoted" in S.COUNTERS and "strategy_side_reasked" in S.COUNTERS
    assert "strategy_alternativa_demoted" in R.STRATEGY_COUNTERS and "strategy_side_reasked" in R.STRATEGY_COUNTERS


def test_item6_the_state_input_reads_the_binding_alternatives(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    state, verified = Z._state_for_input()
    norms = state["analysis_norms"]
    norms.append({"article": "2056 c.c.", "rubrica": "Valutazione dei danni", "elements": [{"eid": "A2056cc:E1", "testo": "il risarcimento dovuto al danneggiato", "comma": 1}]})
    state["binding_pretesa"]["alternative"] = ["2056 c.c."]
    state["binding_pretesa"]["tiers"]["2056 c.c."] = "alternativa"
    inp = S.state_input(state, verified)
    alt = next(a for a in inp["articles"] if a["article"] == "2056 c.c.")
    assert alt["alternative"] is True and alt["in_set"] is False and alt["tier"] == "alternativa"
    assert all(not a.get("alternative") for a in inp["articles"] if a["article"] != "2056 c.c.")
    # alternatives alone are nothing to strategise on
    state["binding_pretesa"]["principale"] = []
    verified["analysis"]["linea_principale"] = verified["analysis"]["linea_subordinata"] = None
    assert S.state_input(state, verified) is None
