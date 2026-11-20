"""v79.15.2 hotfix -- the analysis step's article key (src/rag/analysis_step.py, data/analysis_step.json).

The server's model returned a line's «articolo» in the id form the prompt and the re-ask send («640cp», «1667cc»); 79.15.1
compared it with the display label («640 c.p.») and rejected the line («articolo non tra le norme applicabili: 640cp»).
1. one canonical key (art_key) for every article comparison -- one test per accepted form, the code distinguished, the
   remedy set by its id form, an unknown code, the E-id prefix, the frame's charge, the prose review's sets
2. analysis_fallback_skipped_reason: why the rubrica fallback did not run, a string in the counters, the analysis and
   the «analysis_step» line -- each code; the server's two runs reproduced synthetically end to end
3. the nightly prints the string counter as written

The model faked at its source; every other model path and every outbound socket refused (tests/test_v79_15_0_analysis.py's
sealing for the unit tests, tests/_fixture_79_15_0.py for the end-to-end runs). Synthetic facts and invented names only.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_15_0_fixture_penal as PENAL  # noqa: E402
import test_v79_15_1_fixtures as X  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import verbose_logger  # noqa: E402

_Model, _reply, _line = T._Model, T._reply, T._line


@pytest.fixture
def sealed(monkeypatch, tmp_path):
    """The unit tests' sealing (test_v79_15_0_analysis._sealed) -- not autouse here: the end-to-end runs seal
    themselves (H.run)."""
    from src.rag import ai_chat, applicability as ap, case_frame as cf
    import socket
    monkeypatch.setattr(socket.socket, "connect", T._refused)
    monkeypatch.setattr(socket, "create_connection", T._refused)
    monkeypatch.setattr(ai_chat, "ChatOpenAI", T._refused)
    for name, mod in list(sys.modules.items()):
        if name.startswith("src.rag.") and mod is not None:
            for attr in ("_call_chat", "_call_chat_with_tools", "embedding_model", "_embed_query_with_prefix"):
                if hasattr(mod, attr):
                    monkeypatch.setattr(mod, attr, T._refused)
    monkeypatch.setattr(A, "_default_call", T._refused)
    monkeypatch.setattr(ap, "_default_call", T._refused)
    monkeypatch.setattr(cf, "_default_call", T._refused)
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


LABELS = ["640 c.p.", "641 c.p.", "62-bis c.p.", "640 c.p.p.", "1385 c.c.", "1453 c.c."]


# --- 1. one canonical key ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["A640cp", "640cp", "640 c.p.", "art. 640 c.p.", "Art. 640 cp", "A640CP",
                                  "articolo 640 c.p. (Truffa)", "640 Codice Penale 2026"])
def test_each_accepted_form_is_the_one_key_and_resolves_to_the_display_label(form):
    assert A.art_key(form) == "640cp" and A.art_parts(form) == ("640", "cp")
    assert A.resolve(form, LABELS) == "640 c.p."
    assert A.same_article(form, "640 c.p.") and A.same_article("640 c.p.", form)


@pytest.mark.parametrize("form", ["62-bis c.p.", "62 bis c.p.", "62biscp", "A62biscp", "art. 62-bis c.p."])
def test_a_suffixed_number_is_one_key_however_written(form):
    assert A.art_key(form) == "62biscp" and A.resolve(form, LABELS) == "62-bis c.p."
    assert not A.same_article(form, "62 c.p.")


def test_the_code_is_distinguished_cp_cpp_cc_cpc():
    keys = {A.art_key(x) for x in ("640 c.p.", "640 c.p.p.", "640 c.c.", "640 c.p.c.")}
    assert keys == {"640cp", "640cpp", "640cc", "640cpc"}
    assert not A.same_article("640 c.p.p.", "640 c.p.") and not A.same_article("A640cpp", "640cp")
    assert A.resolve("640 c.p.p.", LABELS) == "640 c.p.p." and A.resolve("A640cpp", LABELS) == "640 c.p.p."
    assert A.resolve("640 c.p.p.", ["640 c.p."]) is None and A.resolve("640cpp", ["640 c.p."]) is None
    assert A.art_key("640 cod. proc. pen.") == "640cpp" and A.art_key("1385 Codice Civile 2026") == "1385cc"


def test_an_unknown_code_matches_nothing_else():
    assert A.art_key("640 xyz") == "640xyz" and A.resolve("640 xyz", LABELS) is None
    assert not A.same_article("640 xyz", "640 c.p.") and not A.is_remedy("1453 xyz") and not A.is_remedy("41xyz")
    assert A.art_key("con artifizi o raggiri") == "" and A.resolve("", LABELS) is None
    assert not A.same_article("", "") and not A.same_article("E1", "E1")


def test_a_bare_number_resolves_only_when_one_norm_has_it():
    assert A.resolve("641", LABELS) == "641 c.p." and A.resolve("640", LABELS) is None     # 640 c.p. and c.p.p.


def test_a_commas_own_label_before_its_articles():
    labels = ["52 c.p.", "52 c.p. co. 2"]
    assert A.art_key("52 c.p. co. 2") == A.art_key("A52cpco2") == "52cp"
    assert A.resolve("A52cpco2", labels) == "52 c.p. co. 2" and A.resolve("52 c.p.", labels) == "52 c.p."
    assert A.resolve("A52cp", list(reversed(labels))) == "52 c.p." and A.art_key("62 n. 2 c.p.") == "62cp"


@pytest.mark.parametrize("label,remedy", [("1453cc", True), ("A1453cc", True), ("1453 c.c.", True), ("A41cp", True),
                                          ("41cp", True), ("A62biscp", True), ("60cp", True), ("A133cp", True),
                                          ("A63cp", False), ("41cpp", False), ("1453cp", False), ("A640cp", False)])
def test_the_remedy_set_by_its_id_form_and_its_ranges(label, remedy):
    assert A.is_remedy(label) is remedy


def test_the_contractual_only_remedy_and_the_contractual_frame_by_the_key():
    assert A.is_remedy("2043cc", contractual=True) and not A.is_remedy("2043cc")
    assert A.is_contractual({"rito": {"valore": "civile"}, "addebito_o_domanda": {"articolo": "1385cc"}})
    assert not A.is_contractual({"rito": {"valore": "civile"}, "addebito_o_domanda": {"articolo": "1385cp"}})


@pytest.mark.parametrize("eid", ["A640cp:E2", "640cp:E2", "640 c.p.:E2", "art. 640 c.p.:E2", "«A640cp:E2»", "E2"])
def test_an_e_id_prefix_in_every_form_names_the_element(eid, sealed):
    inp = T._penal()
    norm = inp["norms"][0]
    labels = [n["article"] for n in inp["norms"]]
    assert A.element_of(norm, eid, labels) is norm["elements"][1]
    assert A.element_of(norm, eid) is norm["elements"][1]
    if eid != "E2":
        assert A.parse_eid(eid) == ("640cp", 2)


def test_an_e_id_of_another_article_or_code_is_not_this_norms(sealed):
    inp = T._penal()
    norm, labels = inp["norms"][0], [n["article"] for n in inp["norms"]]
    for eid in ("A640cpp:E1", "A641cp:E1", "641cp:E1", "A640cp:E9", "640cp:E0", "A640cp.1", "inducendo taluno"):
        assert A.element_of(norm, eid, labels) is None, eid


def _id_form(reply, line_form="640cp", element_form="A640cp"):
    out = json.loads(json.dumps(reply))
    for kind in A.LINE_KINDS:
        if isinstance(out.get(kind), dict):
            out[kind]["articolo"] = line_form
    for it in out.get("elementi") or []:
        it["articolo"] = element_form
    return out


@pytest.mark.parametrize("form", ["A640cp", "640cp", "640 c.p.", "art. 640 c.p.", "Art. 640 cp"])
def test_a_line_in_every_form_is_verified_on_the_first_call_and_logged_as_the_label(form, sealed):
    model = _Model(_id_form(T.GOOD_PENAL, form, form))
    res = A.analyse(T._penal(), call=model)
    a = res["analysis"]
    assert res["status"] == "ok" and len(model.seen) == 1 and res["rejected"] == []
    assert a["linea_principale"]["articolo"] == "640 c.p." and a["linea_principale"]["stato"] == A.VERIFIED
    assert a["frame_charge_missing"] is False and res["counters"]["analysis_frame_charge_missing"] == 0
    assert res["counters"]["analysis_element_missing"] == 0 and list(a["elementi"]) == ["640 c.p."]
    assert res["counters"]["analysis_fallback_skipped_reason"] == A.SKIP_LINE_VERIFIED


def test_a_rejected_line_is_logged_by_its_label_once_resolved_else_as_written(sealed):
    rem = _reply(_line("1453cc", T.E1453[0], "La risoluzione.", ["S2"]))
    rem["linea_principale"]["elemento_in_gioco"] = "A1453cc:E1"
    unknown = _reply(_line("9999cc", "E1", "Altro.", ["S2"]))
    res = A.analyse(T._civil(), call=_Model(rem, unknown))
    assert [(r["articolo"], r["attempt"]) for r in res["rejected"]] == [("1453 c.c.", 1), ("9999cc", 2)]
    assert "rimedio o parte generale" in res["rejected"][0]["why"]
    assert res["rejected"][1]["why"] == "articolo non tra le norme applicabili: 9999cc"


def test_the_frames_charge_in_the_id_form_is_the_charge(sealed):
    frame = dict(T.PENAL_FRAME, addebito_o_domanda={"articolo": "640 c.p.", "descrizione": "truffa", "frase": "x"})
    inp = T._penal(frame=frame)
    assert inp["charge"] == "640 c.p."
    checked = A.verify(_id_form(T.GOOD_PENAL, "A640cp"), inp)
    assert checked["charge_missing"] is False and checked["lines"]["linea_principale"]["articolo"] == "640 c.p."


def test_the_prose_reviews_sets_compare_by_the_key():
    scope = {"excluded": ["1439 c.c."], "fitted": ["640 c.p."], "named": ["641 c.p."], "exempt_codes": ["cpp", "cpc"]}
    assert A.outside(["A1439cc"], scope) == (A.ARTICLE_EXCLUDED, ["A1439cc"])
    assert A.outside(["640cp", "art. 641 c.p."], scope) == ("", [])
    assert A.outside(["640 c.c."], scope) == (A.ARTICLE_UNFITTED, ["640 c.c."])
    assert A.outside(["640 c.p.p."], scope) == ("", [])                    # a procedural code: exempt
    assert A._among("A640cp", ["640 c.p."]) and not A._among("640cpp", ["640 c.p."])


# --- review: the Q&A counterpart by number AND code; a law's own number; a period id; a preposition before the code -

_STATEMENT = {"statements": [{"testo": "L'art. 640 c.p. punisce chi con artifizi o raggiri induce taluno in errore "
                                       "procurandosi un ingiusto profitto", "articoli": ["640 c.p."]}]}
_RULE = "L'art. 640 {code} punisce chi con artifizi o raggiri induce taluno in errore procurandosi un ingiusto profitto."


@pytest.mark.parametrize("code,match", [("c.p.", True), ("cod. pen.", True), ("c.p.p.", False), ("c.c.", False),
                                        ("c.p.c.", False), ("", True)])
def test_the_qa_counterpart_compares_number_and_code(code, match):
    """Review: 79.15.1's counterpart compared the numbers alone -- a rule on «640 c.p.p.» or «640 c.c.» was matched to
    the statement on «640 c.p.» and replaced by it; now only the same article (or a citation with no code) matches."""
    sentence = _RULE.format(code=code).replace("640 .", "640.").replace("640  ", "640 ")
    assert (A.counterpart(sentence, _STATEMENT) is not None) is match
    assert A.counterpart(_RULE.format(code="c.p.").replace("640", "641"), _STATEMENT) is None


def test_the_qa_counterpart_by_a_bare_labels_code():
    """A sentence naming «la norma» with the article from bare_cited: a fitted label decides by its code, «art. 640»
    (no code) by its number."""
    sentence = "La norma punisce chi con artifizi o raggiri induce taluno in errore procurandosi un ingiusto profitto."
    assert A.counterpart(sentence, _STATEMENT, ["640 c.p."]) is not None
    assert A.counterpart(sentence, _STATEMENT, ["art. 640"]) is not None
    assert A.counterpart(sentence, _STATEMENT, ["640 c.p.p."]) is None
    assert A._cites_same({("640", "cp")}, {("640", "cp")}) and not A._cites_same({("640", "cpp")}, {("640", "cp")})


@pytest.mark.parametrize("one,other", [("1 L. n. 241/1990", "1 L. n. 300/1990"),
                                       ("2 legge n. 689/1981", "2 legge n. 300/1981"),
                                       ("3 d.lgs. n. 81/2008", "3 d.lgs. n. 82/2008")])
def test_a_laws_own_number_is_kept_two_laws_of_one_year_stay_apart(one, other):
    """Review: 79.15.2's first cut stripped «n. 241» as a numero, so two laws of one year shared a key."""
    assert A.art_key(one) != A.art_key(other)
    assert not A.same_article(one, other) and A.resolve(other, [one]) is None and A.resolve(one, [one]) == one


def test_a_law_in_its_id_form_and_spelt_out_is_the_label():
    from src.rag.applicability import article_token
    assert A.art_key("1 L. n. 241/1990") == A.art_key("1 L. 241/1990") == A.art_key("1 legge n. 241/1990") == "1l2411990"
    assert A.art_key("A" + article_token("1 L. n. 241/1990")) == "1l2411990"
    assert A.resolve("A" + article_token("1 L. n. 241/1990"), ["1 L. n. 241/1990"]) == "1 L. n. 241/1990"
    assert A.art_key("3 decreto legislativo n. 81/2008") == A.art_key("3 d.lgs. n. 81/2008") == "3dlgs812008"
    assert A.art_key("62 n. 2 c.p.") == A.art_key("A62n2cp") == "62cp"                 # a numero is still dropped


@pytest.mark.parametrize("form", ["A640cp.1", "640cp.1", "A640cp.12"])
def test_a_period_id_given_as_articolo_is_its_article(form):
    assert A.art_key(form) == "640cp" and A.resolve(form, LABELS) == "640 c.p."


def test_a_period_id_of_a_commas_label_and_a_label_with_dots_are_not_cut():
    assert A.resolve("A52cpco2.1", ["52 c.p.", "52 c.p. co. 2"]) == "52 c.p. co. 2"
    assert A.art_key("640 c.p.") == "640cp" and A.art_key("2 L. 300/1970") == "2l3001970"


@pytest.mark.parametrize("form", ["640 del codice penale", "640 del c.p.", "art. 640 della c.p.", "640 di cod. pen."])
def test_a_preposition_before_the_code_is_skipped(form):
    assert A.art_key(form) == "640cp" and A.resolve(form, LABELS) == "640 c.p."
    assert A.art_key("640 del codice di procedura penale") == "640cpp"


def test_a_law_statement_and_an_element_verdict_in_the_id_form_are_read(sealed):
    reply = _id_form(T.GOOD_PENAL, "640cp", "A640cp")
    reply["elementi"][0].update(esito="provato", cita_fatti=["S7"])
    reply["affermazioni_di_diritto"] = [{"testo": "L'art. 640 c.p. punisce chi, con artifizi o raggiri, inducendo "
                                                  "taluno in errore, procura a sé un ingiusto profitto con altrui "
                                                  "danno.", "cita_norma": ["A640cp.1"]}]
    a = A.analyse(T._penal(), call=_Model(reply))["analysis"]
    assert a["elementi"]["640 c.p."][0]["esito"] == "provato"
    assert [s["articoli"] for s in a["affermazioni_di_diritto"]] == [["640 c.p."]]


# --- 2. analysis_fallback_skipped_reason ------------------------------------------------------------------------------

def _no_charge(descrizione=None, istituto=None):
    frame = {k: v for k, v in T.PENAL_FRAME.items() if k != "addebito_o_domanda"}
    if descrizione:
        frame["addebito_o_domanda"] = {"articolo": None, "descrizione": descrizione, "frase": "«procede per truffa»"}
    if istituto:
        frame["istituto"] = {"testo": istituto, "frase": "x"}
    return T._penal(frame=frame)


BAD_LINE = _reply(_line("640cp", "con artifizi o raggiri, scritto", "Gli artifizi non sono provati.", ["S5"]))


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append((event,
                                                                                                          data)))
    return events


@pytest.mark.parametrize("case,reason,fired", [
    ("verified", A.SKIP_LINE_VERIFIED, 0),
    ("frame_charge", A.SKIP_FRAME_CHARGE, 0),
    ("no_named_charge", A.SKIP_NO_NAMED_CHARGE, 0),
    ("no_rubrica_match", A.SKIP_NO_RUBRICA_MATCH, 0),
    ("fired", "", 1),
    ("failed", A.SKIP_STEP_FAILED, 0),
    ("no_input", A.SKIP_NO_INPUT, 0)])
def test_the_reason_the_rubrica_fallback_did_not_run(case, reason, fired, monkeypatch, sealed):
    events = _events(monkeypatch)
    inp, replies = {
        "verified": (_no_charge("truffa"), [_id_form(T.GOOD_PENAL)]),
        "frame_charge": (T._penal(), [BAD_LINE, BAD_LINE]),
        "no_named_charge": (_no_charge(), [BAD_LINE, BAD_LINE]),
        "no_rubrica_match": (_no_charge("truffa ai danni di Giulia Neri", "vendita online"), [BAD_LINE, BAD_LINE]),
        "fired": (_no_charge("truffa"), [BAD_LINE, BAD_LINE]),
        "failed": (T._penal(), [RuntimeError("down")]),
        "no_input": (T._penal(norms=[]), []),
    }[case]
    res = A.analyse(inp, call=_Model(*replies))
    c = res["counters"]
    assert c["analysis_fallback_skipped_reason"] == reason and c["analysis_line_fallback_rubrica"] == fired
    assert isinstance(c["analysis_fallback_skipped_reason"], str)
    line = [d for e, d in events if e == "analysis_step"][0]
    assert line["fallback_skipped_reason"] == reason and list(line)[:3] == ["prompt_hash", "status",
                                                                           "fallback_skipped_reason"]
    if res["analysis"]:
        assert res["analysis"]["fallback_skipped_reason"] == reason
    if case == "frame_charge":
        assert c["analysis_frame_charge_missing"] == 1
        assert res["analysis"]["linea_principale"]["stato"] == A.FRAME_LINE
    if case in ("no_named_charge", "no_rubrica_match"):
        assert c["analysis_line_undetermined"] == 1
    if case == "fired":
        assert res["analysis"]["linea_principale"]["stato"] == A.FALLBACK_LINE


def test_the_reason_survives_the_numeric_sum_and_a_failed_input(monkeypatch, sealed):
    assert A.empty_counters()["analysis_fallback_skipped_reason"] == "" and "analysis_fallback_skipped_reason" in \
        A.COUNTERS
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    events = _events(monkeypatch)
    monkeypatch.setattr(A, "inputs", lambda state: (_ for _ in ()).throw(ValueError("unreadable")))
    res = A.run({}, call=_Model())
    assert res["status"] == "failed" and res["counters"]["analysis_fallback_skipped_reason"] == A.SKIP_STEP_FAILED
    assert [d for e, d in events if e == "analysis_step"][0]["fallback_skipped_reason"] == A.SKIP_STEP_FAILED


# --- the server's two runs, reproduced synthetically end to end ---------------------------------------------------------

def _clone(ns, **kw):
    d = {k: getattr(ns, k) for k in dir(ns) if not k.startswith("__")}
    d.update(kw)
    return types.SimpleNamespace(**d)


def _truffa_id(parsed, attempt):
    """The truffa line as the server's model gave it: articolo «640cp», the element by its E-id."""
    norm, sale = parsed["norms"]["640 c.p."], PENAL._sid(parsed["facts"], "Bruno Gelso ha venduto")
    return {"linea_principale": {"articolo": "640cp", "elemento_in_gioco": norm["eids"][0],
                                 "tesi": "Gli artifizi non sono provati.", "cita_fatti": [sale],
                                 "cita_norma": [norm["periods"][0]]},
            "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}


def _truffa_frame(descrizione):
    frame = json.loads(json.dumps(X.TR_FRAME))
    frame["addebito_o_domanda"] = {"articolo": None, "descrizione": descrizione,
                                   "frase": "«procede per truffa ai sensi dell'art. 640 c.p.»"}
    return frame


def test_server_penal_run_640cp_with_no_frame_article_now_verifies(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, _clone(X.TRUFFA, analysis_reply=_truffa_id, FRAME_REPLY=_truffa_frame("truffa"),
                                              OLDER_FRAME={}))
    c, step = run["state"]["block_counters"], run["by"]["analysis_step"][0][0]
    assert not run["state"]["case_frame"]["frame"]["addebito_o_domanda"]["articolo"]
    assert step["status"] == "ok" and step["rejected"] == [] and c["analysis_calls"] == 1
    assert step["lines"][0]["articolo"] == "640 c.p." and step["lines"][0]["stato"] == A.VERIFIED
    assert c["analysis_fallback_skipped_reason"] == step["fallback_skipped_reason"] == A.SKIP_LINE_VERIFIED
    assert H.section(run["answer"], "sintesi").startswith("Linea principale: art. 640 c.p. — Gli artifizi non sono "
                                                          "provati")


def test_server_civil_run_1667cc_with_no_frame_article_now_verifies(monkeypatch, tmp_path):
    frame = json.loads(json.dumps(X.AP_FRAME))
    frame["addebito_o_domanda"]["articolo"] = None
    run = H.run(monkeypatch, tmp_path, _clone(X.APPALTO, FRAME_REPLY=frame))       # X.APPALTO's reply: «1667cc»
    c, step = run["state"]["block_counters"], run["by"]["analysis_step"][0][0]
    assert step["lines"][0]["articolo"] == "1667 c.c." and step["lines"][0]["eid"] == "A1667cc:E1"
    assert [r["articolo"] for r in step["rejected"]] == ["1667 c.c."]       # the exception element, not the key
    assert c["analysis_line_undetermined"] == 0 and c["analysis_fallback_skipped_reason"] == A.SKIP_LINE_VERIFIED


def _truffa_rejected(parsed, attempt):
    """Both lines rejected on their element (a text, then an E-id 640 c.p. does not have) -- the fallback's turn."""
    got = _truffa_id(parsed, attempt)
    got["linea_principale"]["elemento_in_gioco"] = "artifizi" if attempt == 1 else "A640cp:E9"
    return got


@pytest.mark.parametrize("descrizione,reason,fired", [("truffa", "", 1),
                                                      ("truffa ai danni di Carla Tasso", A.SKIP_NO_RUBRICA_MATCH, 0),
                                                      (None, A.SKIP_NO_NAMED_CHARGE, 0)])
def test_with_no_verified_line_the_counter_says_why_the_fallback_did_not_run(descrizione, reason, fired, monkeypatch,
                                                                             tmp_path):
    run = H.run(monkeypatch, tmp_path, _clone(X.TRUFFA, analysis_reply=_truffa_rejected,
                                              FRAME_REPLY=_truffa_frame(descrizione), OLDER_FRAME={}))
    c, step = run["state"]["block_counters"], run["by"]["analysis_step"][0][0]
    assert [r["articolo"] for r in step["rejected"]] == ["640 c.p.", "640 c.p."]
    assert c["analysis_line_fallback_rubrica"] == fired and c["analysis_line_undetermined"] == 1 - fired
    assert c["analysis_fallback_skipped_reason"] == step["fallback_skipped_reason"] == reason
    assert run["state"]["analysis_step"]["counters"]["analysis_fallback_skipped_reason"] == reason


# --- 3. the nightly tolerates the string counter ----------------------------------------------------------------------

def test_the_nightly_prints_the_string_counter_as_written():
    spec = importlib.util.spec_from_file_location("nightly_sim", HERE.parent / "scripts" / "nightly_sim.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    assert "analysis_fallback_skipped_reason" in nightly.ANALYSIS_STEP_KEYS
    cells = [{"id": "a", "counters": {"analysis_calls": 2, "analysis_fallback_skipped_reason": "no_rubrica_match"},
              "analysis_step": {"status": "ok", "prompt_hash": "h"}},
             {"id": "b", "counters": {"analysis_calls": 1, "analysis_fallback_skipped_reason": ""}}]
    text = "\n".join(nightly.analysis_lines(cells))
    row_a = next(x for x in text.splitlines() if x.startswith("| a |"))
    row_b = next(x for x in text.splitlines() if x.startswith("| b |"))
    assert "| no_rubrica_match |" in row_a and row_b.count("| — |") >= 1
    base = {"cells": [{"id": "a", "counters": {"analysis_fallback_skipped_reason": "line_verified"}}]}
    assert any("line_verified → no_rubrica_match" in x for x in nightly.counter_lines(cells, base))
