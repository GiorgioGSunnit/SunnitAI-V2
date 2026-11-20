"""v79.14.3 -- finish the wiring of 79.14.2 (the model faked; synthetic texts and facts, invented names; statutory text
of 640 c.p., 1439 and 2043 c.c.):

1. allegations never integrate an element: an element only another party's allegation supports is «contestato»
   (status «da provare», never «integrato», never counted as proven), printed «contestato (allegazione di <parte>)»; a
   certain fact among its support (mixed) -- unchanged; the client's own statement -- no allegation for this purpose;
   for a defence client the contested elements are the primary theses («L'accusa deve provare «…»»)
2. the fit's dedicated fact-quote field «fatto_citato»: a «no» and a «dipende» verified on it (never on the perché's
   quotes), a «si» with "" as before; an unverified one keeps the norm
3. an empty plan with a verified charge: the theses built on the frame's charge article, never «Ammettere la condotta»
   under a frame, never the corpus line on a code article
4. excluded means excluded: an article the binding excluded is absent from §4 (schede and tables), §5, §7 and «Fatti
   mancanti» -- «Altri profili» only
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
sys.path.insert(0, str(HERE))

import test_v79_14_2 as T  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as IR  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402

_env = T._env           # no real model call, temporary schema files


# --- 1. allegations never integrate an element ---------------------------------------------------------------------------

ALLEGED = "La società sostiene che Bruno Gelso si è appropriato del denaro della cassa."
CERTAIN = "Bruno Gelso ha prelevato dalla cassa il denaro della società."
OWN = "Bruno Gelso afferma di essersi appropriato del denaro della cassa."


def _sents():
    sents = [{"id": f"S{i}", "text": t, "source": "caso", "certainty": IR.certainty(t), "verbatim": True}
             for i, t in enumerate((ALLEGED, CERTAIN, OWN), 1)]
    flow.mark_sides(sents, {"client": {"name": "Bruno Gelso"}, "analysis": {}})
    return sents


def _settled(supporting, status="provato"):
    req = {"id": "R1", "kind": "requisito", "text": "si appropria il denaro altrui", "comma": 1, "status": status,
           "facts_status": "da provare", "supporting": list(supporting), "link": {"supporting": list(supporting),
                                                                                 "contradicting": []}}
    IR.settle(req, {s["id"]: s for s in _sents()})
    return req


def test_the_speaker_of_an_allegation_and_the_clients_own_statement():
    s1, s2, s3 = _sents()
    assert s1["certainty"] == "allegato" and s1["alleged_by"] == "La società"
    assert s2["certainty"] == "accertato" and s2["alleged_by"] == ""
    assert s3["certainty"] == "allegato" and s3["alleged_by"] == ""         # the client's own: no other party's
    assert flow.speaker_phrase("Secondo la società, Gelso ha incassato il denaro.") == "la società"
    assert flow.speaker_phrase("Il pagamento è avvenuto.") == ""
    unknown = dict(s1)
    assert flow.alleging_party(unknown, {"analysis": {}}, [], "") == ""     # no client known: no other party


def _alleged(text, state):
    sents = [{"id": "S1", "text": text, "source": "caso", "certainty": IR.certainty(text), "verbatim": True}]
    flow.mark_sides(sents, state)
    req = {"id": "R1", "kind": "requisito", "text": "si appropria il denaro altrui", "comma": 1, "status": "provato",
           "facts_status": "da provare", "supporting": ["S1"], "link": {"supporting": ["S1"], "contradicting": []}}
    IR.settle(req, {"S1": sents[0]})
    return sents[0], req


@pytest.mark.parametrize("text", [
    "Lo stesso indagato dichiara di essersi appropriato del denaro della cassa.",
    "L'imputato ammette e dichiara di essersi appropriato del denaro della cassa.",
    "L'indagato, sentito dal pubblico ministero, dichiara di essersi appropriato del denaro della cassa.",
])
def test_the_client_named_by_his_procedural_role_is_his_own_statement_never_an_allegation(text):
    # (review) the client's role from intake (no frame) -- «imputato» covers «indagato» (data/case_frame.json ruoli)
    s, req = _alleged(text, {"client": {"name": "Bruno Gelso"}, "analysis": {}, "intake": {"role": "imputato"}})
    assert s["certainty"] == "allegato" and s["alleged_by"] == "" and not req.get("contested_by")
    assert req["status"] == "da provare"                                    # the existing rule, never «provato»
    assert "imputato" in flow.client_roles({"intake": {"role": "imputato"}}) and \
        "indagato" in flow.client_roles({"intake": {"role": "imputato"}})


def test_the_client_role_is_the_verified_frames_when_a_frame_is_held(monkeypatch):
    monkeypatch.setattr(flow, "client_of", lambda state: "Bruno Gelso")
    from src.rag import case_frame
    monkeypatch.setattr(case_frame, "held", lambda holder: {"frame": {}})
    monkeypatch.setattr(case_frame, "bound", lambda holder, field, posture=None:
                        "persona_offesa" if field == "ruolo" else None)
    state = {"client": {"name": "Bruno Gelso"}, "analysis": {}, "intake": {"role": "imputato"}}
    assert "persona offesa" in flow.client_roles(state) and "imputato" not in flow.client_roles(state)
    s, req = _alleged("L'indagato dichiara che Bruno Gelso gli ha consegnato il denaro.", state)
    assert s["alleged_by"] == "L'indagato" and req.get("contested_by") == "L'indagato"   # another party, here


@pytest.mark.parametrize("text,party,label", [
    ("L'azienda sostiene che Bruno Gelso si è appropriato del denaro della cassa.", "L'azienda", "dell'azienda"),
    ("L'amministratore sostiene che Bruno Gelso si è appropriato del denaro della cassa.", "L'amministratore",
     "dell'amministratore"),
    ("La società, con una lettera del 3 marzo, sostiene che Bruno Gelso si è appropriato del denaro della cassa.",
     "La società", "della società"),
    ("La società ammette e sostiene che Bruno Gelso si è appropriato del denaro della cassa.", "La società",
     "della società"),
    ("Il coindagato Rossi dichiara che Bruno Gelso si è appropriato del denaro della cassa.", "Il coindagato Rossi",
     "del coindagato Rossi"),
])
def test_an_elided_article_an_inciso_or_a_coordinated_verb_still_names_the_alleging_party(text, party, label):
    s, req = _alleged(text, {"client": {"name": "Bruno Gelso"}, "analysis": {}, "intake": {"role": "imputato"}})
    assert s["alleged_by"] == party and req.get("contested_by") == party and req["status"] == "da provare"
    esito = M.subsumption_rows({"article": "646 c.p.", "requisites": [req]}, {"S1": s})[0][2]
    assert esito == f"contestato (allegazione {label})"


def test_an_element_only_an_allegation_supports_is_contestato_never_integrato():
    req = _settled(["S1"])
    assert req["status"] == "da provare" and IR.contested(req) and req["contested_by"] == "La società"
    assert req["contested_ids"] == ["S1"] and "allegazione della società" in req["why"]
    rec = {"article": "646 c.p.", "rubrica": "Appropriazione indebita", "requisites": [req]}
    rows = M.subsumption_rows(rec, {s["id"]: s for s in _sents()})
    assert rows[0][2] == "contestato (allegazione della società)"
    assert rows[0][1].startswith("nessun fatto certo; solo l'allegazione: La società sostiene")
    assert M._counts(rec) == (0, 0, 1)                                     # never counted as proven
    table = M.subsumption_table("646 c.p.", rec, {s["id"]: s for s in _sents()})
    assert "| contestato (allegazione della società) |" in table and "integrato" not in table


def test_mixed_support_is_unchanged_and_the_certain_fact_is_what_integrates_it():
    req = _settled(["S1", "S2"])
    assert req["status"] == "provato" and not IR.contested(req) and "contested_by" not in req
    rec = {"article": "646 c.p.", "requisites": [req]}
    rows = M.subsumption_rows(rec, {s["id"]: s for s in _sents()})
    assert rows[0][2] == "integrato" and rows[0][1].startswith("Bruno Gelso ha prelevato")
    assert "sostiene" not in rows[0][1]


def test_the_clients_own_statement_against_himself_is_not_an_allegation_for_this_purpose():
    req = _settled(["S3"])
    assert not IR.contested(req) and req["status"] == "da provare"        # the 79.8 rule, as before
    rec = {"article": "646 c.p.", "requisites": [req]}
    assert M.subsumption_rows(rec, {s["id"]: s for s in _sents()})[0][2] == "da provare"


def test_a_contested_element_is_the_defence_clients_primary_thesis():
    contested = {"id": "R1", "kind": "requisito", "text": "con artifizi o raggiri", "comma": 1, "status": "da provare",
                 "contested_by": "La società", "contested_ids": ["S1"]}
    integrated = {"id": "R2", "kind": "requisito", "text": "inducendo taluno in errore", "comma": 1,
                  "status": "provato", "supporting": ["S2"]}
    records = {"640 c.p.": {"article": "640 c.p.", "rubrica": "Truffa", "side": "threat", "charged": 1,
                            "requisites": [integrated, contested]}}
    ctx = {"records": records, "assessment": {"article": "640 c.p."}, "posture": T.IMPUTATO, "case": ""}
    cands = M.thesis_candidates(ctx)
    first = cands[0]
    assert first["contested"] and first["contesta"] == ["R1"] and first["kind"] == "contest_open"
    assert first["tesi"].startswith("L'accusa deve provare «con artifizi o raggiri» (R1) dell'art. 640 c.p.")
    assert "allegazione della società" in first["tesi"]
    ctx["candidates"] = cands
    assert M.contested_counts(ctx) == {"elements_contestato": 1, "theses_from_contestato": 1}
    counters = {}
    M.method_counters(counters, ctx)
    assert counters["elements_contestato"] == 1 and counters["theses_from_contestato"] == 1


# --- 2. the fit's dedicated fact-quote field ---------------------------------------------------------------------------

def _judge(item):
    return T._judge(T.CONTRATTO, item)


def test_a_no_with_fatto_citato_is_verified_and_without_it_keeps_the_norm():
    res, rec = _judge(T._item("1439 c.c.", "no", f"«{T.PM}» (S3), non «annullamento del contratto».",
                              fatto=T.PM))
    assert rec["verified"] and rec["fact_quotes"] == [T.PM] and res["counters"]["fit_no_verified"] == 1
    # the 79.14.2 behaviour the brief saw: the fact quoted under perché only -- no longer read
    res, rec = _judge(T._item("1439 c.c.", "no", f"«{T.PM}» (S3)."))
    assert not rec["verified"] and rec["quote_failed"] and rec["reason"] == "fatto_citato mancante"
    assert res["counters"]["fit_no_unverified_quote"] == 1 and ap.effects(res, [])["removed"] == []


def test_a_dipende_must_carry_a_verified_fatto_citato_and_a_si_may_leave_it_empty():
    item = T._item("1439 c.c.", "dipende", "La vendita (S1) dipende dai «raggiri usati da uno dei contraenti».",
                   condizione="Bruno Gelso ha usato raggiri?")
    res, rec = _judge(dict(item, fatto_citato="Bruno Gelso ha venduto a Carla Tasso un orologio"))
    assert rec["verified"] and rec["fit"] == "dipende", rec["reason"]
    res, rec = _judge(dict(item, fatto_citato="Bruno Gelso ha regalato un orologio"))
    assert not rec["verified"] and rec["quote_failed"] and res["counters"]["fit_dipende"] == 0
    assert ap.effects(res, [])["open"] == []                           # unverified: the norm keeps its path
    res, rec = _judge(T._item("640 c.p.", "si", "Bruno Gelso ha venduto un orologio (S1): «con artifizi o raggiri».",
                              fatto=""))
    assert not rec["verified"]          # the 640 text is not 1439's: the article-passage check still applies
    res = ap.judge([T.TRUFFA], T.FACTS, "imputato", case_text=T.CASE_TEXT, call=lambda m, k: json.dumps(
        {"giudizi": [T._item("640 c.p.", "si", "Bruno Gelso ha venduto un orologio (S1): «con artifizi o raggiri».",
                             fatto="")]}))
    assert res["judgments"][0]["verified"] and res["counters"]["fit_si"] == 1


def test_the_prompt_schema_carries_fatto_citato_and_its_hash_is_new():
    text = ap.prompt_system()
    assert '"fatto_citato"' in text and "il \"perche\" si giustifica citando" not in text
    assert ap.prompt_hash() == "2168ae9ef513"


# --- 3. an empty plan with a verified charge -----------------------------------------------------------------------------

def _rescue_records(with_640=True):
    out = {"2043 c.c.": {"article": "2043 c.c.", "rubrica": "Risarcimento per fatto illecito", "side": "threat",
                         "relied_by": "Carla Tasso", "requisites": [T._req(4, "fatto doloso o colposo", "da provare"),
                                                                    T._req(5, "danno ingiusto", "provato", ["S3"])]}}
    if with_640:    # read, never marked the charge (no side): the binding empties the charge set
        out["640 c.p."] = {"article": "640 c.p.", "rubrica": "Truffa", "requisites": [
            T._req(1, "con artifizi o raggiri", "da provare"), T._req(2, "inducendo taluno in errore", "da provare")]}
    return out


CHARGED = {"articolo": "640 c.p.", "descrizione": "truffa", "frase": "«Il pubblico ministero procede per truffa»"}


def test_an_emptied_charge_set_is_rescued_on_the_frames_verified_charge(monkeypatch):
    got = T._method(monkeypatch, T.PENAL_TEXT, T._penal_reply(addebito_o_domanda=CHARGED), _rescue_records(),
                    T.IMPUTATO, "Bruno Gelso")
    ctx, answer = got["ctx"], got["answer"]
    b = ctx["frame_binding"]
    assert b["excluded"] == ["2043 c.c."], b
    assert b["rescued"] and b["kept"] == ["640 c.p."] and not b["empty"], b
    assert b["frame_charge"] == "640 c.p." and b["charge_in_corpus"]
    assert ctx["assessment"]["article"] == "640 c.p."
    on_640 = [c for c in ctx["candidates"] if set(c.get("contesta") or []) & {"R1", "R2"}]
    assert on_640 and on_640[0]["kind"] == "contest_open"
    assert not any(c["tesi"].startswith("Ammettere") for c in ctx["candidates"])
    assert "Ammettere la condotta" not in answer and "corpus non contiene" not in answer
    assert ctx["corpus_line_code"] == 0
    rescue = next(p for e, p in got["events"] if e == "frame_binding_rescue")
    assert rescue == {"charge": "640 c.p.", "rescued": True, "elements": ["R1", "R2"]}
    counters = {}
    M.method_counters(counters, ctx)
    assert counters["frame_binding_rescued"] == 1


def test_no_record_of_the_charge_no_rescue_and_still_no_corpus_line_on_a_code_article(monkeypatch):
    got = T._method(monkeypatch, T.PENAL_TEXT, T._penal_reply(addebito_o_domanda=CHARGED),
                    _rescue_records(with_640=False), T.IMPUTATO, "Bruno Gelso")
    ctx, answer = got["ctx"], got["answer"]
    b = ctx["frame_binding"]
    assert b["empty"] and not b["rescued"] and b["charge_in_corpus"]
    assert "corpus non contiene" not in answer and ctx["corpus_line_code"] == 0
    assert [c["kind"] for c in ctx["candidates"]] == ["contenimento"]
    assert ctx["candidates"][0]["tesi"] == M.CONTAINMENT_LINE           # never «Ammettere la condotta»
    assert next(p for e, p in got["events"] if e == "frame_binding_rescue")["why"] == \
        "nessuna scheda dell'articolo tra le norme lette"


def test_ammettere_la_condotta_is_never_generated_without_a_verified_admission(monkeypatch):
    # (review) «otherwise it is not generated at all»: with the frame held and with no frame (CASE_FRAME off) alike
    for frame_on in (False, True):
        got = T._method(monkeypatch, T.PENAL_TEXT, T._penal_reply(), T._penal_records(), T.IMPUTATO, "Bruno Gelso",
                        frame_on=frame_on)
        assert not any(c["tesi"].startswith("Ammettere") for c in got["ctx"]["candidates"]), frame_on
        assert "mmettere la condotta" not in got["answer"] and not M.frame_admits_conduct(got["state"]), frame_on
        assert any(c["kind"] == "contenimento" and c["tesi"] == M.CONTAINMENT_LINE for c in got["ctx"]["candidates"])
    assert M.posture_fallback(T.IMPUTATO) == ""                                     # the table's line is not enough
    assert M.fallback_thesis({}, [], T.IMPUTATO)["tesi"] == M.CONTAINMENT_LINE
    verified = {**T.IMPUTATO, M.ADMISSION_VERIFIED: True}                           # only a verified admission
    assert M.fallback_thesis({}, [], verified)["tesi"].startswith("Ammettere la condotta")


# --- 4. excluded means excluded -----------------------------------------------------------------------------------------

def test_an_excluded_article_is_absent_from_sections_4_5_7_and_fatti_mancanti(monkeypatch):
    got = T._method(monkeypatch, T.PENAL_TEXT, T._penal_reply(), T._penal_records(), T.IMPUTATO, "Bruno Gelso")
    ctx, plan = got["ctx"], got["plan"]
    assert ctx["frame_binding"]["excluded"] == ["2043 c.c."] and ctx["excluded_dropped"][:1] == ["2043 c.c."]
    plan = {**plan, "obiezioni": [{"obiezione": "Carla Tasso chiederà il risarcimento ai sensi dell'art. 2043 c.c.",
                                   "risposta": "Il danno va provato."}] + list(plan.get("obiezioni") or []),
            "orientamenti": [{"punto": "Il danno ingiusto dell'art. 2043 c.c.", "pronuncia": ""},
                             {"punto": "Gli artifizi dell'art. 640 c.p.", "pronuncia": ""}]}
    norms = ("**Art. 640 c.p. — Truffa**\n\nL'art. 640 c.p. riguarda i fatti narrati.\n\n"
             "**Art. 2043 c.c. — Risarcimento per fatto illecito**\n\nL'art. 2043 c.c. riguarda i fatti narrati.")
    answer = M.compose([("Norme applicate ai fatti", norms)], ctx, plan,
                       profiles=["- art. 2043 c.c. (Risarcimento per fatto illecito): profilo civile"], log={})
    sections = dict(M.answer_sections(answer))
    norme = sections[M.title("norme")]
    before, _, profili = norme.partition("**Altri profili**")
    assert "2043" not in before and "2043 c.c." in profili                 # «Altri profili» only
    assert "Art. 640 c.p." in before
    for key in ("orientamenti", "obiezioni", "fatti"):
        assert "2043" not in sections[M.title(key)], key
    assert "640 c.p" in sections[M.title("orientamenti")]
    assert "2043 c.c." in ctx["excluded_dropped"]
    counters = {}
    M.method_counters(counters, ctx)
    assert counters["excluded_dropped"] == ["2043 c.c."]


def test_a_bold_heading_citing_only_an_excluded_article_drops_its_block(monkeypatch):
    # (review) a heading in another form than «**Art. N …**»
    got = T._method(monkeypatch, T.PENAL_TEXT, T._penal_reply(), T._penal_records(), T.IMPUTATO, "Bruno Gelso")
    ctx, plan = got["ctx"], got["plan"]
    norms = ("**Art. 640 c.p. — Truffa**\n\nL'art. 640 c.p. riguarda i fatti narrati.\n\n"
             "**Il risarcimento (art. 2043 c.c.)**\n\nIl danno ingiusto va risarcito.")
    answer = M.compose([("Norme applicate ai fatti", norms)], ctx, plan, profiles=[], log={})
    norme = dict(M.answer_sections(answer))[M.title("norme")]
    assert "2043" not in norme and "Il danno ingiusto va risarcito" not in norme and "Art. 640 c.p." in norme
    mixed = "**Truffa e risarcimento (artt. 640 c.p. e 2043 c.c.)**\n\nTesto."
    assert M.excluded_blocks_out(mixed, ["2043 c.c."]) == mixed           # a heading citing a kept article stays
    assert M.excluded_blocks_out("**Analisi per elementi**\n\nTesto.", ["2043 c.c."]) == \
        "**Analisi per elementi**\n\nTesto."


def test_a_norm_a_verified_no_removed_is_dropped_too(monkeypatch):
    ctx = {"frame_binding": {}, "fit_removed": ["1439 c.c."], "excluded_articles": []}
    assert M.excluded_labels(ctx) == ["1439 c.c."]
    text = "**Art. 1439 c.c. — Dolo**\n\nIl dolo.\n\n**Art. 640 c.p. — Truffa**\n\nTesto."
    assert M.excluded_blocks_out(text, M.excluded_labels(ctx)) == "**Art. 640 c.p. — Truffa**\n\nTesto."
    assert M.cites_excluded("Si veda l'art. 1439 cod. civ.", ["1439 c.c."]) == ["1439 c.c."]
