"""v79.19.1 -- Antonelo's review of 26 Nov, on branch v79.19 (the model faked; synthetic texts, invented names):

0. the deploy hazard: scripts/deploy.sh and scripts/promote.sh refuse a target whose data/ is a symlink (exit 1)
1. the frame's article from memory: an article read from the facts alone is a hypothesis, a fit candidate, the charge
   only on «si» / «dipende» (frame_hypothesis_confirmed / _dropped); in the analysis step a charge the step rejected,
   or missing beside a verified line, never leads (analysis_frame_charge_overridden); «linea non verificata» prints
   only when no verified line exists
2. the analysis step's timeout: the timed-out call's prompt and partial reply in the calls log; the deadline scales with
   the article block (40 s + 1 s per norm, at most 60 s); the block over max_block_chars sent with elements only
3. empty objections: the verified and the model's objections both 0 -> 79.15.3's objections path
   (objections_fallback_legacy), never the empty line
4. the correction label: parse_article cuts the code or the law's name, the correction event logs the clean label
5. the grading prompt: full key names (voto, questione, perche), one sentence on voto's type; the parser reads both
6. GATES_DEMOTE: the gates demote, never drop (gate_demoted_by); every fit-accepted record printed (records_unprinted
   0); rulings_printed counted
"""

import json
import os
import re
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_14_2_fixture as F  # noqa: E402
import test_v79_15_0_fixture_penal as P  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from test_v79_14_case_frame import _read  # noqa: E402

LITE = ("Assisto Piero Vinci, indagato: dopo una lite ha colpito Luca Ferri con un pugno e Ferri è morto in ospedale "
        "tre giorni dopo. Cosa rischia Piero Vinci?")
LOAN = ("Assisto Anna Pino: ha prestato 9.000 euro a Bruno Faggio con una scrittura privata e Faggio le contesta "
        "l'art. 1218 c.c. per il mancato pagamento di un canone. Come può Anna Pino difendersi?")


def _reply(article=None, facts=None, frase="«ha colpito Luca Ferri con un pugno e Ferri è morto in ospedale tre giorni "
                                           "dopo»"):
    charge = {"articolo": article, "descrizione": "morto", "frase": frase}
    if facts:
        charge["articolo_dai_fatti"] = facts
    return {"parti": [{"nome": "Piero Vinci", "ruolo_nel_fatto": "ha colpito", "lato": "cliente"},
                      {"nome": "Luca Ferri", "ruolo_nel_fatto": "è morto", "lato": "controparte"}],
            "cliente": {"nome": "Piero Vinci", "perche": "«Assisto Piero Vinci»"},
            "ruolo": {"valore": "imputato/indagato", "fonte": "testo", "perche": "«Assisto Piero Vinci, indagato»"},
            "rito": {"valore": "penale", "perche": "«indagato»"}, "stadio": {"valore": "indagini", "perche": "«indagato»"},
            "addebito_o_domanda": [charge], "istituto": None, "quesiti": ["Cosa rischia Piero Vinci?"]}


# --- 0. the deploy hazard ------------------------------------------------------------------------------------------------

def _tree(root: Path, data_symlink: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    (root / "VERSION").write_text("79.19.1+abc1234\n")
    if data_symlink:
        (root / "elsewhere").mkdir(exist_ok=True)
        (root / "data").symlink_to(root / "elsewhere", target_is_directory=True)
    else:
        (root / "data").mkdir(exist_ok=True)
    return root


def test_promote_refuses_a_symlinked_data_dir_before_the_snapshot(tmp_path):
    src = _tree(tmp_path / "v38", False)
    dst = _tree(tmp_path / "opt", True)
    env = {**os.environ, "SRC": str(src), "DST": str(dst), "SNAPSHOTS": str(tmp_path / "snap")}
    got = subprocess.run(["bash", str(ROOT / "scripts" / "promote.sh")], env=env, capture_output=True, text=True)
    assert got.returncode == 1 and "is a symlink" in got.stdout and str(dst / "data") in got.stdout
    assert not (tmp_path / "snap").exists()                 # refused before step 1: nothing written
    # the source side likewise
    src2 = _tree(tmp_path / "v38b", True)
    got = subprocess.run(["bash", str(ROOT / "scripts" / "promote.sh")],
                         env={**env, "SRC": str(src2), "DST": str(_tree(tmp_path / "optb", False))},
                         capture_output=True, text=True)
    assert got.returncode == 1 and "is a symlink" in got.stdout
    # a real directory passes the check (the run then stops later, on the missing venv -- another message)
    dst3 = _tree(tmp_path / "opt3", False)
    got = subprocess.run(["bash", str(ROOT / "scripts" / "promote.sh")], env={**env, "DST": str(dst3)},
                         capture_output=True, text=True)
    assert got.returncode == 1 and "is a symlink" not in got.stdout and "no " in got.stdout


def _fake_bin(tmp_path: Path) -> Path:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    ssh = bin_ / "ssh"           # the remote command (the last argument) run locally
    ssh.write_text("#!/bin/bash\nfor last; do :; done\nbash -c \"$last\"\n")
    rsync = bin_ / "rsync"
    rsync.write_text("#!/bin/bash\nexit 0\n")
    for f in (ssh, rsync):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return bin_


def test_deploy_refuses_a_symlinked_data_dir_on_the_target(tmp_path):
    bin_ = _fake_bin(tmp_path)
    target = _tree(tmp_path / "v38", True)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "SSH_HOST": "fake", "TARGET": str(target),
           "ALLOW_DIRTY": "1"}
    got = subprocess.run(["bash", str(ROOT / "scripts" / "deploy.sh")], env=env, capture_output=True, text=True,
                         cwd=str(ROOT))
    assert got.returncode == 1 and "is a symlink" in got.stdout and str(target / "data") in got.stdout
    ok = _tree(tmp_path / "v38ok", False)
    got = subprocess.run(["bash", str(ROOT / "scripts" / "deploy.sh")], env={**env, "TARGET": str(ok)},
                         capture_output=True, text=True, cwd=str(ROOT))
    assert "is a symlink" not in got.stdout and "(mirrored)" in got.stdout    # past the check, into the sync


def test_the_exclude_list_keeps_the_servers_replay_harness():
    lines = {ln.strip() for ln in (ROOT / "scripts" / "deploy_exclude.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")}
    assert "/replay_cases.py" in lines


# --- 1. the frame's article from memory --------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _frame_on(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")


def test_an_article_from_the_facts_alone_is_a_hypothesis_not_the_charge():
    res = _read(LITE, _reply(None, "584 c.p."))
    a = res["frame"]["addebito_o_domanda"]
    assert a["articolo"] is None and a["ipotesi"] == "584 c.p." and a["da_fatti"] is True
    state = {"case_frame": res}
    cf.settle(state)
    assert cf.bound(state, "addebito_o_domanda") is None and cf.charges(state) == []
    assert cf.alternative_charges(state) == ["584 c.p."]            # a fit candidate
    assert cf.hypotheses(state) == [{"label": "584 c.p.", "stato": "aperta"}]
    # the judgment «si»: the charge, a correction "" -> the article, counted
    made = cf.apply_fit(state, [{"article": "584 c.p.", "fit": "si", "verified": True}])
    assert made and made[0]["old"] == "" and made[0]["new"] == "584 c.p." and made[0]["step"] == "fit"
    assert cf.bound(state, "addebito_o_domanda") == "584 c.p." and cf.alternative_charges(state) == []
    assert cf.hypotheses(state) == [{"label": "584 c.p.", "stato": "confermata"}]
    c = state["frame"]["counters"]
    assert c["frame_hypothesis_confirmed"] == 1 and c["frame_hypothesis_dropped"] == 0
    assert c["frame_corrections_applied"] == 1
    assert flow.frame_audit(state)["frame_hypothesis_confirmed"] == 1


@pytest.mark.parametrize("judgments", [[{"article": "584 c.p.", "fit": "no", "verified": True}],
                                       [{"article": "584 c.p.", "fit": "si", "verified": False}], []])
def test_a_hypothesis_the_judgment_does_not_confirm_is_dropped(judgments):
    state = {"case_frame": _read(LITE, _reply(None, "584 c.p."))}
    cf.settle(state)
    assert cf.apply_fit(state, judgments) == []
    assert cf.bound(state, "addebito_o_domanda") is None and cf.charges(state) == []
    assert cf.alternative_charges(state) == []                       # no longer a candidate
    assert cf.hypotheses(state) == [{"label": "584 c.p.", "stato": "scartata"}]
    assert state["frame"]["counters"]["frame_hypothesis_dropped"] == 1
    assert state["frame"]["counters"]["frame_hypothesis_confirmed"] == 0


def test_a_penal_posture_refuses_a_non_penal_hypothesis():
    # the read knows no rito (a penal rito refuses a civil article at reading time already: charge_rito_dropped)
    reply = _reply(None, "2043 c.c.")
    reply["rito"], reply["stadio"] = None, None
    state = {"case_frame": _read(LITE, reply)}
    cf.settle(state)
    assert cf.alternative_charges(state) == ["2043 c.c."]
    made = cf.settle_posture(state, {"rito": "penale", "family": "penale"})
    assert made and made[0]["field"] == "addebito_o_domanda.ipotesi" and made[0]["new"] == ""
    assert cf.alternative_charges(state) == [] and cf.hypotheses(state) == [{"label": "2043 c.c.", "stato": "scartata"}]


def _norms():
    return [{"article": "646 c.p.", "text": F.ART_646, "rubrica": "Appropriazione indebita",
             "elements": [{"id": "E1", "testo": "si appropria il denaro o la cosa mobile altrui"},
                          {"id": "E2", "testo": "per procurare a sé o ad altri un ingiusto profitto"}]},
            {"article": "640 c.p.", "text": F.ART_640, "rubrica": "Truffa",
             "elements": [{"id": "E1", "testo": "con artifizi o raggiri"}, {"id": "E2", "testo": "induce taluno in errore"}]}]


def _frame(article="646 c.p.", da_fatti=False):
    charge = {"articolo": article, "numero": "646", "codice": "cp", "descrizione": "appropriazione indebita",
              "frase": "«procede per appropriazione indebita»"}
    if da_fatti:
        charge["da_fatti"] = True
    return {"cliente": {"nome": "Dario Melis"}, "ruolo": {"valore": "imputato/indagato"}, "rito": {"valore": "penale"},
            "addebito_o_domanda": charge,
            "parti": [{"nome": "Dario Melis", "lato": "cliente"}, {"nome": "Ferramenta Lunardi", "lato": "controparte"}]}


def _inp(frame=None):
    sents = [{"id": "S1", "text": "Dario Melis custodiva la cassa del negozio.", "certainty": "accertato", "source": "caso"},
             {"id": "S2", "text": "Dario Melis nega di aver trattenuto il denaro.", "certainty": "allegato", "source": "caso"}]
    return A.make_input(frame if frame is not None else _frame(), _norms(), sents, ["Quali rischi?"], "Dario Melis",
                        ["indagato"])


def test_a_confirmed_hypothesis_is_a_fitted_norm_never_the_forced_charge():
    inp = _inp(_frame(da_fatti=True))
    assert inp["charge"] == "" and inp["charge_hypothesis"] == "646 c.p."
    assert not any(n["remedy"] for n in inp["norms"])
    checked = A.verify({"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E1", "tesi": "t",
                                             "cita_fatti": ["S1"], "cita_norma": []}}, inp)
    assert checked["charge_missing"] is False and checked["reasons"] == []
    inp_named = _inp()
    assert inp_named["charge"] == "646 c.p." and inp_named["charge_hypothesis"] == ""


def test_a_rejected_frame_charge_never_leads_the_fitted_fallback_skips_it():
    inp = _inp()
    reply = {"linea_principale": {"articolo": "646 c.p.", "elemento_in_gioco": "A646cp:E9", "tesi": "t",
                                  "cita_fatti": ["S1"], "cita_norma": []}}
    checked = A.verify(reply, inp)
    assert checked["rejected"] and checked["rejected"][0]["articolo"] == "646 c.p." and checked["charge_missing"]
    out = A.finish(checked, inp)
    assert out["linea_principale"]["articolo"] == "640 c.p." and out["linea_principale"]["stato"] == A.FIT_LINE
    assert out["frame_charge_overridden"] is True and out["frame_charge_missing"] is False
    c = checked["counters"]
    assert c["analysis_frame_charge_overridden"] == 1 and c["analysis_frame_charge_missing"] == 1
    assert c[A.SKIPPED_REASON] != A.SKIP_FRAME_CHARGE
    # no verified line: the label «linea non verificata» prints
    assert A.LINE_UNVERIFIED_LABEL in A.line_text(out["linea_principale"])


def test_a_verified_line_leads_the_missing_frame_charge_is_the_subordinate_without_the_unverified_label():
    inp = _inp()
    reply = {"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E1", "tesi": "la truffa",
                                  "cita_fatti": ["S1"], "cita_norma": []},
             "elementi": [{"articolo": "640 c.p.", "elemento": "A640cp:E1", "esito": "da_provare", "cita_fatti": []}]}
    checked = A.verify(reply, inp)
    assert checked["charge_missing"] and checked["rejected"] == []
    out = A.finish(checked, inp)
    assert out["linea_principale"]["articolo"] == "640 c.p." and out["linea_principale"]["stato"] == A.VERIFIED
    sub = out["linea_subordinata"]
    assert sub["articolo"] == "646 c.p." and sub["stato"] == A.FRAME_LINE and sub["posizione"] == "subordinata"
    assert all(e["esito"] == A.DA_PROVARE for e in out["elementi"]["646 c.p."])
    assert checked["counters"]["analysis_frame_charge_overridden"] == 1
    assert checked["counters"][A.SKIPPED_REASON] == A.SKIP_LINE_VERIFIED
    view = A.view({"status": A.OK, "analysis": out}, inp)
    text = A.summary_text(view)
    assert "Linea principale: art. 640 c.p." in text and f"art. 646 c.p. ({A.LABEL_FRAME_SUB})" in text
    assert A.LINE_UNVERIFIED_LABEL not in text


def test_the_text_named_charge_still_leads_when_no_line_is_verified():
    inp = _inp()
    checked = A.verify({"linea_principale": None}, inp)
    out = A.finish(checked, inp)
    assert out["linea_principale"]["articolo"] == "646 c.p." and out["linea_principale"]["stato"] == A.FRAME_LINE
    assert checked["counters"][A.SKIPPED_REASON] == A.SKIP_FRAME_CHARGE
    assert checked["counters"]["analysis_frame_charge_overridden"] == 0
    assert A.LINE_UNVERIFIED_LABEL in A.line_text(out["linea_principale"])


# --- 2. the analysis step's timeout -------------------------------------------------------------------------------------

def test_the_deadline_scales_with_the_article_block(monkeypatch):
    base = dict(A.config())
    monkeypatch.setattr(A, "config", lambda: base)
    assert base["deadline_s"] == 40 and base["deadline_per_article_s"] == 1 and base["deadline_max_s"] == 60
    assert A.deadline_for({"norms": []}) == 40.0
    assert A.deadline_for({"norms": [{}] * 7}) == 47.0
    assert A.deadline_for({"norms": [{}] * 30}) == 60.0


def test_a_timed_out_call_leaves_its_prompt_and_partial_reply_in_the_calls_log(monkeypatch):
    base = dict(A.config())
    base["deadline_s"], base["deadline_per_article_s"] = 0.6, 0
    monkeypatch.setattr(A, "config", lambda: base)
    events = []
    monkeypatch.setattr(A, "_log", lambda e, d: events.append((e, d)))
    inp = _inp()

    def slow(messages, kwargs):
        buf = A._PARTIAL.get()
        buf.append('{"linea_principale": {"articolo": "646 c.p."')
        time.sleep(1.4)
        return '{"linea_principale": null}'
    res = A.analyse(inp, call=slow)
    assert res["status"] == A.TIMEOUT and res["counters"]["analysis_timeouts"] == 1
    call = res["calls"][0]
    assert call["status"] == A.TIMEOUT and call["partial_reply"] == '{"linea_principale": {"articolo": "646 c.p."'
    assert [m["role"] for m in call["prompt"]] == ["system", "user"]
    assert "ARTICOLO: 646 c.p." in call["prompt"][1]["content"] and call["prompt"][0]["content"] == A.prompt_system()
    timeout = next(d for e, d in events if e == "analysis_step_timeout")
    assert timeout["partial_reply"] == call["partial_reply"] and timeout["prompt"] == call["prompt"]
    assert timeout["block"]["chars_full"] > 0 and res["deadline_s"] == 0.6
    time.sleep(1.2)                                 # the late reply lands: logged, nothing reads it
    late = next(d for e, d in events if e == "analysis_step_late_reply")
    assert late["late_reply"] == '{"linea_principale": null}' and late["attempt"] == 1
    from src.rag import verbose_logger
    assert {"prompt", "partial_reply", "late_reply"} <= verbose_logger.FULL_KEYS


def test_the_default_call_streams_into_the_partial_buffer(monkeypatch):
    from src.rag import ai_chat

    class Chunk:
        def __init__(self, text, final=False):
            self.content = text
            self.response_metadata = {"finish_reason": "stop"} if final else {}

        def __add__(self, other):
            out = Chunk(self.content + other.content)
            out.response_metadata = other.response_metadata or self.response_metadata
            return out

    class Model:
        def bind(self, **kwargs):
            return self

        def stream(self, messages):
            for i, piece in enumerate(['{"a": ', '1}']):
                yield Chunk(piece, final=i == 1)

        def invoke(self, messages):
            raise AssertionError("invoke: the buffer set, the call streams")
    buf = []
    got = A._invoke(Model(), [], buf)
    assert got.content == '{"a": 1}' and got.response_metadata == {"finish_reason": "stop"}
    assert buf == ['{"a": ', '1}']
    assert A._invoke(types.SimpleNamespace(invoke=lambda m: "plain"), [], None) == "plain"   # no buffer: one invoke
    assert A._invoke(types.SimpleNamespace(invoke=lambda m: "plain"), [], []) == "plain"     # no stream: one invoke
    assert ai_chat is not None


def test_the_article_block_over_the_cap_is_sent_with_elements_only(monkeypatch):
    base = dict(A.config())
    monkeypatch.setattr(A, "config", lambda: base)
    inp = _inp()
    full = A.block_sizes(inp)
    assert full["reduced"] is False and full["chars_sent"] == full["chars_full"] and full["cap"] == 24000
    human = A.build_messages(inp)[1].content
    assert "[A646cp.1]" in human and base["labels"]["periods_omitted"] not in human
    base["max_block_chars"] = full["chars_full"] - 1
    sizes = A.block_sizes(inp)
    assert sizes["reduced"] is True and sizes["chars_sent"] < sizes["chars_full"] and sizes["norms"] == 2
    human = A.build_messages(inp)[1].content
    assert "[A646cp.1]" not in human and base["labels"]["periods_omitted"] in human
    assert "- [A646cp:E1]" in human and "- [A640cp:E2]" in human            # the elements kept
    res = A.analyse(inp, call=lambda m, k: '{"linea_principale": null}')
    assert res["counters"]["analysis_block_reduced"] == 1 and res["block"]["reduced"] is True
    assert "analysis_block_reduced" in A.COUNTERS


# --- 3. empty objections (the fixture, below) and 4. the correction label ---------------------------------------------------

@pytest.mark.parametrize("raw,label", [
    ("art. 1218 c.c. inadempimento del contratto", "1218 c.c."), ("73, comma 5, d.P.R. 309/1990 detenzione", "73 d.P.R. 309/1990"),
    ("art. 18 l. 300/1970 reintegrazione", "18 l. 300/1970"), ("art. 2 d.lgs. 74/2000 dichiarazione fraudolenta", "2 d.lgs. 74/2000"),
    ("624 c.p. furto", "624 c.p."), ("2051 c.c. danno da cose in custodia", "2051 c.c."),
    ("art. 2043 del codice civile per il danno", "2043 codice civile"), ("1439 c.c. ex d.lgs. 74/2000", "1439 c.c."),
    ("18 legge n. 300 del 1970 reintegra", "18 legge n. 300 del 1970"), ("640-bis c.p.", "640-bis c.p."),
    ("73 T.U. stupefacenti", "73 T.U. stupefacenti")])
def test_parse_article_cuts_the_rest_at_the_code_or_the_law(raw, label):
    got = cf.parse_article(raw)
    assert f"{got['number']} {got['rest']}".strip() == label
    assert cf.code_or_law("c.c. inadempimento") == "c.c." and cf.code_or_law("") == ""


def test_the_correction_event_logs_the_clean_label():
    frase = "«Faggio le contesta l'art. 1218 c.c. per il mancato pagamento di un canone»"
    reply = {"parti": [{"nome": "Anna Pino", "ruolo_nel_fatto": "ha prestato", "lato": "cliente"},
                       {"nome": "Bruno Faggio", "ruolo_nel_fatto": "contesta", "lato": "controparte"}],
             "cliente": {"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"},
             "ruolo": {"valore": "convenuto", "fonte": "testo", "perche": "«Come può Anna Pino difendersi»"},
             "rito": {"valore": "civile", "perche": "«scrittura privata»"}, "stadio": None,
             "addebito_o_domanda": [{"articolo": "1453 c.c. risoluzione per inadempimento", "descrizione": "mancato pagamento",
                                     "frase": frase}], "istituto": None, "quesiti": []}
    res = _read(LOAN, reply)
    a = res["frame"]["addebito_o_domanda"]
    assert a["articolo"] == "1218 c.c." and a["corretto_da"] == "1453 c.c."
    assert res["corrections"][0]["old"] == "1453 c.c." and res["corrections"][0]["new"] == "1218 c.c."


# --- 5. the grading prompt -------------------------------------------------------------------------------------------------

def test_the_grading_prompt_names_its_keys_in_full_and_the_parser_reads_both():
    for text in (dg._GRADED_SYSTEM, dg._graded_system(evidence=True), dg._graded_system(issues=True),
                 dg._graded_system(evidence=True, issues=True, examples=True)):
        assert '"voto": 0, "questione": ' in text and '"perche": ' in text and '"g": 0' not in text
        assert text.count('"voto" è un numero intero da 0 a 3.') == 1
    ids = {"A1": "640 c.p.", "A2": "1453 c.c."}
    new = dg._graded_reply('[{"a": "A1", "voto": 3, "questione": "la truffa", "perche": "p"}, '
                           '{"a": "A2", "voto": "1", "questione": "", "perche": "q"}]', ids)
    assert {k: (v["grade"], v["issue"], v["reason"]) for k, v in new.items()} == {
        "640 c.p.": (3, "la truffa", "p"), "1453 c.c.": (1, "", "q")}
    old = dg._graded_reply('[{"a": "A1", "g": 2, "q": "x", "r": "y"}]', ids)
    assert old == {"640 c.p.": {"article": "640 c.p.", "grade": 2, "issue": "x", "reason": "y"}}
    assert dg._graded_reply('[{"a": "A1", "voto": 7}]', ids) == {}       # no grade 0-3: not read


# --- 6. GATES_DEMOTE ---------------------------------------------------------------------------------------------------------

def _row(n, code="c.p.", fused=None, doc="Codice Penale"):
    out = {"_article": [n, code], "d": {"name": doc}}
    if fused is not None:
        out["_fused_rank"] = fused
    return out


def test_gates_demote_instead_of_dropping_only_with_the_flag(monkeypatch):
    before = [_row("640", fused=1), _row("2043", "c.c.", fused=2), _row("572", fused=3)]
    kept = [before[0], before[2]]
    out = {}
    assert flow.demote_instead("matter_gate", before, kept, out) == kept and out == {}
    monkeypatch.setenv("GATES_DEMOTE", "true")
    rows = flow.demote_instead("matter_gate", before, kept, out)
    assert [r["_article"][0] for r in rows] == ["640", "572", "2043"]
    assert before[1]["_demoted"] == ["matter_gate"] and "_demoted" not in before[0]
    assert out["gate_demoted_by"] == {"matter_gate": 1} and out["gate_demoted"] == {"matter_gate": ["2043 c.c."]}
    # ranked after the undemoted, whatever the fused rank
    order = sorted((flow._rank_of(r, i), r["_article"][0]) for i, r in enumerate(before))
    assert [x for _, x in order] == ["640", "572", "2043"]
    rows = flow.demote_instead("code_scope_gate", rows, [rows[0]], out)
    assert before[1]["_demoted"] == ["matter_gate", "code_scope_gate"] and before[2]["_demoted"] == ["code_scope_gate"]
    assert out["gate_demoted_by"] == {"matter_gate": 1, "code_scope_gate": 2}
    order = sorted((flow._rank_of(r, i), r["_article"][0]) for i, r in enumerate(before))
    assert [x for _, x in order] == ["640", "2043", "572"]          # among the demoted, the fused rank
    assert flow.GATES_DEMOTE_FLAG == "GATES_DEMOTE" and not flow.gates_demote.__doc__


def test_the_family_alarm_readmits_its_graded_articles_demoted(monkeypatch):
    fam = {"alarm": [{"doc": "Codice della strada", "n": 2, "articles": ["186 d.lgs. 285/1992", "187 d.lgs. 285/1992"],
                      "proxy": False}],
           "dropped_graded": {"Codice della strada": {"graded_relevant": 2, "articles": ["186 d.lgs. 285/1992", "187 d.lgs. 285/1992"],
                                                      "grades": {"186 d.lgs. 285/1992": 3, "187 d.lgs. 285/1992": 2}}}}
    dropped = [_row("186", "d.lgs. 285/1992", doc="Codice della strada"), _row("187", "d.lgs. 285/1992", doc="Codice della strada"),
               _row("189", "d.lgs. 285/1992", doc="Codice della strada")]
    assert flow.readmit_alarm_rows(fam, dropped, [], {}) == []           # the flag off
    monkeypatch.setenv("GATES_DEMOTE", "true")
    out = {}
    back = flow.readmit_alarm_rows(fam, dropped, [_row("186", "d.lgs. 285/1992")], out)
    assert [r["_article"][0] for r in back] == ["187"] and back[0]["_grade"] == 2 and back[0]["_demoted"] == ["family_alarm"]
    assert out["gate_demoted_by"] == {"family_alarm": 1}
    proxy = {**fam, "alarm": [{**fam["alarm"][0], "proxy": True}]}
    assert flow.readmit_alarm_rows(proxy, dropped, [], {}) == []         # no grade invented


def test_the_demotion_counters_reach_the_case_counters_and_the_decompose_why():
    state = {"timings": {}, "gate_demoted_by": {"matter_gate": 2, "posture": 1}}
    out = flow.case_counters(state)
    assert out["gate_demoted_by"] == {"matter_gate": 2, "posture": 1} and out["gate_demoted"] == 3
    assert flow.DECOMPOSE_WHY["posture_demoted"]
    found = {"gate_demoted_by": {"grade_funnel": 4}}
    flow.count_demoted(found, "posture", 1, ["590 c.p."])
    assert found["gate_demoted_by"] == {"grade_funnel": 4, "posture": 1} and found["gate_demoted"] == {"posture": ["590 c.p."]}


def test_every_fit_accepted_record_is_named_or_printed_late():
    assert M._names_article("Si applica l'art. 1218 c.c. al caso.", "1218 c.c.")
    assert M._names_article("**Art. 1218 c.c.** — Responsabilità", "1218 c.c.")
    assert M._names_article("gli artt. 1218 e 1223 c.c. regolano", "1223 c.c.")
    assert M._names_article("artt. 1218, 1223 c.c.", "1218 c.c.")
    assert not M._names_article("l'art. 1218-bis c.c.", "1218 c.c.") and not M._names_article("art. 12180 c.c.", "1218 c.c.")
    assert not M._names_article("il 1218 come numero", "1218 c.c.")
    records = {"1218 c.c.": {"rubrica": "Responsabilità del debitore"}, "1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento."},
               "2043 c.c.": {"rubrica": "x", "unprinted": True}}
    ctx = {"fit_accepted": ["1218 c.c.", "1453 c.c.", "2043 c.c.", "1223 c.c."]}
    late = M.unprinted_records(ctx, records, "**Art. 1218 c.c.** — Responsabilità del debitore", ())
    assert late == ["art. 1453 c.c. (Risolubilità del contratto per inadempimento)"]
    assert M.unprinted_records(ctx, records, "x", ["1453 c.c."]) == ["art. 1218 c.c. (Responsabilità del debitore)"]
    text = "**Art. 1218 c.c.**\n\n**" + M.UNPRINTED_HEAD + "**: " + "; ".join(late) + "."
    assert M.unprinted_records(ctx, records, text, ()) == []
    state = {"applicability": {"judgments": [{"article": "1218 c.c.", "fit": "si", "verified": True},
                                             {"article": "1453 c.c.", "fit": "dipende", "verified": True},
                                             {"article": "2043 c.c.", "fit": "no", "verified": True},
                                             {"article": "1223 c.c.", "fit": "si", "verified": False}]}}
    assert M.fit_accepted(state) == ["1218 c.c.", "1453 c.c."] and M.fit_accepted({}) == []


def test_the_rulings_printed_are_counted():
    rec = {"kept": [{"article": "646 c.p.", "id": "r1"}, {"article": "646 c.p.", "id": "r2"}, {"article": "640 c.p.", "id": "r3"}]}
    assert M.rulings_printed({"rulings_record": rec}, [], ()) == 3
    assert M.rulings_printed({"rulings_record": rec}, [], ["640 c.p."]) == 2
    rulings = [{"id": "a", "massima": "m", "article": "646 c.p."}, {"id": "b", "massima": "", "article": "646 c.p."}, "x"]
    assert M.rulings_printed({}, rulings, ()) == 1
    assert {"objections_fallback_legacy", "records_unprinted", "rulings_printed"} <= set(A.RENDER_COUNTERS)


# --- the fixture: items 1, 3, 6b, 6c end to end -----------------------------------------------------------------------------

def _case(**over):
    keys = ("TEXT", "ARTICLES", "FRAME_REPLY", "fit_reply", "link", "analysis_reply", "TITLE", "ANCHOR", "RUBRICA",
            "DOCUMENTS", "CLIENT", "writer_line", "CASE_ID")
    return types.SimpleNamespace(**{**{k: getattr(P, k) for k in keys}, **over})


def _all_open_reply(parsed, attempt):
    """The analysis faked: the line on 646's dolo element, every element «da_provare», nothing cited by a party --
    no allegation, so no verified objection."""
    norms = parsed["norms"]
    el = norms["646 c.p."]["elements"]
    dolo = P._pick(el, "profitto")
    denies = P._sid(parsed["facts"], "Dario Melis nega")
    elementi = [{"articolo": "646 c.p.", "elemento": H.eid(parsed, "646 c.p.", e), "esito": "da_provare", "cita_fatti": [],
                 "parte_che_afferma": None} for e in el]
    return {"linea_principale": {"articolo": "646 c.p.", "elemento_in_gioco": H.eid(parsed, "646 c.p.", dolo),
                                 "tesi": "Il fine di profitto non è provato.", "cita_fatti": [denies],
                                 "cita_norma": [norms["646 c.p."]["periods"][0]]},
            "linea_subordinata": None, "elementi": elementi, "affermazioni_di_diritto": []}


def test_the_fixture_falls_back_to_the_legacy_objections_when_both_sources_are_empty(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, _case(analysis_reply=_all_open_reply, objections=[], CASE_ID="v79191_obj"))
    state, answer, c = run["state"], run["answer"], run["state"]["block_counters"]
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "lawyer_method_compose")] == []
    step = run["by"]["analysis_step"][0][0]
    assert step["status"] == "ok" and step["objections"] == []
    assert c["objections_fallback_legacy"] == 1
    obiezioni = H.section(answer, "obiezioni")
    assert "**Obiezione 1.**" in obiezioni and "non indica obiezioni" not in obiezioni
    assert c["records_unprinted"] == 0 and "rulings_printed" in c
    assert run["by"].get("records_unprinted") is None


def test_the_fixture_with_the_verified_objections_keeps_them_and_prints_every_fitted_record(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, _case(CASE_ID="v79191_pen"))
    c, answer = run["state"]["block_counters"], run["answer"]
    assert c["objections_fallback_legacy"] == 0 and c["records_unprinted"] == 0 and c["rulings_printed"] == 0
    obiezioni = H.section(answer, "obiezioni")          # the verified objections: the allegations as written
    assert "Secondo la Procura" in obiezioni and "La società sostiene" in obiezioni
    assert c["objections_from_allegations"] == 2
    norme = H.section(answer, "norme")
    assert "**Art. 646 c.p.**" in norme and "art. 62-bis c.p." in norme and M.UNPRINTED_HEAD not in norme
    assert c["analysis_frame_charge_overridden"] == 0 and flow.case_counters(run["state"])["frame_hypothesis_confirmed"] == 0


def test_the_fixture_confirms_a_facts_hypothesis_through_the_fit(monkeypatch, tmp_path):
    frame = json.loads(json.dumps(P.FRAME_REPLY))
    frame["addebito_o_domanda"] = {"articolo": None, "articolo_dai_fatti": "646 c.p.", "descrizione": "appropriazione indebita",
                                   "frase": "«Dario Melis lavorava come cassiere presso la società Ferramenta Lunardi e "
                                            "custodiva la cassa del negozio»"}
    run = H.run(monkeypatch, tmp_path, _case(FRAME_REPLY=frame, CASE_ID="v79191_hyp"))
    state, c = run["state"], run["state"]["block_counters"]
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "frame_apply_fit")] == []
    audit = flow.frame_audit(state)                     # the frame's counters reach the answer's through the audit
    assert audit["frame_hypothesis_confirmed"] == 1 and audit["frame_hypothesis_dropped"] == 0
    assert flow.case_counters(state)["frame_hypothesis_confirmed"] == 1
    assert cf.bound(state, "addebito_o_domanda") == "646 c.p." and cf.hypotheses(state)[0]["stato"] == "confermata"
    step = run["by"]["analysis_step"][0][0]
    assert step["status"] == "ok" and c["analysis_frame_charge_missing"] == 0 and c["analysis_frame_charge_overridden"] == 0
    assert "Linea principale: art. 646 c.p." in H.section(run["answer"], "sintesi")
    assert any(x["field"] == "addebito_o_domanda.articolo" and x["old"] == "" and x["new"] == "646 c.p."
               for x in cf.corrections(state))
