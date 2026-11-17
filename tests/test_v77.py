"""v77: delivery and measurement, no pipeline rule.

1. VERSION (src/version.py): read once, «unknown» without the file; the footer «SunnitAI <version>» ends every case-first
   answer exactly once (single_shot, the parere written again, the API's reply), never the document's draft;
   GET /api/version
2. the runtime state in one folder (src/state_paths.py): STATE_DIR, else /opt/chatbot/state on the server
3. the held-out set (src/chatbot/heldout.py): every case-first answer the API serves, one JSON file; the lawyer's pack
   (scripts/heldout_export.py) and his scores read back per version (scripts/heldout_score.py)
4. the nightly records the version, checks the footer, ends with the lawyer's means per version
5. the 31-case sim is no measure: no default batch, no recall or MENTIONED in its report (test_v60m), the intake check's
   grid by default (test_v61)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src import state_paths, version
from src.chatbot import heldout
from src.rag import interactive_flow as flow

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def released(monkeypatch, tmp_path):
    path = tmp_path / "VERSION"
    path.write_text("77.0.0+abc1234\n", encoding="utf-8")
    monkeypatch.setattr(version, "VERSION_FILE", path)
    version.current.cache_clear()
    yield "77.0.0+abc1234"
    version.current.cache_clear()


@pytest.fixture
def api(monkeypatch):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "neo4j"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    import src.chatbot.api as module
    return module


# --- 1. the version and its footer ---------------------------------------------------------------------------------------

def test_the_version_is_the_files_first_line_and_unknown_without_it(monkeypatch, tmp_path, released):
    assert version.current() == released and version.footer() == "SunnitAI 77.0.0+abc1234"
    monkeypatch.setattr(version, "VERSION_FILE", tmp_path / "none")
    version.current.cache_clear()
    assert version.current() == "unknown"


def test_the_footer_ends_the_answer_once_and_replaces_an_older_one(released):
    once = version.with_footer("## Sintesi\n\nIl parere.\n")
    assert once == "## Sintesi\n\nIl parere.\n\nSunnitAI 77.0.0+abc1234"
    assert version.with_footer(once) == once
    assert version.with_footer("Il parere.\n\nSunnitAI 76.0.0+old") == "Il parere.\n\nSunnitAI 77.0.0+abc1234"
    assert version.strip_footer(once) == "## Sintesi\n\nIl parere."
    assert version.has_footer(once, released) and not version.has_footer("Il parere.", released)
    assert version.strip_footer("SunnitAI è un nome nel testo.\nAltro.") == "SunnitAI è un nome nel testo.\nAltro."


def test_the_api_serves_the_version(api, released):
    from fastapi.testclient import TestClient
    assert TestClient(api.app).get("/api/version").json() == {"version": released}


def test_the_parere_written_again_carries_the_footer_and_the_help_text_does_not(monkeypatch, released):
    """_single_update's out(): «regenerate» with the footer; «flow_help» (not a parere) without."""
    state = {"mode": "single_shot", "state": flow.ANSWERED, "timings": {}}
    monkeypatch.setattr(flow, "interaction", lambda *a, **k: {})
    monkeypatch.setattr(flow, "touch", lambda *a, **k: None)
    reply = flow._single_update("come funziona?", state, {"intents": ["domanda_flusso"]}, 0.0)
    assert reply["kind"] == "flow_help" and not version.has_footer(reply["answer"])


# --- 2. the runtime state --------------------------------------------------------------------------------------------------

def test_the_state_folder_is_the_environments_else_the_servers_else_the_checkouts(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "s"))
    assert state_paths.sessions_file() == str(tmp_path / "s" / "sessions.json")
    assert state_paths.system_templates_catalog() == str(tmp_path / "s" / "system_templates" / "catalog_enriched.json")
    assert state_paths.article_cards_file().endswith("s/article_cards.jsonl")
    assert state_paths.logs_dir() == str(tmp_path / "s" / "logs")
    monkeypatch.delenv("STATE_DIR")
    monkeypatch.setattr(state_paths, "SERVER_DIR", tmp_path / "opt" / "state")
    assert state_paths.state_dir() == str(state_paths.ROOT / "state")         # no /opt/chatbot: the checkout's
    (tmp_path / "opt").mkdir()
    assert state_paths.state_dir() == str(tmp_path / "opt" / "state")         # the server's


def test_the_thesis_log_goes_to_the_state_folder(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
        monkeypatch.setenv(name, value)
    from src.rag import lawyer_method
    monkeypatch.delenv("THESIS_LOG_DIR", raising=False)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    path = lawyer_method.thesis_log({"case_id": "c1"}, [], "{}", {}, {}, 1.0)
    assert path and Path(path).parent == tmp_path / "logs"


# --- 3. the held-out set ---------------------------------------------------------------------------------------------------

RESULT = {"answer": "Il parere.\n\nSunnitAI 77.0.0+abc1234", "kind": "answer", "article_citations": [{"id": "2043 c.c."}],
          "state": {"single_shot_timings": {"total_s": 80.1, "round1_s": 30.0, "answer_s": 50.1},
                    "timings": {"thesis_selection_s": 12.5, "writer_s": 20.0}}}


def test_an_answer_is_saved_as_one_file_with_what_the_lawyer_and_the_report_need(monkeypatch, tmp_path, released):
    monkeypatch.setenv("HELDOUT_DIR", str(tmp_path))
    path = heldout.save("s1", "Il caso di Tizio.", RESULT, "single_shot", {"rito": "civile"}, "il testo caricato")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == f"{rec['id']}.json" and rec["id"][:10] == rec["saved_at"][:10]
    assert rec["version"] == released and rec["prompt"] == "Il caso di Tizio." and rec["case_text"] == "il testo caricato"
    assert rec["intake"] == {"rito": "civile"} and rec["article_citations"] == [{"id": "2043 c.c."}]
    assert rec["timings"] == {"total_s": 80.1, "round1_s": 30.0, "answer_s": 50.1, "thesis_selection_s": 12.5,
                              "writer_s": 20.0}
    assert rec["source"] == "test" and rec["kind"] == "answer" and rec["answer"] == RESULT["answer"]


def test_no_folder_no_save_and_a_failed_save_never_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("HELDOUT_DIR", raising=False)
    monkeypatch.setattr(heldout, "SERVER_DIR", tmp_path / "nowhere" / "heldout")
    assert heldout.folder() is None and heldout.save("s", "p", RESULT, "single_shot") is None
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("HELDOUT_DIR", str(blocker / "sub"))
    assert heldout.save("s", "p", RESULT, "single_shot") is None


def test_the_api_saves_the_case_first_answer_with_its_footer(api, monkeypatch, tmp_path, released):
    from fastapi.testclient import TestClient
    from src.db.base import get_db
    monkeypatch.setenv("HELDOUT_DIR", str(tmp_path / "heldout"))
    monkeypatch.setattr(api.chatbot, "_sessions", {})
    monkeypatch.setattr(api.chatbot, "_sessions_file", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(api, "_classify_legal_intent", lambda message, lang: "defensive_draft")
    monkeypatch.setattr(api, "_is_topic_question", lambda message: False)
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    state = {"mode": "single_shot", "state": flow.ANSWERED, "timings": {"writer_s": 9.0}}
    monkeypatch.setattr(flow, "store", lambda anchors, st: anchors)
    monkeypatch.setattr(flow, "single_shot", lambda text, lang="it", request="", **k: {
        "answer": "IL PARERE", "interaction": {}, "state": state, "citations": [], "kind": "answer"})
    api.app.dependency_overrides[get_db] = lambda: None
    try:
        case = "Leonardo è proprietario di un terreno; Domitilla nel 2016 ha chiuso il cancello sulla strada. Come agire?"
        body = TestClient(api.app).post("/api/chat", json={"message": case}).json()
    finally:
        api.app.dependency_overrides.pop(get_db, None)
    assert body["answer"] == "IL PARERE\n\nSunnitAI 77.0.0+abc1234"
    saved = heldout.load(tmp_path / "heldout")
    assert len(saved) == 1 and saved[0]["answer"] == body["answer"] and saved[0]["prompt"] == case
    assert saved[0]["mode"] == "single_shot" and saved[0]["version"] == released and saved[0]["intake"]
    assert saved[0]["timings"] == {"writer_s": 9.0}


def test_a_round_one_reply_is_neither_footed_nor_saved(api):
    assert not api._case_first_answer({"kind": "round1", "answer": "Domande"})
    assert not api._case_first_answer({"kind": "regenerate", "answer": ""})
    assert api._case_first_answer({"kind": "regenerate", "answer": "Il parere"})


def _records(folder: Path, n: int):
    for i in range(n):
        rec = {"id": f"2026-11-1{i}_0000000{i}", "version": "77.0.0+abc" if i < 2 else "77.0.1+def", "source": "live",
               "kind": "answer" if i != 1 else "regenerate", "prompt": f"caso {i}", "answer": f"## Sintesi\n\n1. punto {i}",
               "article_citations": [], "timings": {}}
        (folder / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_the_pack_holds_the_newest_unscored_first_answers_with_a_blank_block(tmp_path):
    export = _script("heldout_export")
    _records(tmp_path, 4)
    out = tmp_path / "pack.md"
    assert export.main(["--dir", str(tmp_path), "-n", "2", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.index("2026-11-13_00000003") < text.index("2026-11-12_00000002")
    assert "2026-11-11_00000001" not in text and "2026-11-10_00000000" not in text   # a regenerate; beyond n
    assert text.count("### Valutazione dell'avvocato") == 2 and text.count("   - voto: ") == 14
    assert "1. Tesi centrale" in text and "7. Strategia" in text
    # the next pack skips the cases already out (exports.jsonl) until they are scored
    assert export.main(["--dir", str(tmp_path), "-n", "5", "--out", str(tmp_path / "p2.md")]) == 0
    assert "2026-11-10_00000000" in (tmp_path / "p2.md").read_text() and \
        "2026-11-13_00000003" not in (tmp_path / "p2.md").read_text()


def _fill(text: str, votes: dict) -> str:
    """The lawyer's pen: votes {case id: [7 votes or None]}, in the pack's order."""
    out, case, k = [], None, 0
    for line in text.split("\n"):
        if line.startswith("<!-- heldout:"):
            case, k = line.split(":")[1].split()[0], 0
        if line == "   - voto: " and case in votes:
            v = votes[case][k]
            k += 1
            line = line + ("" if v is None else str(v))
        if line == "   - note: " and case in votes and k == 1:
            line = line + "la tesi regge\n   ma va detto meglio"
        out.append(line)
    return "\n".join(out)


def test_the_filled_pack_is_recorded_per_version_and_the_table_follows(tmp_path, capsys):
    export, score = _script("heldout_export"), _script("heldout_score")
    _records(tmp_path, 4)
    pack = tmp_path / "pack.md"
    export.main(["--dir", str(tmp_path), "-n", "3", "--with-updates", "--out", str(pack)])
    filled = _fill(pack.read_text(encoding="utf-8"), {"2026-11-13_00000003": [4, 3, 5, 4, 2, 5, None],
                                                      "2026-11-12_00000002": [2, 2, 2, 2, 2, 2, 2]})
    (tmp_path / "filled.md").write_text(filled, encoding="utf-8")
    assert score.main([str(tmp_path / "filled.md"), "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "not scored yet (no voto): 2026-11-11_00000001" in out
    rows = heldout.scores(tmp_path)
    assert {r["id"]: r["mean"] for r in rows} == {"2026-11-13_00000003": 3.83, "2026-11-12_00000002": 2.0}
    assert rows[0]["notes"]["tesi"] == "la tesi regge ma va detto meglio" and rows[0]["sections"]["strategia"] is None
    assert heldout.version_means(rows) == [{"version": "77.0.1+def", "cases": 2, "mean": 2.92,
                                            "sections": {"tesi": 3.0, "categoria": 2.5, "coerenza": 3.5,
                                                         "processuale": 3.0, "mancanze": 2.0, "pregi": 3.5,
                                                         "strategia": 2.0}}]
    assert "| 77.0.1+def | 2 | 2.92 |" in out


def test_a_bad_voto_records_nothing(tmp_path, capsys):
    export, score = _script("heldout_export"), _script("heldout_score")
    _records(tmp_path, 1)
    pack = tmp_path / "pack.md"
    export.main(["--dir", str(tmp_path), "-n", "1", "--out", str(pack)])
    (tmp_path / "f.md").write_text(_fill(pack.read_text(), {"2026-11-10_00000000": [9, 1, 1, 1, 1, 1, 1]}))
    assert score.main([str(tmp_path / "f.md"), "--dir", str(tmp_path)]) == 1
    assert "outside 0-5" in capsys.readouterr().err and heldout.scores(tmp_path) == []


def test_a_case_scored_again_counts_once_with_its_later_score():
    rows = [{"id": "a", "version": "v1", "mean": 1.0, "sections": {"tesi": 1}},
            {"id": "a", "version": "v1", "mean": 3.0, "sections": {"tesi": 3}}]
    assert heldout.version_means(rows) == [{"version": "v1", "cases": 1, "mean": 3.0, "sections": {"tesi": 3.0}}]


# --- 4. the nightly ----------------------------------------------------------------------------------------------------------

def test_the_nightly_checks_the_footer_and_ends_with_the_lawyers_means(tmp_path, released):
    nightly = _script("nightly_sim")
    assert nightly.footer_check("Il parere.\n\nSunnitAI 77.0.0+abc1234")["ok"]
    bad = nightly.footer_check("Il parere.\n\nSunnitAI 76.0.0+old")
    assert not bad["ok"] and "76.0.0+old" in bad["why"]
    (tmp_path / "scores.jsonl").write_text(json.dumps({"id": "x", "version": "77.0.0+abc1234", "mean": 3.5,
                                                       "sections": {"tesi": 3.5}}) + "\n")
    text = nightly.report([{"id": "c1", "checks": [{"check": "version_footer", "ok": True, "why": ""}]}], None, [],
                          released, nightly.lawyer_lines(tmp_path))
    assert "Version: 77.0.0+abc1234" in text and "## The lawyer's scores per version" in text
    assert "| 77.0.0+abc1234 | 1 | 3.5 |" in text
    assert "- none recorded" in "\n".join(nightly.lawyer_lines(tmp_path / "missing"))


# --- the deploy exclude list -----------------------------------------------------------------------------------------------

def test_the_exclude_list_holds_the_pos_entries_in_the_form_rsync_matches_per_directory():
    """The PO's list (17 Nov 2026); data/'s entries without «data/»: each directory is its own rsync."""
    lines = {ln.strip() for ln in (HERE.parent / "scripts" / "deploy_exclude.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")}
    assert {".env*", "__pycache__/", "*.pyc", "logs/", "diagnose_reports/", ".git/", "state/", "/private_documents/",
            "/institutions_*.json", "article_cards*.jsonl", "*.egg-info/"} <= lines
    assert not any(ln.startswith("data/") for ln in lines)


# --- 5. the 31-case sim, retired ---------------------------------------------------------------------------------------------

def test_the_sim_runs_no_default_batch(monkeypatch, capsys):
    dl = _script("diagnose_long")
    monkeypatch.setattr(sys, "argv", ["diagnose_long.py", "--interactive-sim", "--single-shot", "--env", "/nonexistent"])
    monkeypatch.setattr(dl, "_eval_module", lambda: None)
    with pytest.raises(SystemExit):
        dl.main()
    assert "the 31-case batch is retired" in capsys.readouterr().err
