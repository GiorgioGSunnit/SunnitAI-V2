"""maint_feed.py: the feeder tops the queues up to a target, keeps the manifest format, sets aside files already fed,
honours the pause file; the batch check counts what must be looked at before feeding. All on temporary folders."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import maint_feed  # noqa: E402

CODES = [f"snpen_s50_a2026_n{n:05d}_tS.pdf" for n in range(1, 9)]


def make_batch(src, names=CODES, sidecars=True):
    os.makedirs(src, exist_ok=True)
    for n in names:
        with open(os.path.join(src, n), "wb") as fh:
            fh.write(b"%PDF-1.4 test")
        if sidecars:
            with open(os.path.join(src, n[:-4] + ".name"), "w", encoding="utf-8") as fh:
                fh.write(f"Cass. pen. n. {n[-9:-7]} del 2026")


def feed_args(tmp_path, queues, **extra):
    argv = ["feed", "--src", str(tmp_path / "src"), "--batch", "pdf2026", "--manifest", str(tmp_path / "manifest.txt"),
            "--pause-file", str(tmp_path / "feed.pause"), "--state", str(tmp_path / "state.json"),
            "--failed-dir", str(tmp_path / "failed")]
    for q in queues:
        argv += ["--queue", str(q)]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return argv


def test_plan_feed_fills_the_emptiest_queue_first_and_stops_at_the_cap():
    plan = maint_feed.plan_feed({"q1": 1, "q2": 3}, ["a", "b", "c", "d", "e"], cap=10)
    assert plan == [("a", "q2"), ("b", "q2"), ("c", "q1"), ("d", "q2")]       # needs met after 4, "e" stays
    assert maint_feed.plan_feed({"q1": 5}, ["a", "b", "c"], cap=2) == [("a", "q1"), ("b", "q1")]
    assert maint_feed.plan_feed({"q1": 0, "q2": -3}, ["a"], cap=5) == []       # full queues: nothing moves


def test_feed_tops_up_two_queues_writes_the_manifest_and_the_state(tmp_path, capsys):
    make_batch(tmp_path / "src")
    q1, q2 = tmp_path / "q1", tmp_path / "q2"
    os.makedirs(q2)
    with open(q2 / "snpen_s50_a2025_n00001_tS.pdf", "wb") as fh:   # q2 already holds one file
        fh.write(b"x")
    assert maint_feed.main(feed_args(tmp_path, [q1, q2], target=3)) == 0
    assert len(maint_feed.pdfs_in(str(q1))) == 3 and len(maint_feed.pdfs_in(str(q2))) == 3
    assert len(maint_feed.pdfs_in(str(tmp_path / "src"))) == 3                 # 8 - 5 fed
    assert os.path.isfile(q1 / (maint_feed.pdfs_in(str(q1))[0][:-4] + ".name"))  # the sidecar travels with the PDF
    lines = open(tmp_path / "manifest.txt", encoding="utf-8").read().splitlines()
    assert len(lines) == 5 and all(maint_feed.MANIFEST_LINE.match(l) for l in lines)
    assert lines[0].split()[1] == "pdf2026" and lines[0].endswith("_tS.pdf")
    state = json.load(open(tmp_path / "state.json", encoding="utf-8"))
    assert state["fed"] == 5 and state["left"] == 3 and not state["paused"]
    assert state["queues"] == {str(q1): 3, str(q2): 3}
    out = capsys.readouterr().out
    assert "batch=pdf2026 fed 5 (queues now: q1 3, q2 3), 3 left" in out


def test_feed_sets_aside_a_file_already_in_the_manifest_instead_of_feeding_it_twice(tmp_path, capsys):
    make_batch(tmp_path / "src")
    with open(tmp_path / "manifest.txt", "w", encoding="utf-8") as fh:
        fh.write(f"2026-11-01 pdf2025 {CODES[0]}\n")
    assert maint_feed.main(feed_args(tmp_path, [tmp_path / "q1"], target=100, max=2)) == 0
    aside = tmp_path / "src" / maint_feed.ALREADY_FED
    assert sorted(os.listdir(aside)) == [CODES[0][:-4] + ".name", CODES[0]]
    assert maint_feed.pdfs_in(str(tmp_path / "q1")) == CODES[1:3]              # --max 2
    assert "1 already fed set aside" in capsys.readouterr().out


def test_feed_does_nothing_while_the_pause_file_exists(tmp_path, capsys):
    make_batch(tmp_path / "src")
    open(tmp_path / "feed.pause", "w").close()
    assert maint_feed.main(feed_args(tmp_path, [tmp_path / "q1"])) == 0
    assert not os.path.isdir(tmp_path / "q1") or not maint_feed.pdfs_in(str(tmp_path / "q1"))
    assert not os.path.exists(tmp_path / "manifest.txt")
    assert "PAUSED, fed 0" in capsys.readouterr().out
    assert json.load(open(tmp_path / "state.json", encoding="utf-8"))["paused"] is True


def test_feed_reports_a_missing_staging_folder(tmp_path, capsys):
    assert maint_feed.main(feed_args(tmp_path, [tmp_path / "q1"])) == 1
    assert "staging folder not found" in capsys.readouterr().out


def test_check_counts_what_to_look_at_and_can_set_the_already_fed_aside(tmp_path, capsys):
    src = tmp_path / "src"
    make_batch(src, CODES[:4])
    make_batch(src, ["sentenza senza codice.pdf"], sidecars=False)
    with open(src / (CODES[1][:-4] + ".name"), "w", encoding="utf-8") as fh:   # same display name as CODES[0]
        fh.write("Cass. pen. n. 01 del 2026")
    with open(tmp_path / "manifest.txt", "w", encoding="utf-8") as fh:
        fh.write(f"2026-11-01 pdf2025 {CODES[2]}\n")
    argv = ["check", "--src", str(src), "--batch", "pdf2026", "--manifest", str(tmp_path / "manifest.txt")]
    assert maint_feed.main(argv) == 2
    out = capsys.readouterr().out
    assert "5 PDFs; by year 2026 4, ? 1; by branch ? 1, snpen 4" in out
    assert "duplicate file names inside the batch (case-insensitive): 0" in out
    assert "without the Italgiure code" in out and "e.g. sentenza senza codice.pdf" in out
    assert "without a .name sidecar" in out and ": 1" in out
    assert "sidecars whose display name repeats another's: 1" in out
    assert f"already handed to the queue (name in the manifest): 1  e.g. {CODES[2]}" in out
    assert "things to look at before feeding" in out
    assert maint_feed.main(argv + ["--set-aside"]) == 2                        # the other problems remain
    assert os.listdir(src / maint_feed.ALREADY_FED) == sorted([CODES[2][:-4] + ".name", CODES[2]])
    assert "4 left to feed" in capsys.readouterr().out


def test_duplicate_stems_ignore_case_and_extension():
    # on disk only Linux can hold both spellings (be2); Windows folds them, so the check is on names
    assert maint_feed.duplicate_stems(CODES[:2] + ["SNPEN_S50_A2026_N00001_TS.pdf"]) == ["snpen_s50_a2026_n00001_ts"]
    assert maint_feed.duplicate_stems(CODES) == []


def test_check_is_quiet_on_a_clean_batch(tmp_path, capsys):
    make_batch(tmp_path / "src", CODES[:3])
    assert maint_feed.main(["check", "--src", str(tmp_path / "src"), "--batch", "b", "--manifest",
                            str(tmp_path / "none.txt")]) == 0
    out = capsys.readouterr().out
    assert "OK to feed" in out and "e.g." not in out


@pytest.mark.parametrize("name,year,branch", [("snciv@s10@a2019@n01234@tS.pdf", "2019", "snciv"),
                                              ("snpen_s50_a2025_n07217_tO.pdf", "2025", "snpen")])
def test_italgiure_code_gives_year_and_branch(name, year, branch):
    m = maint_feed.ITALGIURE.search(name)
    assert m and m.group(2) == year and m.group(1).lower() == branch
