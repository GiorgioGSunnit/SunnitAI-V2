"""scripts/maint_cards.py: the maintenance agent's job 4, the nightly ruling cards (28 Oct)."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("maint_cards", HERE.parent / "scripts" / "maint_cards.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

LINKS = [{"id": "p1", "branch": "penale", "chars": 100}, {"id": "p2", "branch": "penale", "chars": 100},
         {"id": "p3", "branch": "penale", "chars": 0},                                  # no text: never carded
         {"id": "c1", "branch": "civile", "chars": 100, "outcome": "rigetto"},
         {"id": "c2", "branch": "civile", "chars": 100, "outcome": "estinto"}]           # a closure: skipped


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _args(tmp_path, write=True, **kw):
    return argparse.Namespace(write=write, dir=str(tmp_path), scripts="/s", cwd="/opt/chatbot", python="py",
                              branches=["penale", "civile"], workers=3, stop_at="00:45", **kw)


class Runner:
    """Fake processes: ps, the links refresh (writes the new links, one ruling more), the card runs (card every
    ruling of the branch left, as exp_ruling_cards.pick would)."""

    def __init__(self, tmp_path, ps="", fail_links=False):
        self.tmp, self.ps, self.fail_links, self.calls = tmp_path, ps, fail_links, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        result = type("R", (), {"returncode": 0, "stdout": ""})()
        if cmd[0] == "ps":
            result.stdout = self.ps
        elif cmd[1].endswith("exp_ruling_links.py"):
            if self.fail_links:
                result.returncode = 1
            else:
                _write(Path(cmd[-1]), LINKS + [{"id": "p4", "branch": "penale", "chars": 100}])
        else:
            branch, out = cmd[cmd.index("--branch") + 1], Path(cmd[cmd.index("--out") + 1])
            links = mc.read_jsonl(cmd[cmd.index("--links") + 1])
            done, _ = mc.card_ids(str(out))
            with open(out, "a", encoding="utf-8") as fh:
                for r in links:
                    if r["branch"] == branch and r["id"] not in done and r["chars"] and r.get("outcome") != "estinto":
                        fh.write(json.dumps({"id": r["id"], "branch": branch}) + "\n")
        return result


def test_counts_follow_the_card_scripts_rule():
    assert mc.to_card(LINKS, "penale", {"p1"}) == 1 and mc.to_card(LINKS, "civile", set()) == 1


def test_dry_run_runs_nothing(tmp_path):
    _write(tmp_path / "ruling_links.jsonl", LINKS)
    _write(tmp_path / "cards_pen_v3.jsonl", [{"id": "p1"}])
    runner, said = Runner(tmp_path), []
    line = mc.night(_args(tmp_path, write=False), runner, said.append)
    assert line.startswith("DRY RUN: to card penale 1, civile 1") and runner.calls == []
    assert any("exp_ruling_cards.py" in s and "--stop-at 00:45" in s for s in said)


def test_a_night_refreshes_cards_swaps_and_keeps_the_previous_file(tmp_path):
    _write(tmp_path / "ruling_links.jsonl", LINKS)
    _write(tmp_path / "cards_pen_v3.jsonl", [{"id": "p1"}, {"id": "p9", "errore": "json"}])
    _write(tmp_path / "cards_civ_v1.jsonl", [])
    runner = Runner(tmp_path)
    line = mc.night(_args(tmp_path), runner, lambda s: None)
    assert line.startswith("cards +3 (penale +2, civile +1) · still to card penale 0, civile 0 · failed penale 1, civile 0")
    assert "links 6 rulings" in line and "penale live 2 -> 4 lines" in line
    assert mc.card_ids(str(tmp_path / "cards_pen_v3.jsonl"))[0] == {"p1", "p2", "p4"}
    assert mc.lines(str(tmp_path / "cards_pen_v3.jsonl.prev")) == 2                   # the previous file, kept
    assert not (tmp_path / "work_penale.jsonl").exists()                               # moved in, not copied
    assert all(c[0] == "py" for c in runner.calls[1:])


def test_skips_when_a_card_run_is_going_and_stops_when_the_links_fail(tmp_path):
    _write(tmp_path / "ruling_links.jsonl", LINKS)
    busy = Runner(tmp_path, ps="bash /root/ruling_cards/night_cards.sh\npy /root/ruling_cards/exp_ruling_cards.py --branch penale")
    assert mc.night(_args(tmp_path), busy, lambda s: None) == "SKIPPED: a card run is already going"
    failing = Runner(tmp_path, fail_links=True)
    assert mc.night(_args(tmp_path), failing, lambda s: None).startswith("FAILED: step 1's links refresh")
    assert mc.read_jsonl(str(tmp_path / "ruling_links.jsonl")) == LINKS               # untouched


def test_swap_refuses_a_smaller_work_copy_and_resumes_a_crashed_one(tmp_path):
    live, work = tmp_path / "cards_pen_v3.jsonl", tmp_path / "work_penale.jsonl"
    _write(live, [{"id": "p1"}, {"id": "p2"}])
    _write(work, [{"id": "p1"}])
    assert mc.swap(str(work), str(live)).startswith("refused") and mc.lines(str(live)) == 2
    # a crashed night left a bigger work copy: the next night resumes it instead of starting from the live file
    _write(tmp_path / "ruling_links.jsonl", LINKS)
    _write(work, [{"id": "p1"}, {"id": "p2"}, {"id": "p4-from-crash"}])
    args = _args(tmp_path)
    args.branches = ["penale"]
    mc.night(args, Runner(tmp_path), lambda s: None)
    assert "p4-from-crash" in mc.card_ids(str(live))[0]
