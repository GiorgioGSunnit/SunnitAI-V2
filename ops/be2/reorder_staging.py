"""Changes the ORDER a staged Cassazione batch is fed in, by moving the files that can wait into side folders of the
staging folder. The feeder (maint_feed.py) hands over the PDFs that sit directly in the staging folder, in name order,
and does not look into side folders: what stays at the top is fed first, what is set aside waits until it is brought
back. Nothing is deleted or renamed; every PDF moves together with its .name sidecar.

Why (measured 18 Nov 2026 on the rulings our answers used): judgments of the ordinary sections and everything of the
Joint Sections are what the answers quote; orders come next; decrees, interlocutory orders and the penal filter
section (VII) were never used.

Groups, from the file's Italgiure code and a list of the file names cited by rulings we hold:
    stay                 judgments (tS) of the ordinary sections; everything of the Joint Sections (sU0)
    cited_orders   ->    _later_1_cited_orders    orders (tO) of the ordinary sections that are in the cited list
    orders         ->    _later_2_orders          the other orders of the ordinary sections
    last           ->    _later_3_last            decrees (tD), interlocutory (tI), the penal filter section (s70)
A name without the code stays where it is and is counted.

    python3 -I reorder_staging.py <staging>                                   the plan: counts only, moves nothing
    python3 -I reorder_staging.py <staging> --cited <names file> --set-aside --expect N
    python3 -I reorder_staging.py <staging> --bring-back cited_orders|orders|last|all --expect N
    python3 reorder_staging.py --selftest

--expect is the number of PDFs the run will move, as the plan prints it: a folder that changed since the plan stops
the run before the first move. The feeder runs every ten minutes, on the minute ending in 0: the tool moves only in
a minute ending in 1 to 6, unless the feeder's pause file is there. Prints counts and folder names only.
"""
import datetime as dt
import os
import re
import sys
import tempfile
from collections import Counter

CODE = re.compile(r"(snpen|snciv)[@_ ]s(\w\w)[@_ ]a((?:19|20)\d{2})[@_ ]n(\d+)[@_ ]t([a-z])", re.IGNORECASE)
SIDE = {"cited_orders": "_later_1_cited_orders", "orders": "_later_2_orders", "last": "_later_3_last"}
PAUSE_FILE = "/var/log/maint/feed.pause"


def code_of(name: str):
    """(archive, section, year, number, act) of a file name, lower case; None without the Italgiure code."""
    m = CODE.search(name or "")
    return (m.group(1).lower(), m.group(2).lower(), m.group(3), int(m.group(4)), m.group(5).lower()) if m else None


def group_of(name: str, cited: set) -> str:
    code = code_of(name)
    if code is None:
        return "other"
    archive, section, _, _, act = code
    if section.startswith("u"):
        return "stay"
    if (archive == "snpen" and section == "70") or act in ("d", "i"):
        return "last"
    if act == "s":
        return "stay"
    return "cited_orders" if code in cited else "orders"


def pdfs(folder: str) -> list:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder, f))) \
        if os.path.isdir(folder) else []


def read_cited(path: str) -> set:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return {c for c in (code_of(line) for line in fh) if c}


def plan(staging: str, cited: set) -> dict:
    groups = {"stay": [], "cited_orders": [], "orders": [], "last": [], "other": []}
    for name in pdfs(staging):
        groups[group_of(name, cited)].append(name)
    return groups


def move_all(names: list, src: str, dest: str) -> int:
    """Moves each PDF and its sidecar from src to dest. Stops before the first move if a name is already in dest."""
    os.makedirs(dest, exist_ok=True)
    taken = set(os.listdir(dest))
    clash = [n for n in names if n in taken or os.path.splitext(n)[0] + ".name" in taken]
    if clash:
        raise SystemExit(f"STOP, nothing moved: {len(clash)} names are already in {os.path.basename(dest) or dest}")
    moved = 0
    for name in names:
        sidecar = os.path.splitext(name)[0] + ".name"
        os.rename(os.path.join(src, name), os.path.join(dest, name))
        if os.path.isfile(os.path.join(src, sidecar)):
            os.rename(os.path.join(src, sidecar), os.path.join(dest, sidecar))
        moved += 1
    return moved


def report(staging: str, cited: set) -> list:
    g = plan(staging, cited)
    top = sum(len(v) for v in g.values())
    kinds = Counter()
    for name in g["stay"]:
        code = code_of(name)
        kinds["Joint Sections" if code[1].startswith("u") else "judgments, ordinary sections"] += 1
    lines = [f"staging: {staging}", f"PDFs at the top level (what the feeder sees): {top:,}",
             f"   stay, fed first: {len(g['stay']):,}  (" + ", ".join(f"{k} {n:,}" for k, n in sorted(kinds.items())) + ")"]
    for key in ("cited_orders", "orders", "last"):
        lines.append(f"   would go to {SIDE[key]}: {len(g[key]):,}")
    lines.append(f"   names without the code, left in place: {len(g['other']):,}")
    sides = sorted(d for d in os.listdir(staging) if d.startswith("_") and os.path.isdir(os.path.join(staging, d)))
    lines.append("side folders now: " + (", ".join(f"{d} {len(pdfs(os.path.join(staging, d))):,}" for d in sides) or "none"))
    return lines


def feeder_minute(now=None) -> bool:
    """Too close to a feeder run (cron */10) to move files: any minute not ending in 1 to 6, while the feeder is not paused."""
    return (now or dt.datetime.now()).minute % 10 not in range(1, 7) and not os.path.exists(PAUSE_FILE)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        selftest()
        return 0

    def option(flag):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1] if i + 1 < len(argv) else None
            del argv[i:i + 2]
            return value
        return None
    cited_path, expect, back = option("--cited"), option("--expect"), option("--bring-back")
    set_aside = "--set-aside" in argv
    argv = [a for a in argv if a != "--set-aside"]
    if len(argv) != 1 or (set_aside and back) or (back and back not in list(SIDE) + ["all"]):
        print(__doc__)
        return 1
    staging = argv[0]
    if not os.path.isdir(staging):
        print(f"no such folder: {staging}")
        return 1
    cited = read_cited(cited_path) if cited_path else set()
    if not (set_aside or back):
        print("\n".join(report(staging, cited)))
        g = plan(staging, cited)
        n = sum(len(g[k]) for k in SIDE)
        print(f"cited list: {len(cited):,} names" if cited_path else "cited list: none given (every order counts as not cited)")
        print(f"to set aside: --set-aside --expect {n}")
        return 0
    if feeder_minute():
        print("STOP, nothing moved: too close to a feeder run. Run it again in a minute ending in 1 to 6.")
        return 1
    if set_aside:
        if not cited_path:
            print("STOP, nothing moved: --set-aside needs --cited <names file>")
            return 1
        g = plan(staging, cited)
        todo = [(k, g[k], staging, os.path.join(staging, SIDE[k])) for k in SIDE]
    else:
        keys = list(SIDE) if back == "all" else [back]
        todo = [(k, pdfs(os.path.join(staging, SIDE[k])), os.path.join(staging, SIDE[k]), staging) for k in keys]
    total = sum(len(names) for _, names, _, _ in todo)
    if expect is None or int(expect) != total:
        print(f"STOP, nothing moved: this run would move {total:,} PDFs, --expect says {expect}")
        return 1
    for key, names, src, dest in todo:
        moved = move_all(names, src, dest)
        print(f"{key}: {moved:,} PDFs moved to {os.path.basename(dest.rstrip('/')) if set_aside else 'the top level'}")
    print("\n".join(report(staging, cited)))
    return 0


def selftest() -> None:
    cited = {code_of("snciv@s10@a2025@n00007@tO.clean.pdf")}
    assert group_of("snciv@s10@a2025@n00005@tS.clean.pdf", cited) == "stay"
    assert group_of("snciv@sU0@a2025@n00006@tO.clean.pdf", cited) == "stay"          # Joint Sections: every act
    assert group_of("snciv@s10@a2025@n00007@tO.clean.pdf", cited) == "cited_orders"
    assert group_of("snciv_s10_a2025_n00007_tO_oY.clean.pdf", cited) == "cited_orders"  # underscores, the oY mark
    assert group_of("snciv@s10@a2025@n00008@tO.clean.pdf", cited) == "orders"
    assert group_of("snciv@s30@a2025@n00009@tD.clean.pdf", cited) == "last" and group_of("snciv@sL0@a2025@n1@tI.pdf", cited) == "last"
    assert group_of("snpen@s70@a2024@n00010@tS.clean.pdf", cited) == "last"          # the penal filter section
    assert group_of("snpen@s10@a2024@n00011@tS.clean.pdf", cited) == "stay" and group_of("appunti.pdf", cited) == "other"
    with tempfile.TemporaryDirectory() as tmp:
        names = ["snciv@s10@a2025@n00005@tS.clean.pdf", "snciv@sU0@a2025@n00006@tO.clean.pdf", "snciv@s10@a2025@n00007@tO.clean.pdf",
                 "snciv@s10@a2025@n00008@tO.clean.pdf", "snciv@s30@a2025@n00009@tD.clean.pdf", "appunti.pdf"]
        for n in names:
            open(os.path.join(tmp, n), "w").close()
            open(os.path.join(tmp, os.path.splitext(n)[0] + ".name"), "w").close()
        os.makedirs(os.path.join(tmp, "_no_text"))
        listf = os.path.join(tmp, "cited.txt")
        with open(listf, "w", encoding="utf-8") as fh:
            fh.write("snciv@s10@a2025@n00007@tO.clean.pdf\n")
        global PAUSE_FILE
        PAUSE_FILE = listf                                    # any existing file: the minute guard is off for the test
        assert main([tmp]) == 0
        assert main([tmp, "--cited", listf, "--set-aside", "--expect", "2"]) == 1          # wrong count: nothing moves
        assert len(pdfs(tmp)) == 6
        assert main([tmp, "--cited", listf, "--set-aside", "--expect", "3"]) == 0
        assert pdfs(tmp) == ["appunti.pdf", "snciv@s10@a2025@n00005@tS.clean.pdf", "snciv@sU0@a2025@n00006@tO.clean.pdf"]
        for key, n in (("cited_orders", "n00007"), ("orders", "n00008"), ("last", "n00009")):
            got = sorted(os.listdir(os.path.join(tmp, SIDE[key])))
            assert len(got) == 2 and all(n in x for x in got) and got[0].endswith(".name"), got   # the PDF and its sidecar
        assert main([tmp, "--cited", listf, "--set-aside", "--expect", "0"]) == 0          # again: nothing left to move
        assert main([tmp, "--bring-back", "cited_orders", "--expect", "1"]) == 0
        assert "snciv@s10@a2025@n00007@tO.clean.pdf" in pdfs(tmp) and os.path.isfile(os.path.join(tmp, "snciv@s10@a2025@n00007@tO.clean.name"))
        open(os.path.join(tmp, SIDE["orders"], "snciv@s10@a2025@n00005@tS.clean.pdf"), "w").close()   # a clash at the top
        try:
            main([tmp, "--bring-back", "orders", "--expect", "2"])
            raise AssertionError("a clash must stop the run")
        except SystemExit as stop:
            assert "nothing moved" in str(stop)
        os.remove(os.path.join(tmp, SIDE["orders"], "snciv@s10@a2025@n00005@tS.clean.pdf"))
        assert main([tmp, "--bring-back", "all", "--expect", "2"]) == 0
        assert len(pdfs(tmp)) == 6 and len([f for f in os.listdir(tmp) if f.endswith(".name")]) == 6
        assert all(not pdfs(os.path.join(tmp, d)) for d in SIDE.values())
    late = dt.datetime(2026, 11, 18, 19, 9)
    PAUSE_FILE = "/nonexistent/feed.pause"
    assert feeder_minute(late) and feeder_minute(late.replace(minute=20)) and feeder_minute(late.replace(minute=7))
    assert not feeder_minute(late.replace(minute=13)) and not feeder_minute(late.replace(minute=1)) and not feeder_minute(late.replace(minute=46))
    print("selftest OK")


if __name__ == "__main__":
    sys.exit(main())
