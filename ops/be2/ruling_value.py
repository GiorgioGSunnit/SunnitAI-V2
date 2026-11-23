"""Read-only: which Cassazione rulings do the most work for us, and does the file name tell beforehand?

The cost of loading a ruling grows with its pieces (the Sections the ingest makes of it). What it gives back, as far as
the graph shows: its links to code articles (CITES) and, with --cited, how often the other rulings we hold cite it.

Without --cited (light: three counts, a minute or two):
  per archive and year, per section, per type of decision: rulings, pieces, links, rulings without a link, links per
  piece, and each group's share of the cost (pieces) against its share of the links;
  how concentrated the links are: the share held by the top 10%, 25%, 50% of rulings.

With --cited (reads the text of every ruling in batches, about half an hour: run it off-hours):
  the same groups with the share of rulings cited by another ruling we hold;
  the rulings cited that we do NOT hold, by year: how many, and how few of them cover most of the citations;
  the table itself in ruling_value_cited.csv beside this script: branch, year, number, times cited, citing rulings,
  held (1/0). Numbers only.

With --by-night [--nights N] (as light as the plain run; prints only this):
  the links by the night the 04:00 linker read the ruling (Document.cites_linked_at), the last N nights (default 8),
  and the same nights by loading-priority group (judgments + Joint Sections / orders / last group).

Prints counts only: never a name, a text or a ruling's number.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ruling_value.py              # be2, reads health.env
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ruling_value.py --by-night   # new links, night by night
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ruling_value.py --cited      # off-hours
    python ops/be2/ruling_value.py --selftest                                             # no database
"""
import csv
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

CODE = re.compile(r"(snpen|snciv)[@_ ]s(\w\w)[@_ ]a((?:19|20)\d{2})[@_ ]n(\d+)[@_ ]t([a-z])", re.IGNORECASE)
BATCH, PAUSE = 200, 0.5            # rulings per text query, seconds between two queries (--cited)

# The citations of Cassation rulings inside a text: copied as it is from scripts/unlinked_citations.py (run 3, 29 Oct 2026).
_YEAR = r"(?:19|20)\d{2}"
_DATE = r"(?:\d{1,2}°?\s+[a-zà-ù]+\s+|\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*)"
_PRECEDENT = re.compile(
    r"\b(?P<court>cass\b\.?(?:\s*(?:civ|pen|lav|trib)\.?)?|cassazione|s\.\s?u\.|ss\.\s?uu\.|"
    r"sez(?:ion[ei]|\.)?\s*(?:un(?:ite)?\b\.?|u\b\.?|\d{1,2}\b|[ivx]{1,4}\b|lav\b|trib\b|f(?:er(?:iale)?)?\b))"
    r"(?P<pre>[^;\n]{0,90}?)(?:"
    rf"(?<![\d/])(?:\bn\.?\s*)?(?P<n1>\d{{1,6}})\s*/\s*(?P<y1>{_YEAR})\b"                 # n. 18672/2019, 18672/2019
    rf"|\bn\.?\s*(?P<n2>\d{{1,6}})\s*,?\s*del(?:l['’]\s*|\s+){_DATE}?(?P<y2>{_YEAR})\b"    # n. 5501 del (15/02/)2020
    rf"|{_DATE}(?P<y3>{_YEAR})\s*,?\s*n\.?\s*(?P<n3>\d{{1,6}})\b"                        # 7 aprile 2014, n. 8053
    r")", re.IGNORECASE)
# "Cass. ... d.P.R. n. 115/2002": a law's number, not a ruling's (run 1 counted n. 115/2002, 103/2017, 546/1992).
_LAW_BEFORE_NUMBER = re.compile(
    r"\b(?:d\.?\s?p\.?\s?r\.?|d\.?\s?lgs\.?|lgs\.|d\.?\s?l\.|legge|\bl\.|r\.?\s?d\.?|d\.?\s?m\.?|"
    r"decreto(?:[\s-]+(?:legislativo|legge|ministeriale|del\s+presidente\s+della\s+repubblica))?|"
    r"reg(?:olamento|\.)?(?:\s*\(\s*(?:ue|ce|cee)\s*\))?|direttiva)\s*$", re.IGNORECASE)
# "... ricorso per cassazione ... (Corte cost. n. 186 del 2000)": the Constitutional Court's, not a Cassation ruling
# (run 2's 4th most cited "precedent", from the inadmissibility sentence).
_OTHER_COURT = re.compile(r"\bcost\b|costituzional|\bc\.\s?cost|europea|\bcgue\b|\bcedu\b", re.IGNORECASE)
# A "Sez." with no "Cass" can be another court's: "Cons. Stato, Sez. IV", "Trib. Milano, sez. II", "Corte conti".
_OTHER_COURT_BEFORE = re.compile(
    r"(?:cons(?:iglio|\.)?\s*(?:di\s+)?st(?:ato|\.)|\bt\.?\s?a\.?\s?r\.?\b|corte\s+(?:dei\s+)?conti|c\.\s?conti|"
    r"corte\s+(?:d['’]\s*)?app(?:ello|\.)|\bapp\.\s|trib(?:unale|\.)\s+[a-z]|comm(?:issione|\.)\s*trib|corte\s+giust|"
    r"grande\s+sezione|\bcgue\b|\bcedu\b|corte\s+e\.?d\.?u|corte\s+cost|\bc\.\s?cost)[^;\n]{0,30}$", re.IGNORECASE)
_FOOTER = re.compile(r"corte\s+di\s+cassazione\s*[-–—]\s*copia\s+non\s+ufficiale(?:\s+\d+)?", re.IGNORECASE)
_SUFF = "bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies"
_OWN_CASE = re.compile(r"ricors|r\.\s?g\.|\brgn\b|reg\.\s*gen|iscritt|ruolo", re.IGNORECASE)
FORMS = {"n1": "n. X/year", "n2": "n. X del (date) year", "n3": "date, n. X"}


def precedents(text: str, own: Optional[Tuple[str, str]] = None, forms: Optional[Counter] = None) -> List[Tuple[str, str]]:
    """(number, year) of every Cassation ruling the text cites -- not the case's own docket number ("sul ricorso
    iscritto al n. 15966/2025"), nor the ruling's own (number, year) `own` from its header. `forms` counts the form
    each one was written in."""
    text = _FOOTER.sub(" ", text or "")
    out = []
    for m in _PRECEDENT.finditer(text):
        if _OWN_CASE.search(m.group(0)) or _LAW_BEFORE_NUMBER.search(m.group("pre")[-30:]) \
                or _OTHER_COURT.search(m.group("pre")):
            continue
        if m.group("court").lower().startswith("sez") and _OTHER_COURT_BEFORE.search(text[max(0, m.start() - 60):m.start()]):
            continue
        key = next(k for k in FORMS if m.group(k))
        found = (str(int(m.group(key))), m.group("y" + key[1]))
        if found == own:
            continue
        out.append(found)
        if forms is not None:
            forms[FORMS[key]] += 1
    return out


def code_of(file_name, name):
    """(archive, section, year, number, type) from the Italgiure code of the file name or the display name; else None."""
    m = CODE.search(file_name or "") or CODE.search(name or "")
    return (m.group(1).lower(), m.group(2).lower(), m.group(3), str(int(m.group(4))), m.group(5).upper()) if m else None


def branch_of(archive: str) -> str:
    return "penale" if archive == "snpen" else "civile"


ROUTINE_SHARE = 0.20        # an article linked by this share of an archive's rulings is a routine citation
BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10 ** 9, "21+"))


def routine_articles(articles: dict, archive_of: dict, share: float = ROUTINE_SHARE) -> dict:
    """articles = {ruling id: {(code, article): links}} -> {archive: [((code, article), share of its rulings), ...]},
    the twelve most linked, most frequent first. The ones at `share` or more are the routine ones (the formula of an
    inadmissibility order, the costs): they say little about what the ruling decides."""
    seen, total = defaultdict(Counter), Counter(archive_of.values())
    for doc, arts in articles.items():
        if doc in archive_of:
            seen[archive_of[doc]].update(arts.keys())
    return {a: [(art, n / total[a]) for art, n in c.most_common(12)] for a, c in seen.items()}


def table(title: str, rows: dict, cited: bool) -> list:
    """rows = {group label: [ruling dicts with pieces, links, core, cited]} -> one line per group, largest first."""
    everything = [r for rs in rows.values() for r in rs]
    all_pieces, all_links, all_core = (sum(r[k] for r in everything) or 1 for k in ("pieces", "links", "core"))
    head = (f"  {'group':<16}{'rulings':>9}{'pieces':>8}{'links':>7}{'core':>6}{'no core':>9}{'core/pc':>9}"
            f"{'% cost':>8}{'% links':>9}{'% core':>8}")
    out = [title, head + (f"{'cited':>8}{'cited 3+':>10}" if cited else "")]
    # pieces, links, core = the median per ruling; core = links without the routine ones; no core = rulings with none;
    # core/pc = core links per piece of the whole group; % cost = the group's share of all pieces;
    # % links, % core = its share of all links, of all core links; cited = cited by a ruling we hold
    for label, rs in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        pieces, links, core = (sum(r[k] for r in rs) for k in ("pieces", "links", "core"))
        line = (f"  {label:<16}{len(rs):>9,}{statistics.median(r['pieces'] for r in rs):>8.0f}"
                f"{statistics.median(r['links'] for r in rs):>7.0f}{statistics.median(r['core'] for r in rs):>6.0f}"
                f"{100 * sum(r['core'] == 0 for r in rs) / len(rs):>8.0f}%{core / pieces if pieces else 0:>9.2f}"
                f"{100 * pieces / all_pieces:>7.1f}%{100 * links / all_links:>8.1f}%{100 * core / all_core:>7.1f}%")
        if cited:
            line += (f"{100 * sum(r['cited'] > 0 for r in rs) / len(rs):>7.0f}%"
                     f"{100 * sum(r['cited'] >= 3 for r in rs) / len(rs):>9.0f}%")
        out.append(line)
    return out


def bucket_table(rulings) -> list:
    """What a filter on the number of core links would keep: per size of that number, the rulings, their cost, their links."""
    out = ["BY NUMBER OF CORE LINKS OF THE RULING (countable from its text before the costly step)",
           f"  {'archive':<9}{'core links':<12}{'rulings':>9}{'% rulings':>11}{'% cost':>8}{'% core':>8}"]
    for archive in sorted({r["code"][0] for r in rulings}):
        rs = [r for r in rulings if r["code"][0] == archive]
        pieces, core = sum(r["pieces"] for r in rs) or 1, sum(r["core"] for r in rs) or 1
        for lo, hi, label in BUCKETS:
            b = [r for r in rs if lo <= r["core"] <= hi]
            out.append(f"  {archive:<9}{label:<12}{len(b):>9,}{100 * len(b) / len(rs):>10.1f}%"
                       f"{100 * sum(r['pieces'] for r in b) / pieces:>7.1f}%{100 * sum(r['core'] for r in b) / core:>7.1f}%")
    return out


def concentration(values, shares=(10, 25, 50)) -> str:
    """'top 10% hold 41%, top 25% hold 66%, ...' of the total of `values`."""
    ordered, total = sorted(values, reverse=True), sum(values) or 1
    return ", ".join(f"top {s}% hold {100 * sum(ordered[:max(1, len(ordered) * s // 100)]) / total:.0f}%" for s in shares)


def report(rulings, cited: bool, routine: dict = None) -> list:
    """rulings = dicts code (archive, section, year, number, type), pieces, links, core, cited."""
    out = [f"rulings with an Italgiure code: {len(rulings):,}; pieces {sum(r['pieces'] for r in rulings):,}; "
           f"links to code articles {sum(r['links'] for r in rulings):,}, of which core {sum(r['core'] for r in rulings):,}", ""]
    for archive, arts in sorted((routine or {}).items()):
        out.append(f"{archive}: the most linked articles and the share of its rulings that link each "
                   f"(routine = {ROUTINE_SHARE:.0%} or more, marked *):")
        out.append("   " + ", ".join(f"{a} {c} {s:.0%}{'*' if s >= ROUTINE_SHARE else ''}" for (c, a), s in arts))
    out.append("")
    by = lambda key: (lambda groups: {k: v for k, v in groups.items()})(_group(rulings, key))
    out += table("BY ARCHIVE AND YEAR", by(lambda c: f"{c[0]} {c[2]}"), cited) + [""]
    out += table("BY SECTION (the two characters after «s» in the file name)", by(lambda c: f"{c[0]} s{c[1]}"), cited) + [""]
    out += table("BY TYPE OF DECISION (the letter after «t»)", by(lambda c: f"{c[0]} t{c[4]}"), cited) + [""]
    out += table("BY SECTION AND TYPE (groups of 200 rulings or more)",
                 {k: v for k, v in by(lambda c: f"{c[0]} s{c[1]} t{c[4]}").items() if len(v) >= 200}, cited) + [""]
    out += bucket_table(rulings) + [""]
    for archive in sorted({r["code"][0] for r in rulings}):
        rs = [r for r in rulings if r["code"][0] == archive]
        out.append(f"{archive}: links, {concentration([r['links'] for r in rs])}")
        out.append(f"{archive}: core links, {concentration([r['core'] for r in rs])}")
        if cited:
            out.append(f"{archive}: citations by rulings we hold, {concentration([r['cited'] for r in rs])}")
    return out


KINDS = ("judgments + Joint Sections", "orders", "last group")
NOT_READ = "not read yet"


def kind_of(code) -> str:
    """The loading-priority group of a ruling: the rule of reorder_staging.group_of."""
    archive, section, _, _, act = code
    if section.startswith("u"):
        return KINDS[0]
    if (archive == "snpen" and section == "70") or act in ("D", "I"):
        return KINDS[2]
    return KINDS[0] if act == "S" else KINDS[1]


def night_report(rulings, nights: int = 8) -> list:
    """Links by the night the linker read the ruling (r["linked"], an ISO time or None): a night holds the rulings
    written since the night before, so the newest are «not read yet». Then the same nights by priority group."""
    by_night = defaultdict(list)
    for r in rulings:
        by_night[(r.get("linked") or "")[:10] or NOT_READ].append(r)
    days = sorted(d for d in by_night if d != NOT_READ)[-nights:]
    out = ["LINKS TO CODE ARTICLES, BY THE NIGHT THE LINKER READ THE RULING",
           f"  {'night':<14}{'rulings':>9}{'judgments':>11}{'links':>9}{'core':>9}{'links/ruling':>14}{'core/ruling':>13}{'no link':>9}"]
    # judgments = the first priority group; core = links without the routine articles; no link = rulings with none
    for d in days:
        rs = by_night[d]
        links, core = sum(r["links"] for r in rs), sum(r["core"] for r in rs)
        out.append(f"  {d:<14}{len(rs):>9,}{sum(kind_of(r['code']) == KINDS[0] for r in rs):>11,}{links:>9,}{core:>9,}"
                   f"{links / len(rs):>14.1f}{core / len(rs):>13.1f}{100 * sum(r['links'] == 0 for r in rs) / len(rs):>8.0f}%")
    if by_night.get(NOT_READ):
        out.append(f"  {NOT_READ:<14}{len(by_night[NOT_READ]):>9,}   written since the last night: their links come tonight")
    out += ["", f"THE SAME {len(days)} NIGHTS BY LOADING-PRIORITY GROUP",
            f"  {'group':<28}{'rulings':>9}{'pieces':>9}{'links':>9}{'core':>9}{'links/ruling':>14}{'core/ruling':>13}{'core/piece':>12}"]
    recent = [r for d in days for r in by_night[d]]
    for kind in KINDS:
        rs = [r for r in recent if kind_of(r["code"]) == kind]
        if rs:
            pieces, links, core = (sum(r[k] for r in rs) for k in ("pieces", "links", "core"))
            out.append(f"  {kind:<28}{len(rs):>9,}{pieces:>9,}{links:>9,}{core:>9,}{links / len(rs):>14.1f}"
                       f"{core / len(rs):>13.1f}{core / pieces if pieces else 0:>12.2f}")
    return out


def _group(rulings, key) -> dict:
    groups = defaultdict(list)
    for r in rulings:
        groups[key(r["code"])].append(r)
    return groups


def missing_report(counts: dict, held: set) -> list:
    """counts = {(branch, number, year): times cited}; held = the keys of the rulings in the graph."""
    out = ["RULINGS CITED THAT WE DO NOT HOLD (a citation counts in the citing ruling's own branch)",
           f"  {'branch':<8}{'years':<12}{'rulings':>9}{'citations':>11}{'cited 2+':>10}{'cited 5+':>10}   to cover 50% / 80% of the citations"]
    lost = {k: v for k, v in counts.items() if k not in held and 1990 <= int(k[2]) <= 2026}
    for branch in sorted({k[0] for k in lost}):
        for label, lo, hi in (("2025", 2025, 2025), ("2021-2024", 2021, 2024), ("2017-2020", 2017, 2020),
                              ("2012-2016", 2012, 2016), ("before 2012", 1990, 2011), ("all", 1990, 2026)):
            vals = sorted((v for k, v in lost.items() if k[0] == branch and lo <= int(k[2]) <= hi), reverse=True)
            if not vals:
                continue
            total, need = sum(vals), []
            for share in (0.5, 0.8):
                run = 0
                for i, v in enumerate(vals, 1):
                    run += v
                    if run >= share * total:
                        need.append(i)
                        break
            out.append(f"  {branch:<8}{label:<12}{len(vals):>9,}{total:>11,}{sum(v >= 2 for v in vals):>10,}"
                       f"{sum(v >= 5 for v in vals):>10,}   {need[0]:,} / {need[1]:,} rulings")
    return out


def selftest() -> None:
    assert code_of("snpen@s70@a2025@n00012@tO.clean.pdf", None) == ("snpen", "70", "2025", "12", "O")
    assert code_of(None, "[CIVILE] snciv su0 a2026 n00100 tS.clean") == ("snciv", "u0", "2026", "100", "S") and code_of("x.pdf", "y") is None
    text = ("Come affermato da Sez. 3, n. 12345 del 15/02/2019, Rv. 275000, e da Cass. 7 aprile 2014, n. 8053, nonché "
            "Sez. U, n. 18672/2019; ai sensi del d.P.R. n. 115/2002; sul ricorso iscritto al n. 15966/2025 R.G.; "
            "Corte cost. n. 186 del 2000; Cons. Stato, Sez. IV, n. 555/2020.")
    assert precedents(text) == [("12345", "2019"), ("8053", "2014"), ("18672", "2019")], precedents(text)
    assert precedents("Sez. 3, n. 12345 del 15/02/2019", own=("12345", "2019")) == []
    mk = lambda code, pieces, links, core, cited=0: {"code": code, "pieces": pieces, "links": links, "core": core, "cited": cited}
    rulings = ([mk(("snpen", "10", "2025", str(i), "S"), 12, 20, 15, 4) for i in range(200)]
               + [mk(("snpen", "70", "2025", str(i), "O"), 4, 1 if i % 2 else 0, 0) for i in range(300)]
               + [mk(("snciv", "30", "2026", "7", "O"), 10, 9, 9)])
    arts = {"a": {("c.p.p.", "616"): 2, ("c.p.", "624"): 1}, "b": {("c.p.p.", "616"): 1}, "c": {("c.p.", "575"): 3}, "x": {("c.c.", "2043"): 1}}
    rout = routine_articles(arts, {"a": "snpen", "b": "snpen", "c": "snpen", "x": "snciv"})
    assert rout["snpen"][0] == (("c.p.p.", "616"), 2 / 3) and dict(rout["snpen"])[("c.p.", "575")] == 1 / 3 and len(rout["snciv"]) == 1
    text = "\n".join(report(rulings, cited=True, routine=rout))
    assert "616 c.p.p. 67%*" in text and "624 c.p. 33%*" in text, text
    #                           rulings pieces links core no_core core/pc %cost %links %core cited cited3+
    assert re.search(r"snpen s70\s+300\s+4\s+0\s+0\s+100%\s+0\.00\s+33\.2%\s+3\.6%\s+0\.0%\s+0%\s+0%", text), text     # a third of the cost, none of the core
    assert re.search(r"snpen s10\s+200\s+12\s+20\s+15\s+0%\s+1\.25\s+66\.5%\s+96\.2%\s+99\.7%\s+100%\s+100%", text), text
    assert "snpen s70 tO" in text and "snciv s30 tO" not in text.split("BY SECTION AND TYPE")[1].split("BY NUMBER OF CORE")[0]   # under 200
    assert re.search(r"snpen    0\s+300\s+60\.0%\s+33\.3%\s+0\.0%", text) and re.search(r"snpen    11-20\s+200\s+40\.0%\s+66\.7%\s+100\.0%", text), text
    assert "snpen: links, top 10% hold 24%, top 25% hold 60%, top 50% hold 98%" in text, text
    assert concentration([10, 0, 0, 0], (25, 50)) == "top 25% hold 100%, top 50% hold 100%"
    counts = {("penale", "1", "2019"): 6, ("penale", "2", "2019"): 3, ("penale", "3", "2019"): 1, ("penale", "4", "2025"): 9,
              ("penale", "5", "2026"): 50, ("civile", "8", "2015"): 2, ("penale", "9", "2077"): 4}
    lost = "\n".join(missing_report(counts, held={("penale", "5", "2026")}))
    assert re.search(r"penale  2017-2020\s+3\s+10\s+2\s+1   1 / 2 rulings", lost), lost      # 6 of 10 is half; 6+3 covers 80%
    assert re.search(r"penale  all\s+4\s+19\s+3\s+2   2 / 3 rulings", lost) and "2077" not in lost and "civile  2012-2016" in lost, lost
    assert not re.search(r"n\. ?\d{3,}", text + lost)                                    # counts only, no ruling numbers
    assert kind_of(("snciv", "u0", "2025", "6", "O")) == KINDS[0] and kind_of(("snpen", "70", "2024", "1", "S")) == KINDS[2]
    assert kind_of(("snciv", "10", "2025", "5", "S")) == KINDS[0] and kind_of(("snciv", "10", "2025", "8", "O")) == KINDS[1]
    assert kind_of(("snciv", "30", "2025", "9", "D")) == KINDS[2] and kind_of(("snciv", "l0", "2025", "1", "I")) == KINDS[2]
    for r in rulings:                       # 200 judgments read on the 20th, 300 filter orders on the 19th, one not yet
        r["linked"] = None if r["code"][0] == "snciv" else ("2026-11-20T03:00:09Z" if r["code"][4] == "S" else "2026-11-19T03:00:07Z")
    nights = "\n".join(night_report(rulings))
    #                          night  rulings judgments links core links/ruling core/ruling no_link
    assert re.search(r"2026-11-19\s+300\s+0\s+150\s+0\s+0\.5\s+0\.0\s+50%", nights), nights
    assert re.search(r"2026-11-20\s+200\s+200\s+4,000\s+3,000\s+20\.0\s+15\.0\s+0%", nights), nights
    assert re.search(r"not read yet\s+1 ", nights) and "THE SAME 2 NIGHTS" in nights, nights
    assert re.search(r"judgments \+ Joint Sections\s+200\s+2,400\s+4,000\s+3,000\s+20\.0\s+15\.0\s+1\.25", nights), nights
    assert re.search(r"last group\s+300\s+1,200\s+150\s+0\s+0\.5\s+0\.0\s+0\.00", nights) and "  orders" not in nights, nights
    assert "2026-11-19" not in "\n".join(night_report(rulings, nights=1))                # the last N nights only
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    env_path = sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "/opt/v2tools/health.env"
    cited = "--cited" in sys.argv
    env = {}
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        docs = s.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id, d.file_name AS f, "
                     "d.name AS n, toString(d.cites_linked_at) AS linked, COUNT { (d)-[:CONTAINS]->() } AS pieces").data()
        articles = defaultdict(dict)                        # ruling id -> {(code, article): links}
        for r in s.run("MATCH (d:Document {document_type:'interpretation'})-[:CONTAINS]->(:Section)-[c:CITES]->() "
                       "RETURN d.id AS id, c.code AS code, c.article AS article, count(c) AS n").data():
            articles[r["id"]][(str(r["code"]), str(r["article"]))] = r["n"]
        rulings = [{"id": d["id"], "code": code_of(d["f"], d["n"]), "pieces": d["pieces"], "cited": 0,
                    "linked": d["linked"]} for d in docs]
        rulings = [r for r in rulings if r["code"]]
        routine = routine_articles(articles, {r["id"]: r["code"][0] for r in rulings})
        for r in rulings:
            mine = articles.get(r["id"], {})
            skip = {art for art, share in routine.get(r["code"][0], []) if share >= ROUTINE_SHARE}
            r["links"], r["core"] = sum(mine.values()), sum(n for art, n in mine.items() if art not in skip)
        counts, citing = Counter(), defaultdict(set)
        if cited:
            t0 = time.time()
            for start in range(0, len(rulings), BATCH):
                batch = {r["id"]: r for r in rulings[start:start + BATCH]}
                texts = defaultdict(list)
                for row in s.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
                                 "RETURN d.id AS id, s.plain_text AS t", ids=list(batch)).data():
                    texts[row["id"]].append(row["t"] or "")
                for doc, parts in texts.items():
                    code = batch[doc]["code"]
                    for number, year in precedents("\n".join(parts), own=(code[3], code[2])):
                        key = (branch_of(code[0]), number, year)
                        counts[key] += 1
                        citing[key].add(doc)
                if (start // BATCH) % 25 == 0:
                    print(f"... {min(start + BATCH, len(rulings)):,} of {len(rulings):,} rulings read ({time.time() - t0:.0f}s)", flush=True)
                time.sleep(PAUSE)
    driver.close()
    held = {(branch_of(r["code"][0]), r["code"][3], r["code"][2]): r for r in rulings}
    for key, r in held.items():
        r["cited"] = len(citing.get(key, ()))            # the rulings that cite it, not the times
    if "--by-night" in sys.argv:
        nights = int(sys.argv[sys.argv.index("--nights") + 1]) if "--nights" in sys.argv else 8
        print("\n".join(night_report(rulings, nights)))
        return 0
    print("\n".join(report(rulings, cited, routine)))
    if cited:
        print("\n" + "\n".join(missing_report({k: len(v) for k, v in citing.items()}, set(held))))
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruling_value_cited.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["branch", "year", "number", "times_cited", "citing_rulings", "held"])
            for key in sorted(counts, key=lambda k: -len(citing[k])):
                w.writerow([key[0], key[2], key[1], counts[key], len(citing[key]), int(key in held)])
        print(f"\ntable written: {path} ({len(counts):,} rulings cited; numbers only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
