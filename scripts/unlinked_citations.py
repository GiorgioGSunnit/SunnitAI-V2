"""Read-only (28 Oct): what the CITES linker leaves unlinked, sorted by kind, to size each fix before building it.

scripts/link_ruling_citations.py links an article mention only when a main code follows it ("art. 1490 c.c."); last
night it counted about 278,000 mentions with no code after them. Using the linker's own rule (exp_ruling_links'
_code_after on the 120 characters after the mention), each such mention is sorted into:
  special law            "art. 5 d.lgs. 231/2001", "art. 73 d.P.R. 9 ottobre 1990, n. 309"  -> by law (type, number, year)
  named act              "art. 2 TUIR", "art. 186 c.d.s.", "art. 41-bis ord. pen."          -> by act
  implementing rules     "art. 165 disp. att. c.p.p."
  EU / international     "art. 6 CEDU", "art. 267 TFUE", a directive or regulation
  same act, named before "art. 3 dello stesso decreto", "art. 12 cit."
  code or law named earlier in the sentence  "artt. 2043 c.c. e 2059" (the 2059)
  unknown
and the precedents the rulings cite ("Cass. Sez. U, n. 18672/2019") are counted by year. Each special law and named act
is looked up among our documents: in the database, or not (the list of laws worth acquiring).

Prints counts, law references and document names only, never ruling text (an unknown mention's 30 characters after
the number, lower-cased and with digits masked, at most 20 of them, to show what the rules miss).

    cd /opt/chatbot && venv/bin/python /root/ruling_cards/unlinked_citations.py --sample 6000   # about 2 minutes
    cd /opt/chatbot && venv/bin/python /root/ruling_cards/unlinked_citations.py                 # every ruling
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# --- classification (pure, tested) ---------------------------------------------------------------

_LEAD = re.compile(r"^\s*,?\s*(?:(?:co\.|comma|commi|lett\.|lettera|n\.|num\.|par\.|paragrafo|punto)\s*[\w\-)]+\s*,?\s*){0,4}"
                   r"(?:del(?:la|lo|le)?\s+|dell['’]\s*|degli\s+|dei\s+|in\s+)?", re.IGNORECASE)
_TYPE = (r"(d\.?\s?lgs\.?|lgs\.|d\.?\s?leg\.?vo|decreto\s+legislativo|d\.?\s?p\.?\s?r\.?|decreto\s+del\s+presidente\s+della\s+"
         r"repubblica|d\.?\s?l\.(?!\s*gs)|decreto[-\s]legge|r\.?\s?d\.?|regio\s+decreto|d\.?\s?p\.?\s?c\.?\s?m\.?|"
         r"d\.?\s?m\.?|decreto\s+ministeriale|l\.|legge)")
# "231/2001", "n. 231 del 2001", "n. 231 dell'8 giugno 2001"
_NUM_YEAR = re.compile(_TYPE + r"\s*(?:n\.?|nr\.?|num\.?)?\s*(\d{1,5})\s*(?:/|-|del\s+|dell['’]\s*|,\s*)\s*"
                       r"(?:\d{1,2}\s+\w+\s+)?((?:19|20)?\d{2})\b", re.IGNORECASE)
# "29 settembre 1973, n. 600"
_DATE_NUM = re.compile(_TYPE + r"\s*(?:\d{1,2}°?\s+\w+\s+)?((?:19|20)\d{2})\s*,?\s*(?:n\.?|nr\.?|num\.?)\s*(\d{1,5})\b",
                       re.IGNORECASE)
_NAMED = [
    (r"t\.?\s?u\.?\s?i\.?\s?r\.?(?!\w)|tuir\b", "TUIR (d.P.R. 917/1986)"),
    (r"t\.?\s?u\.?\s?e\.?\s?l\.?(?!\w)|tuel\b", "TUEL (d.lgs. 267/2000)"),
    (r"t\.?\s?u\.?\s?b\.?(?!\w)|tub\b", "TUB (d.lgs. 385/1993)"),
    (r"t\.?\s?u\.?\s?f\.?(?!\w)|tuf\b", "TUF (d.lgs. 58/1998)"),
    (r"c\.?\s?d\.?\s?s\.?(?!\w)|codice\s+della\s+strada|cod\.\s*strada|c\.\s*strada", "Codice della strada (d.lgs. 285/1992)"),
    (r"l\.\s*fall\.?|legge\s+fallimentare", "Legge fallimentare (r.d. 267/1942)"),
    (r"statuto\s+(?:dei\s+diritti\s+)?del\s+contribuente", "Statuto del contribuente (l. 212/2000)"),
    (r"cod(?:ice|\.)\s*(?:del\s+)?cons(?:umo|\.)", "Codice del consumo (d.lgs. 206/2005)"),
    (r"codice\s+(?:dei\s+contratti|degli\s+appalti)", "Codice dei contratti pubblici"),
    (r"codice\s+(?:in\s+materia\s+di\s+protezione\s+dei\s+dati|della\s+privacy)", "Codice privacy (d.lgs. 196/2003)"),
    (r"codice\s+antimafia|cod\.\s*antimafia", "Codice antimafia (d.lgs. 159/2011)"),
    (r"ord(?:inamento)?\.?\s*pen(?:it(?:enziario)?)?\.?(?!\w)", "Ordinamento penitenziario (l. 354/1975)"),
    (r"t\.?\s?u\.?\s*(?:imm|immigrazione)|testo\s+unico\s+(?:sull['’]\s*|dell['’]\s*)?immigrazione",
     "TU immigrazione (d.lgs. 286/1998)"),
    (r"t\.?\s?u\.?\s*stup|testo\s+unico\s+(?:degli\s+|sugli\s+)?stupefacenti", "TU stupefacenti (d.P.R. 309/1990)"),
    (r"codice\s+della\s+crisi", "Codice della crisi d'impresa (d.lgs. 14/2019)"),
    (r"codice\s+del\s+processo\s+tributario|proc\.\s*trib", "Processo tributario (d.lgs. 546/1992)"),
    (r"testo\s+unico|t\.\s?u\.(?!\w)", "testo unico (not named)"),
]
_NAMED_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _NAMED]
_IMPLEMENTING = re.compile(r"(?:disp(?:osizioni)?\.?\s*(?:di\s+)?att(?:uazione|\.)?|norme\s+di\s+attuazione|"
                           r"disp\.?\s*(?:prel|trans)\w*\.?)\s*(?:del(?:la|lo)?\s+|al\s+)?\s*"
                           r"(c\.?\s?p\.?\s?p\.?|c\.?\s?p\.?\s?c\.?|c\.?\s?c\.?|c\.?\s?p\.?|codice\s+\w+(?:\s+\w+)?)?", re.IGNORECASE)
_EU = re.compile(r"cedu\b|conv(?:enzione)?\.?\s+(?:europea|e\.?d\.?u\.?)|tfue\b|t\.?f\.?u\.?e\.|\btue\b|carta\s+dei\s+diritti|"
                 r"carta\s+di\s+nizza|direttiva|reg(?:olamento)?\.?\s*(?:\(?\s*(?:ue|ce|cee)\b)|protocollo\s+(?:addizionale|n\.)",
                 re.IGNORECASE)
_SAME_ACT = re.compile(r"(?:(?:dello?|della|del|nello?|nella)\s+)?(?:stess[oa]|medesim[oa]|citat[oa]|predett[oa]|suddett[oa]|"
                       r"richiamat[oa])\b|cit\.(?!\w)|sopra\s+(?:citat|richiamat)", re.IGNORECASE)
_EARLIER_CODE = re.compile(r"c\.\s?p\.\s?p\.|c\.\s?p\.\s?c\.|c\.\s?c\.|c\.\s?p\.(?!\s?[pc]\.)|cod\.\s*(?:civ|pen|proc)|"
                           r"codice\s+(?:civile|penale|di\s+procedura)|cost\.(?!\w)|costituzione", re.IGNORECASE)
_EARLIER_LAW = re.compile(_TYPE + r"\s*(?:n\.?\s*)?\d", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.;:]\s+(?=[A-ZÀ-Ý(«\"])|\n")
_PRECEDENT = re.compile(r"(?:cass\.?(?:\s*(?:civ|pen|lav|trib)\.?)?|cassazione|sez\.?\s*u(?:n(?:ite)?)?\.?|s\.\s?u\.)"
                        r"[^;\n]{0,90}?\bn\.?\s*(\d{1,6})\s*(?:/|del\s+(?:\d{1,2}\s+\w+\s+)?)\s*((?:19|20)\d{2})\b",
                        re.IGNORECASE)


def _type_of(raw: str) -> str:
    t = re.sub(r"[\s.]", "", raw.lower())
    if t.startswith(("dlgs", "lgs", "dlegvo", "decretolegislativo")):
        return "d.lgs."
    if t.startswith(("dpr", "decretodelpresidente")):
        return "d.P.R."
    if t.startswith(("dpcm",)):
        return "d.P.C.M."
    if t in ("dl",) or t.startswith("decreto-legge") or t.startswith("decretolegge"):
        return "d.l."
    if t.startswith(("rd", "regiodecreto")):
        return "r.d."
    if t.startswith(("dm", "decretoministeriale")):
        return "d.m."
    return "l."


def _year(raw: str) -> str:
    if len(raw) == 4:
        return raw
    n = int(raw)
    return str(2000 + n if n <= 30 else 1900 + n)


def law_of(rest: str) -> Optional[str]:
    """'d.lgs. 231/2001' from the start of the text after a mention (its lead stripped), or None."""
    for pattern, number_first in ((_NUM_YEAR, True), (_DATE_NUM, False)):
        m = pattern.match(rest)
        if m:
            number, year = (m.group(2), m.group(3)) if number_first else (m.group(3), m.group(2))
            return f"{_type_of(m.group(1))} {int(number)}/{_year(year)}"
    return None


def classify(tail: str, head: str) -> Tuple[str, str]:
    """(kind, detail) of an article mention with no code after it, from the 120 characters after the mention and the
    text before it."""
    rest = tail[_LEAD.match(tail).end():]
    law = law_of(rest)
    if law:
        return "special law", law
    imp = _IMPLEMENTING.match(rest)
    if imp and imp.group(0).strip():
        code = re.sub(r"[\s.]", "", (imp.group(1) or "").lower())
        return "implementing rules", {"cpp": "disp. att. c.p.p.", "cpc": "disp. att. c.p.c.", "cc": "disp. att. c.c.",
                                      "cp": "disp. att. c.p."}.get(code, "disp. att. (other)")
    for pattern, label in _NAMED_RE:
        if pattern.match(rest):
            return "named act", label
    if _EU.match(rest):
        return "EU / international", ""
    if _SAME_ACT.match(rest):
        return "same act, named before", ""
    sentence = _SENTENCE_END.split(head)[-1] if head else ""
    if _EARLIER_CODE.search(sentence):
        return "code named earlier in the sentence", ""
    if _EARLIER_LAW.search(sentence):
        return "law named earlier in the sentence", ""
    return "unknown", re.sub(r"\d", "#", " ".join(rest[:30].lower().split()))


_OWN_CASE = re.compile(r"ricors|r\.\s?g\.|iscritt|ruolo", re.IGNORECASE)


def precedents(text: str) -> List[Tuple[str, str]]:
    """(number, year) of every Cassation ruling the text cites -- not the case's own docket number ("sul ricorso
    iscritto al n. 15966/2025")."""
    return [(str(int(m.group(1))), m.group(2)) for m in _PRECEDENT.finditer(text or "")
            if not _OWN_CASE.search(m.group(0))]


# The laws whose corpus document's name does not carry their number: (type number/year) -> the name it starts with.
# Checked against our document names at run time; to review once by hand (Flavio's brief, job 3).
KNOWN_DOCUMENTS = {
    "d.P.R. 917/1986": "Testo unico delle imposte sui redditi", "d.lgs. 267/2000": "Testo unico degli enti locali",
    "d.P.R. 309/1990": "Testo Unico sugli Stupefacenti", "d.lgs. 74/2000": "Legge sui Reati Tributari",
    "d.lgs. 231/2001": "Responsabilità Amministrativa di Società ed Enti", "l. 241/1990": "Legge sul Procedimento Amministrativo",
    "d.lgs. 81/2008": "Testo Unico sulla Sicurezza del Lavoro", "d.P.R. 380/2001": "Testo unico in materia edilizia",
    "l. 898/1970": "Legge sul divorzio", "l. 431/1998": "Legge sulle locazioni abitative",
    "d.lgs. 206/2005": "Codice del Consumo", "d.lgs. 36/2023": "Codice dei contratti pubblici",
    "d.lgs. 196/2003": "Codice della Privacy", "d.lgs. 286/1998": "Testo unico sull'immigrazione",
    "d.lgs. 285/1992": "Codice della strada", "l. 212/2000": "Statuto dei Diritti del Contribuente",
    "d.lgs. 58/1998": "Testo Unico della Finanza", "d.lgs. 209/2005": "Codice delle Assicurazioni Private",
    "d.lgs. 14/2019": "Codice della crisi d'impresa", "d.lgs. 42/2004": "Codice dei beni culturali",
    "l. 354/1975": "Legge sull'ordinamento penitenziario", "d.lgs. 159/2011": "Codice antimafia",
    "d.lgs. 117/2017": "Codice del Terzo Settore", "d.lgs. 28/2010": "Mediazione civile",
    "d.lgs. 175/2016": "Testo unico delle società partecipate", "d.lgs. 151/2001": "Testo unico sulla maternità",
    "l. 247/2012": "Legge Professionale Forense", "d.lgs. 171/2005": "Codice della nautica da diporto",
    "d.lgs. 219/2006": "Codice dei medicinali", "d.lgs. 259/2003": "Codice delle Comunicazioni Elettroniche",
    "d.lgs. 82/2005": "Codice dell'amministrazione digitale", "d.lgs. 174/2016": "Codice di giustizia contabile",
    "d.lgs. 252/2005": "Testo Unico della Previdenza Complementare", "d.lgs. 198/2006": "Codice delle Pari Opportunità",
    "l. 218/1995": "Legge sul diritto internazionale privato", "d.lgs. 34/2018": "Testo Unico in materia di foreste",
    "d.l. 14/2017": "Sicurezza Urbana", "d.lgs. 297/1994": "Testo Unico delle Disposizioni Legislative Vigenti in Materia di Istruzione",
    "d.lgs. 79/2011": "Codice del Turismo", "d.lgs. 385/1993": "Testo unico bancario",
}
_NAMED_LAW = {"TUIR (d.P.R. 917/1986)": "d.P.R. 917/1986", "TUEL (d.lgs. 267/2000)": "d.lgs. 267/2000",
              "TUB (d.lgs. 385/1993)": "d.lgs. 385/1993", "TUF (d.lgs. 58/1998)": "d.lgs. 58/1998",
              "Codice della strada (d.lgs. 285/1992)": "d.lgs. 285/1992", "Legge fallimentare (r.d. 267/1942)": "r.d. 267/1942",
              "Statuto del contribuente (l. 212/2000)": "l. 212/2000", "Codice del consumo (d.lgs. 206/2005)": "d.lgs. 206/2005",
              "Codice privacy (d.lgs. 196/2003)": "d.lgs. 196/2003", "Codice antimafia (d.lgs. 159/2011)": "d.lgs. 159/2011",
              "Ordinamento penitenziario (l. 354/1975)": "l. 354/1975", "TU immigrazione (d.lgs. 286/1998)": "d.lgs. 286/1998",
              "TU stupefacenti (d.P.R. 309/1990)": "d.P.R. 309/1990", "Codice della crisi d'impresa (d.lgs. 14/2019)": "d.lgs. 14/2019",
              "Processo tributario (d.lgs. 546/1992)": "d.lgs. 546/1992"}


def find_document(law: str, docs: List[dict], resolve=None) -> Optional[str]:
    """The name of our document for 'd.lgs. 231/2001': by the table above, else by the number and year in a document's
    name or aliases (the app's resolve_code when given), else None."""
    prefix = KNOWN_DOCUMENTS.get(law)
    if prefix:
        hit = next((d for d in docs if (d.get("name") or "").lower().startswith(prefix.lower())), None)
        if hit:
            return hit["name"]
    number, year = law.split(" ", 1)[1].split("/")
    for d in docs:
        text = " ".join([d.get("name") or ""] + list(d.get("aliases") or []))
        if re.search(rf"\b{number}\b", text) and re.search(rf"\b{year}\b", text):
            return f"{d['name']} (by number and year: to check)"
    if resolve:
        try:
            hit = resolve(law, docs)
            return f"{hit['name']} (by name: to check)" if hit else None
        except Exception:
            return None
    return None


# --- the run --------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="What the CITES linker leaves unlinked, by kind (read-only)")
    p.add_argument("--sample", type=int, default=0, help="read this many random rulings (0: all)")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", default="/tmp/unlinked_citations.json")
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import exp_ruling_links as rl                      # the linker's own rule; sets up the app's path and .env
    rl.read_only()
    from src.rag.defensive_generation import _ARTICLE_RUN_RE, _RUN_NUMBER_RE
    from src.rag.issue_spotting import resolve_code
    from src.rag.main import NEO4J_DATABASE, driver

    t0 = time.time()
    kinds, laws, named, implementing, unknown = Counter(), Counter(), Counter(), Counter(), Counter()
    law_rulings, law_branch = defaultdict(set), defaultdict(Counter)
    linked = bare = 0
    years, cited_precedents, precedent_rulings = Counter(), Counter(), 0
    with driver.session(database=NEO4J_DATABASE) as session:
        rulings = session.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id, "
                              "coalesce(d.file_name, '') + ' ' + coalesce(d.name, '') AS f").data()
        if args.sample and args.sample < len(rulings):
            rulings = random.Random(28).sample(rulings, args.sample)
        branch = {r["id"]: ("penale" if "snpen" in r["f"] else "civile" if "snciv" in r["f"] else "altro") for r in rulings}
        docs = session.run("MATCH (d:Document) WHERE d.document_type IN ['primary', 'special', 'legislation'] "
                           "RETURN d.id AS id, d.name AS name, d.aliases AS aliases").data()
        ids = [r["id"] for r in rulings]
        for start in range(0, len(ids), args.batch):
            if start:
                time.sleep(args.pause)
            texts = defaultdict(list)
            for row in session.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
                                   "RETURN d.id AS doc, s.plain_text AS text", ids=ids[start:start + args.batch]).data():
                if row["text"]:
                    texts[row["doc"]].append(row["text"])
            for doc, parts in texts.items():
                text = rl._PQM_RE.sub(" ; ", " ".join(parts))
                found_precedent = False
                for number, year in precedents(text):
                    years[year] += 1
                    cited_precedents[(number, year)] += 1
                    found_precedent = True
                precedent_rulings += found_precedent
                for m in _ARTICLE_RUN_RE.finditer(text):
                    n = len(_RUN_NUMBER_RE.findall(m.group(1)))
                    tail = text[m.end():m.end() + 120]
                    if rl._code_after(tail):
                        linked += n
                        continue
                    bare += n
                    kind, detail = classify(tail, text[max(0, m.start() - 250):m.start()])
                    kinds[kind] += n
                    if kind == "special law":
                        laws[detail] += n
                        law_rulings[detail].add(doc)
                        law_branch[detail][branch[doc]] += n
                    elif kind == "named act":
                        named[detail] += n
                        key = _NAMED_LAW.get(detail)
                        if key:
                            law_rulings[key].add(doc)
                            law_branch[key][branch[doc]] += n
                    elif kind == "implementing rules":
                        implementing[detail] += n
                    elif kind == "unknown":
                        unknown[detail] += n
            if (start // args.batch) % 40 == 0:
                print(f"  {min(start + args.batch, len(ids)):,}/{len(ids):,} rulings, {time.time() - t0:.0f}s", file=sys.stderr)
    driver.close()

    def pct(n, d):
        return f"{100 * n / d:.0f}%" if d else "-"
    print(f"\nRulings read: {len(ids):,}" + (f" (a random sample of {args.sample:,})" if args.sample else "")
          + f"; article mentions: linked by the code rule {linked:,}, with no code after them {bare:,}")
    print("\nMentions with no code after them, by kind:")
    for kind, n in kinds.most_common():
        print(f"  {n:>9,}  {pct(n, bare):>4}  {kind}")

    merged = Counter(laws)
    for label, n in named.items():
        if label in _NAMED_LAW:
            merged[_NAMED_LAW[label]] += n
    in_db = {law: find_document(law, docs, resolve_code) for law, _ in merged.most_common(args.top * 3)}
    covered = sum(n for law, n in merged.items() if in_db.get(law))
    print(f"\nSpecial laws and named acts: {sum(merged.values()):,} mentions of {len(merged):,} laws; "
          f"in our database: {covered:,} ({pct(covered, sum(merged.values()))} of the mentions, top {args.top * 3} laws checked)")
    print(f"{'mentions':>9}  {'rulings':>7}  {'penale':>7}  {'civile':>7}  law -> our document")
    for law, n in merged.most_common(args.top):
        b = law_branch[law]
        print(f"{n:>9,}  {len(law_rulings[law]):>7,}  {b['penale']:>7,}  {b['civile']:>7,}  {law} -> {in_db.get(law) or 'NOT IN THE DATABASE'}")
    missing = [(law, n) for law, n in merged.most_common(args.top * 3) if not in_db.get(law)]
    print("\nMost cited laws NOT in the database (worth acquiring): "
          + "; ".join(f"{law} ({n:,})" for law, n in missing[:15]))
    if implementing:
        print("\nImplementing rules: " + ", ".join(f"{k} {n:,}" for k, n in implementing.most_common()))
    if named.get("testo unico (not named)"):
        print(f"'Testo unico' with no name: {named['testo unico (not named)']:,}")
    print(f"\nPrecedents cited: {sum(cited_precedents.values()):,} citations of {len(cited_precedents):,} rulings, in "
          f"{precedent_rulings:,} of the rulings read ({pct(precedent_rulings, len(ids))})")
    print("  by year: " + ", ".join(f"{y} {n:,}" for y, n in sorted(years.items(), key=lambda x: -int(x[0]))[:20]))
    decades = Counter(f"{y[:3]}0s" for y in years.elements())
    print("  by decade: " + ", ".join(f"{d} {n:,}" for d, n in sorted(decades.items(), reverse=True)))
    print("  most cited: " + ", ".join(f"n. {num}/{y} ({n})" for (num, y), n in cited_precedents.most_common(15)))
    print("\nUnknown, the most frequent next words (digits as #): "
          + "; ".join(f"'{k}' {n:,}" for k, n in unknown.most_common(20)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rulings": len(ids), "sample": args.sample, "linked": linked, "bare": bare, "kinds": kinds,
                   "laws": {law: {"mentions": n, "rulings": len(law_rulings[law]), "branch": law_branch[law],
                                  "document": in_db.get(law)} for law, n in merged.most_common(300)},
                   "implementing": implementing, "precedent_years": years,
                   "precedents_top": [[f"{num}/{y}", n] for (num, y), n in cited_precedents.most_common(100)]},
                  fh, ensure_ascii=False, indent=1)
    print(f"\nreport: {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
