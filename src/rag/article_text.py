"""One article's text, read the same way by every path: the chat's cross-references
(nodes/retrieval.py), issue spotting, the strategy's code lookups (defensive_generation.
_fetch_articles) and the strategy's rows from searches (fill_texts).

The corpus stores an article several ways (scripts/article_sections.py on prod, 1 Nov). The Sections of type "Articolo"
hold the whole article in order, lists included, cut every ~4000 characters: "61"+"61_1", "582_3"+"582_3_1", "177_9". The
others are partial copies: "N.0.0" a list's opening line or only its date in force ("In vigore dal 11 dicembre 2021"),
"N.k.0" its commi or list items, a bare "N" of another type (or "N_j") one item. The rule, for article N (choose_text):
  1. the "Articolo" chunks that open like the article (N.0.0's first words, else its first comma), joined in suffix
     order (chunks are not on NEXT) with any overlap at a join removed, anything before the opening dropped ("casi
     particolari (1) (6) Chiunque...": the end of a rubrica);
  2. with none: N.0.0, then the bare N, N_j and N.k.0 pieces in NEXT order (else in name order), a piece another holds
     left out. N.k, N.k_j, N.k.2019 are notes and duplicates: never read;
  3. PDF page numbers and editorial notes removed, these as blocks (strip_page_numbers, strip_notes); markers alone left: "(abrogato)" if repealed, else the text
     as it stood.
The row returned keeps the canonical Section ("N.0.0", else "N") with that text in plain_text. An article with neither
keeps its dot fragments ("N.1", "N.2"), the first 3.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IN_FORCE_RE = re.compile(r"\s*(?:in vigore dal\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*)+", re.IGNORECASE)
CHUNK_MIN = 3900          # the BE cuts at 4000: a shorter Section is not followed by a chunk of its own
OPENING_WORDS = 6
OPENING_WITHIN = 300
_NUMBER_RE = re.compile(r"^(\d+(?:-[a-z]+)?)(?:\.|$)", re.IGNORECASE)
_CHUNK_RE = re.compile(r"^(.*)_(\d+)$")

# How an editorial note opens: "(3) Comma così modificato", "(1) Articolo inserito", "(2) Le parole: ...", "(1) La Corte
# costituzionale". A bare "(2)" proves nothing: art. 589 has "...psicotrope.] (2) Nel caso di morte", real text.
_NOTE_HEAD = (r"(?:(?:Comma|Commi|Articolo|Articoli|Numero|Numeri|Lettera|Lettere|Rubrica|Periodo|Capo|Sezione|Titolo|Libro|"
              r"Alinea|Paragrafo|Intestazione|Denominazione|La\s+punteggiatura)\s+(?:[a-zà-ü]+\s+){0,3}?(?:modificat|aggiunt|inserit|abrogat|sostituit|soppress|rinumerat|"
              r"introdott|dichiarat|corrett)|[Ll][ae]\s+parol[ae]\s*[:«\"“]|Le\s+originarie\s+parole|La\s+Corte\s+[Cc]ost|"
              r"I\s+(?:Capi|Titoli|Commi|Numeri|Articoli)\b[^.]{0,60}?stat[ie]\b|"
              r"Il\s+previgente\s+(?:Capo|Titolo|Libro|Sezione|comma|articolo|testo)|"
              r"(?:(?:Successivamente|Infine),\s+)?(?:[Ii]l\s+presente|[Qq]uesto|[Tt]ale|[Ll]a\s+presente|[Qq]uesta)\s+"
              r"(?:comma|articolo|lettera|numero|rubrica|capo|periodo|sezione|titolo)\s+[^.]{0,40}?"
              r"(?:stat[oa]|modificat|aggiunt|inserit|abrogat|sostituit)|L['’]articolo\s+che\s+recitava|Il\s+comma\s+che\s+recitava|"
              r"A\s+norma\s+dell|Vedi\b)")
# After a "(n)", any subject of an amendment: "(2) La misura è stata così modificata dall'art.".
_NOTE_ACT = r"[A-ZÀ-Ü][^.\n]{0,120}?(?:(?:è|sono)\s+stat[oaie]\s+)?così\s+\w+\s+dal|[A-ZÀ-Ü][^.\n]{0,120}?(?:è|sono)\s+stat[oaie]\s+\w+\s+dal"
# Only after a "(n)": comments on the article rather than amendments (card review 2: 582 c.p., 640 c.p.).
_NOTE_COMMENT = (r"Al\s+reato\s+previsto|La\s+condanna\s+per\s+il\s+delitto|Nel\s+presente\s+articolo|"
                 r"Per\s+l['’]aumento\s+della\s+pena|[A-ZÀ-Ü][^.\n]{0,40}?\s+risulta\s+così|Il\s+precedente\s+comma")
_NOTE_RE = re.compile(rf"(?:(?<=\s)|^)\((\d{{1,3}})\)\s+(?={_NOTE_HEAD}|{_NOTE_ACT}|{_NOTE_COMMENT})|^(?={_NOTE_HEAD})", re.MULTILINE)
# Any "(n) Sentence": a note only if its marker was already used and nothing real follows (_next_note).
_MARKED_RE = re.compile(r"(?:(?<=\s)|^)\((\d{1,3})\)\s+(?=[A-ZÀ-Ü])", re.MULTILINE)
_MARKER_RE = re.compile(r"\((\d{1,3})\)")

# A PDF page number left in the text ("determina i 37 tempi": page 37 fell inside the sentence). A number in a law is
# far likelier to be real, so one goes only with neither side able to take it: never after a word that governs a number
# ("comma 2", "L. 69") nor before a unit, a month or a function word ("30 giorni", "5 ottobre", "12 del"). Some page
# numbers stay, by design: article_text_compare.py lists every one removed. Measured on the PDFs: tests.
_NUMBER_BEFORE = frozenset(
    "art artt articolo articoli comma commi n nn numero numeri lett lettera lettere l legge leggi d dl lgs d.lgs d.l d.p.r "
    "d.m r.d r.d.l dpr decreto legislativo sentenza uff g.u euro lire entro oltre almeno fino sino per di da a al ai dal dai "
    "del dei degli dagli sui su tra fra e o ed od meno più circa pari il la lo l' le gli un una ogni nel nei allegato "
    "tabella pag pagina punto paragrafo capo titolo libro sez sezione cap parte anni anno mesi mese giorni giorno ore".split())
_NUMBER_AFTER = frozenset(
    "giorni giorno mesi mese anni anno ore ora minuti settimane settimana gennaio febbraio marzo aprile maggio giugno luglio "
    "agosto settembre ottobre novembre dicembre euro lire per percento volte bis ter quater quinquies sexies septies octies "
    "novies decies e o ed od a ad al alla allo ai alle del della dello dei delle degli di da dal dalla dai dalle con che non "
    "è sono ha hanno si può possono il la lo i le gli un una uno comma commi articolo articoli art lettera lettere ovvero "
    "oppure ma se in nel nella su sul sulla persone membri metri chilometri ettari grammi litri unità dipendenti lavoratori "
    "soci componenti giudici testimoni parti figli anche quando come c.p c.p.p c.c c.p.c".split())
_PAGE_CANDIDATE_RE = re.compile(r"(?<=\s)\d{1,3}(?=\s)")
_PAGE_LINE_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]*$")


def _is_page_number(text: str, match: re.Match) -> bool:
    before, after = text[max(0, match.start() - 60):match.start()].split()[-1:], text[match.end():match.end() + 60].split()[:1]
    if not before or not after:
        return False
    word, following = before[0], after[0].strip(",.;:)»”").lower()
    bare = re.split(r"['’]", word.strip("(«“,.;:").lower())[-1]
    if bare in _NUMBER_BEFORE or re.search(r"\d", word) or re.search(r"\d", following):
        return False
    if word[-1] in ".;:" and after[0][0].isupper():            # between two sentences; under 10, a comma without its dot
        return len(bare) > 3 and int(match.group()) >= 10
    return (word[-1].isalpha() and after[0][0].islower() and len(following) > 2 and following.isalpha()
            and following not in _NUMBER_AFTER)


def page_numbers(text: Optional[str]) -> List[Tuple[int, int]]:
    """(start, end) of each page number in the text: alone on a line (a column of numbers is a table), or inside it."""
    text = text or ""
    lines = list(_PAGE_LINE_RE.finditer(text))
    found = [(m.start(), m.end()) for i, m in enumerate(lines)
             if not (i and lines[i - 1].end() + 1 == m.start()) and not (i + 1 < len(lines) and m.end() + 1 == lines[i + 1].start())]
    found += [(m.start(), m.end()) for m in _PAGE_CANDIDATE_RE.finditer(text) if _is_page_number(text, m)
              and not any(a <= m.start() < b for a, b in found)]
    return sorted(found)


def strip_page_numbers(text: Optional[str]) -> str:
    text = text or ""
    for start, end in reversed(page_numbers(text)):
        tail = text[end:].lstrip(" \t") if text[start - 1:start] in (" ", "\t", "\n") else text[end:]
        text = text[:start] + tail
    return re.sub(r"\n{3,}", "\n\n", text)
_ABROGATED_NOTE_RE = re.compile(r"[^.]{0,90}?\babrogat", re.IGNORECASE)
# Nothing but markers: "(1)", "(…) (1) (1)". What is left of an article whose whole body was a note.
_MARKERS_ONLY_RE = re.compile(r"^(?:\(\d{1,3}\)|\((?:…|\.\.\.)\)|[\s\[\].,;:])*$")
_COMMA_NUMBER_RE = re.compile(r"(?:^|(?<=\s))(\d{1,2})\.\s+[A-ZÀ-Ü]")


def in_force_only(text: Optional[str]) -> bool:
    match = IN_FORCE_RE.match(text or "")
    return bool(match) and match.end() == len(text or "")


def clause_cut(text: str, chars: int) -> str:
    """text within chars, ended only where a clause ends and marked "[…]": a quote that stops on "... quando consta
    mediante" says something else, and the writer repeats it. No clause end in reach: on to the next one, past the budget;
    a text with no punctuation left, whole up to twice the budget, else to its last whole word."""
    if len(text) <= chars:
        return text
    head = text[:chars]
    if " " not in head:
        return head
    stops = [m.end() - 1 for m in re.finditer(r"[.;:](?=\s)", head)]
    pauses = [m.start() for m in re.finditer(r",(?=\s)", head)]
    for found in (stops, pauses):
        if found and found[-1] >= chars // 3:
            return f"{text[:found[-1] + (found is stops)].rstrip()} […]"
    ahead = re.search(r"[.;:,](?=\s)", text[chars:])
    if ahead:
        return f"{text[:chars + ahead.end()].rstrip(' ,')} […]"
    if len(text) <= 2 * chars:
        return text
    return head.rsplit(" ", 1)[0].rstrip() + " […]"


def heading(section: Optional[Dict[str, Any]]) -> str:
    """The article's rubrica: titolo_articolo, else title. A title is generated by a model within 50 tokens: asked to name
    the institute, it may copy the text instead, and the limit cuts it on a word ("La destinazione del padre di famiglia ha
    luogo quando consta mediante"). One that only repeats the article's own words is no rubrica: it goes to the writer as
    one, and the writer writes it into the answer."""
    section = section or {}
    rubrica = " ".join(str(section.get("titolo_articolo") or "").split())
    if rubrica:
        return rubrica
    title = " ".join(str(section.get("title") or "").split())
    return "" if _copies_text(title, section.get("plain_text")) else title


def _copies_text(title: str, text: Any) -> bool:
    """The title repeats the text, punctuation aside: five words or more from its very start, or seven in a row anywhere. A
    real title may well be a phrase of its article ("Destinazione del padre di famiglia")."""
    found = re.findall(r"\w+", title.lower())
    words = re.findall(r"\w+", str(text or "").lower())
    if len(found) >= 5 and words[:len(found)] == found:
        return True
    return len(found) >= 7 and f" {' '.join(found)} " in f" {' '.join(words)} "


def strip_in_force(text: Optional[str]) -> str:
    return IN_FORCE_RE.sub("", text or "", count=1).strip() if IN_FORCE_RE.match(text or "") else (text or "").strip()


def section_quality(section: Optional[Dict[str, Any]]) -> tuple:
    """Sort key among exact duplicates sharing a name: titolo_articolo set (only on a document's current,
    re-ingested copy) beats plain_text length alone (round 3, 18 Oct: length alone won art. 628 for a stale
    duplicate)."""
    return (1 if ((section or {}).get("titolo_articolo") or "").strip() else 0,
            len((section or {}).get("plain_text") or ""))


def article_number(section_name: str) -> Optional[str]:
    """"135-octiesdecies.0.0" -> "135-octiesdecies"; None for names that are not an article's."""
    m = _NUMBER_RE.match((section_name or "").strip())
    return m.group(1) if m else None


def _words_re(text: str, words: int = OPENING_WORDS) -> Optional[re.Pattern]:
    """Text's first words as a pattern, across any spacing or punctuation; None under 3 words."""
    found = re.findall(r"\w+", text or "")[:words]
    return re.compile(r"\W+".join(map(re.escape, found)), re.IGNORECASE) if len(found) >= 3 else None


def _quoted(text: str, start: int, at: int) -> bool:
    """Is `at` inside a quotation opened after `start`? A note can quote the old wording, numbered commi included."""
    span = text[start:at]
    return span.count("«") > span.count("»") or span.count("“") > span.count("”")


def _note_end(text: str, start: int, kept: str, anchors: Sequence[str]) -> int:
    """Where the note block opened at `start` ends: at the next real comma (a piece's opening, or the number after the
    last one read), never inside a quotation; else the end of the text. Notes in a row are one block."""
    last = 0
    for match in _COMMA_NUMBER_RE.finditer(kept):
        last = int(match.group(1))
    found = []
    for anchor in anchors:
        pattern = _words_re(anchor)
        for match in (pattern.finditer(text, start + 1) if pattern else ()):
            if not _quoted(text, start, match.start()):
                found.append(match.start())
                break
    if last:
        for match in re.finditer(rf"(?<=[.;\]»”)])\s+({last + 1})\.\s+[A-ZÀ-Ü]", text[start:]):
            if not _quoted(text, start, start + match.start(1)):
                found.append(start + match.start(1))
                break
    return min(found, default=len(text))


def _next_note(text: str, pos: int, anchors: Sequence[str], kept: str, rubrica: str) -> Optional[re.Match]:
    """The next note from `pos`: one that opens like a note, or before it a "(n) Sentence" that starts the tail of the
    text: its marker already used (in the text or the rubrica, where 582 c.p. has its (1)) and no real comma after it.
    "...psicotrope.] (2) Nel caso di morte", art. 589 c.p., is that marker's first use: real text."""
    note = _NOTE_RE.search(text, pos)
    used = set(_MARKER_RE.findall(rubrica or ""))
    for match in _MARKED_RE.finditer(text, pos, note.start() if note else len(text)):
        if (match.group(1) in used or f"({match.group(1)})" in text[:match.start()]) \
                and _note_end(text, match.start(), kept + text[pos:match.start()], anchors) == len(text):
            return match
    return note


def strip_notes(text: Optional[str], anchors: Sequence[str] = (), rubrica: str = "") -> str:
    """The text without its editorial notes, removed as blocks: what follows a note survives when it is known to be real
    (`anchors`, the openings of the article's commi; the next comma number). A note saying a comma or number was repealed
    turns its inline marker into "(abrogato)"; a leading "(n)" goes with its note; a text left with markers alone becomes
    "(abrogato)" when a note repealed it. Never empty when the text was not: otherwise it stays as it is."""
    return _clean(text, anchors, rubrica) or (text or "").strip()


def _clean(text: Optional[str], anchors: Sequence[str] = (), rubrica: str = "") -> str:
    """strip_notes and strip_page_numbers, "" when nothing of the article is left and no note repealed it."""
    text = strip_page_numbers(text)
    out: List[str] = []
    pos = 0
    abrogated: List[str] = []
    removed = False
    while True:
        match = _next_note(text, pos, anchors, "".join(out), rubrica)
        if not match:
            out.append(text[pos:])
            break
        out.append(text[pos:match.start()])
        removed = True
        end = _note_end(text, match.start(), "".join(out), anchors)
        for note in _NOTE_RE.finditer(text[:end], match.start()):
            if _ABROGATED_NOTE_RE.match(text, note.end()):
                abrogated.append(note.group(1) or "")
        if re.search(r"\bstat[oa]\s+abrogat", text[match.start():end], re.IGNORECASE):   # "L'articolo che recitava: ..."
            abrogated.append("")
        pos = max(end, match.end() + 1)
    kept = "".join(out)
    for number in abrogated:
        if number:
            kept = re.sub(rf"\({number}\)(?!.*\({number}\))", "(abrogato)", kept, count=1, flags=re.DOTALL)
    kept = re.sub(r"[ \t]+\n", "\n", kept).strip()
    if _MARKERS_ONLY_RE.match(kept):
        return "(abrogato)" if abrogated else ""
    return re.sub(r"^(?:\(\d{1,3}\)\s*)+", "", kept) if removed else kept


def _name_key(name: str) -> list:
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"[._]", name)]


def _join(first: str, second: str) -> str:
    """Two chunks, an overlap at the join kept once."""
    for size in range(min(len(first), len(second), 300), 11, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return f"{first} {second}"


def _chunk_groups(number: str, texts: Dict[str, str], kinds: Dict[str, str]) -> List[str]:
    """The article's "Articolo" records, each its chunks joined: "61"+"61_1" one record, "177_9" another. "N_1" follows
    "N" and "N_j_2" follows "N_j_1" only when the one before is full-sized: "N_21" and "N_22" are two records."""
    names = sorted((n for n in texts if kinds.get(n) == "Articolo" and (texts[n] or "").strip()
                    and (n == number or n.startswith((number + "_", number + ".")))), key=_name_key)
    groups: Dict[str, List[str]] = {}
    head = {}
    for name in names:
        match = _CHUNK_RE.match(name)
        before = (match.group(1) if int(match.group(2)) == 1 else f"{match.group(1)}_{int(match.group(2)) - 1}") if match else ""
        if before in head and len(texts[before].strip()) >= CHUNK_MIN:
            head[name] = head[before]
            groups[head[name]].append(name)
        else:
            head[name] = name
            groups[name] = [name]
    out = []
    for chunks in groups.values():
        whole = ""
        for name in chunks:
            part = strip_in_force(texts[name])
            whole = _join(whole, part) if whole else part
        if whole and not in_force_only(whole):
            out.append(whole)
    return out


def _from_opening(group: str, pattern: Optional[re.Pattern]) -> Optional[str]:
    """The record from the article's opening on (a comma's number kept); None when it does not open like the article."""
    match = pattern.search(group[:OPENING_WITHIN]) if pattern else None
    if not match:
        return None
    before = group[:match.start()]
    return group if not re.search(r"[^\W\d]", before) else group[match.start():]


def _reading_order(names: Sequence[str], following: Dict[str, str]) -> List[str]:
    """The names along NEXT, where it links them all; else N and N_j, then N.k.0, in name order."""
    ordered = sorted(names, key=lambda n: ("." in n, _name_key(n)))
    wanted = set(ordered)
    def next_of(name):
        seen = set()
        name = following.get(name)
        while name and name not in wanted and name not in seen:
            seen.add(name)
            name = following.get(name)
        return name
    after = {n: next_of(n) for n in ordered}
    starts = wanted - set(after.values())
    if len(starts) != 1:
        return ordered
    chain = [starts.pop()]
    while after.get(chain[-1]) and after[chain[-1]] not in chain:
        chain.append(after[chain[-1]])
    return chain if len(chain) == len(ordered) else ordered


def _pieces(number: str, texts: Dict[str, str], following: Dict[str, str]) -> List[str]:
    """The article's partial copies in reading order: N.0.0, then N, N_j, N.k.0 (and their chunks)."""
    shape = re.compile(rf"^{re.escape(number)}(?:_\d+|\.[1-9]\d*\.0)?(?:_\d+)*$")
    names = [n for n in texts if shape.match(n) and (texts[n] or "").strip() and not in_force_only(texts[n])]
    ordered = _reading_order(names, following)
    zero = f"{number}.0.0"
    return ([zero] if (texts.get(zero) or "").strip() and not in_force_only(texts[zero]) else []) + ordered


def choose_text(number: str, texts: Dict[str, str], kinds: Optional[Dict[str, str]] = None,
                following: Optional[Dict[str, str]] = None, rubrica: str = "") -> str:
    """The article's whole text from its Sections' {name: plain_text}, {name: type}, {name: the name NEXT leads to} and
    rubrica, by the rule in the module's docstring; "" when none holds more than a date."""
    kinds, following = kinds or {}, following or {}
    pieces = _pieces(number, texts, following)
    anchors = [strip_in_force(texts[n]) for n in pieces if n != f"{number}.0.0" and n != number and "_" not in n]
    zero = strip_notes(strip_in_force(texts.get(f"{number}.0.0")), (), rubrica)
    groups = _chunk_groups(number, texts, kinds)
    has_zero = bool(zero) and not in_force_only(zero)
    opening = _words_re(zero if has_zero else (anchors[0] if anchors else ""))
    matched = [t if has_zero else g for g, t in ((g, _from_opening(g, opening)) for g in groups) if t]  # a comma may be a
    # list item: only N.0.0 proves what comes before it is not the article
    whole = _clean(max(matched, key=len), anchors, rubrica) if matched else ""
    if not whole and not matched and groups and (len(zero) < 60 or not opening):
        whole = _clean(max(groups, key=len), anchors, rubrica)
    if whole:
        return whole
    parts = [_clean(strip_in_force(texts[n]), anchors, rubrica) for n in pieces]
    flat = [" ".join(p.split()) for p in parts]
    kept = []
    for i, part in enumerate(parts):
        if part and flat[i] not in flat[:i] and not any(flat[i] in other and len(other) > len(flat[i]) for other in flat):
            kept.append(part)
    return "\n".join(kept) or _nothing_left(number, texts)


def _nothing_left(number: str, texts: Dict[str, str]) -> str:
    """No text once the notes are out: "(abrogato)" when a Section of the article says so, else what the reader gave
    before notes were removed (N.0.0, else N, else the first comma)."""
    if any(re.search(r"\babrogat", t or "", re.IGNORECASE) for t in texts.values()):
        return "(abrogato)"
    shape = re.compile(rf"^{re.escape(number)}\.[1-9]\d*\.0$")
    for name in [f"{number}.0.0", number] + sorted((n for n in texts if shape.match(n)), key=_name_key):
        text = strip_in_force(texts.get(name))
        if text and not in_force_only(text):
            return text
    return ""


def _best_by_name(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = (row.get("s") or {}).get("name") or ""
        if name and (name not in best or section_quality(row.get("s")) > section_quality(best[name].get("s"))):
            best[name] = row
    return best


def _fields(best: Dict[str, Dict[str, Any]]) -> tuple:
    """choose_text's arguments from {name: row}."""
    sections = {name: row.get("s") or {} for name, row in best.items()}
    return ({n: s.get("plain_text") or "" for n, s in sections.items()},
            {n: s.get("type") or "" for n, s in sections.items()},
            {n: row["next"] for n, row in best.items() if row.get("next")},
            next((s.get("titolo_articolo") for s in sections.values() if (s.get("titolo_articolo") or "").strip()), ""))


def article_rows(number: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """From one document's {"d", "s"} rows for article N: the canonical row carrying the whole text, else its first 3
    dot fragments in name order, else []."""
    best = _best_by_name(rows)
    text = choose_text(number, *_fields(best))
    names = sorted(best, key=_name_key)
    canonical = best.get(f"{number}.0.0") or best.get(number) or (best[names[0]] if text and names else None)
    if canonical:
        section = dict(canonical.get("s") or {})
        if text:
            section["plain_text"] = text
        return [{**canonical, "s": section}]
    fragments = sorted((name for name in best if name.startswith(number + ".")))
    return [best[name] for name in fragments[:3]]


_SECTIONS_QUERY = (
    "UNWIND $items AS it "
    "MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
    "WHERE (s.name = it.num OR s.name STARTS WITH it.num + '.' OR s.name STARTS WITH it.num + '_') {visibility}"
    "OPTIONAL MATCH (s)-[:NEXT]->(after:Section) "
    "RETURN it.doc AS doc, it.num AS num, d, s, after.name AS next")


def _sections(session, items: List[Tuple[str, str]], visibility: str = "", **params) -> Dict[tuple, List[Dict]]:
    """{(doc_id, number): [{"d", "s"}...]} for every item, in one query."""
    if not items:
        return {}
    query = _SECTIONS_QUERY.replace("{visibility}", f"AND {visibility} " if visibility else "")
    found = session.run(query, items=[{"doc": doc, "num": num} for doc, num in items], **params).data()
    grouped: Dict[tuple, List[Dict]] = {}
    for row in found:
        grouped.setdefault((row["doc"], row["num"]), []).append({"d": row["d"], "s": row["s"], "next": row.get("next")})
    return grouped


def fetch_article(session, doc_id: str, number: str, visibility: str = "", **params) -> List[Dict[str, Any]]:
    """Article N of one document: [the canonical row with its text] (see article_rows), never another article
    that merely starts with the same digits. Any spelling of the suffix: the canonical one first ("21-novies"
    finds "21-nonies.0.0"), then the legacy spellings of Sections ingested before the BE normalised them."""
    from .article_suffix import number_variants

    for spelling in number_variants(number):
        rows = article_rows(spelling, _sections(session, [(doc_id, spelling)], visibility, **params)
                            .get((doc_id, spelling), []))
        if rows:
            return rows
    return []


def fill_texts(session, rows: List[Dict[str, Any]]) -> List[str]:
    """In place, for rows other paths found (searches, votes, tables): a row of an article's canonical Section gets the
    article's whole text by the same rule. Returns the sections changed, "doc_name name". One query for all of them."""
    stubs = []
    for row in rows:
        s, d = row.get("s") or {}, row.get("d") or {}
        number = article_number(s.get("name") or "")
        if number and d.get("id") and s.get("name") in (number, f"{number}.0.0"):
            stubs.append((row, d["id"], number))
    if not stubs:
        return []
    grouped = _sections(session, sorted({(doc, num) for _, doc, num in stubs}))
    filled = []
    for row, doc, num in stubs:
        text = choose_text(num, *_fields(_best_by_name(grouped.get((doc, num), []))))
        if text and text != row["s"].get("plain_text"):
            row["s"] = {**row["s"], "plain_text": text}
            filled.append(f"{row['d'].get('name', '')} {row['s']['name']}")
    return filled


def fill_from_rows(number: str, rows: List[Dict[str, Any]]) -> List[str]:
    """In place, without a query, for a path that already read article N's Sections (the chat's article_router): a
    canonical row gets the whole text the same rule chooses from the rows of its own document. Returns the sections
    changed."""
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault((row.get("d") or {}).get("id"), []).append(row)
    changed = []
    for doc_rows in by_doc.values():
        text = choose_text(number, *_fields(_best_by_name(doc_rows)))
        for row in doc_rows:
            s = row.get("s") or {}
            if s.get("name") in (number, f"{number}.0.0") and text and text != s.get("plain_text"):
                row["s"] = {**s, "plain_text": text}
                changed.append(f"{(row.get('d') or {}).get('name', '')} {s.get('name')}")
    return changed


def canonical_only(number: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per document, its canonical rows once filled: the whole article, not its partial copies beside it. A document
    without one keeps its rows."""
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault((row.get("d") or {}).get("id"), []).append(row)
    out: List[Dict[str, Any]] = []
    for doc_rows in by_doc.values():
        zero = [r for r in doc_rows if (r.get("s") or {}).get("name") == f"{number}.0.0"]
        bare = [r for r in doc_rows if (r.get("s") or {}).get("name") == number]
        out += (zero or bare)[:1] or doc_rows
    return out
