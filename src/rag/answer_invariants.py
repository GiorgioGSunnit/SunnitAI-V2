"""ANSWER_INVARIANTS_ENABLED (default off): one general layer of checks on every strategy answer, after its
sections are written (planned or one-call), in place of fixing one defect case by case. Each invariant is logged
with what it found, what it fixed and what remains (target: 0 remaining), as case_analysis_invariants.

  I1 citation closure: every article cited anywhere (the conclusions too) belongs to the checked material -- graded
     2-3 (or, without the grader, the checked pool), named by the user, an offered procedural tool, or the article
     of a computed term; otherwise its whole sentence goes (a heading naming it with its paragraph).
  I2 claim <-> article: every citation in every section is compared with the article's rubrica and opening text
     (the qualification check's comparison, per citation, not per article); a mismatch removes the sentence, unless
     the article is graded 3 or named by the user: then the clause holding it goes when the rest stands alone,
     else the citation is marked "(da verificare: ...)".
  I3 institution -> article within its chapter (CHAPTERS): an institution of the table resolves only inside its
     ranges, the model's own number too; refused outside, the article inside whose rubrica shares most words with
     it is taken (pick_in_chapter). defensive_generation._resolve_institution and the fact rules' institutions.
  I4 ruling relevance: a cited ruling shares a substantive (non-boilerplate) article with the answer through its
     CITES edges, has a substantive holding (not only procedural, unless the case asks a procedural question), and,
     when adverse to the client (RULING_FIT's _fit_adverse), is cited only inside an objection or a reply;
     otherwise the sentences citing it go.
  I5 no internal tokens: status labels in capitals (DECORSO, INCERTO, DA VERIFICARE ...), raw ids (file names,
     document ids, rule ids, [S1]), placeholder brackets, mangled fragments ("1158.0.0", doubled words, stray "**")
     repaired or removed; the terms reach the writer in plain Italian (plain_status).
  I6 readability: one short call per section rewrites only the sentences it finds ungrammatical or incoherent; a
     rewrite that changes any citation number is refused.
  I7 terms: a deadline or duration the answer states comes from the computed term table (or the case's own text,
     or the text of an article the same sentence cites); any other sentence stating one goes.
The guards already in place are counted on the final answer too: explicit questions unanswered, "does not apply"
sentences, orphan fragments (removed: plan_then_write.drop_dangling).
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags

logger = logging.getLogger(__name__)

FLAG = "ANSWER_INVARIANTS_ENABLED"


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- I3: the chapter of each main institution --------------------------------------------------------------------
# (pattern on the institution's name, code, ranges, unless: a pattern that takes the name out of the entry). First
# match wins: the specific before the general (usucapione before servitù, passaggio coattivo before servitù).
_CRIMES = (r"furt|rapin|omicid|lesion|truff|estorsion|ricettazion|stupefacent|violenz|minacc|diffamazion|corruzion|"
           r"peculat|concussion|fals|evasion|resistenza|danneggiament|appropriazion|usur|bancarott|maltrattament|"
           r"persecutori|sequestr|riciclagg|incendi|frode")
CHAPTERS: List[Tuple[str, str, List[List[int]], str]] = [
    (r"usucapi", "c.c.", [[1158, 1167], [1061, 1061]], ""),
    (r"passaggio coattiv|servit\w* coattiv|indennit\w* per il passaggio", "c.c.", [[1051, 1055]], ""),
    (r"negatori|confessori|rivendica|accertamento della servit|regolamento di confini|apposizione di termini", "c.c.",
     [[948, 951], [1079, 1079]], ""),
    (r"servit|destinazione del padre di famiglia", "c.c.", [[1027, 1099]], ""),
    (r"possess|reintegrazion|spoglio|azione di manutenzione|turbativ", "c.c.", [[1140, 1172]], ""),
    (r"extracontrattual|fatto illecito|aquilian|cose in custodia|danno cagionato da animali|rovina di edific|"
     r"circolazione di veicoli|attivit\w* pericolos", "c.c.", [[2043, 2059]], ""),
    (r"appalt|difformit\w* (?:e|dell)", "c.c.", [[1655, 1677]], ""),
    (r"vizi (?:della cosa|del bene|occulti|redibitori)|garanzia per (?:i )?vizi|cosa vendut|azione redibitoria|"
     r"quanti minoris|aliud pro alio", "c.c.", [[1490, 1497]], ""),
    (r"successio|eredit|\berede\b|legittim|testament|\blegat[oi]\b|collazion|azione di riduzione", "c.c.",
     [[456, 768]], ""),
    (r"comunione (?:legale|de residuo|dei beni)|beni personali del coniuge", "c.c.", [[177, 197]], ""),
    (r"clausola penale|riduzione della penale|caparra", "c.c.", [[1382, 1386]], ""),
    (r"inadempiment|risoluzione (?:del contratto|per inadempimento)|mora del debitore|eccessiva onerosit|"
     r"impossibilit\w* sopravvenut", "c.c.", [[1218, 1229], [1453, 1469]], ""),
    (r"prescrizion|decadenz", "c.c.", [[2934, 2969]], r"servit|usucap|reato"),
    (r"condomin|parti comuni", "c.c.", [[1117, 1139]], ""),
    (r"locazion|conduttor|locator", "c.c.", [[1571, 1654]], ""),
    (r"immission", "c.c.", [[844, 844]], ""),
    (r"distanz\w* (?:nelle|tra le) costruzion|vedut|luci e vedute", "c.c.", [[873, 907]], ""),
    (r"tentativ|delitto tentato|concorso di persone|concorso (?:formale|materiale)|reato continuato|"
     r"legittima difesa|stato di necessit|\bdolo\b|\bcolpa\b|elemento psicologico|recidiv|circostanz|attenuant|"
     r"aggravant|imputabilit|cause? di giustificazione|desistenza|recesso attivo|rapporto di causalit|"
     r"nesso (?:di )?causal|estinzione del reato|prescrizione del reato|sospensione condizionale|non punibilit|"
     r"particolare tenuit", "c.p.", [[39, 240]], _CRIMES),
]


def chapter(name: str, code: str) -> Optional[List[List[int]]]:
    """The ranges of article numbers the institution `name` of code `code` resolves within, or None (no entry)."""
    low = " ".join((name or "").lower().split())
    for pattern, entry_code, ranges, unless in CHAPTERS:
        if entry_code == code and re.search(pattern, low) and not (unless and re.search(unless, low)):
            return ranges
    return None


def in_ranges(number: str, ranges: List[List[int]]) -> bool:
    m = re.match(r"\d+", str(number or ""))
    return bool(m) and any(lo <= int(m.group(0)) <= hi for lo, hi in ranges)


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower())
            if w not in {"della", "delle", "dello", "degli", "nella", "nelle", "dalla", "come", "sono", "diritto",
                         "codice", "norme", "articolo", "altri", "altre"}}


def pick_in_chapter(session, doc_id: str, name: str, ranges: List[List[int]],
                    words: Optional[List[str]] = None) -> Optional[str]:
    """The number of the article inside `ranges` of document `doc_id` whose rubrica shares most content words with
    the institution's name (then its keywords); None when none shares one. Ties: the lowest number."""
    rows = session.run(
        "UNWIND $ranges AS r MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) "
        "WITH r, s, split(s.name, '.')[0] AS num "
        "WHERE toInteger(split(num, '-')[0]) >= r[0] AND toInteger(split(num, '-')[0]) <= r[1] "
        "RETURN DISTINCT num, coalesce(s.titolo_articolo, s.title, '') AS rubrica",
        doc=doc_id, ranges=ranges).data()
    target = _stems(name)
    extra = _stems(" ".join(w for w in words or [] if isinstance(w, str)))
    best = None
    for r in rows:
        num, rubrica = str(r.get("num") or ""), _stems(r.get("rubrica") or "")
        if not re.match(r"^\d+(?:-[a-z]+)?$", num):
            continue
        score = (len(target & rubrica), len(extra & rubrica))
        if score[0] + score[1] == 0:
            continue
        key = (score, -int(re.match(r"\d+", num).group(0)))
        if best is None or key > best[0]:
            best = (key, num)
    return best[1] if best else None


# --- I5: internal tokens --------------------------------------------------------------------------------------------
_STATUS_WORDS = {"NON DECORSO": "non ancora decorso", "DECORSO": "già decorso", "INCERTO": "incerto",
                 "NON VERIFICABILE": "non verificabile con le date del caso", "DA VERIFICARE": "da verificare",
                 "IN CORSO": "in corso", "PRECLUSO": "precluso", "SCADUTO": "scaduto", "NON APPLICABILE": "non applicabile",
                 "APPLICABILE": "applicabile"}
_STATUS_RE = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, _STATUS_WORDS), key=len, reverse=True)) + r")\b")


def plain_status(text: str) -> str:
    """A term's status in plain Italian ("termine annuale DECORSO" -> "termine annuale già decorso")."""
    return _STATUS_RE.sub(lambda m: _STATUS_WORDS[m.group(0)], text or "")


_MARK_RE = re.compile(r"\[(DA VERIFICARE)(?::\s*([^\]]*))?\]")
_RAW_ID_RE = re.compile(
    r"\bsn(?:civ|pen)[\w@.\-]*|\bLEGAL_DOC::\S+|\b[\w\-@]+\.clean\.pdf\b|\bregola\s+[a-z]+(?:_[a-z]+)+|"
    r"\[S\d+\]|\b[a-z]{3,}(?:_[a-z]{2,}){1,}\b")
_PLACEHOLDER_RE = re.compile(r"\[(?:\s*(?:nome|data|importo|indirizzo|luogo|numero|citt[aà]|inserire|completare|"
                             r"x+|\.\.\.|…|•)[^\]]{0,30})\]|<\s*(?:nome|data|importo|inserire)[^>]{0,30}>|\bXXX+\b",
                             re.IGNORECASE)
_BRACKETED_RE = re.compile(r"\[([^\]\n]{1,80})\](?!\()")
_SECTION_NAME_RE = re.compile(r"\b(\d+(?:-[a-z]+)?)\.(\d+)\.0\b")
_DOUBLED_RE = re.compile(r"\b([a-zà-ù]{2,})(\s+\1)+\b", re.IGNORECASE)


def internal_tokens(text: str) -> List[str]:
    """The internal tokens in text (I5)."""
    found = [m.group(0) for m in _STATUS_RE.finditer(text or "")]
    found += [m.group(0) for m in _RAW_ID_RE.finditer(text or "")]
    found += [m.group(0) for m in _PLACEHOLDER_RE.finditer(text or "")]
    found += [m.group(0) for m in _BRACKETED_RE.finditer(text or "")]
    found += [m.group(0) for m in _SECTION_NAME_RE.finditer(text or "")]
    found += [m.group(0) for m in _DOUBLED_RE.finditer(text or "") if m.group(1).lower() not in {"no", "via"}]
    found += [ln.strip() for ln in (text or "").split("\n") if ln.count("**") % 2]
    return found


def repair_tokens(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): the internal tokens repaired -- an unverified mark as "(da verificare...)", a status in plain
    Italian, a section name as its article ("1158.0.0" -> "1158", "2051.2.0" -> "2051, comma 2"), a doubled word
    once, a stray "**" taken out, other brackets opened; raw ids and the sentences holding a placeholder removed."""
    from . import plan_then_write as ptw
    log: List[Dict[str, Any]] = []

    def note(kind, old, new=""):
        log.append({"kind": kind, "token": old[:120], "now": new[:120]})
    spans = [(s, e) for s, e in ptw._sentence_spans(text) if _PLACEHOLDER_RE.search(text[s:e])]
    for s, e in spans:
        note("placeholder", text[s:e].strip())
    text = ptw._drop_spans(text, spans) if spans else text

    def mark(m):
        new = f"(da verificare{': ' + m.group(2).strip() if m.group(2) else ''})"
        note("status", m.group(0), new)
        return new
    text = _MARK_RE.sub(mark, text)

    def status(m):
        note("status", m.group(0), _STATUS_WORDS[m.group(0)])
        return _STATUS_WORDS[m.group(0)]
    text = _STATUS_RE.sub(status, text)

    def raw(m):
        note("raw_id", m.group(0))
        return ""
    text = _RAW_ID_RE.sub(raw, text)

    def section(m):
        new = m.group(1) if m.group(2) == "0" else f"{m.group(1)}, comma {m.group(2)}"
        note("mangled", m.group(0), new)
        return new
    text = _SECTION_NAME_RE.sub(section, text)

    def doubled(m):
        if m.group(1).lower() in {"no", "via"}:
            return m.group(0)
        note("mangled", m.group(0), m.group(1))
        return m.group(1)
    text = _DOUBLED_RE.sub(doubled, text)

    def bracket(m):
        note("bracket", m.group(0), m.group(1))
        return m.group(1)
    text = _BRACKETED_RE.sub(bracket, text)
    lines = []
    for ln in text.split("\n"):
        if ln.count("**") % 2:
            note("mangled", ln.strip(), ln.replace("**", "").strip())
            ln = ln.replace("**", "")
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", re.sub(r"(?<=\S)[ \t]{2,}", " ", text))
    return text, log


# --- I7: durations --------------------------------------------------------------------------------------------------
_NUMBERS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7,
            "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "quindici": 15, "diciotto": 18, "venti": 20,
            "vent'": 20, "ventiquattro": 24, "trenta": 30, "trent'": 30, "quaranta": 40, "quarantacinque": 45,
            "sessanta": 60, "novanta": 90, "cento": 100, "centoventi": 120, "centottanta": 180}
_ADJECTIVES = {"annual": (1, "ann"), "biennal": (2, "ann"), "triennal": (3, "ann"), "quinquennal": (5, "ann"),
               "decennal": (10, "ann"), "ventennal": (20, "ann")}
_DURATION_RE = re.compile(
    r"\b(\d+|" + "|".join(sorted((re.escape(w) for w in _NUMBERS), key=len, reverse=True)) + r")\s*"
    r"(giorn|mes|ann)\w*|\b(annual|biennal|triennal|quinquennal|decennal|ventennal)[ei]\b", re.IGNORECASE)


def durations(text: str) -> set:
    """{(n, "giorn" | "mes" | "ann")} stated in text: "8 giorni", "otto giorni", "vent'anni", "termine annuale"."""
    out = set()
    for m in _DURATION_RE.finditer(text or ""):
        if m.group(3):
            out.add(_ADJECTIVES[m.group(3).lower()])
            continue
        word = m.group(1).lower()
        n = int(word) if word.isdigit() else _NUMBERS.get(word)
        if n:
            out.add((n, m.group(2).lower()))
    return out


# --- I2 / I6: the calls ---------------------------------------------------------------------------------------------
_I2_BATCH = 20
_READABILITY_SYSTEM = (
    "Sei un avvocato italiano e rileggi una sezione di un parere. Ti vengono date le sue frasi, numerate. Indica SOLO "
    "le frasi sgrammaticate, incoerenti, troncate o senza senso compiuto, e per ciascuna una riscrittura corretta "
    "che dica la stessa cosa: non aggiungere né togliere fatti, norme o pronunce, e lascia identici tutti i numeri "
    "di articoli e di pronunce. Le frasi corrette non vanno indicate. Rispondi solo con un array JSON (vuoto se "
    "nessuna): [{\"i\": numero, \"frase\": \"la riscrittura\"}].")
_CITATION_NUMBER_RE = re.compile(r"\b\d+(?:-[a-z]+)?(?:/\d{2,4})?\b", re.IGNORECASE)


def citation_numbers(text: str) -> List[str]:
    """The numbers of articles and rulings a sentence cites, sorted (I6 refuses a rewrite that changes them)."""
    from . import defensive_generation as dg
    refs = [n for m in dg._ARTICLE_RUN_RE.finditer(text or "") for n in dg._RUN_NUMBER_RE.findall(m.group(1))]
    refs += [f"{a}/{b}" for a, b in dg._decision_numbers(text or "")]
    return sorted(str(r).lower() for r in refs)


def _sections(text: str) -> List[Tuple[str, str]]:
    """[(heading line or "", body)] of an answer by its "**N. Title**" headings."""
    parts = re.split(r"(?m)^(\*\*\d+\.\s[^*\n]+\*\*)[ \t]*$", text)
    out = [("", parts[0])] if parts[0].strip() else []
    out += [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return out


def _join(sections: List[Tuple[str, str]]) -> str:
    return "\n\n".join((f"{h}\n\n{b.strip()}" if h else b.strip()) for h, b in sections).strip()


# --- the layer ------------------------------------------------------------------------------------------------------

def _counts(found: int, fixed: int, remaining: int, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"found": found, "fixed": fixed, "remaining": remaining, "entries": entries[:40]}


def allowed_refs(article_rows: List[Dict[str, Any]], named: List[tuple], signals: Optional[Dict[str, Any]],
                 procedural: List[tuple]) -> List[tuple]:
    """I1's checked material: articles graded 2-3 (every article of the checked pool when nothing was graded), the
    user's own, the offered procedural tools, the articles of the computed terms."""
    graded = any(r.get("_grade") is not None for r in article_rows)
    rows = [r["_article"] for r in article_rows if r.get("_article")
            and (not graded or (isinstance(r.get("_grade"), int) and r["_grade"] >= 2))]
    tools = [tuple(ref) for t in (signals or {}).get("tools") or [] for ref in t.get("refs") or []]
    terms = []
    for t in (signals or {}).get("terms") or []:
        m = re.match(r"\s*(\d+(?:-[a-z]+)?)\s+(.+?)\s*$", str(t.get("ref") or ""))
        if m:
            terms.append((m.group(1), m.group(2)))
    return list(dict.fromkeys([tuple(r) for r in rows] + list(named) + tools + list(procedural) + terms))


def i1_closure(text: str, allowed: List[tuple], documents: List[Dict[str, Any]], known: set) -> Tuple[str, Dict]:
    from . import plan_then_write as ptw
    bad = list(dict.fromkeys(r for _, _, refs in ptw._citations(text, documents, known) for r in refs
                             if not any(ptw._same_ref(r, a) for a in allowed)))
    if not bad:
        return text, _counts(0, 0, 0, [])
    text, log = ptw.remove_citations(text, bad, documents, known, "answer")
    left = [r for _, _, refs in ptw._citations(text, documents, known) for r in refs
            if not any(ptw._same_ref(r, a) for a in allowed)]
    entries = [{"refs": e.get("removed"), "sentence": e.get("sentence"), "how": e.get("how"),
                "kind": e.get("kind") or "citation"} for e in log]
    return text, _counts(len(bad), len(bad) - len(set(left)), len(set(left)), entries)


def _clause_without(sentence: str, number: str) -> str:
    """The sentence without the clause that only cites article `number` -- a parenthesis "(art. N c.c.)" or a tail
    "ai sensi / a norma / in base / come previsto / in virtù / ex ... art. N c.c." -- when what is left stands alone
    (six words at least); "" otherwise (the citation is then marked)."""
    num = re.escape(number)
    cite = rf"\bartt?\.\s*(?:[\w-]+\s*(?:,|e)\s*)*{num}\b"
    for pattern in (rf"\s*\([^()]*{cite}[^()]*\)",
                    rf"(?:,\s*)?\b(?:ai sensi|a norma|in base|come previsto|secondo quanto previsto|in virtù|"
                    rf"conformemente|ex)\b[^,;()]{{0,30}}?{cite}[^,;()]*?(?=\s*[,;]|\s*\.?\s*$)"):
        new = re.sub(pattern, "", sentence, count=1, flags=re.IGNORECASE)
        if new != sentence:
            new = re.sub(r"\s+([,;.])", r"\1", new).strip()
            new = new if new.endswith((".", "!", "?")) else new.rstrip(" ,;") + "."
            return new if len(re.findall(r"\w+", new)) >= 6 else ""
    return ""


def i2_consistency(text: str, rows_by_ref: Dict[tuple, Dict[str, Any]], protected: set,
                   documents: List[Dict[str, Any]], known: set, chat_json: Callable) -> Tuple[str, Dict]:
    """Every citation in every sentence against its article's rubrica and opening text."""
    from . import defensive_generation as dg
    from . import plan_then_write as ptw
    items = []
    for s, e in ptw._sentence_spans(text):
        sentence = text[s:e]
        for cs, ce, refs in ptw._citations(sentence, documents, known):
            for ref in refs:
                row = next((r for k, r in rows_by_ref.items() if ptw._same_ref(k, ref)), None)
                heading = ((row or {}).get("s") or {}).get("titolo_articolo") or ((row or {}).get("s") or {}).get("title")
                if row and heading:
                    items.append({"span": (s, e), "cite": (cs, ce), "ref": ref, "heading": heading.strip(),
                                  "opening": " ".join(str((row.get("s") or {}).get("plain_text") or "").split())[:300],
                                  "sentence": " ".join(sentence.split())})
    if not items:
        return text, _counts(0, 0, 0, [])
    system = dg._QUALIFICATION_SYSTEM_TYPED.replace(" Nel dubbio è compatibile.",
                                                     dg._QUALIFICATION_DIRECTION + " Nel dubbio è compatibile.")

    def batch(start):
        part = items[start:start + _I2_BATCH]
        human = "\n\n".join(f"[A{start + k}] art. {it['ref'][0]} {it['ref'][1]}\nRubrica: {it['heading']}\n"
                            f"Testo: {it['opening'] or '-'}\nFrase: {it['sentence']}" for k, it in enumerate(part))
        try:
            return chat_json([SystemMessage(content=system.replace("l'identificativo dato",
                                                                   "l'identificativo tra parentesi quadre (A0, A1...)")),
                              HumanMessage(content=human)], max_tokens=120 * len(part) + 100, temperature=0)
        except Exception as exc:
            logger.warning("answer invariants: I2 batch failed, its citations kept: %s", exc)
            return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(batch, range(0, len(items), _I2_BATCH)))
    bad: Dict[int, Dict[str, Any]] = {}
    for reply in replies:
        for v in reply if isinstance(reply, list) else []:
            m = re.search(r"A(\d+)", str((v or {}).get("articolo") or "")) if isinstance(v, dict) else None
            if m and v.get("compatibile") is False and str(v.get("tipo") or "") != "formulazione":
                k = int(m.group(1))
                if 0 <= k < len(items):
                    bad[k] = {**items[k], "why": str(v.get("perche") or "")[:200]}
    entries, drop, replace, marks = [], [], {}, 0
    for it in bad.values():
        span, ref = it["span"], it["ref"]
        keep = any(ptw._same_ref(ref, p) for p in protected)
        entry = {"ref": f"{ref[0]} {ref[1]}", "sentence": it["sentence"][:300], "why": it["why"]}
        if not keep:
            drop.append(span)
            entry["how"] = "sentence dropped"
        else:
            current = replace.get(span, text[span[0]:span[1]])
            cut = _clause_without(current, ref[0])
            if cut:
                replace[span] = cut
                entry["how"] = "clause dropped"
            else:
                label = f"art. {ref[0]} {ref[1]}"
                replace[span] = current.rstrip(" .") + f" (da verificare: l'{label} riguarda «{it['heading']}»)."
                entry["how"] = "marked"
                marks += 1
        entries.append(entry)
    for span in sorted(set(drop) | set(replace), reverse=True):
        s, e = span
        text = text[:s] + ("" if span in drop else replace[span]) + text[e:]
    text = ptw._tidy(text) if drop else text
    return text, _counts(len(bad), len(bad) - marks, marks, entries)


_PROCEDURAL_QUESTION_RE = re.compile(r"ricors|appell|impugn|cassazion|competen|giurisdizion|\brito\b|procedur|"
                                     r"inammissib|notific|termine per (?:proporre|impugnare)|esecuzion", re.IGNORECASE)
_PROCEDURAL_CODES = {"c.p.c.", "c.p.p.", "c.p.a."}
_OBJECTION_LINE_RE = re.compile(r"^\W*(?:obiezione|replica)\b", re.IGNORECASE)
_OBJECTION_WORDS_RE = re.compile(r"obiezion|replica|controparte|potrebbe (?:invocare|sostenere|obiettare)",
                                 re.IGNORECASE)


def i4_rulings(text: str, ruling_rows: List[Dict[str, Any]], cites: Dict[str, set], case_text: str,
               documents: List[Dict[str, Any]], known: set) -> Tuple[str, Dict]:
    """The rulings the answer cites, each kept only when it shares a substantive article with the answer, has a
    substantive holding, and (adverse) sits in an objection or a reply."""
    from . import defensive_generation as dg
    from . import plan_then_write as ptw
    from . import source_tiers
    plate = source_tiers.boilerplate()
    answer = {(str(n), ptw._code_key(c)) for _, _, refs in ptw._citations(text, documents, known) for n, c in refs}
    procedural_case = bool(_PROCEDURAL_QUESTION_RE.search(case_text or ""))
    spans = ptw._sentence_spans(text)
    drop, entries, found = set(), [], 0
    for row in ruling_rows:
        label = dg._ruling_label(row)
        numbers = dg._decision_numbers(label)
        citing = [(s, e) for s, e in spans if dg._ruling_is_cited(label, " ".join(text[s:e].lower().split()),
                                                                   dg._decision_numbers(text[s:e]))]
        if not citing or not (numbers or label):
            continue
        refs = {(str(n), ptw._code_key(c)) for n, c in cites.get((row.get("d") or {}).get("id"), set())}
        substantive = refs - {(n, ptw._code_key(c)) for n, c in plate}
        why = ""
        if not substantive & answer:
            why = "no substantive article in common with the answer"
        elif (all(c in _PROCEDURAL_CODES for _, c in substantive) or re.search(r"inammissib|improcedib",
                                                                              str(row.get("_card_esito") or ""), re.I)) \
                and not procedural_case:
            why = "procedural holding only"
        if why:
            bad = citing
        elif row.get("_fit_adverse"):
            why = "adverse, cited outside an objection or a reply"
            bad = [(s, e) for s, e in citing
                   if not _OBJECTION_LINE_RE.match(text[text.rfind("\n", 0, s) + 1:e])
                   and not _OBJECTION_WORDS_RE.search(text[s:e])]
        else:
            bad = []
        if bad:
            found += 1
            drop.update(bad)
            entries.append({"ruling": label, "why": why, "sentences": len(bad)})
    if not drop:
        return text, _counts(0, 0, 0, [])
    text = ptw._drop_spans(text, sorted(drop))
    return text, _counts(found, found, 0, entries)


def i6_readability(sections: List[Tuple[str, str]], call_chat: Callable) -> Tuple[List[Tuple[str, str]], Dict]:
    """One call per section; a flagged sentence replaced by its rewriting when the citation numbers are the same and
    its length is in proportion."""
    from . import defensive_generation as dg
    from . import plan_then_write as ptw

    def one(body):
        spans = ptw._sentence_spans(body)
        if not spans:
            return []
        numbered = "\n".join(f"{k}. {' '.join(body[s:e].split())}" for k, (s, e) in enumerate(spans, 1))
        try:
            reply = dg.parse_json_reply(call_chat([SystemMessage(content=_READABILITY_SYSTEM),
                                                   HumanMessage(content=numbered)],
                                                  max_tokens=150 + 40 * len(spans), temperature=0))
        except Exception as exc:
            logger.warning("answer invariants: I6 call failed, section kept: %s", exc)
            return []
        return [(spans[int(v["i"]) - 1], " ".join(str(v.get("frase") or "").split()))
                for v in reply if isinstance(v, dict) and str(v.get("i", "")).isdigit()
                and 1 <= int(v["i"]) <= len(spans)] if isinstance(reply, list) else []
    with ThreadPoolExecutor(max_workers=max(1, len(sections))) as pool:
        flagged = list(pool.map(lambda pair: one(pair[1]), sections))
    out, entries, found, fixed = [], [], 0, 0
    for (heading, body), flags in zip(sections, flagged):
        for (s, e), new in sorted(dict(flags).items(), reverse=True):
            old = body[s:e]
            found += 1
            ok = bool(new) and citation_numbers(old) == citation_numbers(new) and 0.5 * len(old) <= len(new) <= 1.8 * len(old)
            entries.append({"sentence": " ".join(old.split())[:300], "rewritten": new[:300],
                            "how": "rewritten" if ok else "refused (citations or length changed)"})
            if ok:
                body = body[:s] + new + body[e:]
                fixed += 1
        out.append((heading, body))
    return out, _counts(found, fixed, found - fixed, entries)


def i7_terms(text: str, signals: Optional[Dict[str, Any]], case_text: str, rows_by_ref: Dict[tuple, Dict[str, Any]],
             documents: List[Dict[str, Any]], known: set) -> Tuple[str, Dict]:
    """Sentences stating a duration not in the term table, the case's text, or the text of an article they cite."""
    from . import plan_then_write as ptw
    table = set()
    for t in (signals or {}).get("terms") or []:
        table |= durations(f"{t.get('length') or ''} {t.get('text') or ''}")
    in_case = durations(case_text)
    drop, entries = [], []
    for s, e in ptw._sentence_spans(text):
        sentence = text[s:e]
        stated = durations(sentence)
        if not stated:
            continue
        cited = set()
        for _, _, refs in ptw._citations(sentence, documents, known):
            for ref in refs:
                row = next((r for k, r in rows_by_ref.items() if ptw._same_ref(k, ref)), None)
                cited |= durations(str(((row or {}).get("s") or {}).get("plain_text") or ""))
        stray = stated - table - in_case - cited
        if stray:
            drop.append((s, e))
            entries.append({"sentence": " ".join(sentence.split())[:300],
                            "durations": [f"{n} {u}" for n, u in sorted(stray)]})
    if not drop:
        return text, _counts(0, 0, 0, [])
    return ptw._drop_spans(text, drop), _counts(len(drop), len(drop), 0, entries)


def guards(text: str, request: str, facts: str) -> Dict[str, Any]:
    """The guards already in place, counted on the final answer."""
    from . import plan_then_write as ptw
    sections = {h: b for h, b in _sections(text)}
    first = next((b for h, b in sections.items() if re.match(r"\*\*1\.", h or "")), "")
    questions = ptw.explicit_questions(request, facts)
    sentences = [text[s:e] for s, e in ptw._sentence_spans(text)]
    return {"questions": len(questions),
            "questions_unanswered": [q for q in questions if first and not ptw._answers(q, first)],
            "not_apply": sum(1 for x in sentences if ptw._cites(x) and ptw._NOT_APPLY_RE.search(x))}


def enforce(draft: str, *, facts: str, request: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
            ruling_rows: List[Dict[str, Any]], documents: List[Dict[str, Any]], session=None,
            chat_json: Optional[Callable] = None, call_chat: Optional[Callable] = None,
            cites: Optional[Dict[str, set]] = None) -> Tuple[str, Dict[str, Any]]:
    """(draft, log): the answer with I1, I2, I4, I5, I6, I7 enforced in that order (I5 first too, so a mangled
    citation is read), then the orphan fragments dropped and the guards counted; text after a "⚠️" line untouched.
    Logged as case_analysis_invariants."""
    from . import defensive_generation as dg
    from . import method_candidates
    from . import plan_then_write as ptw
    chat_json = chat_json or dg._chat_json
    call_chat = call_chat or dg._call_chat
    t0 = time.time()
    cut = draft.find("\n\n⚠️")
    body, tail = (draft[:cut], draft[cut:]) if cut >= 0 else (draft, "")
    case_text = dg._case_text(facts, request)
    signals = (analysis or {}).get("fact_signals") or {}
    named = dg._named_refs(case_text, documents)
    procedural = method_candidates.candidate_refs([], method_candidates.case_lists(analysis)[1])
    allowed = allowed_refs(article_rows, named, signals, procedural)
    known = {r["_article"] for r in article_rows if r.get("_article")} | set(allowed)
    rows_by_ref: Dict[tuple, Dict[str, Any]] = {}
    for row in article_rows:
        if row.get("_article"):
            rows_by_ref.setdefault(row["_article"], row)
    log: Dict[str, Any] = {}

    body, repairs = repair_tokens(body)
    first_repairs = len(repairs)
    body, log["I1"] = i1_closure(body, allowed, documents, known)
    cited = list(dict.fromkeys(r for _, _, refs in ptw._citations(body, documents, known) for r in refs))
    missing = [r for r in cited if not any(ptw._same_ref(r, k) for k in rows_by_ref)]
    if missing and session is not None:
        try:
            for row in dg._fetch_any(session, documents, missing):
                rows_by_ref.setdefault(row.get("_article"), row)
        except Exception as exc:
            logger.warning("answer invariants: article fetch failed: %s", exc)
    protected = {r["_article"] for r in article_rows if r.get("_article") and r.get("_grade") == 3} | set(named)
    body, log["I2"] = i2_consistency(body, rows_by_ref, protected, documents, known, chat_json)
    body, log["I7"] = i7_terms(body, signals, case_text, rows_by_ref, documents, known)
    cited_rulings = [r for r in ruling_rows if (r.get("d") or {}).get("id")]
    if cites is None:
        cites = {}
        if session is not None and cited_rulings:
            from . import source_tiers
            try:
                cites = source_tiers.ruling_citations(session, [(r.get("d") or {}).get("id") for r in cited_rulings])
            except Exception as exc:
                logger.warning("answer invariants: CITES of the rulings not read, I4 skipped: %s", exc)
                cites = None
    if cites is not None:
        body, log["I4"] = i4_rulings(body, cited_rulings, cites, case_text, documents, known)
    else:
        log["I4"] = {**_counts(0, 0, 0, []), "skipped": True}
    sections, log["I6"] = i6_readability(_sections(body), call_chat)
    body = _join(sections)
    body, orphans = ptw.drop_dangling(body, "answer")
    body, more = repair_tokens(body)
    remaining = internal_tokens(body)
    log["I5"] = _counts(first_repairs + len(more), first_repairs + len(more), len(remaining),
                        repairs + more + [{"kind": "remaining", "token": t[:120]} for t in remaining])
    after_i1 = sum(1 for e in log["I1"]["entries"] if e.get("kind") == "dangling")
    log["guards"] = {**guards(body, request, facts), "orphans_removed": len(orphans) + after_i1}
    log["I3"] = {"note": "counted at resolution: case_analysis_institution_range"}
    vlog_payload = {k: v for k, v in log.items()}
    dg.vlog("case_analysis_invariants", vlog_payload, duration_ms=(time.time() - t0) * 1000)
    return body + tail, log
