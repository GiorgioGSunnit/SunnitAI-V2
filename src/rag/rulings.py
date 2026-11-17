"""v78 (RULINGS_RECORD, default off): the Cassazione rulings into the record. The graph holds the rulings of 2025-2026
(Document.document_type 'interpretation', their sections with plain_text and abstract, (Section)-[:CITES]->(article
Section) written by scripts/link_ruling_citations.py, which stamps Document.cites_linked_at). Per answer:

  1. fetch     for each record article (lawyer_method's records: the charge or claim, the alternatives, the named
               articles, the client's), the interpretation sections that CITES its Section, grouped by ruling Document:
               file_name, document_number, date_enacted, the citing sections' abstract (plain_text read later, for the
               shortlist only). Only a ruling with cites_linked_at is kept (the CITES edges are the linker's: by
               construction); found / linked counted per article.
  2. estremi   from the file name only (parse_estremi): archive, section code, year, number, act; the chamber from
               data/ruling_sections.json (provisional until Antonelo confirms it); date_enacted gives the date when it
               reads as one, else «data n.d.»; document_number never overrides the file name.
  3. rank      the abstract is a tag and a quoted passage: it only ranks -- its noun overlap with the elements in
               dispute shortlists, then the linking call grades each shortlisted ruling (pertinenza 0-2, as the
               applicability grader does an article); the order: Sezioni Unite first, then that score, then the date.
               GRADED_PER_ARTICLE graded per article (pertinenza 0 dropped), of which PER_ARTICLE printed and
               PER_ANSWER per answer (round robin over the articles).
  4. passage   the same call picks, among the citing section's own sentences, the one that states the principle on the
               element (its id; the sentence printed is ours, verbatim); none -> the estremi alone, no quote. The word
               «massima» is never printed.
  5. orientamento  per controverted point (an element a candidate thesis contests): the call's «verso» of each graded
               passage on that element (up to GRADED_PER_ARTICLE, printed or not) -- for or against its integration; MIN_ORIENTATION or more with a majority ->
               «orientamento prevalente» / «minoritario» with the counts, else «orientamento da verificare» with the
               rulings listed.
  6. §5        section_text: the rulings by article, the orientations, the lawyer's pronunce di riferimento
               (landmark_rulings), the fixed CORPUS_LINE.

The registry (registry_block) is the material of the thesis call, the objections call and the norms writer, whose rule
flips from «Non citare pronunce» to WRITER_RULE; lawyer_method's ruling check drops any ruling not in it (v71).
cap_realism: a prevailing line for the integration of an element lowers every thesis contesting it to «basso».
Logged as rulings_for_record; the record (state["rulings_record"]) goes to the held-out set and the nightly.
"""

import json
import logging
import os
import re
import time
from datetime import date
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import feature_flags

logger = logging.getLogger(__name__)

FLAG = "RULINGS_RECORD"
SECTIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                             "ruling_sections.json")
CORPUS_LINE = "Corpus giurisprudenziale disponibile: Cassazione 2025–2026."
PASSAGE_LABEL = "passaggio della motivazione"
REGISTRY_TITLE = "REGISTRO DELLE PRONUNCE"
WRITER_RULE = " Cita solo le pronunce del REGISTRO DELLE PRONUNCE, con gli estremi indicati."
PER_ARTICLE = 3
PER_ANSWER = 8
FETCH_PER_ARTICLE = 1000        # citing sections read per article (metadata and abstract only), the Sezioni Unite
                                # first, then the newest (FETCH_QUERY's order): a cap never drops a Sezioni Unite
GRADED_PER_ARTICLE = 6          # rulings per article graded by the linking call: the orientation counts them all,
                                # the top PER_ARTICLE are printed (with 3 graded, an orientation needed all three)
SHORTLIST_MAX = 18

SENTENCES_PER_RULING = 5        # candidate sentences per ruling in the call (16 x 5 x 500 chars: the call's material)
SENTENCE_CHARS = (40, 500)      # a candidate sentence's length, min and max
MIN_ORIENTATION = 3
LINK_TOKENS = 2000
FOR, AGAINST, NEUTRAL = "a favore", "contro", "neutro"

# v79 (E2): the reader never sees «registro» -- «Pronunce sulle norme applicate»
SECTION_TITLES = {"rulings": "**Pronunce sulle norme applicate**", "orientations": "**Orientamenti sui punti controversi**",
                  "landmarks": "**Pronunce di riferimento indicate dal consulente**"}
NO_RULINGS_LINE = "Nessuna pronuncia del corpus cita le norme applicate."
PREVALENT = ("orientamento prevalente: elemento {side} ({n} pronunce su {tot}); orientamento minoritario: elemento "
             "{other} ({m} su {tot})")
UNANIMOUS = "orientamento prevalente: elemento {side} ({n} pronunce su {tot})"
TO_VERIFY = "orientamento da verificare"
SIDES = {FOR: "integrato", AGAINST: "non integrato"}

# v78 (the PO's review): a quotable sentence states a rule -- one of these verbs --, holds the graded element's nouns,
# and holds none of the procedural markers (a motivo, the court below, a party's complaint, a reference's «dep.»/«Rv.»)
_RULE_RE = re.compile(r"\b(?:richied\w*|si\s+configura\w*|configura(?:no)?\b|integra(?:no)?\b|(?:non\s+)?occorr\w*|"
                      r"sussist\w*|(?:è|sia|sono|siano|risulta|risultano)\s+(?:necessari|sufficient|richiest|integrat|"
                      r"costituit|esclus|configurabil)\w*|presuppon\w*|si\s+realizza\w*|deve\s+ritenersi|va\s+ritenut\w*|va\s+esclus\w*|"
                      r"(?:è|resta|rimane)\s+esclus\w*|esclud\w*|rientra(?:no)?\b|costituisc\w*|si\s+intend\w*|"
                      r"consist\w*|ricorr(?:e|ono)\b)", re.IGNORECASE)
_PROCEDURAL_RE = re.compile(r"\bmotiv[oi]\s+di\s+ricorso|\b(?:primo|secondo|terzo|quarto|quinto|sesto|unico)\s+motivo\b|"
                            r"\bcorte\s+territoriale\b|\bcorte\s+d['’]\s?appello\b|\bricorrent[ei]\b|\blament\w*|"
                            r"\bsi\s+duol\w*|\bdeduc[eo]\w*|\bdep\.|\brv\.|\baddebito\s+mosso|\b(?:sentenza|ordinanza)\s+"
                            r"impugnata\b|\bil\s+difensore\b|\bricorso\s+per\s+cassazione\b|\b(?:è|era)\s+stat[oa]\s+"
                            r"condannat\w*|\bimputat[oa]\s+(?:del|di)\s+(?:reato|delitto)", re.IGNORECASE)
DEMOTED_SECTIONS = ("7",)       # Sez. 7 (inammissibilità) -- last, with the ordinanze
DEMOTED_ACTS = ("O", "I")

_ITALGIURE_RE = re.compile(r"(snpen|snciv)[@_ ]s(\w)(\w)[@_ ]a((?:19|20)\d{2})[@_ ]n0*(\d+)[@_ ]t([a-z])", re.I)
_DATE_RES = (re.compile(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})"), re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-]((?:19|20)\d{2})"))
_WORD_RE = re.compile(r"[a-zàèéìòù]{4,}", re.I)
_STOP = set("""
alla alle allo anche ancora avere aveva base caso cioè come con contro cosa così dalla dalle dallo degli della delle
dello dell detto dopo dove essere fatto fatti fine hanno infatti invece loro mentre nella nelle nello nonché oltre
ogni ovvero perché però poiché presente prima quale quali quando quanto quella quelle quello questa queste questo
sensi senza sono stata stato stessa stesso sulla sulle sullo tale tali tutti tutto venga viene sentenza ricorso
ricorrente corte appello cassazione giudice motivo motivi articolo codice comma decisione pronuncia impugnata
""".split())

_LINK_SYSTEM = (    # my draft (v78): the linking call -- one per answer, every shortlisted ruling
    "Sei un avvocato italiano. Per ciascuna PRONUNCIA ricevi la norma che cita, gli ELEMENTI di quella norma in "
    "discussione nel caso (ciascuno con il suo id) e alcune FRASI della motivazione, ciascuna con il suo id "
    "([P1.1], [P1.2]…). Per ogni pronuncia indica:\n"
    "- \"frase\": l'id della sola frase che enuncia il principio di diritto su uno degli ELEMENTI (la regola che la "
    "Corte applica a quell'elemento); vuoto se nessuna lo enuncia: il racconto dei fatti del giudizio, dello "
    "svolgimento del processo o delle tesi delle parti non è un principio;\n"
    "- \"elemento\": l'id dell'elemento su cui è quella frase (vuoto senza frase);\n"
    "- \"pertinenza\": 2 se il principio riguarda l'elemento, 1 se riguarda la norma ma non quell'elemento, 0 se "
    "riguarda altro;\n"
    "- \"verso\": rispetto all'integrazione dell'elemento, \"a favore\" se il principio porta a ritenerlo integrato "
    "(ne allarga i presupposti, respinge la lettura che lo nega), \"contro\" se porta a escluderlo (richiede qualcosa "
    "di più, accoglie la lettura che lo nega), \"neutro\" altrimenti.\n"
    "Non inventare frasi: indica solo id ricevuti. Rispondi SOLO con un JSON: {\"P1\": {\"frase\": \"P1.3\", "
    "\"elemento\": \"R2\", \"pertinenza\": 2, \"verso\": \"a favore\"}, …}")


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def _log(event: str, data: Any, duration_ms: Optional[float] = None) -> None:
    try:
        from .verbose_logger import vlog
        vlog(event, data, duration_ms)
    except Exception:           # logging never fails an answer
        logger.debug("rulings: %s not logged", event, exc_info=True)


def _plain(text: Any) -> str:
    return " ".join(str(text or "").split())


# --- 2. the estremi, from the file name ------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def sections_map() -> Dict[str, Any]:
    try:
        with open(SECTIONS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("rulings: %s unreadable (%s): no chamber in the estremi", SECTIONS_FILE, exc)
        return {"codes": {}, "sezioni_unite": [], "confirmed": False}


def parse_estremi(file_name: str) -> Optional[Dict[str, Any]]:
    """The Italgiure code of a file name -- «snpen@s10@a2025@n00001@tS.clean.pdf», «snciv_sU0_a2026_n12726_tD» -- as
    {archive penale|civile, section «10», year, number (int), act «S»}; None when the name holds none."""
    m = _ITALGIURE_RE.search(str(file_name or ""))
    if not m:
        return None
    archive, sec, sub, year, number, act = m.groups()
    return {"archive": "penale" if archive.lower() == "snpen" else "civile", "section": (sec + sub).upper(),
            "year": year, "number": int(number), "act": act.upper()}


def chamber(p: Dict[str, Any]) -> str:
    """«Sez. Un.», «Sez. 3», «Sez. 6-1»; "" for a code data/ruling_sections.json does not map."""
    code, sub = p["section"][:1], p["section"][1:]
    name = (sections_map().get("codes") or {}).get(code)
    if not name:
        return ""
    return f"{name}-{sub}" if sub and sub != "0" else name


def sezioni_unite(p: Optional[Dict[str, Any]]) -> bool:
    return bool(p) and p["section"][:1] in set(sections_map().get("sezioni_unite") or [])


def ruling_date(value: Any, year: str = "") -> Optional[date]:
    """date_enacted as a date when it reads as one (ISO, or dd/mm/yyyy) within a year of the number's; else None."""
    text = str(value or "")
    for i, rx in enumerate(_DATE_RES):
        m = rx.search(text)
        if not m:
            continue
        y, mo, d = (m.group(1), m.group(2), m.group(3)) if i == 0 else (m.group(3), m.group(2), m.group(1))
        try:
            found = date(int(y), int(mo), int(d))
        except ValueError:
            return None
        if year and abs(found.year - int(year)) > 1:
            return None
        return found
    return None


def render_estremi(p: Dict[str, Any], date_enacted: Any = None) -> str:
    """«Cass. pen., Sez. Un., n. 12345/2025 (12.03.2025)», «Cass. civ., Sez. 3, n. 678/2026 (data n.d.)»."""
    court = "Cass. pen." if p["archive"] == "penale" else "Cass. civ."
    room = chamber(p)
    when = ruling_date(date_enacted, p["year"])
    return (f"{court}, " + (f"{room}, " if room else "") + f"n. {p['number']}/{p['year']} "
            + (f"({when.day:02d}.{when.month:02d}.{when.year})" if when else "(data n.d.)"))


# --- 3. ranking ------------------------------------------------------------------------------------------------------------

def stems(text: str) -> set:
    return {w.lower()[:6] for w in _WORD_RE.findall(text or "") if w.lower() not in _STOP}


def overlap(text: str, elements: Sequence[Dict[str, Any]]) -> int:
    return len(stems(text) & set().union(*[stems(e.get("text") or "") for e in elements])) if elements else 0


def demoted(p: Optional[Dict[str, Any]]) -> bool:
    """v78: a Sez. 7 ruling or an ordinanza -- ranked after every other."""
    return bool(p) and (p["section"][:1] in DEMOTED_SECTIONS or p["act"] in DEMOTED_ACTS)


def _order_key(r: Dict[str, Any]) -> Tuple:
    when = r.get("date")
    return (0 if r.get("su") else 1, 1 if demoted(r.get("parsed")) else 0, -(r.get("score") or 0),
            -(when.toordinal() if when else 0), r["id"])


def rubrica_stems(rubrica: str) -> set:
    return stems(rubrica)


def on_rubrica(abstract: str, rubrica: str) -> bool:
    """v78: the citing section's abstract shares the article's rubrica nouns -- all of them (two at most: «lesione
    personale»), two of a longer one; an article with no rubrica: True."""
    own = rubrica_stems(rubrica)
    if not own:
        return True
    return len(own & stems(abstract)) >= min(2, len(own))


def element_noun_hit(sentence: str, element: Dict[str, Any]) -> bool:
    """v78: the sentence holds the element's nouns (issue_records.element_nouns: one of one or two, two of more)."""
    from . import issue_records
    own = issue_records.element_nouns(element.get("text") or "")
    if not own:
        return False
    return len(own & issue_records.element_nouns(sentence)) >= (1 if len(own) <= 2 else 2)


def quotable(sentence: str, element: Optional[Dict[str, Any]] = None) -> bool:
    """v78: a sentence that states a rule, names no procedural step, and (with an element) holds its nouns."""
    if not _RULE_RE.search(sentence or "") or _PROCEDURAL_RE.search(sentence or ""):
        return False
    return element is None or element_noun_hit(sentence, element)


def round_robin(by_article: Dict[str, List[Dict[str, Any]]], cap: int) -> List[Dict[str, Any]]:
    """The first of each article, then the second of each…, until cap (the articles in the record's order)."""
    out, depth = [], 0
    while len(out) < cap and any(len(v) > depth for v in by_article.values()):
        for rows in by_article.values():
            if len(rows) > depth and len(out) < cap:
                out.append(rows[depth])
        depth += 1
    return out


# --- 1. the fetch ----------------------------------------------------------------------------------------------------------

FETCH_QUERY = (
    "MATCH (law:Document {id: $doc})-[:CONTAINS]->(t:Section) "
    "WHERE t.name = $num OR t.name = $num + '.0.0' OR t.name STARTS WITH $num + '.' "
    "MATCH (rs:Section)-[:CITES]->(t) MATCH (d:Document)-[:CONTAINS]->(rs) WHERE d.document_type = 'interpretation' "
    "RETURN d.id AS ruling, d.file_name AS file_name, d.name AS name, d.document_number AS document_number, "
    "toString(d.date_enacted) AS date_enacted, d.cites_linked_at IS NOT NULL AS linked, rs.id AS section, "
    "rs.abstract AS abstract ORDER BY coalesce(d.file_name, '') =~ $su DESC, coalesce(toString(d.date_enacted), '') DESC "
    "LIMIT $limit")
TEXT_QUERY = "UNWIND $ids AS id MATCH (s:Section {id: id}) RETURN s.id AS section, s.plain_text AS plain_text"


def group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The citing sections grouped by ruling Document."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        g = out.setdefault(str(r["ruling"]), {"id": str(r["ruling"]), "file_name": r.get("file_name") or "",
                                              "name": r.get("name") or "", "document_number": r.get("document_number"),
                                              "date_enacted": r.get("date_enacted"), "linked": bool(r.get("linked")),
                                              "sections": [], "abstract": ""})
        if r.get("section") and r["section"] not in g["sections"]:
            g["sections"].append(r["section"])
            g["abstract"] = _plain(f"{g['abstract']} {r.get('abstract') or ''}")
    return out


def su_pattern() -> str:
    """The Cypher regex of a Sezioni Unite file name (data/ruling_sections.json's codes)."""
    codes = "".join(re.escape(c) for c in sections_map().get("sezioni_unite") or []) or "\\x00"
    return f"(?i).*sn(?:pen|civ)[@_ ]s[{codes}].*"


def fetch(session, doc: str, num: str, limit: int = FETCH_PER_ARTICLE) -> Dict[str, Dict[str, Any]]:
    rows = session.run(FETCH_QUERY, doc=doc, num=num, limit=limit, su=su_pattern()).data()
    out = group(rows)
    if len(rows) >= limit:
        _log("rulings_fetch_capped", {"doc": doc, "article": num, "sections": len(rows), "rulings": len(out)})
    return out


def texts(session, section_ids: Sequence[str]) -> Dict[str, str]:
    if not section_ids:
        return {}
    return {r["section"]: r.get("plain_text") or "" for r in session.run(TEXT_QUERY, ids=list(section_ids)).data()}


# --- 4. the passages: candidate sentences, the linking call ------------------------------------------------------------------

_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-ZÀÈÉÌÒÙ«\"(])")


# --- v79 (F2): a passage as the reader gets it ------------------------------------------------------------------------------

# the stamp the PDF leaves on every page («Corte di Cassazione - copia non ufficiale», its page count) and page counts
STAMP_RE = re.compile(r"\s*corte\s+di\s+cassazione\s*[-–—]?\s*copia\s+non\s+ufficiale(?:\s*[-–—]?\s*\d{1,4}(?!\s*/))?",
                      re.IGNORECASE)
PAGE_RE = re.compile(r"\s*\b(?:pag(?:ina|\.)?|p\.)\s*\d{1,4}(?:\s*(?:di|/)\s*\d{1,4})?\b", re.IGNORECASE)
# a sentence that starts inside a reference («n. 1234/2025, ha affermato …», «Rv. 2811», «, che …») is never quoted
_MID_REFERENCE_RE = re.compile(r"^(?:[a-zà-ù0-9,;:.)\]/-]|(?:n|nn|rv|dep|sez|cit|pag|cfr)\.\s)", re.IGNORECASE)
_UPPER_START_RE = re.compile(r"^[«\"“(]?[A-ZÀÈÉÌÒÙ]")


def strip_stamp(text: str) -> str:
    """v79 (F2): the text without the PDF's stamp and its page counts."""
    return " ".join(PAGE_RE.sub(" ", STAMP_RE.sub(" ", text or "")).split())


def starts_clean(sentence: str) -> bool:
    """v79 (F2): the sentence starts a sentence -- a capital (or an opening quote), never inside a reference."""
    s = (sentence or "").strip()
    return bool(_UPPER_START_RE.match(s)) and not re.match(r"^(?:Sez|Rv|Cfr|Pag|Dep|Nn?)\.\s", s)


NAMES_ENV = "RULINGS_PARTY_NAMES"      # v79 (F2): «redact» (default) or «keep» -- the decision is the PO's, pending
REDACTED = "[omissis]"
_TITLE_NAME_RE = re.compile(r"\b(?:sig(?:\.|nor|nora)?|dott(?:\.|or)|avv\.|ing\.|geom\.)\s+([A-ZÀ-Ý][a-zà-ÿ']+)")
_NAME_PAIR_RE = re.compile(r"(?<=[\w,;:)] )([A-ZÀ-Ý][a-zà-ÿ']+|[A-ZÀ-Ý]{2,})(\s+(?:[A-ZÀ-Ý][a-zà-ÿ']+|[A-ZÀ-Ý]{2,}))+")
_NOT_NAMES = set("""Corte Cassazione Suprema Sezione Sezioni Unite Tribunale Appello Assise Stato Repubblica Codice Penale
Civile Procura Procuratore Generale Pubblico Ministero Giudice Pace Costituzione Costituzionale Consiglio Ministro Legge
Decreto Regione Comune Provincia Italia Italiana Italiano Europea Europeo Unione Convenzione Carta Diritti Uomo Testo
Unico Polizia Carabinieri Guardia Finanza Agenzia Entrate Banca Lavoro Riesame Sorveglianza Minorenni Gip Gup Gup Pm
Corti Territoriale Distrettuale Superiore Magistratura Collegio Presidente Relatore Consigliere Cedu Edu Inps Inail
Spa Srl Sas Snc Ufficio Registro Imprese Camera Commercio Asl Usl Ente Fondo Istituto Nazionale Previdenza Sociale
Penali Civili Sezione Feriale Unita""".split())


def redact_names(text: str) -> str:
    """v79 (F2): the parties' names out of a quoted passage -- a name after a title («sig. Rossi»), and a run of two or
    more capitalised words inside a sentence none of which is an office, a court or an act («Mario Rossi»,
    «ROSSI Mario») -- as REDACTED; RULINGS_PARTY_NAMES=keep leaves them."""
    if (os.getenv(NAMES_ENV) or "redact").strip().lower() == "keep":
        return text or ""
    out = _TITLE_NAME_RE.sub(lambda m: m.group(0)[:m.start(1) - m.start(0)] + REDACTED, text or "")

    def pair(m: "re.Match") -> str:
        words = m.group(0).split()
        return m.group(0) if any(w.capitalize().strip("'") in _NOT_NAMES for w in words) else REDACTED
    return _NAME_PAIR_RE.sub(pair, out)


def printed(passage: str) -> str:
    """v79 (F2): a kept passage as §5 prints it -- no stamp, no party's name."""
    return redact_names(strip_stamp(passage))


# --- v79 (F1): the registry's labels as the reader gets them --------------------------------------------------------------

_LABEL_RE = re.compile(r"\[(G\d{1,2})\]|\((G\d{1,2})\)|(?<![\w/.-])(G\d{1,2})(?![\w/-])")


def reference(r: Dict[str, Any]) -> str:
    """The ruling's reference for the prose -- «Cass. pen., Sez. 5, n. 1234/2025, 12.03.2025», its date when held."""
    e = str(r.get("estremi") or "")
    e = re.sub(r"\s*\(data n\.d\.\)$", "", e)
    return re.sub(r"\s*\((\d{2}\.\d{2}\.\d{4})\)$", r", \1", e)


def labels_out(text: str, record: Optional[Dict[str, Any]], log: Optional[List[str]] = None) -> str:
    """v79 (F1): every registry label the writer emits («[G1]», «(G1)», «G1») replaced by its ruling's reference; a
    label no registry ruling holds drops its sentence (logged)."""
    from .lawyer_method import _by_sentence
    if not text or not _LABEL_RE.search(text):
        return text or ""
    kept = (record or {}).get("kept") or []

    def ruling(label: str) -> Optional[Dict[str, Any]]:
        i = int(label[1:])
        return kept[i - 1] if 1 <= i <= len(kept) else None

    def review(sentence: str) -> str:
        found = [next(g for g in m.groups() if g) for m in _LABEL_RE.finditer(sentence)]
        if not found:
            return sentence
        if any(ruling(x) is None for x in found):
            if log is not None:
                log.append(f"{', '.join(found)}: {sentence[:120]}")
            return ""
        return _LABEL_RE.sub(lambda m: f"({reference(ruling(m.group(2)))})" if m.group(2)
                             else reference(ruling(m.group(1) or m.group(3))), sentence)
    return _by_sentence(text, review)


# --- v79 (F3): the nightly's check of the rulings as printed ---------------------------------------------------------------

_HEADLESS_RE = re.compile(r"\bSez\.\s*(?:Un\.|\d{1,2})|\bn\.\s*\d{1,6}\s*/\s*(?:19|20)\d{2}\b")
_ACT_BEFORE_RE = re.compile(r"(?:d\.\s?lgs\.?|d\.\s?p\.\s?r\.|d\.\s?l\.|legge|l\.|decreto|reg\.|regolamento|dpr|"
                            r"direttiva|\(ue\)|\(ce\))[^.;]{0,12}$", re.IGNORECASE)


def printed_problems(answer: str) -> List[str]:
    """v79 (F3): what the parere must not show -- a registry label («G1»), the PDF's stamp, a ruling's citation without
    its head («Sez. 5, n. 12/2025» with no «Cass.» before it)."""
    out = [f"etichetta {m.group(0)}" for m in _LABEL_RE.finditer(answer or "")]
    out += ["timbro «copia non ufficiale»" for _ in STAMP_RE.finditer(answer or "")]
    for m in _HEADLESS_RE.finditer(answer or ""):
        before = (answer or "")[max(0, m.start() - 45):m.start()]
        if "Cass." not in before and not _ACT_BEFORE_RE.search(before):
            out.append(f"citazione senza la testa: «{(answer or '')[m.start():m.end() + 10].strip()}»")
    return out


def sentences(text: str) -> List[str]:
    from . import sentence_split
    return [s for s in (_plain(x) for x in sentence_split.split(strip_stamp(text or ""), _SPLIT_RE)) if s]


def candidate_sentences(text: str, elements: Sequence[Dict[str, Any]], article: str) -> List[str]:
    """The SENTENCES_PER_RULING sentences of a citing section nearest to the elements (noun overlap; naming the article
    counts one more), in their order in the text; too short or too long ones never."""
    number = re.escape(str(article).split(" ")[0])
    named = re.compile(rf"\bart(?:icol[oi]|t?\.)\s*{number}\b", re.I)
    lo, hi = SENTENCE_CHARS
    scored = [(overlap(s, elements) + (1 if named.search(s) else 0), i, s) for i, s in enumerate(sentences(text))
              if lo <= len(s) <= hi and starts_clean(s) and quotable(s)        # v79 (F2): never mid-reference
              and any(element_noun_hit(s, e) for e in elements)]
    best = sorted(scored, key=lambda x: (-x[0], x[1]))[:SENTENCES_PER_RULING]
    return [s for _, _, s in sorted((x for x in best if x[0] > 0), key=lambda x: x[1])]


def link_material(short: Sequence[Dict[str, Any]], case: str = "") -> str:
    parts = [f"CASO (per orientarti, non per scegliere le frasi):\n{_plain(case)[:1500]}"] if case else []
    for i, r in enumerate(short, 1):
        r["_pid"] = f"P{i}"
        els = "\n".join(f"  [{e['id']}] {_plain(e.get('text'))[:200]}" for e in r["elements"])
        frasi = "\n".join(f"  [P{i}.{k}] {s}" for k, s in enumerate(r["candidates"], 1))
        parts.append(f"PRONUNCIA [P{i}] — art. {r['article']}\nELEMENTI:\n{els or '  (nessuno)'}\nFRASI:\n"
                     f"{frasi or '  (nessuna)'}")
    return "\n\n".join(parts)


def default_call(messages: Sequence[Any], max_tokens: int) -> Any:
    from . import defensive_generation as dg
    return dg._chat_json(list(messages), max_tokens=max_tokens, temperature=0)


def link(short: List[Dict[str, Any]], case: str = "", call: Optional[Callable] = None) -> Dict[str, Any]:
    """The linking call over the shortlist, its answer applied in place: r["passage"] (our sentence, verbatim),
    r["element_id"], r["pertinenza"], r["verso"]. {"ok", "seconds", "error"}."""
    from langchain_core.messages import HumanMessage, SystemMessage
    if not short:
        return {"ok": True, "seconds": 0.0}
    messages = [SystemMessage(content=_LINK_SYSTEM), HumanMessage(content=link_material(short, case))]
    t0 = time.time()
    try:
        raw = (call or default_call)(messages, LINK_TOKENS)
    except Exception as exc:        # noqa: BLE001 -- no call: ranked by overlap, estremi only, every point to verify
        return {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}"[:200]}
    raw = raw if isinstance(raw, dict) else {}
    for r in short:
        got = raw.get(r["_pid"]) if isinstance(raw.get(r["_pid"]), dict) else {}
        sid = str(got.get("frase") or "").strip().strip("[]")
        m = re.fullmatch(rf"{r['_pid']}\.(\d+)", sid)
        k = int(m.group(1)) if m else 0
        r["passage"] = r["candidates"][k - 1] if 1 <= k <= len(r["candidates"]) else ""
        ids = {e["id"] for e in r["elements"]}
        el = str(got.get("elemento") or "").strip().strip("[]").upper()
        r["element_id"] = el if el in ids else (r["elements"][0]["id"] if r["passage"] and len(ids) == 1 else "")
        graded_on = next((e for e in r["elements"] if e["id"] == r["element_id"]), None)
        if r["passage"] and (graded_on is None or not quotable(r["passage"], graded_on)):
            r["passage"], r["element_id"] = "", ""          # v78: no rule on that element -- the estremi alone
        try:
            r["pertinenza"] = max(0, min(2, int(got.get("pertinenza"))))
        except (TypeError, ValueError):
            r["pertinenza"] = None
        verso = _plain(got.get("verso")).lower()
        r["verso"] = verso if verso in (FOR, AGAINST, NEUTRAL) else NEUTRAL
    return {"ok": bool(raw), "seconds": round(time.time() - t0, 1)}


# --- 5. the orientations -----------------------------------------------------------------------------------------------------

def orientation(element: Dict[str, Any], kept: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One controverted point: its kept passages on the element, counted for and against its integration."""
    on = [r for r in kept if r.get("element_id") == element["id"] and r.get("passage")]
    n_for = sum(1 for r in on if r.get("verso") == FOR)
    n_against = sum(1 for r in on if r.get("verso") == AGAINST)
    tot = n_for + n_against
    out = {"element_id": element["id"], "article": element["article"], "element": _plain(element.get("text")),
           "for": n_for, "against": n_against, "rulings": [r["estremi"] for r in on]}
    if tot >= MIN_ORIENTATION and n_for != n_against:
        side = FOR if n_for > n_against else AGAINST
        out.update(status="prevalente", side=side)
    else:
        out.update(status="da_verificare")
    return out


def orientation_line(o: Dict[str, Any]) -> str:
    head = f"- «{o['element'][:160]}» (art. {o['article']}): "
    if o["status"] == "prevalente":
        n, m = (o["for"], o["against"]) if o["side"] == FOR else (o["against"], o["for"])
        other = AGAINST if o["side"] == FOR else FOR
        tot = o["for"] + o["against"]
        text = (PREVALENT.format(side=SIDES[o["side"]], n=n, tot=tot, other=SIDES[other], m=m) if m
                else UNANIMOUS.format(side=SIDES[o["side"]], n=n, tot=tot))
        return head + text + "."
    listed = "; ".join(o["rulings"])
    return head + TO_VERIFY + (f" (pronunce: {listed})" if listed else " (nessuna pronuncia del corpus sul punto)") + "."


# --- the answer's registry -----------------------------------------------------------------------------------------------------

def _live(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    from . import lawyer_method
    return [r for r in lawyer_method._live(rec) if r.get("id")]


def controverted(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The elements a candidate thesis contests, each with its article: [{id, text, article}], in the record's order."""
    from . import lawyer_method
    records = ctx.get("records") or {}
    cands = ctx.get("candidates") or lawyer_method.thesis_candidates(ctx)
    wanted = {i for c in cands for i in c.get("contesta") or []}
    return [{**r, "article": label} for label, rec in records.items() for r in _live(rec) if r["id"] in wanted]


def build(ctx: Dict[str, Any], rows: Sequence[Dict[str, Any]], session, call: Optional[Callable] = None,
          landmarks: Optional[Dict[str, List[Dict[str, str]]]] = None) -> Dict[str, Any]:
    """The answer's rulings record: {"kept": [...], "orientations": [...], "landmarks": {...}, "log": {...}}, logged as
    rulings_for_record. rows: the answer's article rows (each record's (doc id, number) is read from them)."""
    from .topic_groups import unit_key
    from .topic_step import _code
    _key = lambda ref: (str(ref[0]), _code(ref[1]))               # interactive_flow._key, without its imports
    _parse_label = lambda label: tuple(str(label).partition(" ")[::2])
    records = ctx.get("records") or {}
    disputed = controverted(ctx)
    by_label_rows = {}
    for row in rows or []:
        if row.get("_article") and all(unit_key(row)):
            by_label_rows.setdefault(_key(row["_article"]), row)
    log: Dict[str, Any] = {"articles": {}, "estremi": {"parsed": 0, "unparsed": 0, "section_unmapped": 0},
                           "passages": {"quoted": 0, "none": 0}, "orientation": {"computed": 0, "da_verificare": 0},
                           "sections_confirmed": bool(sections_map().get("confirmed"))}
    t0 = time.time()
    shortlist: Dict[str, List[Dict[str, Any]]] = {}
    for label, rec in records.items():
        row = by_label_rows.get(_key(_parse_label(label)))
        art_log = log["articles"].setdefault(label, {"found": 0, "linked": 0, "unparsed": 0, "off_rubrica": 0,
                                                            "graded": 0, "kept": 0})
        if row is None:
            art_log["unread"] = True
            continue
        doc, num = unit_key(row)
        found = fetch(session, doc, num)
        art_log["found"] = len(found)
        elements = [{**r, "article": label} for r in _live(rec)]
        mine = [e for e in disputed if e["article"] == label] or elements
        cands = []
        for g in found.values():
            if not g["linked"]:
                continue
            art_log["linked"] += 1
            p = parse_estremi(g["file_name"])
            if p is None:
                art_log["unparsed"] += 1
                log["estremi"]["unparsed"] += 1
                continue
            log["estremi"]["parsed"] += 1
            if not on_rubrica(g["abstract"], str(rec.get("rubrica") or "")):
                art_log["off_rubrica"] += 1          # v78: its abstract is not on the article: dropped for it
                continue
            if not chamber(p):
                log["estremi"]["section_unmapped"] += 1
            cands.append({"id": g["id"], "article": label, "estremi": render_estremi(p, g["date_enacted"]),
                          "su": sezioni_unite(p), "date": ruling_date(g["date_enacted"], p["year"]),
                          "score": overlap(g["abstract"], mine), "sections": g["sections"], "elements": mine,
                          "parsed": p})
        shortlist[label] = sorted(cands, key=_order_key)[:GRADED_PER_ARTICLE]
    short = round_robin(shortlist, SHORTLIST_MAX)
    body = texts(session, [s for r in short for s in r["sections"]])
    for r in short:
        r["candidates"] = candidate_sentences(" ".join(body.get(s) or "" for s in r["sections"]), r["elements"],
                                              r["article"])
    call_log = link(short, ctx.get("case") or "", call)
    log["link_call"] = call_log
    by_article: Dict[str, List[Dict[str, Any]]] = {}
    dropped = []
    for r in short:
        if call_log.get("ok") and r.get("pertinenza") == 0:
            dropped.append(r["estremi"])
            continue
        r["score"] = (r.get("score") or 0) + 3 * (r.get("pertinenza") or 0)
        if not call_log.get("ok"):
            r.update(passage="", element_id="", verso=NEUTRAL)
        by_article.setdefault(r["article"], []).append(r)
    graded = {k: sorted(v, key=_order_key) for k, v in by_article.items()}         # the orientation's
    for k, v in graded.items():
        log["articles"][k]["graded"] = len(v)
    kept = round_robin({k: graded[k][:PER_ARTICLE] for k in records if k in graded}, PER_ANSWER)     # the printed
    for r in kept:
        log["articles"][r["article"]]["kept"] += 1
        log["passages"]["quoted" if r.get("passage") else "none"] += 1
    orientations = [orientation(e, [r for v in graded.values() for r in v]) for e in disputed]
    for o in orientations:
        log["orientation"]["computed" if o["status"] == "prevalente" else "da_verificare"] += 1
    log["dropped_unrelated"] = dropped
    out = {"kept": [{k: r.get(k) for k in ("id", "article", "estremi", "su", "element_id", "passage", "verso",
                                           "pertinenza")} | {"date": r["date"].isoformat() if r.get("date") else None,
                                                             "element": next((_plain(e.get("text")) for e in r["elements"]
                                                                              if e["id"] == r.get("element_id")), "")}
                    for r in kept],
           "orientations": orientations, "landmarks": landmarks or {}, "log": log}
    _log("rulings_for_record", log, (time.time() - t0) * 1000)
    return out


# --- 6. what is printed, and what the calls receive ----------------------------------------------------------------------------

def ruling_line(r: Dict[str, Any]) -> str:
    return f"- {r['estremi']}" + (f", {PASSAGE_LABEL}: «{printed(r['passage'])}»" if r.get("passage") else "") + "."


def landmark_lines(landmarks: Dict[str, List[Dict[str, str]]]) -> List[str]:
    return [f"- art. {label}: {x['estremi']} — {x['principio']}" for label, items in (landmarks or {}).items()
            for x in items]


def section_text(record: Dict[str, Any], points: Optional[List[str]] = None) -> str:
    """§5, rendered by the code: the rulings by article, the orientations, the lawyer's pronunce di riferimento, the
    fixed corpus line. points: «Orientamento sul punto: …» for each point to verify (for §10)."""
    parts = []
    kept = record.get("kept") or []
    if kept:
        lines = [SECTION_TITLES["rulings"]]
        for label in dict.fromkeys(r["article"] for r in kept):
            mine = [r for r in kept if r["article"] == label]
            lines.append(f"\nArt. {label}:")
            lines += [ruling_line(r) for r in mine]
        parts.append("\n".join(lines))
    else:
        parts.append(NO_RULINGS_LINE)
    orients = record.get("orientations") or []
    if orients:
        parts.append(SECTION_TITLES["orientations"] + "\n\n" + "\n".join(orientation_line(o) for o in orients))
        if points is not None:
            points += [f"Orientamento sul punto: {o['element'][:120]} (art. {o['article']})" for o in orients
                       if o["status"] != "prevalente"]
    marks = landmark_lines(record.get("landmarks") or {})
    if marks:
        parts.append(SECTION_TITLES["landmarks"] + "\n\n" + "\n".join(marks))
    parts.append(CORPUS_LINE)
    return "\n\n".join(parts)


def citable(record: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The rulings the parere may cite: the registry's and the lawyer's -- {id, label, massima} as lawyer_method's ruling
    check reads them."""
    if not record:
        return []
    out = [{"id": r["id"], "label": r["estremi"], "massima": printed(r.get("passage") or ""), "article": r["article"],
            "element": r.get("element") or "" if r.get("passage") else ""} for r in record.get("kept") or []]
    shown = {r["estremi"] for r in record.get("kept") or []}
    listed = {x: o for o in record.get("orientations") or [] for x in o.get("rulings") or [] if x not in shown}
    out += [{"id": f"orientamento:{i}", "label": x, "massima": "", "article": o["article"], "element": o["element"]}
            for i, (x, o) in enumerate(listed.items())]          # graded on the point, listed in §5
    out += [{"id": f"consulente:{label}:{i}", "label": x["estremi"], "massima": x["principio"], "article": label,
             "element": ""}
            for label, items in (record.get("landmarks") or {}).items() for i, x in enumerate(items)]
    return out


def registry_block(record: Optional[Dict[str, Any]]) -> str:
    """The material of the thesis call, the objections call and the norms writer ("" without a record)."""
    if not record:
        return ""
    lines = []
    for i, r in enumerate(record.get("kept") or [], 1):
        on = f", elemento [{r['element_id']}] «{_plain(r.get('element'))[:120]}»" if r.get("element_id") else ""
        quote = f": {PASSAGE_LABEL} «{printed(r['passage'])}» (verso l'integrazione: {r.get('verso')})" if r.get("passage") \
            else ": nessun passaggio sull'elemento (citabile solo con gli estremi)"
        lines.append(f"- [G{i}] {r['estremi']} — art. {r['article']}{on}{quote}")
    orients = [orientation_line(o) for o in record.get("orientations") or []]
    marks = landmark_lines(record.get("landmarks") or {})
    out = [f"{REGISTRY_TITLE} (le sole pronunce citabili, con gli estremi indicati):\n" + ("\n".join(lines) or "(nessuna)")]
    if orients:
        out.append("ORIENTAMENTI CALCOLATI SUI PUNTI CONTROVERSI:\n" + "\n".join(orients))
    if marks:
        out.append("PRONUNCE DI RIFERIMENTO INDICATE DAL CONSULENTE:\n" + "\n".join(marks))
    return "\n\n".join(out)


REALISM_RULE = ("\n\nREGISTRO DELLE PRONUNCE: un passaggio che enuncia un principio sull'elemento contestato può cambiare "
                "il realismo di una tesi: un orientamento prevalente contro la contestazione (l'elemento ritenuto "
                "integrato) porta il realismo della tesi che lo contesta a basso.")


def cap_realism(plan: Dict[str, Any], record: Optional[Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None
                ) -> Dict[str, Any]:
    """A prevailing line for the integration of an element: every thesis contesting it «basso» (in place, logged)."""
    from . import lawyer_method
    against = {o["element_id"]: o for o in (record or {}).get("orientations") or []
               if o.get("status") == "prevalente" and o.get("side") == FOR}
    for t in plan.get("soluzioni") or []:
        hit = [i for i in t.get("contesta") or [] if i in against]
        if hit and t.get("realismo") != lawyer_method.BASSO:
            if log is not None:
                log.append({"tesi": str(t.get("tesi"))[:120], "was": t.get("realismo"), "elements": hit})
            t["realismo"] = lawyer_method.BASSO
    return plan


# --- the nightly's check, the structure check's reading of §5 --------------------------------------------------------------

def quoted(section: str) -> int:
    """How many rulings §5 prints with a quoted passage."""
    return len(re.findall(rf"^- Cass\. [^\n]*, {PASSAGE_LABEL}: «", section or "", re.M))


def section_lines_ok(section: str) -> List[str]:
    """The §5 lines the code would not print (a written orientation): [] when every line is the code's."""
    bad = []
    for ln in (section or "").split("\n"):
        s = ln.strip()
        if not s or s in SECTION_TITLES.values() or s in (CORPUS_LINE, NO_RULINGS_LINE) or re.match(r"^Art\. .+:$", s):
            continue
        if re.match(r"^- Cass\. (?:pen|civ)\., ", s) or re.match(r"^- «.+» \(art\. [^)]+\): ", s) or \
                re.match(r"^- art\. [^:]+: .+ — .+", s):
            continue
        bad.append(s[:120])
    return bad
