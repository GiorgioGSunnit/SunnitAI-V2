"""TOPIC_STEP (default false): after retrieval, the case's topics, one per institution the case names, each anchored on the
article whose rubrica is that institution. Phase 1: built and logged only (case_analysis_topics, analysis["topics"]); no
plan or section reads them yet. scripts/diagnose_long.py --topics-only runs the facts, the frame and retrieval, builds
them and scores their recall, with no plan and no writing.

build():
  1. the institutions, each once (by its normalised name), in this order of sources: the fired fact rules'
     institutions; the rules that fire once the facts' unknown yes/no fields are taken as yes (a work's "visible",
     "before_separation", an access's "landlocked": "regola da verificare", the conditions to verify among its facts --
     Leonardo v52: the road's existence at the separation left blank, destinazione_padre_famiglia not fired); the fired
     regimes (REGIME_CANDIDATES, by their action's name); the frame's central issues; the analysis questioni's
     institutions.
  2. the anchor: the article whose rubrica is the institution's name, normalised (lower case, no accents, punctuation
     or articles and prepositions) -- the same words ("passaggio coattivo" = "Passaggio coattivo.") or the name ending
     with all of them ("servitù per destinazione del padre di famiglia" ends with "Destinazione del padre di
     famiglia.", the words left out only category words: "divisione di un fondo comune" is no "Fondo comune") -- in the
     case's allowed codes (vote_pool.allowed_codes); fallback, the regime table (the regime's first article, a regime
     whose trigger the name holds), then the strongest two-signal pool article (_multi_signal) the institution resolved
     to (analysis["institution_refs"], the fired rules' rule_institutions).
  3. its articles: the anchor, then, alternately, its Sezione's articles nearest by number and its REFERS_TO neighbours
     (out and in, within the allowed codes) cited by the fewest articles, a hub (more than HUB_INBOUND citing articles)
     never -- at most
     ARTICLES_MAX; then every two-signal pool article of the anchor's Capo.
  4. the computed facts it touches: the terms, tools and fired rules whose articles are among its articles; a rule to
     verify, its conditions to verify; the parties to sue for a real action.
  5. a topic sharing more than half of its articles (of the smaller set) with a stronger one is merged into it, its
     facts kept; the strongest TOPICS_MAX kept: an exact anchor before a regime's, a regime's before the pool's, then by
     source as in 1. The title is the institution's name.

v54 (round-1 quality, the h3 simulation): a generic rubrica (GENERIC_RUBRICHE, or one of at most GENERIC_SHORT_WORDS words
found more than GENERIC_SHORT_COUNT times across the codes) is never an anchor, and a name of fewer than NAME_MIN_WORDS
content words is anchored by rubrica only from a rule, a term or a regime; every computed term DA VERIFICARE is a topic on
its own article and its Sezione (source "termine"), a topic anchored on a DECORSO term's article is a fact, never a topic;
a topic no rule, term or regime supports (supported) and one whose every action the action gate gives to the other side
go to `others` (the "altri profili") -- unless no topic at all is supported: then the unsupported ones are offered
(NO_SUPPORT_OFFERED, rule (1) still applied to their anchors); the gated actions leave the titles (without_gated). Open
terms: DA VERIFICARE, INCERTO and NON VERIFICABILE.

v55 (the v54-interactive review): a regime (and a topic_tables entry, TOPIC_TABLES) anchors on its own first article
before any rubrica match, its own articles first among the topic's; a rubrica or pool anchor in the civil code lies within
the institution's areas' book (AREA_RANGES: its own area and the case's), a pool anchor's rubrica shares a content word
of the name (shares_name) -- v54 l4's 585 c.c., n2's 270-ter c.p.; two topics one of whose titles' content words are all
in the other's are one (same_name_merged); a fired rule marked unstated (interactive_flow.mark_unstated) is a rule to
verify; with only table topics supported, the frame's are offered next to them.

links: where the units and the links are read -- GraphLinks (the corpus, read-only: titolo_articolo, Unit HAS_ARTICLE,
REFERS_TO) in the pipeline, CsvLinks (scripts/article_links.py's units.csv and refers_to.csv, a rubrica table) offline.
"""

import copy
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import feature_flags

FLAG = "TOPIC_STEP"
TOPICS_MIN, TOPICS_MAX = 3, 6
ARTICLES_MAX = 8
HUB_INBOUND = 40            # REFERS_TO into an article from more articles than this: a hub (v52 corpus: p99 is 42)
DUPLICATE_SHARE = 0.5
SPECIFIC_SHARE = 0.2        # a word in more of the pool's rubriche than this is not specific
_SOURCE_ORDER = ("regola", "regola da verificare", "termine", "regime", "tabella", "frame", "questione")
_METHOD_ORDER = ("exact", "term", "suffix", "rule", "regime", "table", "pool")
# v54 (round-1 quality): the sources that support a topic by the computed facts; a frame's or a questione's institution
# with none of them (only its rubrica or pool match) is no topic but one of the "altri profili"
SUPPORTED_SOURCES = ("regola", "regola da verificare", "termine", "regime", "tabella")
OPEN_TERM_STATUS = ("DA VERIFICARE", "INCERTO", "NON VERIFICABILE")     # v54 decision: all three are topics
DECIDED_TERM_STATUS = ("DECORSO",)
# a rubrica that names a category, not an institution: never an anchor (normalised by tokens(); the PO's list)
GENERIC_RUBRICHE = {tuple(tokens_) for tokens_ in (
    ["prescrizione"], ["decadenza"], ["nullita"], ["annullabilita"], ["definizione"], ["definizioni"],
    ["disposizioni", "generali"], ["disposizione", "generale"], ["norme", "applicabili"], ["norma", "applicabile"],
    ["rinvio"], ["oggetto"], ["ambito", "applicazione"], ["applicazione"], ["contenuto"], ["effetti"], ["forma"],
    ["termini"], ["termine"], ["competenza"], ["sanzioni"], ["abrogazioni"], ["entrata", "vigore"],
    ["disposizioni", "transitorie"], ["disposizioni", "finali"], ["casi", "particolari"], ["eccezioni"], ["limiti"])}
GENERIC_SHORT_WORDS = 2         # a rubrica this short ...
GENERIC_SHORT_COUNT = 3         # ... found more often than this across the codes is generic too
UNSUPPORTED_WHY = "nessuna regola, termine o regime dei fatti"
NO_SUPPORT_OFFERED = "nessun argomento sostenuto dai fatti calcolati: offerto dal quadro del caso"
NAME_MIN_WORDS = 2              # content words an institution needs to be anchored by its rubrica alone
REGIME_NAMES = {"usucapione": "usucapione", "passaggio_coattivo": "passaggio coattivo",
                "responsabilita_contrattuale": "responsabilità contrattuale",
                "impossibilita_sopravvenuta": "impossibilità sopravvenuta della prestazione",
                "procedimento_amministrativo": "provvedimento amministrativo", "ottemperanza": "giudizio di ottemperanza"}
# v55: the parts of the civil code an area's institutions anchor in (c.c. article numbers), Libro VI (trascrizione, prove,
# prescrizione: CIVIL_COMMON) for every civil area; an anchor in c.c. outside every range of the institution's areas
# (its own and the case's) is none -- v54 l4: "Separazione de facto" on 585 c.c. (successioni) in a family case
AREA_RANGES = {"famiglia": [(1, 455)], "successioni": [(456, 809)], "diritti_reali": [(810, 1172)],
               "contratti": [(1173, 2059)], "locazione": [(1173, 2059)], "consumo": [(1173, 2059)],
               "responsabilita_civile": [(1173, 2059), (2229, 2238)], "lavoro": [(1173, 1229), (2060, 2246)],
               "societario": [(2247, 2642)], "societa": [(2247, 2642)], "impresa": [(2082, 2642)]}
CIVIL_COMMON = [(2643, 2969)]
_UNKNOWN_YES = {"works_or_signs": ("visible", "before_separation"), "access": ("landlocked",)}
_STOPWORDS = {"il", "lo", "la", "i", "gli", "le", "l", "un", "una", "uno", "di", "del", "dello", "della", "dei", "degli",
              "delle", "da", "dal", "dallo", "dalla", "dai", "dalle", "in", "nel", "nello", "nella", "nei", "nelle", "a",
              "al", "allo", "alla", "ai", "agli", "alle", "e", "ed", "o", "per", "con", "su", "sul", "sulla", "tra", "fra",
              "d", "art", "c", "p"}
_STEM_STOP = {"della", "delle", "dello", "degli", "nella", "nelle", "questione", "centrale", "diritto", "diritti", "azione",
              "azioni", "tutela", "parte", "parti", "norma", "norme", "codice", "civile", "penale"}


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- names ----------------------------------------------------------------------------------------------------------

def tokens(text: Any) -> List[str]:
    """A name or a rubrica normalised: lower case, no accents, no punctuation, no articles or prepositions."""
    plain = unicodedata.normalize("NFKD", str(text or "").lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", plain) if w not in _STOPWORDS]


# the words a name may carry before its rubrica: a category, not another institution ("servitù per destinazione del
# padre di famiglia" is "Destinazione del padre di famiglia"; "divisione di un fondo comune" is no "Fondo comune")
CATEGORY_WORDS = {"servitu", "azione", "azioni", "diritto", "diritti", "reato", "delitto", "contravvenzione", "istituto",
                  "obbligo", "obblighi", "contratto", "principio", "regime", "disciplina", "fattispecie", "ipotesi",
                  "questione", "profilo", "tutela", "rimedio", "norma", "configurabilita", "sussistenza"}


def name_match(name: str, rubrica: str) -> str:
    """"exact" (the same words, in any order), "suffix" (the name ends with all of the rubrica's, two at least, after category words
    only: CATEGORY_WORDS) or ""."""
    a, b = tokens(name), tokens(rubrica)
    if not a or not b:
        return ""
    if a == b or (len(a) == len(b) and sorted(a) == sorted(b)):        # "Legittima difesa" = "Difesa legittima"
        return "exact"
    if len(b) >= 2 and len(a) > len(b) and a[-len(b):] == b and set(a[:-len(b)]) <= CATEGORY_WORDS:
        return "suffix"
    return ""


def _stems(text: Any) -> Set[str]:
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _STEM_STOP}


def _title(name: str) -> str:
    name = " ".join(str(name or "").split()).strip(" .;:")
    return name[:1].upper() + name[1:]


def _code(code: Any) -> str:
    from .plan_then_write import _code_key
    text = " ".join(str(code or "").split())
    return _code_key(text) if text else ""


def _label(ref: Sequence[str]) -> str:
    return f"{ref[0]} {ref[1]}"


def _number(article: str) -> float:
    m = re.match(r"(\d+)", str(article))
    return float(m.group(1)) if m else 0.0


# --- the links ------------------------------------------------------------------------------------------------------

class CsvLinks:
    """Offline: units [(code, level, unit name, article)], refers [(from label, to label)], rubriche {(number, code):
    rubrica}."""

    def __init__(self, units: Iterable[Sequence[str]], refers: Iterable[Sequence[str]], rubriche: Dict[tuple, str]):
        self.unit_of: Dict[Tuple[str, str, str], str] = {}
        self.members: Dict[Tuple[str, str, str], List[str]] = {}
        for code, level, unit, article in units:
            key = (code, level.upper(), unit)
            self.unit_of[(code, level.upper(), str(article))] = unit
            self.members.setdefault(key, []).append(str(article))
        pairs = {(f, t) for f, t in refers if f != t}
        self.out: Dict[str, List[str]] = {}
        self.into: Dict[str, List[str]] = {}
        for f, t in pairs:
            self.out.setdefault(f, []).append(t)
            self.into.setdefault(t, []).append(f)
        self.inbound = Counter(t for _, t in pairs)
        self.rubriche = {(str(n), _code(c)): r for (n, c), r in rubriche.items() if r}
        self.counts = Counter(tuple(tokens(r)) for r in self.rubriche.values())

    def rubrica_count(self, rubrica: str) -> int:
        """How many articles of the codes have this rubrica (normalised)."""
        return self.counts.get(tuple(tokens(rubrica)), 0)

    def rubrica_matches(self, name: str, codes: Optional[set]) -> List[Tuple[tuple, str, str]]:
        out = []
        for ref, rubrica in self.rubriche.items():
            if codes and ref[1] not in codes:
                continue
            how = name_match(name, rubrica)
            if how:
                out.append((ref, rubrica, how))
        return out

    def unit_name(self, ref: Sequence[str], level: str) -> str:
        return self.unit_of.get((_code(ref[1]), level.upper(), str(ref[0]))) or ""

    def unit(self, ref: Sequence[str], level: str) -> List[tuple]:
        code = _code(ref[1])
        unit = self.unit_of.get((code, level, str(ref[0])))
        return [(a, code) for a in self.members.get((code, level, unit), [])] if unit else []

    def refers(self, ref: Sequence[str]) -> List[Tuple[tuple, int]]:
        label = _label((ref[0], _code(ref[1])))
        near = dict.fromkeys(self.out.get(label, []) + self.into.get(label, []))
        return [(tuple(lab.split(" ", 1)), self.inbound[lab]) for lab in near if " " in lab]

    def refers_out(self, ref: Sequence[str]) -> List[Tuple[tuple, int]]:
        """v57: the articles this one cites (outbound REFERS_TO only), each with how many articles cite it."""
        label = _label((ref[0], _code(ref[1])))
        return [(tuple(lab.split(" ", 1)), self.inbound[lab]) for lab in dict.fromkeys(self.out.get(label, []))
                if " " in lab]

    def rubrica_of(self, ref: Sequence[str]) -> str:
        return self.rubriche.get((str(ref[0]), _code(ref[1])), "")


class GraphLinks:
    """The corpus, read-only: rubriche (Section.titolo_articolo), Unit HAS_ARTICLE, REFERS_TO."""

    _RUBRICA = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                "AND toLower(s.titolo_articolo) CONTAINS $word RETURN d.name AS doc, s.name AS section, "
                "s.titolo_articolo AS rubrica LIMIT 300")
    _ANCHOR = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.name = $num OR s.name STARTS WITH $num + '.' "
               "OR s.name STARTS WITH $num + '_' ")
    _UNIT = _ANCHOR + ("MATCH (u:Unit {level: $level})-[:HAS_ARTICLE]->(s) MATCH (u)-[:HAS_ARTICLE]->(t:Section)"
                       "<-[:CONTAINS]-(td:Document) RETURN DISTINCT td.name AS doc, t.name AS section")
    _REFERS = _ANCHOR + ("MATCH (s)-[:REFERS_TO]-(t:Section)<-[:CONTAINS]-(td:Document) "
                         "OPTIONAL MATCH (src:Section)-[:REFERS_TO]->(t) "
                         "RETURN td.name AS doc, t.name AS section, count(DISTINCT src) AS inbound")

    _COUNT = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE s.titolo_articolo IS NOT NULL "
              "AND toLower(s.titolo_articolo) STARTS WITH $first RETURN DISTINCT d.id AS doc, s.name AS section, "
              "s.titolo_articolo AS rubrica LIMIT 5000")

    def __init__(self, session, documents: List[Dict[str, Any]]):
        self.session = session
        self.documents = documents or []
        self._counts: Dict[Tuple[str, ...], int] = {}

    def rubrica_count(self, rubrica: str) -> int:
        """How many articles across the corpus have this rubrica (normalised), read once per rubrica."""
        key = tuple(tokens(rubrica))
        if key not in self._counts:
            first = (re.findall(r"[\wà-ù']+", str(rubrica or "").lower()) or [""])[0]
            rows = self.session.run(self._COUNT, first=first).data() if first else []
            self._counts[key] = len({(r["doc"], r["section"]) for r in rows if tuple(tokens(r["rubrica"])) == key})
        return self._counts[key]

    def _docs(self, codes: Optional[set]) -> List[str]:
        return [d["id"] for d in self.documents if d.get("id") and (not codes or _code(d.get("name")) in codes)]

    def _doc_of(self, code: str) -> Optional[str]:
        return next((d["id"] for d in self.documents if d.get("id") and _code(d.get("name")) == _code(code)), None)

    @staticmethod
    def _ref(doc: str, section: str) -> tuple:
        return (str(section or "").split(".")[0].split("_")[0], _code(doc))

    def rubrica_matches(self, name: str, codes: Optional[set]) -> List[Tuple[tuple, str, str]]:
        words = [w for w in re.findall(r"[\wà-ù']+", str(name or "").lower()) if w not in _STOPWORDS]
        if not words:
            return []
        rows = self.session.run(self._RUBRICA, docs=self._docs(codes), word=words[-1]).data()
        out = []
        for r in rows:
            how = name_match(name, r["rubrica"])
            if how:
                out.append((self._ref(r["doc"], r["section"]), r["rubrica"], how))
        return out

    _UNIT_NAME = _ANCHOR + "MATCH (u:Unit {level: $level})-[:HAS_ARTICLE]->(s) RETURN u.name AS name LIMIT 1"

    def unit_name(self, ref: Sequence[str], level: str) -> str:
        doc = self._doc_of(ref[1])
        if not doc:
            return ""
        rows = self.session.run(self._UNIT_NAME, doc=doc, num=str(ref[0]), level=level).data()
        return str(rows[0]["name"] or "") if rows else ""

    def unit(self, ref: Sequence[str], level: str) -> List[tuple]:
        doc = self._doc_of(ref[1])
        if not doc:
            return []
        rows = self.session.run(self._UNIT, doc=doc, num=str(ref[0]), level=level).data()
        return list(dict.fromkeys(self._ref(r["doc"], r["section"]) for r in rows))

    def refers(self, ref: Sequence[str]) -> List[Tuple[tuple, int]]:
        doc = self._doc_of(ref[1])
        if not doc:
            return []
        rows = self.session.run(self._REFERS, doc=doc, num=str(ref[0])).data()
        return [(self._ref(r["doc"], r["section"]), int(r["inbound"] or 0)) for r in rows]

    _REFERS_OUT = _ANCHOR + ("MATCH (s)-[:REFERS_TO]->(t:Section)<-[:CONTAINS]-(td:Document) "
                             "OPTIONAL MATCH (src:Section)-[:REFERS_TO]->(t) "
                             "RETURN td.name AS doc, t.name AS section, count(DISTINCT src) AS inbound")
    _RUBRICHE = ("UNWIND $refs AS a MATCH (d:Document {id: a.doc})-[:CONTAINS]->(s:Section) WHERE s.name = a.num OR "
                 "s.name STARTS WITH a.num + '.' OR s.name STARTS WITH a.num + '_' "
                 "RETURN a.doc AS doc, a.num AS num, s.titolo_articolo AS rubrica")

    def refers_out(self, ref: Sequence[str]) -> List[Tuple[tuple, int]]:
        """v57: the articles this one cites (outbound REFERS_TO only), each with how many articles cite it."""
        doc = self._doc_of(ref[1])
        if not doc:
            return []
        rows = self.session.run(self._REFERS_OUT, doc=doc, num=str(ref[0])).data()
        return [(self._ref(r["doc"], r["section"]), int(r["inbound"] or 0)) for r in rows]

    def rubrica_of(self, ref: Sequence[str]) -> str:
        return self.rubriche_of([ref]).get((str(ref[0]), _code(ref[1])), "")

    def rubriche_of(self, refs: Sequence[Sequence[str]]) -> Dict[tuple, str]:
        """{(number, code): rubrica} of refs, read once each."""
        cache = self.__dict__.setdefault("_rubriche", {})
        wanted = [(str(r[0]), _code(r[1])) for r in refs if (str(r[0]), _code(r[1])) not in cache]
        args = [{"doc": self._doc_of(c), "num": n} for n, c in wanted if self._doc_of(c)]
        if args:
            by_doc = {d["id"]: _code(d.get("name")) for d in self.documents if d.get("id")}
            for row in self.session.run(self._RUBRICHE, refs=args).data():
                cache.setdefault((str(row["num"]), by_doc.get(row["doc"], "")), str(row["rubrica"] or ""))
        for key in wanted:
            cache.setdefault(key, "")
        return {(str(r[0]), _code(r[1])): cache.get((str(r[0]), _code(r[1])), "") for r in refs}


# --- the institutions -----------------------------------------------------------------------------------------------

def _relaxed(facts: Dict[str, Any]) -> Dict[str, Any]:
    """The facts with every unknown yes/no field (_UNKNOWN_YES) taken as yes."""
    out = copy.deepcopy(facts or {})
    for key, fields in _UNKNOWN_YES.items():
        for item in out.get(key) or []:
            if isinstance(item, dict):
                for field in fields:
                    if not str(item.get(field) or "").strip():
                        item[field] = "sì"
    return out


def rules_to_verify(signals: Dict[str, Any], areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The fact rules not fired that fire once the unknown yes/no fields are taken as yes: [{id, institutions, codice,
    to_verify: [the conditions that failed on the facts as they are]}]."""
    from . import fact_signals
    facts = (signals or {}).get("facts")
    if not facts:
        return []
    fired = {r["id"] for r in signals.get("fired") or []}
    terms = signals.get("terms") or []
    try:
        relaxed = fact_signals.fire_rules(_relaxed(facts), terms, areas=areas)
        explained = {r["id"]: r for r in fact_signals.explain_rules(facts, terms, areas=areas)}
    except Exception as exc:
        fact_signals._log_exception("topic_step.rules_to_verify", exc)
        return []
    out = []
    for rule in relaxed:
        if rule["id"] in fired:
            continue
        failing = [c["text"] for c in (explained.get(rule["id"]) or {}).get("conditions") or [] if not c["matched"]]
        out.append({"id": rule["id"], "institutions": list(rule.get("institutions") or []), "codice": rule.get("codice"),
                    "to_verify": failing})
    return out


_DURATION_RE = r"\b(?:da|per)\s+(?:più di |oltre |almeno |circa )?(?:\d+|due|tre|dieci|venti|trenta)\s+anni|\bfin dal\b|\bsin dal\b"


def possession_over_time(facts: Dict[str, Any]) -> bool:
    """v58 (v57-l3: an owner's flat made an usucapione topic): a possession or use over time of what is not the
    possessor's own -- a possession whose holder is no owner of the case (property_history, a party «proprietario»), of
    a way or of a thing named as another party's, or one with a start or a stated duration."""
    from . import fact_signals
    possession = [p for p in (facts or {}).get("possession") or [] if isinstance(p, dict) and p.get("who")]
    owners = {fact_signals._low(h.get("owner")) for h in (facts or {}).get("property_history") or [] if h.get("owner")}
    owners |= {fact_signals._low(p.get("name")) for p in (facts or {}).get("parties") or []
               if re.search(r"proprietari", fact_signals._low(p.get("role")))}
    names = [fact_signals._low(x.get("name")) for x in (facts or {}).get("parties") or [] if x.get("name")]
    for p in possession:
        who, asset = fact_signals._low(p.get("who")), fact_signals._low(f"{p.get('asset')} {p.get('use')}")
        if who not in owners and not any(o and o in who for o in owners):
            return True
        if re.search(fact_signals._WAY, asset) or any(n and n not in who and re.search(rf"\b{re.escape(n)}\b", asset)
                                                      for n in names):
            return True             # a way, or another party's thing: h3's road over the neighbours' plots
        if str(p.get("since") or "").strip() or re.search(_DURATION_RE, fact_signals._low(f"{p.get('how')} {p.get('use')}")):
            return True
    return False


def term_institutions(signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The computed terms still to verify (OPEN_TERM_STATUS), each an institution of source "termine": its name the
    term's label, its refs the term's articles."""
    from .plan_then_write import _code_key
    out = []
    over_time = possession_over_time((signals or {}).get("facts") or {})
    for term in (signals or {}).get("terms") or []:
        if term.get("status") not in OPEN_TERM_STATUS:
            continue
        if str(term.get("id") or "").startswith("usucapione") and (signals or {}).get("facts") and not over_time:
            continue                # v58: no possession over time, no usucapione topic
        ref = str(term.get("ref") or "")
        # the longer abbreviations first: "c.p.a." is no "c.p." (v54-interactive n4: 120 c.p.a. read as 120 c.p.)
        code = re.search(r"(c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?p\.\s?a\.|c\.\s?p\.|c\.\s?c\.|d\.\s?lgs\.?\s*[\d/]+)",
                         ref)
        numbers = re.findall(r"\b\d+(?:-[a-z]+)?\b", ref[:code.start()] if code else ref)
        refs = [(n, _code_key(code.group(1)) if code else "c.c.") for n in numbers]
        if refs:
            out.append({"name": str(term.get("label") or term.get("id")), "term": term.get("id"), "refs": refs,
                        "status": term.get("status")})
    return out


def decided_refs(signals: Dict[str, Any]) -> Set[tuple]:
    """The articles of the computed terms already decided against an action (DECIDED_TERM_STATUS: possessoria
    DECORSO): a fact of the case, never a topic."""
    out: Set[tuple] = set()
    for t in term_institutions({"terms": [{**t, "status": OPEN_TERM_STATUS[0]} for t in (signals or {}).get("terms") or []
                                          if t.get("status") in DECIDED_TERM_STATUS]}):
        out.update((str(n), _code(c)) for n, c in t["refs"])
    return out


def generic_rubrica(rubrica: str, links=None) -> bool:
    """A category's rubrica (GENERIC_RUBRICHE), or one of at most GENERIC_SHORT_WORDS words found more than
    GENERIC_SHORT_COUNT times across the codes."""
    key = tuple(tokens(rubrica))
    if key in GENERIC_RUBRICHE:
        return True
    if links is not None and len(key) <= GENERIC_SHORT_WORDS and hasattr(links, "rubrica_count"):
        try:
            return links.rubrica_count(rubrica) > GENERIC_SHORT_COUNT
        except Exception as exc:
            from . import fact_signals
            fact_signals._log_exception("topic_step.generic_rubrica", exc)
    return False


def content_words(name: str) -> List[str]:
    return [w for w in tokens(name) if w not in CATEGORY_WORDS]


def _content_stems(text: Any) -> Set[str]:
    return {w[:6] for w in content_words(text) if len(w) >= 4}


# v56 (h3's altri profili: "Abuso del diritto di proprietà" on 2814 c.c., «nuda proprietà»; a "prescrizione" on 1449
# c.c., «Prescrizione.»): a word every part of the codes uses never makes a rubrica the name's
GENERIC_STEMS = {w[:6] for key in GENERIC_RUBRICHE for w in key if len(w) >= 4} | {
    "propri", "diritt", "obblig", "genera", "dispos", "partic", "specia", "discip", "beni", "respon", "danni",
    "danno"}


def shares_name(name: str, rubrica: str) -> bool:
    """The rubrica holds a content word of the name (its first six letters: "coniugi" is "coniuge"), one no part of the
    codes owns (GENERIC_STEMS)."""
    return bool((_content_stems(name) & _content_stems(rubrica)) - GENERIC_STEMS)


def area_fits(ref: Sequence[str], areas: Sequence[str]) -> bool:
    """A c.c. article within the AREA_RANGES of the areas (CIVIL_COMMON for all); any other code, or no area with
    ranges, fits."""
    if _code(ref[1]) != "c.c.":
        return True
    ranges = [r for a in areas or [] for r in AREA_RANGES.get(str(a or "").strip().lower(), [])]
    if not ranges:
        return True
    number = _number(ref[0])
    return any(lo <= number <= hi for lo, hi in ranges + CIVIL_COMMON)


def institutions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{name, source, rule?, regime?, refs, to_verify}] each once by its normalised name, in the sources' order."""
    from . import fact_signals
    frame = analysis.get("frame") or {}
    signals = analysis.get("fact_signals") or {}
    areas = frame.get("areas") or signals.get("areas") or []
    found: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()

    def add(name, source, **extra):
        key = tuple(tokens(name))
        if key and key not in seen:
            seen.add(key)
            found.append({"name": " ".join(str(name).split()), "source": source, "refs": [], "to_verify": [], **extra})
    for rule in signals.get("fired") or []:
        for name in rule.get("institutions") or []:
            required = [(str(r[0]), _code(r[1])) for r in rule.get("requires") or [] if len(r) == 2]
            if rule.get("unstated"):        # v55: fired on a field the case does not state (interactive_flow.mark_unstated)
                add(name, "regola da verificare", rule=rule.get("id"), to_verify=list(rule.get("unstated_conditions") or []),
                    requires=required)
            else:
                add(name, "regola", rule=rule.get("id"), requires=required)
    for rule in rules_to_verify(signals, areas):
        for name in rule["institutions"]:
            add(name, "regola da verificare", rule=rule["id"], to_verify=rule["to_verify"])
    for term in term_institutions(signals):
        add(term["name"], "termine", term=term["term"], refs=[(str(n), _code(c)) for n, c in term["refs"]],
            to_verify=[f"{term['name']}: termine {str(term['status']).lower()}"])
    if fact_signals.regime_candidates_enabled():
        try:
            regimes = fact_signals.regime_candidates(analysis, areas)
        except Exception as exc:
            fact_signals._log_exception("topic_step.regimes", exc)
            regimes = []
        for regime in regimes:
            if regime["id"] == "usucapione" and not possession_over_time(signals.get("facts") or {}):
                continue            # v58: the usucapione regime too
            name = (fact_signals.real_action_name(signals, regime) if regime["id"] == fact_signals.REAL_ACTION_REGIME
                    else REGIME_NAMES.get(regime["id"], regime["id"].replace("_", " ")))
            add(name, "regime", regime=regime["id"], refs=[(str(n), _code(c)) for n, c in regime.get("refs") or []])
    from . import topic_tables
    if topic_tables.enabled():
        try:
            tables = topic_tables.matches(analysis, areas)
        except Exception as exc:
            fact_signals._log_exception("topic_step.tables", exc)
            tables = []
        for table in tables:
            add(table["name"], "tabella", table=table["id"], refs=[(str(n), _code(c)) for n, c in table["refs"]],
                questions=table["questions"], table_name=table.get("table_name") or table["name"])
    for issue in frame.get("central_issue_candidates") or []:
        if isinstance(issue, dict) and issue.get("nome"):
            add(issue["nome"], "frame", **({"area": issue["area"]} if issue.get("area") else {}))
    for q in analysis.get("questioni") or []:
        for inst in (q.get("istituti") or []) if isinstance(q, dict) else []:
            if isinstance(inst, dict) and inst.get("nome"):
                add(inst["nome"], "questione")
    return found


# --- the topics -----------------------------------------------------------------------------------------------------

def _pool(rows: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in rows or []:
        ref = row.get("_article") if isinstance(row, dict) else None
        if ref:
            out.setdefault((str(ref[0]), _code(ref[1])), row)
    return out


def _weight(row: Dict[str, Any]) -> float:
    rank = row.get("_fused_rank")
    return (1.0 + (1.0 if row.get("_central") else 0.0) + (1.0 if row.get("_institution") else 0.0)
            + (1.0 / (1.0 + rank / 10.0) if isinstance(rank, (int, float)) else 0.0))


def _rubrica(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return str(s.get("titolo_articolo") or s.get("title") or "")


def _two_signal(pool: Dict[tuple, Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    return {ref: row for ref, row in pool.items() if row.get("_multi_signal")}


def anchor(inst: Dict[str, Any], links, codes: Optional[set], pool: Dict[tuple, Dict[str, Any]],
           regimes: List[Dict[str, Any]], resolved: Optional[List[tuple]] = None,
           areas: Sequence[str] = ()) -> Tuple[Optional[tuple], str, str]:
    """(the anchor, how, its rubrica): a regime's or a table's own first article; the rubrica's exact or suffix match (an
    exact one first, then the one in the pool, then the longest rubrica, then the lowest number); the regime table; the
    pool's two-signal article among those the institution resolved to (resolved) whose rubrica shares a content word of
    the name (shares_name). v55: a rubrica or pool anchor within the institution's areas (area_fits)."""
    if inst.get("source") == "termine" and inst.get("refs"):         # a term: its own article
        return tuple(inst["refs"][0]), "term", ""
    if inst.get("source") in ("regime", "tabella") and inst.get("refs"):        # v55: its table's article, not a rubrica's
        return tuple(inst["refs"][0]), "regime" if inst["source"] == "regime" else "table", ""
    matches = []
    by_rubrica = len(content_words(inst["name"])) >= NAME_MIN_WORDS or inst.get("source") in SUPPORTED_SOURCES
    try:
        matches = links.rubrica_matches(inst["name"], codes) if links is not None and by_rubrica else []
        matches = [m for m in matches if not generic_rubrica(m[1], links) and area_fits(m[0], areas)]
    except Exception as exc:
        from . import fact_signals
        fact_signals._log_exception("topic_step.rubrica_matches", exc)
    if matches:
        ref, rubrica, how = min(matches, key=lambda m: (m[2] != "exact", m[0] not in pool, -len(tokens(m[1])),
                                                         _number(m[0][0])))
        return ref, how, rubrica
    for ref in inst.get("requires") or []:      # v55: a fired rule's required article whose rubrica shares the name
        rubrica = _rubrica(pool[ref]) if ref in pool else (getattr(links, "rubriche", None) or {}).get(ref, "")
        if rubrica and shares_name(inst["name"], rubrica) and area_fits(ref, areas):
            return ref, "rule", rubrica
    refs = list(inst.get("refs") or [])
    if not refs:
        low = str(inst["name"]).lower()
        hits = [(max(len(m.group(0)) for t in regime.get("triggers") or [] for m in [re.search(rf"\b{t}", low)] if m),
                 regime) for regime in regimes if any(re.search(rf"\b{t}", low) for t in regime.get("triggers") or [])]
        # the regime whose trigger is the longest match ("passaggio coattivo" over "servit"), then the narrower regime
        refs = list(max(hits, key=lambda h: (h[0], -len(h[1].get("triggers") or [])))[1]["refs"]) if hits else []
    if refs:
        return tuple(refs[0]), "regime", ""
    two = _two_signal(pool)
    mine = [r for r in (resolved or []) if r in two and (not codes or r[1] in codes) and area_fits(r, areas)
            and shares_name(inst["name"], _rubrica(two[r])) and not generic_rubrica(_rubrica(two[r]), links)]
    if mine:            # the pool's two-signal articles found for this institution, the strongest
        ref = max(mine, key=lambda r: (_weight(two[r]), -mine.index(r)))
        return ref, "pool", _rubrica(two[ref])
    return None, "", ""


# v57: the Capo's two-signal articles are planned (within the cap, after the Sezione's): on the replay, planned they hold
# 93/125 expected norms and 7 of v55's 26 misused anchors; listed only, 80/125 and 1 -- the misuse was the gloss, which
# the answer gives only to the articles that earned it (interactive_flow.earned_refs)
CAPO_PLANNED = True
SEZIONE_SHARE = 1 / 3          # v57: a word in more of a Sezione's rubriche than this says nothing within it
NEAR_WINDOW = 0                # v57: no Sezione article by position alone (k = 1-3 kept no expected norm more)


# v57: the books of the two codes the links do not divide (units.csv has no Libro I-II of the c.p.)
CODE_BOOKS = {"c.p.": [(1, 240), (241, 649), (650, 734)],
              "c.c.": [(1, 455), (456, 809), (810, 1172), (1173, 2059), (2060, 2642), (2643, 2969)]}


def _book(ref: Sequence[str], links=None) -> str:
    code = _code(ref[1])
    number = _number(ref[0])
    for i, (lo, hi) in enumerate(CODE_BOOKS.get(code, [])):
        if lo <= number <= hi:
            return f"{code} {i + 1}"
    try:
        return links.unit_name(ref, "LIBRO") if links is not None and hasattr(links, "unit_name") else ""
    except Exception:
        return ""


def _rubriche(links, refs: Sequence[tuple]) -> Dict[tuple, str]:
    if hasattr(links, "rubriche_of"):
        return links.rubriche_of(refs)
    return {(str(r[0]), _code(r[1])): (links.rubrica_of(r) if hasattr(links, "rubrica_of") else "") for r in refs}


def articles(ref: tuple, links, pool: Dict[tuple, Dict[str, Any]], codes: Optional[set] = None, name: str = "",
             matched: Optional[Sequence[tuple]] = None) -> List[tuple]:
    """articles_split's planned then its tail."""
    planned, tail = articles_split(ref, links, pool, codes, name, matched)
    return planned + tail


def articles_split(ref: tuple, links, pool: Dict[tuple, Dict[str, Any]], codes: Optional[set] = None, name: str = "",
                   matched: Optional[Sequence[tuple]] = None) -> Tuple[List[tuple], List[tuple]]:
    """v57 (v55-all's 37 misused: anchors glossed against the facts): a topic's anchors are its matched article(s) (ref,
    or matched: a table's or a regime's own), then its Sezione's articles whose rubrica shares a word of the topic's name
    no part of the codes owns (shares_name) and that is not in more than SEZIONE_SHARE of the Sezione's rubriche, nearest
    by number; then the Capo's articles retrieval found by two signals (_two_signal); then the articles the matched one
    cites (outbound REFERS_TO, its own code and book -- CODE_BOOKS --, no hub), least cited first; at
    most ARTICLES_MAX. Never an article citing it, never another code's (a procedural citer -- 380/381 c.p.p. -- reaches
    the procedural section only through a table), never another book's (640 c.p. cites 32-quater and 218, Libro I).
    The Capo's two-signal articles: the replay's expected norms 1061, 641, 1163, 41 c.p. come only from them (without,
    topic recall 83/125)."""
    out = list(dict.fromkeys([tuple(r) for r in (matched or [])] + [tuple(ref)]))[:ARTICLES_MAX]
    if links is None:
        return out, []
    try:
        mates = [r for r in links.unit(ref, "SEZIONE") if r != ref]
        rubriche = _rubriche(links, mates)
        counts = Counter(st for r in mates for st in _content_stems(rubriche.get((str(r[0]), _code(r[1])), "")))
        common = {st for st, n in counts.items() if len(mates) >= 3 and n > SEZIONE_SHARE * len(mates)}
        wanted = _content_stems(name) - GENERIC_STEMS - common
        ordered = sorted(mates + [ref], key=lambda r: (_number(r[0]), str(r[0])))
        at = ordered.index(ref)
        near = {r for r in ordered[max(0, at - NEAR_WINDOW):at + NEAR_WINDOW + 1] if r != ref}
        sezione = sorted((r for r in mates if r in near or wanted & _content_stems(rubriche.get((str(r[0]), _code(r[1])), ""))),
                         key=lambda r: (abs(_number(r[0]) - _number(ref[0])), _number(r[0])))
        cites = links.refers_out(ref) if hasattr(links, "refers_out") else []
        own = _code(ref[1])
        book = _book(ref, links)
        cited = [r for r, n in sorted(cites, key=lambda x: (x[1], x[0])) if r != ref and n <= HUB_INBOUND
                 and _code(r[1]) == own and (not book or _book(r, links) in ("", book))]
        capo = set(links.unit(ref, "CAPO"))
        found = [r for r in sorted(_two_signal(pool), key=lambda r: _number(r[0])) if r in capo]
    except Exception as exc:
        from . import fact_signals
        fact_signals._log_exception("topic_step.articles", exc)
        return out, []
    for r in sezione + (found if CAPO_PLANNED else []) + cited:
        if len(out) >= ARTICLES_MAX:
            break
        if r not in out:
            out.append(r)
    # past the cap, listed but never planned (interactive_flow reads a topic's first TOPIC_ARTICLES): the Capo's
    # two-signal articles left out
    return out, [r for r in found if r not in out]          # (planned, listed only)


def _facts(inst: Dict[str, Any], refs: List[tuple], signals: Dict[str, Any]) -> List[str]:
    from . import fact_signals
    numbers = {str(r[0]) for r in refs}
    out = [f"da verificare: {c}" for c in inst.get("to_verify") or []]
    for t in signals.get("terms") or []:
        if set(re.findall(r"\b\d+(?:-[a-z]+)?\b", str(t.get("ref") or ""))) & numbers:
            out.append(f"termine: {' '.join(str(t.get('text') or '').split())}")
    for tool in signals.get("tools") or []:
        if {str(r[0]) for r in tool.get("refs") or []} & numbers:
            matched = "; ".join(str(m) for m in tool.get("matched") or [] if m)
            out.append(f"strumento: {tool.get('id')}" + (f" ({matched})" if matched else ""))
    for rule in signals.get("fired") or []:
        if rule.get("unstated"):            # v55: its conclusion is not a fact; its conditions are the topic's to verify
            continue
        cited = set(re.findall(r"\bart\.\s*(\d+(?:-[a-z]+)?)", str(rule.get("conclusion") or "")))
        if cited & numbers or rule.get("id") == inst.get("rule"):
            out.append(f"regola: {rule.get('id')}" + (f" — {rule['conclusion']}" if rule.get("conclusion") else ""))
    if inst.get("regime") == fact_signals.REAL_ACTION_REGIME:
        out += [f"parte da convenire: {d['name']} ({d.get('why', '')})" for d in signals.get("defendants") or []
                if isinstance(d, dict) and d.get("name")]
    return list(dict.fromkeys(out))


def supported(topic: Dict[str, Any]) -> bool:
    """A topic the computed facts support: from a fired rule, a rule to verify, a computed term or a regime, or touching
    a fired rule, a term or a condition to verify (its facts)."""
    if str(topic.get("source") or "").startswith(SUPPORTED_SOURCES):
        return True
    return any(str(f).startswith(("regola:", "termine:", "da verificare:")) for f in topic.get("facts") or [])


def _sezione(ref: tuple, links) -> List[tuple]:
    """A term's articles: its own, then its Sezione's nearest by number (at most ARTICLES_MAX)."""
    mates = []
    if links is not None:
        try:
            mates = [r for r in links.unit(ref, "SEZIONE") if r != ref]
        except Exception as exc:
            from . import fact_signals
            fact_signals._log_exception("topic_step.sezione", exc)
    mates.sort(key=lambda r: (abs(_number(r[0]) - _number(ref[0])), _number(r[0])))
    return [ref] + mates[:ARTICLES_MAX - 1]


def without_gated(title: str, gate: Dict[str, Any]) -> str:
    """The title without the actions the action gate gives to the other side ("Azione confessoria e negatoria" with
    949 c.c. gated: "Azione confessoria"); "" when nothing is left."""
    from . import fact_signals
    words = [fact_signals.ACTION_WORDS[a] for label in gate or {} for a in
             [fact_signals.ACTION_ARTICLES.get(tuple(str(label).split(" ", 1)))] if a]
    if not words:
        return title
    parts = re.split(r"\s*(?:,|/|\be\b|\bed\b)\s*", title)
    kept = [p for p in parts if p and not any(re.search(w, p, re.IGNORECASE) for w in words)]
    if len(kept) == len(parts):
        return title
    return _title(" e ".join(kept)) if kept else ""


def _merge(into: Dict[str, Any], topic: Dict[str, Any]) -> None:
    if str(topic.get("source") or "").startswith("tabella"):      # v57 (H): a table merged in, for its name
        into.setdefault("tables", []).append({"title": _title(topic.get("table_name") or topic["title"]),
                                              "own": list(topic.get("own") or []), "source": topic["source"]})
    into["merged"].append(topic["title"])
    into["merged"] += [m for m in topic.get("merged") or [] if m not in into["merged"]]
    into["merged_sources"] = list(dict.fromkeys(list(into.get("merged_sources") or []) + [topic.get("source")]
                                                + list(topic.get("merged_sources") or [])))
    into["facts"] = list(dict.fromkeys(into["facts"] + topic["facts"]))
    into["questions"] = list(dict.fromkeys(list(into.get("questions") or []) + list(topic.get("questions") or [])))
    # v55: a regime's or a table's own articles go with it (n5: the l. 241 regime into the one-article TAR term topic);
    # v56: right after the anchor, before the neighbours (v55-p1: 575 and 589 c.p. past the eight articles read)
    # v59 (v58-p1: 52 and 55 c.p., already the Capo's neighbours of 43, stayed unmatched -- bare, «non trattate» --
    # while the three questions were the table's): every own article of the table is the topic's own (matched), the
    # ones already among its articles moved up after the anchor, inside the planned ones
    refs = list(into["refs"])
    planned = into.get("planned", len(refs))
    own = [r for r in topic.get("own") or [] if r != refs[0]]
    new = [r for r in own if r not in refs]
    rest = [r for r in refs[1:] if r not in own]
    into["refs"] = refs[:1] + own + rest
    into["planned"] = max(planned + len(new), 1 + len(own))
    into["own"] = list(dict.fromkeys(list(into.get("own") or []) + list(topic.get("own") or [])))


def _refs_of_label(label: Any) -> List[Tuple[str, str]]:
    """«art. 2947 c.c.», «artt. 29 e 41 c.p.a.», «2947 c.c.» -> [(number, code)]."""
    m = re.match(r"\s*(?:artt?\.\s*)?((?:\d+(?:-[a-z]+)?(?:\s*,\s*|\s+e\s+))*\d+(?:-[a-z]+)?)\s+(.+?)\s*$",
                 re.sub(r"\(.*?\)", "", str(label or "")))
    return [(n, _code(m.group(2))) for n in re.split(r"\s*,\s*|\s+e\s+", m.group(1)) if n] if m else []


def _topic_keys(topic: Dict[str, Any]) -> Set[Tuple[str, str]]:
    labels = list(topic.get("articles") or []) + [f"{r[0]} {r[1]}" for r in topic.get("refs") or [] if len(r) == 2]
    return {k for lab in labels for k in _refs_of_label(lab)}


def _term_words(term: Dict[str, Any]) -> Set[str]:
    return _content_stems(re.sub(r"\(.*?\)", "", str(term.get("label") or term.get("id") or "").replace("_", " "))) \
        - GENERIC_STEMS - {"termin", "prescr", "azione", "decorr"}


def _term_only(topic: Dict[str, Any], term: Dict[str, Any]) -> bool:
    """A topic made of the term's articles alone (a term's own topic: «Prescrizione del risarcimento…» on 2947 c.c.)."""
    keys, mine = set(_refs_of_label(term.get("ref"))), _topic_keys(topic)
    return bool(mine) and mine <= keys and (str(topic.get("source") or "").startswith("termine")
                                            or bool(_term_words(term) & _content_stems(topic.get("title"))))


# v59: the matters of the c.c. a term belongs to (its regime), by article: a term is carried by a topic of its matter
CIVIL_MATTERS = (("diritti reali", ((810, 1116), (1140, 1172))), ("condominio", ((1117, 1139),)),
                 ("contratti", ((1173, 2042),)), ("fatto illecito", ((2043, 2059),)))
# a term whose article is not its matter's (the prescriptions sit in Libro VI): the matter it runs for
TERM_MATTERS = {"prescrizione_extracontrattuale": "fatto illecito", "prescrizione_circolazione": "fatto illecito",
                "prescrizione_ordinaria": "contratti", "denuncia_conformita_consumo": "contratti",
                "responsabilita_conformita_consumo": "contratti", "azione_conformita_consumo": "contratti"}


def _matter(key: Tuple[str, str]) -> str:
    """The c.c. matter of an article (CIVIL_MATTERS), else its code."""
    m = re.match(r"\d+", key[0])
    if key[1] == "c.c." and m:
        return next((name for name, spans in CIVIL_MATTERS for lo, hi in spans if lo <= int(m.group(0)) <= hi), "c.c.")
    return key[1]


def carries_term(term: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> bool:
    """v59 (v58-l3: the usucapione's, negatoria's and rivendicazione's lines in round 1 and §4 beside condominium and
    damages topics; v58-h10: 2947 c.c. in an administrative case): a computed term is shown only when a listed topic
    carries it -- holds one of its articles, or is of its matter (CIVIL_MATTERS, TERM_MATTERS; another code: the same
    code) --, never by its own term-only topic alone while other topics are listed (alone, it is the answer's)."""
    keys = set(_refs_of_label(term.get("ref")))
    matters = {TERM_MATTERS.get(str(term.get("id") or ""))} - {None} or {_matter(k) for k in keys}
    others = [t for t in topics if not _term_only(t, term)]
    if not others:
        return len(others) < len(topics)
    if any(keys & _topic_keys(t) for t in others):
        return True
    return any(matters & {_matter(k) for k in _topic_keys(t)} for t in others)


def term_topics_out(topics: List[Dict[str, Any]], terms: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v59 (v58-h10 topic 1, «Prescrizione del risarcimento del danno extracontrattuale» on 2947 c.c. beside the TAR's
    and the ottemperanza's): a term's own topic the term is not carried by (carries_term) is no topic."""
    out = []
    for topic in topics:
        mine = [t for t in terms if _term_only(topic, t)]
        if mine and not any(carries_term(t, topics) for t in mine):
            continue
        out.append(topic)
    return out


def named_by_table(topic: Dict[str, Any]) -> None:
    """v57 (H: v56-p1 topic 1 "Elemento psicologico del reato" on 52/55/575/589): a topic a table merged into takes the
    table's name when the table's own articles are at least half of its planned ones (the old title among its merged);
    the table's questions are already its own (_merge). v60 (v59-n2 «Custodia cautelare in carcere», 5 of 7; v59-p1
    «Eccesso colposo di legittima difesa», 4 of 7): the table's own name ("table_name"), not the sub-name its "anchors"
    gave (topic_tables.matches), also for the table's own topic."""
    planned = [tuple(r) for r in topic["refs"][:topic.get("planned", len(topic["refs"]))]]
    mine = [{"title": topic["table_name"], "own": topic.get("own") or []}] if topic.get("table_name") else []
    for table in mine + list(topic.get("tables") or []):
        own = {tuple(r) for r in table["own"]}
        # v59 (v58-p1: 4 of 8, still «Elemento psicologico del reato»): at half or more
        if planned and len(own & set(planned)) * 2 >= len(planned) and table["title"] != topic["title"]:
            topic["merged"] = [table["title"] if m == table["title"] else m for m in topic["merged"]
                               if m != table["title"]] + [topic["title"]]
            topic["title"] = table["title"]
            return


def same_name_merged(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v55: two topics one of whose titles' content words (their first six letters) are all in the other's ("Truffa" /
    "Truffa mediante artifizi", "Usucapione" / "Usucapione ordinaria di beni immobili") are one: the stronger (the
    earlier, topics in rank order) is kept, with the other's title among its merged, its facts and its anchor article
    (last, when not among its articles)."""
    kept: List[Dict[str, Any]] = []
    for topic in topics:
        mine = _content_stems(topic["title"])
        twin = next((k for k in kept if mine and _content_stems(k["title"])
                     and (mine <= _content_stems(k["title"]) or _content_stems(k["title"]) <= mine)), None)
        if twin is None:
            kept.append(topic)
            continue
        _merge(twin, topic)
        ref = tuple(topic["anchor"].split(" ", 1))
        refs = [tuple(r) for r in twin["refs"]]
        if ref not in refs:
            cut = twin.get("planned", len(refs))           # the absorbed anchor is planned, before the tail
            twin["refs"] = refs[:cut] + [ref] + refs[cut:]
            twin["planned"] = cut + 1
    return kept


def _overlap(a: Sequence[tuple], b: Sequence[tuple]) -> float:
    small = min(len(a), len(b))
    return len(set(a) & set(b)) / small if small else 0.0


def allowed(analysis: Dict[str, Any], case_text: str = "") -> Optional[set]:
    from . import vote_pool
    codes = vote_pool.allowed_codes(vote_pool.case_areas(analysis), vote_pool.named_codes(case_text))
    return {_code(c) for c in codes} if codes else None


def build(analysis: Dict[str, Any], rows: Iterable[Dict[str, Any]], links=None, case_text: str = "",
          others: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """[{title, anchor, how, rubrica, articles: ["1062 c.c.", ...], facts, source, merged}] (module docstring). v54: a
    topic anchored on a decided term's article (possessoria DECORSO) is none; a topic the computed facts do not support
    (supported) and a topic whose every action is gated to the other side go to `others` when given (the "altri
    profili"), never among the topics; the gated actions leave the titles."""
    from . import fact_signals
    analysis = analysis or {}
    pool = _pool(rows)
    signals = analysis.get("fact_signals") or {}
    codes = allowed(analysis, case_text)
    areas = (analysis.get("frame") or {}).get("areas") or signals.get("areas") or []
    regimes = []
    if fact_signals.regime_candidates_enabled():
        try:
            table = {r["id"]: r for r in fact_signals.REGIME_CANDIDATES}
            regimes = [{**r, "triggers": table[r["id"]]["triggers"], "refs": [(str(n), _code(c)) for n, c in r["refs"]]}
                       for r in fact_signals.regime_candidates(analysis, areas)]
        except Exception as exc:
            fact_signals._log_exception("topic_step.build", exc)
    resolved: Dict[Tuple[str, ...], List[tuple]] = {}
    for name, labels in list((analysis.get("institution_refs") or {}).items()) + [
            (f.get("nome"), f.get("refs")) for f in analysis.get("rule_institutions") or [] if isinstance(f, dict)]:
        refs = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in labels or []]
        resolved.setdefault(tuple(tokens(name)), []).extend((str(r[0]), _code(r[1])) for r in refs if len(r) == 2)
    built = []
    decided = decided_refs(signals)
    for i, inst in enumerate(institutions(analysis)):
        own = [inst["area"]] if inst.get("area") else []
        ref, how, rubrica = anchor(inst, links, codes, pool, regimes, resolved.get(tuple(tokens(inst["name"]))),
                                   areas=own + list(areas))
        if ref is None:
            continue
        if (str(ref[0]), _code(ref[1])) in decided:
            continue
        own = [tuple(r) for r in inst.get("refs") or []] if how in ("table", "regime") else []
        head, tail = articles_split(ref, links, pool, codes, inst["name"], own)     # v57: matched, Sezione, cited
        refs, planned = head + tail, len(head)
        source = inst["source"] + (f" {inst.get('rule') or inst.get('regime') or inst.get('term') or inst.get('table')}"
                                   if inst.get("rule") or inst.get("regime") or inst.get("term") or inst.get("table")
                                   else "")
        built.append({"title": _title(inst["name"]), "anchor": _label(ref), "how": how, "rubrica": rubrica,
                      "refs": refs, "planned": planned, "source": source, "facts": _facts(inst, refs, signals), "merged": [],
                      "questions": list(inst.get("questions") or []),
                      **({"table_name": _title(inst["table_name"])} if inst.get("table_name") else {}),
                      "own": [tuple(r) for r in inst.get("refs") or []] if how in ("table", "regime") else [],
                      "rank": (_METHOD_ORDER.index(how), _SOURCE_ORDER.index(inst["source"]), i)})
    built.sort(key=lambda t: t["rank"])
    kept: List[Dict[str, Any]] = []
    for topic in built:
        twin = next((k for k in kept if _overlap(k["refs"], topic["refs"]) > DUPLICATE_SHARE), None)
        if twin:
            _merge(twin, topic)
            continue
        kept.append(topic)
    kept = same_name_merged(kept)
    for topic in kept:
        named_by_table(topic)
    out, rest = [], []
    gate = analysis.get("action_gate") or {}
    for t in kept:
        topic = {"title": without_gated(t["title"], gate), "anchor": t["anchor"], "how": t["how"],
                 "rubrica": t["rubrica"], "articles": [_label(r) for r in t["refs"]],
                 "planned": t.get("planned", len(t["refs"])), "facts": t["facts"],
                 **({"merged_sources": t["merged_sources"]} if t.get("merged_sources") else {}),
                 "source": t["source"], "merged": t["merged"], **({"questions": t["questions"]} if t["questions"] else {}),
                 # v57: the matched article(s) -- the anchor, a table's or a regime's own -- the only ones the answer
                 # applies to the facts with the fired rules' and the user's (interactive_flow.earned_refs)
                 "matched": list(dict.fromkeys([t["anchor"]] + [_label(r) for r in t.get("own") or []]))}
        if not topic["title"]:
            rest.append({**topic, "title": t["title"], "why": "azione della controparte"})
        elif not supported(topic):
            rest.append({**topic, "why": UNSUPPORTED_WHY})
        else:
            out.append(topic)
    if not any(not str(t["source"]).startswith("tabella") for t in out):
        # v54 decision: no topic the facts support (a criminal case, no rule fired): the frame's are offered; v55: next to
        # the tables' (a draft table never hides the frame's exact anchors), in rank order
        offered = [{**t, "why": NO_SUPPORT_OFFERED} for t in rest if t.get("why") == UNSUPPORTED_WHY]
        position = {t["title"]: i for i, t in enumerate(kept)}
        out = sorted(out + offered, key=lambda t: position.get(t["title"], len(kept)))
        rest = [t for t in rest if t.get("why") != UNSUPPORTED_WHY]
    out = term_topics_out(out, signals.get("terms") or [])          # v59
    if others is not None:
        case_words = _content_stems(case_text) - GENERIC_STEMS
        for topic in rest:                      # the "altri profili" are labelled by their anchor's Capo
            if len(others) >= TOPICS_MAX:
                break
            try:
                topic["capo"] = links.unit_name(tuple(topic["anchor"].split(" ", 1)), "CAPO") if links is not None \
                    and hasattr(links, "unit_name") else ""
            except Exception as exc:
                from . import fact_signals
                fact_signals._log_exception("topic_step.capo", exc)
                topic["capo"] = ""
            if not relevant_other(topic, links, pool, case_words):
                continue
            others.append(topic)
    return out[:TOPICS_MAX]


def relevant_other(topic: Dict[str, Any], links, pool: Dict[tuple, Dict[str, Any]], case_words: Set[str]) -> bool:
    """v58 (v57 round 1: «827 beni dello Stato», 2828-2833 as «danni all'immobile», «Capo V esercizio delle servitù»):
    an "altro profilo" whose Capo shares no word of its title no part of the codes owns (the generic-word rule, at Capo
    level) is none; nor one whose articles' rubriche share no such word with the case's text (when it is given)."""
    if topic.get("capo") and not shares_name(topic["title"], re.sub(r"^\s*capo\s+[ivxlc\d-]+\s*[–-]\s*", "",
                                                                     topic["capo"], flags=re.IGNORECASE)):
        return False
    if not case_words:
        return True
    refs = [tuple(a.split(" ", 1)) for a in topic.get("articles") or [] if " " in a]
    rubriche = dict(_rubriche(links, refs)) if links is not None else {}
    for ref in refs:
        rubriche[(str(ref[0]), _code(ref[1]))] = rubriche.get((str(ref[0]), _code(ref[1]))) or \
            _rubrica(pool.get((str(ref[0]), _code(ref[1]))) or {})
    return any(_content_stems(r) - GENERIC_STEMS & case_words for r in rubriche.values() if r)


def recall(topics: Sequence[Dict[str, Any]], expected: Sequence[Sequence[str]], key) -> Dict[str, Any]:
    """{"expected", "in_topics", "in_top3", "share", "share_top3", "missing"}: expected items (each a list of
    alternative keys) found among the articles of all topics and of the first three; key(label) -> an item's key."""
    every = {key(a) for t in topics for a in t["articles"]}
    top = {key(a) for t in topics[:3] for a in t["articles"]}
    found = [any(k in every for k in alts) for alts in expected]
    found3 = [any(k in top for k in alts) for alts in expected]
    n = len(expected)
    return {"expected": n, "in_topics": sum(found), "in_top3": sum(found3),
            "share": round(sum(found) / n, 3) if n else None, "share_top3": round(sum(found3) / n, 3) if n else None,
            "missing": [list(alts) for alts, ok in zip(expected, found) if not ok]}
