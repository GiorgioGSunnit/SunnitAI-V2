"""v62: the posture consumers -- the client's posture (intake's rito, role and stage, its prior decisions) read once per
answer, and what it drives:

  of(state, analysis)        the posture: {rito, family, role, stage, querela_stated, prior_decisions, entry}. The rito
                             and the role are intake's when its rules or the model read them, else the existing logic's
                             (intake.resolve: the frame's area, the client's side); the stage is intake's (none
                             without an intake: today's behaviour everywhere the stage decides).
  entry_for(rito, role, stage)
                             the act family and the goal of the main line (data/posture_acts.json); None for a cell no
                             entry matches -- civile/attore/pre_filing among them -- whose plan stays today's exactly.
  apply_plan(plan, posture)  the plan's main line from the entry: plan["atto"] (the act) and plan["obiettivo"] (the
                             goal); v70: the entry's «subordinate» lines added to plan["subordinate"] (the imputato's
                             riti alternativi and sentencing); the plan unchanged without an entry.
  mark_sides(state, posture) each record's side (rec["side"]): a «threat» when the client is imputato or convenuto and
                             the article is the charge or the claim -- named in the case's charge or claim sentences
                             (by number, or by its rubrica: charged_by_name) or relied on by the other side
                             (rec["relied_by"]); else a «weapon». v63: an article among the client's alternative
                             qualifications (alternative_rows) is a weapon whatever the charge sentence names
                             (rec["alternative"]); rec["charged"] marks the charge, whoever the client. Logged.
                             v65: an article the user names with its verb (intake's named_articles) takes the side
                             the verb gives -- weapon when the client invokes it, threat when it is used against the
                             client --, whatever the role.
  choose_charge(...)         v79.10.5 (E1): the charge by rule, held across the pipeline -- the user's article, else the
                             first read's hypothesis with most elements integrato, else the store's rubrica match;
                             logged with its rule (charge_rule), kept in state["charge"].
  counterparty_label(...)    v65: the counterparty's label, the opposite of the client's role (attore <-> convenuto,
                             imputato <-> persona offesa / parte civile).
  alternative_rows(state)    v63: the first read's alternative qualifications, each with the row of its article (its
                             rubrica named by the qualification) or None.
  procedure_offences(...)    v63: the procedural section's offences -- the charged articles and (v64) the
                             alternatives that fit the facts, never every graded candidate nor a rejected alternative.

No case-specific logic: every rule reads the posture's fields, the records and the case's own sentences.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ACTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "posture_acts.json")
PENAL_RITI = ("penale", "penale_gdp")
CIVIL_RITI = ("civile", "lavoro", "famiglia", "esecuzione")
ACCUSED_ROLES = ("imputato", "convenuto")
UNKNOWN = "sconosciuto"

_TABLE: Dict[str, Any] = {}


def table() -> List[Dict[str, Any]]:
    """data/posture_acts.json's entries (read once; [] when missing or unreadable, logged)."""
    if "entries" not in _TABLE:
        try:
            with open(ACTS_FILE, encoding="utf-8") as f:
                _TABLE["entries"] = [e for e in json.load(f).get("entries") or [] if isinstance(e, dict)]
        except (OSError, ValueError) as exc:
            logger.warning("posture: %s not read: %s", ACTS_FILE, exc)
            _TABLE["entries"] = []
    return _TABLE["entries"]


def family(rito: str) -> str:
    """penale | civile | amministrativo | "" -- the procedure family of a rito."""
    rito = str(rito or "")
    if rito in PENAL_RITI:
        return "penale"
    if rito in CIVIL_RITI:
        return "civile"
    return "amministrativo" if rito == "amministrativo" else ""


def entry_for(rito: str, role: str, stage: str) -> Optional[Dict[str, Any]]:
    """The table's first entry whose rito, role and stage lists hold the cell's; None for no match or an unknown field."""
    if UNKNOWN in (rito, role, stage) or not (rito and role and stage):
        return None
    return next((e for e in table() if rito in (e.get("rito") or []) and role in (e.get("role") or [])
                 and stage in (e.get("stage") or [])), None)


# a querela or a denuncia-querela the case says was filed
_QUERELA_STATED_RE = re.compile(
    r"\b(?:(?:ha|aveva|hanno|avevano)\s+)?(?:sporto|presentato|proposto|depositato|sporge|presenta|propone|deposita|"
    r"sporgono|presentano)\s+(?:formale\s+)?(?:(?:denuncia|esposto)(?:-|\s+e\s+))?querela\b|\bquerela\s+"
    r"(?:presentata|sporta|proposta|depositata)\b|\bdenuncia-querela\b", re.IGNORECASE)


def querela_stated(case: str) -> bool:
    return bool(_QUERELA_STATED_RE.search(case or ""))


def of(state: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The answer's posture (see the module's doc). Never raises: an unreadable intake is no intake."""
    from . import intake as intake_mod, interactive_flow
    raw = (state or {}).get("intake") or {}
    resolved = intake_mod.resolve(raw, analysis or {}) if raw else None
    if resolved is None:            # no intake (an old state, the replay): the existing logic's rito and role only
        resolved = intake_mod.resolve({"rito": UNKNOWN, "role": UNKNOWN, "stage": "", "confidence": {}},
                                      analysis or {}) or {}
    rito = str(resolved.get("rito") or UNKNOWN)
    if rito == UNKNOWN:
        families = interactive_flow.frame_families(analysis or {})
        rito = families[0] if families and families[0] in ("civile", "penale", "amministrativo") else UNKNOWN
    role = str(resolved.get("role") or UNKNOWN)
    stage = str(resolved.get("stage") or "") or UNKNOWN
    case = str((analysis or {}).get(interactive_flow.CASE_KEY) or (state or {}).get("text") or "")
    read = str((raw or {}).get("rito") or "")
    resist = str(resolved.get("resisting") or "") if role == "convenuto" else ""
    if role == "attore" and family(rito) == "civile" and (resolved.get("confidence") or {}).get("role") != "rule":
        # v79.10.2 (B1): the party resisting a demand is never the claimant -- a model's or a default's «attore» on a
        # case (the precisazioni too) that shows the client resisting is the defending role
        resist = resist or intake_mod.resisting(case, str((resolved.get("client") or {}).get("name") or ""))
        if resist:
            role = "convenuto"
    from . import case_frame
    if case_frame.held(state) is not None:      # v79.14.0 (CASE_FRAME): the frame's rito, role and stage -- never
        for c in resolved.get("frame_conflicts") or []:         # resolve's own (it lost): kept in the state
            case_frame.conflict(state, c.get("field"), c.get("step"), c.get("proposed"), c.get("frame"))
        rito = case_frame.defer(state, "rito", "posture.of", rito)     # this step's own (logged, it loses)
        role = case_frame.defer(state, "ruolo", "posture.of", role)
        stage = case_frame.defer(state, "stadio", "posture.of", stage)
        resist = resist if role == "convenuto" else ""
        read = case_frame.bound(state, "rito") or read
    out = {"rito": rito, "family": family(rito), "role": role, "stage": stage, "resisting": resist,
           "querela_stated": querela_stated(case), "prior_decisions": list(resolved.get("prior_decisions") or []),
           "from_intake": bool(raw), "intake_rito": read if read != UNKNOWN else ""}      # v75: the tables' key
    out["entry"] = entry_for(rito, role, stage)
    return out


def apply_plan(plan: Dict[str, Any], posture: Optional[Dict[str, Any]],
               rites: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The plan's main line from the posture's entry -- the act (plan["atto"]) and the goal (plan["obiettivo"]); the
    plan itself, unchanged, without an entry (the default cell: today's plan exactly). v79.10.3 (C1): with the charge's
    riti (lawyer_method.charge_rites), an entry's subordinate line naming a rito the charge does not admit is that
    rito's own line instead -- «da verificare sulla pena edittale» when the pena is unknown, none when it precludes
    it; v79.10.7: the patteggiamento's «possibile, dipende dalla pena in concreto» likewise."""
    entry = (posture or {}).get("entry")
    if not entry or not plan:
        return plan
    out = dict(plan)
    if entry.get("act"):
        out["atto"] = entry["act"]
    if entry.get("goal"):
        out["obiettivo"] = entry["goal"]
    if entry.get("subordinate"):        # v70: the imputato's riti alternativi and sentencing, whatever the model planned
        have = [str(x) for x in out.get("subordinate") or []]
        out["subordinate"] = have + [x for x in entry_subordinate(entry, rites) if x not in have]
    return out


def reapply_rites(plan: Dict[str, Any], posture: Optional[Dict[str, Any]], before: Optional[Sequence[Dict[str, Any]]],
                  after: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """v79.10.7 (review): the plan's subordinate lines re-read on other riti statuses (the method's, with the attenuanti
    it graded) -- the entry's lines apply_plan added for «before» replaced by those «after» gives; the model's own
    lines kept. The plan unchanged without an entry's subordinate lines."""
    entry = (posture or {}).get("entry") or {}
    if not plan or not entry.get("subordinate"):
        return plan
    old, new = entry_subordinate(entry, before), entry_subordinate(entry, after)
    kept = [str(x) for x in plan.get("subordinate") or [] if x not in old or x in new]
    return {**plan, "subordinate": kept + [x for x in new if x not in kept]}


def entry_subordinate(entry: Dict[str, Any], rites: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """v79.10.3 (C1): the entry's subordinate lines with the charge's riti applied (apply_plan) -- unchanged without
    them (rites None: no penal charge read)."""
    lines = [str(x) for x in entry.get("subordinate") or []]
    if not rites:
        return lines
    from . import lawyer_method
    by_id = {str(x.get("id") or ""): x for x in rites}
    out: List[str] = []
    for line in lines:
        rite = by_id.get(lawyer_method.rite_of(line))
        if not rite or rite.get("status") == lawyer_method.AVAILABLE:
            out.append(line)
        elif rite.get("status") in (lawyer_method.TO_VERIFY_RITE, lawyer_method.POSSIBLE):   # v79.10.7: its own words
            out.append(str(rite.get("line") or ""))
    return [x for x in out if x]


# the case's sentences stating the charge or the claim: «il PM contesta …», «imputato di …», «a norma degli articoli …»,
# «chiede la condanna …» -- v79 (C2): the wordings are data (data/charge_wordings.json), never code
CHARGE_WORDINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                                    "charge_wordings.json")
_CHARGE: Dict[str, Any] = {}


def charge_re() -> "re.Pattern":
    """The charge sentence's wordings (data/charge_wordings.json), one expression, read once; a missing or broken file:
    an expression matching nothing (logged)."""
    if "re" not in _CHARGE:
        try:
            with open(CHARGE_WORDINGS_FILE, encoding="utf-8") as f:
                patterns = [w["pattern"] for w in json.load(f).get("wordings") or [] if w.get("pattern")]
            _CHARGE["re"] = re.compile("|".join(patterns) or r"(?!x)x", re.IGNORECASE)
        except (OSError, ValueError, KeyError, re.error) as exc:
            logger.warning("posture: %s not read: %s", CHARGE_WORDINGS_FILE, exc)
            _CHARGE["re"] = re.compile(r"(?!x)x")
    return _CHARGE["re"]


# v79 (C3): an article of a special law (data/special_laws.json) -- «art. 73 d.P.R. 309/1990», «art. 73, comma 5, del
# d.P.R. n. 309 del 1990», «art. 55 del d.lgs. 81/2008» -- as «<number> <act> <number>/<year>»
_SPECIAL_CITE_RE = re.compile(
    r"\bart(?:t|icol[oi])?\.?\s*(?P<art>\d{1,4}(?:-(?:bis|ter|quater|quinquies|sexies))?)\b(?P<gap>[^;]{0,40}?)"
    r"\b(?:del(?:la|l['’])?\s+)?(?P<act>d\.\s?p\.\s?r\.|d\.\s?lgs\.?|decreto\s+legislativo|decreto\s+del\s+presidente\s+della\s+"
    r"repubblica|legge|l\.)\s*(?:\d{1,2}\s+[a-zà-ù]+\s+(?P<dy>\d{4})\s*,?\s*)?(?:n\.\s*)?(?P<num>\d{1,4})"
    r"(?:\s*/\s*(?P<y1>\d{4}|\d{2})\b|,?\s+del\s+(?P<y2>\d{4})\b)?", re.IGNORECASE)
_ACT_NAMES = (("d.P.R.", r"d\.\s?p\.\s?r|presidente"), ("d.lgs.", r"lgs|legislativo"), ("l.", r"legge|l\."))
_CODE_SIGLA_RE = re.compile(r"\bc\.\s?[pc]\.|\bart(?:t|icol[oi])?\.?\s*\d", re.IGNORECASE)


def special_cites(sentence: str) -> List[Tuple[int, str]]:
    """v79 (C3): [(position, «<article> <act> <n>/<year>»)] of the special-law articles a sentence cites -- an act of the
    special laws' table only (special_laws.law_of: its number and year)."""
    from . import special_laws
    out = []
    for m in _SPECIAL_CITE_RE.finditer(sentence or ""):
        year = m.group("y1") or m.group("y2") or m.group("dy")
        if not year or _CODE_SIGLA_RE.search(m.group("gap") or ""):
            continue
        year = year if len(year) == 4 else ("19" if int(year) > 30 else "20") + year
        act = next(name for name, pattern in _ACT_NAMES if re.search(pattern, m.group("act"), re.IGNORECASE))
        text = f"{act} {int(m.group('num'))}/{year}"
        if special_laws.law_of(text):
            out.append((m.start(), f"{m.group('art').lower()} {text}"))
    return out


# a list of articles closed by its code's name («a norma degli articoli 582 - 583, comma 1, numero 1 – 585 – 61, numero
# 4 – del codice penale»): the numbers, the commi's and items' left out
_ARTICLE_LIST_RE = re.compile(r"\bartt?(?:icol[oi]|\.)\s+(.{1,200}?)\s+del (codice penale|codice civile|codice di procedura "
                              r"penale|codice di procedura civile)\b", re.IGNORECASE)
_LIST_CODES = {"codice penale": "c.p.", "codice civile": "c.c.", "codice di procedura penale": "c.p.p.",
               "codice di procedura civile": "c.p.c."}
_SUB_RE = re.compile(r"\b(?:comma|co\.|numero|num\.|n\.|lett(?:era|\.)?)\s*\w+", re.IGNORECASE)


def sentence_articles(sentence: str) -> List[str]:
    """v71: the articles a sentence names, as labels -- each reference, and the lists «degli articoli 582 - 583 … del
    codice penale»."""
    from . import citations, interactive_flow
    found = list(citations.article_cites(sentence or ""))          # v72: the shared tolerant parser, with positions
    for m in _ARTICLE_LIST_RE.finditer(sentence or ""):
        code = _LIST_CODES[m.group(2).lower()]
        body = _SUB_RE.sub(lambda x: " " * len(x.group(0)), m.group(1))       # the positions kept
        found += [(m.start(1) + n.start(), (n.group(0).lower(), code))
                  for n in re.finditer(r"\b\d{1,4}(?:-(?:bis|ter|quater|quinquies))?\b", body)]
    labels = [(at, interactive_flow._label(interactive_flow._key(ref))) for at, ref in found]
    labels += special_cites(sentence)                              # v79 (C3): a special law's article
    labels.sort(key=lambda x: x[0])
    return list(dict.fromkeys(label for _, label in labels))


def intake_family(posture: Optional[Dict[str, Any]]) -> str:
    """v75: the key of the ratione temporis, stragiudiziale, mediazione and riti lines -- the rito intake read, never
    the frame's area (a civil phishing case whose word lit the penal area keeps its civil lines): «lavoro» for the
    labour rito, else its family; "" when intake read none. A posture without the key (an older state, a test) keeps its
    own rito."""
    posture = posture or {}
    rito = str(posture["intake_rito"] if "intake_rito" in posture else posture.get("rito") or "")
    if rito in ("", UNKNOWN):
        return ""
    return "lavoro" if rito == "lavoro" else family(rito)


def _name_re(name: str) -> "re.Pattern":
    words = [w for w in re.findall(r"[A-ZÀ-Ü][\wà-ÿ'’]+", name or "") if len(w) >= 3] or [str(name or "").strip()]
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words if w) + r")\b")


def client_text(case: str, scope: Optional[Dict[str, Any]]) -> str:
    """v79.6 (item 2): the case as one client's charge reads it -- each sentence cut into its clauses (at «;», «:» and
    before another client's name after a comma); a clause naming only other clients is dropped, one naming no client
    goes with the clause before it (a sentence's first clause: everyone's). Never the whole case for a scope with no
    other client."""
    from . import intake
    if not scope or not scope.get("others"):
        return case or ""
    mine = _name_re(scope["name"])
    others = [_name_re(x) for x in scope["others"]]
    kept: List[str] = []
    for sentence in intake.sentences(case or ""):
        owner, mine_ = "all", []
        for clause in re.split(r"\s*[;:]\s*|,\s+(?=(?:mentre\s+|e\s+)?[A-ZÀ-Ü][\wà-ÿ]+\s)", sentence):
            me = bool(mine.search(clause))
            them = any(o.search(clause) for o in others)
            owner = "me" if me else ("them" if them else owner)
            if owner != "them" and clause.strip():
                mine_.append(clause.strip().rstrip("."))
        if mine_:                   # the sentence's clauses kept together: its charge verb stays with its articles
            kept.append(", ".join(mine_))
    return ". ".join(kept) + ("." if kept else "")


def charge_labels(case: str) -> List[str]:
    """The articles the case's charge or claim sentences name («Il PM contesta a Sempronio l'art 584 cp», «imputato …
    a norma degli articoli 582 - 583 … del codice penale»), as labels. v73: the in-flow fallback reads the case's text
    as intake does -- its charge sentences (intake.prior_decisions' «contestazione» entries: «contesta / imputa /
    chiede la condanna», the shared tolerant parser) first, then this module's (the claim's «chiede il pagamento», «a
    norma degli articoli …») --, so the charge is marked with or without intake."""
    from . import intake
    out: List[str] = list(intake_charges({"prior_decisions": intake.prior_decisions(case or "")}))
    for sentence in intake.sentences(case or ""):
        if charge_re().search(sentence):
            out += [x for x in sentence_articles(sentence) if x not in out]
    return out


def intake_charges(posture: Optional[Dict[str, Any]]) -> List[str]:
    """v71: the charge as intake read it -- the articles of its «contestazione» entries (intake.prior_decisions)."""
    out: List[str] = []
    for d in (posture or {}).get("prior_decisions") or []:
        if d.get("outcome") == CHARGE_OUTCOME:
            out += [x for x in d.get("articles") or [] if x not in out]
    return out


CHARGE_OUTCOME = "contestazione"
FRAME_CHARGE_RULE = "case_frame"        # v79.14.0: choose_charge's rule when the verified frame holds the charge
CLAIM_ROLES = {"attore": "principale", "convenuto": "controparte"}     # v75: whose line is the attore's claim
PENAL_LINE = "principale"       # v79 (C4): a criminal case's charge, when none is stated: the read's main hypothesis


# v76: the claim as the citazione pleads it -- petitum and causa petendi («chiedeva al Tribunale di accertare /
# condannare …», «ha convenuto … chiedendo la condanna …»)
_PLEADED_RE = re.compile(
    r"\bchied\w*\s+(?:al\s+(?:tribunale|giudice|giudice\s+di\s+pace)\b[^.;]{0,60}?\s+)?(?:di\s+)?(?:accertare|dichiarare|"
    r"condannare|pronunciare|ordinare|disporre|risolvere|annullare)\b|\bchiedendo\s+(?:al\s+\w+\s+)?(?:di\s+)?"
    r"(?:accertare|dichiarare|condannare|la\s+condanna|la\s+risoluzione|il\s+risarcimento|l['’]accertamento|"
    r"la\s+declaratoria|il\s+pagamento)", re.IGNORECASE)
PLEADED_MATCH = 2           # content stems a hypothesis shares with the pleaded sentence
_PLEADING_WORDS = "chiede chiedeva chiedendo chiesto accertare dichiarare condannare pronunciare ordinare disporre " \
                  "tribunale giudice citazione convenuto attore domanda"


def pleaded_claim(case: str) -> str:
    """v76: the sentence of the case that pleads the claim (petitum and causa petendi); "" when none does."""
    from . import intake
    return next((x for x in intake.sentences(case or "") if _PLEADED_RE.search(x)), "")


def claim_from_read(state: Dict[str, Any], posture: Optional[Dict[str, Any]], rows: Sequence[Dict[str, Any]] = (),
                    log: Optional[Dict[str, Any]] = None) -> List[str]:
    """v75: a civil case whose claim names no article («Tizio chiedeva il risarcimento dei danni»): the claim is the
    first read's principal hypothesis for the attore's claim -- the client's «principale» line when the client is the
    attore, the other side's («controparte») when the client is the convenuto --, its articles the rows the first read's
    search found for that hypothesis's institutions (tagged _first_read), fattispecie only, in the read's order. []
    for another family or role, or when the read has no such line."""
    from . import interactive_flow as flow, issue_records
    fam = str((posture or {}).get("family") or "")
    hyps = ((state or {}).get("read") or {}).get("hypotheses") or []
    if fam == "penale":         # v79 (C4): no charge stated -- the read's main hypothesis, whoever the client
        line = PENAL_LINE
        hyp = next((h for h in hyps if h.get("line") == PENAL_LINE), hyps[0] if hyps else None)
    elif fam != "civile":
        return []
    else:
        line = CLAIM_ROLES.get(str((posture or {}).get("role") or ""))
        hyp = next((h for h in hyps if h.get("line") == line), None)
    pleaded = pleaded_claim(flow.case_text(state or {})) if line == "controparte" else ""
    if pleaded:                 # v76: the convenuto's claim is the pleaded one -- the hypothesis that matches it
        read = (state or {}).get("read") or {}
        names = " ".join([str(p.get("name") or "") for p in read.get("parties") or []]
                         + [str((read.get("client") or {}).get("name") or "")])
        noise = _stems(names) | _stems(_PLEADING_WORDS)
        m = _PLEADED_RE.search(pleaded)
        words = _stems(pleaded[m.start():] if m else pleaded) - noise          # the petitum and the causa petendi
        scored = sorted(((len(words & (_stems(" ".join([str(h.get("claim") or "")] + [str(x) for x in
                                                                                     h.get("institutions") or []]))
                                       - noise)), i) for i, h in enumerate(hyps)), reverse=True)
        best = scored[0] if scored else (0, -1)
        if log is not None:
            log["pleaded"] = pleaded[:200]
            if hyp is not None and (best[0] < PLEADED_MATCH or hyps[best[1]] is not hyp):
                log["unmatched_hypothesis"] = str(hyp.get("claim") or "")[:200]     # an alternative, not the claim
        hyp = hyps[best[1]] if best[0] >= PLEADED_MATCH else None
    if not line or not hyp:
        return []
    pool = [r for r in list(rows or []) + flow._restored_rows((state or {}).get("rows"))
            + flow._restored_rows((state or {}).get("reserve")) if r.get("_article") and r.get("_first_read")]
    out: List[str] = []
    for inst in hyp.get("institutions") or []:
        for r in pool:
            label = flow._label(flow._key(r["_article"]))
            if flow._low(r["_first_read"]) == flow._low(" ".join(str(inst).split())) and label not in out \
                    and issue_records.counted(label, flow._row_rubrica(r)) and (fam != "penale" or penal_article(label)):
                out.append(label)
    return out


# v79.12.4 (item 3): the decomposition's reason for a first read's hypothesis other than its main one (decompose_records)
# -- every hypothesis is split so the charge rule can score it; one not chosen is graded, never printed (mark_sides)
READ_CANDIDATE_WHY = "ipotesi della prima lettura"

# v79.10.5 (E1): the rules that choose the charge, in their order
CHARGE_RULES = ("user_article", "read_hypothesis", "rubrica_match")
NO_CHARGE_RULE = "none"
_LINE_RANK = {"principale": 0, "subordinata": 1, "controparte": 2}


def read_hypotheses(state: Dict[str, Any], posture: Optional[Dict[str, Any]], rows: Sequence[Dict[str, Any]] = (),
                    log: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], List[str]]]:
    """v79.10.5 (E1): the first read's hypotheses a charge can come from, each with its articles (the rows the read's
    search found for its institutions, fattispecie only: claim_from_read's) -- a criminal case: every hypothesis; a
    civil one: those of the role's line (CLAIM_ROLES), or only the one matching the pleaded claim (v76) when the case
    pleads one; [] for another family."""
    from . import interactive_flow as flow, issue_records
    fam = str((posture or {}).get("family") or "")
    hyps = [h for h in ((state or {}).get("read") or {}).get("hypotheses") or [] if isinstance(h, dict)]
    if fam == "civile":
        principal = claim_from_read(state, posture, rows, log)
        if (log or {}).get("pleaded"):          # v76: the pleaded claim decides, as claim_from_read reads it
            return [({"line": "controparte", "claim": "", "institutions": []}, principal)] if principal else []
        line = CLAIM_ROLES.get(str((posture or {}).get("role") or ""))
        hyps = [h for h in hyps if line and h.get("line") == line]
    elif fam != "penale":
        return []
    pool = [r for r in list(rows or []) + flow._restored_rows((state or {}).get("rows"))
            + flow._restored_rows((state or {}).get("reserve")) if r.get("_article") and r.get("_first_read")]
    out: List[Tuple[Dict[str, Any], List[str]]] = []
    for h in hyps:
        labels: List[str] = []
        for inst in h.get("institutions") or []:
            for r in pool:
                label = flow._label(flow._key(r["_article"]))
                if flow._low(r["_first_read"]) == flow._low(" ".join(str(inst).split())) and label not in labels \
                        and issue_records.counted(label, flow._row_rubrica(r)) and (fam != "penale" or penal_article(label)):
                    labels.append(label)
        out.append((h, labels))
    return out


def element_score(rec: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    """v79.10.5 (E1): a record's (integrato, escluso, elements) -- its live elements as the reader gets them."""
    from . import lawyer_method
    live = lawyer_method._live(rec or {}) if rec else []
    return (sum(1 for r in live if r.get("status") == lawyer_method.PROVATO),
            sum(1 for r in live if r.get("status") == lawyer_method.NON_RISULTA), len(live))


def rubricated_match(state: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[str]:
    """v79.10.5 (E1): the records whose rubrica (the store's) holds every content stem of a name the first read gave a
    hypothesis's institution as rubricated («Lesione personale»), in the read's order."""
    from .topic_step import GENERIC_STEMS
    out: List[str] = []
    for h in ((state or {}).get("read") or {}).get("hypotheses") or []:
        for name in (h.get("rubricated") or []) if isinstance(h, dict) else []:
            stems = _stems(str(name))
            if not stems or not (stems - GENERIC_STEMS):
                continue
            for label, rec in (records or {}).items():
                if label not in out and stems <= _stems(str((rec or {}).get("rubrica") or "")):
                    out.append(label)
    return out


VICTIM_ROLES = ("persona_offesa", "parte_civile")         # lawyer_method.VICTIM_ROLES (no import cycle at load)


def victim(posture: Optional[Dict[str, Any]]) -> bool:
    """v79.12.4 (item 1): the client is the persona offesa or the parte civile."""
    return str((posture or {}).get("role") or "") in VICTIM_ROLES


# v79.12.4 (item 3, review): an attempt -- the offence's event, by definition, did not happen (a tentato omicidio with
# the victim injured and alive): the event table never refuses its hypothesis
_ATTEMPT_RE = re.compile(r"\btentat\w*|\bart(?:icolo|\.)?\s*56\b(?:\s*c\.\s*p\.)?", re.IGNORECASE)


def attempted(hypothesis: Optional[Dict[str, Any]]) -> bool:
    """v79.12.4 (item 3, review): the first read's hypothesis names an attempt (its claim, institutions or any word of
    it: «tentato omicidio», «art. 56») -- the event table never refuses it (event_refuses)."""
    words = " ".join(str(v) for v in (hypothesis or {}).values() if isinstance(v, str)) + " " + " ".join(
        str(x) for x in (hypothesis or {}).get("institutions") or [])
    return bool(_ATTEMPT_RE.search(words))


def event_refuses(label: str, event: str, hypothesis: Optional[Dict[str, Any]] = None) -> bool:
    """v79.12.4 (item 3): the certain facts' event does not fit the article (lawyer_method.event_mismatch: a death and
    590) -- never for a hypothesis naming an attempt (attempted: injuries fit a tentato omicidio)."""
    from . import lawyer_method
    return not attempted(hypothesis) and lawyer_method.event_mismatch(label, event)


def certain_event(state: Optional[Dict[str, Any]], case: str = "") -> str:
    """v79.12.4 (item 3): the event the case's certain facts narrate (lawyer_method.event_of: «morte», «lesioni», "") --
    the state's sentences when it has them (interactive_flow.certain_event), else the case's own stated as accertato."""
    from . import interactive_flow, lawyer_method
    if (state or {}).get("sentences"):
        return interactive_flow.certain_event(state)
    return lawyer_method.event_of(lawyer_method.certain_text({"case": case or ""}))


def max_pena(label: str, rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """v79.12.4 (item 3): the article's maximum pena in years from the procedural store (lawyer_method.charge_pena and
    pena_of) -- inf for the ergastolo, None when the store states none (another code's article, no row, no pena)."""
    from . import lawyer_method
    text, _ = lawyer_method.charge_pena(label, rows or [])
    if not text:
        return None
    p = lawyer_method.pena_of(text)
    if not p.get("known"):
        return None
    return float("inf") if p.get("ergastolo") else float(p.get("high") or 0.0)


def heaviest(labels: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    """v79.12.4 (item 3): among several charged articles the one with the highest maximum pena from the store -- never
    the list's order; an article whose pena the store does not state ranks below every known one; ties, and none
    known, keep the given order. "" for no label."""
    labels = [x for x in dict.fromkeys(labels or []) if x]
    if not labels:
        return ""
    ranked = sorted(enumerate(labels), key=lambda x: (-(max_pena(x[1], rows) if max_pena(x[1], rows) is not None
                                                       else float("-inf")), x[0]))
    return ranked[0][1]


def choose_charge(state: Dict[str, Any], posture: Optional[Dict[str, Any]], records: Dict[str, Dict[str, Any]],
                  case: str, rows: Sequence[Dict[str, Any]] = (), user: Sequence[str] = (),
                  repealed: Sequence[str] = (), read_log: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v79.10.5 (E1): the charge (or the claim) by rule, the same for the same facts -- {"article", "rule", "order",
    "scores", "by_name", "from_read"}:
      user_article     the user's article (intake's charge, the case's charge or claim sentences: user, in order), else
                       the charge those sentences name in words, a record by its rubrica («imputato di furto»:
                       charged_by_name, by_name) -- the user's charge, never displaced by the read;
      read_hypothesis  else the first read's hypothesis whose articles hold most elements integrato (read_hypotheses;
                       an article with a record): the most integrato, then the fewest escluso, then one with elements,
                       then the line (principale, subordinata, controparte), then the label -- never the read's order,
                       so the same facts read twice give the same charge;
      rubrica_match    else the store's rubrica: a record whose rubrica holds a name the read gave a hypothesis as
                       rubricated (rubricated_match);
      none             else nothing.
    order: the charge first, then the other charged articles (the user's, the chosen hypothesis's, those named by
    rubrica). Logged (charge_rule) and kept in state["charge"] {article, rule}; the counter charge_rule.
    v79.12.4: (item 1) the persona offesa's or the parte civile's charge is the querela's offence and only it -- a civil
    article the charge or claim sentences name (2043 c.c. …) is a remedy under it («remedies»), never the charge, and
    every rule reads penal articles only; (item 3) a first read's hypothesis (and a rubrica match) whose offence the
    certain facts' event does not fit (lawyer_method.event_mismatch: a death and 590) is never the charge
    («event_refused»); among the charged articles the one with the highest maximum pena from the store («exposure»:
    heaviest) sets the exposure and the riti, never the order. Counters charge_event_mismatch (must be 0 off the
    user's article), charge_event_refused, charge_victim_remedies."""
    from . import issue_records
    repealed = set(repealed or [])
    records = records or {}
    is_victim = victim(posture)
    user = [x for x in dict.fromkeys(user or []) if x not in repealed]
    by_name = [x for x in charged_by_name(case, records, rows) if x not in repealed]
    remedies: List[str] = []
    if is_victim:               # v79.12.4 (item 1): the querela's offence only -- a civil article is its remedy
        remedies = [x for x in dict.fromkeys(user + by_name) if not penal_article(x)]
        user = [x for x in user if penal_article(x)]
        by_name = [x for x in by_name if penal_article(x)]
    event = certain_event(state, case)
    refused: List[str] = []
    scores: Dict[str, List[int]] = {}
    article, rule, extra = "", NO_CHARGE_RULE, []
    if user or by_name:
        article, rule = (user or by_name)[0], "user_article"
    hyps = [] if article else read_hypotheses(state, posture, (), read_log)
    best: Optional[Tuple] = None
    for h, labels in hyps:
        for label in labels:
            rec = records.get(label)
            if label in repealed or not rec or not issue_records.counted(label, str(rec.get("rubrica") or "")):
                continue
            if is_victim and not penal_article(label):
                continue
            if event_refuses(label, event, h):          # v79.12.4 (item 3): never 590 where the victim died
                refused.append(label)
                continue
            proved, excluded, total = element_score(rec)
            scores[label] = [proved, excluded, total]
            key = (-proved, excluded, 0 if total else 1, _LINE_RANK.get(str(h.get("line") or ""), 3), label)
            if best is None or key < best[0]:
                best = (key, label, [x for x in labels if x in records and x not in repealed
                                     and not event_refuses(x, event, h)
                                     and (not is_victim or penal_article(x))], h)
    chosen_h: Optional[Dict[str, Any]] = None
    if not article and best is not None:
        article, rule, extra, chosen_h = best[1], "read_hypothesis", best[2], best[3]
    if not article:
        named = [x for x in rubricated_match(state, records) if x not in repealed
                            and issue_records.counted(x, str(records[x].get("rubrica") or ""))
                            and (str((posture or {}).get("family") or "") != "penale" or penal_article(x))
                            and (not is_victim or penal_article(x))]
        attempt = {"case": case} if _ATTEMPT_RE.search(case or "") else None    # (review) a stated attempt
        refused += [x for x in named if event_refuses(x, event, attempt) and x not in refused]
        named = [x for x in named if not event_refuses(x, event, attempt)]
        if named:
            article, rule, extra, chosen_h = named[0], "rubrica_match", named, attempt
    from . import case_frame
    framed = case_frame.bound(state, "addebito_o_domanda", posture) if isinstance(state, dict) else None
    if framed and article != framed and framed not in repealed:     # v79.14.0 (CASE_FRAME): the frame's charge --
        case_frame.defer(state, "addebito_o_domanda", "posture.choose_charge", article)     # the rule's logged, loses
        article, rule, extra, chosen_h = framed, FRAME_CHARGE_RULE, [], None
    order = list(dict.fromkeys(([article] if article else []) + user + extra + by_name))
    # v79.12.4 (item 3): the heaviest charged fattispecie -- (review) never a circumstance (583 c.p., counted False),
    # the same filter main_threat and charge_rites read
    exposure = heaviest([x for x in order if x in records
                         and issue_records.counted(x, str(records[x].get("rubrica") or ""))], rows) or article
    out = {"article": article, "rule": rule, "order": order, "scores": scores, "by_name": by_name,
           "from_read": list(dict.fromkeys([article] + extra)) if rule == "read_hypothesis" else [],
           "exposure": exposure, "event": event, "event_refused": refused, "remedies": remedies}
    if isinstance(state, dict):
        state["charge"] = {"article": article, "rule": rule,             # v79.12.4 (item 3): a heavier charged one
                           **({"exposure": exposure} if exposure and exposure != article else {})}
        counters = state.setdefault("block_counters", {})
        if not (state.get("client_scope") or {}).get("others") or "charge_rule" not in counters:
            counters["charge_rule"] = rule
            # v79.12.4 (item 3): a chosen charge the certain facts' event does not fit (the user's own article only can
            # be one) and the read's hypotheses refused for it; (item 1) the victim's civil articles kept as remedies
            counters["charge_event_mismatch"] = int(event_refuses(article, event, chosen_h))
            counters["charge_event_refused"] = len(refused)
            counters["charge_victim_remedies"] = len(remedies)
    from . import interactive_flow
    interactive_flow._log("charge_rule", {"article": article, "rule": rule, "order": order, "scores": scores,
                                          "user": user, "by_name": by_name, "exposure": exposure, "event": event,
                                          "event_refused": refused, "remedies": remedies,
                                          "client": (state.get("client_scope") or {}).get("name", "")
                                          if isinstance(state, dict) else ""})
    return out


def penal_article(label: str) -> bool:
    """v79 (C4): an article a criminal charge can be -- the c.p.'s, or a special law's (data/special_laws.json)."""
    from . import special_laws
    # v79.12.4 (item 1, review): a comma label («582 c.p. co. 2», a special comma's record) is its article's code
    code = _COMMA_TAIL_RE.sub("", str(label or "").strip()).partition(" ")[2].strip()
    return code == "c.p." or bool(special_laws.law_of(code))


_COMMA_TAIL_RE = re.compile(r"\s+co(?:mma|\.)\s*\d+.*$", re.IGNORECASE)        # lawyer_method._COMMA_TAIL_RE


def _stems(text: str) -> set:
    from .topic_step import _content_stems
    return set(_content_stems(text or ""))


def alternative_record(rec: Dict[str, Any], alternatives: Sequence[Dict[str, Any]]) -> bool:
    """The record's article is one of the first read's alternative qualifications (its rubrica holds every content stem
    of the alternative's rubricated name, or of its name)."""
    rubrica = _stems(rec.get("rubrica") or "")
    if not rubrica:
        return False
    for a in alternatives or []:
        for name in (a.get("rubricated"), a.get("name")):
            stems = _stems(str(name or ""))
            if stems and stems <= rubrica:
                return True
    return False


def alternative_rows(state: Dict[str, Any], rows: Sequence[Dict[str, Any]] = ()
                     ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """v63: the first read's alternative qualifications, in its order, each with the row of its article: the case's row
    whose rubrica holds every content stem of the qualification's rubricated name or of its name, else the row the
    first read's search found for it (tagged with its name) whose rubrica shares one of its words; None when neither.
    An article goes to one qualification only (the first)."""
    from . import interactive_flow as flow
    from .cenni import named_by
    from .topic_step import shares_name
    pool = [r for r in list(rows or []) + flow._restored_rows((state or {}).get("rows"))
            + flow._restored_rows((state or {}).get("reserve")) if r.get("_article")]
    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    taken: set = set()
    for a in ((state or {}).get("read") or {}).get("alternatives") or []:
        names = [str(x) for x in (a.get("rubricated"), a.get("name")) if x]
        free = [r for r in pool if flow._key(r["_article"]) not in taken]
        row = next((r for name in names for r in free if named_by(name, flow._row_rubrica(r))), None)
        if row is None:
            row = next((r for r in free if a.get("name") and flow._low(r.get("_first_read")) == flow._low(a["name"])
                        and any(shares_name(n, flow._row_rubrica(r)) for n in names)), None)
        if row is not None:
            taken.add(flow._key(row["_article"]))
        out.append((a, row))
    return out


def alternative_labels(state: Dict[str, Any], rows: Sequence[Dict[str, Any]] = ()) -> List[str]:
    from . import interactive_flow as flow
    return [flow._label(flow._key(r["_article"])) for _, r in alternative_rows(state, rows) if r]


def charged_by_name(case: str, records: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]] = ()) -> List[str]:
    """v63: the records the case's charge or claim sentences name by name («procedimento penale per il delitto di
    maltrattamenti in famiglia», «imputato di lesione personale aggravata»): every content stem of the article's rubrica,
    or of the institution the first read found it for (its row's _first_read), in the sentence -- one of them no part
    of the codes owns; within a sentence, a name another match's holds is the less specific one («Omicidio» under
    «Omicidio preterintenzionale») and is left out."""
    from . import intake, interactive_flow as flow
    from .topic_step import GENERIC_STEMS
    names: List[Tuple[str, set]] = [(label, _stems(rec.get("rubrica") or "")) for label, rec in (records or {}).items()]
    for row in rows or []:
        if row.get("_article") and row.get("_first_read"):
            label = flow._label(flow._key(row["_article"]))
            if label in (records or {}):
                names.append((label, _stems(str(row["_first_read"]))))
    out: List[str] = []
    for sentence in intake.sentences(case or ""):
        if not charge_re().search(sentence):
            continue
        words = _stems(sentence)
        hits = [(label, stems) for label, stems in names if stems and stems - GENERIC_STEMS and stems <= words]
        for label, stems in hits:
            if not any(stems < other for x, other in hits if x != label) and label not in out:
                out.append(label)
    return out


def mark_sides(state: Dict[str, Any], posture: Dict[str, Any]) -> Dict[str, Any]:
    """Each record's side (the module's doc), with its reason (rec["side_why"]); rec["charged"] (its place in the
    charge, from 1), rec["alternative"]; the log {role, charged, alternatives, threats: {label: why}}."""
    from . import interactive_flow
    records = (state or {}).get("records") or {}
    role = str((posture or {}).get("role") or "")
    accused = role in ACCUSED_ROLES
    case = interactive_flow.case_text(state)
    rows = interactive_flow._restored_rows((state or {}).get("rows")) + \
        interactive_flow._restored_rows((state or {}).get("reserve"))
    # the charge's order: v71 -- intake's charge first («contesta / imputa / chiede la condanna ai sensi di art. N»)
    aliases = (state or {}).get("charge_aliases") or {}     # v79 (C3): a special law's label -> its store document's
    scope = (state or {}).get("client_scope")                # v79.6 (item 2): one client's charge among several
    if scope and scope.get("others"):
        case = client_text(case, scope)
    from_intake = [aliases.get(x, x) for x in (intake_charges(posture) if not scope else [])]
    from_case = [aliases.get(x, x) for x in charge_labels(case)]
    repealed = set((state or {}).get("abrogated") or [])    # v79.6 (item 3): a repealed article is never the charge
    read_log: Dict[str, Any] = {}
    # v79.10.5 (E1): the charge by rule -- the user's article, the read's hypothesis with most elements integrato, the
    # store's rubrica match -- logged with its rule; the order's first is it, whatever a later step says
    choice = choose_charge(state, posture, records, case, rows, from_intake + from_case, repealed, read_log)
    order = list(choice["order"])
    from_read = list(choice.get("from_read") or [])
    hypothesis = choice["rule"] == "read_hypothesis" and str((posture or {}).get("family") or "") == "penale" \
        and choice["article"] not in choice.get("by_name", [])            # v79 (C4): no charge the user stated
    charged = set(order)
    first = order[0] if order else ""
    alternatives = ((state or {}).get("read") or {}).get("alternatives") or []
    alt_rows = alternative_rows(state)
    alt = {interactive_flow._label(interactive_flow._key(r["_article"])) for _, r in alt_rows if r}
    alt_facts = {interactive_flow._label(interactive_flow._key(r["_article"])): str(a.get("facts") or "")
                 for a, r in alt_rows if r}
    named = {a["article"]: a for a in ((state or {}).get("intake") or {}).get("named_articles") or []
             if isinstance(a, dict) and a.get("article") and a.get("use")}
    threats: Dict[str, str] = {}
    for label, rec in records.items():
        # v79 (C1): the first charged article stays the charge even when the read lists it among the alternatives
        own = (label in alt or alternative_record(rec, alternatives)) and label != first
        for key, value in (("charged", order.index(label) + 1 if label in charged else 0), ("alternative", own),
                           ("alternative_facts", alt_facts.get(label, "") if own else ""),     # v65: the read's verdict
                           ("charge_hypothesis", hypothesis and label in charged),             # v79 (C4)
                           # v79.12.4 (item 3): the charged article with the highest maximum pena -- main_threat's
                           ("exposure", label in charged and label == choice.get("exposure") and len(charged) > 1),
                           # v79.12.4 (item 1): the victim's civil article, a remedy under the querela's offence
                           ("remedy_of", choice["article"] if label in (choice.get("remedies") or []) else "")):
            if value:
                rec[key] = value
            else:
                rec.pop(key, None)
        if rec.get("decomposed_for") == READ_CANDIDATE_WHY:    # v79.12.4 (item 3): a hypothesis not chosen is graded,
            if label in charged:                               # never printed
                rec.pop("unprinted", None)
            else:
                rec["unprinted"] = True
        why = ""
        if label in named:                  # v65: the user's own words decide -- invoked by the client, or used against it
            why = f"usata contro il cliente («{named[label].get('verb')}»)" if named[label]["use"] == "threat" else ""
        elif accused and not own:           # v63: the client's own alternative is never the charge
            if label in charged:
                why = "nominata nell'addebito o nella domanda del caso"
            elif rec.get("relied_by"):
                why = f"la controparte vi si fonda ({rec['relied_by']})"
        if why:
            rec["side"], rec["side_why"] = "threat", why
            threats[label] = why
        else:
            rec["side"] = "weapon"
            rec.pop("side_why", None)
    # v79.12.4 (item 3): every charged article decomposed -- the charge's articles with no record split into elements
    # (must be 0; the record's absence logged as charge_unmarked too)
    from . import issue_records as issue_records_
    undecomposed = [x for x in order if x not in records or not issue_records_.decomposed(records[x])]
    if isinstance(state, dict):
        counters_ = state.setdefault("block_counters", {})
        if not (scope and scope.get("others")) or "charged_undecomposed" not in counters_:
            counters_["charged_undecomposed"] = len(undecomposed)
    out = {"role": role, "charged": order, "alternatives": sorted(x for x, r in records.items()
                                                                            if r.get("alternative")),
           "threats": threats, "charge_from": {"intake": from_intake, "case": from_case,       # v73
                                               "first_read": from_read, **read_log,               # v75, v76
                                               **({"hypothesis": True} if hypothesis else {}),    # v79
                                               **({"aliases": aliases} if aliases else {})},
           "charge_unmarked": [x for x in from_intake + from_case if x not in records],
           "charged_undecomposed": undecomposed, "exposure": choice.get("exposure") or "",        # v79.12.4 (item 3)
           "event_refused": list(choice.get("event_refused") or []),
           "remedies": list(choice.get("remedies") or []),                                       # v79.12.4 (item 1)
           "charge_rule": {k: choice.get(k) for k in ("article", "rule", "scores")}}               # v79.10.5 (E1)
    # v79.10.4 (D3): on the client's records, whose each deontic element is -- the counterparty's never contradicted
    try:
        from . import issue_records
        by_id = {x["id"]: x for x in (state or {}).get("sentences") or [] if isinstance(x, dict) and x.get("id")}
        bound = issue_records.client_bindings(records, issue_records.role_stems(client_role_words(state)), by_id,
                                              role)
        out["bindings"] = bound
        counters = state.setdefault("block_counters", {}) if isinstance(state, dict) else {}
        scoped = bool((state or {}).get("client_scope"))       # several clients: the highest client's
        for key, value in (("counterparty_elements", bound["counterparty_elements"]),
                           ("counterparty_breaches", bound["counterparty_breaches"]),
                           ("counterparty_contradicted", issue_records.counterparty_contradicted(records, by_id))):
            counters[key] = max(int(counters.get(key) or 0), value) if scoped else value
    except Exception as exc:        # noqa: BLE001 -- the sides stand without the bindings
        interactive_flow._log("record_bindings_failed", {"error": f"{type(exc).__name__}: {exc}"[:200]})
    interactive_flow._log("record_sides", out)
    return out


def client_role_words(state: Dict[str, Any]) -> List[str]:
    """v79.10.4 (D3): the client's own role words -- the first read's client role and each listed client's, the
    extracted parties' role for the client's name («acquirente», «committente»); with several clients (client_scope)
    the scoped client's only; [] when none is known."""
    from . import interactive_flow
    st = state or {}
    read = st.get("read") or {}
    scoped = interactive_flow._low((st.get("client_scope") or {}).get("name"))
    try:
        client = scoped or (interactive_flow._low(interactive_flow.client_of(st))
                            if st.get("analysis") or st.get("client") else "")
        facts = interactive_flow._facts(interactive_flow._restored_analysis(st["analysis"])) if st.get("analysis") else {}
    except Exception:               # noqa: BLE001
        client, facts = scoped, {}
    same = lambda name: bool(client) and bool(name) and (name == client or name in client or client in name)  # noqa: E731
    out = [] if scoped else [str((read.get("client") or {}).get("role") or "")]
    out += [str(c.get("role") or "") for c in read.get("clients") or [] if isinstance(c, dict)
            and (not scoped or same(interactive_flow._low(c.get("name"))))]
    for p in (facts or {}).get("parties") or [] if isinstance(facts, dict) else []:
        if isinstance(p, dict) and same(interactive_flow._low(p.get("name"))):
            out.append(str(p.get("role") or ""))
    return [x for x in out if x.strip()]


def procedure_offences(records: Dict[str, Dict[str, Any]], offences: Sequence[Tuple[str, str]],
                       fits: Optional[Sequence[str]] = None) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """v63: (the offences of the procedural section, the log) -- with a charge the case states (rec["charged"]), only
    the charged articles and the alternatives that fit the facts; the topics' other offences, graded candidates, are
    left out. v64: an alternative fits by the alternatives section's verdict (fits: the labels it says «SÌ»), else --
    no such section -- by its record (issue_records.assertable); a rejected one never. None of them among the offences:
    the first charged offence with a record. No charge stated: the offences as they are."""
    from . import interactive_flow as flow, issue_records, rule_quote
    charged = sorted((k for k, r in (records or {}).items() if r.get("charged")), key=lambda k: records[k]["charged"])
    if not charged:
        return list(offences), {}
    alternatives = {k for k, r in records.items() if r.get("alternative")}
    sound = {k for k in alternatives if records[k].get("alternative_facts") != "no"      # v65: SÌ needs a record with
             and not issue_records.contradicted(records[k])}                           # no contradicted element
    fitting = set(fits) & sound if fits is not None else {k for k in sound if issue_records.assertable(records[k])}
    focus = set(charged) | fitting
    kept = [tuple(o) for o in offences if f"{o[0]} c.p." in focus]
    if not kept:
        first = next((k for k in charged if rule_quote.offence(flow._parse_label(k)) or _late_offence(k)), "")
        if first:
            kept = [(flow._parse_label(first)[0], str(records[first].get("rubrica") or ""))]
    return kept, {"focus": sorted(focus), "fits_by": "sezione" if fits is not None else "record",
                  "rejected": sorted(alternatives - fitting), "kept": [f"{n} c.p." for n, _ in kept],
                  "dropped": [f"{o[0]} c.p." for o in offences if tuple(o) not in kept]}


# v65: the labels of the two sides by the client's role -- the client's own, the counterparty's (its opposite); before
# the trial the accused is «indagato», once a parte civile is stated the victim is «parte civile»
_ROLE_LABELS = {"attore": "attore", "convenuto": "convenuto", "imputato": "imputato", "persona_offesa": "persona offesa",
                "parte_civile": "parte civile"}
_OPPOSITE = {"attore": "convenuto", "convenuto": "attore", "imputato": "persona_offesa", "persona_offesa": "imputato",
             "parte_civile": "imputato"}
EARLY_STAGES = ("pre_filing", "indagini")


def role_label(role: str, stage: str = "") -> str:
    """The label of a role («persona offesa»; the accused «indagato» before the trial); "" for none."""
    if role == "imputato" and stage in EARLY_STAGES:
        return "indagato"
    return _ROLE_LABELS.get(role or "", "")


def counterparty_label(role: str, stage: str = "", civil_party: bool = False) -> str:
    """v65: the counterparty's label -- the opposite of the client's role (attore <-> convenuto, imputato <->
    persona offesa, or parte civile once one is constituted); "" for a role with no opposite (terzo, unknown)."""
    other = _OPPOSITE.get(role or "", "")
    if other == "persona_offesa" and civil_party:
        other = "parte_civile"
    return role_label(other, stage)


def _late_offence(label: str) -> bool:
    """A contravvenzione (c.p. Libro III, artt. 650-734): an offence rule_quote.offence (Libro II) does not count."""
    from .plan_then_write import _code_key
    parts = label.split(" ", 1)
    m = re.match(r"\d+", parts[0]) if parts else None
    return bool(m) and len(parts) == 2 and _code_key(parts[1]) == "c.p." and 650 <= int(m.group(0)) <= 734
