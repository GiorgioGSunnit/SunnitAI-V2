"""RENDER_COMPUTED (default false; acts with PLAN_THEN_WRITE_ENABLED): "Profili processuali e termini" is rendered from the
computed block by template, never written by the model.

v52: the procedural section was the one where the model kept rewriting what the code had already computed -- a term
DECORSO proposed as an action, the mediation said not mandatory, the parties to sue as «eventuali proprietari», the
trascrizione without its art. 2653 c.c. With the flag:

  1. render(): the section from the computed facts (fact_signals), in this order, each paragraph under its bold title:
     **Mediazione** (only when mandatory: its conclusion line, art. 5 d.lgs. 28/2010) first; **Termini** (each computed
     term with its status and article, its dates as events -- "cancello apposto nel 2016; avvocato contattato nel
     2018"); **Strumenti processuali** (each tool with its articles and "Nel caso: ..."); **Parti da convenire** (the
     PARTIES_ENFORCED line, else the parties to sue with why); **Competenza** ("Cause relative a diritti reali su
     immobili: giudice del luogo in cui si trova l'immobile (<the facts' comune>)", art. 21 c.p.c.); **Trascrizione della
     domanda giudiziale (art. 2653, n. 1, c.c.)** with the action transcribed; then the criminal profile and the heads of
     damage when computed. The parties' names capitalised. No model call; a tool or a term whose article is the other
     side's (the action gate) is not rendered.
  2. The model writes the other five sections; the rendered text reaches sections 1, 4 and 6 as read-only context
     (READ_ONLY_TITLE, READ_ONLY_RULE): not rewritten, not contradicted.
  3. The articles the action gate tags adverse (analysis["action_gate"]) leave the plan's remedies -- each issue's
     norms, the main remedies (rimedi) -- and appear only in the objections block (OBJECTIONS_TITLE, plan
     ["obiezioni_norme"]); after writing, a sentence citing one with no word of an objection or a reply is dropped.
  4. restore(): after the answer's last edits (defensive_generation), the section is the rendered text again.

Logged as case_analysis_render_computed ({blocks, chars, gated, restored}). Off, nothing changes.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import feature_flags

FLAG = "RENDER_COMPUTED"
SECTION_TITLE = "Profili processuali e termini"
READ_ONLY_TITLE = "PROFILI PROCESSUALI GIÀ REDATTI (SOLA CONSULTAZIONE)"
READ_ONLY_RULE = ("È il testo della sezione «Profili processuali e termini», ricavato dai fatti calcolati: non riscriverlo e "
                  "non contraddirlo (termini, strumenti, parti, mediazione); se serve, rinvia a quella sezione.")
OBJECTIONS_TITLE = "OBIEZIONI DELLA CONTROPARTE"
OBJECTIONS_RULE = ("Queste norme fondano azioni che spettano alla controparte, non a {party}: citale solo come argomento "
                   "della controparte, con la replica di {party}; mai tra i rimedi o le domande di {party}.")
_SEPARATE = {"competenza_territoriale": "Competenza", "trascrizione_domanda": "Trascrizione"}
_LITIS = "litisconsorzio"
RENDERED_KEY = "_rendered_section5"
TRASCRIZIONE_TITLE = "Trascrizione della domanda giudiziale (art. 2653, n. 1, c.c.)"
_COMUNE_RE = re.compile(r"\b(?:in|nel)\s+comune\s+di\s+([A-ZÀ-Ü][\w'’-]+(?:\s+[A-ZÀ-Ü][\w'’-]+)*)|\bcomune\s+di\s+"
                        r"([A-ZÀ-Ü][\w'’-]+(?:\s+[A-ZÀ-Ü][\w'’-]+)*)")


def _comune(analysis: Dict[str, Any]) -> str:
    """The comune the facts place the land in ("terreni in comune di Manciano"): the extracted facts, then the frame's."""
    import json
    signals = _signals(analysis)
    texts = [json.dumps(signals.get("facts") or {}, ensure_ascii=False)]
    texts += [str(f) for f in ((analysis or {}).get("frame") or {}).get("facts") or []]
    for text in texts:
        m = _COMUNE_RE.search(text)
        if m:
            return m.group(1) or m.group(2)
    return ""


def _transcribed_action(tool: Dict[str, Any], gated: set) -> str:
    """The action whose demand is transcribed, from the tool's computed fact; the negatoria left out when the action gate
    gives it to the other side ("azione confessoria/negatoria" -> "azione confessoria")."""
    m = next((re.search(r"\((azione [^)]+)\)", str(x)) for x in tool.get("matched") or [] if "(azione" in str(x)), None)
    action = m.group(1) if m else ""
    if "949 c.c." in gated:
        action = action.replace("confessoria/negatoria", "confessoria")
    return action


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def _signals(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return (analysis or {}).get("fact_signals") or {}


def _gated(analysis: Dict[str, Any]) -> set:
    return set((analysis or {}).get("action_gate") or {})


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).strip()
    return text if not text or text.endswith((".", "!", "?", "»")) else text + "."


# a dated event of a term, in a few words: "Domitilla appone un cancello chiuso a chiave" -> "cancello apposto"
_EVENT_LABELS = (
    (r"\b(cancell\w*|sbarr\w*|recinzion\w*|lucchett\w*|catena|transenn\w*|muro|paletti)\b.*|"
     r"\b(?:appon|pon|install|colloc|chiud|erig|costru)\w*\b.*?\b(cancell\w*|sbarr\w*|recinzion\w*|lucchett\w*|catena|muro)",
     lambda m: f"{(m.group(1) or m.group(2)).lower()} apposto"),
    (r"\b(?:avvocat\w*|legale)\b", lambda m: "avvocato contattato"),
    (r"\bdiffid\w*", lambda m: "diffida inviata"),
    (r"\b(?:citazion\w*|ricors\w*|domanda giudiziale)\b", lambda m: "domanda proposta"),
    (r"\bvend\w*|\bcompravendit\w*", lambda m: "vendita"),
    (r"\bconsegn\w*", lambda m: "consegna"),
    (r"\bscopr\w*|\bscopert\w*", lambda m: "scoperta"),
    (r"\bultimo uso\b", lambda m: "ultimo uso"),
)


def _event_label(what: str) -> str:
    low = " ".join(str(what or "").lower().split())
    for pattern, label in _EVENT_LABELS:
        m = re.search(pattern, low)
        if m:
            return label(m)
    return " ".join(str(what or "").split())


def _dated(term: Dict[str, Any], text: str) -> str:
    """The term's "(Domitilla appone un cancello chiuso a 2016, Leonardo si rivolge ad un avvocato 2018)" as "(cancello
    apposto nel 2016; avvocato contattato nel 2018)"; a text without both events as they are written, unchanged."""
    text = re.sub(r"\b(dall['’]ultimo uso|dall['’]ultimo esercizio|dalla consegna|dalla scoperta),\s*(\d{4})\b", r"\1 nel \2", text)
    ends = [e for e in (term.get("from"), term.get("to")) if isinstance(e, dict) and e.get("what") and e.get("date")]
    if len(ends) != 2:
        return text
    written = ", ".join(f"{e['what']} {e['date']}" for e in ends)
    return text.replace(f"({written})", "(" + "; ".join(f"{_event_label(e['what'])} nel {e['date']}" for e in ends) + ")")


def _term_line(term: Dict[str, Any]) -> str:
    from . import answer_invariants
    text = _dated(term, str(term.get("text") or "").strip())
    if answer_invariants.enabled():
        text = answer_invariants.plain_status(text)
    ref = str(term.get("ref") or "").strip()
    if ref and not ref.lower().startswith("art"):
        ref = f"art. {ref}"
    if not ref or ref in text:
        return "- " + _sentence(text)
    return "- " + _sentence(f"{text}, {ref}" if text.rstrip().endswith(")") else f"{text} ({ref})")


def _tool_line(tool: Dict[str, Any]) -> str:
    # a rubrica-based text ("Foro per le cause relative a diritti reali. (art. 21 c.p.c.).") without its inner stop
    text = _sentence(re.sub(r"\.\s+\((artt?\.)", r" (\1", str(tool.get("text") or "")))
    matched = [str(m).strip() for m in tool.get("matched") or [] if str(m).strip()]
    return text + (f" Nel caso: {'; '.join(matched).rstrip('.')}." if matched else "")


def _term_gated(term: Dict[str, Any], gated: set) -> bool:
    """A term of an action the action gate gives the other side (the negatoria's "non soggetta a prescrizione", 949 c.c.)."""
    ref = str(term.get("ref") or "")
    return any(re.search(rf"\b{re.escape(g.split(' ')[0])}\b", ref) and g.split(" ", 1)[1] in ref for g in gated)


def _tool_gated(tool: Dict[str, Any], gated: set) -> bool:
    return any(f"{r[0]} {r[1]}" in gated for r in tool.get("refs") or [] if len(r) >= 2)


_CIVIL_CODES = frozenset({"c.c.", "c.p.c."})
# v56: the heads of damage only for the injured side -- the danneggiato, the erede, the crime's victim
DAMAGED_RE = r"danneggiat|\bered[ei]\b|persona offesa|parte civile|vittim"


def civil_tools_allowed(analysis: Dict[str, Any]) -> bool:
    """v56 (v55-p1, v55-h10 §4): a tool on c.c. / c.p.c. articles only (696, 700 c.p.c.) is offered only when the case's
    main area (its first: pool_gates.gate_areas) allows the civil codes (pool_gates.AREA_CODES, POOL_GATE_AREA's); no
    known area: offered."""
    from . import pool_gates
    areas = [a for a in pool_gates.gate_areas(analysis) if a not in pool_gates.NO_CODES]
    return not areas or bool(pool_gates.AREA_CODES.get(areas[0], frozenset()) & _CIVIL_CODES)


def _civil_only(tool: Dict[str, Any]) -> bool:
    from .plan_then_write import _code_key
    codes = {_code_key(str(r[1])) for r in tool.get("refs") or [] if len(r) == 2}
    return bool(codes) and codes <= _CIVIL_CODES


def client_damaged(analysis: Dict[str, Any], party: str = "") -> bool:
    """The client is the danneggiato or the erede (DAMAGED_RE): in its name and position, or its role in the facts."""
    facts = _signals(analysis).get("facts") or {}
    roles = " ".join(str(p.get("role") or "") for p in facts.get("parties") or []
                     if isinstance(p, dict) and re.search(r"\bcliente\b", str(p.get("role") or ""), re.IGNORECASE))
    text = f"{party} {(analysis or {}).get('parte_assistita') or ''} {(analysis or {}).get('posizione') or ''} {roles}"
    return bool(re.search(DAMAGED_RE, text.lower()))


PROCEDURE_TITLE = "Procedibilità e competenza"
DEFAULT_PROCEDURE = ("d'ufficio", "tribunale")        # v58: a delitto not in topic_tables.CRIMINAL_PROCEDURE
OFFENCES_MAX = 5


def procedure_lines(found: Sequence[tuple], offences: Sequence[tuple], dated: bool) -> List[str]:
    """The «Procedibilità e competenza» lines: each table offence (topic_tables.CRIMINAL_PROCEDURE) found or among the
    chosen topics' delitti, with its own procedibilità and competenza; the other delitti (offences: [(number, rubrica)]),
    by default procedibili d'ufficio before the tribunale, in one line (v59: v58-h7 had five identical lines); while the
    date of the fact is unknown, what runs from it."""
    from . import topic_tables
    table = topic_tables.CRIMINAL_PROCEDURE
    shown = list(dict.fromkeys([tuple(r) for r in found] + [(str(n), "c.p.") for n, _ in offences
                                                             if (str(n), "c.p.") in table]))
    lines = []
    for ref in shown:
        name, how, court = table[ref]
        lines.append(f"- {name[:1].upper() + name[1:]} (art. {ref[0]} {ref[1]}): procedibile {how}"
                     + (f"; competenza: {court}" if court else "") + ".")
    others = []
    for number, rubrica in [tuple(o) for o in offences][:OFFENCES_MAX]:
        if (str(number), "c.p.") in shown:
            continue
        name = " ".join(str(rubrica or "").split()).rstrip(" .") or "Delitto"
        others.append(f"{name[:1].upper() + name[1:]} (art. {number} c.p.)")
    if others:
        lines.append(f"- {DEFAULT_LIST_JOIN.join(others)}: "
                     + (DEFAULT_LINE_MANY if len(others) > 1 else DEFAULT_LINE_ONE).format(*DEFAULT_PROCEDURE) + ".")
    if not dated:
        querela = any("querela" in table[r][1] for r in shown)
        lines.append("- Data del fatto: non nota. Da essa decorre la prescrizione del reato (art. 157 c.p.)"
                     + ("; per i reati a querela, i tre mesi per proporla (art. 124 c.p.)" if querela else "") + ".")
    return lines


DEFAULT_LINE_ONE = "procedibile {0}; competenza: {1}"
DEFAULT_LINE_MANY = "procedibili {0}; competenza: {1}"
DEFAULT_LIST_JOIN = "; "


def criminal_procedure(analysis: Dict[str, Any]) -> str:
    """v57 (J): in a criminal case, each offence the case's tables hold (topic_tables.offences) with its procedibilità
    and competenza; while the date of the fact is unknown, what runs from it -- the prescription (art. 157 c.p.) and,
    for an offence a querela starts, its three months (art. 124 c.p.). "" otherwise. v59 (v58-p2: truffa chosen, no
    §4, the frame's first area not penale): the chosen topics' delitti (analysis["offences"]) make it a criminal answer
    whatever the frame's areas. v60 (v59-n1: usura framed civile, the one topic 43, 45 c.p., no delitto among them): chosen
    topics all in c.p./c.p.p. (analysis["topic_refs"]) make it a criminal answer too."""
    from . import fact_signals, pool_gates, topic_tables
    from .plan_then_write import _code_key
    areas = pool_gates.gate_areas(analysis)
    offences = [tuple(o) for o in (analysis or {}).get("offences") or []]
    refs = [tuple(r) for r in (analysis or {}).get("topic_refs") or [] if len(r) == 2]
    criminal = (bool(areas) and fact_signals.family(areas[0]) == "penale"
                or bool(refs) and all(_code_key(str(c)) in ("c.p.", "c.p.p.") for _, c in refs))
    if not criminal and not offences:
        return ""
    found = topic_tables.offences(analysis, areas) if criminal else []
    facts = _signals(analysis).get("facts") or {}
    dated = any(str(e.get("date") or "").strip() and fact_signals.parse_date(str(e.get("date")))
                and re.search(fact_signals._OFFENCE + "|" + fact_signals._DAMAGE + r"|spar|ucci|mort",
                              str(e.get("what") or "").lower()) for e in facts.get("events") or [])
    return "\n".join(procedure_lines(found, offences, dated))


# v59 (v55-l4, n9: usucapione chosen, no «Mediazione»): the chosen topics' c.c. articles whose matter needs mediation
# first (art. 5, co. 1, d.lgs. 28/2010) when the facts' own reading (fact_signals.mediation) found none -- by article
MEDIATION_RANGES = (("condominio", 1117, 1139), ("diritti reali", 832, 1172), ("successioni ereditarie", 456, 768),
                    ("patti di famiglia", 768, 769), ("locazione", 1571, 1614), ("comodato", 1803, 1812),
                    ("contratti assicurativi, bancari e finanziari", 1834, 1932),
                    ("somministrazione", 1559, 1570), ("contratto d'opera", 2222, 2238),
                    ("società di persone", 2251, 2324), ("affitto di aziende", 2562, 2562))


def mediation_of(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The mediation the facts require (fact_signals.mediation), else the one the chosen topics' matter requires
    (analysis["topic_refs"]: their anchors and matched articles, interactive_flow): {"required", "subject", "text"}."""
    from . import fact_signals
    from .plan_then_write import _code_key
    mediation = _signals(analysis).get("mediation") or {}
    if mediation.get("required"):
        return mediation
    for number, code in [tuple(r) for r in (analysis or {}).get("topic_refs") or [] if len(r) == 2]:
        m = re.match(r"\d+", str(number))
        if not m or _code_key(str(code)) != "c.c.":
            continue
        subject = next((name for name, lo, hi in MEDIATION_RANGES if lo <= int(m.group(0)) <= hi), None)
        if subject:
            return {"required": True, "subject": subject,
                    "text": f"Mediazione obbligatoria ({fact_signals.MEDIATION_REF}): la materia è {subject}, il suo "
                            "esperimento è condizione di procedibilità della domanda."}
    return mediation


def blocks(analysis: Dict[str, Any], party: str = "") -> List[Tuple[str, str]]:
    """[(title, body)] of the section, in the template's order (render's docstring); a block without content is left out."""
    from . import fact_signals
    signals = _signals(analysis)
    gated = _gated(analysis)
    out: List[Tuple[str, str]] = []
    mediation = mediation_of(analysis)
    if mediation.get("required") and mediation.get("text"):
        out.append(("Mediazione", _sentence(mediation["text"])))
    terms = [t for t in signals.get("terms") or [] if t.get("text") and not _term_gated(t, gated)]
    listed = (analysis or {}).get("listed_topics")
    if listed:              # v59 (v58-l3 §4: the usucapione with no such topic): the terms the listed topics carry
        from . import topic_step
        terms = [t for t in terms if topic_step.carries_term(t, listed)]
    if terms:
        out.append(("Termini", "\n".join(_term_line(t) for t in terms)))
    tools = [t for t in signals.get("tools") or [] if not _tool_gated(t, gated)]
    if not civil_tools_allowed(analysis):
        tools = [t for t in tools if not _civil_only(t)]
    parties = ""
    try:
        parties = fact_signals.parties_line(signals, analysis)
    except Exception as exc:            # logged by the caller's step; the plain list below stands
        fact_signals._log_exception("render_computed.parties_line", exc)
    general = [t for t in tools if t.get("id") not in _SEPARATE and t.get("id") != fact_signals.CRIMINAL_TOOL
               and not (t.get("id") == _LITIS and parties)]
    if general:
        out.append(("Strumenti processuali", "\n".join(f"- {_tool_line(t)}" for t in general)))
    if parties:
        out.append(("Parti da convenire", _sentence(parties)))
    elif signals.get("defendants"):
        names = "; ".join(f"{d['name']} ({d['why']})" if d.get("why") else str(d["name"])
                          for d in signals["defendants"] if d.get("name"))
        if names:
            out.append(("Parti da convenire", _sentence(f"Parti da convenire: {names}")))
    competenza = next((t for t in tools if t.get("id") == "competenza_territoriale"), None)
    if competenza:
        place = _comune(analysis)
        out.append(("Competenza", "Cause relative a diritti reali su immobili: giudice del luogo in cui si trova "
                                  "l'immobile" + (f" ({place})" if place else "") + " (art. 21 c.p.c.)."))
    trascrizione = next((t for t in tools if t.get("id") == "trascrizione_domanda"), None)
    if trascrizione:
        action = _transcribed_action(trascrizione, gated)
        out.append((TRASCRIZIONE_TITLE, f"Va trascritta la domanda giudiziale" + (f" ({action})" if action else "") + "."))
    criminal = next((t for t in tools if t.get("id") == fact_signals.CRIMINAL_TOOL), None)
    if criminal:
        out.append(("Profilo penale", _tool_line(criminal)))
    procedure = criminal_procedure(analysis)
    if procedure:
        out.append((PROCEDURE_TITLE, procedure))
    damages = fact_signals.damages_block(signals) if signals.get("damages") else ""
    if damages.strip() and client_damaged(analysis, party):         # the heads of damage are the injured side's: for an accused client, the other side's
        side = signals.get("side") or fact_signals.client_side(party or str((analysis or {}).get("parte_assistita") or ""))
        title = {"accused": "Voci di danno che la controparte può far valere", "victim": "Voci di danno"}.get(
            side, "Voci di danno da considerare")
        out.append((title, damages.strip()))
    return out


def _named(text: str, analysis: Dict[str, Any]) -> str:
    """The parties' names with their capital where the computed facts wrote them lower case ("fondo di leonardo")."""
    names = {str(p.get("name") or "").strip() for p in (_signals(analysis).get("facts") or {}).get("parties") or []
             if isinstance(p, dict)}
    for name in sorted((n for n in names if n and n[:1].isupper()), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name.lower())}\b", name, text)
    return text


def render(analysis: Dict[str, Any], party: str = "") -> str:
    """The section's text, each block a paragraph under its bold title; "" with nothing computed. party: the client's
    name and position (the plan's), for the side of the heads of damage."""
    return _named("\n\n".join(f"**{title}**\n\n{body}" for title, body in blocks(analysis, party)), analysis)


def read_only_block(rendered: str) -> str:
    return f"{READ_ONLY_TITLE}:\n{rendered}\n{READ_ONLY_RULE}" if rendered.strip() else ""


# --- the action gate's articles: out of the remedies, into the objections -------------------------------------------

def move_gated(plan: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """(plan, moved): each article the action gate tags adverse out of every issue's norms and out of the main remedies,
    listed in plan["obiezioni_norme"] [{article, reason, issue}]; an issue left with no norm keeps its argument."""
    gate = (analysis or {}).get("action_gate") or {}
    if not gate or not plan:
        return plan, []
    moved: List[Dict[str, str]] = []
    issues = []
    for q in plan.get("questioni") or []:
        out = [r for r in q.get("norme") or [] if f"{r[0]} {r[1]}" in gate]
        for r in out:
            moved.append({"article": f"{r[0]} {r[1]}", "reason": gate[f"{r[0]} {r[1]}"], "issue": q.get("titolo") or ""})
        issues.append({**q, "norme": [r for r in q.get("norme") or [] if f"{r[0]} {r[1]}" not in gate]} if out else q)
    plan = {**plan, "questioni": issues}
    remedies = plan.get("rimedi") or {}
    if remedies.get("article"):
        number = re.sub(r"^art\.\s*", "", remedies["article"]).split(" ")[0]
        hit = next((k for k in gate if k.split(" ")[0] == number), None)
        if hit:
            plan = {k: v for k, v in plan.items() if k != "rimedi"}
            if not any(m["article"] == hit for m in moved):
                moved.append({"article": hit, "reason": gate[hit], "issue": "rimedi"})
    if moved:
        objections = list(plan.get("obiezioni_norme") or [])
        have = {o["article"] for o in objections}
        objections += [m for m in moved if m["article"] not in have and not have.add(m["article"])]
        plan = {**plan, "obiezioni_norme": objections,
                "norme_contrarie": list(dict.fromkeys(list(plan.get("norme_contrarie") or [])
                                                      + [o["article"] for o in objections]))}
    return plan, moved


_ROLE_WORDS = {"resists_servitude": "contesta una servitù altrui", "claims_ownership": "rivendica la proprietà",
               "claims_servitude": "vanta una servitù", "possessor": "possiede il bene",
               "landlocked_owner": "è proprietario del fondo intercluso"}


def _italian(reason: str) -> str:
    """The action gate's reason in the writer's words: "spetta a chi resists_servitude o ..." -> "spetta a chi contesta
    una servitù altrui o ..."."""
    return re.sub(r"\b(" + "|".join(_ROLE_WORDS) + r")\b", lambda m: _ROLE_WORDS[m.group(1)], str(reason or ""))


def objections_block(plan: Dict[str, Any]) -> str:
    items = plan.get("obiezioni_norme") or []
    if not items:
        return ""
    from .plan_then_write import _code_key
    party = plan.get("parte_assistita") or "la parte assistita"
    lines = []
    for o in items:
        number, _, code = o["article"].partition(" ")
        lines.append(f"- art. {number} {_code_key(code)}: {_italian(o['reason'])}")
    return f"{OBJECTIONS_TITLE}:\n" + "\n".join(lines) + "\n" + OBJECTIONS_RULE.format(party=party)


_OBJECTION_WORDS_RE = re.compile(r"controparte|obiezion|obiett|eccepi|potrebbe(?:ro)?\s+(?:invocare|sostenere|richiamare|"
                                 r"agire|opporre)|replica|tesi avversaria|contrari", re.IGNORECASE)


def gated_as_support(title: str, text: str, plan: Dict[str, Any],
                     documents: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): each sentence citing an objections-block article with no word of an objection or a reply dropped."""
    from .plan_then_write import _citations, _drop_spans, _same_ref, _sentence_spans
    refs = [tuple(o["article"].split(" ", 1)) for o in plan.get("obiezioni_norme") or []]
    if not refs:
        return text, []
    spans, log = [], []
    for s, e in _sentence_spans(text):
        sentence = text[s:e]
        cited = [ref for _, _, rs in _citations(sentence, documents or []) for ref in rs]
        hit = next((f"{r[0]} {r[1]}" for r in refs if any(_same_ref(r, c) for c in cited)), None)
        if hit and not _OBJECTION_WORDS_RE.search(sentence):
            spans.append((s, e))
            log.append({"section": title, "kind": "adverse_article_as_support", "article": hit,
                        "how": "sentence dropped", "sentence": " ".join(sentence.split())[:300]})
    return (_drop_spans(text, spans) or text, log) if spans else (text, [])


# --- the answer: the rendered section back after the last edits ------------------------------------------------------

def _section_bounds(draft: str, title: str = SECTION_TITLE) -> Optional[Tuple[int, int]]:
    """(start of its text, end) of the section titled `title` in a planned answer ("**5. Profili ...**")."""
    head = re.search(rf"(?m)^\*\*\d+\.\s*{re.escape(title)}\*\*[ \t]*\n", draft or "")
    if not head:
        return None
    nxt = re.search(r"(?m)^\*\*\d+\.\s*[^*\n]+\*\*[ \t]*$", draft[head.end():])
    end = head.end() + nxt.start() if nxt else len(draft)
    return head.end(), end


def restore(draft: str, analysis: Dict[str, Any]) -> Tuple[str, bool]:
    """(draft, changed): the section's text the rendered one again, when an edit after writing changed it."""
    rendered = (analysis or {}).get(RENDERED_KEY)
    bounds = _section_bounds(draft) if rendered else None
    if not bounds:
        return draft, False
    s, e = bounds
    if draft[s:e].strip() == rendered.strip():
        return draft, False
    tail = "\n\n" if e < len(draft) else ""
    return draft[:s] + "\n" + rendered.strip() + tail + draft[e:].lstrip("\n"), True
