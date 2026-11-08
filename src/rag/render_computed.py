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


def blocks(analysis: Dict[str, Any], party: str = "") -> List[Tuple[str, str]]:
    """[(title, body)] of the section, in the template's order (render's docstring); a block without content is left out."""
    from . import fact_signals
    signals = _signals(analysis)
    gated = _gated(analysis)
    out: List[Tuple[str, str]] = []
    mediation = signals.get("mediation") or {}
    if mediation.get("required") and mediation.get("text"):
        out.append(("Mediazione", _sentence(mediation["text"])))
    terms = [t for t in signals.get("terms") or [] if t.get("text") and not _term_gated(t, gated)]
    if terms:
        out.append(("Termini", "\n".join(_term_line(t) for t in terms)))
    tools = [t for t in signals.get("tools") or [] if not _tool_gated(t, gated)]
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
    damages = fact_signals.damages_block(signals) if signals.get("damages") else ""
    if damages.strip():         # the heads of damage are the injured side's: for an accused client, the other side's
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
