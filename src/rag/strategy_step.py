"""v79.18.0 (B): the strategy as a verified step (flag ANALYSIS_STEP, data/strategy_step.json), after the analysis step
and the rulings record, in place of the retired objections call (interactive_flow.method_objections).

One model call (thinking off, temperature 0, a JSON reply; one re-ask within «deadline_s») on an input of ids and
verbatim texts only (state_input, from the flow's state after the analysis step; rulings_for, the registry's rulings on
the set's articles): the pretesa and the client's side; the governs-pretesa set with its tiers (interactive_flow.
binding_pretesa «tiers»: si verificato > dipende verificato > non verificato; the verified lines' articles too) -- each
article's schema elements with their verified esito and cited S-ids (the analysis step's tables; an article the step
did not assess: every element «da_provare», said so); the facts (S-ids, the record's stato and speaker); the
counterparty's allegations; the registry's rulings on those articles (G-ids as the registry numbers them, their
verified passage); the quesiti.

The reply: linee (2-3: articolo, eid, tipo fondare | contestare | riqualificare | eccepire, rapporto principale |
subordinata | alternativa | cumulativa, perche_prima, cita_fatti, cita_pronunce, obiezione {testo, risposta_cita_fatti
| «nessuna risposta nei fatti», cita_pronunce}); passi (the principale's da_provare / contestato elements, what would
prove each); and (item 12) passi_immediati, contro_difesa (one per allegation), verifiche_documentali -- all id-cited.

Verification (verify, finish): every id exists; a linea's articolo is in the set, its eid belongs to it; «contestare»
only on a contestato / da_provare element, «fondare» only on a provato one -- else rejected, re-asked once; fewer than
two verified linee -- re-asked once. forza is the code's, from the tables, by side (forza): the defence's the count of
contestato + da_provare on the line's article (more = stronger); a claimant's per element (item 10): the eid's esito
first (provato > contestato > da_provare > escluso), then fewer further conditions in the same comma, then the provato
count -- a remedy whose own element is da_provare never ranks above one whose element is provato. The ranking rule
(rank): a linea may stand above one with strictly greater forza only when it cites a G-id; otherwise the code reorders
by forza, strategy_rank_reordered = 1, both orders logged; ties keep the model's order. The first ranked linea is the
principale. An obiezione without risposta_cita_fatti prints «senza risposta nei fatti narrati», never an invented
answer. A contro_difesa entry the model did not give is added by the code as «senza risposta»; a passo the model did
not give for an open element of the principale is added as a label.

Rendering (the code's labels, data «rendering»): §1's line sentences from the ranked linee (summary_text), §6
(solutions_text) in the order passi immediati -> linee per forza (the table, forza shown as «provato x / contestato y /
da provare z») -> risposta ai quesiti -> contro-difesa -> verifiche documentali, §7 the obiezioni with their verified
answers (objection_items: the cited facts as written, the rulings by their registry label, which compose renders as
estremi), §9's risk word from the principale's forza by the data's thresholds (risk_word), §10 the passi (passi_points).
The step failing, timing out or answering nothing: the analysis step's rendering as before (lawyer_method.compose).

Counters (COUNTERS, into block_counters): strategy_calls, strategy_reasked, strategy_lines, strategy_rank_reordered,
strategy_line_rejected, strategy_objections_unanswered, strategy_rulings_cited, strategy_seconds (the brief's) and
strategy_failed, strategy_bad_id, strategy_passi_added, strategy_contro_difesa_added, strategy_entries_uncited (mine).
Logged «strategy_step» with the prompt's hash (prompt_hash).
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait as _wait
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FLAG = "ANALYSIS_STEP"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "strategy_step.json")
COUNTERS = ("strategy_calls", "strategy_reasked", "strategy_lines", "strategy_rank_reordered", "strategy_line_rejected",
            "strategy_objections_unanswered", "strategy_rulings_cited", "strategy_seconds",
            "strategy_failed", "strategy_bad_id", "strategy_passi_added", "strategy_contro_difesa_added",
            "strategy_entries_uncited",
            # v79.18.1 (C3, C4, C5): no «riqualificare» linea for a penal defence with another set article; an
            # obiezione whose risposta rests on an element whose esito does not fit the side; a passo that is no
            # concrete fact to acquire
            "strategy_riqualificazione_missing", "strategy_objection_mismatch", "strategy_passi_not_concrete",
            # v79.18.2 (item 2): every proposed line rejected for the side's rules -- the re-ask states the side; (item
            # 6) a riqualificazione line on an «alternativa» article moved below the first set line (never principale)
            "strategy_side_reasked", "strategy_alternativa_demoted",
            "strategy_risk_capped",      # v79.18.3 (item 1): a favorevole word capped at incerto (unverified set / one element)
            # v79.18.5: item 1 a factual sentence of a free-text field with no S-id (dropped), any other sentence the
            # Q&A's reviewer dropped (an unknown id, an unfitted article, a rule without its A-id); item 3 the ids rendered
            # as text and the raw ids a final sweep still found; item 5 the objections with a verified accusa_forte and the
            # objections dropped for want of any S-id
            "strategy_sentences_dropped_no_fact", "strategy_sentences_dropped", "strategy_ids_rendered",
            "residues_dropped", "strategy_objection_strongest", "strategy_objection_dropped")
OK, FAILED, MALFORMED, TIMEOUT, NO_INPUT, OFF = "ok", "failed", "malformed", "timeout", "no_input", "off"
PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO = "provato", "contestato", "da_provare", "escluso"
ESITO_RANK = {PROVATO: 3, CONTESTATO: 2, DA_PROVARE: 1, ESCLUSO: 0}
OPEN = (CONTESTATO, DA_PROVARE)
SIDE_CLAIMANT, SIDE_DEFENCE, SIDE_NEUTRAL = "attore", "difesa", "neutro"
TIER_ALTERNATIVA = "alternativa"         # v79.18.2 (item 6): a «no» in the pretesa's Titolo / Capo -- riqualificare only
TIER_UNVERIFIED_SET = "non_verificato"   # v79.18.3 (item 1): the binding's lowest tier (interactive_flow.TIER_UNVERIFIED)
FAVOREVOLE, INCERTO, SFAVOREVOLE = "favorevole", "incerto", "sfavorevole"
PRINCIPALE, SUBORDINATA = "principale", "subordinata"
_S_RE = re.compile(r"^S\d+$")
_G_RE = re.compile(r"^G\d+$")
_FACT_ID_RE = re.compile(r"\[S\d+\]")                                   # v79.18.5 (item 1): an inline fact id
_ANY_ID_RE = re.compile(r"\[(?:S\d+|G\d+|A[0-9a-z]+(?::E\d+|\.\d+)?)\]")     # every inline id the renderer turns to text
_RESIDUE_RE = re.compile(r"\[?\b(?:S\d{1,3}|G\d{1,3}|A[0-9]+[a-z]+(?::E\d+|\.\d+)?)\b\]?")   # (item 3) a raw id left on the page
SITE_STRATEGY = "strategia"              # v79.18.5 (item 1): the reviewer's site for the strategy's free text

_DATA: Dict[str, Any] = {}


def enabled() -> bool:
    from . import feature_flags
    return feature_flags.enabled(FLAG)


def config() -> Dict[str, Any]:
    if "d" not in _DATA:
        with open(DATA_FILE, encoding="utf-8") as f:
            _DATA["d"] = json.load(f)
    return _DATA["d"]


def prompt_system() -> str:
    return "\n".join(config().get("prompt_system") or [])


def prompt_hash() -> str:
    return hashlib.sha256(prompt_system().encode("utf-8")).hexdigest()[:12]


def rendering() -> Dict[str, Any]:
    return config().get("rendering") or {}


def _log(event: str, data: Dict[str, Any]) -> None:
    try:
        from .verbose_logger import vlog
        vlog(event, data)
    except Exception:           # noqa: BLE001 -- logging never fails the answer
        pass


def _flat(value: Any) -> str:
    return " ".join(str(value or "").split())


def _stop(text: Any) -> str:
    return _flat(text).rstrip(" .;:")


def _cell(text: Any) -> str:
    return _flat(text).replace("|", "/") or "—"


def empty_counters() -> Dict[str, Any]:
    return {k: 0 for k in COUNTERS}


def _as():
    from . import analysis_step
    return analysis_step


def _ap():
    from . import applicability
    return applicability


# --- the input --------------------------------------------------------------------------------------------------------

def state_input(state: Dict[str, Any], verified: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The step's input from the flow's state after the analysis step and verified_object: {pretesa, client, side,
    charge, articles [{article, token, rubrica, tier, in_set, valutato, elements [{eid, testo, comma, esito, cita_fatti,
    valutato, eccezione, procedurale, parte_che_afferma}]}], facts [{id, text, certainty, said_by}], allegations [S-ids
    the counterparty asserts], quesiti, quesiti_articles {Qn: labels}, lines (the analysis step's), honesty}. None with
    no verified view, no set article with elements (nothing to strategise) or the flag off."""
    if not enabled() or not isinstance(verified, dict):
        return None
    A = _as()
    a = verified.get("analysis") if isinstance(verified.get("analysis"), dict) else {}
    binding = state.get("binding_pretesa") if isinstance(state.get("binding_pretesa"), dict) else {}
    norms = [n for n in state.get("analysis_norms") or [] if isinstance(n, dict) and n.get("article")]
    tiers = dict(binding.get("tiers") or {})
    in_set = [str(x) for x in binding.get("principale") or []]
    alternative = [str(x) for x in binding.get("alternative") or []]        # v79.18.2 (item 6): riqualificazione only
    labels = list(in_set)
    for kind in A.LINE_KINDS:
        x = a.get(kind)
        if x and x.get("articolo") and x.get("stato") == A.VERIFIED and not any(
                A.same_article(x["articolo"], y) for y in labels):
            labels.append(str(x["articolo"]))
    labels += [x for x in alternative if not any(A.same_article(x, y) for y in labels)]     # v79.18.2 (item 6)
    esiti = a.get("elementi") if isinstance(a.get("elementi"), dict) else {}
    articles = []
    for label in labels:
        n = next((x for x in norms if A.same_article(x["article"], label)), None)
        if not n or n.get("remedy") or not n.get("elements"):
            continue
        got = {e.get("eid"): e for e in (esiti.get(n["article"]) or esiti.get(label) or []) if isinstance(e, dict)}
        elements = []
        for e in n["elements"]:
            if not isinstance(e, dict) or not e.get("eid"):
                continue
            v = got.get(e["eid"]) or {}
            elements.append({"eid": str(e["eid"]), "testo": _flat(e.get("testo")), "comma": e.get("comma"),
                             "esito": v.get("esito") if v.get("esito") in ESITO_RANK else DA_PROVARE,
                             "cita_fatti": [str(i) for i in v.get("cita_fatti") or []],
                             "valutato": bool(v), "eccezione": bool(e.get("eccezione")),
                             "procedurale": bool(A.procedural_element(e)),
                             "parte_che_afferma": v.get("parte_che_afferma")})
        if not elements:
            continue
        label_in_set = any(A.same_article(label, y) for y in in_set)
        is_alt = any(A.same_article(label, y) for y in alternative) and not label_in_set
        articles.append({"article": str(n["article"]), "token": _ap().article_token(n["article"]),
                         "rubrica": _flat(n.get("rubrica")), "tier": tiers.get(label) or tiers.get(n["article"])
                         or (TIER_ALTERNATIVA if is_alt else "linea" if not label_in_set else "non_verificato"),
                         "in_set": label_in_set, "alternative": is_alt, "valutato": bool(got), "elements": elements})
    if not any(a for a in articles if not a.get("alternative")):      # nothing to strategise on alternatives alone
        return None
    maps = state.get("analysis_maps") if isinstance(state.get("analysis_maps"), dict) else {}
    sents = maps.get("sentences") or {}
    facts = [{"id": str(sid), "text": _flat(s.get("text")), "certainty": str(s.get("certainty") or ""),
              "said_by": _flat(s.get("said_by"))} for sid, s in sents.items() if isinstance(s, dict) and _flat(s.get("text"))]
    facts.sort(key=lambda f: int(f["id"][1:]) if f["id"][1:].isdigit() else 0)
    from . import case_frame, interactive_flow as flow
    held = case_frame.held(state)
    frame = case_frame._frame(held) if held else {}
    client = _flat(verified.get("client") or flow.client_of(state))
    mark_own(facts, frame, client)                      # v79.18.1 (C1): the client's own facts
    quesiti = flow.fit_pretesa_quesiti(state)[1] if held else [str(q) for q in flow.asked_questions(state)]
    side, side_source = side_of(a.get("side"), frame)           # v79.18.2 (item 2): the pretesa's tipo when unverified
    parties = [client] + [_flat(x.get("nome")) for x in (frame.get("parti") or []) if isinstance(x, dict) and _flat(x.get("nome"))]
    return {"pretesa": _flat(verified.get("pretesa_line") or (case_frame.pretesa_text(held) if held else "")),
            "client": client, "side": side, "side_source": side_source,
            "parties": list(dict.fromkeys(x for x in parties if x)),         # v79.18.5 (item 1): the factual test's names
            "charge": _flat(a.get("frame_charge") or binding.get("pretesa_article")),
            "articles": articles, "facts": facts,
            "allegations": allegations(facts, frame, client),
            "quesiti": [str(q) for q in quesiti if _flat(q)],
            "quesiti_articles": {str(k): [str(x) for x in v] for k, v in (binding.get("quesiti") or {}).items()},
            "lines": [{"kind": k, "articolo": a[k].get("articolo"), "eid": a[k].get("eid"), "stato": a[k].get("stato")}
                      for k in A.LINE_KINDS if a.get(k)],
            "honesty": _flat(a.get("honesty")), "rulings": [],
            "penal": bool(A.penal_frame(frame)), "rito": A.frame_rito(frame)}       # v79.18.1 (C3)


def side_of(side: Any, frame: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """v79.18.2 (item 2): (the strategy's side, its source) -- the analysis step's side when the verified frame gave one
    («frame»); else, the frame's ruolo unverified, the side the pretesa's tipo implies (data «side_by_tipo»: accusa /
    opposizione -> difesa, domanda / ricorso / impugnazione -> attore; «pretesa_tipo:<tipo>»); else neutro («»)."""
    got = str(side or SIDE_NEUTRAL)
    if got in (SIDE_CLAIMANT, SIDE_DEFENCE):
        return got, "frame"
    f = frame if isinstance(frame, dict) else {}
    from . import case_frame
    pretesa = f.get(case_frame.PRETESA_FIELD) if isinstance(f.get(case_frame.PRETESA_FIELD), dict) else {}
    tipo = _flat(pretesa.get("tipo")).lower()
    by_tipo = config().get("side_by_tipo") or {}
    if tipo and str(by_tipo.get(tipo) or "") in (SIDE_CLAIMANT, SIDE_DEFENCE):
        return str(by_tipo[tipo]), f"pretesa_tipo:{tipo}"
    return SIDE_NEUTRAL, ""


def mark_own(facts: List[Dict[str, Any]], frame: Dict[str, Any], client: str) -> None:
    """v79.18.1 (C1): each fact's «own» -- stated accertato, or alleged by the client (analysis_step.speaker: CLIENT)."""
    A = _as()
    inp_like = {"frame": frame or {}, "client": client, "client_roles": []}
    for f in facts:
        own = f.get("certainty") == "accertato"
        if not own and f.get("said_by"):
            try:
                own = A.speaker(f["said_by"], inp_like)[0] == A.CLIENT
            except Exception:       # noqa: BLE001
                own = False
        f["own"] = bool(own)


def allegations(facts: Sequence[Dict[str, Any]], frame: Dict[str, Any], client: str) -> List[str]:
    """The S-ids of the facts a party other than the client asserts: the record's allegato with a speaker who is not
    the client (analysis_step.speaker), else -- no speaker recorded -- a sentence reporting a non-client party's
    assertion (asserting_party)."""
    A = _as()
    inp_like = {"frame": frame or {}, "client": client, "client_roles": []}
    out = []
    for f in facts:
        if f.get("certainty") != "allegato":
            continue
        who = f.get("said_by") or ""
        try:
            if who:
                kind, _ = A.speaker(who, inp_like)
                if kind == A.OTHER:
                    out.append(f["id"])
            elif A.asserting_party(f.get("text") or "", inp_like):
                out.append(f["id"])
        except Exception:       # noqa: BLE001 -- an unreadable speaker is no allegation
            continue
    return out


def rulings_for(record: Optional[Dict[str, Any]], articles: Sequence[str]) -> List[Dict[str, Any]]:
    """The registry's rulings on the set's articles, each with its registry label (G<i>, the registry's own numbering,
    so compose's labels_out renders it as the ruling's reference): [{id, estremi, article, passage, element_id}]."""
    A = _as()
    out = []
    for i, r in enumerate((record or {}).get("kept") or [], 1):
        if not isinstance(r, dict) or not any(A.same_article(r.get("article"), x) for x in articles):
            continue
        passage = ""
        if r.get("passage"):
            try:
                from . import rulings as rulings_mod
                passage = rulings_mod.printed(str(r["passage"]))
            except Exception:   # noqa: BLE001
                passage = _flat(r["passage"])
        out.append({"id": f"G{i}", "estremi": _flat(r.get("estremi")), "article": str(r.get("article") or ""),
                    "passage": passage, "element_id": r.get("element_id") or ""})
    return out


# --- the prompt -------------------------------------------------------------------------------------------------------

def _article_block(art: Dict[str, Any], sents: Dict[str, str]) -> List[str]:
    labels = config().get("labels") or {}
    head = f"- A{art['token']}: art. {art['article']}" + (f" ({art['rubrica']})" if art.get("rubrica") else "") \
        + f" [{str(art.get('tier') or '').replace('_', ' ')}]"
    if art.get("alternative"):          # v79.18.2 (item 6): a riqualificazione candidate only
        head += f" — {labels.get('alternativa') or 'solo «riqualificare»'}"
    out = [head, f"  {labels.get('elements') or 'elementi'}:"]
    if not art.get("valutato"):
        out.append(f"  ({labels.get('unassessed') or 'esiti non valutati'})")
    for e in art.get("elements") or []:
        cited = ", ".join(i for i in e.get("cita_fatti") or [] if i in sents) or "—"
        out.append(f"  - [{e['eid']}] {e['testo']} — {e['esito']} — {cited}")
    return out


def build_messages(inp: Dict[str, Any]) -> List[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    labels = config().get("labels") or {}
    none = labels.get("none") or "(nessuno)"
    sents = {f["id"]: f["text"] for f in inp.get("facts") or []}
    parts = [f"{labels.get('pretesa') or 'PRETESA'}: {inp.get('pretesa') or none}",
             f"{labels.get('client') or 'Parte assistita'}: {inp.get('client') or none} ({inp.get('side') or SIDE_NEUTRAL})",
             "", f"{labels.get('set') or 'INSIEME'}:"]
    for art in inp.get("articles") or []:
        parts += _article_block(art, sents)
    parts += ["", f"{labels.get('facts') or 'FATTI'}:"]
    for f in inp.get("facts") or []:
        tag = f.get("certainty") or ""
        if f.get("said_by"):
            tag += f"; {f['said_by']}"
        if f.get("own"):
            tag += f"; {labels.get('own_fact') or 'fatto proprio del cliente'}"       # v79.18.1 (C1)
        parts.append(f"{f['id']} [{tag}] {f['text']}")
    if penal_defence(inp):              # v79.18.1 (C3): the riqualificare requirement said, with the set
        others = [a["article"] for a in alternative_articles(inp)]
        parts += ["", f"{labels.get('penal_defence') or 'DIFESA PENALE'}: " + (", ".join(f"art. {x}" for x in others) or none)]
    parts += ["", f"{labels.get('allegations') or 'ALLEGAZIONI DELLA CONTROPARTE'}: "
              + (", ".join(inp.get("allegations") or []) or none)]
    parts += ["", f"{labels.get('rulings') or 'PRONUNCE'}:"]
    rulings = inp.get("rulings") or []
    parts += [f"- {r['id']}: {r['estremi']} — art. {r['article']} — "
              + (f"«{r['passage']}»" if r.get("passage") else (labels.get("no_passage") or "nessun passaggio"))
              for r in rulings] or [f"- {none}"]
    parts += ["", f"{labels.get('questions') or 'QUESITI'}:"]
    parts += [f"Q{i}: {q}" for i, q in enumerate(inp.get("quesiti") or [], 1)] or [f"- {none}"]
    parts += ["", f"{labels.get('lines') or 'LINEE'}:"]
    parts += [f"- {x.get('kind')}: art. {x.get('articolo')} {x.get('eid') or ''} ({x.get('stato')})"
              for x in inp.get("lines") or [] if x.get("articolo")] or [f"- {none}"]
    return [SystemMessage(content=prompt_system()), HumanMessage(content="\n".join(parts))]


def allowed_text(inp: Dict[str, Any]) -> str:
    rows = []
    labels = config().get("labels") or {}
    for art in inp.get("articles") or []:
        ids = ", ".join(f"{e['eid']} ({e['esito']})" for e in art.get("elements") or [])
        alt = f" — {labels.get('alternativa') or 'solo «riqualificare»'}" if art.get("alternative") else ""
        rows.append(f"- A{art['token']} (art. {art['article']}){alt}: {ids}")
    return "\n".join(rows) or f"- {(config().get('labels') or {}).get('none') or '(nessuno)'}"


def side_statement(inp: Dict[str, Any]) -> str:
    """v79.18.2 (item 2): the re-ask's line naming the client's side and its rules (data «reask_side», «side_rules»)."""
    side = inp.get("side") or SIDE_NEUTRAL
    rules = config().get("side_rules") or {}
    line = str(config().get("reask_side") or "PARTE E LATO: {client} — {side}. {regola}")
    return line.format(client=inp.get("client") or (config().get("labels") or {}).get("none") or "(nessuno)",
                       side=side, regola=str(rules.get(side) or rules.get(SIDE_NEUTRAL) or ""))


def _side_mismatch(why: str) -> bool:
    """v79.18.2 (item 2): the rejection is the side's -- «contestare» on a non-open element or «fondare» on a non-provato
    one (the model argued the other side's case)."""
    return "«contestare» su" in str(why or "") or "«fondare» su" in str(why or "")


def reask_messages(messages: List[Any], reply: str, reasons: Sequence[str], inp: Dict[str, Any],
                   side_mismatch: bool = False) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage
    out = []
    for line in config().get("reask") or []:
        if "{riqualificazione}" in line:        # v79.18.1 (C3): the penal defence's alternatives, when any
            others = [a["article"] for a in alternative_articles(inp)] if penal_defence(inp) else []
            if others:
                out.append(f"{(config().get('labels') or {}).get('penal_defence') or 'DIFESA PENALE'}: "
                           + ", ".join(f"art. {x}" for x in others))
            continue
        if "{lato}" in line:                    # v79.18.2 (item 2): the side stated when every line missed it
            if side_mismatch:
                out.append(side_statement(inp))
            continue
        out.append(line.replace("{reasons}", "\n".join(f"- {r}" for r in reasons)).replace("{allowed}", allowed_text(inp)))
    return list(messages) + [AIMessage(content=str(reply or "")), HumanMessage(content="\n".join(out))]


def penal_defence(inp: Dict[str, Any]) -> bool:
    """v79.18.1 (C3): the client defends a penal pretesa (the frame's rito penal, the side difesa)."""
    return bool(inp.get("penal")) and (inp.get("side") or SIDE_NEUTRAL) == SIDE_DEFENCE


def alternative_articles(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.18.1 (C3): the set's articles other than the pretesa's (the charge) -- a riqualificazione's candidates."""
    charge = inp.get("charge") or ""
    return [a for a in inp.get("articles") or [] if (a.get("in_set", True) or a.get("alternative"))
            and not (charge and _as().same_article(a["article"], charge))]


# --- the call ---------------------------------------------------------------------------------------------------------

def request_kwargs() -> Dict[str, Any]:
    out = dict(_ap().request_kwargs())
    out["max_tokens"] = int(config().get("max_tokens") or 2500)
    return out


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model bound to the kwargs; the reply without its reasoning, on the request's stage meter --
    as analysis_step._default_call."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = ai_chat.chat_model.bind(**kwargs).invoke(messages)
    ai_chat._stage_call("strategy_step", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def call_once(messages: List[Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]],
              budget: float) -> Dict[str, Any]:
    """One call on a worker thread, waited at most budget seconds (as analysis_step.call_once); never raises."""
    started = time.time()
    if budget <= 0:
        return {"status": TIMEOUT, "reply": "", "data": None, "seconds": 0.0, "error": "nessun tempo residuo"}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="strategy_step")
    try:
        fut = pool.submit(contextvars.copy_context().run, call or _default_call, messages, request_kwargs())
        done, _ = _wait([fut], timeout=budget)
        if fut not in done:
            return {"status": TIMEOUT, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"oltre {round(budget, 1)} s"}
        try:
            reply = fut.result()
        except Exception as exc:    # noqa: BLE001
            return {"status": FAILED, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    try:
        data = _as().parse_reply(reply)
    except Exception as exc:        # noqa: BLE001
        return {"status": MALFORMED, "reply": str(reply or "")[:2000], "data": None,
                "seconds": round(time.time() - started, 3), "error": str(exc)[:300]}
    return {"status": OK, "reply": str(reply or ""), "data": data, "seconds": round(time.time() - started, 3),
            "error": ""}


# --- verification -----------------------------------------------------------------------------------------------------

def _ids(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    return [str(x).strip() for x in value or [] if str(x or "").strip()] if isinstance(value, list) else []


def article_of(raw: Any, inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The input's article a reply names («A1385cc», «1385cc», «art. 1385 c.c.», the label), else None."""
    labels = [a["article"] for a in inp.get("articles") or []]
    label = _as().resolve(raw, labels)
    return next((a for a in inp.get("articles") or [] if a["article"] == label), None) if label else None


def element_of(art: Dict[str, Any], eid: Any) -> Optional[Dict[str, Any]]:
    """The article's element an id names («A1385cc:E4»; «E4» alone on this article), else None."""
    A = _as()
    parsed = A.parse_eid(_flat(eid))
    if parsed is None:
        return None
    key, n = parsed
    if key and key != A.art_key(art["article"]):
        return None
    for e in art.get("elements") or []:
        got = A.parse_eid(e["eid"])
        if got is not None and got[1] == n:
            return e
    return None


def _sids(value: Any, inp: Dict[str, Any], counters: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    known = {f["id"] for f in inp.get("facts") or []}
    ids = [i for i in _ids(value) if _S_RE.match(i)]
    good = [i for i in dict.fromkeys(ids) if i in known]
    bad = [i for i in ids if i not in known]
    counters["strategy_bad_id"] += len(bad) + sum(1 for i in _ids(value) if not _S_RE.match(i))
    return good, bad


def _gids(value: Any, inp: Dict[str, Any], counters: Dict[str, Any]) -> List[str]:
    known = {r["id"] for r in inp.get("rulings") or []}
    ids = _ids(value)
    good = [i for i in dict.fromkeys(ids) if i in known]
    counters["strategy_bad_id"] += len([i for i in ids if i not in known])
    return good


def _eids(value: Any, inp: Dict[str, Any], counters: Dict[str, Any]) -> List[Tuple[str, str]]:
    """[(article, eid)] of the element ids a list names, the unknown ones counted."""
    out = []
    for raw in _ids(value):
        hit = None
        parsed = _as().parse_eid(raw)
        for art in inp.get("articles") or [] if parsed and parsed[0] else []:     # the article's token required
            e = element_of(art, raw)
            if e is not None:
                hit = (art["article"], e["eid"])
                break
        if hit is None:
            counters["strategy_bad_id"] += 1
        elif hit not in out:
            out.append(hit)
    return out


def _no_answer(value: Any) -> bool:
    words = [str(w).lower() for w in config().get("no_answer_words") or []]
    return isinstance(value, str) and any(w in value.lower() for w in words)


def _esito_fits(esito: str, side: str) -> bool:
    """v79.18.1 (C4): an element's esito supports the client's answer -- the defence stands on a contestato, da_provare
    or escluso element, a claimant on a provato one."""
    return esito in (CONTESTATO, DA_PROVARE, ESCLUSO) if side == SIDE_DEFENCE else esito == PROVATO


def _verify_objection(raw: Any, inp: Dict[str, Any], counters: Dict[str, Any], n: int = 0
                      ) -> Tuple[Optional[Dict[str, Any]], str]:
    """(the verified obiezione, the mismatch reason or ""): v79.18.1 (C4) the risposta declares the elements and
    rulings it rests on («risposta_cita_elementi», verified by id); an element whose esito does not fit the side is a
    mismatch -- the caller re-asks once, then the answer prints as unanswered."""
    if not isinstance(raw, dict) or not _flat(raw.get("testo")):
        return None, ""
    answer = raw.get("risposta_cita_fatti")
    sids, _bad = _sids(answer, inp, counters) if not _no_answer(answer) else ([], [])
    gids = _gids(raw.get("cita_pronunce"), inp, counters)
    # v79.18.5 (item 5): the accusa_forte -- the fact pattern the counterparty relies on, by S-ids (inline or listed)
    forte = raw.get("accusa_forte") if isinstance(raw.get("accusa_forte"), dict) else {}
    forte_text = _flat(forte.get("testo"))
    forte_sids, _b2 = _sids(list(_ids(forte.get("cita_fatti"))) + [x.strip("[]") for x in _FACT_ID_RE.findall(forte_text)],
                            inp, counters)
    own_sids, _b3 = _sids([x.strip("[]") for x in _FACT_ID_RE.findall(_flat(raw.get("testo")))], inp, counters)
    rests = raw.get("risposta_cita_elementi")
    eids = _eids([x for x in _ids(rests) if not _G_RE.match(x)], inp, counters)
    rest_g = _gids([x for x in _ids(rests) if _G_RE.match(x)], inp, counters)
    side = inp.get("side") or SIDE_NEUTRAL
    bad = []
    for art, eid in eids:
        a = next((x for x in inp.get("articles") or [] if x["article"] == art), {})
        e = next((x for x in a.get("elements") or [] if x["eid"] == eid), {})
        if not _esito_fits(e.get("esito") or DA_PROVARE, side):
            bad.append(f"{eid} ({e.get('esito')})")
    why = (f"linea {n}: la risposta all'obiezione si regge su {', '.join(bad)}: esito che non la sostiene per la parte "
           f"{side}") if bad else ""
    return {"testo": _flat(raw.get("testo")), "risposta_cita_fatti": sids, "cita_pronunce": gids,
            "risposta_cita_elementi": [eid for _a, eid in eids] + rest_g, "mismatch": bool(bad),
            "accusa_forte": {"testo": forte_text, "cita_fatti": forte_sids} if forte_text and forte_sids else None,
            "grounded": bool(own_sids or forte_sids),         # v79.18.5 (item 5): an objection with no S-id is dropped
            "senza_risposta": not sids or bool(bad)}, why     # counted in finish, on the final linee only


def _verify_line(raw: Any, n: int, inp: Dict[str, Any], counters: Dict[str, Any], seen: set
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(raw, dict):
        return None, f"linea {n}: non è un oggetto"
    art = article_of(raw.get("articolo"), inp)
    if art is None:
        return None, (f"linea {n}: articolo non nell'insieme: {_flat(raw.get('articolo'))[:60] or '(vuoto)'}; ammessi "
                      + ", ".join(f"A{a['token']}" for a in inp.get("articles") or []))
    e = element_of(art, raw.get("eid"))
    if e is None:
        return None, (f"linea {n}: eid «{_flat(raw.get('eid'))[:40] or '(vuoto)'}» non è un elemento di art. "
                      f"{art['article']} ({', '.join(x['eid'] for x in art['elements'])})")
    tipo = _flat(raw.get("tipo")).lower()
    if tipo not in (config().get("tipi") or []):
        return None, f"linea {n}: tipo «{tipo or '(vuoto)'}» non ammesso: {' | '.join(config().get('tipi') or [])}"
    if art.get("alternative") and tipo != "riqualificare":      # v79.18.2 (item 6)
        return None, (f"linea {n}: art. {art['article']} è un'alternativa (fit «no» nello stesso titolo/capo della "
                      f"pretesa): ammette solo «riqualificare»")
    if tipo == "contestare" and e["esito"] not in OPEN:
        return None, (f"linea {n}: «contestare» su {e['eid']} con esito {e['esito']}: ammesso solo su un elemento "
                      f"contestato o da_provare")
    if tipo == "fondare" and e["esito"] != PROVATO:
        return None, f"linea {n}: «fondare» su {e['eid']} con esito {e['esito']}: ammesso solo su un elemento provato"
    if (art["article"], e["eid"]) in seen:
        return None, f"linea {n}: ripete la linea su {e['eid']}"
    rapporto = _flat(raw.get("rapporto")).lower()
    if rapporto not in (config().get("rapporti") or []):
        rapporto = "alternativa"
    sids, _bad = _sids(raw.get("cita_fatti"), inp, counters)
    gids = _gids(raw.get("cita_pronunce"), inp, counters)
    condizione = ""                      # v79.18.1 (C2): a G-id attached to the eid -- the jurisprudential condition
    for g in _gids(raw.get("condizione"), inp, counters)[:1]:
        ruling = next((r for r in inp.get("rulings") or [] if r["id"] == g), {})
        if _as().same_article(ruling.get("article"), art["article"]):
            condizione = g
        else:
            counters["strategy_bad_id"] += 1
    obiezione, why_ob = _verify_objection(raw.get("obiezione"), inp, counters, n)
    return {"articolo": art["article"], "token": art["token"], "eid": e["eid"], "elemento": e["testo"],
            "esito": e["esito"], "tipo": tipo, "rapporto": rapporto, "rapporto_modello": rapporto,
            "perche_prima": _flat(raw.get("perche_prima")), "cita_fatti": sids, "cita_pronunce": gids,
            "condizione": condizione, "obiezione": obiezione, "tier": art.get("tier"),
            "alternative": bool(art.get("alternative")), "reasons": [why_ob] if why_ob else []}, ""


def verify(reply: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """One reply checked (pure): {"linee": the verified ones (the model's order), "rejected": [{n, why}], "reasons":
    [the re-ask's], "raw": the reply, "counters"}."""
    counters = empty_counters()
    reply = reply if isinstance(reply, dict) else {}
    raw_lines = reply.get("linee") if isinstance(reply.get("linee"), list) else []
    linee, rejected, reasons = [], [], []
    seen: set = set()
    for n, raw in enumerate(raw_lines, 1):
        line, why = _verify_line(raw, n, inp, counters, seen)
        if line is None:
            counters["strategy_line_rejected"] += 1
            rejected.append({"n": n, "articolo": _flat((raw or {}).get("articolo")) if isinstance(raw, dict) else "",
                             "why": why})
            reasons.append(why)
            continue
        seen.add((line["articolo"], line["eid"]))
        reasons += line.pop("reasons", [])              # v79.18.1 (C4): an objection's mismatch, re-asked once
        linee.append(line)
    if not raw_lines:
        reasons.append("«linee» mancante o vuota")
    # v79.18.2 (item 2): every proposed line rejected for the side's rules -- the re-ask states the side explicitly
    side_mismatch = bool(raw_lines) and not linee and all(_side_mismatch(r["why"]) for r in rejected)
    lo = int(config().get("lines_min") or 2)
    if len(linee) < lo:
        reasons.append(f"servono almeno {lo} linee verificate (ne restano {len(linee)})")
    if penal_defence(inp) and alternative_articles(inp) and not riqualificazione_of(linee, inp):    # v79.18.1 (C3)
        reasons.append("difesa penale: manca una linea «riqualificare» su un altro articolo dell'insieme ("
                       + ", ".join(f"A{a['token']}" for a in alternative_articles(inp)) + ")")
    for item in reply.get("passi") if isinstance(reply.get("passi"), list) else []:     # v79.18.1 (C5)
        if isinstance(item, dict):
            why = passo_issue(item, inp)
            if why:
                reasons.append(why)
    return {"linee": linee[:int(config().get("lines_max") or 3)], "rejected": rejected, "reasons": reasons,
            "raw": reply, "counters": counters, "side_mismatch": side_mismatch}


def riqualificazione_of(linee: Sequence[Dict[str, Any]], inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """v79.18.1 (C3): the first «riqualificare» linea on an article other than the pretesa's, None without one."""
    charge = inp.get("charge") or ""
    return next((x for x in linee if x.get("tipo") == "riqualificare"
                 and not (charge and _as().same_article(x["articolo"], charge))), None)


def _element_by_eid(eid: Any, inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for art in inp.get("articles") or []:
        e = element_of(art, eid)
        if e is not None and (_as().parse_eid(_flat(eid)) or ("", 0))[0]:
            return e
    return None


def passo_issue(item: Dict[str, Any], inp: Dict[str, Any]) -> str:
    """v79.18.1 (C5): why a passo is no concrete fact to acquire -- a restatement of its element (restatement_overlap of
    the element's words in the passo) or no acquisition word (passi_words); "" when it is one, or its eid is unknown."""
    e = _element_by_eid(item.get("eid"), inp)
    cosa = _flat(item.get("cosa"))
    if e is None or not cosa:
        return ""
    el_words = {w for w in re.findall(r"[a-zà-ü]{4,}", _as().fold(e.get("testo")))}
    mine = {w for w in re.findall(r"[a-zà-ü]{4,}", _as().fold(cosa))}
    share = len(el_words & mine) / len(el_words) if el_words else 0.0
    if share >= float(config().get("restatement_overlap") or 0.6):
        return f"passo {item.get('eid')}: riformula l'elemento invece di indicare la prova da acquisire"
    pats = [str(p) for p in config().get("passi_words") or [] if str(p or "").strip()]
    has_means = bool(pats) and bool(re.search("|".join(f"(?:{p})" for p in pats), _as().fold(cosa)))
    if pats and not has_means:
        return f"passo {item.get('eid')}: non nomina una prova concreta da acquisire (referto, perizia, testimoni, documenti…)"
    concl = [str(p) for p in config().get("passi_conclusion_words") or [] if str(p or "").strip()]     # v79.18.5 (item 4)
    if concl and re.search("|".join(f"(?:{p})" for p in concl), _as().fold(cosa)):
        return f"passo {item.get('eid')}: chiede una conclusione giuridica, non un fatto da acquisire"
    return ""


# --- forza and the ranking --------------------------------------------------------------------------------------------

def defence_rank(e: Dict[str, Any], inp: Dict[str, Any], linea: Optional[Dict[str, Any]] = None) -> int:
    """v79.18.1 (C1): the defence's order -- (a) the element contestato by the client's own facts (a cited fact «own»:
    accertato or the client's), (b) da_provare with a ruling the linea cites (cita_pronunce or its condizione) setting
    a condition the facts do not show, (c) a bare da_provare; a contestato on another party's words alone between (b)
    and (c); provato or escluso last. Data «defence_rank»."""
    cfg = config().get("defence_rank") or {}
    own = {f["id"] for f in inp.get("facts") or [] if f.get("own")}
    cites_ruling = bool((linea or {}).get("cita_pronunce") or (linea or {}).get("condizione"))
    esito = e.get("esito")
    cited = set(e.get("cita_fatti") or []) | set((linea or {}).get("cita_fatti") or [])
    if esito == CONTESTATO and cited & own:
        return int(cfg.get("contestato_fatti_propri", 4))
    if esito == DA_PROVARE and cites_ruling:
        return int(cfg.get("da_provare_con_pronuncia", 3))
    if esito == CONTESTATO:
        return int(cfg.get("contestato", 2))
    if esito == DA_PROVARE:
        return int(cfg.get("da_provare", 1))
    return int(cfg.get("altro", 0))


def forza(label: str, eid: str, inp: Dict[str, Any], linea: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The line's strength, computed from the article's verified table by side: {provato, contestato, da_provare,
    escluso, totale, key, testo, valutato}. The defence's key (v79.18.1, C1): defence_rank of the eid, then the open
    count (contestato + da_provare) on the article. A claimant's (item 10): the eid's esito rank, then minus the
    further conditions in the same comma, then the provato count. Ties compare equal."""
    art = next((a for a in inp.get("articles") or [] if _as().same_article(a["article"], label)), None) or {}
    els = art.get("elements") or []
    n = {k: sum(1 for e in els if e.get("esito") == k) for k in ESITO_RANK}
    e = next((x for x in els if x.get("eid") == eid), None) or {}
    if (inp.get("side") or SIDE_NEUTRAL) == SIDE_DEFENCE:
        key: Tuple[Any, ...] = (defence_rank(e, inp, linea), n[CONTESTATO] + n[DA_PROVARE])
    else:
        further = sum(1 for x in els if x.get("eid") != eid and x.get("comma") is not None
                      and x.get("comma") == e.get("comma")) if e.get("comma") is not None else 0
        key = (ESITO_RANK.get(e.get("esito"), 1), -further, n[PROVATO])
    cfg = rendering()
    testo = str(cfg.get("forza") or "provato {provato} / contestato {contestato} / da provare {da_provare}").format(**n)
    if n[ESCLUSO]:
        testo += str(cfg.get("forza_escluso") or " / escluso {escluso}").format(**n)
    own = {f["id"] for f in inp.get("facts") or [] if f.get("own")}          # v79.18.3 (item 1): what the risk reads
    cited = set(e.get("cita_fatti") or []) | set((linea or {}).get("cita_fatti") or [])
    return {**n, "totale": len(els), "key": list(key), "testo": testo, "valutato": bool(art.get("valutato")),
            "esito": e.get("esito") or "", "fatti_propri": bool(e.get("esito") == CONTESTATO and cited & own),
            "pronuncia": bool((linea or {}).get("cita_pronunce") or (linea or {}).get("condizione"))}


def rank(linee: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """(the linee ranked, whether the code moved any): a linea stands above one with strictly greater forza only when
    it cites a G-id; else the stronger one moves up past it; ties keep the model's order."""
    out: List[Dict[str, Any]] = []
    for line in linee:
        out.append(line)
        i = len(out) - 1
        while i > 0 and tuple(out[i]["forza"]["key"]) > tuple(out[i - 1]["forza"]["key"]) \
                and not out[i - 1].get("cita_pronunce"):
            out[i], out[i - 1] = out[i - 1], out[i]
            i -= 1
    return out, [x["eid"] for x in out] != [x["eid"] for x in linee]


def risk_of(a: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """v79.18.3 (item 1): §9's word for the client from the principale, by side (data «risk»): {word, capped, why}.
    Claimant (attore, neutro): favorevole = the principale's element provato AND (provati_min or more provato elements on
    its article OR a G-cite on the line); sfavorevole = the principale's element da_provare / escluso; else incerto (provato
    without that support, or contestato). Defence: favorevole = the principale's element contestato by the client's own
    facts AND a G-cite; sfavorevole = the article's elements all provato and none contestato; else incerto (contestato or
    da_provare with a G-cite but no own fact, an own fact without a G-cite, a bare da_provare). Any side: a set built only
    from unverified tiers (a recall-empty set), or a one-element article, caps the word at incerto (capped, the counter
    strategy_risk_capped). {} without a verified strategy."""
    if not isinstance(a, dict) or not a.get("linee"):
        return {}
    cfg_all = config().get("risk") or {}
    side = a.get("side") or SIDE_NEUTRAL
    cfg = cfg_all.get(side) or cfg_all.get(SIDE_NEUTRAL) or {}
    cap = cfg_all.get("cap") or {}
    f = a["linee"][0].get("forza") or {}
    tot = int(f.get("totale") or 0)
    if not tot:
        return {}
    esito = f.get("esito") or a["linee"][0].get("esito") or ""
    fav, sfav = cfg.get("favorevole") or {}, cfg.get("sfavorevole") or {}
    if side == SIDE_DEFENCE:
        if sfav.get("tutti_provati", True) and f.get(PROVATO, 0) == tot and not f.get(CONTESTATO, 0):
            word, why = SFAVOREVOLE, "ogni elemento dell'accusa provato, nessuno contestato"
        elif esito in (fav.get("esito") or [CONTESTATO]) and (not fav.get("fatti_propri", True) or f.get("fatti_propri")) \
                and (not fav.get("pronuncia", True) or f.get("pronuncia")):
            word, why = FAVOREVOLE, "elemento contestato dai fatti propri del cliente, con pronuncia"
        else:
            word, why = INCERTO, f"elemento {esito}" + (", senza pronuncia" if not f.get("pronuncia") else "") \
                + (", senza fatti propri" if esito == CONTESTATO and not f.get("fatti_propri") else "")
    else:
        if esito in (sfav.get("esito") or [DA_PROVARE, ESCLUSO]):
            word, why = SFAVOREVOLE, f"elemento della linea principale {esito}"
        elif esito in (fav.get("esito") or [PROVATO]) and (f.get(PROVATO, 0) >= int(fav.get("provati_min", 2))
                                                           or (fav.get("oppure_pronuncia", True) and f.get("pronuncia"))):
            word, why = FAVOREVOLE, ("elemento provato, " + (f"{f.get(PROVATO)} elementi provati" if f.get(PROVATO, 0)
                                                               >= int(fav.get("provati_min", 2)) else "con pronuncia"))
        else:
            word, why = INCERTO, f"elemento {esito}" + (" senza altri elementi provati né pronuncia" if esito == PROVATO else "")
    capped = ""
    if word == FAVOREVOLE:
        if cap.get("tier_non_verificato", True) and a.get("set_non_verificato"):
            capped = "insieme da soli livelli non verificati"
        elif tot < int(cap.get("elementi_min", 2)):
            capped = "articolo con un solo elemento"
    if capped:
        word = INCERTO
    return {"word": word, "capped": bool(capped), "why": why + (f"; tetto incerto: {capped}" if capped else "")}


def risk_word(result: Optional[Dict[str, Any]]) -> str:
    """§9's word for the client (risk_of, stored by finish as analysis["risk"]); "" without a verified strategy."""
    a = (result or {}).get("analysis") if isinstance(result, dict) else None
    if not a or not a.get("linee"):
        return ""
    got = a.get("risk") if isinstance(a.get("risk"), dict) else risk_of(a)
    return str((got or {}).get("word") or "")


# --- the result -------------------------------------------------------------------------------------------------------

def _passi(raw: Any, principale: Dict[str, Any], inp: Dict[str, Any], counters: Dict[str, Any]
           ) -> List[Dict[str, Any]]:
    """The principale's open elements, each with the model's passo when it gave one (its eid verified, its S-id
    verified), else a code label (strategy_passi_added)."""
    art = next((a for a in inp.get("articles") or [] if a["article"] == principale["articolo"]), None) or {}
    given: Dict[str, Dict[str, Any]] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        e = element_of(art, item.get("eid")) if art else None
        if e is None or e["esito"] not in OPEN:
            counters["strategy_bad_id"] += 1
            continue
        if passo_issue(item, inp):          # v79.18.1 (C5): a restatement or no concrete proof -- the code's label
            counters["strategy_passi_not_concrete"] += 1
            continue
        sids, _bad = _sids(item.get("cita_fatti"), inp, counters)
        given.setdefault(e["eid"], {"eid": e["eid"], "elemento": e["testo"], "esito": e["esito"],
                                    "cosa": _flat(item.get("cosa")), "cita_fatti": sids[:1], "codice": False})
    out = []
    for e in art.get("elements") or []:
        if e["esito"] not in OPEN:
            continue
        if e["eid"] in given and given[e["eid"]]["cosa"]:
            out.append(given[e["eid"]])
        else:
            counters["strategy_passi_added"] += 1
            out.append({"eid": e["eid"], "elemento": e["testo"], "esito": e["esito"], "cosa": "", "cita_fatti": [],
                        "codice": True})
    return out


def _cited_entries(raw: Any, inp: Dict[str, Any], counters: Dict[str, Any], text_key: str = "testo"
                   ) -> List[Dict[str, Any]]:
    """Item 12's id-cited entries (passi_immediati, verifiche_documentali): kept with at least one valid id (an
    element's or a fact's), else dropped (strategy_entries_uncited)."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not _flat(item.get(text_key)):
            continue
        sids, _bad = _sids(item.get("cita_fatti"), inp, counters)
        eids = _eids(item.get("cita_elementi"), inp, counters)
        if not sids and not eids:
            counters["strategy_entries_uncited"] += 1
            continue
        out.append({"testo": _flat(item.get(text_key)), "cita_fatti": sids, "cita_elementi": [x[1] for x in eids]})
    return out


def _contro_difesa(raw: Any, linee: Sequence[Dict[str, Any]], inp: Dict[str, Any], counters: Dict[str, Any]
                   ) -> List[Dict[str, Any]]:
    """One entry per counterparty allegation (inp «allegations»), the model's when it gave one on that S-id (its
    cita_linea an eid of a verified linea, its S-ids verified; «senza risposta» said so), else the code's «senza
    risposta» (strategy_contro_difesa_added)."""
    by_eid = {x["eid"]: x for x in linee}
    given: Dict[str, Dict[str, Any]] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        sid = _flat(item.get("allegazione"))
        if sid not in (inp.get("allegations") or []):
            counters["strategy_bad_id"] += 1
            continue
        sids, _bad = _sids(item.get("cita_fatti"), inp, counters)
        linea = _flat(item.get("cita_linea"))
        if linea and linea not in by_eid:
            counters["strategy_bad_id"] += 1
            linea = ""
        risposta = _flat(item.get("risposta"))
        none = not risposta or _no_answer(risposta) or (not sids and not linea)
        given.setdefault(sid, {"allegazione": sid, "risposta": "" if none else risposta, "cita_linea": linea or None,
                               "cita_fatti": sids, "senza_risposta": none, "codice": False})
    out = []
    for sid in inp.get("allegations") or []:
        if sid in given:
            out.append(given[sid])
        else:
            counters["strategy_contro_difesa_added"] += 1
            out.append({"allegazione": sid, "risposta": "", "cita_linea": None, "cita_fatti": [], "senza_risposta": True,
                        "codice": True})
    return out


def material(inp: Dict[str, Any], vm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v79.18.5 (item 1): the reviewer's material for the strategy's free text -- compose's verified material when
    given (its maps: the periods and the sentences; the fitted set), else built from the input (the facts as the
    sentences, the input's articles as the fitted set, no periods)."""
    if isinstance(vm, dict) and isinstance(vm.get("maps"), dict) and vm["maps"]:
        return vm
    return {"maps": {"periods": {}, "sentences": {f["id"]: {"text": f.get("text")} for f in inp.get("facts") or []},
                     "norms": []},
            "fitted": [a["article"] for a in inp.get("articles") or []], "named": []}


_NAME_STOP = frozenset({"srl", "spa", "snc", "sas", "societa", "società", "ditta", "studio", "comune", "agenzia", "delle",
                        "della", "dello", "degli", "entrate", "procura", "repubblica", "tribunale"})


def factual(sentence: str, inp: Dict[str, Any]) -> bool:
    """v79.18.5 (item 1): the sentence asserts a fact of the case -- it names a party (inp «parties»: the frame's parti
    and the client, by a whole word of three letters or more that is no company form) or holds a figure (a date, an
    amount, a count)."""
    if re.search(r"\[A[0-9a-z]+\.\d+\]", sentence or ""):        # a law statement with its period id is no fact claim
        return False
    bare = re.sub(r"\[[^\]]*\]", " ", sentence or "")                 # the ids out, then the article mentions
    bare = re.sub(r"\bartt?\.\s*\d+(?:-[a-z]+)?(?:\s+(?:c\.[a-z]\.[a-z]?\.?|d\.[a-z]+\.?|l\.|cost\.)\S*)?", " ", bare, flags=re.IGNORECASE)
    if re.search(r"\d", bare):
        return True
    low = f" {_as().fold(sentence)} "
    for name in inp.get("parties") or []:
        for w in re.findall(r"[a-zà-ü]{3,}", _as().fold(name)):
            if w not in _NAME_STOP and f" {w} " in low:
                return True
    return False


def review_free_text(text: Any, inp: Dict[str, Any], vm: Optional[Dict[str, Any]], counters: Dict[str, Any],
                     listed: Sequence[str] = (), events: Optional[List[Dict[str, Any]]] = None) -> str:
    """v79.18.5 (item 1): a free-text field of the reply, sentence by sentence through the Q&A's reviewer
    (verified_material.review_sentence: an unknown id, an article outside the fitted set, a rule attributed to an
    article with no A-id of it -- dropped; item 2: a rule sentence with its A-id printed as the article's verbatim
    passage); a sentence asserting a fact of the case (factual) with no S-id -- inline, or the field's own cita_fatti
    (listed) when the field is one sentence -- is dropped (strategy_sentences_dropped_no_fact). The kept sentences, their
    ids rendered as text (strategy_ids_rendered)."""
    from . import verified_material as V
    mat = material(inp, vm)
    sentences = [x for x in V.sentences_of(" ".join(str(text or "").split())) if x.strip()]
    kept: List[str] = []
    for sentence in sentences:
        grounded = bool(_FACT_ID_RE.search(sentence)) or (len(sentences) == 1 and bool(listed))
        if factual(sentence, inp) and not grounded:
            counters["strategy_sentences_dropped_no_fact"] += 1
            if events is not None:
                events.append({"sede": SITE_STRATEGY, "esito": _as().DROPPED, "testo": sentence[:200], "articoli": [],
                               "motivo": "fatto_senza_id"})
            continue
        n_ids = len(_ANY_ID_RE.findall(sentence))
        done = V.review_sentence(sentence, mat, SITE_STRATEGY, events if events is not None else [])
        if done.strip():
            counters["strategy_ids_rendered"] += n_ids
            kept.append(done.strip())
        else:
            counters["strategy_sentences_dropped"] += 1
    return " ".join(kept)


def review_fields(linee: List[Dict[str, Any]], raw: Dict[str, Any], inp: Dict[str, Any], vm: Optional[Dict[str, Any]],
                  counters: Dict[str, Any], events: Optional[List[Dict[str, Any]]] = None) -> None:
    """v79.18.5 (items 1, 5): the linee's perche_prima and obiezione texts reviewed in place; an objection whose text
    is dropped or that no S-id grounds (its own inline ids or a verified accusa_forte) is dropped
    (strategy_objection_dropped); an objection with a verified accusa_forte counted (strategy_objection_strongest)."""
    for x in linee:
        x["perche_prima"] = review_free_text(x.get("perche_prima"), inp, vm, counters, x.get("cita_fatti") or [], events)
        o = x.get("obiezione")
        if not o:
            continue
        forte = o.get("accusa_forte") or {}
        if forte.get("testo"):
            forte["testo"] = review_free_text(forte["testo"], inp, vm, counters, forte.get("cita_fatti") or [], events)
        o["testo"] = review_free_text(o.get("testo"), inp, vm, counters, (forte or {}).get("cita_fatti") or [], events)
        if not o["testo"] or not o.get("grounded"):
            counters["strategy_objection_dropped"] += 1
            x["obiezione"] = None
            continue
        if forte.get("testo") and forte.get("cita_fatti"):
            counters["strategy_objection_strongest"] += 1
        else:
            o["accusa_forte"] = None


def finish(checked: Dict[str, Any], inp: Dict[str, Any], vm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The verified strategy from a checked reply: the linee with their forza, ranked (rank; the first the principale,
    another «principale» made «subordinata»), the passi, item 12's entries, the ranking orders and the side."""
    counters = checked["counters"]
    linee = [dict(x, forza=forza(x["articolo"], x["eid"], inp, x)) for x in checked["linee"]]
    review_events: List[Dict[str, Any]] = []
    review_fields(linee, checked.get("raw") or {}, inp, vm, counters, review_events)     # v79.18.5 (items 1, 5)
    for x in linee:                     # v79.18.1 (C4): a mismatch left after the re-ask -- the answer prints as unanswered
        if x.get("obiezione") and x["obiezione"].get("mismatch"):
            counters["strategy_objection_mismatch"] += 1
            x["obiezione"]["senza_risposta"] = True
    model_order = [f"{x['articolo']} {x['eid']}" for x in linee]
    ranked, reordered = rank(linee)
    if ranked and ranked[0].get("alternative") and any(not x.get("alternative") for x in ranked):
        first = next(i for i, x in enumerate(ranked) if not x.get("alternative"))   # v79.18.2 (item 6): never the
        ranked.insert(0, ranked.pop(first))                                          # principale, whatever its forza
        counters["strategy_alternativa_demoted"] += 1
        reordered = True
    counters["strategy_rank_reordered"] = int(reordered)
    counters["strategy_lines"] = len(ranked)
    for i, x in enumerate(ranked):
        if i == 0:
            x["rapporto"] = PRINCIPALE
        elif x["rapporto"] == PRINCIPALE:
            x["rapporto"] = SUBORDINATA
    cited = {g for x in ranked for g in x.get("cita_pronunce") or []}
    cited |= {g for x in ranked for g in ((x.get("obiezione") or {}).get("cita_pronunce") or [])}
    counters["strategy_rulings_cited"] = len(cited)
    counters["strategy_objections_unanswered"] = sum(1 for x in ranked if x.get("obiezione")
                                                     and x["obiezione"].get("senza_risposta"))
    raw = checked.get("raw") or {}
    principale = ranked[0] if ranked else None
    riq = None                          # v79.18.1 (C3): the penal defence's qualificazione alternativa
    if penal_defence(inp):
        hit = riqualificazione_of(ranked, inp)
        riq = {"linea": hit["eid"] if hit else None, "articolo": hit["articolo"] if hit else None,
               "n": ranked.index(hit) + 1 if hit else None,
               "alternative": [a["article"] for a in alternative_articles(inp)]}
        if hit is None:
            counters["strategy_riqualificazione_missing"] = 1
    set_arts = [x for x in inp.get("articles") or [] if not x.get("alternative")]       # v79.18.3 (item 1): the cap
    set_non_verificato = bool(set_arts) and all(str(x.get("tier") or "") in (TIER_UNVERIFIED_SET, "linea") for x in set_arts)
    risk = risk_of({"linee": ranked, "side": inp.get("side") or SIDE_NEUTRAL, "set_non_verificato": set_non_verificato})
    counters["strategy_risk_capped"] = int(bool(risk.get("capped")))
    passi = _passi(raw.get("passi"), principale, inp, counters) if principale else []
    for p in passi:                     # v79.18.5 (item 1): the passo's sentence reviewed; emptied -> the code's label
        if p.get("cosa"):
            p["cosa"] = review_free_text(p["cosa"], inp, vm, counters, p.get("cita_fatti") or [], review_events)
            if not p["cosa"]:
                p["codice"] = True
    immediati = _cited_entries(raw.get("passi_immediati"), inp, counters)
    verifiche = _cited_entries(raw.get("verifiche_documentali"), inp, counters)
    for group in (immediati, verifiche):
        for p in group:
            p["testo"] = review_free_text(p["testo"], inp, vm, counters, p.get("cita_fatti") or [], review_events)
    immediati = [p for p in immediati if p.get("testo")]
    verifiche = [p for p in verifiche if p.get("testo")]
    contro = _contro_difesa(raw.get("contro_difesa"), ranked, inp, counters)
    for c in contro:
        if c.get("risposta"):
            c["risposta"] = review_free_text(c["risposta"], inp, vm, counters, c.get("cita_fatti") or [], review_events)
            if not c["risposta"] and not c.get("cita_fatti") and not c.get("cita_linea"):
                c["senza_risposta"] = True
    return {"linee": ranked, "principale": principale, "side": inp.get("side") or SIDE_NEUTRAL,
            "set_non_verificato": set_non_verificato, "risk": risk,                                  # v79.18.3 (item 1)
            "review_events": review_events,                                                          # v79.18.5 (item 1)
            "rulings_estremi": {r["id"]: r.get("estremi") or r["id"] for r in inp.get("rulings") or []},   # (item 3)
            "elementi_testi": {e["eid"]: e.get("testo") or "" for art in inp.get("articles") or []
                               for e in art.get("elements") or []},
            "side_source": inp.get("side_source") or "",                                            # v79.18.2 (item 2)
            "alternative": [a["article"] for a in inp.get("articles") or [] if a.get("alternative")],   # v79.18.2 (item 6)
            "riqualificazione": riq,
            "speakers": {f["id"]: f["said_by"] for f in inp.get("facts") or [] if f.get("said_by")},   # v79.18.1 (C7)
            "reordered": reordered, "model_order": model_order,
            "final_order": [f"{x['articolo']} {x['eid']}" for x in ranked],
            "passi": passi, "passi_immediati": immediati, "contro_difesa": contro, "verifiche_documentali": verifiche,
            "rejected": checked.get("rejected") or [], "rulings_cited": sorted(cited, key=lambda g: int(g[1:]))}


def merge(first: Dict[str, Any], again: Dict[str, Any]) -> Dict[str, Any]:
    """The re-ask's verification onto the first's: the second reply's linee, then the first's not repeated, up to
    lines_max; the second reply's other fields (passi, item 12) when it has them, else the first's."""
    lo, hi = int(config().get("lines_min") or 2), int(config().get("lines_max") or 3)
    linee = list(again["linee"])
    if len(linee) < lo:
        seen = {(x["articolo"], x["eid"]) for x in linee}
        linee += [x for x in first["linee"] if (x["articolo"], x["eid"]) not in seen]
    raw = dict(first.get("raw") or {})
    raw.update({k: v for k, v in (again.get("raw") or {}).items() if v not in (None, [], {}, "")})
    return {**again, "linee": linee[:hi], "raw": raw, "rejected": first["rejected"] + again["rejected"]}


def failed_result(status: str, error: str = "") -> Dict[str, Any]:
    counters = empty_counters()
    counters["strategy_failed"] = int(status not in (OK, OFF, NO_INPUT))
    return {"status": status, "error": error, "analysis": None, "counters": counters, "prompt_hash": prompt_hash(),
            "seconds": 0.0, "calls": [], "reasons": [], "rejected": []}


def analyse(inp: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
            started: Optional[float] = None, vm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The step on a ready input: at most 2 calls within «deadline_s» from started; {"status", "error", "analysis"
    (None when nothing verified), "counters", "prompt_hash", "seconds", "calls", "reasons", "rejected"}; logged
    «strategy_step»."""
    t0 = started if started is not None else time.time()
    deadline = float(config().get("deadline_s") or 40)
    counters = empty_counters()
    calls: List[Dict[str, Any]] = []
    status, error, analysis, reasons, rejected = NO_INPUT, "", None, [], []
    if inp.get("articles") and inp.get("facts"):
        messages = build_messages(inp)
        first = call_once(messages, call, deadline - (time.time() - t0))
        calls.append({"attempt": 1, "status": first["status"], "seconds": first["seconds"], "error": first["error"]})
        status, error = first["status"], first["error"]
        if first["status"] == OK:
            checked = verify(first["data"], inp)
            reasons = list(checked["reasons"])
            rejected = [dict(r, attempt=1) for r in checked["rejected"]]
            if checked["reasons"]:
                side_mismatch = bool(checked.get("side_mismatch"))
                second = call_once(reask_messages(messages, first["reply"], checked["reasons"], inp, side_mismatch),
                                   call, deadline - (time.time() - t0))
                calls.append({"attempt": 2, "status": second["status"], "seconds": second["seconds"],
                              "error": second["error"]})
                counters["strategy_reasked"] = 1
                counters["strategy_side_reasked"] = int(side_mismatch)       # v79.18.2 (item 2)
                if side_mismatch:
                    reasons.append(side_statement(inp))                      # logged with the re-ask's reasons
                if second["status"] == OK:
                    again = verify(second["data"], inp)
                    for k in ("strategy_line_rejected", "strategy_bad_id"):     # the first reply's too
                        again["counters"][k] += checked["counters"][k]
                    again["counters"]["strategy_side_reasked"] = 0              # counted above, once
                    rejected += [dict(r, attempt=2) for r in again["rejected"]]
                    reasons += [r for r in again["reasons"] if r not in reasons]
                    checked = merge(checked, again)
                else:
                    logger.warning("strategy_step: re-ask %s: %s (the first reply's verification stands)",
                                   second["status"], second["error"])
            if checked["linee"]:
                analysis = finish(checked, inp, vm)
            else:
                status, error = FAILED, "nessuna linea verificata"
                counters["strategy_failed"] = 1
            for k, v in checked["counters"].items():
                counters[k] = counters.get(k, 0) + v
        else:
            counters["strategy_failed"] = 1
            logger.warning("strategy_step: %s: %s (the analysis step's rendering stands)", first["status"], first["error"])
    counters["strategy_calls"] = len(calls)
    seconds = round(time.time() - t0, 3)
    counters["strategy_seconds"] = seconds
    result = {"status": status, "error": error, "analysis": analysis, "counters": counters,
              "prompt_hash": prompt_hash(), "seconds": seconds, "calls": calls, "reasons": reasons,
              "rejected": rejected}
    _log("strategy_step", {"prompt_hash": result["prompt_hash"], "status": status, "error": error, "seconds": seconds,
                           "calls": calls, "counters": counters, "side": inp.get("side") or SIDE_NEUTRAL,
                           "side_source": inp.get("side_source") or "",                      # v79.18.2 (item 2)
                           "alternative": [a["article"] for a in inp.get("articles") or [] if a.get("alternative")],
                           "set": [a["article"] for a in inp.get("articles") or []],
                           "rulings": [r["id"] for r in inp.get("rulings") or []],
                           "allegations": list(inp.get("allegations") or []),
                           "ranked": replay_summary(result).get("ranked"),
                           "reordered": bool((analysis or {}).get("reordered")),
                           "model_order": (analysis or {}).get("model_order") or [],
                           "final_order": (analysis or {}).get("final_order") or [],
                           "rejected": rejected, "reasons": reasons,
                           "passi": [p["eid"] for p in (analysis or {}).get("passi") or []],
                           "riqualificazione": (analysis or {}).get("riqualificazione"),        # v79.18.1 (C3)
                           "contro_difesa": [(c["allegazione"], not c["senza_risposta"])
                                             for c in (analysis or {}).get("contro_difesa") or []]})
    return result


def run(ctx: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None) -> Dict[str, Any]:
    """The step on the method's context: ctx["strategy_input"] (state_input, set by the flow) with the registry's
    rulings on its articles (ctx["rulings_record"]); the result into ctx["strategy"], its seconds into
    ctx["timings"]["strategy_s"]. Flag off: status off; no input: no_input. Never raises."""
    if not enabled():
        return failed_result(OFF)
    inp = ctx.get("strategy_input")
    if not isinstance(inp, dict) or not inp.get("articles"):
        result = failed_result(NO_INPUT)
    else:
        inp = dict(inp)
        try:
            inp["rulings"] = rulings_for(ctx.get("rulings_record"), [a["article"] for a in inp["articles"]])
            result = analyse(inp, call=call, vm=ctx.get("verified_material") if isinstance(ctx.get("verified_material"), dict) else None)
        except Exception as exc:        # noqa: BLE001 -- the step never fails the answer
            logger.warning("strategy_step: failed: %s", exc)
            result = failed_result(FAILED, f"{type(exc).__name__}: {exc}"[:300])
    ctx["strategy"] = result
    ctx.setdefault("timings", {})["strategy_s"] = result.get("seconds", 0.0)
    return result


def replay_summary(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the replay row and the state keep: {status, ranked [{articolo, eid, tipo, rapporto, forza}], reordered,
    model_order, final_order, counters, seconds, prompt_hash, error, objections [(answered)], contro_difesa}."""
    result = result if isinstance(result, dict) else {}
    a = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
    return {"status": result.get("status") or "", "error": result.get("error") or "",
            "ranked": [{"articolo": x["articolo"], "eid": x["eid"], "tipo": x["tipo"], "rapporto": x["rapporto"],
                        "forza": (x.get("forza") or {}).get("testo") or "", "key": (x.get("forza") or {}).get("key"),
                        "pronunce": list(x.get("cita_pronunce") or [])} for x in a.get("linee") or []],
            "reordered": bool(a.get("reordered")), "model_order": list(a.get("model_order") or []),
            "final_order": list(a.get("final_order") or []),
            "objections": [{"linea": x["eid"], "answered": not (x.get("obiezione") or {}).get("senza_risposta", True)}
                           for x in a.get("linee") or [] if x.get("obiezione")],
            "contro_difesa": [{"sid": c["allegazione"], "answered": not c["senza_risposta"]}
                              for c in a.get("contro_difesa") or []],
            "riqualificazione": a.get("riqualificazione"),                                  # v79.18.1 (C3)
            "side": a.get("side") or "", "side_source": a.get("side_source") or "",         # v79.18.2 (item 2)
            "alternative": list(a.get("alternative") or []),                                # v79.18.2 (item 6)
            "counters": dict(result.get("counters") or {}), "seconds": result.get("seconds", 0.0),
            "prompt_hash": result.get("prompt_hash") or prompt_hash(), "risk": risk_word(result),
            "risk_why": str((a.get("risk") or {}).get("why") or ""),                     # v79.18.3 (item 1)
            "risk_capped": bool((a.get("risk") or {}).get("capped"))}


# --- rendering --------------------------------------------------------------------------------------------------------

def _texts(v: Dict[str, Any]) -> Dict[str, str]:
    return {k: str(x or "") for k, x in (v.get("sentences") or {}).items()}


def _quoted(ids: Sequence[str], texts: Dict[str, str]) -> str:
    cfg = rendering()
    got = [str(cfg.get("fatto") or "«{testo}»").format(testo=_stop(texts[i])) for i in ids or [] if texts.get(i)]
    return str(cfg.get("fatti_sep") or "; ").join(got)


def _word(kind: str, value: Any) -> str:
    """v79.18.5 (item 3): the page's word for a tipo / rapporto label (data rendering «tipo_words» / «rapporto_words»);
    the label itself when the data has no word."""
    words = rendering().get(kind) or {}
    return str(words.get(str(value or "")) or value or "")


def ordinal(n: int) -> str:
    """v79.18.5 (item 3): «prima linea», «seconda linea» … (data rendering «ordinali»), never «Linea 2»."""
    cfg = rendering()
    words = [str(x) for x in cfg.get("ordinali") or []]
    word = words[n - 1] if 0 < n <= len(words) else f"{n}ª"
    return str(cfg.get("linea_ordinale") or "{ordinale} linea").format(ordinale=word)


def _g_text(a: Dict[str, Any], g: str) -> str:
    """A ruling's reference for the page: its estremi (analysis «rulings_estremi»), never the G-id."""
    return str((a.get("rulings_estremi") or {}).get(g) or g)


def _e_text(a: Dict[str, Any], eid: str) -> str:
    """An element's schema text for the page (analysis «elementi_testi»), never the E-id."""
    return _flat((a.get("elementi_testi") or {}).get(eid)) or str(eid)


def swept(text: str, counters: Optional[Dict[str, Any]] = None) -> str:
    """v79.18.5 (item 3): the final sweep on a rendered block -- a raw id (S3, G1, A640cp:E1, A640cp.2) the renderer
    did not turn into text is removed and counted (residues_dropped)."""
    found = _RESIDUE_RE.findall(text or "")
    if not found:
        return text or ""
    if counters is not None:
        counters["residues_dropped"] = counters.get("residues_dropped", 0) + len(found)
    out = _RESIDUE_RE.sub("", text or "")
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"\s+([,.;:)])", r"\1", out)


def line_sentence(linea: Dict[str, Any], side: str, v: Dict[str, Any]) -> str:
    """«art. X — <tesi>»: the tesi the analysis step's data gives for the side and the eid's esito
    (analysis_step rendering.tesi), {elemento} the schema text, {fatti} the linea's cited facts as written."""
    cfg = _as().rendering().get("tesi") or {}
    esito = linea.get("esito") if linea.get("esito") in ESITO_RANK else DA_PROVARE
    template = str(((cfg.get(side) or cfg.get(SIDE_NEUTRAL) or {}).get(esito)) or "«{elemento}»")
    fatti = _quoted(linea.get("cita_fatti") or [], _texts(v)) or str(cfg.get("nessun_fatto") or "nessun fatto citato")
    return f"art. {linea['articolo']} — " + _stop(template.format(elemento=linea.get("elemento") or "",
                                                                 art=linea["articolo"], fatti=fatti))


def summary_lines(result: Dict[str, Any], v: Dict[str, Any], counters: Optional[Dict[str, Any]] = None
                  ) -> List[Tuple[str, str]]:
    a = (result or {}).get("analysis") or {}
    out = []
    for i, x in enumerate(a.get("linee") or []):
        label = _as().LABEL_MAIN if i == 0 else (_as().LABEL_SUB if i == 1 else ordinal(i + 1).capitalize())
        out.append((label, swept(line_sentence(x, a.get("side") or SIDE_NEUTRAL, v), counters)))
    return out


def summary_text(v: Dict[str, Any], result: Dict[str, Any]) -> str:
    """§1 with the strategy: the pretesa line, the honesty line when the set is empty, then the ranked linee."""
    lines = summary_lines(result, v)
    if not lines:
        return _as().summary_text(v)
    return _as().summary_text(v, lines=lines)


def objection_items(result: Dict[str, Any], v: Dict[str, Any], counters: Optional[Dict[str, Any]] = None
                    ) -> List[Dict[str, str]]:
    """§7's entries: each ranked linea's obiezione (the model's sentence, reviewed and rendered) with its accusa_forte
    (v79.18.5, item 5: the facts the counterparty relies on, as written) and its verified answer -- the cited facts as
    written, the rulings by their estremi (item 3: never a G-id); none cited: the data's «senza_risposta»."""
    a = (result or {}).get("analysis") or {}
    texts = _texts(v)
    cfg = rendering()
    out = []
    for x in a.get("linee") or []:
        o = x.get("obiezione")
        if not o or not o.get("testo"):
            continue
        if o.get("senza_risposta"):
            answer = str(cfg.get("senza_risposta") or "senza risposta nei fatti narrati")
        else:
            answer = _quoted(o.get("risposta_cita_fatti") or [], texts) or \
                str(cfg.get("senza_risposta") or "senza risposta nei fatti narrati")
        if o.get("cita_pronunce"):
            answer += " " + " ".join(f"({_g_text(a, g)})" for g in o["cita_pronunce"])
        answer = answer.strip().rstrip(".")
        testo = _stop(o["testo"]).rstrip(".")
        forte = o.get("accusa_forte") or {}
        if forte.get("testo") or forte.get("cita_fatti"):         # v79.18.5 (item 5): the facts the counterparty relies on
            fatti = _stop(forte["testo"]).rstrip(".") if forte.get("testo") else _quoted(forte.get("cita_fatti") or [], texts)
            testo += ". " + str(cfg.get("accusa_forte") or "La controparte farà leva su: {fatti}").format(fatti=fatti).rstrip(".")
        out.append({"obiezione": swept(testo, counters), "risposta": swept((answer[:1].upper() + answer[1:]) + ".", counters),
                    "fonte": "strategia", "linea": x["eid"]})
    return out


def quesiti_map(inp: Optional[Dict[str, Any]], by_question: Optional[Dict[str, Sequence[str]]] = None
                ) -> Dict[str, List[str]]:
    """{Qn: the articles that answer the n-th quesito}: the fit's governs tags (inp «quesiti_articles»), and for a
    quesito the fit tagged nothing for, the articles its own search brought that the answer holds (by_question, the
    method's ctx «quesiti_articles» keyed by the question's text)."""
    inp = inp or {}
    out = {str(k): [str(x) for x in v] for k, v in (inp.get("quesiti_articles") or {}).items()}
    if by_question:
        from . import lawyer_method
        for n, q in enumerate(inp.get("quesiti") or [], 1):
            if out.get(f"Q{n}"):
                continue
            for text, labels in by_question.items():
                if text == q or lawyer_method.same_line(str(text), str(q)):
                    out[f"Q{n}"] = [str(x) for x in labels]
                    break
    return out


def _quesiti_text(a: Dict[str, Any], inp_quesiti: Sequence[str], quesiti_articles: Dict[str, Sequence[str]]) -> List[str]:
    cfg = rendering()
    out = []
    for i, q in enumerate(inp_quesiti or [], 1):
        labels = list(quesiti_articles.get(f"Q{i}") or [])
        hits = [str(cfg.get("quesito_linea") or "art. {art} ({linea})").format(art=x["articolo"], n=k + 1, linea=ordinal(k + 1))
                for k, x in enumerate(a.get("linee") or [])
                if any(_as().same_article(x["articolo"], y) for y in labels)]
        out.append(str(cfg.get("quesito") or "Q{n} «{quesito}»: {linee}").format(
            n=i, quesito=_stop(q), linee=", ".join(hits) or str(cfg.get("quesito_senza_linea") or "nessuna linea")))
    return out


def solutions_text(result: Dict[str, Any], v: Dict[str, Any], quesiti: Sequence[str] = (),
                   quesiti_articles: Optional[Dict[str, Sequence[str]]] = None,
                   counters: Optional[Dict[str, Any]] = None) -> str:
    """§6 (item 12's order): passi immediati -> the ranked linee with their forza (the table) -> risposta ai quesiti ->
    contro-difesa -> verifiche documentali. Labels and the model's one-sentence texts only; the facts as written.
    v79.18.5 (item 3): the tipo and rapporto as words, the lines by ordinal, the elements by their text, the rulings
    by their estremi, no id on the page (swept)."""
    a = (result or {}).get("analysis") or {}
    cfg = rendering()
    titles = cfg.get("titoli") or {}
    texts = _texts(v)
    parts: List[str] = []
    els = lambda ids: "; ".join(f"«{_e_text(a, e)}»" for e in ids)         # noqa: E731
    # passi immediati
    rows = [f"- {str(cfg.get('immediato') or '{testo}').format(testo=_stop(p['testo']))}"
            + (f": {_quoted(p['cita_fatti'], texts)}" if p.get("cita_fatti") else "")
            + (f" ({els(p['cita_elementi'])})" if p.get("cita_elementi") else "") + "."
            for p in a.get("passi_immediati") or []]
    parts.append(f"{titles.get('passi_immediati') or '**Passi immediati**'}\n\n"
                 + ("\n".join(rows) or str(cfg.get("nessun_passo_immediato") or "Nessun passo immediato indicato.")))
    # the linee per forza
    table = list(cfg.get("tabella") or [])
    for i, x in enumerate(a.get("linee") or [], 1):
        f = x.get("forza") or {}
        forza_text = f.get("testo") or "—"
        if not f.get("valutato"):
            forza_text += f" ({cfg.get('non_valutati') or 'elementi non valutati'})"
        elemento = _flat(x.get("elemento")) + (str(cfg.get("condizione") or " [{g}]").format(g=_g_text(a, x["condizione"]))
                                                if x.get("condizione") else "")       # v79.18.1 (C2); (item 3) the estremi
        table.append(f"| {ordinal(i).capitalize()} | art. {_cell(x['articolo'])} | "
                     f"{_cell(elemento)} | {_cell(_word('tipo_words', x.get('tipo')))} | {_cell(_word('rapporto_words', x.get('rapporto')))} | "
                     f"{_cell(x.get('perche_prima') or cfg.get('nessuno') or '—')} | {_cell(forza_text)} | "
                     f"{_cell(_quoted(x.get('cita_fatti') or [], texts) or cfg.get('nessun_fatto') or 'nessun fatto citato')} | "
                     f"{_cell('; '.join(_g_text(a, g) for g in x.get('cita_pronunce') or []) or cfg.get('nessuno') or 'nessuno')} |")
    block = "\n".join(table)
    if a.get("reordered"):          # (item 3): the model's order by article and element text, never by ids
        order = []
        for item in a.get("model_order") or []:
            art, _, eid = str(item).rpartition(" ")
            order.append(f"art. {art} «{_e_text(a, eid)}»" if art else str(item))
        block += "\n\n" + str(cfg.get("riordinata") or "Ordine ricalcolato dal codice per forza ({ordine}).").format(
            ordine="; ".join(order))
    parts.append(f"{titles.get('linee') or '**Linee per forza**'}\n\n{block}")
    riq = a.get("riqualificazione")     # v79.18.1 (C3): the penal defence's qualificazione alternativa, always said
    if riq:
        text = str(cfg.get("riqualificazione_linea") or "art. {art} ({linea})").format(art=riq["articolo"], n=riq["n"], linea=ordinal(riq["n"])) \
            if riq.get("linea") else str(cfg.get("riqualificazione_nessuna") or "nessuna qualificazione alternativa ({articoli})").format(
                articoli=", ".join(f"art. {x}" for x in riq.get("alternative") or []) or str(cfg.get("nessuno") or "nessuno"))
        parts.append(f"{cfg.get('riqualificazione_titolo') or '**Qualificazione alternativa**'}\n\n{text[:1].upper() + text[1:]}.")
    # risposta ai quesiti
    if quesiti:
        parts.append(f"{titles.get('quesiti') or '**Risposta ai quesiti**'}\n\n"
                     + "\n".join(f"- {x}" for x in _quesiti_text(a, quesiti, quesiti_articles or {})))
    # contro-difesa
    rows = []
    for c in a.get("contro_difesa") or []:
        if c.get("senza_risposta"):
            risposta = str(cfg.get("contro_senza") or "senza risposta")
        else:
            risposta = _stop(c.get("risposta")) or _quoted(c.get("cita_fatti") or [], texts)
            if c.get("cita_fatti") and c.get("risposta"):
                risposta += f": {_quoted(c['cita_fatti'], texts)}"
            linea = next((x for x in a.get("linee") or [] if x["eid"] == c.get("cita_linea")), None)
            if linea:
                risposta += str(cfg.get("contro_linea") or " (linea: art. {art}, «{elemento}»)").format(
                    art=linea["articolo"], elemento=linea.get("elemento") or "")
        who = _flat((a.get("speakers") or {}).get(c["allegazione"]))
        form = cfg.get("contestazione_accusa") if _as().is_authority(who) else cfg.get("contro")   # v79.18.1 (C7)
        rows.append("- " + str(form or "Allegazione «{frase}»: {risposta}").format(
            sid="", frase=_stop(texts.get(c["allegazione"], "")), risposta=risposta) + ".")
    parts.append(f"{titles.get('contro_difesa') or '**Contro-difesa**'}\n\n"
                 + ("\n".join(rows) or str(cfg.get("nessuna_allegazione") or "Nessuna allegazione della controparte.")))
    # verifiche documentali
    rows = [f"- {str(cfg.get('verifica') or '{testo}').format(testo=_stop(p['testo']))}"
            + (f": {_quoted(p['cita_fatti'], texts)}" if p.get("cita_fatti") else "")
            + (f" ({els(p['cita_elementi'])})" if p.get("cita_elementi") else "") + "."
            for p in a.get("verifiche_documentali") or []]
    parts.append(f"{titles.get('verifiche') or '**Verifiche documentali**'}\n\n"
                 + ("\n".join(rows) or str(cfg.get("nessuna_verifica") or "Nessuna verifica documentale indicata.")))
    return swept("\n\n".join(parts), counters)


def passi_elements(result: Dict[str, Any]) -> List[str]:
    """The schema texts of the elements the passi name (§10 prints each such element once: the passo, not the
    analysis's question on it)."""
    return [_flat(p.get("elemento")) for p in ((result or {}).get("analysis") or {}).get("passi") or [] if p.get("elemento")]


def passi_points(result: Dict[str, Any], v: Dict[str, Any], counters: Optional[Dict[str, Any]] = None) -> List[str]:
    """§10's passi: one line per open element of the principale -- the model's sentence (its cited fact as written)
    or the code's label."""
    a = (result or {}).get("analysis") or {}
    cfg = rendering()
    texts = _texts(v)
    art = (a.get("principale") or {}).get("articolo") or ""
    out = []
    for p in a.get("passi") or []:
        if p.get("codice") or not p.get("cosa"):
            out.append(str(cfg.get("passo_codice") or "«{elemento}» (art. {art}): da provare").format(
                elemento=p.get("elemento") or "", art=art))
            continue
        cosa = _stop(p["cosa"])
        if p.get("cita_fatti"):
            cosa += f" ({_quoted(p['cita_fatti'], texts)})"
        out.append(str(cfg.get("passo_punto") or "Passo: «{elemento}» (art. {art}) — {cosa}").format(
            elemento=p.get("elemento") or "", art=art, cosa=cosa))
    return [swept(x.rstrip(".") + ".", counters) for x in out]
