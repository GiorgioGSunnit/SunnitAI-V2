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
            "strategy_entries_uncited")
OK, FAILED, MALFORMED, TIMEOUT, NO_INPUT, OFF = "ok", "failed", "malformed", "timeout", "no_input", "off"
PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO = "provato", "contestato", "da_provare", "escluso"
ESITO_RANK = {PROVATO: 3, CONTESTATO: 2, DA_PROVARE: 1, ESCLUSO: 0}
OPEN = (CONTESTATO, DA_PROVARE)
SIDE_CLAIMANT, SIDE_DEFENCE, SIDE_NEUTRAL = "attore", "difesa", "neutro"
FAVOREVOLE, INCERTO, SFAVOREVOLE = "favorevole", "incerto", "sfavorevole"
PRINCIPALE, SUBORDINATA = "principale", "subordinata"
_S_RE = re.compile(r"^S\d+$")
_G_RE = re.compile(r"^G\d+$")

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
    labels = list(in_set)
    for kind in A.LINE_KINDS:
        x = a.get(kind)
        if x and x.get("articolo") and x.get("stato") == A.VERIFIED and not any(
                A.same_article(x["articolo"], y) for y in labels):
            labels.append(str(x["articolo"]))
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
        articles.append({"article": str(n["article"]), "token": _ap().article_token(n["article"]),
                         "rubrica": _flat(n.get("rubrica")), "tier": tiers.get(label) or tiers.get(n["article"])
                         or ("linea" if not label_in_set else "non_verificato"),
                         "in_set": label_in_set, "valutato": bool(got), "elements": elements})
    if not articles:
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
    quesiti = flow.fit_pretesa_quesiti(state)[1] if held else [str(q) for q in flow.asked_questions(state)]
    return {"pretesa": _flat(verified.get("pretesa_line") or (case_frame.pretesa_text(held) if held else "")),
            "client": client, "side": a.get("side") or SIDE_NEUTRAL,
            "charge": _flat(a.get("frame_charge") or binding.get("pretesa_article")),
            "articles": articles, "facts": facts,
            "allegations": allegations(facts, frame, client),
            "quesiti": [str(q) for q in quesiti if _flat(q)],
            "quesiti_articles": {str(k): [str(x) for x in v] for k, v in (binding.get("quesiti") or {}).items()},
            "lines": [{"kind": k, "articolo": a[k].get("articolo"), "eid": a[k].get("eid"), "stato": a[k].get("stato")}
                      for k in A.LINE_KINDS if a.get(k)],
            "honesty": _flat(a.get("honesty")), "rulings": []}


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
        parts.append(f"{f['id']} [{tag}] {f['text']}")
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
    for art in inp.get("articles") or []:
        ids = ", ".join(f"{e['eid']} ({e['esito']})" for e in art.get("elements") or [])
        rows.append(f"- A{art['token']} (art. {art['article']}): {ids}")
    return "\n".join(rows) or f"- {(config().get('labels') or {}).get('none') or '(nessuno)'}"


def reask_messages(messages: List[Any], reply: str, reasons: Sequence[str], inp: Dict[str, Any]) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage
    out = [line.replace("{reasons}", "\n".join(f"- {r}" for r in reasons)).replace("{allowed}", allowed_text(inp))
           for line in config().get("reask") or []]
    return list(messages) + [AIMessage(content=str(reply or "")), HumanMessage(content="\n".join(out))]


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


def _verify_objection(raw: Any, inp: Dict[str, Any], counters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict) or not _flat(raw.get("testo")):
        return None
    answer = raw.get("risposta_cita_fatti")
    sids, _bad = _sids(answer, inp, counters) if not _no_answer(answer) else ([], [])
    gids = _gids(raw.get("cita_pronunce"), inp, counters)
    return {"testo": _flat(raw.get("testo")), "risposta_cita_fatti": sids, "cita_pronunce": gids,
            "senza_risposta": not sids}         # counted in finish, on the final linee only


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
    return {"articolo": art["article"], "token": art["token"], "eid": e["eid"], "elemento": e["testo"],
            "esito": e["esito"], "tipo": tipo, "rapporto": rapporto, "rapporto_modello": rapporto,
            "perche_prima": _flat(raw.get("perche_prima")), "cita_fatti": sids, "cita_pronunce": gids,
            "obiezione": _verify_objection(raw.get("obiezione"), inp, counters), "tier": art.get("tier")}, ""


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
        linee.append(line)
    if not raw_lines:
        reasons.append("«linee» mancante o vuota")
    lo = int(config().get("lines_min") or 2)
    if len(linee) < lo:
        reasons.append(f"servono almeno {lo} linee verificate (ne restano {len(linee)})")
    return {"linee": linee[:int(config().get("lines_max") or 3)], "rejected": rejected, "reasons": reasons,
            "raw": reply, "counters": counters}


# --- forza and the ranking --------------------------------------------------------------------------------------------

def forza(label: str, eid: str, inp: Dict[str, Any]) -> Dict[str, Any]:
    """The line's strength, computed from the article's verified table by side: {provato, contestato, da_provare,
    escluso, totale, key, testo, valutato}. The defence's key: contestato + da_provare on the article (more = stronger).
    A claimant's (item 10): the eid's esito rank, then minus the further conditions in the same comma, then the provato
    count. Ties compare equal."""
    art = next((a for a in inp.get("articles") or [] if _as().same_article(a["article"], label)), None) or {}
    els = art.get("elements") or []
    n = {k: sum(1 for e in els if e.get("esito") == k) for k in ESITO_RANK}
    e = next((x for x in els if x.get("eid") == eid), None) or {}
    if (inp.get("side") or SIDE_NEUTRAL) == SIDE_DEFENCE:
        key: Tuple[Any, ...] = (n[CONTESTATO] + n[DA_PROVARE],)
    else:
        further = sum(1 for x in els if x.get("eid") != eid and x.get("comma") is not None
                      and x.get("comma") == e.get("comma")) if e.get("comma") is not None else 0
        key = (ESITO_RANK.get(e.get("esito"), 1), -further, n[PROVATO])
    cfg = rendering()
    testo = str(cfg.get("forza") or "provato {provato} / contestato {contestato} / da provare {da_provare}").format(**n)
    if n[ESCLUSO]:
        testo += str(cfg.get("forza_escluso") or " / escluso {escluso}").format(**n)
    return {**n, "totale": len(els), "key": list(key), "testo": testo, "valutato": bool(art.get("valutato"))}


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


def risk_word(result: Optional[Dict[str, Any]]) -> str:
    """§9's word for the client from the principale's forza (data «risk», by side): the share of the measured esiti
    at favorevole_min or more -> favorevole, at sfavorevole_max or less -> sfavorevole, else incerto; an escluso
    element -> sfavorevole for the pretesa's holder, favorevole for the defence. "" without a verified strategy."""
    a = (result or {}).get("analysis") if isinstance(result, dict) else None
    if not a or not a.get("linee"):
        return ""
    f = a["linee"][0].get("forza") or {}
    side = a.get("side") or SIDE_NEUTRAL
    cfg = (config().get("risk") or {}).get(side) or (config().get("risk") or {}).get(SIDE_NEUTRAL) or {}
    tot = int(f.get("totale") or 0)
    if not tot:
        return ""
    if f.get(ESCLUSO):
        return FAVOREVOLE if side == SIDE_DEFENCE else SFAVOREVOLE
    measure = f.get(PROVATO, 0) if str(cfg.get("misura") or "provato") == "provato" else \
        f.get(CONTESTATO, 0) + f.get(DA_PROVARE, 0)
    share = measure / tot
    if share >= float(cfg.get("favorevole_min", 1.0)):
        return FAVOREVOLE
    if share <= float(cfg.get("sfavorevole_max", 0.0)):
        return SFAVOREVOLE
    return INCERTO


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


def finish(checked: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """The verified strategy from a checked reply: the linee with their forza, ranked (rank; the first the principale,
    another «principale» made «subordinata»), the passi, item 12's entries, the ranking orders and the side."""
    counters = checked["counters"]
    linee = [dict(x, forza=forza(x["articolo"], x["eid"], inp)) for x in checked["linee"]]
    model_order = [f"{x['articolo']} {x['eid']}" for x in linee]
    ranked, reordered = rank(linee)
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
    return {"linee": ranked, "principale": principale, "side": inp.get("side") or SIDE_NEUTRAL,
            "reordered": reordered, "model_order": model_order,
            "final_order": [f"{x['articolo']} {x['eid']}" for x in ranked],
            "passi": _passi(raw.get("passi"), principale, inp, counters) if principale else [],
            "passi_immediati": _cited_entries(raw.get("passi_immediati"), inp, counters),
            "contro_difesa": _contro_difesa(raw.get("contro_difesa"), ranked, inp, counters),
            "verifiche_documentali": _cited_entries(raw.get("verifiche_documentali"), inp, counters),
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
            started: Optional[float] = None) -> Dict[str, Any]:
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
                second = call_once(reask_messages(messages, first["reply"], checked["reasons"], inp), call,
                                   deadline - (time.time() - t0))
                calls.append({"attempt": 2, "status": second["status"], "seconds": second["seconds"],
                              "error": second["error"]})
                counters["strategy_reasked"] = 1
                if second["status"] == OK:
                    again = verify(second["data"], inp)
                    for k in ("strategy_line_rejected", "strategy_bad_id"):     # the first reply's too
                        again["counters"][k] += checked["counters"][k]
                    rejected += [dict(r, attempt=2) for r in again["rejected"]]
                    reasons += [r for r in again["reasons"] if r not in reasons]
                    checked = merge(checked, again)
                else:
                    logger.warning("strategy_step: re-ask %s: %s (the first reply's verification stands)",
                                   second["status"], second["error"])
            if checked["linee"]:
                analysis = finish(checked, inp)
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
                           "set": [a["article"] for a in inp.get("articles") or []],
                           "rulings": [r["id"] for r in inp.get("rulings") or []],
                           "allegations": list(inp.get("allegations") or []),
                           "ranked": replay_summary(result).get("ranked"),
                           "reordered": bool((analysis or {}).get("reordered")),
                           "model_order": (analysis or {}).get("model_order") or [],
                           "final_order": (analysis or {}).get("final_order") or [],
                           "rejected": rejected, "reasons": reasons,
                           "passi": [p["eid"] for p in (analysis or {}).get("passi") or []],
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
            result = analyse(inp, call=call)
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
            "counters": dict(result.get("counters") or {}), "seconds": result.get("seconds", 0.0),
            "prompt_hash": result.get("prompt_hash") or prompt_hash(), "risk": risk_word(result)}


# --- rendering --------------------------------------------------------------------------------------------------------

def _texts(v: Dict[str, Any]) -> Dict[str, str]:
    return {k: str(x or "") for k, x in (v.get("sentences") or {}).items()}


def _quoted(ids: Sequence[str], texts: Dict[str, str]) -> str:
    cfg = rendering()
    got = [str(cfg.get("fatto") or "«{testo}»").format(testo=_stop(texts[i])) for i in ids or [] if texts.get(i)]
    return str(cfg.get("fatti_sep") or "; ").join(got)


def line_sentence(linea: Dict[str, Any], side: str, v: Dict[str, Any]) -> str:
    """«art. X — <tesi>»: the tesi the analysis step's data gives for the side and the eid's esito
    (analysis_step rendering.tesi), {elemento} the schema text, {fatti} the linea's cited facts as written."""
    cfg = _as().rendering().get("tesi") or {}
    esito = linea.get("esito") if linea.get("esito") in ESITO_RANK else DA_PROVARE
    template = str(((cfg.get(side) or cfg.get(SIDE_NEUTRAL) or {}).get(esito)) or "«{elemento}»")
    fatti = _quoted(linea.get("cita_fatti") or [], _texts(v)) or str(cfg.get("nessun_fatto") or "nessun fatto citato")
    return f"art. {linea['articolo']} — " + _stop(template.format(elemento=linea.get("elemento") or "",
                                                                 art=linea["articolo"], fatti=fatti))


def summary_lines(result: Dict[str, Any], v: Dict[str, Any]) -> List[Tuple[str, str]]:
    a = (result or {}).get("analysis") or {}
    out = []
    for i, x in enumerate(a.get("linee") or []):
        label = _as().LABEL_MAIN if i == 0 else (_as().LABEL_SUB if i == 1 else
                                                 str(rendering().get("linea") or "Linea {n}").format(n=i + 1))
        out.append((label, line_sentence(x, a.get("side") or SIDE_NEUTRAL, v)))
    return out


def summary_text(v: Dict[str, Any], result: Dict[str, Any]) -> str:
    """§1 with the strategy: the pretesa line, the honesty line when the set is empty, then the ranked linee."""
    lines = summary_lines(result, v)
    if not lines:
        return _as().summary_text(v)
    return _as().summary_text(v, lines=lines)


def objection_items(result: Dict[str, Any], v: Dict[str, Any]) -> List[Dict[str, str]]:
    """§7's entries: each ranked linea's obiezione (the model's sentence) with its verified answer -- the cited facts as
    written, the rulings by their registry label «(G1)» (compose renders it as the reference); none cited: the data's
    «senza_risposta»."""
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
            answer += " " + " ".join(f"({g})" for g in o["cita_pronunce"])
        answer = answer.strip().rstrip(".")
        out.append({"obiezione": _stop(o["testo"]), "risposta": (answer[:1].upper() + answer[1:]) + ".",
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
        hits = [str(cfg.get("quesito_linea") or "art. {art} (linea {n})").format(art=x["articolo"], n=k + 1)
                for k, x in enumerate(a.get("linee") or [])
                if any(_as().same_article(x["articolo"], y) for y in labels)]
        out.append(str(cfg.get("quesito") or "Q{n} «{quesito}»: {linee}").format(
            n=i, quesito=_stop(q), linee=", ".join(hits) or str(cfg.get("quesito_senza_linea") or "nessuna linea")))
    return out


def solutions_text(result: Dict[str, Any], v: Dict[str, Any], quesiti: Sequence[str] = (),
                   quesiti_articles: Optional[Dict[str, Sequence[str]]] = None) -> str:
    """§6 (item 12's order): passi immediati -> the ranked linee with their forza (the table) -> risposta ai quesiti ->
    contro-difesa -> verifiche documentali. Labels and the model's one-sentence texts only; the facts as written."""
    a = (result or {}).get("analysis") or {}
    cfg = rendering()
    titles = cfg.get("titoli") or {}
    texts = _texts(v)
    parts: List[str] = []
    # passi immediati
    rows = [f"- {str(cfg.get('immediato') or '{testo}').format(testo=_stop(p['testo']))}"
            + (f": {_quoted(p['cita_fatti'], texts)}" if p.get("cita_fatti") else "")
            + (f" [{', '.join(p['cita_elementi'])}]" if p.get("cita_elementi") else "") + "."
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
        table.append(f"| {str(cfg.get('linea') or 'Linea {n}').format(n=i)} | art. {_cell(x['articolo'])} | "
                     f"{_cell(x.get('elemento'))} | {_cell(x.get('tipo'))} | {_cell(x.get('rapporto'))} | "
                     f"{_cell(x.get('perche_prima'))} | {_cell(forza_text)} | "
                     f"{_cell(_quoted(x.get('cita_fatti') or [], texts) or cfg.get('nessun_fatto') or 'nessun fatto citato')} | "
                     f"{_cell(' '.join(f'({g})' for g in x.get('cita_pronunce') or []) or cfg.get('nessuno') or 'nessuno')} |")
    block = "\n".join(table)
    if a.get("reordered"):
        block += "\n\n" + str(cfg.get("riordinata") or "Ordine ricalcolato dal codice per forza ({ordine}).").format(
            ordine=", ".join(a.get("model_order") or []))
    parts.append(f"{titles.get('linee') or '**Linee per forza**'}\n\n{block}")
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
        rows.append("- " + str(cfg.get("contro") or "Allegazione {sid} «{frase}»: {risposta}").format(
            sid=c["allegazione"], frase=_stop(texts.get(c["allegazione"], "")), risposta=risposta) + ".")
    parts.append(f"{titles.get('contro_difesa') or '**Contro-difesa**'}\n\n"
                 + ("\n".join(rows) or str(cfg.get("nessuna_allegazione") or "Nessuna allegazione della controparte.")))
    # verifiche documentali
    rows = [f"- {str(cfg.get('verifica') or '{testo}').format(testo=_stop(p['testo']))}"
            + (f": {_quoted(p['cita_fatti'], texts)}" if p.get("cita_fatti") else "")
            + (f" [{', '.join(p['cita_elementi'])}]" if p.get("cita_elementi") else "") + "."
            for p in a.get("verifiche_documentali") or []]
    parts.append(f"{titles.get('verifiche') or '**Verifiche documentali**'}\n\n"
                 + ("\n".join(rows) or str(cfg.get("nessuna_verifica") or "Nessuna verifica documentale indicata.")))
    return "\n\n".join(parts)


def passi_elements(result: Dict[str, Any]) -> List[str]:
    """The schema texts of the elements the passi name (§10 prints each such element once: the passo, not the
    analysis's question on it)."""
    return [_flat(p.get("elemento")) for p in ((result or {}).get("analysis") or {}).get("passi") or [] if p.get("elemento")]


def passi_points(result: Dict[str, Any], v: Dict[str, Any]) -> List[str]:
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
    return [x.rstrip(".") + "." for x in out]
