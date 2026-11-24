"""v79.17.0 (item 4): prose written from verified material, not censored after.

Under ANALYSIS_STEP with a verified analysis, every writer call for the Q&A, §4 (per article) and §7 receives ONE
material block (block): the pretesa, the client and its side, the verified line(s) with their elements' esiti, the
articles tagged for that question / section with their cited passages as period ids («[A1385cc.1] text»), and the
record's facts as sentence ids («[S3] text»). The writer must cite the ids inline; the renderer (render_ids) turns
«[A1385cc.1]» into «art. 1385 c.c.» and «[S3]» into the fact's quote.

The post-processing of that prose is one rule (review): a sentence citing an article outside the fitted set (the
analysis input's norms, the remedies, the articles the user named) or an id that does not exist is dropped -- no
replacement, no dependent-sentence drop, no lemma matching. The events go to the prose review's list (analysis_step
.prose_counters reads them: qa_sentences_dropped, norms_sentences_dropped, objections_sentences_dropped,
prose_unfitted_article_dropped, prose_unknown_id_dropped). An answer that lost two or more sentences is written once
more with the same material (lawyer_method.answers_block); a fragment or an article's text alone is never an answer
(fragment_or_copy).

Inputs: the maps analysis_step.id_maps builds from the step's input (the periods by id, the sentences by id, the
norms), the view (analysis_step.view), the pretesa (case_frame.pretesa), the fit judgments (their cita_norma and
governs), the record's facts (fact_record.facts) and the questions asked.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis_step import (ARTICLE_EXCLUDED, ARTICLE_UNFITTED, DROPPED, ESITO_LABELS, SITE_NORMS, SITE_OBJECTIONS,
                            SITE_QA, _cited, _flat, _lines, bare_cited, line_text, same_article)

UNKNOWN_ID = "id_inesistente"
RULE_WITHOUT_ID = "regola_senza_id"         # v79.18.1 (C9): a rule attributed to article X with no A-id of X
LAW_VERBATIM = "legge_verbatim"             # v79.18.5 (item 2): a rule sentence printed as the article's passage
LAW_VERBATIM_SITES = (SITE_NORMS, SITE_OBJECTIONS, "strategia")      # §4, §7 and the strategy's fields (not the Q&A)
TITLE = "MATERIALE VERIFICATO (il solo materiale di questa sezione)"
ID_RULE = (" Cita ogni norma con l'id del periodo su cui ti fondi tra parentesi quadre (es. [A1385cc.1]) e ogni fatto con "
           "il suo id (es. [S3]): il codice sostituisce gli id con il riferimento alla norma e con la citazione del fatto. "
           "Solo le norme e i fatti del MATERIALE VERIFICATO: una frase che ne cita altri viene tolta. Non riportare il "
           "testo di una norma: applicala ai fatti.")
NORMS_ORDER_TITLE = "NORME NELL'ORDINE DEI PARAGRAFI"
FRAGMENT_WORDS = 6
COPY_OVERLAP = 0.9
_ID_RE = re.compile(r"\[(A[0-9a-z]+\.\d+|S\d+)\]")
# the review's sentence split: a stop, then a capital or a quote; never after «art.», «n.», «co.», «lgs.», «d.», «l.»
# (a code's «c.c.» may end a sentence: «… ai sensi dell'art. 1439 c.c. Pertanto, …» is two)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü«\"])")
_NO_END = re.compile(r"(?:\b(?:artt?|n|co|comma|lgs|d|l|lett|cfr|v)\.|\b[A-Z]\.)$")


def sentences_of(body: str) -> List[str]:
    out, start = [], 0
    for m in _SENT_SPLIT.finditer(body):
        if _NO_END.search(body[start:m.start()].rstrip()):
            continue
        out.append(body[start:m.start()])
        start = m.end()
    out.append(body[start:])
    return [x for x in out if x.strip()]
_WORD_RE = re.compile(r"[a-zà-ü0-9]+")


# --- the material -----------------------------------------------------------------------------------------------------

def build(view: Dict[str, Any], maps: Dict[str, Any], pretesa: Optional[Dict[str, Any]] = None, client: str = "",
          judgments: Sequence[Dict[str, Any]] = (), binding: Optional[Dict[str, Any]] = None,
          record_facts: Sequence[Dict[str, Any]] = (), quesiti: Sequence[str] = (),
          named: Sequence[str] = (), quesiti_articles: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """The material every writer call reads from: {"pretesa" (one line), "client", "side", "lines": [text], "esiti":
    {article: [«elemento» esito]}, "articles": {label: {"rubrica", "governs", "passages": [(id, text)]}}, "fitted":
    [labels], "named": [labels], "facts": [(sid, tag, text)], "quesiti": [texts], "maps"}; {} with no verified view."""
    if not isinstance(view, dict) or not isinstance(view.get("analysis"), dict) or not maps:
        return {}
    a = view["analysis"]
    side = {"attore": "parte che agisce", "difesa": "parte che resiste"}.get(str(a.get("side") or ""), "")
    lines = [f"{label}: {line_text(x)}" for label, x in _lines(view)]
    esiti = {art: [f"«{_flat(e.get('elemento'))}» {ESITO_LABELS.get(e.get('esito'), 'da provare')}" for e in els]
             for art, els in (a.get("elementi") or {}).items()}
    cited_by: Dict[str, List[str]] = {}
    governs_by: Dict[str, List[str]] = {}
    for j in judgments or []:
        if not isinstance(j, dict) or not j.get("article"):
            continue
        cited_by[str(j["article"])] = [str(x) for x in j.get("cita_norma") or []]
        if isinstance(j.get("governs"), list):
            governs_by[str(j["article"])] = list(j["governs"])
    for x in (a.get("linea_principale"), a.get("linea_subordinata")):
        if x and x.get("articolo"):
            cited_by.setdefault(x["articolo"], [])
            cited_by[x["articolo"]] = list(dict.fromkeys(cited_by[x["articolo"]] + [str(i) for i in x.get("cita_norma") or []]))
    periods = maps.get("periods") or {}
    articles: Dict[str, Dict[str, Any]] = {}
    for n in maps.get("norms") or []:
        label = str(n.get("article") or "")
        if not label:
            continue
        mine = [(pid, p["text"]) for pid, p in periods.items() if p.get("article") == label]
        cited = [(pid, text) for pid, text in mine if pid in (cited_by.get(label) or [])]
        articles[label] = {"rubrica": _flat(n.get("rubrica")), "governs": governs_by.get(label),
                           "passages": cited or mine[:2], "remedy": bool(n.get("remedy")), "fit": n.get("fit")}
    facts = []
    by_fid = {str(f.get("id") or ""): f for f in record_facts or [] if isinstance(f, dict) and f.get("id")}
    for sid, s in (maps.get("sentences") or {}).items():
        rec = by_fid.get(str(s.get("fact") or "")) or {}
        tag = str(rec.get("stato") or s.get("certainty") or "")
        if rec.get("chi"):
            tag += f", {rec['chi']}"
        facts.append((sid, tag, _flat(s.get("text"))))
    p = pretesa or {}
    what = unquoted(_flat(p.get("contenuto")) or _flat(p.get("frase")))
    pretesa_line = (f"{p.get('tipo') or 'domanda'}: «{what}»" + (f" (art. {p['articolo']})" if p.get("articolo") else "")
                    + (f", contro {p['contro']}" if p.get("contro") else "")) if p else ""
    qs = [_flat(q) for q in quesiti or [] if _flat(q)]
    by_n: Dict[int, List[str]] = {}          # the question's own retrieval articles (interactive_flow.quesiti_articles),
    for q, labels in (quesiti_articles or {}).items():      # the tag when the fit said nothing of that question
        n = next((i for i, x in enumerate(qs, 1) if _flat(q) == x or _flat(q)[:60] == x[:60]), None)
        if n:
            by_n[n] = [str(x) for x in labels if str(x) in articles]
    return {"pretesa": pretesa_line, "client": client or "", "side": side, "lines": lines, "esiti": esiti,
            "articles": articles, "fitted": list(articles), "named": [str(x) for x in named or []],
            "facts": facts, "quesiti": qs, "quesiti_articles": by_n,
            "binding": dict(binding or {}), "line_articles": [x.get("articolo") for _, x in _lines(view)
                                                              if x.get("articolo")], "maps": maps}


def unquoted(text: Any) -> str:
    """The pretesa's words without their guillemets and a final stop (an abbreviation's dot kept: «c.c.»)."""
    out = _flat(text).strip().strip("«»\"“”").strip().rstrip("?! ").strip()
    if out.endswith(".") and not re.search(r"(?<![A-Za-z])[A-Za-z]\.$", out):
        out = out[:-1].rstrip()
    return out


def articles_for(vm: Dict[str, Any], site: str, question_n: Optional[int] = None,
                 only: Sequence[str] = ()) -> List[str]:
    """The articles a call's block carries: the lines' always; for a question Q<n> those whose verdict governs it; for
    §4 and §7 those governing the pretesa, then the other fitted non-remedy norms when the binding is not applied."""
    arts = vm.get("articles") or {}
    if only:                    # one article alone (the named-article retry)
        return [x for x in only if x in arts]
    out = [x for x in vm.get("line_articles") or [] if x in arts]
    b = vm.get("binding") or {}
    if site == SITE_QA and question_n:
        tok = f"Q{question_n}"
        tagged = [x for x, a in arts.items() if tok in (a.get("governs") or [])]
        # the verdict tags it; with no verdict on a question's article its own retrieval stands (79.16.0 B2)
        out += tagged or [x for x in (vm.get("quesiti_articles") or {}).get(question_n) or []
                          if not isinstance(arts.get(x, {}).get("governs"), list)]
    else:
        out += [x for x, a in arts.items() if "pretesa" in (a.get("governs") or [])]
        if not b.get("applied"):
            out += [x for x, a in arts.items() if not a.get("remedy") and a.get("fit") != "no"]
    return list(dict.fromkeys(out))


def block(vm: Dict[str, Any], site: str, question: str = "", question_n: Optional[int] = None,
          only: Sequence[str] = ()) -> str:
    """One writer call's material block: the pretesa, the client and its side, the lines with their esiti, the
    section's articles with their cited passages by id, the facts by id; for §4 the paragraph order too."""
    if not vm:
        return ""
    lines = [TITLE]
    if vm.get("pretesa"):
        lines.append(f"Pretesa: {vm['pretesa']}.")
    lines.append(f"Parte assistita: {vm.get('client') or 'non indicata'}" + (f" ({vm['side']})" if vm.get("side") else "") + ".")
    if question:
        lines.append(f"Quesito{f' Q{question_n}' if question_n else ''}: «{_flat(question)}»")
    for x in vm.get("lines") or []:
        lines.append(x + ".")
    for art, els in (vm.get("esiti") or {}).items():
        lines.append(f"Esiti dell'art. {art}: " + "; ".join(els) + ".")
    labels = articles_for(vm, site, question_n, only)
    if labels:
        lines.append("ARTICOLI (con i periodi citati, da citare con il loro id):")
        for label in labels:
            a = vm["articles"][label]
            head = f"- art. {label}" + (f" ({a['rubrica']})" if a.get("rubrica") else "")
            if isinstance(a.get("governs"), list):
                head += " [governa: " + (", ".join(a["governs"]) or "solo contesto") + "]"
            lines.append(head)
            for pid, text in a.get("passages") or []:
                lines.append(f"  [{pid}] {_flat(text)}")
        if site == SITE_NORMS:
            lines.append(f"{NORMS_ORDER_TITLE}:")
            lines += [f"{i}. art. {label}" for i, label in enumerate(labels, 1)]
    if vm.get("facts"):
        lines.append("FATTI DEL CASO (da citare con il loro id):")
        lines += [f"[{sid}] [{tag}] {text}" if tag else f"[{sid}] {text}" for sid, tag, text in vm["facts"]]
    return "\n".join(lines)


# --- the one rule and the renderer --------------------------------------------------------------------------------------

def _article_of_id(pid: str, vm: Dict[str, Any]) -> str:
    return str(((vm.get("maps") or {}).get("periods") or {}).get(pid, {}).get("article") or "")


def _known(pid: str, vm: Dict[str, Any]) -> bool:
    maps = vm.get("maps") or {}
    return pid in (maps.get("periods") or {}) or pid in (maps.get("sentences") or {})


def outside_fitted(labels: Sequence[str], vm: Dict[str, Any]) -> List[str]:
    """The cited articles outside the fitted set and the user's named ones."""
    allowed = list(vm.get("fitted") or []) + list(vm.get("named") or [])
    return [x for x in labels if not any(same_article(x, y) for y in allowed)]


def render_ids(text: str, vm: Dict[str, Any]) -> str:
    """«[A1385cc.1]» -> «art. 1385 c.c.» (left out when the words before already name it), «[S3]» -> «fact»."""
    maps = vm.get("maps") or {}
    sentences = maps.get("sentences") or {}

    def sub(m: "re.Match[str]") -> str:
        pid = m.group(1)
        if pid.startswith("S"):
            s = sentences.get(pid)
            return f"«{_flat(s.get('text')).rstrip('.')}»" if s else m.group(0)
        label = _article_of_id(pid, vm)
        if not label:
            return m.group(0)
        before = text[max(0, m.start() - 60):m.start()]
        return "" if f"art. {label}" in before or f"artt. {label}" in before else f"art. {label}"
    out = _ID_RE.sub(sub, text)
    out = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"[ \t]{2,}", " ", out))
    return re.sub(r"\(\s*\)", "", out).strip()


def review_sentence(sentence: str, vm: Dict[str, Any], site: str, events: List[Dict[str, Any]],
                    excluded: Sequence[str] = ()) -> str:
    """The one rule on one sentence: dropped (""), with its event, when it cites an id that does not exist or an
    article outside the fitted set (an excluded one -- the binding's, a verified «no» -- logged as such); else the
    sentence with its ids rendered."""
    text = _flat(sentence)
    if not text:
        return sentence
    ids = _ID_RE.findall(text)
    unknown = [i for i in ids if not _known(i, vm)]
    if unknown:
        events.append({"sede": site, "esito": DROPPED, "testo": text[:200], "articoli": [], "motivo": UNKNOWN_ID,
                       "id": unknown[:6]})
        return ""
    cited = list(dict.fromkeys(_cited(text) + [_article_of_id(i, vm) for i in ids if i.startswith("A")]))
    scope = {"fitted": list(vm.get("fitted") or []), "named": list(vm.get("named") or []), "excluded": list(excluded or [])}
    cited = list(dict.fromkeys(cited + list(bare_cited(text, cited, scope))))     # «l'art. 1439» by number alone too
    gone = [x for x in cited if x and any(same_article(x, y) for y in excluded or [])]
    if gone:
        events.append({"sede": site, "esito": ARTICLE_EXCLUDED, "testo": text[:200], "articoli": gone})
        return ""
    loose = outside_fitted([x for x in cited if x], vm)
    if loose:
        events.append({"sede": site, "esito": ARTICLE_UNFITTED, "testo": text[:200], "articoli": loose})
        return ""
    missing = rule_without_id(text, cited, ids, vm)       # v79.18.1 (C9)
    if missing:
        events.append({"sede": site, "esito": DROPPED, "testo": text[:200], "articoli": missing, "motivo": RULE_WITHOUT_ID})
        return ""
    quoted = law_verbatim(text, ids, vm) if site in LAW_VERBATIM_SITES else ""     # v79.18.5 (item 2)
    if quoted:
        events.append({"sede": site, "esito": LAW_VERBATIM, "testo": text[:200],
                       "articoli": list(dict.fromkeys(_article_of_id(i, vm) for i in ids if i.startswith("A")))})
        return quoted
    return render_ids(sentence, vm)


def rule_without_id(text: str, cited: Sequence[str], ids: Sequence[str], vm: Dict[str, Any]) -> List[str]:
    """v79.18.1 (C9): the articles a sentence attributes a rule to (it names them and holds a rule verb --
    analysis_step's qa_statements verbs) without citing one of their period ids; [] when every named article has its
    A-id, or the sentence states no rule."""
    from . import analysis_step
    if not analysis_step._qa_verb_re().search(text or ""):
        return []
    with_id = {_article_of_id(i, vm) for i in ids if i.startswith("A")}
    return [x for x in cited if x and not any(same_article(x, y) for y in with_id if y)]


def law_verbatim(text: str, ids: Sequence[str], vm: Dict[str, Any]) -> str:
    """v79.18.5 (item 2): a sentence stating what a norm provides (a qa_statements verb) that cites period ids is not
    paraphrased: «art. X («<passage verbatim>»)» for each cited article, its cited periods' texts in order; "" when the
    sentence states no rule or cites no A-id."""
    from . import analysis_step
    a_ids = [i for i in ids if i.startswith("A") and _known(i, vm)]
    if not a_ids or not analysis_step._qa_verb_re().search(text or ""):
        return ""
    periods = (vm.get("maps") or {}).get("periods") or {}
    by_article: Dict[str, List[str]] = {}
    for pid in dict.fromkeys(a_ids):
        label = _article_of_id(pid, vm)
        passage = _flat((periods.get(pid) or {}).get("text"))
        if label and passage:
            by_article.setdefault(label, []).append(passage.rstrip("."))
    if not by_article:
        return ""
    return " ".join(f"art. {label} (" + "; ".join(f"«{x}»" for x in texts) + ")." for label, texts in by_article.items())


def review_text(text: str, vm: Dict[str, Any], site: str, events: List[Dict[str, Any]],
                excluded: Sequence[str] = ()) -> Tuple[str, int]:
    """A text sentence by sentence through review_sentence (table rows, headings and bold labels untouched): (the
    text, the sentences dropped)."""
    out_lines: List[str] = []
    dropped = 0
    for line in str(text or "").split("\n"):
        raw = line.strip()
        if not raw or raw.startswith("|") or raw.startswith("**") or raw.startswith("#"):
            out_lines.append(line)
            continue
        lead = re.match(r"^(\s*(?:[-•*]|\d{1,2}\.|\[\d{1,2}\])\s*)", line)
        prefix = lead.group(1) if lead else ""
        kept = []
        for sentence in sentences_of(line[len(prefix):]):
            done = review_sentence(sentence, vm, site, events, excluded)
            if done.strip():
                kept.append(done.strip())
            elif sentence.strip():
                dropped += 1
        if kept:
            out_lines.append(prefix + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip(), dropped


def review_sentences(text: str, vm: Dict[str, Any], site: str, events: List[Dict[str, Any]],
                     excluded: Sequence[str] = ()) -> Tuple[List[str], int]:
    """An answer's sentences through the one rule: (the kept sentences, rendered; the number dropped)."""
    kept, dropped = [], 0
    for sentence in sentences_of(" ".join(str(text or "").split())):
        done = review_sentence(sentence, vm, site, events, excluded)
        if done.strip():
            kept.append(done.strip())
        elif sentence.strip():
            dropped += 1
    return kept, dropped


def _tokens(text: str) -> set:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if len(w) > 2}


def fragment_or_copy(sentences: Sequence[str], vm: Dict[str, Any]) -> str:
    """Why an answer must not print: «frammento» (no sentence of FRAGMENT_WORDS words ending with a stop) or «copia»
    (the answer is an article's text: COPY_OVERLAP of its words in one period's); "" when it may."""
    full = [s for s in sentences if len(_WORD_RE.findall(s)) >= FRAGMENT_WORDS and s.rstrip()[-1:] in ".!?»"]
    if not full:
        return "frammento"
    mine = _tokens(" ".join(sentences))
    for p in ((vm.get("maps") or {}).get("periods") or {}).values():
        theirs = _tokens(p.get("text"))
        if mine and theirs and len(mine & theirs) / len(mine) >= COPY_OVERLAP:
            return "copia"
    return ""


def sites() -> Tuple[str, str, str]:
    return SITE_QA, SITE_NORMS, SITE_OBJECTIONS
