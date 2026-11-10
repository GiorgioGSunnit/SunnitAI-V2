"""v60g (e): the one check of the interactive answer, run last, against the record.

Before it, no step deletes: with a deferring ledger (answer_ledger, defer=True) every step's removals are put back and
recorded as proposals; the steps' rewrites and markers stand. This check reads every sentence of the answer -- the
procedural section (code-written) and the «Altri profili» list aside -- against the per-article record and the case,
and is the only one that removes; each removal goes into the ledger with its rule and reason (and the steps that had
proposed it), so the record says why every sentence the writer wrote is not in the answer.

Removed (each rule a reference, never a heuristic of its own):
  fuori_dal_record          the sentence cites an article neither in the record (the topic's planned articles) nor in
                            the computed material (terms, tools, the procedural section), nor named by the case
  esito_contraddetto        it says an article does not apply while its record's verdict is not «non applicabile», or
                            that it applies while the verdict is «non applicabile»
  norma_non_applicabile     (v60h 5) a §2 paragraph under an article whose record's verdict is «non applicabile»: the
                            article is one item of «Norme considerate e non applicabili», never a block
  norma_senza_testo         a §2 paragraph on an article whose text was not read (the record has no requisite)
  applicazione_non_chiesta  a §2 paragraph on an article the writer was not asked to apply, whose record has no fact for it
  virgolette_fuori_testo    a quotation, attributed to an article, its text and rubrica do not hold (rule_quote's measure)
  comma_inesistente         it cites a comma the article does not have
  contro_assistito          it proposes an action, a complaint or a claim against the client (the state's client)
  ripetizione               it repeats an earlier sentence of its section (repetition_guard's measure: four content
                            stems at least -- «Questo requisito è provato.» twice is two requisites)
  frammento                 a fragment: no subject (headless; a list's leading dash aside) or a cut citation's tail
                            (text_tidy.lead_fragment)
Marked, never removed:
  istituto_rubrica          it names an institution for an article («… (art. N)», «… ex art. N», «art. N (…)») whose
                            nouns share no stem with the article's rubrica, its text, or the names the record gives it
                            (its topic, the first read's institution): «(da verificare: art. N, rubrica «…»)»
  contraddizione_interna    it denies what the sentence before it in its paragraph asserts (readback's measure)
A proposal no rule confirms is rejected: the sentence stays (logged with the steps that proposed it).
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REMOVE_RULES = ("fuori_dal_record", "esito_contraddetto", "norma_non_applicabile", "norma_senza_testo",
                "applicazione_non_chiesta",
                "virgolette_fuori_testo", "comma_inesistente", "contro_assistito", "ripetizione", "frammento")
MARK_RULES = ("istituto_rubrica", "contraddizione_interna")
QUOTE_MIN = 25
_QUOTE_RE = re.compile(r"«([^»]{%d,})»" % QUOTE_MIN)
_AGAINST_RE = re.compile(r"\b(?:quereli?\w*|denunci\w*|agire|agisca|citare|convenire|convenuto in giudizio|"
                         r"chiedere (?:il )?risarciment\w*|domanda(?:re)? di risarcimento|azione)\b[^.;]{0,60}?"
                         r"\b(?:contro|nei confronti d\w*|a carico d\w*)\s+(?:il |la |lo |l['’]\s*|del |della )?"
                         r"(?:sig(?:\.|nor[ae]?)\s+|dott\.(?:ssa)?\s+)?", re.IGNORECASE)
# v60g (3): an institution named for an article -- the words just before «(art. N …)», «ex art. N», «ai sensi
# dell'art. N», «di cui all'art. N», or inside «art. N … (…)» after it
_BEFORE_RE = re.compile(r"(?:\(\s*|\b(?:ex|di cui all['’]|previst[oa] dall['’]|disciplinat[oa] dall['’])\s*)"
                        r"artt?\.\s*(?P<num>\d+(?:-[a-z]+)?)", re.IGNORECASE)
_AFTER_RE = re.compile(r"\bartt?\.\s*(?P<num>\d+(?:-[a-z]+)?)[^(),;.]{0,12}\(\s*(?P<inst>[a-zà-ü'’ ]{6,60})\)",
                       re.IGNORECASE)
_VERB_END_RE = re.compile(r"(?:are|ere|ire|ato|ata|ati|ate|ito|ita|iti|ite|uto|uta|ando|endo|ebbe|ono|isce|iscono)$")
INSTITUTION_WORDS = 3
_INSTITUTION_STOP = {"della", "delle", "dello", "degli", "dalla", "dalle", "nella", "nelle", "sulla", "sulle", "alla",
                     "alle", "questo", "questa", "quello", "quella", "norma", "articolo", "comma", "codice", "sensi",
                     "previsto", "prevista", "disciplinato", "disciplinata", "secondo", "anche", "inoltre", "quindi",
                     "pertanto", "infatti", "applica", "applicabile", "richiamo", "base", "materia", "tema", "ipotesi",
                     "caso", "fatti", "fatto", "parte", "cliente",
                     # v60g: the words of any application, never an institution's name
                     "fattispecie", "requisiti", "requisito", "sussistenza", "presupposti", "presupposto", "disciplina",
                     "applicazione", "termine", "termini", "rientra", "rientrano", "integra", "integrano", "prescrive",
                     "dispone", "prevede", "stabilisce", "effetti", "effetto", "conseguenze", "ambito", "profilo",
                     "profili", "verifica", "decorrenza", "natura", "condotta", "violazione", "rispetto", "tutela",
                     "azione", "domanda", "richiesta", "giudizio", "reato", "delitto", "norme", "regola", "principio"}


def _stems(text: str) -> Set[str]:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{5,}", str(text or "").lower()) if w not in _INSTITUTION_STOP}


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-zà-ü0-9]+", str(text or "").lower()))


def _label_of(ref: Sequence[str]) -> str:
    from . import citations
    return citations.label(ref)


# --- the rules ------------------------------------------------------------------------------------------------------

def _outside(cited: Sequence[Tuple[str, str]], allowed: Set[str]) -> List[str]:
    return [_label_of(r) for r in cited if _label_of(r) not in allowed]


def _verdict_contradicted(sentence: str, label: str, records: Dict[str, Dict[str, Any]]) -> str:
    """The reason when the sentence says of the article the opposite of its record's verdict; "" otherwise."""
    from . import issue_records, readback
    from .interactive_flow import declares_inapplicable
    rec = records.get(label)
    if not rec or not rec.get("requisites"):
        return ""
    verdict = rec.get("verdict") or issue_records.verdict(rec)
    if verdict != issue_records.NOT_APPLICABLE and declares_inapplicable(sentence):
        return f"dichiara non applicabile l'art. {label}, che il record dà «{verdict}»"
    if verdict == issue_records.NOT_APPLICABLE and readback._polarity(sentence) is True:
        return f"dice applicabile l'art. {label}, che il record dà non applicabile ({issue_records.verdict_reason(rec)})"
    return ""


def _quote_outside(sentence: str, labels: Sequence[str], texts: Dict[str, str], rubriche: Dict[str, str]) -> str:
    """A quotation (QUOTE_MIN characters or more) none of the sentence's articles' texts or rubriche holds, by
    rule_quote's own measure (rule_quote._holds: its words, or nine tenths of its content words); a quotation cut by
    an ellipsis is held piece by piece."""
    from .rule_quote import _holds
    own = [lab for lab in labels if texts.get(lab)]
    if not own or len(own) < len(set(labels)):
        return ""               # an article whose text was not read: nothing to hold the quotation against
    sentence = re.sub(r"\[DA VERIFICARE:[^\]]*\]|\(da verificare[^)]*\)", " ", sentence, flags=re.IGNORECASE)
    for quote in _QUOTE_RE.findall(sentence):
        pieces = [p for p in re.split(r"\s*(?:\[\s*(?:…|\.\.\.)\s*\]|\(\s*(?:…|\.\.\.)\s*\)|…|\.\.\.)\s*", quote)
                  if len(p.strip()) >= 12]
        ground = [f"{rubriche.get(lab, '')} {texts[lab]}" for lab in own]
        if pieces and not any(all(_holds(p, g) for p in pieces) for g in ground):
            return f"virgolette «{quote[:60]}…» che non sono nel testo dell'art. {own[0]}"
    return ""


def _phantom_comma(sentence: str, documents, commi: Dict[str, Set[int]]) -> str:
    from .interactive_flow import _ART_COMMA_RE, _COMMA_OF_RE
    from . import citations
    for pattern in (_ART_COMMA_RE, _COMMA_OF_RE):
        for m in pattern.finditer(sentence):
            number, comma = (m.group(1), m.group(2)) if pattern is _ART_COMMA_RE else (m.group(2), m.group(1))
            labels = [_label_of(r) for r in citations.refs_in(sentence[m.start():m.end() + 30], documents)
                      if str(r[0]) == str(number)]
            for lab in labels:
                have = commi.get(lab)
                if have and int(comma) not in have:
                    return f"cita il comma {comma} dell'art. {lab}, che ha i commi {sorted(have)}"
    return ""


_ACT_ON_RE = re.compile(r"\b(?:querelar\w*|denunciar\w*|citar\w*|convenir\w*|diffidar\w*|chiamar\w* in giudizio)\s+"
                        r"(?:anche\s+)?(?:il |la |lo |l['’]\s*)?(?:sig(?:\.|nor[ae]?)\s+|dott\.(?:ssa)?\s+)?",
                        re.IGNORECASE)


def _against_client(sentence: str, client: str) -> str:
    if not client:
        return ""
    names = [w for w in re.findall(r"[A-ZÀ-Ü][\w'’]{2,}", client)] or [client]
    for pattern in (_AGAINST_RE, _ACT_ON_RE):
        for m in pattern.finditer(sentence):
            rest = sentence[m.end():m.end() + 60]
            if any(re.match(rf"\s*{re.escape(n)}\b", rest) for n in names):
                return f"propone un'azione contro la parte assistita ({client})"
    return ""


def _nouns(words: Sequence[str]) -> List[str]:
    """The content nouns among words: five letters or more, no function word, no verb form."""
    return [w for w in words if len(w) >= 5 and w.lower() not in _INSTITUTION_STOP and not _VERB_END_RE.search(w.lower())]


def _institution(sentence: str, documents, rubriche: Dict[str, str], texts: Dict[str, str],
                 names: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, str]]:
    """(label, institution) when the sentence names an institution for an article -- the nouns among the
    INSTITUTION_WORDS words before its citation, or in the parenthesis after it -- sharing no stem with the article's
    rubrica, its text, or the names the record gives it (names: its topic's title, the first read's institution that
    found it)."""
    from . import citations
    found = []
    for m in _BEFORE_RE.finditer(sentence):
        head = re.sub(r"\b(?:ex|di cui all['’]|previst[oa] dall['’]|disciplinat[oa] dall['’])\s*$", "",
                      sentence[:m.start()], flags=re.IGNORECASE)
        words = re.findall(r"[a-zà-ü'’]+", head.lower())[-INSTITUTION_WORDS:]
        found.append((m, _nouns(words)))
    for m in _AFTER_RE.finditer(sentence):
        found.append((m, _nouns(re.findall(r"[a-zà-ü'’]+", m.group("inst").lower()))))
    for m, nouns in found:
        mine = _stems4(" ".join(nouns))
        if not mine:
            continue
        labels = [_label_of(r) for r in citations.refs_in(sentence[m.start():m.end() + 30], documents)
                  if str(r[0]) == m.group("num")]
        for lab in labels:
            if not (rubriche.get(lab) and texts.get(lab)):
                continue                # without the article's own words, nothing to check against
            ground = _stems4(rubriche[lab]) | _stems4(texts[lab]) | _stems4((names or {}).get(lab, ""))
            if not mine & ground:
                return lab, " ".join(nouns)
    return None


def _stems4(text: str) -> Set[str]:
    """Four-letter stems of the words of four letters or more («vizio» and «vizi» one)."""
    return {w[:4] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _INSTITUTION_STOP}


def _repeats(sentence: str, earlier: List[Tuple[set, set, set, str]]) -> str:
    from . import repetition_guard as rg
    mine = rg.stems(sentence)
    if len(mine) < rg.MIN_STEMS:
        return ""
    refs, numbers = rg._refs(sentence), rg._numbers(sentence)
    for other, other_refs, other_numbers, said in earlier:
        if len(mine & other) >= rg.OVERLAP * len(mine) and refs <= other_refs and numbers <= other_numbers:
            return f"ripete «{said[:80]}»"
    earlier.append((mine, refs, numbers, sentence))
    return ""


def _fragment(sentence: str) -> str:
    from . import text_tidy
    from .interactive_flow import _headless
    sentence = re.sub(r"^\s*[–—-]\s+(?=[A-ZÀ-Ü«(\"“])", "", sentence)       # «— Il venditore …»: a list's dash
    if _headless(sentence):
        return "frammento senza soggetto"
    if not text_tidy.lead_fragment(sentence.strip()):
        return "frammento di una citazione tagliata"
    return ""


def _denies_previous(sentence: str, previous: str) -> bool:
    from . import readback
    if not previous or readback._CONDITIONAL_RE.search(sentence) or readback._CONDITIONAL_RE.search(previous):
        return False
    a, b = readback._claim_stems(previous), readback._claim_stems(sentence)
    if len(a) < 4 or len(b) < 4:
        return False
    if bool(readback._NEGATION_RE.search(previous)) == bool(readback._NEGATION_RE.search(sentence)):
        return False
    return len(a & b) / max(1, len(a | b)) >= readback.CLAIM_OVERLAP


def _mark(sentence: str, mark: str) -> str:
    body = sentence.rstrip()
    stop = re.search(r"[.!?]+$", body)
    if not stop:
        return body + mark + sentence[len(body):]
    return body[:stop.start()] + mark + stop.group(0) + sentence[len(body):]


# --- the check ------------------------------------------------------------------------------------------------------

def check(sections: List[Tuple[str, str]], ctx: Dict[str, Any], ledger=None) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """(sections, log). ctx: records, allowed (labels), glossed (labels), texts / rubriche / names ({label: …}), commi
    ({label: {numbers}}), documents, client, skip (titles), norms_title, numbering (the writer's «[N]» refs)."""
    from . import answer_ledger, citations
    from .plan_then_write import _sentence_spans
    records = ctx.get("records") or {}
    allowed = set(ctx.get("allowed") or ())
    glossed = set(ctx.get("glossed") or ())
    texts, rubriche, commi = ctx.get("texts") or {}, ctx.get("rubriche") or {}, ctx.get("commi") or {}
    documents = ctx.get("documents") or []
    skip = set(ctx.get("skip") or ())
    log: Dict[str, Any] = {"removed": [], "marked": [], "rejected": [], "rules": {}}
    if ledger is not None:
        ledger.steps.append(answer_ledger.CHECK_STEP)
    out = []
    removed_norms: Set[str] = set()
    for title, text in sections:
        if title in skip or not (text or "").strip():
            out.append((title, text))
            continue
        units = answer_ledger.tracked(text, ctx.get("numbering") or (), documents)
        cuts: List[Tuple[int, int]] = []
        marks: Dict[Tuple[int, int], str] = {}
        earlier: List[Tuple[set, set, set, str]] = []
        previous = ""
        for u in units:
            s, e = u["start"], u["end"]
            sentence = text[s:e]
            cited = citations.refs_in(sentence, documents)
            labels = [_label_of(r) for r in cited] or ([u["heading"]] if u["heading"] else [])
            in_norms = title == ctx.get("norms_title") and u["heading"]
            rule, why = "", ""
            outside = _outside(cited, allowed)
            if outside:
                rule, why = ("fuori_dal_record", f"cita art. {', '.join(outside)}: non è tra le norme degli argomenti scelti "
                             "né nel materiale calcolato")
            if not rule:
                for lab in list(dict.fromkeys(labels))[:3]:
                    why = _verdict_contradicted(sentence, lab, records)
                    if why:
                        rule = "esito_contraddetto"
                        break
            if not rule and in_norms:
                from . import issue_records
                rec = records.get(u["heading"])
                if rec is not None and (rec.get("verdict") or issue_records.verdict(rec)) == issue_records.NOT_APPLICABLE:
                    rule, why = ("norma_non_applicabile", f"paragrafo sull'art. {u['heading']}: esito del record «non "
                                 "applicabile», una riga in «Norme considerate e non applicabili»")
                elif rec is not None and not rec.get("requisites") and not texts.get(u["heading"]):
                    rule, why = "norma_senza_testo", f"paragrafo sull'art. {u['heading']}, il cui testo non è stato letto"
                elif rec is not None and glossed and u["heading"] not in glossed:
                    from . import issue_records
                    if not issue_records.support(rec):
                        rule, why = ("applicazione_non_chiesta", f"paragrafo sull'art. {u['heading']}: applicazione "
                                     "non chiesta e nessun fatto a sostegno nel record")
            if not rule:
                why = _quote_outside(sentence, labels, texts, rubriche)
                rule = "virgolette_fuori_testo" if why else ""
            if not rule:
                why = _phantom_comma(sentence, documents, commi)
                rule = "comma_inesistente" if why else ""
            if not rule:
                why = _against_client(sentence, ctx.get("client") or "")
                rule = "contro_assistito" if why else ""
            if not rule:
                why = _repeats(sentence, earlier)
                rule = "ripetizione" if why else ""
            if not rule:
                why = _fragment(sentence)
                rule = "frammento" if why else ""
            proposers = ledger.proposed_by(u["norm"]) if ledger is not None else []
            if rule:
                cuts.append((s, e))
                removed_norms.add(u["norm"])
                log["rules"][rule] = log["rules"].get(rule, 0) + 1
                log["removed"].append({"section": title, "rule": rule, "reason": why, "sentence": u["text"][:200],
                                       "proposed_by": proposers})
                if ledger is not None:
                    ledger.remove(why, title, sentence, u["articles"], rule, proposers)
                continue
            mark = ""
            hit = _institution(sentence, documents, rubriche, texts, ctx.get("names"))
            if hit:
                mark = f" (da verificare: art. {hit[0]}, rubrica «{rubriche.get(hit[0], '')[:80]}»)"
                log["marked"].append({"section": title, "rule": "istituto_rubrica", "article": hit[0],
                                      "institution": hit[1], "sentence": u["text"][:200]})
            elif _denies_previous(sentence, previous):
                mark = " (da verificare: contraddice la frase precedente)"
                log["marked"].append({"section": title, "rule": "contraddizione_interna", "sentence": u["text"][:200]})
            if mark and "(da verificare" not in sentence:
                marks[(s, e)] = _mark(sentence, mark)
                log["rules"][log["marked"][-1]["rule"]] = log["rules"].get(log["marked"][-1]["rule"], 0) + 1
            if proposers:
                log["rejected"].append({"section": title, "proposed_by": proposers, "sentence": u["text"][:200]})
            previous = sentence if "\n" not in text[e:e + 2] else ""
        lead, trail = text[:len(text) - len(text.lstrip("\n"))], text[len(text.rstrip("\n")):]
        for s, e in sorted(set(cuts) | set(marks), reverse=True):
            text = text[:s] + marks[(s, e)] + text[e:] if (s, e) in marks else text[:s] + text[e:]
        if cuts:
            text = lead + _tidy(text) + trail
        out.append((title, text))
    return out, log


def _tidy(text: str) -> str:
    """After the cuts: no double spaces, no line left with a mark or punctuation alone, no triple breaks."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\s*(?:\[\d{1,2}\])?\s*[.;:,]?\s*", ln) or not ln.strip()]
    text = "\n".join(lines)
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.;:,])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")
