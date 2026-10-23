"""PLAN_THEN_WRITE_ENABLED: the parere as a plan, then its sections written from the plan.

Flavio's handover of 22 Oct (6.3, "the main lever"): qwen2.5-32b is reliable on short focused
prompts, not on one long one. The lawyer's review of the held-out drafts found the wrong side
argued, the central question missed or conceded, rulings without court, section and number,
and no practical content. The PM's test cases of 23 Oct showed the same: the pupil's case was
argued for the school although the stance check had named the parents as the client, and the
burglary case's procedural question got "processo ordinario".

  1. The plan, one call at temperature 0, as JSON: the client and their goal, the central
     question in one sentence, an answer to each of the user's questions, and per issue the
     governing articles (from NORME, by their text), the rulings (from GIURISPRUDENZA, by full
     reference), the client's argument, the strongest opposing one and the rebuttal; then the
     subordinate requests and the weak points. Articles and rulings not among those given are
     removed in code. If the stance check (defensive_generation._opposing_arguments) finds an
     argument for the other side, the plan is made once more with that list.
  2. The practical checks of the case's posture and area (practical_checklist.py).
  3. Six sections written in parallel, each call seeing only its own material.

Any failure (no plan, a plan without an argued issue, a section failing twice) returns None:
the pipeline then writes the parere in one call, as before.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import defensive_generation as dg
from . import method_candidates, practical_checklist

logger = logging.getLogger(__name__)

_PLAN_TOKENS = 2000
_MAX_ISSUES = 5

_PLAN_SYSTEM = (
    "Sei un avvocato italiano con lunga esperienza di contenzioso. Prima di scrivere un parere ne prepari il "
    "PIANO, nell'interesse della parte assistita. Ricevi il caso, i quesiti, l'analisi dei fatti, le questioni "
    "giuridiche, le NORME (testi dalla banca dati, ciascuna con il suo identificativo tra parentesi quadre) e "
    "la GIURISPRUDENZA (passi di pronunce, ciascuna con il suo riferimento tra parentesi quadre).\n\n"
    "REGOLE:\n"
    "- Tutto il piano è nell'interesse della parte assistita: chi chiede il parere o la strategia, o il "
    "cliente da difendere.\n"
    "- La questione centrale è la domanda da cui dipende l'esito per la parte assistita, in una frase: va "
    "risolta per prima e mai concessa.\n"
    "- Per ogni questione scegli le norme che la disciplinano leggendone il testo in NORME e indicale con "
    "l'identificativo esatto (\"art. 1384 c.c.\"); mai norme che non sono in NORME, mai a memoria.\n"
    "- Pronunce: solo quelle di GIURISPRUDENZA, con il riferimento esatto tra parentesi quadre, e solo se il "
    "passo riguarda davvero la questione; altrimenti lista vuota. I fatti di una pronuncia non sono fatti "
    "del caso.\n"
    "- L'argomento di ogni questione deve escludere, ridurre o contrastare la responsabilità o la pretesa "
    "contro la parte assistita, o sostenerne la pretesa. Un fatto che favorisce la controparte è "
    "un'obiezione da superare, non un argomento.\n"
    "- Considera le tesi alternative o subordinate quando i fatti lo consentono: una qualificazione diversa e "
    "meno grave, una riduzione, un rito o un rimedio più favorevole.\n"
    "- Non aggiungere fatti che il caso non contiene.\n\n"
    "Rispondi SOLO con un oggetto JSON:\n"
    '{"parte_assistita": "...", "posizione": "...", "obiettivo": "che cosa vuole ottenere", '
    '"questione_centrale": "una frase", '
    '"risposte": [{"quesito": "...", "risposta": "risposta diretta in una o due frasi"}], '
    '"questioni": [{"titolo": "...", "norme": ["art. N codice"], "pronunce": ["riferimento"], '
    '"argomento": "...", "obiezione": "...", "replica": "..."}], '
    '"subordinate": ["..."], "punti_deboli": ["..."]}\n'
    "Le questioni, da due a cinque, in ordine di forza per la parte assistita."
)

SECTION_TITLES = ("Risposte ai quesiti", "Inquadramento e questione centrale", "Norme e giurisprudenza",
                  "Applicazione al caso", "Profili processuali e termini", "Conclusioni")


# --- 1. the plan -------------------------------------------------------------------------------

def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _texts(value: Any) -> List[str]:
    return [t for t in (_text(v) for v in (value if isinstance(value, list) else [])) if t]


def _matching_reference(text: str, references: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """The reference among `references` a plan's ruling names, by the reference itself or its
    decision number and year."""
    low = text.lower()
    numbers = dg._decision_numbers(text)
    for reference in references:
        if reference.lower() in low or numbers & dg._decision_numbers(reference):
            return reference
    return None


def clean_plan(raw: Any, known: set, references: Dict[str, List[Dict[str, Any]]]) -> Tuple[Optional[dict], dict]:
    """(plan, removed): the model's plan with only articles among `known` and rulings among
    `references`; issues without an argument dropped. None when no argued issue is left."""
    removed: Dict[str, List[str]] = {"norme": [], "pronunce": []}
    if not isinstance(raw, dict):
        return None, removed
    issues = []
    for q in raw.get("questioni") if isinstance(raw.get("questioni"), list) else []:
        if not isinstance(q, dict) or not _text(q.get("argomento")):
            continue
        norme: List[tuple] = []
        for item in _texts(q.get("norme")):
            refs = dg._refs_in(item)
            if not refs:
                removed["norme"].append(item)
            for ref in refs:
                if ref not in known:
                    removed["norme"].append(f"{ref[0]} {ref[1]}")
                elif ref not in norme:
                    norme.append(ref)
        pronunce: List[str] = []
        for item in _texts(q.get("pronunce")):
            reference = _matching_reference(item, references)
            if not reference:
                removed["pronunce"].append(item)
            elif reference not in pronunce:
                pronunce.append(reference)
        issues.append({"titolo": _text(q.get("titolo")) or f"Questione {len(issues) + 1}", "norme": norme,
                       "pronunce": pronunce, "argomento": _text(q.get("argomento")),
                       "obiezione": _text(q.get("obiezione")), "replica": _text(q.get("replica"))})
    if not issues:
        return None, removed
    answers = [{"quesito": _text(a.get("quesito")), "risposta": _text(a.get("risposta"))}
               for a in (raw.get("risposte") if isinstance(raw.get("risposte"), list) else [])
               if isinstance(a, dict) and _text(a.get("risposta"))]
    return {"parte_assistita": _text(raw.get("parte_assistita")), "posizione": _text(raw.get("posizione")),
            "obiettivo": _text(raw.get("obiettivo")), "questione_centrale": _text(raw.get("questione_centrale")),
            "risposte": answers, "questioni": issues[:_MAX_ISSUES],
            "subordinate": _texts(raw.get("subordinate")), "punti_deboli": _texts(raw.get("punti_deboli"))}, removed


def _plan_human(facts: str, request: str, analysis: Dict[str, Any], articles: str, rulings: str,
                principles: str) -> str:
    base = dg._parere_human(facts, analysis, articles, rulings, request, principles)
    base = base[:base.rfind("Redigi ora il parere.")].rstrip() if "Redigi ora il parere." in base else base
    return f"PARTE ASSISTITA (dall'analisi del caso): {dg._stance_party(analysis)}\n\n{base}\n\nPrepara ora il piano."


def _arguments_text(plan: Dict[str, Any]) -> str:
    """The plan's arguments as the stance check reads a section 3."""
    return "3. Argomenti\n" + "\n".join(f"#### {q['titolo']}\n{q['argomento']}" for q in plan["questioni"])


def make_plan(facts: str, request: str, analysis: Dict[str, Any], articles: str, rulings: str, principles: str,
              known: set, references: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The cleaned plan, made once more if its arguments favour the other side; None on failure."""
    human = _plan_human(facts, request, analysis, articles, rulings, principles)
    messages = [SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=human)]
    t0 = time.time()
    try:
        raw = dg._chat_json(messages, max_tokens=_PLAN_TOKENS, temperature=0)
        plan, removed = clean_plan(raw, known, references)
        opposing = dg._opposing_arguments(facts, request, analysis, _arguments_text(plan)) if plan else []
        if opposing:
            listing = "\n".join(f"- {o['argomento']}: {o['perche']}" for o in opposing)
            retry = (f"{human}\n\nIl tuo primo piano:\n{json.dumps(raw, ensure_ascii=False)}\n\nUna revisione ha "
                     f"trovato che questi argomenti giovano alla controparte, non a {dg._stance_party(analysis)}:\n"
                     f"{listing}\n\nRifai il piano: ogni argomento deve escludere, ridurre o contrastare la "
                     "responsabilità o la pretesa contro la parte assistita, o sostenerne la pretesa; un fatto "
                     "che favorisce la controparte va trattato come obiezione da superare.")
            raw = dg._chat_json([SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=retry)],
                                max_tokens=_PLAN_TOKENS, temperature=0)
            plan, removed = clean_plan(raw, known, references)
    except Exception as exc:
        logger.warning("plan-then-write: no plan, the parere is written in one call: %s", exc)
        return None
    finally:
        dg.vlog("case_analysis_plan_llm", duration_ms=(time.time() - t0) * 1000)
    dg.vlog("case_analysis_plan", {
        "issues": len(plan["questioni"]) if plan else 0, "replanned": bool(opposing),
        "opposing": [o["argomento"] for o in opposing],
        "norme": sorted({f"{n} {c}" for q in (plan or {}).get("questioni", []) for n, c in q["norme"]}),
        "pronunce": sorted({p for q in (plan or {}).get("questioni", []) for p in q["pronunce"]}),
        "removed": removed, "references_available": len(references)})
    if plan is None:
        logger.warning("plan-then-write: the plan has no argued issue, the parere is written in one call")
        return None
    if not plan["parte_assistita"]:
        plan["parte_assistita"] = str(analysis.get("parte_assistita") or "").strip()
    return plan


# --- 3. the sections -------------------------------------------------------------------------

def _plan_text(plan: Dict[str, Any], answers: bool = False, issues: str = "titles", subordinate: bool = False,
               weak: bool = False) -> str:
    """The plan as the writer reads it; `issues` is "titles", "full" or "" (none)."""
    party = plan["parte_assistita"] or "la parte assistita"
    lines = [f"Parte assistita: {party}" + (f" ({plan['posizione']})" if plan["posizione"] else "")]
    if plan["obiettivo"]:
        lines.append(f"Obiettivo: {plan['obiettivo']}")
    if plan["questione_centrale"]:
        lines.append(f"Questione centrale: {plan['questione_centrale']}")
    if answers and plan["risposte"]:
        lines.append("Risposte ai quesiti:")
        lines += [f"- {a['quesito']} -> {a['risposta']}" if a["quesito"] else f"- {a['risposta']}"
                  for a in plan["risposte"]]
    if issues:
        lines.append("Questioni, in ordine di forza:")
        for i, q in enumerate(plan["questioni"], 1):
            norms = ", ".join(f"art. {n} {c}" for n, c in q["norme"]) or "-"
            lines.append(f"{i}. {q['titolo']} (norme: {norms}; pronunce: {', '.join(q['pronunce']) or '-'})")
            if issues == "full":
                lines += [f"   Argomento: {q['argomento']}"]
                lines += [f"   Obiezione della controparte: {q['obiezione']}"] if q["obiezione"] else []
                lines += [f"   Replica: {q['replica']}"] if q["replica"] else []
            else:
                lines.append(f"   {q['argomento']}")
    if subordinate and plan["subordinate"]:
        lines += ["Tesi o domande subordinate:"] + [f"- {s}" for s in plan["subordinate"]]
    if weak and plan["punti_deboli"]:
        lines += ["Punti deboli:"] + [f"- {s}" for s in plan["punti_deboli"]]
    return "PIANO:\n" + "\n".join(lines)


def _plan_refs(plan: Dict[str, Any]) -> List[tuple]:
    return list(dict.fromkeys(ref for q in plan["questioni"] for ref in q["norme"]))


def _article_texts(article_rows: List[Dict[str, Any]], refs: List[tuple], chars: int) -> str:
    rows = sorted((r for r in article_rows if r.get("_article") in refs), key=lambda r: refs.index(r["_article"]))
    return dg._source_block(rows, chars)


def _ruling_texts(plan: Dict[str, Any], references: Dict[str, List[Dict[str, Any]]], chars: int) -> str:
    cited = list(dict.fromkeys(p for q in plan["questioni"] for p in q["pronunce"]))
    return dg._source_block([references[p][0] for p in cited if references.get(p)], chars)


def _section_system(plan: Dict[str, Any], lang: str) -> str:
    from .language import language_display_name
    party = plan["parte_assistita"] or "la parte assistita"
    position = f" ({plan['posizione']})" if plan["posizione"] else ""
    return (
        "Sei un avvocato italiano con lunga esperienza di contenzioso. Scrivi UNA sezione di un parere motivato "
        f"nell'interesse di {party}{position}, seguendo il PIANO già deciso: stessa parte, stessa questione "
        "centrale, stessi argomenti, stesse norme e pronunce.\n\n"
        "REGOLE:\n"
        "- Scrivi solo il testo della sezione, senza il suo titolo e senza anticipare le altre sezioni.\n"
        "- Non concedere mai un elemento contestato dell'accusa o della pretesa: una posizione che lo presuppone "
        "va presentata solo in via subordinata.\n"
        "- Norme: cita solo quelle del PIANO o dei TESTI forniti, sempre con il codice (\"art. 1384 c.c.\", mai "
        "\"art. 1384\" da solo); riporta tra virgolette solo frasi presenti nei testi forniti.\n"
        "- Pronunce: cita solo quelle del PIANO, con il riferimento esatto indicato; senza riferimento non "
        "richiamare orientamenti giurisprudenziali. I fatti di una pronuncia non sono fatti del caso.\n"
        "- Non aggiungere fatti, date, importi o atti che il caso non contiene: se un dato manca, indica che cosa "
        "va verificato.\n"
        "- Prosa argomentata in paragrafi; elenchi puntati solo per prove da acquisire e attività da compiere.\n"
        f"- Scrivi in {language_display_name(lang)}."
    )


def _facts_list(analysis: Dict[str, Any]) -> str:
    return "\n".join(f"- {f.get('fatto', '')}" for f in analysis.get("fatti", []) if f.get("fatto"))


def section_jobs(plan: Dict[str, Any], facts: str, request: str, analysis: Dict[str, Any],
                 article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
                 principles: str) -> List[Tuple[str, str, str, int]]:
    """(title, instruction, material, max_tokens) for each section, in order."""
    party = plan["parte_assistita"] or "la parte assistita"
    case = f"CASO:\n{dg._case_text(facts, request)}"
    frame = analysis.get("frame") or {}
    tasks = "\n".join(f"- {t}" for t in frame.get("tasks") or []) or request.strip()
    facts_list = _facts_list(analysis)
    refs = _plan_refs(plan)
    # Article text per section within a fixed total, so that a plan naming many articles still
    # leaves each call well inside the context with the case text (up to ~5,000 tokens).
    def per_article(total_chars: int, most: int) -> int:
        return max(300, min(most, total_chars // max(1, len(refs))))
    areas = method_candidates.areas_for(frame.get("areas") or [], analysis.get("area") or "")
    posture = frame.get("posture_type") or ""
    checks = practical_checklist.checks_for(areas, posture)
    procedural = [ref for ref in method_candidates.candidate_refs([], posture) if ref not in refs]

    def block(title: str, body: str) -> str:
        return f"{title}:\n{body}" if body.strip() else ""

    def material(*parts: str) -> str:
        return "\n\n".join(p for p in parts if p)

    return [
        (SECTION_TITLES[0],
         "SEZIONE: Risposte ai quesiti. Una risposta diretta e motivata a ciascun quesito, nell'ordine in cui è "
         "posto, a partire dalle risposte del PIANO; se la questione centrale non è tra i quesiti, rispondi prima "
         "a essa. Ogni risposta in due-quattro frasi, con la norma su cui si fonda.",
         material(case, block("QUESITI", tasks), _plan_text(plan, answers=True)), 800),
        (SECTION_TITLES[1],
         "SEZIONE: Inquadramento e questione centrale. La qualificazione giuridica della vicenda (la norma "
         "principale e quelle alternative o subordinate, e perché), la questione centrale e perché decide "
         f"l'esito; poi ogni fatto del caso valutato dal punto di vista di {party}: favorevole, sfavorevole o "
         f"neutro per {party}, e perché. Un fatto può essere insieme favorevole e sfavorevole: dillo.",
         material(case, block("FATTI DEL CASO", facts_list), _plan_text(plan),
                  block("TESTI DELLE NORME", _article_texts(article_rows, refs, per_article(8000, 700)))), 1300),
        (SECTION_TITLES[2],
         "SEZIONE: Norme e giurisprudenza. Per ogni norma del PIANO: che cosa stabilisce, citando il testo "
         "fornito, e come si ripartisce l'onere della prova; per ogni pronuncia del PIANO: il principio che "
         "afferma, con il suo riferimento, e che cosa comporta per questo caso. Solo norme e pronunce del PIANO.",
         material(case, _plan_text(plan),
                  block("TESTI DELLE NORME", _article_texts(article_rows, refs, per_article(15000, dg._ARTICLE_CHARS))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 1200)),
                  block("PRINCIPI DALLA GIURISPRUDENZA", principles)), 1500),
        (SECTION_TITLES[3],
         "SEZIONE: Applicazione al caso. Per ogni questione del PIANO, nell'ordine, con un sottotitolo: "
         f"l'argomento di {party} applicato ai fatti del caso, con le norme e le pronunce del piano; le prove da "
         "acquisire; l'obiezione più forte della controparte e la replica. Poi le tesi subordinate del PIANO. "
         "Prosa, senza ripetere per ogni questione lo stesso schema a voci.",
         material(case, block("FATTI DEL CASO", facts_list), _plan_text(plan, issues="full", subordinate=True),
                  block("TESTI DELLE NORME", _article_texts(article_rows, refs, per_article(9000, 600))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 600))), 2200),
        (SECTION_TITLES[4],
         "SEZIONE: Profili processuali e termini. La fase in cui si trova la vicenda e che cosa consente di fare "
         f"a {party}: gli atti da compiere e i loro termini, con l'articolo che li prevede; a chi rivolgersi o chi "
         "convenire; le voci di danno o le domande da proporre, anche in via subordinata. Usa le VERIFICHE solo "
         "se i fatti del caso le rendono pertinenti, dicendo perché; un termine che dipende da una data che il "
         "caso non indica è una verifica da fare.",
         material(case, block("FASE", frame.get("procedural_posture") or ""),
                  _plan_text(plan, issues="", subordinate=True),
                  block("VERIFICHE PER QUESTA FASE E MATERIA", "\n".join(f"- {c}" for c in checks)),
                  block("TESTI DELLE NORME PROCESSUALI", _article_texts(article_rows, procedural, 500))), 1200),
        (SECTION_TITLES[5],
         "SEZIONE: Conclusioni. Gli argomenti in ordine di forza, come nel PIANO; una stima motivata del rischio, "
         f"tenendo conto dei punti deboli; la condotta consigliata a {party}.",
         _plan_text(plan, subordinate=True, weak=True), 800),
    ]


def _without_heading(text: str, title: str) -> str:
    """The section's text without a heading line the model repeated anyway."""
    text = (text or "").strip()
    first, _, rest = text.partition("\n")
    bare = re.sub(r"^[#>*\s]*(?:\d+\.\s*)?|[*#:\s]+$", "", first)
    return rest.strip() if bare.lower() == title.lower() else text


def write_sections(plan: Dict[str, Any], jobs: List[Tuple[str, str, str, int]], lang: str) -> Optional[List[tuple]]:
    """[(title, text)] for the jobs, written in parallel; a failed section is tried once more.
    None if one still fails or comes back empty."""
    system = _section_system(plan, lang)

    def write(job):
        title, instruction, material, max_tokens = job
        messages = [SystemMessage(content=f"{system}\n\n{instruction}"), HumanMessage(content=material)]
        try:
            return dg._call_chat(messages, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("plan-then-write: section %r failed, trying once more: %s", title, exc)
            return dg._call_chat(messages, max_tokens=max_tokens)

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            texts = list(pool.map(write, jobs))
    except Exception as exc:
        logger.warning("plan-then-write: a section failed twice, the parere is written in one call: %s", exc)
        return None
    finally:
        dg.vlog("case_analysis_sections_llm", duration_ms=(time.time() - t0) * 1000)
    sections = [(job[0], _without_heading(text, job[0])) for job, text in zip(jobs, texts)]
    if any(not text for _, text in sections):
        logger.warning("plan-then-write: an empty section, the parere is written in one call")
        return None
    return sections


def write_parere(facts: str, request: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                 ruling_rows: List[Dict[str, Any]], articles: str, rulings: str, principles: str,
                 lang: str = "it") -> Optional[str]:
    """The parere from a plan, section by section, or None (the caller writes it in one call).
    `articles` and `rulings` are the draft prompt's own blocks (_fit_parere_prompt)."""
    references: Dict[str, List[Dict[str, Any]]] = {}
    for row in ruling_rows:
        reference = dg._ruling_reference(row)
        if reference:
            references.setdefault(reference, []).append(row)
    known = {r["_article"] for r in article_rows if r.get("_article")}
    plan = make_plan(facts, request, analysis, articles, rulings, principles, known, references)
    if plan is None:
        return None
    sections = write_sections(plan, section_jobs(plan, facts, request, analysis, article_rows, references,
                                                 principles), lang)
    if sections is None:
        return None
    return "\n\n".join(f"**{i}. {title}**\n\n{text}" for i, (title, text) in enumerate(sections, 1))
