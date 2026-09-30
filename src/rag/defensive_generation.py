"""
Defensive legal document generation pipeline.
Given an uploaded judicial document, this module:
1. Classifies the proceeding type
2. Extracts opposing claims and cited articles
3. Queries Neo4j for counter-arguments and favorable jurisprudence
4. Generates a structured defensive draft
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import _call_chat

logger = logging.getLogger(__name__)

_PROCEEDING_TYPES = {"civile", "penale", "amministrativo"}
_PROCEEDING_SUBTYPES = {
    "atto_citazione", "decreto_ingiuntivo", "ricorso", "appello", "altro",
}


def _safe_defaults() -> Dict[str, Any]:
    return {
        "proceeding_type": "civile",
        "proceeding_subtype": "altro",
        "court": "",
        "opposing_party": "",
        "opposing_claims": "",
        "cited_articles": [],
        "key_facts": "",
    }


def classify_proceeding(text: str, lang: str = "it") -> dict:
    """Classify an uploaded judicial document's proceeding type via one LLM call.

    Best-effort: any parsing or LLM failure returns _safe_defaults() rather
    than raising, since the caller (run_defensive_pipeline) must still be
    able to produce a draft with generic section headings.
    """
    system = (
        "Sei un esperto di diritto italiano. Analizza il seguente documento "
        "giudiziario e restituisci un oggetto JSON con queste chiavi esatte:\n"
        "- proceeding_type: \"civile\", \"penale\", o \"amministrativo\"\n"
        "- proceeding_subtype: \"atto_citazione\", \"decreto_ingiuntivo\", \"ricorso\", "
        "\"appello\", o \"altro\"\n"
        "- court: nome del tribunale (stringa vuota se non trovato)\n"
        "- opposing_party: nome della parte avversa (stringa vuota se non trovato)\n"
        "- opposing_claims: riassunto delle pretese della controparte (max 300 caratteri)\n"
        "- cited_articles: lista degli articoli di legge citati dalla controparte\n"
        "- key_facts: fatti chiave allegati dalla controparte (max 300 caratteri)\n"
        "Restituisci SOLO il JSON, niente altro."
    )
    try:
        raw = _call_chat(
            [SystemMessage(content=system), HumanMessage(content=text)],
            max_tokens=600,
        )
    except Exception as exc:
        logger.warning("classify_proceeding: LLM call failed: %s", exc)
        return _safe_defaults()

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        logger.warning("classify_proceeding: JSON parse failed: %s", exc)
        return _safe_defaults()

    if not isinstance(parsed, dict):
        return _safe_defaults()

    result = _safe_defaults()
    proceeding_type = parsed.get("proceeding_type")
    if proceeding_type in _PROCEEDING_TYPES:
        result["proceeding_type"] = proceeding_type

    proceeding_subtype = parsed.get("proceeding_subtype")
    if proceeding_subtype in _PROCEEDING_SUBTYPES:
        result["proceeding_subtype"] = proceeding_subtype

    for key in ("court", "opposing_party", "opposing_claims", "key_facts"):
        value = parsed.get(key)
        if isinstance(value, str):
            result[key] = value

    cited_articles = parsed.get("cited_articles")
    if isinstance(cited_articles, list):
        result["cited_articles"] = [a for a in cited_articles if isinstance(a, str)]

    return result


_STRUCTURES = {
    ("civile", "atto_citazione"): """
Redigi una COMPARSA DI RISPOSTA con questa struttura:
1. INTESTAZIONE — Tribunale, parti, numero di ruolo
2. IN FATTO — Ricostruzione dei fatti dal punto di vista del convenuto
3. IN DIRITTO — Argomentazioni giuridiche e articoli a difesa
4. CONCLUSIONI — Richiesta di rigetto delle domande attoree con condanna alle spese. Nota: eventuali domande riconvenzionali devono essere formulate espressamente e depositate almeno 70 giorni prima dell'udienza.
5. FIRMA E DATA
""",
    ("civile", "decreto_ingiuntivo"): """
Redigi un ATTO DI OPPOSIZIONE A DECRETO INGIUNTIVO con questa struttura:
1. INTESTAZIONE — Tribunale, parti, estremi del decreto ingiuntivo opposto
2. PREMESSE IN FATTO — Ricostruzione sintetica dei fatti
3. MOTIVI DI OPPOSIZIONE — Argomentazioni giuridiche e fattuali
4. CONCLUSIONI — Petitum preciso (revoca/sospensione del decreto)
5. FIRMA E DATA
""",
    ("civile", "ricorso"): """
Redigi una MEMORIA DIFENSIVA con questa struttura:
1. INTESTAZIONE — Tribunale, parti, numero di ruolo
2. PREMESSE — Contesto procedurale
3. IN FATTO — Fatti rilevanti per la difesa
4. IN DIRITTO — Argomentazioni giuridiche
5. CONCLUSIONI — Richieste al giudice
6. FIRMA E DATA
""",
    ("penale", "*"): """
Redigi una MEMORIA DIFENSIVA PENALE con questa struttura:
1. INTESTAZIONE — Tribunale/GIP, imputato, reato contestato
2. IN FATTO — Ricostruzione dei fatti dalla prospettiva della difesa
3. IN DIRITTO — Inquadramento giuridico, cause di giustificazione, attenuanti
4. CONCLUSIONI — Richiesta di assoluzione/archiviazione/attenuanti
5. FIRMA E DATA
""",
    ("amministrativo", "*"): """
Redigi un RICORSO AMMINISTRATIVO con questa struttura:
1. INTESTAZIONE — TAR/Consiglio di Stato, ricorrente, atto impugnato
2. IN FATTO — Fatti rilevanti
3. MOTIVI DI RICORSO — Illegittimità per violazione di legge, eccesso di potere, incompetenza
4. CONCLUSIONI — Annullamento/sospensione dell'atto impugnato
5. FIRMA E DATA
""",
}


def _defensive_structure_prompt(proceeding_type: str, subtype: str) -> str:
    key = (proceeding_type, subtype)
    if key in _STRUCTURES:
        return _STRUCTURES[key]
    for (pt, ps), struct in _STRUCTURES.items():
        if pt == proceeding_type and ps == "*":
            return struct
    return _STRUCTURES[("civile", "atto_citazione")]


_LEGAL_DEFENSES = {
    ("civile", "decreto_ingiuntivo"): """
I MOTIVI DI OPPOSIZIONE tipici per un decreto ingiuntivo sono:
- Contestazione del credito (importo errato, pagamenti già effettuati non contabilizzati)
- Eccezione di inadempimento ex art. 1460 c.c. (il creditore non ha adempiuto le proprie obbligazioni)
- Contestazione delle prove documentali (fatture, contratti)
- Vizi formali del procedimento monitorio
- Prescrizione del credito
- Compensazione con crediti propri del debitore
USA SOLO questi motivi se applicabili ai fatti del documento.
NON inventare motivi non pertinenti al caso concreto.
NON usare terminologia di altri rami del diritto (diritto penale, diritto tributario, etc.).
""",
    ("civile", "atto_citazione"): """
Le ECCEZIONI tipiche per una comparsa di risposta sono:
- Contestazione dei fatti allegati dall'attore
- Eccezione di prescrizione
- Eccezione di difetto di legittimazione attiva/passiva
- Contestazione del nesso causale
- Concorso di colpa della controparte ex art. 1227 c.c.
- Difetto di prova del danno
USA SOLO questi motivi se applicabili ai fatti del documento.
NON inventare fatti o rapporti non presenti nel documento.
""",
}


def _legal_defenses_prompt(proceeding_type: str, subtype: str) -> str:
    """Subtype-specific list of typical legal grounds, or "" when none apply.

    Unlike _defensive_structure_prompt, there is no wildcard/ultimate fallback
    here — the catalog only covers the subtypes it's actually been reviewed
    for, and injecting an unrelated proceeding's grounds would be worse than
    injecting nothing.
    """
    return _LEGAL_DEFENSES.get((proceeding_type, subtype), "")


def generate_defensive_draft(
    proceeding: dict,
    document_text: str,
    citations: list,
    lang: str = "it",
    extra_instructions: str = "",
    article_texts: list = None,
) -> str:
    structure = _defensive_structure_prompt(
        proceeding.get("proceeding_type", "civile"),
        proceeding.get("proceeding_subtype", "altro"),
    )
    legal_defenses = _legal_defenses_prompt(
        proceeding.get("proceeding_type", "civile"),
        proceeding.get("proceeding_subtype", "altro"),
    )

    citations_text = ""
    if citations:
        citations_text = "\n".join(
            f"- {c['document_name']}, {s['name']}: {s.get('plain_text', '')[:400]}"
            for c in citations[:6] for s in (c.get("sections") or [])[:2]
        )

    system = (
        "Sei un avvocato esperto di diritto italiano. "
        "Redigi una bozza di documento difensivo professionale basandoti sul documento giudiziario fornito. "
        + (f"\n\n{legal_defenses}" if legal_defenses else "")
        + f"\n\n{structure}"
        "\n\nIMPORTANTE:"
        "\n- Usa un linguaggio giuridico formale italiano"
        "\n- Cita esplicitamente gli articoli di legge pertinenti"
        "\n- Usa [DA COMPILARE] per i campi che richiedono dati specifici non disponibili"
        "\n- Questa è una BOZZA — indica chiaramente che richiede revisione da parte dell'avvocato"
        "\n- Attieniti ESCLUSIVAMENTE ai fatti contenuti nel documento. Non inventare rapporti "
        "contrattuali, contesti o circostanze non esplicitamente menzionati."
        "\n- NON citare mai il testo di articoli di legge a memoria — usa SOLO i testi forniti "
        "nella sezione 'TESTO DEGLI ARTICOLI' qui sopra"
        "\n- Se il testo di un articolo non è nella sezione sopra, scrivi [TESTO DA VERIFICARE] "
        "invece del testo"
        "\n- NON usare le sentenze del corpus come fonte di argomentazioni giuridiche se non "
        "pertinenti al tipo di causa. Le sentenze servono SOLO come precedenti giurisprudenziali "
        "per supportare argomenti già fondati sui fatti del documento."
        + (f"\n\nNormativa di riferimento dal corpus legale:\n{citations_text}" if citations_text else "")
        + (f"\n\nIstruzioni aggiuntive: {extra_instructions}" if extra_instructions else "")
    )

    if article_texts:
        article_text_block = "\n".join(
            f"- {c['document_name']}, {s['name']}:\n  \"{s.get('plain_text', '')[:600]}\""
            for c in article_texts[:5] for s in (c.get("sections") or [])[:2]
            if s.get("plain_text")
        )
        if article_text_block:
            system += (
                "\n\nTESTO DEGLI ARTICOLI CITATI DALLA CONTROPARTE (fonte ufficiale — usa "
                "ESCLUSIVAMENTE questi testi, non citare mai a memoria):\n"
                + article_text_block
            )

    human = (
        f"DOCUMENTO GIUDIZIARIO DA CONTRASTARE:\n\n{document_text[:8000]}\n\n"
        f"ANALISI DEL PROCEDIMENTO:\n"
        f"- Tipo: {proceeding.get('proceeding_type', 'N/A')}\n"
        f"- Sottotipo: {proceeding.get('proceeding_subtype', 'N/A')}\n"
        f"- Tribunale: {proceeding.get('court', 'N/A')}\n"
        f"- Controparte: {proceeding.get('opposing_party', 'N/A')}\n"
        f"- Pretese avversarie: {proceeding.get('opposing_claims', 'N/A')}\n"
        f"- Articoli citati dalla controparte: {', '.join(proceeding.get('cited_articles', []))}\n"
        f"- Fatti chiave: {proceeding.get('key_facts', 'N/A')}\n\n"
        "Redigi ora il documento difensivo:"
    )

    draft = _call_chat(
        [SystemMessage(content=system), HumanMessage(content=human)],
        max_tokens=4000,
    )

    if citations:
        draft += (
            "\n\n⚠️ Le sentenze citate provengono dal corpus documentale e devono "
            "essere verificate dall'avvocato prima dell'uso."
        )

    return draft


def run_defensive_pipeline(
    document_text: str, user_message: str, session_lang: str = "it"
) -> dict:
    """Full pipeline: classify -> RAG -> draft.

    Returns {"draft": str, "proceeding": dict, "citations": list}. Each stage
    is best-effort: a RAG or classification failure degrades to defaults/empty
    citations rather than aborting the draft.
    """
    # Lazy imports: defensive_generation lives in the rag package but needs
    # rag.main (fine, same package) and chatbot.api's _merge_citations (the
    # only rag -> chatbot dependency here); importing at call time avoids a
    # module-load-time circular import with api.py, which imports this module.
    from .main import run as rag_run
    from .answer_processing import _extract_citations
    from ..chatbot.api import _merge_citations

    proceeding = classify_proceeding(document_text[:5000], session_lang)

    citations: list = []

    if proceeding.get("cited_articles"):
        query1 = "articoli " + ", ".join(proceeding["cited_articles"][:5])
        try:
            r1 = rag_run(query1, session_language=session_lang, skip_calculation=True)
            citations = _extract_citations(r1.get("raw_result", []))
        except Exception as exc:
            logger.warning("run_defensive_pipeline: RAG query 1 failed: %s", exc)

    # RAG query 3: fetch actual text of articles cited by opposing party
    article_citations = []
    if proceeding.get("cited_articles"):
        for article_ref in proceeding["cited_articles"][:5]:
            try:
                r3 = rag_run(article_ref, session_language=session_lang, skip_calculation=True)
                article_citations = _merge_citations(
                    article_citations,
                    _extract_citations(r3.get("raw_result", [])),
                )
            except Exception as exc:
                logger.warning("run_defensive_pipeline: RAG article lookup failed for %s: %s", article_ref, exc)

    try:
        r2 = rag_run(user_message, session_language=session_lang, skip_calculation=True)
        citations = _merge_citations(citations, _extract_citations(r2.get("raw_result", [])))
    except Exception as exc:
        logger.warning("run_defensive_pipeline: RAG query 2 failed: %s", exc)

    draft = generate_defensive_draft(
        proceeding, document_text, citations, session_lang,
        extra_instructions=user_message,
        article_texts=article_citations,
    )

    return {"draft": draft, "proceeding": proceeding, "citations": citations}
