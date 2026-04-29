"""Document generation: Italian civil procedure opposition act (atto di opposizione a decreto ingiuntivo)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import _call_chat

def _placeholder(lang: str) -> str:
    if lang == "es":
        return "[COMPLETAR]"
    if lang == "en":
        return "[TO BE COMPLETED]"
    return "[DA COMPILARE]"

_GENERATION_COMBOS = [
    (
        {
            "genera", "generami", "scrivi", "scrivimi", "redigi", "redigimi",
            "crea", "creami", "prepara", "preparami", "drafta",
            "elabora", "elaborami", "formulami", "formula",
        },
        {"opposizione", "atto", "ricorso", "memoria", "decreto", "ingiuntivo"},
    ),
]


def is_generation_request(user_message: str) -> bool:
    """Return True if the message appears to request document generation.

    Uses keyword detection only — no LLM call.
    A message qualifies when it contains at least one action verb AND at least
    one document/legal-context noun from the defined combo pairs, OR when it
    contains a standalone strong trigger word.
    """
    lowered = user_message.lower()
    tokens = set(re.findall(r"[a-zÀ-ɏ]+", lowered))

    for action_set, target_set in _GENERATION_COMBOS:
        if tokens & action_set and tokens & target_set:
            return True

    strong = {"drafta", "redigimi", "generami", "scrivimi"}
    if tokens & strong:
        return True

    return False


def extract_case_details(user_message: str) -> Dict[str, str]:
    """Extract structured case details from a free-text user message via LLM.

    Returns a dict with keys:
        plaintiff, defendant, injunction_reference, court, amount, grounds, date
    Any field not mentioned in the message is returned as an empty string.
    """
    system = (
        "Sei un estrattore di informazioni legali per la procedura civile italiana. "
        "Leggi il messaggio dell'utente ed estrai i dettagli strutturati per un "
        "atto di opposizione a decreto ingiuntivo. "
        "Restituisci un oggetto JSON con esattamente queste chiavi: "
        "plaintiff, defendant, injunction_reference, court, amount, grounds, date. "
        "Usa il testo italiano originale per i valori. "
        "Se un campo non e' menzionato, restituisci una stringa vuota per quella chiave. "
        "Restituisci SOLO l'oggetto JSON, senza testo aggiuntivo, senza markdown."
    )
    human = f"Messaggio dell'utente:\n{user_message}"
    raw = _call_chat(
        [SystemMessage(content=system), HumanMessage(content=human)],
        max_tokens=400,
    )

    import json
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = {}

    keys = ["plaintiff", "defendant", "injunction_reference", "court", "amount", "grounds", "date"]
    return {k: str(parsed.get(k, "") or "") for k in keys}


def _field(value: Optional[str], label: str, lang: str = "it") -> str:
    """Return value if non-empty, else a labelled placeholder."""
    v = (value or "").strip()
    return v if v else f"{_placeholder(lang)} ({label})"


def _format_retrieved_sections(retrieved_sections: List[Any], lang: str = "it") -> str:
    """Format retrieved knowledge-base sections into a readable context block."""
    if not retrieved_sections:
        if lang == "es":
            return "(ninguna seccion recuperada de la base de conocimiento)"
        if lang == "en":
            return "(no sections retrieved from the knowledge base)"
        return "(nessuna sezione recuperata dalla knowledge base)"

    lines: List[str] = []
    for i, item in enumerate(retrieved_sections, 1):
        if isinstance(item, dict):
            title = item.get("title") or item.get("heading") or f"Sezione {i}"
            text = item.get("text") or item.get("text_en") or item.get("content") or ""
            source = item.get("document_title") or item.get("source") or ""
            entry = f"[{i}] {title}"
            if source:
                entry += f" (Fonte: {source})"
            if text:
                entry += f"\n{text[:600]}"
                if len(text) > 600:
                    entry += "..."
        else:
            entry = f"[{i}] {str(item)[:600]}"
        lines.append(entry)

    return "\n\n".join(lines)


def _opposition_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en derecho civil espanol. "
            "Redacta un escrito de oposicion a decreto monitorio conforme al art. 815 de la Ley de Enjuiciamiento Civil. "
            "El documento debe ser formalmente correcto, en terminologia juridica espanola precisa, y estructurado "
            "en las siguientes secciones en el orden indicado:\n\n"
            "1. ENCABEZAMIENTO — Juzgado competente, partes (oponente y acreedor), numero y fecha del decreto monitorio.\n"
            "2. ANTECEDENTES DE HECHO — Exposicion sintetica de los hechos relevantes.\n"
            "3. MOTIVOS DE OPOSICION — Argumentos juridicos y facticos. "
            "Cita las secciones de la base de conocimiento proporcionadas en el contexto cuando sean pertinentes, "
            "indicando la fuente entre parentesis.\n"
            "4. SUPLICO — Peticion precisa: lo que el oponente solicita al Juzgado "
            "(anulacion/suspension del decreto, desestimacion de las pretensiones de la parte contraria, condena en costas).\n"
            "5. FIRMA Y FECHA — Lugar, fecha, firma del letrado.\n\n"
            f"Usa {ph} para datos que falten y que el cliente debera completar. "
            "No anadas advertencias meta-legales en el propio documento. "
            "Redacta directamente el escrito, sin preambulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert civil litigation lawyer. "
            "Draft an opposition to an injunction/payment order under applicable civil procedure rules. "
            "The document must be formally correct, in precise legal language, and structured "
            "in the following sections in the order given:\n\n"
            "1. HEADING — Competent court, parties (opposing party and creditor), reference number and date of the order.\n"
            "2. STATEMENT OF FACTS — Concise factual background.\n"
            "3. GROUNDS OF OPPOSITION — Legal and factual arguments. "
            "Cite sections from the knowledge base provided in the context where relevant, "
            "indicating the source in parentheses.\n"
            "4. RELIEF SOUGHT — Precise petitum: what the opposing party requests from the court "
            "(revocation/suspension of the order, dismissal of the claimant's claims, costs award).\n"
            "5. SIGNATURE AND DATE — Place, date, lawyer's signature.\n\n"
            f"Use {ph} for any missing data that the client must supply. "
            "Do not add meta-legal disclaimers to the document itself. "
            "Write the document directly, without preamble or explanation."
        )
    # Default: Italian
    return (
        "Sei un avvocato esperto in diritto civile italiano. "
        "Redigi un atto di opposizione a decreto ingiuntivo ai sensi degli artt. 645 e ss. c.p.c. "
        "Il documento deve essere formalmente corretto, in italiano giuridico preciso, e strutturato "
        "nelle seguenti sezioni nell'ordine indicato:\n\n"
        "1. INTESTAZIONE — Tribunale competente, parti (opponente e opposto), "
        "numero e data del decreto ingiuntivo.\n"
        "2. PREMESSE IN FATTO — Ricostruzione sintetica dei fatti rilevanti.\n"
        "3. MOTIVI DI OPPOSIZIONE — Argomenti giuridici e fattuali. "
        "Cita le sezioni della knowledge base fornite nel contesto ove pertinenti, "
        "indicando la fonte tra parentesi.\n"
        "4. CONCLUSIONI — Petitum preciso: cosa chiede l'opponente al Tribunale "
        "(revoca/sospensione del decreto, rigetto delle domande avversarie, condanna alle spese).\n"
        "5. FIRMA E DATA — Luogo, data, firma del difensore.\n\n"
        f"Usa {ph} per dati mancanti che il cliente dovra' integrare. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente l'atto, senza prefazioni o spiegazioni."
    )


def generate_opposition_act(
    case_details: Dict[str, Any],
    retrieved_sections: List[Any],
    session_lang: str = "it",
) -> str:
    """Generate an Italian opposition act (atto di opposizione a decreto ingiuntivo).

    Args:
        case_details: Dict with keys plaintiff, defendant, injunction_reference,
                      court, amount, grounds, date. Any can be empty/None.
        retrieved_sections: List of section dicts from the RAG knowledge base.
        session_lang: Session language code (output is always Italian).

    Returns:
        The full generated act as a string.
    """
    lang = session_lang or "it"
    ph = _placeholder(lang)

    plaintiff = _field(case_details.get("plaintiff"), "nome opponente / opposing party name", lang)
    defendant = _field(case_details.get("defendant"), "nome opposto/creditore / creditor name", lang)
    injunction_ref = _field(case_details.get("injunction_reference"), "n. decreto ingiuntivo / order reference", lang)
    court = _field(case_details.get("court"), "tribunale / court", lang)
    amount = _field(case_details.get("amount"), "importo / amount", lang)
    grounds = (case_details.get("grounds") or "").strip() or ph + " (grounds of opposition)"
    date = _field(case_details.get("date"), "data / date", lang)

    sections_block = _format_retrieved_sections(retrieved_sections, lang)

    human_content = (
        "Case details:\n"
        f"- Opposing party (plaintiff): {plaintiff}\n"
        f"- Creditor (defendant): {defendant}\n"
        f"- Order reference: {injunction_ref}\n"
        f"- Court: {court}\n"
        f"- Amount disputed: {amount}\n"
        f"- Date: {date}\n"
        f"- Grounds of opposition (client's free text):\n{grounds}\n\n"
        "Relevant knowledge base sections (use as references in the Grounds section):\n"
        f"{sections_block}\n\n"
        "Draft the complete document following exactly the structure specified in the system prompt."
    )

    return _call_chat(
        [
            SystemMessage(content=_opposition_system(lang)),
            HumanMessage(content=human_content),
        ],
        max_tokens=2000,
    )
