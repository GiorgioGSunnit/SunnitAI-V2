"""Document generation: Italian civil procedure opposition act (atto di opposizione a decreto ingiuntivo)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import _call_chat

logger = logging.getLogger(__name__)

DOCUMENT_TYPE_REGISTRY = {
    "opposition_act": {
        "keywords": ["opposizione", "decreto ingiuntivo", "opporsi", "atto di opposizione", "oposicion", "monitorio", "opposition", "opposizione al decreto", "opporsi al decreto ingiuntivo"],
        "fields": ["plaintiff", "defendant", "injunction_reference", "court", "amount", "grounds", "date"],
        "label": "Atto di Opposizione a Decreto Ingiuntivo",
    },
    "rental_basic": {
        "keywords": ["cedolare secca", "affitto semplice", "locazione privata", "contratto base affitto", "affitto", "affitto con cedolare", "locazione con cedolare"],
        "fields": ["locatore", "conduttore", "indirizzo_immobile", "canone_mensile", "deposito_cauzionale", "data_inizio", "durata_anni", "cedolare_secca"],
        "label": "Contratto di Locazione con Cedolare Secca",
    },
    "rental_standard": {
        "keywords": ["locazione abitativa", "contratto 3+2", "affitto residenziale", "legge 431", "accordo territoriale", "canone concordato", "locazione", "contratto di locazione abitativa", "locazione 3+2"],
        "fields": ["locatore", "conduttore", "indirizzo_immobile", "riferimenti_catastali", "canone_annuale", "deposito_cauzionale", "data_inizio", "data_fine", "accordo_territoriale", "iban_locatore"],
        "label": "Contratto di Locazione Abitativa (3+2)",
    },
    "rental_student": {
        "keywords": ["locazione studenti", "affitto universitario", "studente universitario", "fuori sede", "locazione universitaria", "contratto", "contratto per studenti", "affitto per studenti", "locazione per universitari"],
        "fields": ["locatore", "conduttore", "indirizzo_immobile", "canone_mensile", "deposito_cauzionale", "data_inizio", "data_fine", "corso_studi", "nome_universita", "comune_universita"],
        "label": "Locazione Abitativa per Studenti Universitari",
    },
    "rental_transitional": {
        "keywords": ["locazione transitoria", "affitto temporaneo", "contratto transitorio", "locazione breve", "esigenza transitoria", "affitto di natura transitoria", "locazione di natura transitoria"],
        "fields": ["locatore", "conduttore", "indirizzo_immobile", "canone_mensile", "deposito_cauzionale", "data_inizio", "data_fine", "motivazione_transitorietà", "accordo_territoriale"],
        "label": "Locazione Abitativa di Natura Transitoria",
    },
    "rental_free_rent": {
        "keywords": ["canone libero", "locazione canone libero", "affitto 4+4", "contratto libero", "affitto a canone libero", "locazione a canone libero"],
        "fields": ["locatore", "conduttore", "indirizzo_immobile", "riferimenti_catastali", "canone_annuale", "canone_mensile", "deposito_cauzionale", "iban_locatore", "data_efficacia"],
        "label": "Contratto di Locazione ad Uso Abitativo a Canone Libero",
    },
    "rental_commercial": {
        "keywords": ["locazione commerciale", "affitto commerciale", "uso commerciale", "locazione ufficio", "affitto negozio", "uso diverso abitazione", "legge 392", "locazione ad uso commerciale", "affitto per ufficio", "contratto commerciale immobile"],
        "fields": ["parte_locatrice", "parte_conduttrice", "indirizzo_immobile", "riferimenti_catastali", "canone_annuale", "spese_conduzione", "deposito_cauzionale", "uso_destinato", "data_inizio", "data_fine", "foro_competente"],
        "label": "Contratto di Locazione ad Uso Commerciale",
    },
    "rental_cancellation": {
        "keywords": ["disdetta locazione", "recesso contratto affitto", "disdetta conduttore", "rescissione locazione", "recesso affitto", "disdetta del contratto di locazione", "disdetta dell'affitto", "recesso dal contratto", "recesso dal contratto di locazione", "comunicazione di recesso locazione"],
        "fields": ["conduttore", "indirizzo_conduttore", "locatore", "indirizzo_immobile", "data_stipula", "data_registrazione", "data_consegna", "preavviso_mesi", "modalita_invio"],
        "label": "Disdetta Contratto di Locazione",
    },
    "insurance_cancellation": {
        "keywords": ["disdetta polizza", "disdetta assicurazione", "recesso polizza", "cancellazione polizza", "disdetta contratto assicurazione", "disdetta della polizza", "disdetta del contratto assicurativo", "disdetta assicurativa", "rescissione polizza", "recesso dalla polizza", "cancellazione dell'assicurazione"],
        "fields": ["assicurato", "indirizzo_assicurato", "compagnia_assicurativa", "numero_polizza", "data_scadenza", "luogo", "data"],
        "label": "Disdetta Polizza Assicurativa",
    },
    "insurance_declaration": {
        "keywords": ["dichiarazione sostitutiva polizza", "dichiarazione assicurativa", "sostitutiva assicurazione", "dichiarazione polizza", "dichiarazione di polizza", "sostitutiva assicurativa"],
        "fields": ["dichiarante", "luogo_nascita", "data_nascita", "residenza", "qualita", "organizzazione", "sede_organizzazione", "data_inizio", "data_fine", "attivita_autorizzate"],
        "label": "Dichiarazione Sostitutiva di Polizza Assicurativa",
    },
    "employment_dismissal_appeal": {
        "keywords": ["impugnativa licenziamento", "impugnare licenziamento", "contestare licenziamento", "ricorso licenziamento", "opposizione licenziamento", "licenziamento illegittimo", "reintegrazione lavoro", "impugnativa di licenziamento", "impugnare il licenziamento", "contestare il licenziamento", "reintegra", "licenziamento senza giusta causa"],
        "fields": ["lavoratore", "indirizzo_lavoratore", "datore_lavoro", "indirizzo_datore_lavoro", "data_inizio_rapporto", "data_fine_rapporto", "qualifica", "mansione", "data_licenziamento", "modalita_licenziamento", "motivi_illegittimita", "modalita_invio", "data"],
        "label": "Impugnativa di Licenziamento",
    },
    "employment_termination": {
        "keywords": ["licenziamento giusta causa", "lettera licenziamento", "licenziare dipendente", "risoluzione rapporto lavoro", "licenziamento immediato", "lettera di licenziamento", "licenziamento per giusta causa", "licenziare", "risoluzione del rapporto di lavoro"],
        "fields": ["datore_lavoro", "indirizzo_datore_lavoro", "dipendente", "indirizzo_dipendente", "data_inizio_rapporto", "qualifica", "giusta_causa", "modalita_invio", "luogo", "data"],
        "label": "Lettera di Licenziamento per Giusta Causa",
    },
    "franchising_contract": {
        "keywords": ["contratto di franchising", "franchising", "affiliazione commerciale", "franchisor", "franchisee", "contratto di affiliazione"],
        "fields": ["franchisor", "indirizzo_franchisor", "franchisee", "indirizzo_franchisee", "settore_attivita", "area_esclusiva", "fee_ingresso", "royalties_percentuale", "incasso_minimo_annuale", "durata_anni", "foro_competente", "luogo", "data"],
        "label": "Contratto di Franchising",
    },
    "demand_letter": {
        "keywords": ["diffida", "messa in mora", "lettera di diffida", "intimazione", "sollecito legale", "burofax", "demand letter", "lettera di messa in mora"],
        "fields": ["mittente", "destinatario", "oggetto_controversia", "importo_dovuto", "termine_adempimento", "conseguenze", "data"],
        "label": "Diffida / Lettera di Messa in Mora",
    },
    "appeal": {
        "keywords": ["ricorso", "appello", "impugnazione", "opposizione al provvedimento", "recurso", "appeal", "petition", "atto di appello", "ricorso al tribunale", "impugnazione del provvedimento"],
        "fields": ["appellante", "controparte", "tribunale", "provvedimento_impugnato", "motivi", "petitum", "data"],
        "label": "Ricorso / Atto di Appello",
    },
    "power_of_attorney": {
        "keywords": ["procura", "delega", "mandato", "rappresentanza legale", "poder notarial", "power of attorney", "atto di procura", "delega notarile", "procura speciale"],
        "fields": ["conferente", "procuratore", "oggetto_poteri", "limitazioni", "durata", "data", "notaio"],
        "label": "Procura / Delega",
    },
    "sale_agreement": {
        "keywords": ["contratto di compravendita", "vendita", "acquisto", "trasferimento proprietà", "compraventa", "sale agreement", "contratto di vendita", "atto di compravendita", "contratto per la vendita"],
        "fields": ["venditore", "acquirente", "descrizione_bene", "prezzo", "modalita_pagamento", "data_consegna", "garanzie", "data"],
        "label": "Contratto di Compravendita",
    },
}


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


def is_generation_request(message: str) -> bool:
    msg = message.lower()
    action_verbs = [
        "genera", "scrivi", "redigi", "crea", "prepara", "stendi",
        "elabora", "redigimi", "generami", "scrivimi", "fammi",
        "produce", "draft", "write", "create", "redacta",
        "voglio", "vorrei", "ho bisogno di",
    ]
    strong_triggers = [
        "redigimi", "generami", "scrivimi", "fammi un atto",
        "fammi un contratto", "fammi una lettera",
    ]
    if any(t in msg for t in strong_triggers):
        return True
    all_keywords = [
        kw for entry in DOCUMENT_TYPE_REGISTRY.values()
        for kw in entry["keywords"]
    ]
    has_verb = any(v in msg for v in action_verbs)
    has_keyword = any(k in msg for k in all_keywords)
    return has_verb and has_keyword


def classify_document_type(message: str, lang: str) -> str:
    type_list = "\n".join(
        f"- {key}: {entry['label']}"
        for key, entry in DOCUMENT_TYPE_REGISTRY.items()
    )
    rental_hints = (
        "Per distinguere i tipi di locazione:\n"
        "- cedolare secca o affitto privato semplice → rental_basic\n"
        "- studenti, universitari, fuori sede → rental_student\n"
        "- transitorio, temporaneo, esigenza transitoria → rental_transitional\n"
        "- canone libero, 4+4 → rental_free_rent\n"
        "- commerciale, ufficio, negozio, B2B → rental_commercial\n"
        "- locazione abitativa generica senza altri segnali → rental_standard\n"
    )
    system = (
        "Sei un classificatore di richieste di documenti legali. "
        "Dato un messaggio utente, restituisci esattamente una delle seguenti chiavi "
        "che meglio corrisponde al tipo di documento richiesto:\n\n"
        f"{type_list}\n"
        "- unknown: nessuno dei precedenti\n\n"
        f"{rental_hints}\n"
        "Restituisci SOLO la chiave, senza testo aggiuntivo."
    )
    human = f"Messaggio utente:\n{message}"
    try:
        result = _call_chat(
            [SystemMessage(content=system), HumanMessage(content=human)],
            max_tokens=20,
        ).strip().lower()
        if result not in DOCUMENT_TYPE_REGISTRY and result != "unknown":
            logger.warning(f"classify_document_type got unexpected value: {result!r}, falling back to unknown")
            return "unknown"
        return result
    except Exception as e:
        logger.warning(f"classify_document_type failed: {e}, falling back to unknown")
        return "unknown"


def extract_document_fields(user_message: str, doc_type: str, lang: str) -> Dict[str, str]:
    entry = DOCUMENT_TYPE_REGISTRY[doc_type]
    fields = entry["fields"]
    label = entry["label"]
    fields_list = ", ".join(fields)

    if lang == "es":
        system = (
            f"Eres un extractor de información legal. "
            f"Lee el mensaje del usuario y extrae los detalles estructurados para un documento de tipo '{label}'. "
            f"Devuelve un objeto JSON con exactamente estas claves: {fields_list}. "
            "Usa el texto original del usuario para los valores. "
            "Si un campo no se menciona, devuelve una cadena vacía para esa clave. "
            "Devuelve SOLO el objeto JSON, sin texto adicional, sin markdown."
        )
    elif lang == "en":
        system = (
            f"You are a legal information extractor. "
            f"Read the user message and extract structured details for a document of type '{label}'. "
            f"Return a JSON object with exactly these keys: {fields_list}. "
            "Use the original user text for values. "
            "If a field is not mentioned, return an empty string for that key. "
            "Return ONLY the JSON object, with no extra text, no markdown."
        )
    else:
        system = (
            f"Sei un estrattore di informazioni legali. "
            f"Leggi il messaggio dell'utente ed estrai i dettagli strutturati per un documento di tipo '{label}'. "
            f"Restituisci un oggetto JSON con esattamente queste chiavi: {fields_list}. "
            "Usa il testo originale dell'utente per i valori. "
            "Se un campo non è menzionato, restituisci una stringa vuota per quella chiave. "
            "Restituisci SOLO l'oggetto JSON, senza testo aggiuntivo, senza markdown."
        )

    human = f"Messaggio dell'utente:\n{user_message}"
    raw = _call_chat(
        [SystemMessage(content=system), HumanMessage(content=human)],
        max_tokens=400,
    )

    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("extract_document_fields: JSON parse failed for doc_type=%r, raw=%r", doc_type, raw[:200])
        parsed = {}

    return {k: str(parsed.get(k, "") or "") for k in fields}


def extract_case_details(user_message: str) -> Dict[str, str]:
    return extract_document_fields(user_message, "opposition_act", "it")


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
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
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
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
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
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente l'atto, senza prefazioni o spiegazioni."
    )


def _rental_basic_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos urbanos españoles. "
            "Redacta un contrato de arrendamiento de vivienda con tributación simplificada. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Arrendador, arrendatario, D.N.I./N.I.E., fecha y lugar de firma.\n"
            "2. OBJETO Y DURACIÓN — Descripción del inmueble, duración en años, fecha de inicio, "
            "prórroga tácita con preaviso de 3 meses.\n"
            "3. RENTA Y RÉGIMEN FISCAL — Importe mensual, día de pago, régimen fiscal aplicable, "
            "sin actualización por IPC salvo pacto expreso.\n"
            "4. OBLIGACIONES DEL ARRENDATARIO — Uso residencial, prohibición de subarriendo, "
            "mantenimiento ordinario, suministros a cargo del arrendatario.\n"
            "5. DEPÓSITO Y CLÁUSULAS FINALES — Depósito máximo 2 mensualidades, gastos contractuales, "
            "mediación obligatoria, firmas con doble suscripción de cláusulas generales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in residential tenancy law. "
            "Draft a basic residential tenancy agreement with simplified tax treatment. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Landlord, tenant, tax/ID numbers, date and place of signing.\n"
            "2. PROPERTY AND TERM — Description of the property, duration in years, start date, "
            "tacit renewal with 3-month notice to terminate.\n"
            "3. RENT AND TAX REGIME — Monthly amount, payment date, applicable tax regime, "
            "no index-linking unless expressly agreed.\n"
            "4. TENANT OBLIGATIONS — Residential use only, no subletting, routine maintenance, "
            "utilities at tenant's expense.\n"
            "5. DEPOSIT AND FINAL CLAUSES — Deposit capped at 2 monthly rents, contract costs, "
            "mandatory mediation, signatures with express approval of general clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi un contratto di locazione con cedolare secca. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Locatore, conduttore, C.F., data e luogo di stipula.\n"
        "2. OGGETTO E DURATA — Descrizione dell'immobile, durata in anni, data di inizio, "
        "rinnovo tacito con disdetta 3 mesi prima.\n"
        "3. CANONE E CEDOLARE SECCA — Importo mensile, giorno di pagamento, regime cedolare secca, "
        "nessun aggiornamento ISTAT.\n"
        "4. OBBLIGHI DEL CONDUTTORE — Uso abitativo, divieto di sublocazione, manutenzione ordinaria, "
        "utenze a carico del conduttore.\n"
        "5. DEPOSITO E CLAUSOLE FINALI — Deposito massimo 2 mensilità, spese contrattuali, "
        "mediazione obbligatoria, firme con doppia sottoscrizione artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_standard_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos urbanos españoles. "
            "Redacta un contrato de arrendamiento de vivienda habitual de renta pactada (3+2 años). "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Arrendador y arrendatario con datos completos, referencias catastrales, "
            "certificado de eficiencia energética.\n"
            "2. DURACIÓN Y PRÓRROGA — 3 años + prórroga de 2 años, condiciones de desistimiento, "
            "prórroga tácita.\n"
            "3. RENTA Y DEPÓSITO — Renta anual pactada, IBAN, pagos mensuales, actualización por IPC 75%, "
            "depósito máximo 3 mensualidades con intereses.\n"
            "4. CARGAS Y OBLIGACIONES — Distribución de gastos conforme a normativa, uso residencial, "
            "prohibición de subarriendo, entrega y devolución, obras con consentimiento escrito.\n"
            "5. CLÁUSULAS FINALES Y FIRMAS — Comisión paritaria, RGPD, fuero competente, "
            "doble suscripción de cláusulas generales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in residential tenancy law. "
            "Draft a standard assured shorthold tenancy agreement (3+2 year term). "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Landlord and tenant with full details, land registry references, "
            "energy performance certificate.\n"
            "2. TERM AND RENEWAL — 3-year term plus 2-year extension, termination conditions, "
            "tacit renewal.\n"
            "3. RENT AND DEPOSIT — Annual rent as agreed, IBAN, monthly instalments, "
            "75% index-linking, deposit capped at 3 monthly rents with interest.\n"
            "4. OBLIGATIONS — Schedule of charges per applicable regulations, residential use only, "
            "no subletting, handover and return, alterations require written consent.\n"
            "5. FINAL CLAUSES AND SIGNATURES — Joint committee, GDPR, jurisdiction, "
            "express approval of general clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi un contratto di locazione abitativa ai sensi della L. 431/98 (3+2 anni). "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Locatore e conduttore con dati completi, riferimenti catastali, "
        "prestazione energetica.\n"
        "2. DURATA E RINNOVO — 3 anni + proroga 2 anni L. 431/98, condizioni di disdetta, "
        "rinnovo tacito.\n"
        "3. CANONE E DEPOSITO — Canone annuale da accordo territoriale, IBAN, rate mensili, "
        "ISTAT 75%, deposito massimo 3 mensilità con interessi.\n"
        "4. ONERI E OBBLIGHI — Tabella oneri D.M. infrastrutture, uso abitativo, "
        "divieto di sublocazione, consegna e riconsegna, modifiche con consenso scritto.\n"
        "5. CLAUSOLE FINALI E FIRME — Commissione paritetica, GDPR, foro competente, "
        "doppia sottoscrizione artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_student_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos para estudiantes universitarios. "
            "Redacta un contrato de arrendamiento de temporada para estudiante universitario. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Arrendador y arrendatario con datos completos, referencias catastrales.\n"
            "2. DURACIÓN Y NATURALEZA TRANSITORIA — De 6 meses a 3 años, renovación automática, "
            "declaración expresa de matriculación universitaria con nombre del curso y centro, "
            "acuerdo territorial aplicable.\n"
            "3. RENTA Y DEPÓSITO — Renta mensual, transferencia anticipada antes del día 5, "
            "depósito máximo 3 mensualidades, eventuales garantías adicionales.\n"
            "4. OBLIGACIONES DE LAS PARTES — Suministros a cargo del arrendatario, uso residencial, "
            "prohibición de subarriendo con resolución de pleno derecho, desistimiento por causa grave "
            "con preaviso de 3 meses, entrega y devolución del inmueble.\n"
            "5. CLÁUSULAS FINALES Y FIRMAS — Comisión paritaria, RGPD, fuero competente, "
            "doble suscripción de cláusulas generales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in student residential tenancy law. "
            "Draft a student tenancy agreement for a university student. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Landlord and tenant with full details, land registry references.\n"
            "2. TERM AND TRANSITIONAL NATURE — 6 months to 3 years, automatic renewal, "
            "explicit declaration of university enrolment with course name and institution, "
            "applicable territorial agreement.\n"
            "3. RENT AND DEPOSIT — Monthly rent, advance bank transfer by day 5, "
            "deposit capped at 3 monthly rents, any additional guarantees.\n"
            "4. OBLIGATIONS OF THE PARTIES — Utilities at tenant's expense, residential use only, "
            "no subletting with automatic termination clause, early termination for serious cause "
            "with 3-month notice, handover and return of the property.\n"
            "5. FINAL CLAUSES AND SIGNATURES — Joint committee, GDPR, jurisdiction, "
            "express approval of general clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi un contratto di locazione abitativa per studenti universitari ai sensi della L. 431/98. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Locatore e conduttore con dati completi, riferimenti catastali.\n"
        "2. DURATA E NATURA TRANSITORIA — Da 6 mesi a 3 anni, rinnovo automatico, "
        "dichiarazione esplicita di frequenza del corso universitario con nome del corso e ateneo, "
        "accordo territoriale applicabile.\n"
        "3. CANONE E DEPOSITO — Canone mensile, bonifico anticipato entro il giorno 5, "
        "deposito massimo 3 mensilità, eventuali garanzie aggiuntive.\n"
        "4. OBBLIGHI DELLE PARTI — Utenze a carico del conduttore, uso abitativo, "
        "divieto di sublocazione con risoluzione di diritto, recesso per gravi motivi con preavviso "
        "di 3 mesi, consegna e riconsegna dell'immobile.\n"
        "5. CLAUSOLE FINALI E FIRME — Commissione paritetica, GDPR, foro competente, "
        "doppia sottoscrizione artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_transitional_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos de temporada y transitorios. "
            "Redacta un contrato de arrendamiento de naturaleza transitoria. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Arrendador y arrendatario con datos completos, referencias catastrales.\n"
            "2. DURACIÓN Y NECESIDAD TRANSITORIA — De 1 a 18 meses, cese automático sin denuncia, "
            "motivación transitoria documentada obligatoria para duraciones superiores a 30 días, "
            "acuerdo territorial aplicable, consecuencias del incumplimiento de las modalidades de firma.\n"
            "3. RENTA Y DEPÓSITO — Renta mensual, transferencia anticipada antes del día 5, "
            "depósito máximo 3 mensualidades, eventuales garantías adicionales.\n"
            "4. OBLIGACIONES DE LAS PARTES — Suministros a cargo del arrendatario, uso residencial "
            "con lista de convivientes, prohibición de subarriendo, desistimiento por causa grave "
            "con preaviso de 30 días no aplicable a contratos de 30 días o menos, "
            "entrega y devolución del inmueble.\n"
            "5. CLÁUSULAS FINALES Y FIRMAS — Comisión paritaria, RGPD, fuero competente, "
            "doble suscripción de cláusulas generales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in short-term and transitional residential tenancy law. "
            "Draft a transitional residential tenancy agreement. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Landlord and tenant with full details, land registry references.\n"
            "2. TERM AND TRANSITIONAL NEED — 1 to 18 months, automatic expiry without notice, "
            "documented transitional reason mandatory for terms exceeding 30 days, "
            "applicable territorial agreement, consequences of non-compliance with signing requirements.\n"
            "3. RENT AND DEPOSIT — Monthly rent, advance bank transfer by day 5, "
            "deposit capped at 3 monthly rents, any additional guarantees.\n"
            "4. OBLIGATIONS OF THE PARTIES — Utilities at tenant's expense, residential use with list "
            "of occupants, no subletting, early termination for serious cause with 30-day notice "
            "not applicable to agreements of 30 days or less, handover and return of the property.\n"
            "5. FINAL CLAUSES AND SIGNATURES — Joint committee, GDPR, jurisdiction, "
            "express approval of general clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi un contratto di locazione abitativa di natura transitoria ai sensi della L. 431/98. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Locatore e conduttore con dati completi, riferimenti catastali.\n"
        "2. DURATA ED ESIGENZA TRANSITORIA — Da 1 a 18 mesi, cessazione automatica senza disdetta, "
        "motivazione transitoria documentata obbligatoria per durate superiori a 30 giorni, "
        "accordo territoriale applicabile, conseguenze dell'inadempimento delle modalità di stipula.\n"
        "3. CANONE E DEPOSITO — Canone mensile, bonifico anticipato entro il giorno 5, "
        "deposito massimo 3 mensilità, eventuali garanzie aggiuntive.\n"
        "4. OBBLIGHI DELLE PARTI — Utenze a carico del conduttore, uso abitativo con elenco dei "
        "conviventi, divieto di sublocazione, recesso per gravi motivi con preavviso di 30 giorni "
        "non applicabile a contratti di durata ≤ 30 giorni, consegna e riconsegna dell'immobile.\n"
        "5. CLAUSOLE FINALI E FIRME — Commissione paritetica, GDPR, foro competente, "
        "doppia sottoscrizione artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_free_rent_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos urbanos de renta libre. "
            "Redacta un contrato de arrendamiento de vivienda a renta libre (4+4 años). "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO Y PREMISAS — Arrendador y arrendatario con datos completos, "
            "declaraciones preliminares de propiedad, referencias catastrales, certificado energético.\n"
            "2. DURACIÓN Y DESISTIMIENTO — 4+4 años, preaviso de desistimiento del arrendador "
            "de 6 meses con causa tasada, desistimiento del arrendatario con preaviso de 6 meses "
            "por carta certificada.\n"
            "3. OBLIGACIONES Y RENTA — Uso residencial, prohibición de subarriendo, renta anual "
            "con pagos mensuales anticipados antes del día 5 por transferencia IBAN, "
            "actualización IPC 75% desde el mes 13, gastos ordinarios a cargo del arrendatario, "
            "gastos extraordinarios a cargo del arrendador.\n"
            "4. DEPÓSITO Y GARANTÍAS — Depósito máximo 3 mensualidades con intereses legales anuales, "
            "no imputable a renta, eventual garantía suplementaria.\n"
            "5. DISPOSICIONES FINALES Y FIRMAS — Fecha de eficacia, derecho español, RGPD, "
            "registro 50% cada parte, fuero competente, doble suscripción de cláusulas generales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in open-market residential tenancy law. "
            "Draft an open-market tenancy agreement (4+4 year term). "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING AND RECITALS — Landlord and tenant with full details, ownership declarations, "
            "land registry references, energy performance certificate.\n"
            "2. TERM AND TERMINATION — 4+4 years, landlord's 6-month notice with statutory grounds, "
            "tenant's 6-month notice by recorded delivery.\n"
            "3. OBLIGATIONS AND RENT — Residential use only, no subletting, annual rent with monthly "
            "advance payments by day 5 via IBAN bank transfer, 75% index-linking from month 13, "
            "routine repairs at tenant's expense, major repairs at landlord's expense.\n"
            "4. DEPOSIT AND GUARANTEES — Deposit capped at 3 monthly rents with annual statutory "
            "interest, not offset against rent, any supplementary guarantee.\n"
            "5. FINAL PROVISIONS AND SIGNATURES — Effective date, governing law, GDPR, registration "
            "costs split 50/50, jurisdiction, express approval of general clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi un contratto di locazione ad uso abitativo a canone libero ai sensi della L. 431/98 (4+4 anni). "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE E PREMESSE — Locatore e conduttore con dati completi, "
        "dichiarazioni preliminari di proprietà, riferimenti catastali, attestato energetico D.Lgs. 28/2011.\n"
        "2. DURATA E RECESSO — 4+4 anni L. 431/98, preavviso di disdetta del locatore 6 mesi "
        "con motivazione tassativa, recesso del conduttore con preavviso di 6 mesi tramite raccomandata.\n"
        "3. OBBLIGHI E CANONE — Uso abitativo, divieto di sublocazione, canone annuale con rate "
        "mensili anticipate entro il giorno 5 tramite bonifico IBAN, ISTAT 75% dal tredicesimo mese, "
        "spese ordinarie a carico del conduttore, spese straordinarie a carico del locatore.\n"
        "4. DEPOSITO E GARANZIE — Deposito massimo 3 mensilità con interessi legali annuali "
        "art. 11 L. 392/78, non imputabile a canone, eventuale garanzia supplementare.\n"
        "5. DISPOSIZIONI FINALI E FIRME — Data di efficacia, diritto italiano, GDPR, "
        "registrazione 50% ciascuna parte, foro competente, doppia sottoscrizione artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_commercial_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos de locales de negocio. "
            "Redacta un contrato de arrendamiento para uso distinto del de vivienda (6+6 años). "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Parte arrendadora y arrendataria con razón social, N.I.F., "
            "representante legal, referencias catastrales, declaraciones de conformidad del inmueble, "
            "certificado energético.\n"
            "2. DURACIÓN Y DESISTIMIENTO — 6+6 años, preaviso de denuncia de 12 meses, "
            "renuncia del arrendador a denegar la renovación en el primer vencimiento, "
            "desistimiento de la arrendataria con preaviso de 6 meses.\n"
            "3. RENTA Y PAGO — Renta anual más IVA, gastos de conducción y calefacción, "
            "4 pagos trimestrales anticipados 01/01-01/04-01/07-01/10, "
            "actualización IPC 75% desde el segundo año.\n"
            "4. USO MEJORAS Y CARGAS — Destino de uso especificado, prohibición de cambio de destino, "
            "mejoras solo con consentimiento escrito quedan en beneficio del arrendador, "
            "prohibición de subarriendo salvo grupo societario, todos los gastos a cargo de la "
            "arrendataria incluidos tributos locales.\n"
            "5. DEPÓSITO VARIOS Y FIRMAS — Depósito de 3 mensualidades por transferencia, "
            "visita del arrendador con preaviso de 24 horas, comunicaciones por burofax, "
            "timbre y registro 50% arrendataria, fuero competente exclusivo, "
            "firmas con aprobación de cláusulas esenciales.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el contrato, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in commercial property tenancy law. "
            "Draft a commercial lease agreement (6+6 year term). "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Landlord and tenant with company name, VAT number, legal representative, "
            "land registry references, compliance declarations, energy performance certificate.\n"
            "2. TERM AND TERMINATION — 6+6 years, 12-month notice to terminate, landlord waives "
            "right to refuse renewal at first expiry, tenant's early termination with 6-month notice.\n"
            "3. RENT AND PAYMENT — Annual rent plus VAT, service and heating charges, "
            "4 quarterly advance instalments 01/01-01/04-01/07-01/10, "
            "75% index-linking from the second year.\n"
            "4. USE IMPROVEMENTS AND CHARGES — Specified permitted use, no change of use, "
            "improvements only with written consent and revert to landlord, no subletting except "
            "within group companies, all outgoings at tenant's expense including local taxes.\n"
            "5. DEPOSIT MISCELLANEOUS AND SIGNATURES — 3-month deposit by bank transfer, "
            "landlord access with 24-hour notice, notices by recorded delivery or PEC, "
            "stamp duty and registration 50% tenant, exclusive jurisdiction clause, "
            "signatures with express approval of essential clauses.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni commerciali italiane. "
        "Redigi un contratto di locazione ad uso commerciale ai sensi della L. 392/78 (6+6 anni). "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Parte locatrice e conduttrice con ragione sociale, P.IVA, "
        "rappresentante legale, riferimenti catastali, dichiarazioni di conformità dell'immobile, "
        "attestato energetico.\n"
        "2. DURATA E RECESSO — 6+6 anni L. 392/78, preavviso di disdetta 12 mesi, "
        "rinuncia della locatrice al diniego di rinnovo alla prima scadenza, "
        "recesso della conduttrice con preavviso di 6 mesi.\n"
        "3. CANONE E PAGAMENTO — Canone annuale oltre IVA, spese di conduzione e riscaldamento, "
        "4 rate trimestrali anticipate 01/01-01/04-01/07-01/10, "
        "ISTAT 75% dal secondo anno.\n"
        "4. USO MIGLIORIE E ONERI — Destinazione d'uso specificata, divieto di cambio di destinazione, "
        "migliorie solo con consenso scritto e restano al locatore, divieto di sublocazione salvo "
        "gruppo societario, tutti gli oneri a carico della conduttrice inclusi TARI e TASI.\n"
        "5. DEPOSITO VARIE E FIRME — Deposito 3 mensilità tramite bonifico, visita della locatrice "
        "con preavviso di 24 ore, comunicazioni tramite PEC, bollo e registro 50% conduttrice, "
        "foro competente esclusivo, firme con approvazione delle clausole essenziali artt. 1341-1342 c.c.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il contratto, senza prefazioni o spiegazioni."
    )


def _rental_cancellation_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un abogado experto en arrendamientos urbanos. "
            "Redacta una carta de desistimiento de contrato de arrendamiento de vivienda. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Datos del arrendatario (nombre, DNI, domicilio), datos del arrendador "
            "(nombre, domicilio), medio de envío (carta certificada con acuse de recibo o correo electrónico "
            "certificado), objeto de la comunicación.\n"
            "2. DECLARACIÓN DE DESISTIMIENTO — Manifestación expresa de desistir del contrato de "
            "arrendamiento, fecha de firma del contrato, número de referencia de registro, "
            "dirección del inmueble arrendado.\n"
            "3. PLAZO DE ENTREGA — Fecha en la que el inmueble será entregado libre de personas y enseres, "
            "plazo de preaviso en meses/días desde la recepción de la presente comunicación.\n"
            "4. CIERRE Y FIRMA — Fórmula de cierre, lugar, fecha, firma del arrendatario.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente la carta, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in residential tenancy law. "
            "Draft a notice of termination of a residential tenancy agreement. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Tenant's details (name, ID, address), landlord's details (name, address), "
            "method of delivery (recorded delivery letter or certified email), subject of the notice.\n"
            "2. NOTICE OF TERMINATION — Express statement of intent to terminate the tenancy agreement, "
            "date the agreement was signed, land registry or tax registration reference, "
            "address of the rented property.\n"
            "3. HANDOVER DATE — Date by which the property will be returned vacant and free of belongings, "
            "notice period in months/days from receipt of this communication.\n"
            "4. CLOSING AND SIGNATURE — Closing formula, place, date, tenant's signature.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un avvocato esperto in locazioni abitative italiane. "
        "Redigi una disdetta di contratto di locazione da parte del conduttore. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Dati del conduttore (nome, cognome, C.F., indirizzo), dati del locatore "
        "(nome, cognome, indirizzo), modalità di invio (raccomandata A/R o PEC), oggetto della comunicazione.\n"
        "2. DICHIARAZIONE DI RECESSO — Manifestazione espressa di voler recedere anticipatamente dal "
        "contratto di locazione, data di stipula del contratto, data di registrazione presso l'Agenzia "
        "delle Entrate, indirizzo dell'immobile locato.\n"
        "3. TERMINE DI CONSEGNA — Data entro cui l'immobile sarà consegnato libero da cose e persone, "
        "preavviso in mesi/giorni dalla ricezione della presente comunicazione.\n"
        "4. CHIUSURA E FIRMA — Formula di chiusura, luogo, data, firma del conduttore.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente la lettera, senza prefazioni o spiegazioni."
    )


def _insurance_cancellation_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un experto en derecho de seguros. "
            "Redacta una carta de desistimiento o no renovación de póliza de seguro por vencimiento natural. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Lugar y fecha, destinatario (nombre compañía aseguradora, dirección), "
            "objeto con número de póliza y fecha de vencimiento.\n"
            "2. CUERPO DE LA CARTA — Solicitud formal de no renovación de la póliza por vencimiento natural, "
            "referencia a las condiciones generales de la póliza y a la normativa vigente, "
            "petición de acuse de recibo.\n"
            "3. FIRMA — Fórmula de cierre cordial, firma del asegurado.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente la carta, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in insurance law. "
            "Draft a notice of cancellation or non-renewal of an insurance policy at natural expiry. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Place and date, addressee (insurance company name, address), "
            "subject with policy number and expiry date.\n"
            "2. BODY OF THE LETTER — Formal request not to renew the policy at natural expiry, "
            "reference to the general policy conditions and applicable regulations, "
            "request for acknowledgement of receipt.\n"
            "3. SIGNATURE — Cordial closing formula, policyholder's signature.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un esperto in diritto assicurativo. "
        "Redigi una lettera di disdetta di polizza assicurativa per naturale scadenza. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Luogo e data, destinatario (nome compagnia assicurativa, indirizzo), "
        "oggetto con numero polizza e data di scadenza.\n"
        "2. CORPO DELLA LETTERA — Richiesta formale di disdetta per naturale scadenza, riferimento "
        "alle condizioni di polizza e alle norme vigenti, richiesta di ricevuta di ritorno.\n"
        "3. FIRMA — Formula di chiusura cordiale, firma dell'assicurato.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente la lettera, senza prefazioni o spiegazioni."
    )


def _insurance_declaration_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un experto en derecho de seguros y documentación oficial. "
            "Redacta una declaración sustitutiva de póliza de seguro para una actividad artístico-cultural. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Fac-símil en papel con membrete, título declaración sustitutiva.\n"
            "2. DATOS DEL DECLARANTE — Nombre y apellidos, lugar y fecha de nacimiento, domicilio, "
            "calidad en que actúa, nombre de la organización, sede.\n"
            "3. PERÍODO DE REFERENCIA — Fechas desde/hasta incluidos días de montaje/desmontaje, "
            "actividades autorizadas objeto de la cobertura.\n"
            "4. DECLARACIONES — Póliza contra accidentes para actividades artístico-culturales, "
            "cobertura de daños a cosas de terceros, aviso APL sobre señalización de espacios.\n"
            "5. CONSENTIMIENTO Y FIRMA — Consentimiento para el tratamiento de datos personales, "
            "fecha, firma, adjuntar documento de identidad.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el documento, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in insurance law and official documentation. "
            "Draft a substitute declaration of insurance policy for an artistic-cultural activity. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Facsimile on headed paper, title of the substitute declaration.\n"
            "2. DECLARANT'S DETAILS — Full name, place and date of birth, address, "
            "capacity in which acting, name of the organisation, registered office.\n"
            "3. REFERENCE PERIOD — Dates from/to including set-up/dismantling days, "
            "authorised activities covered by the policy.\n"
            "4. DECLARATIONS — Accident policy for artistic-cultural activities, "
            "coverage for damage to third-party property, APL notice regarding signage of venues.\n"
            "5. CONSENT AND SIGNATURE — Consent to personal data processing, "
            "date, signature, attach identity document.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un esperto in diritto assicurativo e documentazione ufficiale. "
        "Redigi una dichiarazione sostitutiva di polizza assicurativa per un'attività artistico-culturale. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Fac-simile su carta intestata, titolo dichiarazione sostitutiva.\n"
        "2. DATI DEL DICHIARANTE — Nome e cognome, luogo e data di nascita, residente in, "
        "qualità in cui agisce, nome dell'organizzazione, sede.\n"
        "3. PERIODO DI RIFERIMENTO — Date dal/al inclusi giorni di allestimento/disallestimento, "
        "attività autorizzate oggetto della copertura.\n"
        "4. DICHIARAZIONI — Polizza contro infortuni per attività artistico-culturali, copertura danni "
        "a cose di terzi, avvertenza APL su segnalazione degli spazi.\n"
        "5. CONSENSO E FIRMA — Consenso al trattamento dei dati personali ai sensi della L. 196/2003, "
        "data, firma, allegare documento di identità.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il documento, senza prefazioni o spiegazioni."
    )


def _employment_dismissal_appeal_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un experto en derecho laboral. "
            "Redacta una impugnación formal de despido para proteger los derechos del trabajador. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Trabajador remitente con domicilio, empleador destinatario con dirección, "
            "modalidad de envío (carta certificada con acuse de recibo o PEC), objeto: impugnación de despido.\n"
            "2. DATOS DE LA RELACIÓN LABORAL — Período de trabajo desde/hasta, categoría y puesto, "
            "fecha y modalidad del despido (si fue solo verbal, especificarlo).\n"
            "3. MOTIVOS DE ILEGITIMIDAD — Exposición de las razones por las que el despido se considera nulo y/o ilegítimo.\n"
            "4. IMPUGNACIÓN Y REQUERIMIENTO — Impugnación formal del despido, requerimiento al empleador "
            "para que garantice la reincorporación en un plazo de 7 días o abone las indemnizaciones económicas correspondientes, "
            "advertencia de recurso ante la autoridad judicial en caso de incumplimiento.\n"
            "5. RESERVAS Y FIRMA — Reserva de reclamar todos los conceptos pendientes no abonados, fecha, firma del trabajador.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el documento, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in employment law. "
            "Draft a formal dismissal appeal letter to protect the worker's rights. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Worker as sender with address, employer as recipient with address, "
            "delivery method (registered letter with acknowledgement of receipt or PEC), subject: dismissal appeal.\n"
            "2. EMPLOYMENT DETAILS — Period of employment from/to, job title and duties, "
            "date and method of dismissal (if only verbal, state so explicitly).\n"
            "3. GROUNDS OF ILLEGITIMACY — Statement of the reasons why the dismissal is considered void and/or unlawful.\n"
            "4. FORMAL APPEAL AND NOTICE — Formal contestation of the dismissal, notice to the employer "
            "to ensure reinstatement within 7 days or pay the applicable financial compensation, "
            "warning of judicial recourse in the event of non-compliance.\n"
            "5. RESERVATIONS AND SIGNATURE — Reservation of the right to claim all outstanding entitlements not yet paid, date, worker's signature.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un esperto in diritto del lavoro. "
        "Redigi un'impugnativa formale di licenziamento a tutela dei diritti del lavoratore. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Lavoratore mittente con indirizzo, datore di lavoro destinatario con indirizzo, "
        "modalità di invio (raccomandata A/R o PEC), oggetto: impugnativa di licenziamento.\n"
        "2. DATI DEL RAPPORTO DI LAVORO — Periodo di lavoro dal/al, qualifica e mansione, "
        "data e modalità del licenziamento (se solo verbale, specificarlo).\n"
        "3. MOTIVI DI ILLEGITTIMITÀ — Esposizione delle ragioni per cui il licenziamento è ritenuto nullo e/o illegittimo.\n"
        "4. IMPUGNATIVA E DIFFIDA — Impugnazione formale del licenziamento, diffida al datore di lavoro "
        "a garantire la reintegrazione entro 7 giorni o a corrispondere le indennità economiche spettanti, "
        "avvertimento di ricorso all'autorità giudiziaria in caso di inottemperanza.\n"
        "5. RISERVE E FIRMA — Riserva di rivendicare tutte le spettanze non corrisposte, data, firma del lavoratore.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il documento, senza prefazioni o spiegazioni."
    )


def _employment_termination_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un experto en derecho laboral. "
            "Redacta una carta de despido por causa justificada en nombre del empleador. "
            "El documento debe ser formalmente correcto y estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO — Empleador remitente con dirección, empleado destinatario con dirección, "
            "modalidad de envío (carta certificada con acuse de recibo o entrega en mano), objeto: despido por causa justificada.\n"
            "2. COMUNICACIÓN DE DESPIDO — Comunicación oficial de resolución inmediata de la relación laboral "
            "conforme al art. 54 del Estatuto de los Trabajadores, indicación de la causa justificada específica.\n"
            "3. EFECTOS Y CIERRE — Efecto inmediato del despido, invitación a recoger los efectos personales, "
            "lugar, fecha, firma del empleador.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el documento, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in employment law. "
            "Draft a letter of dismissal for just cause on behalf of the employer. "
            "The document must be formally correct and structured in the following sections:\n\n"
            "1. HEADING — Employer as sender with address, employee as recipient with address, "
            "delivery method (registered letter with acknowledgement of receipt or hand delivery), subject: dismissal for just cause.\n"
            "2. NOTICE OF DISMISSAL — Official communication of immediate termination of the employment relationship "
            "pursuant to art. 2119 of the Civil Code, specifying the particular just cause.\n"
            "3. EFFECTS AND CLOSING — Immediate effect of the dismissal, invitation to collect personal belongings, "
            "place, date, employer's signature.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un esperto in diritto del lavoro. "
        "Redigi una lettera di licenziamento per giusta causa per conto del datore di lavoro. "
        "Il documento deve essere formalmente corretto e strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE — Datore di lavoro mittente con indirizzo, dipendente destinatario con indirizzo, "
        "modalità di invio (raccomandata A/R o consegna a mano), oggetto: licenziamento per giusta causa.\n"
        "2. COMUNICAZIONE DI LICENZIAMENTO — Comunicazione ufficiale di risoluzione immediata del rapporto di lavoro "
        "ai sensi dell'art. 2119 c.c., indicazione della giusta causa specifica.\n"
        "3. EFFETTI E CHIUSURA — Effetto immediato del licenziamento, invito a ritirare gli effetti personali, "
        "luogo, data, firma del datore di lavoro.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il documento, senza prefazioni o spiegazioni."
    )


def _franchising_contract_system(lang: str) -> str:
    ph = _placeholder(lang)
    if lang == "es":
        return (
            "Eres un experto en derecho mercantil y contratos de franquicia. "
            "Redacta un contrato de franquicia completo y formalmente correcto estructurado en las siguientes secciones:\n\n"
            "1. ENCABEZAMIENTO Y CONSIDERANDOS — Franquiciador y franquiciado con datos personales completos, "
            "declaraciones preliminares (sector de actividad, know-how, marca, local comercial del franquiciado).\n"
            "2. OBJETO, EXCLUSIVIDAD Y CANON — Concesión de know-how y marca con exclusividad territorial, "
            "canon de entrada con servicios incluidos, servicios adicionales e importe correspondiente, "
            "autorizaciones intuitu personae no cedibles.\n"
            "3. ROYALTIES Y OBLIGACIONES — Porcentaje de royalties sobre volumen de negocio anual, "
            "facturación mínima anual, obligaciones del franquiciador (know-how, procedimientos, manuales, asistencia, formación), "
            "obligaciones del franquiciado (gestión autónoma, prohibición de traslado de sede, confidencialidad, respeto de la política empresarial).\n"
            "4. DURACIÓN, RESOLUCIÓN Y CONTROLES — Duración del contrato con prórroga tácita y preaviso de 6 meses, "
            "cláusula resolutoria expresa, efectos de la resolución (cese del uso de marca y signos distintivos), "
            "nulidad por información falsa, controles del franquiciador en el punto de venta.\n"
            "5. DISPOSICIONES FINALES Y FIRMAS — Resolución de controversias mediante Cámara de Comercio con foro competente, "
            "tratamiento de datos RGPD, lugar, fecha, firmas del franquiciador y del franquiciado.\n\n"
            f"Usa {ph} para cada dato ausente sin excepción — no inventes ni asumas fechas, nombres, códigos, números ni ningún otro dato no proporcionado explícitamente: usa {ph} para fechas, números de identificación, direcciones y referencias numéricas. "
            "No añadas advertencias meta-legales. "
            "Redacta directamente el documento, sin preámbulos ni explicaciones."
        )
    if lang == "en":
        return (
            "You are an expert in commercial law and franchise agreements. "
            "Draft a complete and formally correct franchise contract structured in the following sections:\n\n"
            "1. HEADING AND RECITALS — Franchisor and franchisee with full personal details, "
            "preliminary declarations (sector of activity, know-how, trademark, franchisee's commercial premises).\n"
            "2. SUBJECT MATTER, EXCLUSIVITY AND FEE — Grant of know-how and trademark with territorial exclusivity, "
            "entry fee with included services, additional services and corresponding amount, "
            "intuitu personae authorisations not transferable.\n"
            "3. ROYALTIES AND OBLIGATIONS — Royalty percentage on annual turnover, "
            "annual minimum turnover, franchisor's obligations (know-how, procedures, manuals, assistance, training), "
            "franchisee's obligations (independent management, prohibition on relocating premises, confidentiality, compliance with company policy).\n"
            "4. DURATION, TERMINATION AND CONTROLS — Contract duration with tacit renewal and 6-month prior notice for termination, "
            "express termination clause, effects of termination (cessation of trademark and distinctive signs), "
            "voidance for false information, franchisor's controls at the point of sale.\n"
            "5. FINAL PROVISIONS AND SIGNATURES — Dispute resolution through the Chamber of Commerce with competent jurisdiction, "
            "GDPR data processing, place, date, signatures of franchisor and franchisee.\n\n"
            f"Use {ph} for every missing piece of information without exception — never invent or assume dates, names, codes, numbers or any other data not explicitly provided: use {ph} for dates, tax codes, addresses and reference numbers. "
            "Do not add meta-legal disclaimers. "
            "Write the document directly, without preamble or explanation."
        )
    return (
        "Sei un esperto in diritto commerciale e contratti di franchising. "
        "Redigi un contratto di franchising completo e formalmente corretto strutturato nelle seguenti sezioni:\n\n"
        "1. INTESTAZIONE E PREMESSE — Franchisor e franchisee con dati anagrafici completi, "
        "dichiarazioni preliminari (settore attività, know-how, marchio, locale commerciale franchisee).\n"
        "2. OGGETTO, ESCLUSIVA E FEE — Concessione know-how e marchio con esclusiva per area, "
        "fee di ingresso con servizi inclusi, servizi ulteriori e relativo importo, "
        "autorizzazioni intuitu personae non cedibili.\n"
        "3. ROYALTIES E OBBLIGHI — Percentuale royalties su volume d'affari annuo, "
        "incasso minimo annuale, obblighi del franchisor (know-how, procedure, manuali, assistenza, formazione), "
        "obblighi del franchisee (gestione autonoma, divieto trasferimento sede, riservatezza, rispetto policy aziendale).\n"
        "4. DURATA, RISOLUZIONE E CONTROLLI — Durata contratto con rinnovo tacito e disdetta 6 mesi prima, "
        "clausola risolutiva espressa, effetti risoluzione (cessazione marchio e segni distintivi), "
        "annullamento per false informazioni, controlli franchisor sul punto vendita.\n"
        "5. DISPOSIZIONI FINALI E FIRME — Risoluzione controversie tramite Camera di Commercio con foro competente, "
        "trattamento dati GDPR, luogo, data, firme franchisor e franchisee.\n\n"
        f"Usa {ph} per ogni dato mancante senza eccezioni — non inventare né assumere date, nomi, codici, numeri o qualsiasi altro dato non esplicitamente fornito: usa {ph} per date, codici fiscali, indirizzi e riferimenti numerici. "
        "Non aggiungere avvertenze meta-legali sul documento stesso. "
        "Scrivi direttamente il documento, senza prefazioni o spiegazioni."
    )


def generate_document(
    user_message: str,
    doc_type: str,
    lang: str,
    citations: list = None,
) -> Dict[str, Any]:
    """Generic document generator dispatching on doc_type via DOCUMENT_TYPE_REGISTRY.

    Returns:
        {"draft": str, "case_details": Dict[str, str], "doc_type": str}
    """
    _SYSTEM_FN = {
        "opposition_act": _opposition_system,
        "rental_basic": _rental_basic_system,
        "rental_standard": _rental_standard_system,
        "rental_student": _rental_student_system,
        "rental_transitional": _rental_transitional_system,
        "rental_free_rent": _rental_free_rent_system,
        "rental_commercial": _rental_commercial_system,
        "rental_cancellation": _rental_cancellation_system,
        "insurance_cancellation": _insurance_cancellation_system,
        "insurance_declaration": _insurance_declaration_system,
        "employment_dismissal_appeal": _employment_dismissal_appeal_system,
        "employment_termination": _employment_termination_system,
        "franchising_contract": _franchising_contract_system,
    }

    system_fn = _SYSTEM_FN.get(doc_type)
    if system_fn is None:
        logger.warning(
            "generate_document: no system prompt function for doc_type=%r, falling back to _opposition_system",
            doc_type,
        )
        system_fn = _opposition_system

    fields_dict = extract_document_fields(user_message, doc_type, lang)

    system_content = system_fn(lang)
    if citations:
        citations_text = "\n".join(
            f"- {cit.get('document_name', '')}, sezione {sec.get('name', '')}"
            for cit in citations
            for sec in (cit.get("sections") or [])
            if sec.get("name")
        )
        if citations_text:
            system_content += (
                "\nGround the legal arguments in these specific retrieved documents and cite them "
                "explicitly in the Motivi di opposizione section:\n" + citations_text
            )

    ph = _placeholder(lang)
    field_lines = "\n".join(
        f"- {k}: {v if v else ph}"
        for k, v in fields_dict.items()
    )
    human_content = (
        f"Document fields:\n{field_lines}\n\n"
        "Draft the complete document following exactly the structure specified in the system prompt."
    )

    raw_output = _call_chat(
        [SystemMessage(content=system_content), HumanMessage(content=human_content)],
        max_tokens=2000,
    )

    return {"draft": raw_output, "case_details": fields_dict, "doc_type": doc_type}


def generate_opposition_act(
    case_details: Dict[str, Any],
    retrieved_sections: List[Any],
    session_lang: str = "it",
    citations: list = None,
) -> str:
    """Thin wrapper around generate_document for backwards compatibility."""
    lang = session_lang or "it"
    lines = [f"{k}: {v}" for k, v in case_details.items() if v]
    if retrieved_sections:
        lines.append(
            f"\nRelevant knowledge base sections:\n{_format_retrieved_sections(retrieved_sections, lang)}"
        )
    return generate_document("\n".join(lines), "opposition_act", lang, citations)["draft"]
