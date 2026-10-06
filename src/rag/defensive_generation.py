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
        "cited_codes": [],
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
        "- cited_codes: lista delle abbreviazioni dei codici citati dalla controparte "
        "(es. [\"c.c.\", \"c.p.c.\", \"c.p.\", \"c.p.p.\"])\n"
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

    cited_codes = parsed.get("cited_codes")
    if isinstance(cited_codes, list):
        result["cited_codes"] = [c for c in cited_codes if isinstance(c, str)]

    return result


_STRATEGY_STRUCTURE = """
Redigi un'ANALISI LEGALE COMPLETA strutturata nelle seguenti 5 sezioni obbligatorie.
Usa esattamente questi titoli in grassetto. Non omettere nessuna sezione né sottosezione.

**1. Premessa e Inquadramento della Questione**

**1a. Descrizione dei Fatti**
- Identifica chi è il cliente e qual è il suo obiettivo reale (non solo ciò che chiede formalmente, ma ciò che vuole ottenere: risarcimento economico, tutela della reputazione, risoluzione rapida, etc.)
- Ricostruzione cronologica dei fatti: chi, cosa, quando, dove, come — senza interpretazioni
- ATTENZIONE: distingui chiaramente ciò che il cliente sa con certezza da ciò che crede o ricorda
- Controparti e soggetti terzi coinvolti (testimoni, assicurazioni, enti, periti)
- Cosa è già stato fatto: diffide, trattative, atti, procedimenti pendenti, scadenze già decorse
- ATTENZIONE: non inventare MAI cifre, date o fatti non presenti nel documento.
  Usa [DA VERIFICARE] per qualsiasi dato non confermato dai documenti.

**1b. Acquisizione e Vaglio dei Documenti**
- Elenca i documenti disponibili (contratti, corrispondenza, PEC, email, messaggi, atti notificati, provvedimenti)
- Verifica autenticità, completezza e data certa di ciascun documento
- Se applicabile: individua i documenti mancanti e chi potrebbe averli

**1c. Termini e Urgenze**
- Prescrizione e decadenza applicabili
- Termini processuali in corso (impugnazioni, opposizioni, costituzione in giudizio)
- Necessità di misure urgenti: cautelari, sequestri, diffide, interruzione della prescrizione
- Conflitti di interessi e profili deontologici da verificare prima di accettare l'incarico

---

**2. Qualificazione Giuridica**

**2a. Identificazione**
- Traduci i fatti in questioni di diritto: qual è il rapporto giuridico in gioco
- Individua le norme applicabili (sostanziali e processuali) e le possibili fattispecie alternative
- Verifica diritto intertemporale e fonti sovranazionali se rilevanti
- IMPORTANTE: cita SOLO norme del ramo giuridico pertinente (cause civili → c.c. e c.p.c.; cause penali → c.p. e c.p.p.; cause amministrative → leggi amministrative). NON citare norme penali in cause civili e viceversa.

**2b. Ricerca: Norme, Dottrina, Giurisprudenza**
- Testo vigente delle norme e relative modifiche recenti
- Orientamenti giurisprudenziali prevalenti, in particolare Cassazione e Sezioni Unite; segnala eventuali contrasti tra orientamenti
- Dottrina per i punti controversi
- Usa SOLO i testi forniti nella sezione 'TESTO DEGLI ARTICOLI' — per qualsiasi articolo non presente scrivi [TESTO DA VERIFICARE]
- Usa SOLO le sentenze fornite nel corpus legale — non citare giurisprudenza a memoria

**2c. Profili Processuali**
- Rito applicabile ed eventuali condizioni di procedibilità (mediazione, negoziazione assistita, querela, etc.)
- Alternative al giudizio: transazione, arbitrato, soluzioni stragiudiziali
- Giurisdizione e competenza (materia, valore, territorio)
- Legittimazione e interesse ad agire

---

**3. Prova**
- Per ogni fatto rilevante: chi deve provarlo (onere della prova) e con quali mezzi
- Punti di forza probatori e lacune; cosa si può ancora acquisire
- Prova della controparte: cosa potrebbe produrre contro il cliente
- Valutazione complessiva della solidità probatoria della posizione del cliente

---

**4. Analisi di Forza e Rischio**
- Tesi principale e tesi subordinate, in ordine dal più al meno favorevole
- Possibili eccezioni e difese della controparte, con la replica consigliata per ciascuna
- Stima della probabilità di successo (alta/media/bassa) con motivazione
- Stima dei tempi processuali e dei costi (compresa la soccombenza)
- Recuperabilità del credito o effettiva utilità del risultato atteso

---

**5. Strategia e Opzioni per il Cliente**

**5a. Quadro delle Opzioni**
- Elenca tutte le strade percorribili con pro e contro di ciascuna
- Raccomandazione motivata, lasciando al cliente la decisione informata

**5b. Piano Operativo**
- Prossimi atti da compiere e relative scadenze
- Documenti da raccogliere o produrre

**5c. Gestione dell'Incarico**
- Preventivo e modalità di compenso: [DA COMPILARE]
- Mandato, informativa privacy, eventuale copertura assicurativa: [DA COMPILARE]
- Aspettative: tempi stimati, esiti possibili, frequenza di aggiornamento al cliente
- Apertura del fascicolo e scadenzario: [DA COMPILARE]
"""


def _extract_deepdive_topics(draft: str, proceeding: dict) -> list:
    """Extract 2-3 main argument topics from the draft for deepdive suggestions.

    Best-effort: a failure here must never block the draft itself — the
    caller just skips the suggestion when this returns [].
    """
    try:
        raw = _call_chat([
            SystemMessage(content=(
                "Sei un assistente legale. Dal seguente documento difensivo, estrai i 3-5 temi "
                "più importanti da approfondire, preferibilmente dalla sezione '4. Analisi di Forza e Rischio' "
                "o '5. Strategia e Opzioni'. Restituiscili come etichette brevi "
                "(max 5 parole ciascuna, in italiano, minuscolo). "
                "Restituisci SOLO una lista JSON di stringhe, niente altro. "
                "Esempio: [\"prescrizione del credito\", \"onere della prova\", \"mediazione obbligatoria\"]"
            )),
            HumanMessage(content=draft[:3000]),
        ], max_tokens=100)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        topics = json.loads(cleaned)
        if isinstance(topics, list):
            return [t for t in topics if isinstance(t, str)][:3]
    except Exception as exc:
        logger.warning("_extract_deepdive_topics failed: %s", exc)
    return []


def _format_deepdive_suggestion(topics: list) -> str:
    """Format topics as a natural suggestion with markdown deepdive links."""
    if not topics:
        return ""
    links = [f"[{t}](deepdive:{t.lower().replace(' ', '-')})" for t in topics]
    if len(links) == 1:
        parts = links[0]
    elif len(links) == 2:
        parts = f"{links[0]} o {links[1]}"
    else:
        parts = f"{', '.join(links[:-1])} o {links[-1]}"
    return f"Vuoi approfondire in particolare {parts}?"


def _generate_legal_reasoning(
    proceeding: dict,
    document_text: str,
    draft_sections_1_3: str,
    citations: list,
    lang: str = "it",
) -> str:
    """Second focused LLM call for sections 4 and 5 — deep legal reasoning."""

    citations_text = ""
    if citations:
        citations_text = "\n".join(
            f"- {c['document_name']}, {s['name']}: {s.get('plain_text','')[:400]}"
            for c in citations[:6] for s in (c.get('sections') or [])[:2]
        )

    system = (
        "Sei un avvocato esperto di diritto italiano con 20 anni di esperienza in contenzioso. "
        "Hai già prodotto le sezioni 1-3 dell'analisi legale (premessa, qualificazione, prova). "
        "Ora devi produrre SOLO le sezioni 4 e 5, con ragionamento giuridico concreto e approfondito.\n\n"
        "**4. Analisi di Forza e Rischio**\n"
        "- Sviluppa la tesi principale con argomentazione specifica ai fatti: non dire genericamente "
        "'il gesto era intenzionale' ma spiega PERCHÉ sulla base dei fatti concreti (modalità, "
        "contesto, sequenza temporale, violazione delle regole di gioco)\n"
        "- Per ogni tesi subordinata, indica l'articolo specifico e perché si applica a questo caso\n"
        "- Per le eccezioni della controparte: anticipale e confutale con argomenti specifici\n"
        "- Stima probabilità di successo con motivazione concreta basata sui fatti, non generica\n"
        "- Indica tempi realistici e costi stimati\n\n"
        "**5. Strategia e Opzioni per il Cliente**\n"
        "- Per ogni opzione indica il fondamento normativo specifico (es. 'appello della parte "
        "civile ai soli effetti civili ex art. 576 c.p.p.')\n"
        "- Il piano operativo deve indicare l'atto concreto da compiere CON il termine specifico "
        "(es. '15 giorni dalla notifica della sentenza motivata') e i documenti esatti da produrre\n"
        "- La raccomandazione deve spiegare PERCHÉ quella strada è preferibile rispetto alle altre "
        "in questo caso specifico\n\n"
        "REGOLE:\n"
        "- Ragiona sui fatti specifici di questo caso, non in astratto\n"
        "- Cita le norme con il numero esatto dell'articolo\n"
        "- Se citi giurisprudenza usa SOLO quella fornita nel corpus o scrivi [DA VERIFICARE]\n"
        "- NON inventare cifre, date o fatti non presenti nel documento\n"
        "- Usa [DA COMPILARE] solo per dati che il cliente deve fornire\n"
        + (f"\n\nFonti dal corpus legale:\n{citations_text}" if citations_text else "")
    )

    human = (
        f"ANALISI DEL CASO (sezioni 1-3 già prodotte):\n{draft_sections_1_3[:3000]}\n\n"
        f"FATTI ORIGINALI:\n{document_text[:4000]}\n\n"
        f"Tipo procedimento: {proceeding.get('proceeding_type', 'N/A')} — "
        f"{proceeding.get('proceeding_subtype', 'N/A')}\n"
        f"Articoli contestati: {', '.join(proceeding.get('cited_articles', []))}\n\n"
        "Produci ora le sezioni 4 e 5 con ragionamento giuridico concreto:"
    )

    return _call_chat(
        [SystemMessage(content=system), HumanMessage(content=human)],
        max_tokens=3000,
    )


def generate_defensive_draft(
    proceeding: dict,
    document_text: str,
    citations: list,
    lang: str = "it",
    extra_instructions: str = "",
    article_texts: list = None,
) -> str:
    citations_text = ""
    if citations:
        citations_text = "\n".join(
            f"- {c['document_name']}, {s['name']}: {s.get('plain_text', '')[:400]}"
            for c in citations[:6] for s in (c.get("sections") or [])[:2]
        )

    system = (
        "Sei un avvocato esperto di diritto italiano con 20 anni di esperienza in contenzioso civile e penale. "
        "Analizza il caso descritto e produci un'analisi legale concreta e approfondita — "
        "non una descrizione generica di cosa si potrebbe fare, ma il ragionamento giuridico effettivo: "
        "quali norme si applicano e perché, quali argomenti reggono e quali no, "
        "quale teoria del caso è più solida, quali sono i punti deboli specifici di questo caso. "
        "Ragiona come se stessi preparando il fascicolo per un'udienza la prossima settimana. "
        "Segui la struttura in 5 sezioni ma riempi ogni sezione con analisi concreta, non con placeholder.\n\n"
        + _STRATEGY_STRUCTURE
        + "\n\nREGOLE FONDAMENTALI:"
        "\n- Attieniti ESCLUSIVAMENTE ai fatti contenuti nel documento e nella conversazione. "
        "Non inventare rapporti contrattuali, contesti o circostanze non esplicitamente menzionati."
        "\n- NON citare mai il testo di articoli di legge a memoria — usa SOLO i testi forniti "
        "nella sezione 'TESTO DEGLI ARTICOLI' o scrivi [TESTO DA VERIFICARE]"
        "\n- NON usare le sentenze del corpus come fonte di argomentazioni se non pertinenti. "
        "Le sentenze servono SOLO come precedenti a supporto di argomenti già fondati sui fatti."
        "\n- Usa [DA COMPILARE] per i campi che richiedono dati specifici non disponibili"
        "\n- Questa è una BOZZA che richiede revisione da parte dell'avvocato"
        "\n- NON inventare MAI cifre, importi, date, nomi o fatti specifici non esplicitamente "
        "presenti nel documento fornito. Se un dato non è nel documento scrivi [DA VERIFICARE] "
        "al suo posto. È preferibile un campo vuoto a un dato inventato."
        "\n- Per i fatti incerti o non documentati usa formule come 'secondo quanto dichiarato', "
        "'da verificare', 'non risulta dai documenti disponibili' — mai affermare come certo "
        "ciò che non lo è."
        "\n- Ogni argomento deve essere specifico al caso concreto — NON usare formule generiche "
        "come 'contestare la fatturazione' o 'verificare i documenti'. Spiega PERCHÉ quella "
        "specifica contestazione regge in questo caso, citando i fatti e le norme."
        "\n- Per la sezione 4, stima la probabilità di successo con una motivazione concreta "
        "basata sui fatti del caso, non una generica valutazione 'media/alta/bassa'."
        "\n- Per la sezione 5, indica il prossimo atto concreto da compiere (es. 'depositare "
        "atto di appello entro X giorni dalla notifica della sentenza motivata') non "
        "descrizioni generiche."
        + (f"\n\nNormativa di riferimento dal corpus legale (sezione 2):\n{citations_text}" if citations_text else "")
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
                "\n\nTESTO DEGLI ARTICOLI CITATI DALLA CONTROPARTE (fonte ufficiale):\n"
                + article_text_block
            )

    human = (
        f"DOCUMENTO GIUDIZIARIO:\n\n{document_text[:8000]}\n\n"
        f"ANALISI DEL PROCEDIMENTO:\n"
        f"- Tipo: {proceeding.get('proceeding_type', 'N/A')}\n"
        f"- Sottotipo: {proceeding.get('proceeding_subtype', 'N/A')}\n"
        f"- Tribunale: {proceeding.get('court', 'N/A')}\n"
        f"- Controparte: {proceeding.get('opposing_party', 'N/A')}\n"
        f"- Pretese avversarie: {proceeding.get('opposing_claims', 'N/A')}\n"
        f"- Articoli citati dalla controparte: {', '.join(proceeding.get('cited_articles', []))}\n"
        f"- Fatti chiave: {proceeding.get('key_facts', 'N/A')}\n\n"
        "Redigi ora la strategia difensiva completa in tutte e 5 le sezioni:"
    )

    draft = _call_chat(
        [SystemMessage(content=system), HumanMessage(content=human)],
        max_tokens=4000,
    )

    # Second focused call — replace sections 4 and 5 with deeper legal reasoning
    try:
        _split_marker = "**4."
        if _split_marker in draft:
            _sections_1_3 = draft[:draft.index(_split_marker)].strip()
        else:
            _sections_1_3 = draft[:3000]

        _sections_4_5 = _generate_legal_reasoning(
            proceeding, document_text, _sections_1_3, citations, lang
        )

        if _split_marker in draft:
            draft = _sections_1_3 + "\n\n" + _sections_4_5
    except Exception as _exc:
        logger.warning("generate_defensive_draft: second LLM call failed, using first draft: %s", _exc)

    topics = _extract_deepdive_topics(draft, proceeding)
    if topics:
        draft += "\n\n" + _format_deepdive_suggestion(topics)

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
        codes = " ".join(proceeding.get("cited_codes", []))
        query1 = f"articoli {', '.join(proceeding['cited_articles'][:5])} {codes}".strip()
        try:
            r1 = rag_run(query1, session_language=session_lang, skip_calculation=True)
            citations = _extract_citations(r1.get("raw_result", []))
        except Exception as exc:
            logger.warning("run_defensive_pipeline: RAG query 1 failed: %s", exc)

    # RAG query 3: fetch actual text of articles cited by opposing party
    article_citations = []
    if proceeding.get("cited_articles"):
        for article_ref in proceeding["cited_articles"][:5]:
            codes = " ".join(proceeding.get("cited_codes", []))
            scoped_ref = f"{article_ref} {codes}".strip()
            try:
                r3 = rag_run(scoped_ref, session_language=session_lang, skip_calculation=True)
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


def generate_deepdive_analysis(topic: str, chat_history: list, session_lang: str = "it") -> str:
    """Generate a focused legal analysis of one defensive argument topic.

    chat_history is the session's build_history_messages() output — already
    containing the original draft as an assistant turn, which this scans for
    to ground the analysis in the specific case rather than the topic alone.
    """
    from .main import run as rag_run
    from .answer_processing import _extract_citations

    citations: list = []
    try:
        r = rag_run(topic, session_language=session_lang, skip_calculation=True)
        citations = _extract_citations(r.get("raw_result", []))
    except Exception as exc:
        logger.warning("generate_deepdive_analysis: RAG failed: %s", exc)

    citations_text = ""
    if citations:
        citations_text = "\n".join(
            f"- {c['document_name']}, {s['name']}: {s.get('plain_text', '')[:400]}"
            for c in citations[:5] for s in (c.get("sections") or [])[:2]
        )

    original_draft = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant" and "BOZZA DOCUMENTO DIFENSIVO" in msg.get("content", ""):
            original_draft = msg["content"][:3000]
            break

    system = (
        "Sei un avvocato esperto di diritto italiano. "
        f"L'utente vuole approfondire il seguente argomento difensivo: '{topic}'. "
        "Fornisci un'analisi giuridica dettagliata e approfondita di questo specifico argomento, includendo:\n"
        "- Fondamento normativo (articoli di legge applicabili)\n"
        "- Orientamento giurisprudenziale prevalente\n"
        "- Strategia difensiva consigliata\n"
        "- Prove e documenti utili a supporto\n"
        "- Possibili obiezioni della controparte e come controbatterle\n"
        "Usa un linguaggio giuridico formale italiano. "
        "NON citare articoli di legge a memoria — usa SOLO le fonti fornite o scrivi [DA VERIFICARE]."
        + (f"\n\nFonti dal corpus legale:\n{citations_text}" if citations_text else "")
        + (f"\n\nBozza difensiva di riferimento:\n{original_draft}" if original_draft else "")
    )

    return _call_chat(
        [SystemMessage(content=system), HumanMessage(content=f"Approfondisci: {topic}")],
        max_tokens=2000,
    )
