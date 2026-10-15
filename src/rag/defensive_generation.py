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

from . import feature_flags
from .ai_chat import _call_chat
from .verbose_logger import vlog

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
- Stima dei tempi processuali e dei costi (compresa la soccombenza), presentata esplicitamente come indicativa e non vincolante
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

    Oct 2026: the prompt's example topics ("prescrizione del credito, onere della
    prova, mediazione obbligatoria") came back word for word under a pothole
    parere. No examples now, and a topic the draft does not discuss is dropped.
    """
    try:
        raw = _call_chat([
            SystemMessage(content=(
                "Sei un assistente legale. Dal seguente testo difensivo, estrai i 3 argomenti più "
                "importanti da approfondire, con le parole usate nel testo. Restituiscili come etichette "
                "brevi (max 5 parole ciascuna, in italiano, minuscolo). "
                "Restituisci SOLO una lista JSON di stringhe, niente altro."
            )),
            HumanMessage(content=draft[:3000]),
        ], max_tokens=100)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        topics = json.loads(cleaned)
        if isinstance(topics, list):
            return [t for t in topics if isinstance(t, str) and _found_in(t, draft)][:3]
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
        "- Per ogni tesi subordinata, indica l'articolo specifico e perché si applica a questo caso, "
        "solo se quell'articolo e quel contenuto risultano dalle fonti fornite\n"
        "- Per le eccezioni della controparte: anticipale e confutale con argomenti specifici\n"
        "- Stima probabilità di successo con motivazione concreta basata sui fatti, non generica\n"
        "- Indica tempi e costi SOLO come stima orientativa, dicendo esplicitamente che sono "
        "indicativi e non vincolanti (es. 'tempi indicativi: ...', 'costo orientativo: ...')\n\n"
        "**5. Strategia e Opzioni per il Cliente**\n"
        "- Per ogni opzione indica il fondamento normativo specifico (es. 'impugnazione nel termine "
        "di legge, se la posizione processuale del cliente lo consente'). Il rimedio concreto e "
        "l'articolo citato devono essere quelli effettivamente disponibili al ruolo del cliente in "
        "questo procedimento — mai un'opzione riservata alla controparte o a un'altra parte del "
        "procedimento\n"
        "- Il piano operativo deve indicare l'atto concreto da compiere CON il termine specifico "
        "(es. '15 giorni dalla notifica della sentenza motivata') e i documenti esatti da produrre\n"
        "- La raccomandazione deve spiegare PERCHÉ quella strada è preferibile rispetto alle altre "
        "in questo caso specifico\n\n"
        "REGOLE:\n"
        "- Ragiona sui fatti specifici di questo caso, non in astratto\n"
        "- Cita le norme con il numero esatto dell'articolo\n"
        "- Se citi giurisprudenza usa SOLO quella fornita nel corpus o scrivi [DA VERIFICARE]\n"
        "- NON inventare cifre, date o fatti non presenti nel documento\n"
        "- Ogni articolo, pena o fattispecie citata deve essere supportata dal testo delle fonti "
        "fornite: se un articolo non è tra le fonti, o le fonti non ne confermano il contenuto "
        "indicato, scrivi (da verificare) invece di descriverlo a memoria. Non attribuire mai a un "
        "articolo un contenuto, una pena o una fattispecie che il suo testo non contiene.\n"
        "- Le opzioni processuali e i rimedi proposti devono corrispondere al ruolo del cliente nel "
        "procedimento (es. indagato/imputato oppure persona offesa/parte civile): non proporre mai "
        "un rimedio riservato a un'altra parte del procedimento.\n"
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
        "\n- Ogni articolo, pena o fattispecie citata deve essere supportata dal testo delle fonti "
        "fornite: se un articolo pertinente non è tra le fonti, o le fonti non ne confermano il "
        "contenuto indicato, scrivi (da verificare) invece di descriverlo a memoria. Non attribuire "
        "mai a un articolo un contenuto, una pena o una fattispecie che il suo testo non contiene."
        "\n- Le opzioni processuali e i rimedi proposti devono corrispondere al ruolo del cliente nel "
        "procedimento (es. indagato/imputato oppure persona offesa/parte civile): non proporre mai "
        "un rimedio riservato a un'altra parte del procedimento."
        "\n- Indica tempi e costi SOLO come stima orientativa, segnalando esplicitamente che sono "
        "indicativi e non vincolanti — non presentarli come cifre certe."
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
    # Every strategy request gets the reasoned parere (run_case_analysis_pipeline):
    # without a document the chat passes the message as both arguments; with
    # one, the document holds the case and the message is the request. The
    # flow below always produced an analysis too ("ANALISI LEGALE COMPLETA"),
    # with the problems described above run_case_analysis_pipeline; it stays
    # for reference, no longer called.
    request = "" if document_text.strip() == user_message.strip() else user_message
    result = run_case_analysis_pipeline(document_text, session_lang, request=request)
    return {"draft": result["draft"], "proceeding": _safe_defaults(), "citations": result["citations"]}

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


# ---------------------------------------------------------------------------
# Case analysis from facts (no uploaded document): a reasoned parere
# ---------------------------------------------------------------------------
# Oct 2026: a bar-exam style case (pensioner falls in a pothole, defend the
# Comune) went through run_defensive_pipeline with the user's text as the
# "judicial document". No article was "cited by the opposing party", so no
# article text was fetched and the model wrote 2043/2044 from memory (wrongly),
# never reached art. 2051 c.c., filled the intake checklist with invented facts
# and copied the instructions' examples. Here the legal questions are worked out
# first, the articles are read from the database by number, case law is found
# by full-text search (the vector index gives unrelated results for these
# queries), and every article or decision number the answer cites is checked
# against what was retrieved.

_CODE_PREFIXES = {
    "c.c.": "Codice Civile",
    "c.p.c.": "Codice di procedura civile",
    "c.p.": "Codice Penale",
    "c.p.p.": "Codice di procedura penale",
    "c.p.a.": "Codice del processo amministrativo",
    "c.d.s.": "Codice della strada",
}
_CODE_SPELLINGS = [   # longest first, so "c.p.c." is not read as "c.p."; dots optional ("cpp")
    (r"c\.?\s*p\.?\s*c\.?(?!\w)|cod\.\s*proc\.\s*civ\.", "c.p.c."),
    (r"c\.?\s*p\.?\s*p\.?(?!\w)|cod\.\s*proc\.\s*pen\.", "c.p.p."),
    (r"c\.?\s*p\.?\s*a\.?(?!\w)", "c.p.a."),
    (r"c\.?\s*d\.?\s*s\.?(?!\w)|codice della strada", "c.d.s."),
    (r"c\.?\s*c\.?(?!\w)|cod\.\s*civ\.|codice civile", "c.c."),
    (r"c\.?\s*p\.?(?!\w)|cod\.\s*pen\.|codice penale", "c.p."),
]
# Articles a court ruling cites for the substance of a case. Procedure codes
# are left out: every Cassazione ruling cites art. 360 and 380-bis c.p.c.
_SUBSTANTIVE_CODES = {"c.c.", "c.p."}
# The backbone of any civil damages claim: fault, burden of proof,
# prescription, the injured party's contribution.
_CIVIL_DAMAGES_NORMS = [("2043", "c.c."), ("2697", "c.c."), ("2947", "c.c."), ("1227", "c.c.")]
_DAMAGES_RE = re.compile(r"\b(dann[io]|risarciment|risarcitori)", re.IGNORECASE)
# Contract-only damages rules (debtor's liability, default, interest on late
# payment, foreseeable damage): they do not apply to a claim in tort, whose
# damages follow art. 2056 c.c. (1223, 1226, 1227). Runs 9-10 used 1218, 1219
# and 1225 for the pothole claim, arguing "damage limited to what was foreseeable".
# Art. 2056 itself is not added: the choice takes at most 6 articles, and in
# run 11 it pushed out art. 2947 c.c. (prescription).
_CONTRACT_ONLY_NORMS = {("1218", "c.c."), ("1219", "c.c."), ("1224", "c.c."), ("1225", "c.c.")}
# Any sign of a contract between the parties (or of a relationship treated as
# one: hospital care, employment) keeps the contract rules among the candidates.
_CONTRACT_RE = re.compile(
    r"\b(contratt|acquist|vend|compravendit|appalt|locazion|locat|affitt|mutu[oi]\b|fornitur|incaric|"
    r"mandat|abbonament|obbligazion|debitor|creditor|fattur|ospedal|struttura sanitaria|clinic|"
    r"datore di lavoro|lavorator|dipendent)",
    re.IGNORECASE,
)
_ARTICLE_NUM = r"(\d+(?:-?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?(?:\.\d+)?)"
_CITED_ARTICLE_RE = re.compile(
    r"\b(?:art(?:t)?\.?|articol[oi])\s*" + _ARTICLE_NUM
    + r"(?:\s*,?\s*(?:co\.|comma)\s*\d+(?:\s*,\s*n\.\s*\d+)?)?\s*,?\s*(?:del\s+)?("
    + "|".join(p for p, _ in _CODE_SPELLINGS) + r")",
    re.IGNORECASE,
)
_CITED_DECISION_RE = re.compile(r"\b(\d{2,6})\s*/\s*((?:19|20)\d{2})\b")
_LAW_BEFORE_RE = re.compile(
    r"(d\.\s*lgs\.?|d\.\s*l\.|\bl\.|legge|d\.\s*p\.\s*r\.|\bdpr|reg\.|regolamento|dir\.|direttiva)\s*(?:n\.\s*)?$"
)
_UNVERIFIED = " [DA VERIFICARE]"

_ARTICLE_CHARS = 1500
_MAX_RULINGS = 10
# The rulings the writer reads, and how much of each: the passages that cite
# the chosen articles or discuss the case's concepts (the court's reasoning),
# not the passage the search happened to match, which is often the facts.
_PASSAGE_RULINGS = 5
_PASSAGE_CHARS = 2500
_PARERE_TOKENS = 5000
# The case as read by the analysis and the writer (~3,400 tokens): an uploaded
# act can be long, and the 20k-token context also holds sources and the answer.
_FACTS_CHARS = 12000

_CASE_ANALYSIS_SYSTEM = (
    "Sei un avvocato italiano esperto. Leggi il caso e prepara l'impostazione di un parere. "
    "Restituisci SOLO un oggetto JSON con queste chiavi:\n"
    '- "parte_assistita": la parte nel cui interesse va redatto il parere; se il testo non la indica, '
    "la parte che chiede assistenza\n"
    '- "posizione": la sua posizione processuale o sostanziale\n'
    '- "area": il ramo del diritto che regola la pretesa o il fatto, una tra "civile", "penale", '
    '"amministrativo", "lavoro", "tributario", "altro" (una richiesta di risarcimento verso un ente '
    'pubblico è "civile")\n'
    '- "richiesta": in una frase, che cosa chiede il caso\n'
    '- "ricerca_fatti": da 4 a 8 parole concrete che descrivono i fatti materiali (cose, luoghi, '
    "eventi, soggetti: non termini giuridici), con cui cercare sentenze su fatti simili\n"
    '- "questioni": da 3 a 6 questioni giuridiche specifiche del caso, in ordine di importanza: prima il '
    "fondamento della responsabilità o del reato e i suoi presupposti, poi le cause di esclusione o di "
    "attenuazione e le difese possibili, poi la quantificazione; termini, prescrizione e procedibilità "
    'solo se il caso li pone. Ciascuna è {"titolo": stringa, "norme": lista di articoli nel formato '
    '"art. NUMERO SIGLA" con sigla tra c.c., c.p.c., c.p., c.p.p., c.p.a., c.d.s., "parole_chiave": '
    "da 4 a 8 nomi tecnici di istituti giuridici e concetti con cui le sentenze trattano quella "
    'questione, non parole generiche}\n'
    '- "fatti": i fatti del testo uno per uno, senza accorparli, compresi i dettagli che sembrano '
    "secondari: età e condizioni delle persone, abitudini, luoghi, orari e condizioni di visibilità, "
    "chi era presente e chi ha visto che cosa, che cosa risulta da verbali o dichiarazioni, tempi "
    'trascorsi. Ciascuno è {"fatto": stringa, "effetto": "favorevole", "sfavorevole", "favorevole e '
    'sfavorevole" o "neutro" per la parte assistita, "perche": stringa breve}\n'
    "Non inventare fatti. Rispondi SOLO con il JSON."
)

_ARTICLE_CHOICE_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati un caso e alcuni articoli di legge candidati, con il loro "
    "testo. Scegli gli articoli che un parere su questo caso deve usare: quelli che disciplinano "
    "direttamente la vicenda, le difese o le eccezioni, l'onere della prova, i termini rilevanti. Escludi "
    "quelli che riguardano altro, anche se contengono parole simili. Restituisci SOLO una lista JSON di "
    "stringhe, ciascuna identica a un identificativo tra parentesi quadre, in ordine di importanza, al "
    "massimo 6. Escludi gli articoli che non si applicano a questi fatti, ad esempio quelli su reati, "
    "contratti o procedimenti diversi da quelli del caso."
)
# Which codes the keyword search reads, by area: a civil case does not need the
# criminal prescription articles that "prescrizione" also matches.
# The Highway Code is left out: in every test run its matches were noise
# ("auto" found a drink-driving rule in a used-car sale); when it matters, the
# user's text or the rulings name its articles.
_CODE_SEARCH_PREFIXES = {
    "civile": ["Codice Civile", "Codice di procedura civile"],
    "lavoro": ["Codice Civile", "Codice di procedura civile"],
    "penale": ["Codice Penale", "Codice di procedura penale"],
}
_ALL_CODE_PREFIXES = ["Codice Civile", "Codice di procedura civile", "Codice Penale",
                      "Codice di procedura penale"]


def _case_text(facts: str, request: str = "") -> str:
    """The case as the model reads it: the user's request (when the case is an
    uploaded document) and the text, cut to _FACTS_CHARS with a visible mark."""
    text = facts if len(facts) <= _FACTS_CHARS else facts[:_FACTS_CHARS] + "\n[... documento troncato ...]"
    return (f"RICHIESTA DELL'UTENTE: {request}\n\nDOCUMENTO CARICATO:\n{text}" if request else text)


def _analyse_case(facts: str, lang: str, request: str = "") -> Dict[str, Any]:
    """The legal questions, the norms and the role of each fact. Best-effort:
    any failure returns empty lists, and the parere is written from the facts."""
    empty = {"parte_assistita": "", "posizione": "", "area": "", "richiesta": "", "ricerca_fatti": [],
             "questioni": [], "fatti": []}
    try:
        raw = _call_chat([SystemMessage(content=_CASE_ANALYSIS_SYSTEM),
                          HumanMessage(content=_case_text(facts, request))],
                         max_tokens=1500)
        parsed = json.loads(re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip())
    except Exception as exc:
        logger.warning("case analysis failed: %s", exc)
        return empty
    if not isinstance(parsed, dict):
        return empty
    result = {**empty, **{k: v for k, v in parsed.items() if k in empty}}
    result["questioni"] = [q for q in result["questioni"] if isinstance(q, dict)][:6] \
        if isinstance(result["questioni"], list) else []
    for q in result["questioni"]:
        q["norme"] = _as_list(q.get("norme"))
        q["parole_chiave"] = _as_list(q.get("parole_chiave"))
    result["fatti"] = [f for f in result["fatti"] if isinstance(f, dict)][:15] \
        if isinstance(result["fatti"], list) else []
    result["ricerca_fatti"] = _as_list(result["ricerca_fatti"])[:8]
    # A damages claim against a public body is a civil matter; the model keeps
    # filing it as "amministrativo", which widens the code search to every code.
    if result["area"] == "amministrativo" and _DAMAGES_RE.search(facts):
        result["area"] = "civile"
    return result


def _as_list(value) -> List[str]:
    """A list of strings from either a JSON list or one comma-separated string:
    the model returns both from one run to the next, and a string iterated as a
    list searched the database letter by letter."""
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _code_of(text: str) -> Optional[str]:
    for pattern, abbr in _CODE_SPELLINGS:
        if re.fullmatch(pattern, text.strip(), re.IGNORECASE):
            return abbr
    return None


def _article_num(raw: str) -> str:
    """'415bis' / '415-BIS' / '2051.1' -> '415-bis' / '2051' (as the codes name them)."""
    num = raw.lower().split(".")[0]
    return re.sub(r"(\d)-?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)", r"\1-\2", num)


def _refs_in(text: str) -> List[tuple]:
    """(article number, code) for every article cited in a text, in order, repeats kept."""
    refs = []
    for m in _CITED_ARTICLE_RE.finditer(text or ""):
        code = _code_of(m.group(2))
        if code:
            refs.append((_article_num(m.group(1)), code))
    return refs


def _norm_refs(analysis: Dict[str, Any]) -> List[tuple]:
    """(article number, code abbreviation) for every norm the analysis names."""
    refs = []
    for q in analysis.get("questioni", []):
        for norm in q.get("norme") or []:
            if isinstance(norm, str):
                refs.extend(r for r in _refs_in(norm) if r not in refs)
    return refs[:12]


def _cited_by_rulings(ruling_rows: List[Dict[str, Any]]) -> List[tuple]:
    """Substantive articles cited by at least two different rulings, most cited first."""
    docs_citing: Dict[tuple, set] = {}
    for row in ruling_rows:
        doc_id = (row.get("d") or {}).get("id")
        for ref in _refs_in((row.get("s") or {}).get("plain_text", "")):
            if ref[1] in _SUBSTANTIVE_CODES:
                docs_citing.setdefault(ref, set()).add(doc_id)
    return sorted((r for r, docs in docs_citing.items() if len(docs) >= 2), key=lambda r: -len(docs_citing[r]))


def _articles_cited_in_rulings(session, doc_ids: List[str]) -> Dict[tuple, int]:
    """For each substantive article, how many of these rulings cite it anywhere.

    The whole ruling, not just the passage the search matched: a search on the
    facts ("buca stradale, passeggiata, anziano") lands on the passage that tells
    the facts, while art. 2051 c.c. is cited in the reasoning further on.
    """
    ids = [i for i in dict.fromkeys(doc_ids) if i]
    if not ids:
        return {}
    cited_by: Dict[tuple, set] = {}
    for row in session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids RETURN d.id AS id, s.plain_text AS text",
        ids=ids,
    ).data():
        for ref in _refs_in(row.get("text") or ""):
            if ref[1] in _SUBSTANTIVE_CODES:
                cited_by.setdefault(ref, set()).add(row.get("id"))
    return {ref: len(docs) for ref, docs in cited_by.items()}


def _candidate_articles(facts: str, analysis: Dict[str, Any], ruling_rows: List[Dict[str, Any]],
                        code_refs: List[tuple], ruling_counts: Optional[Dict[tuple, int]] = None,
                        similar_counts: Optional[Dict[tuple, int]] = None) -> tuple:
    """(articles to keep whatever happens, candidates for the model to choose from).

    The model knows the legal vocabulary but not reliably the numbers: for the
    pothole case it named art. 50, 1219, 2667 c.c., where the rulings on the same
    issue cite 2051 and 1227; for an injury in a race, art. 61 c.p. instead of
    590. So the candidates come from the database: articles found in the codes by
    the question's keywords, articles the rulings cite, the damages backbone for
    a civil damages claim. The model's own numbers only join the list, and every
    candidate is chosen with its real text in front of the model.
    """
    keep = [r for r in dict.fromkeys(_refs_in(facts)) if r[1] in _CODE_PREFIXES]
    if ruling_counts:
        agreed = sorted((r for r, n in ruling_counts.items() if n >= 2), key=lambda r: -ruling_counts[r])
        # Kept outright when three or more rulings cite it, two of them rulings
        # on similar facts: the model's choice dropped the decisive article from
        # run to run. Only the case's own code (a criminal case got art. 2697
        # c.c.), and the similar-facts condition keeps out what rulings cite in
        # general (art. 416-bis c.p. joined a racing accident case).
        own_code = {"penale": "c.p."}.get(analysis.get("area"), "c.c.")
        similar_counts = similar_counts or {}
        keep += [r for r in agreed[:3] if ruling_counts[r] >= 3 and similar_counts.get(r, 0) >= 2
                 and r[1] == own_code and r not in keep]
    else:
        agreed = _cited_by_rulings(ruling_rows)
    candidates: List[tuple] = []
    # What the rulings agree on first: they were decided on these questions.
    for ref in agreed[:6] + code_refs:
        if ref not in candidates and ref not in keep:
            candidates.append(ref)
    if analysis.get("area") != "penale" and _DAMAGES_RE.search(facts):
        candidates += [r for r in _CIVIL_DAMAGES_NORMS if r not in candidates and r not in keep]
    candidates += [r for r in _norm_refs(analysis) if r not in candidates and r not in keep]
    if not _CONTRACT_RE.search(facts):   # a tort; articles the user's text names stay in `keep`
        candidates = [r for r in candidates if r not in _CONTRACT_ONLY_NORMS]
    return keep, candidates[:24]


def _search_code_articles(session, analysis: Dict[str, Any]) -> List[tuple]:
    """Articles of the main codes whose text matches each question's keywords."""
    abbr_of = {prefix: abbr for abbr, prefix in _CODE_PREFIXES.items()}
    refs: List[tuple] = []
    for q in analysis.get("questioni", []):
        words = [w for w in (q.get("parole_chiave") or []) if isinstance(w, str)]
        terms = _lucene_query(" ".join([q.get("titolo") or ""] + words))
        if not terms:
            continue
        try:
            found = session.run(
                "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
                "MATCH (d:Document)-[:CONTAINS]->(node) "
                "WHERE d.document_type = 'primary' AND any(p IN $prefixes WHERE d.name STARTS WITH p) "
                "RETURN d.name AS doc, node.name AS sec, score ORDER BY score DESC LIMIT 10",
                t=terms, prefixes=_CODE_SEARCH_PREFIXES.get(analysis.get("area"), _ALL_CODE_PREFIXES),
            ).data()
        except Exception as exc:
            logger.warning("code search failed for %r: %s", terms, exc)
            continue
        taken = 0
        for row in found:
            prefix = next((p for p in sorted(abbr_of, key=len, reverse=True)
                           if (row.get("doc") or "").startswith(p)), None)
            m = re.match(r"^(\d+(?:-[a-z]+)?)(?:[._]|$)", (row.get("sec") or "").strip(), re.IGNORECASE)
            if not prefix or not m:
                continue
            ref = (_article_num(m.group(1)), abbr_of[prefix])
            if ref not in refs:
                refs.append(ref)
                taken += 1
            if taken == 4:
                break
    return refs


def _choose_articles(facts: str, candidate_rows: List[Dict[str, Any]]) -> Optional[List[tuple]]:
    """The candidates the model judges relevant, reading their text; None if the call fails."""
    texts: Dict[tuple, List[str]] = {}
    for row in candidate_rows:
        texts.setdefault(row["_article"], []).append(" ".join(((row.get("s") or {}).get("plain_text") or "").split()))
    if not texts:
        return []
    listing = "\n".join(f"[art. {n} {c}] {' '.join(t)[:400]}" for (n, c), t in texts.items())
    try:
        raw = _call_chat([SystemMessage(content=_ARTICLE_CHOICE_SYSTEM),
                          HumanMessage(content=f"CASO:\n{facts[:6000]}\n\nARTICOLI CANDIDATI:\n{listing}")],
                         max_tokens=200)
        chosen = json.loads(re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip())
    except Exception as exc:
        logger.warning("article choice failed: %s", exc)
        return None
    if not isinstance(chosen, list):
        return None
    refs = []
    for item in chosen:
        for ref in _refs_in(item if isinstance(item, str) else ""):
            if ref in texts and ref not in refs:
                refs.append(ref)
    return refs[:6]


def _lucene_terms(text: str) -> str:
    return " ".join(re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", text).split())


def _lucene_query(text: str) -> str:
    """Search terms plus a word-stem form of each longer word ("vizio" -> "vizi*"):
    the full-text index does not reduce words to their stem, so the model's
    "vizio nascosto" never matched art. 1495 c.c., which says "denunzia i vizi"."""
    out = []
    for word in _lucene_terms(text).split():
        word = word.strip(",;.")
        if not word:
            continue
        out.append(word)
        if len(word) >= 5 and word.isalpha() and word.lower() not in _STOPWORDS:
            out.append(word[:-1].lower() + "*")
    return " ".join(dict.fromkeys(out))


_STOPWORDS = {"della", "delle", "dello", "degli", "nella", "nelle", "nello", "negli", "sulla", "sulle",
              "dalla", "dalle", "alla", "alle", "quale", "quali", "come", "dopo", "prima", "senza",
              "anche", "sono", "essere", "stato", "stata", "questo", "questa", "verso", "contro"}


def _rows_to_sources(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .answer_processing import _extract_citations
    return _extract_citations(rows)


def _fetch_articles(session, refs: List[tuple]) -> List[Dict[str, Any]]:
    """The text of each article, read from its code by number."""
    rows = []
    for num, abbr in refs:
        prefix = _CODE_PREFIXES.get(abbr)
        if not prefix:
            continue
        found = session.run(
            "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
            "WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
            "AND (s.name = $n + '.0.0' OR s.name STARTS WITH $n + '.') "
            "RETURN d, s ORDER BY s.name LIMIT 6",
            prefix=prefix, n=num,
        ).data()
        for row in found:
            row["_article"] = (num, abbr)
        rows.extend(found)
    return rows


def _search_case_law(session, analysis: Dict[str, Any], numbers: Optional[List[str]] = None,
                     exclude_docs: Optional[set] = None) -> List[Dict[str, Any]]:
    """Rulings for each legal question, by full-text search on its keywords
    (plus `numbers`: articles already verified, never the model's guesses, since
    a wrong number pulls in rulings on an unrelated article); at most two per
    question, one passage per ruling."""
    rows, seen_docs = [], set(exclude_docs or ())
    # The commentary in the corpus is criminal law (Codice Penale Commentato):
    # only criminal cases search it.
    doc_types = ["interpretation", "special"] if analysis.get("area") == "penale" else ["interpretation"]
    for index, q in enumerate(analysis.get("questioni", [])):
        words = [w for w in (q.get("parole_chiave") or []) if isinstance(w, str)]
        terms = _lucene_query(" ".join([q.get("titolo") or ""] + words + list(numbers or [])))
        if not terms:
            continue
        try:
            found = session.run(
                "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
                "MATCH (d:Document)-[:CONTAINS]->(node) "
                "WHERE d.document_type IN $types "
                "RETURN d, node AS s, score ORDER BY score DESC LIMIT 12",
                t=terms, types=doc_types,
            ).data()
        except Exception as exc:
            logger.warning("case-law search failed for %r: %s", terms, exc)
            continue
        taken = 0
        for row in found:
            doc_id = (row.get("d") or {}).get("id")
            if not doc_id or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            row["_question"] = index
            rows.append(row)
            taken += 1
            if taken == 2 or len(rows) >= _MAX_RULINGS:
                break
        if len(rows) >= _MAX_RULINGS:
            break
    return rows


def _search_similar_facts(session, analysis: Dict[str, Any], exclude_docs: set) -> List[Dict[str, Any]]:
    """Rulings on similar facts, found by the facts' own words ("buca manto
    stradale caduta pedone comune"), not by legal labels: the model may frame a
    pothole claim as "responsabilità amministrativa", but rulings on falls in a
    pothole all cite art. 2051 c.c."""
    terms = _lucene_query(" ".join(analysis.get("ricerca_fatti") or []))
    if not terms:
        return []
    doc_types = ["interpretation", "special"] if analysis.get("area") == "penale" else ["interpretation"]
    try:
        found = session.run(
            "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) "
            "WHERE d.document_type IN $types "
            "RETURN d, node AS s, score ORDER BY score DESC LIMIT 12",
            t=terms, types=doc_types,
        ).data()
    except Exception as exc:
        logger.warning("similar-facts search failed for %r: %s", terms, exc)
        return []
    rows, seen = [], set(exclude_docs)
    for row in found:
        doc_id = (row.get("d") or {}).get("id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            rows.append(row)
        if len(rows) == 4:
            break
    return rows


def _ruling_label(row: Dict[str, Any]) -> str:
    return re.sub(r"^\s*\[[A-Z]+\]\s*", "", (row.get("d") or {}).get("name") or "")


def _source_block(rows: List[Dict[str, Any]], chars: int) -> str:
    """One labelled block per article (its sections joined) or per ruling (its
    chosen passage, else the matched section), each cut to `chars`."""
    grouped: Dict[str, List[str]] = {}
    for row in rows:
        _s = row.get("s") or {}
        # abstract fallback: special/dottrina books (Commentato, Fiscalita) are
        # routinely "abstract-only" — plain_text empty, content in abstract
        # (same field the ingestion step embeds from — see BE's "abstract-only
        # sections" commit). plain_text-only here silently dropped their text.
        text = row.get("_passage") or " ".join((_s.get("plain_text") or _s.get("abstract") or "").split())
        if not text:
            continue
        label = (f"art. {row['_article'][0]} {row['_article'][1]}" if row.get("_article")
                 else _ruling_label(row))
        grouped.setdefault(label, []).append(text)
    return "\n\n".join(f"[{label}]\n{' '.join(texts)[:chars]}" for label, texts in grouped.items())


def _reading_order(similar: List[Dict[str, Any]], second: List[Dict[str, Any]],
                   first: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rulings in the order the writer reads them, without repeats: two on
    similar facts, then one for each legal question, then the rest. In search
    order the five read in depth would be the four similar-facts rulings and one
    on the first question, none on the defences."""
    per_question, questions = [], set()
    for row in second + first:
        if row.get("_question") not in questions:
            questions.add(row.get("_question"))
            per_question.append(row)
    ordered, seen = [], set()
    for row in similar[:2] + per_question + similar[2:] + second + first:
        doc_id = (row.get("d") or {}).get("id")
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(row)
    return ordered[:_MAX_RULINGS]


def _ruling_sections(session, doc_ids: List[str]) -> Dict[str, List[str]]:
    """Every section of each ruling, in reading order ("2_3" before "10_1")."""
    ids = [i for i in dict.fromkeys(doc_ids) if i]
    if not ids:
        return {}
    parts: Dict[str, List[tuple]] = {}
    rows = session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
        "RETURN d.id AS id, s.name AS name, s.plain_text AS text, s.abstract AS abstract",
        ids=ids,
    ).data()
    for i, row in enumerate(rows):
        # abstract fallback, same reason as _source_block: special/dottrina
        # sections are often abstract-only.
        text = " ".join((row.get("text") or row.get("abstract") or "").split())
        if text:
            order = tuple(int(n) for n in re.findall(r"\d+", row.get("name") or ""))
            parts.setdefault(row.get("id"), []).append((order, i, text))
    return {doc: [text for _, _, text in sorted(found)] for doc, found in parts.items()}


def _passage_terms(analysis: Dict[str, Any]) -> List[tuple]:
    """The case's concepts and fact words, each as the stems of its longer words
    ("custodia della strada" -> ("custodi", "strad")): a section discusses a
    concept when it contains all of them, in any order and form ("strade in
    custodia"). Four-letter words stay whole: "caso" cut to "cas" matches "Cassazione"."""
    phrases = list(analysis.get("ricerca_fatti") or [])
    for q in analysis.get("questioni", []):
        phrases += q.get("parole_chiave") or []
    terms = []
    for phrase in phrases:
        words = [w for w in re.findall(r"\w+", phrase.lower()) if len(w) >= 4 and w not in _STOPWORDS]
        term = tuple(w[:-1] if len(w) >= 5 else w for w in words)
        if term and term not in terms:
            terms.append(term)
    return terms


def _has_term(low: str, term: tuple) -> bool:
    return all(stem in low for stem in term)


def _best_passage(sections: List[str], selected: List[tuple], terms: List[tuple], chars: int) -> str:
    """The sections of one ruling that cite the chosen articles (worth three
    concepts each) or discuss the case's concepts, best first up to `chars`,
    then put back in reading order. Empty when no section scores."""
    wanted = set(selected)

    def score(text: str) -> int:
        low = text.lower()
        return 3 * len(set(_refs_in(text)) & wanted) + sum(1 for t in terms if _has_term(low, t))

    ranked = sorted(((score(t), i) for i, t in enumerate(sections)), key=lambda x: (-x[0], x[1]))
    picked, used = [], 0
    for points, i in ranked:
        room = chars - used
        if not points or room < 300:
            break
        text = sections[i] if len(sections[i]) <= room else _window(sections[i], wanted, terms, room)
        picked.append((i, text))
        used += len(text) + 5
    return " […] ".join(text for _, text in sorted(picked))


def _window(text: str, wanted: set, terms: List[tuple], chars: int) -> str:
    """`chars` of a long section, from the sentence before its first mention
    of a chosen article or a concept."""
    low = text.lower()
    hits = [m.start() for m in _CITED_ARTICLE_RE.finditer(text)
            if (_article_num(m.group(1)), _code_of(m.group(2))) in wanted]
    hits += [low.find(t[0]) for t in terms if _has_term(low, t)]
    start = max(0, min(hits, default=0) - chars // 5)
    if start:
        dot = text.rfind(". ", 0, start)
        start = dot + 2 if dot >= 0 and start - dot < 300 else start
    return ("… " if start else "") + text[start:start + chars].strip()


_PENAL_CODES = {"c.p.", "c.p.p."}
_CIVIL_CODES = {"c.c.", "c.p.c."}


def _other_branch(texts: List[str], area: str) -> bool:
    """A civil ruling for a criminal case, or the reverse, judged by the codes it
    cites most. Run 9, Oct 2026: the search for a race crash found a civil
    ruling on a pothole, and the parere cited it four times on criminal fault."""
    codes = [code for text in texts for _, code in _refs_in(text)]
    penal = sum(code in _PENAL_CODES for code in codes)
    civil = sum(code in _CIVIL_CODES for code in codes)
    return civil > penal if area == "penale" else penal > civil


def _ruling_passages(session, ruling_rows: List[Dict[str, Any]], selected: List[tuple],
                     analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The first _PASSAGE_RULINGS rulings of the case's own branch (civil or
    criminal), each with its best passage as `_passage` (the matched section
    when no section scores)."""
    try:
        sections = _ruling_sections(session, [(r.get("d") or {}).get("id") for r in ruling_rows])
    except Exception as exc:
        logger.warning("reading the rulings failed: %s", exc)
        sections = {}
    terms = _passage_terms(analysis)
    out = []
    for row in ruling_rows:
        texts = sections.get((row.get("d") or {}).get("id")) or [(row.get("s") or {}).get("plain_text") or ""]
        if _other_branch(texts, analysis.get("area") or ""):
            continue
        passage = _best_passage(sections.get((row.get("d") or {}).get("id")) or [], selected, terms,
                                _PASSAGE_CHARS)
        out.append({**row, "_passage": passage} if passage else row)
        if len(out) == _PASSAGE_RULINGS:
            break
    return out


_PRINCIPLES_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati un caso, le sue questioni giuridiche e alcuni passi di "
    "pronunce, ciascuno con un identificativo tra parentesi quadre ([S1], [S2]...). Per ogni questione "
    "elenca i principi di diritto che le pronunce affermano e che servono a decidere il caso: che cosa "
    "deve provare ciascuna parte, quando la responsabilità è esclusa o ridotta, quali circostanze contano "
    "e come il giudice le valuta. Una riga per principio, nella forma \"- principio [Sn]\", con "
    "l'identificativo del passo da cui lo ricavi, formulato con le parole del passo. Riporta solo ciò che "
    "il giudice afferma, non le tesi delle parti né i fatti di quel giudizio, e nulla che i passi non "
    "contengano. Usa solo i passi che trattano lo stesso istituto giuridico delle questioni del caso; "
    "ignora gli altri anche se usano parole simili. Salta le questioni che nessun passo tratta. Formato:\n"
    "QUESTIONE 1: titolo\n- principio [Sn]\n"
    "Rispondi solo con l'elenco."
)


_ADMINISTRATIVE_LIABILITY_RE = re.compile(r"responsabilit\S*\s+amministrativ\w*", re.IGNORECASE)


def _civil_title(text: str, area: str) -> str:
    """In a civil case "responsabilità amministrativa" (public employees before
    the Corte dei conti) becomes "responsabilità civile della P.A.": the analysis
    keeps using it for damages claims against a Comune, and run 10 copied it
    into the parere as a heading."""
    if area != "civile" or re.search(r"erarial|corte dei conti|dipendent", text, re.IGNORECASE):
        return text
    return _ADMINISTRATIVE_LIABILITY_RE.sub(
        lambda m: ("R" if m.group(0)[0].isupper() else "r") + "esponsabilità civile della P.A.", text)


def _issue_titles(analysis: Dict[str, Any]) -> str:
    """The legal questions, numbered, as the principles step and the writer read them."""
    return "\n".join(f"{i}. {_civil_title(q.get('titolo') or '', analysis.get('area') or '')}"
                     for i, q in enumerate(analysis.get("questioni", []), 1))


def _found_in(principle: str, passage: str) -> bool:
    """Most of the principle's longer words (cut to a 7-letter stem, so
    "prevedibilità" matches "prevedibile") appear in the passage it cites. Run 9
    attributed principles to rulings that do not state them; those are dropped."""
    stems = {w[:min(len(w) - 1, 7)] for w in re.findall(r"\w+", principle.lower()) if len(w) >= 6}
    low = passage.lower()
    return not stems or sum(stem in low for stem in stems) >= 0.6 * len(stems)


def _extract_principles(facts: str, analysis: Dict[str, Any], ruling_rows: List[Dict[str, Any]],
                        request: str = "") -> str:
    """The legal principles the rulings state, by question, each with the
    ruling's name: the writer used the raw passages only for loose paraphrase.
    Lines without a valid passage id are dropped; empty on any failure."""
    rows = [r for r in ruling_rows if r.get("_passage") or (r.get("s") or {}).get("plain_text")]
    if not rows:
        return ""
    labels = [_ruling_label(r) for r in rows]
    texts = [r.get("_passage") or " ".join(r["s"]["plain_text"].split())[:_PASSAGE_CHARS] for r in rows]
    passages = "\n\n".join(f"[S{i}] {label}\n{text}" for i, (label, text) in enumerate(zip(labels, texts), 1))
    issues = _issue_titles(analysis)
    try:
        raw = _call_chat([SystemMessage(content=_PRINCIPLES_SYSTEM),
                          HumanMessage(content=f"CASO:\n{_case_text(facts, request)[:3000]}\n\n"
                                               f"QUESTIONI:\n{issues or '(non indicate)'}\n\n"
                                               f"PASSI DELLE PRONUNCE:\n{passages}")],
                         max_tokens=1200)
    except Exception as exc:
        logger.warning("principles extraction failed: %s", exc)
        return ""
    lines = []
    for line in raw.splitlines():
        line = line.strip().strip("*#").strip()
        if re.match(r"questione\b", line, re.IGNORECASE):
            lines.append(_civil_title(line.replace("**", ""), analysis.get("area") or ""))
            continue
        ids = [int(n) for group in re.findall(r"\[([^\]]*)\]", line) for n in re.findall(r"S(\d+)", group)]
        ids = [n for n in dict.fromkeys(ids) if 1 <= n <= len(labels)]
        body = re.sub(r"\s*\[[^\]]*S\d+[^\]]*\]", "", line).lstrip("-•* ").strip()
        body = re.sub(r"^\d+[.)]\s*", "", body)
        ids = [n for n in ids if _found_in(body, texts[n - 1])]
        if ids and body:
            lines.append(f"- {body} [{'; '.join(labels[n - 1] for n in ids)}]")
    # A question heading stays only with principles under it.
    return "\n".join(line for i, line in enumerate(lines)
                     if line.startswith("- ") or (i + 1 < len(lines) and lines[i + 1].startswith("- ")))


def _mark_unverified(text: str, articles: List[tuple], sources_text: str) -> str:
    """Flag article and decision numbers the retrieved sources do not contain,
    the first time each appears."""
    known = {(n, a) for n, a in articles}
    flagged, out, pos = set(), [], 0
    for m in _CITED_ARTICLE_RE.finditer(text):
        ref = (_article_num(m.group(1)), _code_of(m.group(2)))
        if ref[1] in _CODE_PREFIXES and ref not in known and ref not in flagged:
            flagged.add(ref)
            out.append(text[pos:m.end()] + _UNVERIFIED)
            pos = m.end()
    text = "".join(out) + text[pos:]

    compact = re.sub(r"\s+", " ", sources_text)
    out, pos = [], 0
    for m in _CITED_DECISION_RE.finditer(text):
        # Only numbers introduced as decisions ("Cass. n.", "ord.", "sent."),
        # not laws ("d.lgs. 28/2010", "legge n. 128/2001").
        before = text[max(0, m.start() - 30):m.start()].lower()
        if _LAW_BEFORE_RE.search(before) or not re.search(r"cass|sent|ord|cost|sez|n\.", before):
            continue
        num, year = m.group(1), m.group(2)
        if (num, year) in flagged:
            continue
        if re.search(rf"\b{num}\s*(?:/|-|\s+del\s+){year}\b", compact):
            continue
        flagged.add((num, year))
        out.append(text[pos:m.end()] + _UNVERIFIED)
        pos = m.end()
    return "".join(out) + text[pos:]


# Round 3, item 4 (STRATEGY_DRAFT_RULES_ENABLED, default false): a lawyer's
# review of the Sempronio draft. Arguments restated the charge instead of
# defending; a reply contradicted the facts (a "late rescue" when the rescue was
# immediate) and gave the prosecutor a thesis the charge excludes; the
# procedural section was placeholders; mental element, causation, alternative
# qualifications, mitigating circumstances and the sentence range were missing;
# a favourable fact was worded as an admission. And a contract-under-threat
# draft gave art. 1449 c.c. (rescission) as the term for annulment (1442).
# Rules, not case law; articles named here that the sources lack are marked
# [DA VERIFICARE] by _mark_unverified like any other.
_PROCEDURAL_SECTION = (
    "**5. Profili processuali e operativi**: solo quelli pertinenti al caso, tra prescrizione o "
    "decadenza, condizioni di procedibilità, competenza, attività istruttorie, terzi da coinvolgere, "
    "possibilità di definizione stragiudiziale.\n"
)
_PROCEDURAL_SECTION_CONCRETE = (
    "**5. Profili processuali e operativi**: concreti, mai segnaposto. Nei casi penali: l'autorità "
    "giudiziaria competente; la prescrizione del reato, calcolata dalla pena dell'articolo in NORME; la "
    "procedibilità (d'ufficio o a querela); i riti alternativi praticabili; le misure cautelari possibili; "
    "la costituzione di parte civile. Nei casi civili: il giudice competente; la prescrizione o la "
    "decadenza, dal testo in NORME; le condizioni di procedibilità; le prove e le consulenze da chiedere; "
    "la possibilità di definizione stragiudiziale. Ogni punto si ricava dalle fonti fornite oppure è "
    "seguito da \"(da verificare)\".\n"
)
_DRAFT_RULES = (
    "- Ogni articolo, pena o termine che citi deve risultare dal testo in NORME o GIURISPRUDENZA; "
    "altrimenti scrivi accanto \"(da verificare)\". Non attribuire mai a un articolo un contenuto, una "
    "pena o un termine che il suo testo non riporta.\n"
    "- Gli argomenti della sezione 3 sono difensivi: ciascuno esclude, riduce o riqualifica la "
    "responsabilità o la pretesa contro la parte assistita (manca un elemento, opera una causa di "
    "esclusione, si applica una qualificazione meno grave, ricorre un'attenuante); non riformulare "
    "l'accusa o la pretesa avversaria e non limitarti a descrivere le circostanze.\n"
    "- Le repliche attribuite alla controparte devono essere coerenti con i fatti del caso e con la sua "
    "posizione: non attribuirle circostanze contrarie ai fatti né una tesi esclusa dalla norma che essa "
    "stessa invoca.\n"
    "- Non presentare un fatto favorevole alla parte assistita come un'ammissione: un aiuto prestato, "
    "delle scuse o un pagamento non sono un riconoscimento di responsabilità.\n"
    "- Nei casi penali considera, quando le fonti lo consentono: le diverse letture dell'elemento "
    "soggettivo; il nesso causale e le cause sopravvenute (art. 41, comma 2, c.p.), con la prova tecnica "
    "necessaria, ad esempio una consulenza medico-legale; le qualificazioni alternative, come tesi "
    "principale o subordinata; le circostanze attenuanti (artt. 62 e 62-bis c.p.); una forbice di pena "
    "realistica, calcolata dalle norme in NORME.\n"
)


def _parere_system(analysis: Dict[str, Any], lang: str) -> str:
    from .language import language_display_name
    party = analysis.get("parte_assistita") or "la parte assistita"
    position = f" ({analysis['posizione']})" if analysis.get("posizione") else ""
    draft_rules = feature_flags.enabled("STRATEGY_DRAFT_RULES_ENABLED")
    return (
        "Sei un avvocato italiano con lunga esperienza di contenzioso. Redigi un PARERE MOTIVATO "
        f"nell'interesse di {party}{position}, completo e approfondito come quello di un avvocato esperto "
        "per il proprio cliente.\n\n"
        "STRUTTURA (titoli numerati in grassetto):\n"
        "**1. Inquadramento**: la qualificazione giuridica della vicenda o della pretesa (la norma "
        "principale e quelle subordinate o alternative, e perché) e una valutazione di OGNI fatto del "
        "caso: se è favorevole, sfavorevole o neutro per la parte assistita, e perché. Alcuni fatti sono "
        "insieme favorevoli e sfavorevoli: dillo.\n"
        "**2. Quadro normativo e giurisprudenziale**: per ogni norma pertinente, cosa stabilisce e come si "
        "ripartisce l'onere della prova; i principi affermati dalla giurisprudenza, ciascuno con la "
        "pronuncia che lo afferma, e che cosa comportano per questo caso.\n"
        f"**3. Argomenti a favore di {party}**: in ordine di forza, ciascuno con un sottotitolo. Per "
        "ciascuno: i fatti del caso su cui si fonda, la norma e i principi giurisprudenziali che lo "
        "sostengono, le prove da acquisire, la probabile replica della controparte e come superarla.\n"
        "**4. Punti deboli e rischio**: i fatti e gli argomenti sfavorevoli valutati con franchezza, e una "
        "stima motivata del rischio.\n"
        + (_PROCEDURAL_SECTION_CONCRETE if draft_rules else _PROCEDURAL_SECTION)
        + "**6. Conclusioni**: gli argomenti in ordine di forza e la condotta consigliata.\n\n"
        "REGOLE:\n"
        "- Usa ogni fatto elencato in ANALISI DEI FATTI e spiega perché aiuta o danneggia la parte "
        "assistita. Non aggiungere fatti, date, importi o atti che il caso non contiene: se un dato "
        "manca, indica che cosa va verificato o acquisito.\n"
        "- Norme: cita ogni articolo di NORME con il numero e il codice indicati tra parentesi quadre "
        "prima del suo testo, mai con un altro numero. Riporta tra virgolette solo testi presenti in "
        "NORME; per norme non presenti descrivine il contenuto senza virgolette.\n"
        "- Giurisprudenza: cita solo le pronunce presenti in GIURISPRUDENZA, con l'identificativo "
        "indicato tra parentesi quadre; senza fonte puoi richiamare un orientamento solo in termini "
        "generali, senza numeri né date. I PRINCIPI DALLA GIURISPRUDENZA sono ricavati da quei passi: "
        "fondaci il ragionamento e applica ciascuno ai fatti del caso. Riporta tra virgolette solo frasi "
        "presenti nei passi di GIURISPRUDENZA.\n"
        "- Ogni affermazione giuridica va collegata a un fatto del caso: niente considerazioni astratte.\n"
        "- Nella valutazione dei fatti considera anche le circostanze che il testo indica di passaggio "
        "(abitudini, orari, il motivo per cui una persona si trovava sul posto, chi è arrivato dopo il "
        "fatto e che cosa ha potuto vedere): spesso sono decisive.\n"
        "- Usa solo le norme e le pronunce pertinenti al caso; quelle non pertinenti ignorale, senza "
        "elencarle né commentarle.\n"
        "- Prescrizione, decadenza e altri termini sono un argomento solo se il caso indica le date "
        "necessarie; altrimenti indicali solo come verifica da fare.\n"
        "- Se il caso nomina un atto o una fase processuale in corso (un avviso, una notifica, un "
        "termine), spiega che cosa consente di fare alla parte assistita ed entro quando, in base al "
        "testo in NORME: è spesso la prima cosa da decidere.\n"
        "- Proponi solo rimedi, atti e iniziative che la legge consente alla parte assistita nella sua "
        "posizione nel procedimento (per esempio indagato o persona offesa, attore o convenuto): mai "
        "un'iniziativa riservata a un'altra parte.\n"
        "- La responsabilità civile di un ente pubblico verso un privato per un danno non è "
        "'responsabilità amministrativa' (che riguarda i dipendenti pubblici davanti alla Corte dei conti).\n"
        "- Prosa argomentata in paragrafi: non ripetere per ogni argomento uno schema a voci (fatti, norma, "
        "giurisprudenza...); elenchi puntati solo per prove da acquisire e attività da compiere.\n"
        + (_DRAFT_RULES if draft_rules else "")
        + "- Lunghezza: indicativamente 1.500-2.500 parole.\n"
        f"- Scrivi in {language_display_name(lang)}."
    )


def _parere_human(facts: str, analysis: Dict[str, Any], articles: str, rulings: str, request: str = "",
                  principles: str = "") -> str:
    facts_list = "\n".join(
        f"- {f.get('fatto', '')} ({f.get('effetto', 'neutro')}: {f.get('perche', '')})"
        for f in analysis.get("fatti", []) if f.get("fatto")
    )
    # Titles only: the analysis' own article numbers are unreliable, and given
    # here the writer used them as labels for the verified texts in NORME.
    issues = _issue_titles(analysis)
    return (
        f"CASO:\n{_case_text(facts, request)}\n\n"
        + (f"ANALISI DEI FATTI:\n{facts_list}\n\n" if facts_list else "")
        + (f"QUESTIONI GIURIDICHE:\n{issues}\n\n" if issues else "")
        + f"NORME (testo dalla banca dati):\n{articles or '(nessun testo recuperato)'}\n\n"
        + (f"PRINCIPI DALLA GIURISPRUDENZA (per questione):\n{principles}\n\n" if principles else "")
        + f"GIURISPRUDENZA (passi dalla banca dati):\n{rulings or '(nessuna pronuncia recuperata)'}\n\n"
        + "Redigi ora il parere."
    )


def _select_articles(session, driver, database: str, facts: str, request: str,
                     analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The articles a parere is written from, and how they were picked:
    rulings on the questions and on similar facts, the candidates (articles
    the user's text names, articles the rulings agree on, code search,
    damages backbone, the model's own numbers), the model's choice among them
    read with their real text, and the articles the chosen ones cross-refer to.

    Shared by run_case_analysis_pipeline and scripts/check_applicability.py, so
    the diagnosis sees exactly what the pipeline does.
    """
    from .nodes.retrieval import broaden_article_context

    first_rulings = _search_case_law(session, analysis)
    similar = _search_similar_facts(session, analysis, {(r.get("d") or {}).get("id") for r in first_rulings})
    counts = _articles_cited_in_rulings(
        session, [(r.get("d") or {}).get("id") for r in (similar + first_rulings)[:10]])
    similar_counts = _articles_cited_in_rulings(session, [(r.get("d") or {}).get("id") for r in similar])
    case = _case_text(facts, request)
    keep, candidates = _candidate_articles(case, analysis, similar + first_rulings,
                                           _search_code_articles(session, analysis), counts, similar_counts)
    candidate_rows = _fetch_articles(session, keep + candidates)
    chosen = _choose_articles(case, [r for r in candidate_rows if r["_article"] not in keep])
    if chosen is None:   # the choice failed: what the rulings cite, and the damages backbone
        backbone = set(_cited_by_rulings(similar + first_rulings)) | set(_CIVIL_DAMAGES_NORMS)
        chosen = [r for r in candidates if r in backbone][:8]
    selected = keep + [r for r in chosen if r not in keep]
    article_rows = sorted((r for r in candidate_rows if r["_article"] in selected),
                          key=lambda r: selected.index(r["_article"]))
    # Cross-references within the selected articles' own text (e.g. 584
    # citing 581/582) — the same deterministic, grounded expansion
    # article_router's "fired" path gets from broaden_article_context;
    # this pipeline never goes through that graph path (it does its own
    # retrieval), so without this call those articles were never added.
    # Reuses the function as-is: it keys each row's own d.id as the
    # cross-reference scope, so an empty law_hint here is correct, not
    # a shortcut. Tags added rows _source=CROSSREF_SOURCE, same as the
    # graph path.
    article_rows = broaden_article_context(
        {"raw_result": article_rows, "law_hint_doc_id": "", "user_id": "", "tenant_id": ""},
        driver, database,
    ).get("raw_result", article_rows)
    # broaden_article_context's added rows carry d/s but no `_article`
    # tuple (that tag is this file's own convention, set by
    # _fetch_articles). Without it: a bare row["_article"] below
    # crashes the whole pipeline on the first query that adds a
    # crossref row, and _mark_unverified would wrongly flag a cited
    # 581/582 as "[DA VERIFICARE]" despite being DB-sourced.
    abbr_of = {prefix: abbr for abbr, prefix in _CODE_PREFIXES.items()}
    for row in article_rows:
        if "_article" not in row:
            d_name = (row.get("d") or {}).get("name") or ""
            abbr = next((a for p, a in abbr_of.items() if d_name.startswith(p)), None)
            if abbr:
                row["_article"] = (_article_num((row.get("s") or {}).get("name") or ""), abbr)
    # Issue spotting (round 3, item 2, ISSUE_SPOTTING_ENABLED): the case
    # analysis above names norms only as "art. N" + one of six code
    # abbreviations, so laws outside those codes (Codice del Consumo, L.
    # 392/1978) can never be proposed. The same step as the chat's
    # issue_spotting node adds them, resolved against the real document names
    # and already through the applicability check.
    spotted = []
    if feature_flags.enabled("ISSUE_SPOTTING_ENABLED"):
        try:
            from .doc_lookup import _fetch_doc_names
            from .issue_spotting import spotted_rows
            spotted = spotted_rows(session, _fetch_doc_names(driver, database), case, article_rows)
        except Exception as exc:
            logger.warning("issue spotting in the parere failed, nothing added: %s", exc)
        article_rows = article_rows + spotted
    return {"first_rulings": first_rulings, "similar": similar, "keep": keep,
            "named": [r for r in keep if r in _refs_in(case)], "candidates": candidates,
            "chosen": chosen, "selected": selected, "article_rows": article_rows, "spotted": spotted}


# Round 3, item 3. 6db1125's check asked whether an article "applies to the
# facts": for Sempronio it dropped 589/590, the alternative charges a defence
# argues for, and kept 116 (concorso anomalo, which needs several agents), which
# was exempt because the rulings agreed on it. Now only the articles the user's
# own text names are exempt, an alternative qualification counts as applicable,
# and an article is dropped only when the model names the requirement in its
# text and the fact of the case that contradicts it. On production (15 Oct) the
# model kept 116 as an "alternativa" with an invented reason, so "alternativa" is
# only for articles that describe an offence or a claim the same facts could
# meet, and the model checks the text's requirements (who and how many act,
# conduct, event, relationship) against the facts before choosing.
_APPLICABILITY_SYSTEM = (
    "Sei un giurista. Per ogni articolo elencato leggi il suo testo reale e confrontalo con i fatti del "
    "caso. Prima di classificarlo individua i requisiti che il testo pone (chi agisce e quanti soggetti "
    "devono esserci, quale condotta, quale evento, quale rapporto tra le parti) e verifica ciascuno nei "
    "fatti. Classifica ciascun articolo:\n"
    "- \"applicabile\": disciplina i fatti, anche solo in parte, o serve alla difesa o alla tesi della "
    "parte assistita;\n"
    "- \"alternativa\": descrive un reato, un diritto o un'azione che gli stessi fatti possono integrare "
    "al posto di quello contestato o richiesto, anche in via subordinata. Vale solo per le norme che "
    "descrivono una fattispecie: per le altre (regole generali, concorso di persone, circostanze, "
    "procedura) scegli tra \"applicabile\" e \"incompatibile\";\n"
    "- \"incompatibile\": un fatto preciso del caso contraddice un requisito scritto nel testo "
    "dell'articolo.\n"
    "Usa \"incompatibile\" solo se puoi indicare quel requisito e quel fatto; in ogni dubbio usa "
    "\"applicabile\". Rispondi SOLO con un array JSON, un oggetto per articolo, nessun altro testo:\n"
    '[{"article": "<come indicato>", "verdict": "applicabile|alternativa|incompatibile", '
    '"requisito": "<il requisito del testo, se incompatibile>", "fatto": "<il fatto del caso, se '
    'incompatibile>", "reason": "<una frase>"}]'
)


def _applicability_key(label: str) -> str:
    """"Art. 116 c.p." / "116 c.p." -> "116 c.p.": the model doesn't always
    echo the label exactly."""
    return " ".join(re.sub(r"^\s*art(?:icolo)?\.?\s*", "", str(label), flags=re.IGNORECASE).lower().split())


def _applicability_candidates(facts: str, request: str, article_rows: List[Dict[str, Any]]) -> tuple:
    """({article: rows} to check, {label: text}). Exempt, never sent to the
    model: rows without an article tag, cross-reference rows (grounded in a
    chosen article's own text) and the articles the user's own text names."""
    from .nodes.cypher import CROSSREF_SOURCE

    named = set(_refs_in(_case_text(facts, request)))
    to_check: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows:
        article = row.get("_article")
        if not article or row.get("_source") == CROSSREF_SOURCE or article in named:
            continue
        to_check.setdefault(article, []).append(row)
    texts = {}
    for article, rows in to_check.items():
        text = " ".join(
            " ".join(((r.get("s") or {}).get("plain_text") or (r.get("s") or {}).get("abstract") or "").split())
            for r in rows
        ).strip()
        if text:
            texts[f"{article[0]} {article[1]}"] = text[:_ARTICLE_CHARS]
    return to_check, texts


def _applicability_verdicts(facts: str, request: str, texts: Dict[str, str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """{label key: verdict} from one model call; None if the call or the reply fails."""
    if not texts:
        return {}
    human = (
        f"Fatti del caso:\n{_case_text(facts, request)[:2000]}\n\n"
        "Articoli candidati (testo reale dalla banca dati):\n"
        + "\n\n".join(f"[{label}]\n{text}" for label, text in texts.items())
    )
    try:
        raw = _call_chat([SystemMessage(content=_APPLICABILITY_SYSTEM), HumanMessage(content=human)],
                         max_tokens=1200)
        clean = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
        verdicts = json.loads(clean)
    except Exception as exc:
        logger.warning("applicability check failed, keeping every candidate article: %s", exc)
        vlog("case_analysis_applicability", {"checked": len(texts), "error": str(exc)[:160]})
        return None
    if not isinstance(verdicts, list):
        return None
    return {_applicability_key(v["article"]): v for v in verdicts
            if isinstance(v, dict) and isinstance(v.get("article"), str)}


def _drops_article(verdict: Optional[Dict[str, Any]]) -> bool:
    """Only an "incompatibile" verdict that names both the requirement in the
    article's text and the contradicting fact drops an article."""
    return bool(verdict) and str(verdict.get("verdict", "")).strip().lower() == "incompatibile" \
        and bool(str(verdict.get("requisito") or "").strip()) and bool(str(verdict.get("fatto") or "").strip())


def _filter_applicable_articles(facts: str, request: str,
                                article_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The article rows without those the model shows to be incompatible with
    the facts, in their original order. Fails open: on any error, unparseable
    reply or missing verdict, the rows are kept."""
    to_check, texts = _applicability_candidates(facts, request, article_rows)
    if not texts:
        return article_rows
    verdicts = _applicability_verdicts(facts, request, texts)
    if verdicts is None:
        return article_rows
    dropped_articles, dropped = set(), []
    for article in to_check:
        label = f"{article[0]} {article[1]}"
        verdict = verdicts.get(_applicability_key(label))
        if _drops_article(verdict):
            dropped_articles.add(article)
            dropped.append({"article": label, "requisito": str(verdict.get("requisito"))[:200],
                            "fatto": str(verdict.get("fatto"))[:200], "reason": str(verdict.get("reason", ""))[:200]})
    vlog("case_analysis_applicability", {
        "checked": sorted(texts),
        "verdicts": {label: (verdicts.get(_applicability_key(label)) or {}).get("verdict", "missing")
                     for label in texts},
        "dropped": dropped,
    })
    return [r for r in article_rows if r.get("_article") not in dropped_articles]


def run_case_analysis_pipeline(facts: str, session_lang: str = "it", request: str = "") -> dict:
    """A reasoned parere on a case: facts described in the message, or an
    uploaded document (`facts`) with the user's message as `request`.

    Returns {"draft": str, "citations": list} like run_defensive_pipeline.
    """
    import neo4j
    from .main import driver, NEO4J_DATABASE

    analysis = _analyse_case(facts, session_lang, request)
    article_rows, ruling_rows, candidates, selected = [], [], [], []
    try:
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            picked = _select_articles(session, driver, NEO4J_DATABASE, facts, request, analysis)
            first_rulings, similar = picked["first_rulings"], picked["similar"]
            candidates, selected, article_rows = picked["candidates"], picked["selected"], picked["article_rows"]
            # Second search with the verified article numbers: rulings on the
            # article itself, not on whatever else the keywords match.
            second = _search_case_law(session, analysis, numbers=[n for n, _ in selected[:4]],
                                      exclude_docs={(r.get("d") or {}).get("id") for r in similar})
            ruling_rows = _ruling_passages(session, _reading_order(similar, second, first_rulings),
                                           selected, analysis)
    except Exception as exc:
        logger.warning("case analysis retrieval failed: %s", exc)

    # Per-article applicability (round 3, item 3): off unless
    # APPLICABILITY_CHECK_ENABLED. Runs after article_rows/ruling_rows are
    # final, before both the citations and the draft prompt use them.
    if feature_flags.enabled("APPLICABILITY_CHECK_ENABLED"):
        article_rows = _filter_applicable_articles(facts, request, article_rows)

    articles = _source_block(article_rows, _ARTICLE_CHARS)
    rulings = _source_block(ruling_rows, _PASSAGE_CHARS + 100)
    principles = _extract_principles(facts, analysis, ruling_rows, request)
    draft = _call_chat(
        [SystemMessage(content=_parere_system(analysis, session_lang)),
         HumanMessage(content=_parere_human(facts, analysis, articles, rulings, request, principles))],
        max_tokens=_PARERE_TOKENS,
    )
    found = sorted({row["_article"] for row in article_rows if row.get("_article")})
    draft = _mark_unverified(draft, found, articles + "\n" + rulings)

    topics = _extract_deepdive_topics(draft[draft.find("**3."):] if "**3." in draft else draft, {})
    if topics:
        draft += "\n\n" + _format_deepdive_suggestion(topics)
    if ruling_rows or _UNVERIFIED in draft:
        draft += (
            "\n\n⚠️ Le sentenze citate provengono dal corpus documentale; i riferimenti segnati "
            "[DA VERIFICARE] non sono stati trovati nelle fonti consultate. Verificare tutto prima dell'uso."
        )
    # A row whose text lived only in `abstract` (special/dottrina books, e.g.
    # "Codice Penale Commentato") fails _extract_citations' length gate and
    # comes back with sections: [] — a citation naming a source with nothing
    # in it. Drop it here rather than let it reach the user.
    citations = [c for c in _rows_to_sources(article_rows + ruling_rows) if c.get("sections")]
    return {"draft": draft, "citations": citations,
            "analysis": analysis, "articles": found, "principles": principles,
            "candidates": [f"art. {n} {c}" for n, c in candidates],
            "norms_requested": [f"art. {n} {c}" for n, c in selected]}


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
        _content = msg.get("content", "")
        if msg.get("role") == "assistant" and (
            "BOZZA DOCUMENTO DIFENSIVO" in _content or "BOZZA STRATEGIA DIFENSIVA" in _content
        ):
            original_draft = msg["content"][:3000]
            break

    system = (
        "Sei un avvocato esperto di diritto italiano. "
        f"Nel caso descritto nella bozza difensiva di riferimento, l'utente vuole approfondire "
        f"specificamente: '{topic}'.\n\n"
        "Fornisci un'analisi giuridica concreta e approfondita di questo argomento "
        "APPLICATA AL CASO SPECIFICO — non una trattazione generale dell'istituto. "
        "Rispondi come se stessi preparando questa sezione per l'udienza:\n"
        "- Come si applica questo argomento ai fatti specifici di questo caso\n"
        "- Quale norma esatta regola questa situazione e perché\n"
        "- Quale orientamento giurisprudenziale è più favorevole al cliente in questo caso\n"
        "- Quali prove specifiche di questo caso supportano o indeboliscono questo argomento\n"
        "- Quali obiezioni farà la controparte su questo punto specifico e come replicare\n"
        "NON fare una trattazione generale — ogni punto deve riferirsi ai fatti del caso. "
        "NON citare articoli o sentenze a memoria — usa SOLO le fonti fornite o scrivi [DA VERIFICARE]. "
        "NON inventare cifre o fatti non presenti nella bozza di riferimento."
        + (f"\n\nFonti dal corpus legale:\n{citations_text}" if citations_text else "")
        + (f"\n\nBOZZA DIFENSIVA DEL CASO (usa questi fatti come riferimento):\n{original_draft}" if original_draft else "")
        + "\n\nATTENZIONE CRITICA: "
        "\n- L'art. 2697 c.c. è la norma generale sull'onere della prova — NON inventare altri articoli"
        "\n- NON citare MAI numeri di sentenza specifici (es. n. 12345) a meno che non siano "
        "esplicitamente presenti nelle fonti del corpus fornite sopra"
        "\n- Se non hai una sentenza specifica dal corpus, scrivi 'orientamento consolidato della "
        "Cassazione [DA VERIFICARE]' senza inventare numeri"
    )

    return _call_chat(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Approfondisci '{topic}' in relazione al caso descritto nella bozza."),
        ],
        max_tokens=2000,
    )
