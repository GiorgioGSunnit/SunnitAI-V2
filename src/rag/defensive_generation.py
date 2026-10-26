"""
Defensive legal document generation pipeline.
Given an uploaded judicial document, this module:
1. Classifies the proceeding type
2. Extracts opposing claims and cited articles
3. Queries Neo4j for counter-arguments and favorable jurisprudence
4. Generates a structured defensive draft
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from . import article_text, feature_flags
from .ai_chat import _call_chat, embedding_model, parse_json_reply
from .article_suffix import SUFFIX, normalize_number, number_variants
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

_PROCEEDING_TYPES = {"civile", "penale", "amministrativo"}
_PROCEEDING_SUBTYPES = {
    "atto_citazione", "decreto_ingiuntivo", "ricorso", "appello", "altro",
}

_CALL_CHAT_PARAMS = frozenset(inspect.signature(_call_chat).parameters)


def _chat(messages, **kwargs):
    """_call_chat, dropping any kwarg the resolved _call_chat doesn't accept.

    Prod, 17 Oct: this module started passing temperature=0 before ai_chat.py's
    matching change had deployed everywhere it runs, and every case-analysis
    LLM call crashed with "_call_chat() got an unexpected keyword argument
    'temperature'" -- on the flag-off path too, since _analyse_case (v1) also
    passes it. Dropping an unsupported kwarg here is always safe: chat_model is
    already constructed with temperature=0 in every ai_chat.py this module has
    ever shipped with, so an older _call_chat gives the same temperature anyway.
    """
    accepted = {k: v for k, v in kwargs.items() if k in _CALL_CHAT_PARAMS}
    dropped = kwargs.keys() - accepted.keys()
    if dropped:
        logger.warning("_chat: _call_chat does not accept %s; calling without them "
                        "(mismatched deploy of ai_chat.py?)", sorted(dropped))
    return _call_chat(messages, **accepted)


_JSON_RETRY_SUFFIX = (
    "\n\nRispondi SOLO con il JSON richiesto, senza alcuna spiegazione, nota, blocco di codice o testo "
    "introduttivo: SOLO il JSON, a partire dal primo carattere della risposta."
)


def _chat_json(messages, max_tokens, **chat_kwargs):
    """_chat() + parse_json_reply, with one retry -- the same system message
    plus a stricter closing instruction -- if the first reply has no
    parseable JSON at all. parse_json_reply already tolerates code fences
    and text around the JSON; a reply with no JSON in it anywhere (prod, 18
    Oct: "no JSON value in the model's reply", _choose_articles) needs a
    different answer, not more lenient parsing. Raises ValueError, same as
    parse_json_reply, if the retry also fails -- every caller already
    catches that and fails open with a WARNING, unchanged.

    Round 8 (19 Oct): the applicability check failed this way on prod for
    every one of 12 candidate articles ("check failed: every article would
    be kept") with nothing to show WHY -- the warning logged only the
    exception, never the reply that didn't parse. Both failures are logged
    here, with the raw reply, so the next one is diagnosable without
    needing to reproduce it live."""
    raw = _chat(messages, max_tokens=max_tokens, **chat_kwargs)
    try:
        return parse_json_reply(raw)
    except ValueError as exc:
        logger.info("_chat_json: first reply had no parseable JSON (%s); retrying with a stricter "
                    "instruction. raw reply: %r", exc, raw[:300])
    retried = [SystemMessage(content=messages[0].content + _JSON_RETRY_SUFFIX), *messages[1:]]
    raw = _chat(retried, max_tokens=max_tokens, **chat_kwargs)
    try:
        return parse_json_reply(raw)
    except ValueError as exc:
        logger.warning("_chat_json: retry also had no parseable JSON (%s); failing open. raw reply: %r",
                       exc, raw[:500])
        raise


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

    try:
        parsed = parse_json_reply(raw)
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
        topics = parse_json_reply(raw)
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
# "609-bis", "609bis" and "609 bis" (r14: h7's "Art. 609 bis c.p." escaped the check).
_ARTICLE_NUM = r"(\d+(?:\s*-?\s*" + SUFFIX + r"(?![a-z]))?(?:\.\d+)?)"
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
# The draft prompt must leave room for _PARERE_TOKENS in the 20k-token context.
# Prod, 19 Oct (held-out h9): a long case plus the fusion pool's uncapped
# articles came to ~15,018 input tokens + 5,000 and the user got an error.
# _fit_parere_prompt trims the sources until this estimate fits; the estimate
# counts 3 characters per token, more than this model's tokenizer uses on
# Italian legal text (~3.5, see _FACTS_CHARS), so it errs on the safe side.
_DRAFT_INPUT_BUDGET = 13000
_CHARS_PER_TOKEN = 3.0

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

# CASE_ANALYSIS_ENABLED (default false). Extends _analyse_case's schema with a
# structured fact model and, per legal issue, a category and norms that can
# name any code or law — not just the six sigle _CODE_PREFIXES knows — so a
# case needing Codice del Consumo or a numbered law isn't structurally unable
# to reach it. See _analyse_case_v2 / _select_articles_v2.
CASE_ANALYSIS_FLAG = "CASE_ANALYSIS_ENABLED"
CASE_ANALYSIS_SOURCE = "case_analysis"

# CASE_POOL_FUSION_ENABLED (default false; has no effect unless
# CASE_ANALYSIS_ENABLED is also on). Adds a ranked article pool on top of
# _select_articles_v2's own selection: CURRENT (v2's full "selected" --
# v1's code-search + _choose_articles union institutions) union BASELINE
# (vector+fulltext on the raw case text) union DISTILL (vector+fulltext per
# distilled bullet), ranked by fusion (how many of the three agree, then
# best rank) and capped at _FUSION_POOL_CAP -- the exact three sources
# validated as FUSION_BASELINE on scripts/exp_distill.py (round 6, 19 Oct:
# 27/47 @8, 32/47 @12). Round 8 (19 Oct): the first version of this
# replaced CURRENT with institution-resolution alone (to skip v1's own
# search + _choose_articles call) -- a simplification exp_distill.py's own
# numbers never tested, and the actual cause of a prod parity bug (wrong-
# code institution matches, a book in the ranked pool, a naive-concatenation
# merge bug). See _select_articles_fusion, which now calls
# _select_articles_v2 itself for CURRENT and shares its merge functions
# with scripts/exp_distill.py, instead of reimplementing either.
_CASE_POOL_FUSION_FLAG = "CASE_POOL_FUSION_ENABLED"
# section_embeddings_v2: same default scripts/exp_distill.py uses; override
# via the environment if the live index is named differently.
_CASE_POOL_VECTOR_INDEX = os.getenv("CASE_POOL_VECTOR_INDEX", "section_embeddings_v2")
_FUSION_SEARCH_K = 20
# How many SINGLE-signal (found by only one of CURRENT/BASELINE/DISTILL)
# candidates fill out the pool. Added on top, uncapped: articles named by
# the user's own text or agreed on by rulings (_select_articles_v2's own
# "keep" -- never in doubt), AND (round 9, 19 Oct) any article found by two
# or more of the three signals, regardless of its fused rank -- a rank-only
# cap could previously cut a multi-signal article in favour of weaker
# single-signal ones ranked ahead of it only by tie-break. Also raised from
# 12 to 16, for the single-signal fill this now actually governs (a prod
# diagnostic, loss_funnel.py, found l2's 582 and l4's 337-ter cut at rank
# 31/18 -- if either is genuinely single-signal, this cap increase alone
# won't reach it either; re-run the funnel to see which it was).
_FUSION_POOL_CAP = 16
# HYBRID_NARRATIVE_ENABLED: the map's and the analysis' issues searched as queries, at most this many.
_HYBRID_ISSUE_QUERIES = 8

# HEADING_INDEX_ENABLED (default false): resolve an institution against a
# dedicated vector index over every primary Section's titolo_articolo (its
# official rubrica), by MEANING rather than shared words -- the gap
# article_by_title's word/stem matching structurally can't close. "tentativo"
# (h5) never title-matches art. 56's rubrica "Delitto tentato": article_by_
# title's own tolerance rule allows zero extra title words for a one-word
# institution name, and "Delitto tentato" has one extra word ("Delitto")
# over the single stem "tentativo"/"tentato" shares -- a real rubrica for
# the right institution, rejected only because of a word-count rule, not
# because it's the wrong article. A semantic index can tell they're the
# same institution without needing every word to match.
#
# Scripts/build_heading_index.py builds it (DB write -- not run here, see
# that script's own docstring for the exact CREATE VECTOR INDEX statement
# and size/time estimate). _heading_index_match falls back to None (no
# different from the index not existing yet) on any query failure, so
# turning this flag on before the index is built changes nothing.
HEADING_INDEX_FLAG = "HEADING_INDEX_ENABLED"
_HEADING_VECTOR_INDEX = os.getenv("HEADING_VECTOR_INDEX", "section_heading_embeddings")
# Neo4j's vector index has no pre-filter: queryNodes ranks across every
# Section with a heading embedding (every primary code and law, once built),
# and only the institution's own document's hits are kept afterward. k=200
# is a hedge against the institution's own article being outranked, across
# the whole corpus, by closer-sounding headings in OTHER codes -- cheap
# (one ANN query), not free (a corpus with many codes could still occasionally
# push the right heading past 200; there is no way to bound this without a
# pre-filtered vector query, which this Neo4j version may or may not support).
_HEADING_INDEX_K = 200


def _heading_index_match(session, doc_id: str, term: str) -> Optional[str]:
    """The article number whose titolo_articolo is closest, by meaning, to
    `term`, among Sections of `doc_id` -- or None if the index doesn't
    exist yet, the query fails, or no candidate's heading shares enough of
    `term`'s own words to trust the match.

    The overlap floor ("at least half the stems", same bar _issue_
    article_refs uses) is deliberately looser than article_by_title's own
    "zero to two extra words" rule -- the whole point of this path is to
    catch a DIFFERENT wording of the same institution (that is what the
    embedding similarity is for), not to re-require a word-for-word match
    on top of it. It is not zero, either: an embedding's top hit can still
    be a wrong institution that merely sounds similar, and nothing else
    here would catch that."""
    from .issue_spotting import _stems

    stems = _stems(term)
    if not stems:
        return None
    try:
        embedding = list(embedding_model.embed_query(term))
        rows = session.run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
            "RETURN node.name AS name, node.titolo_articolo AS titolo_articolo, score "
            "ORDER BY score DESC LIMIT 5",
            index=_HEADING_VECTOR_INDEX, k=_HEADING_INDEX_K, embedding=embedding, doc_id=doc_id,
        ).data()
    except Exception as exc:
        logger.info("heading index unavailable, skipping: %s", exc)
        return None
    needed = max(1, (len(stems) + 1) // 2)
    for row in rows:
        name = (row.get("name") or "").strip()
        m = _ARTICLE_SECTION_RE.match(name)
        if not m:
            continue
        title_stems = _stems(row.get("titolo_articolo") or "")
        if title_stems and len(stems & title_stems) >= needed:
            return _article_num(name)
    return None


_CASE_CATEGORIES = {
    "qualificazione", "elemento_soggettivo", "nesso_causale",
    "qualificazione_alternativa", "circostanza", "procedura",
}
_FACT_MODEL_KEYS = {
    "parti", "condotta", "indizi_intenzionali", "evento",
    "nesso_causale", "condotta_successiva", "fase_procedurale",
}

_CASE_ANALYSIS_V2_SYSTEM = (
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
    '- "fatti_strutturati": un oggetto con queste chiavi, ciascuna una stringa (vuota se il caso non la '
    'indica): "parti" (chi sono i soggetti coinvolti e il loro ruolo), "condotta" (che cosa ha fatto '
    'ciascuno), "indizi_intenzionali" (elementi del testo che indicano dolo, colpa, o la loro assenza), '
    '"evento" (che cosa è accaduto come conseguenza), "nesso_causale" (la sequenza che lega condotta ed '
    'evento, passaggio per passaggio), "condotta_successiva" (che cosa ha fatto ciascuno dopo il fatto: '
    'soccorso, fuga, confessione, denuncia...), "fase_procedurale" (in che fase si trova il procedimento '
    'o la vicenda, se il caso lo indica)\n'
    '- "questioni": da 3 a 6 questioni giuridiche specifiche del caso, in ordine di importanza: prima la '
    "qualificazione principale e i suoi presupposti, poi l'elemento soggettivo, il nesso causale, le "
    "qualificazioni alternative, le circostanze, poi la procedura (termini, prescrizione, procedibilità) "
    'solo se il caso la pone. Ciascuna è {"titolo": stringa, "categoria": una tra "qualificazione", '
    '"elemento_soggettivo", "nesso_causale", "qualificazione_alternativa", "circostanza", "procedura", '
    '"istituti": da 2 a 6 istituti giuridici che disciplinano la questione, ciascuno {"nome": il nome '
    'dell\'istituto con le stesse parole con cui è intitolato nel codice (per esempio "omicidio '
    'colposo", "morte o lesione come conseguenza di altro delitto", "elemento psicologico del reato" — '
    'mai un numero da solo, il nome dell\'istituto, non la tua opinione su di esso), "codice": il nome '
    'del codice o della legge come lo scriveresti in un atto — per i codici tradizionali usa c.c., '
    'c.p.c., c.p., c.p.p., c.p.a., c.d.s.; per qualsiasi altra legge scrivi il nome per intero, per '
    'esempio "Codice del Consumo" o "l. 392/1978" — non limitarti ai sei codici tradizionali se il caso '
    'richiede un\'altra legge, "numero_indicativo": il numero dell\'articolo se lo ricordi con '
    'certezza, altrimenti stringa vuota — non indovinare: un numero sbagliato è peggio di nessun '
    'numero, verrà comunque verificato sul testo reale prima di essere usato}, "parole_chiave": da 4 a '
    '8 nomi tecnici di istituti giuridici e concetti con cui le sentenze trattano quella questione, non '
    'parole generiche}. Per le questioni di categoria "qualificazione_alternativa": l\'istituto '
    'alternativo deve avere lo stesso esito materiale dei fatti dell\'istituto principale — se i fatti '
    'hanno causato la morte della vittima, l\'alternativa è un\'altra fattispecie che ha la morte come '
    'esito (per esempio omicidio colposo, o morte o lesione come conseguenza di altro delitto), mai una '
    'fattispecie di lesione personale che presuppone la sua sopravvivenza, e viceversa se la vittima è '
    'sopravvissuta; proponi l\'istituto giusto per l\'esito, non il più generico. Quando la '
    'responsabilità civile può fondarsi sia su un rapporto contrattuale sia su un fatto illecito verso '
    'la stessa parte, prepara entrambe come questioni di categoria "qualificazione" distinte, non solo '
    'quella extracontrattuale.\n'
    '- "fatti": i fatti del testo uno per uno, senza accorparli, compresi i dettagli che sembrano '
    "secondari: età e condizioni delle persone, abitudini, luoghi, orari e condizioni di visibilità, "
    "chi era presente e chi ha visto che cosa, che cosa risulta da verbali o dichiarazioni, tempi "
    'trascorsi. Ciascuno è {"fatto": stringa, "effetto": "favorevole", "sfavorevole", "favorevole e '
    'sfavorevole" o "neutro" per la parte assistita, "perche": stringa breve}\n'
    "Non inventare fatti. Rispondi SOLO con il JSON."
)

# ISSUE_CHECKLISTS_ENABLED (default false): general issue checklists, by
# area, appended to _CASE_ANALYSIS_V2_SYSTEM. Round 10 (19 Oct): the held-
# out set found a real generalisation gap, not a retrieval bug -- h1, h4-h8
# reached run_case_analysis_pipeline and still missed most expected
# articles, because the model's own "questioni" never raised the issue in
# the first place (nothing to resolve an institution for, nothing for
# fusion to search on). These are deliberately METHODS to check for, never
# article numbers or institution names (an "istituto" the model invents to
# satisfy a checklist item would be worse than missing it) -- the model
# still decides, from the facts, whether each one actually applies, and
# "questioni" still only lists the ones that do. Developed and tuned on the
# tuning set (eval_narrative_cases.json's non-heldout cases) only, per the
# standing rule never to tune on the held-out set itself.
ISSUE_CHECKLISTS_FLAG = "ISSUE_CHECKLISTS_ENABLED"
_ISSUE_CHECKLISTS_BLOCK = (
    "\n\nPer l'area individuata, controlla anche se i fatti pongono una di queste questioni tipiche di "
    "quell'area. Sono un controllo, non un elenco da esaurire: aggiungi a \"questioni\" solo quelle che i "
    "fatti pongono davvero, con il loro istituto vero (mai un numero di articolo, mai un istituto inventato "
    "solo per coprire la voce); ignora le altre senza commentarle.\n"
    "- penale: la forma di manifestazione del reato (tentativo, concorso di persone, concorso di reati, "
    "reato continuato); l'elemento soggettivo e la sua forma esatta (dolo diretto, dolo eventuale, colpa "
    "con le sue forme); il nesso causale quando condotta ed evento sono distinti; le circostanze aggravanti "
    "o attenuanti che i fatti indicano; le cause di esclusione della responsabilità (legittima difesa, "
    "stato di necessità, consenso dell'avente diritto, adempimento di un dovere); qualificazioni "
    "alternative con lo stesso esito materiale dei fatti; la procedibilità (querela, condizioni).\n"
    "- obbligazioni e contratti: la natura della responsabilità (contrattuale, extracontrattuale, o "
    "entrambe verso la stessa parte); l'inadempimento e la ripartizione dell'onere della prova; i rimedi "
    "disponibili (risoluzione, recesso, riduzione del prezzo, risarcimento, eccezione di inadempimento); "
    "una clausola penale e i presupposti della sua riduzione giudiziale; vizi della cosa e garanzie; "
    "prescrizione o decadenza quando i fatti indicano le date necessarie.\n"
    "- diritti reali e possesso: il titolo di acquisto (usucapione, contratto, successione) quando è in "
    "discussione; la tutela possessoria rispetto a quella petitoria; la costituzione, l'estinzione o "
    "l'esercizio di una servitù, inclusa l'interclusione del fondo; i rapporti di comunione o condominio "
    "(parti comuni, obblighi dell'amministratore, responsabilità per i danni).\n"
    "- successioni: l'apertura della successione e il titolo per succedere; l'accettazione dell'eredità, "
    "espressa o tacita (quali atti la implicano, quali restano atti conservativi che non la implicano), e "
    "l'accettazione con beneficio d'inventario; la responsabilità per i debiti del de cuius e i suoi "
    "limiti; la rinuncia all'eredità e i suoi effetti.\n"
    "- diritto amministrativo: la legittimità del provvedimento e i suoi vizi tipici (incompetenza, "
    "violazione di legge, eccesso di potere); l'autotutela amministrativa; il silenzio o l'inerzia della "
    "pubblica amministrazione; l'esecuzione di un giudicato e il giudizio di ottemperanza quando "
    "l'amministrazione non vi ha dato seguito."
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

# CASE_POOL_FUSION_ENABLED: the case as neutral fact bullets, in statutory
# vocabulary, to search with -- same prompt validated on scripts/exp_distill.py
# (round 6, 19 Oct), ported here unchanged.
_FUSION_DISTILL_SYSTEM = (
    "Sei un avvocato italiano. Riscrivi il caso che segue come fatti giuridici neutri, in 5-8 punti brevi, "
    "uno per ciascuna di queste categorie quando il caso la prevede (saltala se non si applica): "
    "\"soggetti e ruoli\", \"condotta\", \"elementi sull'intenzione\", \"evento/esito\", \"nesso causale\", "
    "\"condotta successiva\", \"posizione processuale\", \"cosa si chiede\"; per i fatti penali aggiungi, "
    "se pertinente, \"circostanze\" (per esempio: vittima in condizione di minorata difesa o particolarmente "
    "vulnerabile, provocazione, danno riparato, recidiva, numero di persone coinvolte). Per ciascun punto "
    "usa il termine giuridico tipico con cui il codice descrive quella condotta, quell'evento o quello "
    "stato soggettivo, non una parafrasi generica: per esempio, se i fatti lo giustificano, \"artifizi o "
    "raggiri\" (non \"inganno\"), \"induzione in errore\", \"violenza o minaccia\", \"impossessamento\", "
    "\"sottrazione\", \"custodia\" o \"detenzione\" della cosa, \"dolo\" o \"colpa\" (con la forma specifica: "
    "negligenza, imprudenza, imperizia), \"inadempimento\", \"mora\", \"vizio della cosa\", \"nesso di "
    "causalità\". Elimina i dettagli irrilevanti (nomi propri non necessari, descrizioni di contesto, "
    "aggettivi, orari e luoghi non giuridicamente rilevanti). Non usare MAI un numero di articolo né il "
    "nome di un codice o di una legge: descrivi solo i fatti, con il vocabolario giuridico italiano. "
    "Restituisci SOLO un array JSON di oggetti {\"categoria\": una delle etichette sopra, \"testo\": il "
    "punto, breve}."
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


def _analyse_case(facts: str, lang: str, request: str = "",
                  frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The legal questions, the norms and the role of each fact. Best-effort:
    any failure returns empty lists, and the parere is written from the facts."""
    empty = {"parte_assistita": "", "posizione": "", "area": "", "richiesta": "", "ricerca_fatti": [],
             "questioni": [], "fatti": []}
    try:
        parsed = _chat_json([SystemMessage(content=_CASE_ANALYSIS_SYSTEM),
                            HumanMessage(content=_case_text(facts, request) + _frame_block(frame))],
                           max_tokens=1500, temperature=0)
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


def _analyse_case_v2(facts: str, lang: str, request: str = "",
                     frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Like _analyse_case, plus a structured fact model ("fatti_strutturati":
    parti, condotta, indizi_intenzionali, evento, nesso_causale,
    condotta_successiva, fase_procedurale) and, per legal issue, a category
    and a list of institutions ("istituti": {nome, codice, numero_indicativo})
    naming any code or law, not just the six _CODE_PREFIXES knows.

    Deliberately asks for the institution's name, not an article number:
    checked on prod (16 Oct, check_applicability.py on Sempronio), the
    model's own numbers were frequently wrong from memory — "elemento
    soggettivo" -> art. 12 c.p. (should be 43), "reati alternativi" -> 597,
    598 c.p. (should be 589, 586), 589 never proposed at all. _resolve_institution
    resolves the name against the real rubriche (the institution's own words
    usually match them closely — "omicidio colposo" names art. 589 exactly);
    numero_indicativo is kept only as a last-resort hint, verified against
    that article's own title before ever being trusted (see
    _number_matches_institution).

    Used by run_case_analysis_pipeline when CASE_ANALYSIS_ENABLED. Same
    fail-open contract as _analyse_case: any failure returns the same shape
    with empty values, and the parere is written from the facts alone."""
    empty = {"parte_assistita": "", "posizione": "", "area": "", "richiesta": "", "ricerca_fatti": [],
             "questioni": [], "fatti": [], "fatti_strutturati": {}}
    system = _CASE_ANALYSIS_V2_SYSTEM
    if feature_flags.enabled(ISSUE_CHECKLISTS_FLAG):
        system += _ISSUE_CHECKLISTS_BLOCK
    t0 = time.time()
    try:
        parsed = _chat_json([SystemMessage(content=system),
                            HumanMessage(content=_case_text(facts, request) + _frame_block(frame))],
                           max_tokens=2200 if feature_flags.enabled(ISSUE_CHECKLISTS_FLAG) else 1800,
                           temperature=0)
    except Exception as exc:
        logger.warning("case analysis (v2) failed: %s", exc)
        return empty
    finally:
        vlog("case_analysis_v2_llm", duration_ms=(time.time() - t0) * 1000)
    if not isinstance(parsed, dict):
        return empty
    result = {**empty, **{k: v for k, v in parsed.items() if k in empty}}
    result["questioni"] = [q for q in result["questioni"] if isinstance(q, dict)][:6] \
        if isinstance(result["questioni"], list) else []
    for q in result["questioni"]:
        istituti_raw = q.get("istituti")
        q["istituti"] = [
            {"nome": str(i.get("nome") or "").strip(), "codice": str(i.get("codice") or "").strip(),
             "numero_indicativo": str(i.get("numero_indicativo") or "").strip()}
            for i in (istituti_raw if isinstance(istituti_raw, list) else [])
            if isinstance(i, dict) and str(i.get("nome") or "").strip()
        ][:6]
        q["parole_chiave"] = _as_list(q.get("parole_chiave"))
        categoria = str(q.get("categoria") or "").strip().lower()
        q["categoria"] = categoria if categoria in _CASE_CATEGORIES else ""
    result["fatti"] = [f for f in result["fatti"] if isinstance(f, dict)][:15] \
        if isinstance(result["fatti"], list) else []
    result["ricerca_fatti"] = _as_list(result["ricerca_fatti"])[:8]
    fatti_strutturati = result.get("fatti_strutturati")
    result["fatti_strutturati"] = {
        k: str(v).strip() for k, v in fatti_strutturati.items()
        if k in _FACT_MODEL_KEYS and isinstance(v, (str, int, float)) and str(v).strip()
    } if isinstance(fatti_strutturati, dict) else {}
    if result["area"] == "amministrativo" and _DAMAGES_RE.search(facts):
        result["area"] = "civile"
    return result


# --- STRUCTURED_ANALYSIS_ENABLED (requires CASE_ANALYSIS_ENABLED) -----------
#
# Handover 19 Oct, 6.3: one general method that gives the same analysis however
# the case is worded, in short focused calls rather than one long prompt. Call
# 1 reads the case sentence by sentence and returns its facts in neutral form;
# call 2 fills a fixed grid for the case's area, slot by slot, and names the
# legal institution for each slot the facts actually raise, in words, never by
# number. The result has _analyse_case_v2's shape, so institution resolution,
# fusion, the rulings search, the check and the draft are unchanged. Any
# failure falls back to _analyse_case_v2.
#
# Every slot can be "not pertinent": lists of things to consider made the model
# invent material in the round-3 draft previews (16 Oct), so the grid asks
# whether the facts pose each point, not to cover them all.
STRUCTURED_ANALYSIS_FLAG = "STRUCTURED_ANALYSIS_ENABLED"
_STRUCTURED_MAX_QUESTIONS = 7
_STRUCTURED_MAX_FACTS = 20
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ý\"«“'(])|\n+")

_STRUCTURED_FACTS_SYSTEM = (
    "Sei un avvocato italiano. Ti viene dato un caso diviso in frasi numerate. Estrai i fatti giuridicamente "
    "rilevanti, in forma neutra. Restituisci SOLO un oggetto JSON con queste chiavi:\n"
    '- "parte_assistita": la parte nel cui interesse va redatto il parere; se il testo non la indica, la parte '
    "che chiede assistenza\n"
    '- "posizione": la sua posizione processuale o sostanziale\n'
    '- "area": il ramo del diritto che regola la vicenda, una tra "civile", "penale", "amministrativo", '
    '"lavoro", "tributario", "altro" (una richiesta di risarcimento verso un ente pubblico è "civile")\n'
    '- "richiesta": in una frase, che cosa chiede il caso\n'
    '- "ricerca_fatti": da 4 a 8 parole concrete che descrivono i fatti materiali (cose, luoghi, eventi, '
    "soggetti: non termini giuridici), con cui cercare sentenze su fatti simili\n"
    '- "fatti": per ogni frase che contiene fatti rilevanti, uno o più oggetti {"frase": il numero della '
    'frase, "fatto": il fatto in forma neutra e breve (chi, che cosa ha fatto, con quale mezzo, con quale '
    'risultato), senza nomi propri non necessari e senza valutazioni, "tipo": uno tra "soggetti", "rapporto", '
    '"condotta", "mezzo", "intenzione", "evento", "danno", "condotta_successiva", "tempi", "procedura", '
    '"atto", "richiesta", "effetto": "favorevole", "sfavorevole", "favorevole e sfavorevole" o "neutro" per '
    'la parte assistita, "perche": stringa breve}\n'
    "Salta le frasi senza fatti rilevanti (contesto, cortesie, ripetizioni). Non aggiungere fatti che il testo "
    "non contiene e non dedurre stati d'animo o intenzioni che il testo non indica. Non indicare articoli di "
    "legge. Rispondi SOLO con il JSON."
)

_STRUCTURED_GRID_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati i fatti di un caso, già estratti in forma neutra, e una griglia "
    "di analisi per la sua area. Per ogni voce della griglia decidi se i fatti la pongono davvero. Restituisci "
    "SOLO un array JSON, un oggetto per voce, nell'ordine della griglia:\n"
    '{"voce": l\'identificativo della voce, "pertinente": true o false, "valutazione": una frase che applica la '
    'voce ai fatti (vuota se non pertinente), "istituti": da 1 a 3 istituti giuridici che la disciplinano, '
    'ciascuno {"nome": il nome dell\'istituto con le parole con cui è intitolato nel codice o nella legge, '
    '"codice": per i codici tradizionali c.c., c.p.c., c.p., c.p.p., c.p.a., c.d.s.; per qualsiasi altra '
    'legge il nome per intero, per esempio "Codice del Consumo"}, "parole_chiave": da 3 a 6 termini tecnici '
    "con cui le sentenze trattano la questione}\n"
    "Regole: una voce è pertinente solo se un fatto elencato la pone; nel dubbio è false. Una voce non "
    "pertinente ha istituti e parole_chiave vuoti. Mai un numero di articolo. Mai un istituto scelto solo per "
    "riempire una voce. Usa il nome specifico dell'istituto, non quello della categoria a cui appartiene. Le "
    "qualificazioni alternative devono avere lo stesso esito materiale dei fatti (se la vittima è morta, "
    "fattispecie che hanno la morte come esito). Rispondi SOLO con il JSON."
)

# (slot id, label, what it asks, categoria for the v2 shape), by area.
_STRUCTURED_GRIDS = {
    "penale": [
        ("qualificazione", "Qualificazione del fatto",
         "il reato o i reati che la condotta descritta integra, e il ruolo di ciascun soggetto", "qualificazione"),
        ("elemento_soggettivo", "Elemento soggettivo",
         "lo stato psicologico richiesto e quello che risulta dai fatti: dolo diretto o eventuale, colpa e sua "
         "forma, preterintenzione", "elemento_soggettivo"),
        ("nesso_causale", "Nesso causale",
         "il rapporto di causalità tra condotta ed evento, con le concause e le cause sopravvenute, quando "
         "condotta ed evento sono distinti", "nesso_causale"),
        ("forma_del_reato", "Forma del reato",
         "solo se i fatti pongono un tentativo, un concorso di più persone nel reato o un reato che solo certi "
         "soggetti possono commettere; un reato consumato da una sola persona non pone questa voce",
         "qualificazione"),
        ("circostanze", "Circostanze",
         "circostanze aggravanti o attenuanti indicate dai fatti, compresa la condotta successiva al fatto "
         "(soccorso, riparazione del danno), ciascuna con il nome della circostanza specifica, non della "
         "categoria: per le circostanze valide per ogni reato scrivi \"circostanze attenuanti comuni\", "
         "\"circostanze attenuanti generiche\" o \"circostanze aggravanti comuni\", mai solo \"circostanze "
         "attenuanti\"", "circostanza"),
        ("cause_di_giustificazione", "Cause di esclusione della punibilità",
         "legittima difesa, stato di necessità, consenso dell'avente diritto, adempimento di un dovere, "
         "esercizio di un diritto, eccesso colposo", "circostanza"),
        ("qualificazioni_alternative", "Qualificazioni alternative",
         "reati diversi che gli stessi fatti possono integrare al posto di quello principale, con lo stesso "
         "esito materiale: se la vittima è morta, solo fattispecie che hanno la morte come esito (per esempio "
         "omicidio colposo, morte o lesione come conseguenza di altro delitto), mai una lesione personale",
         "qualificazione_alternativa"),
        ("procedura", "Procedura",
         "procedibilità (querela o d'ufficio), fase del procedimento e rimedi, solo se i fatti li indicano",
         "procedura"),
    ],
    "civile": [
        ("rapporto", "Rapporto tra le parti",
         "contratto (quale tipo), fatto illecito, proprietà, possesso, condominio, successione, famiglia, "
         "lavoro, consumo", "qualificazione"),
        ("obbligazione_o_diritto", "Obbligazione o diritto",
         "l'obbligazione o il diritto in gioco e chi ne è titolare", "qualificazione"),
        ("inadempimento_o_lesione", "Inadempimento o lesione",
         "l'inadempimento, il vizio o la lesione del diritto, e su chi grava l'onere della prova",
         "qualificazione"),
        ("titolo_della_responsabilita", "Titolo della responsabilità",
         "colpa, dolo, diligenza richiesta, oppure responsabilità senza colpa (cose in custodia, attività "
         "pericolose, fatto degli ausiliari)", "elemento_soggettivo"),
        ("danno_e_nesso", "Danno e nesso causale",
         "il danno lamentato e il nesso di causalità con l'inadempimento o con il fatto", "nesso_causale"),
        ("rimedi", "Rimedi",
         "adempimento, riparazione o sostituzione, riduzione del prezzo, risoluzione, recesso, risarcimento, "
         "azioni a tutela della proprietà o del possesso", "qualificazione"),
        ("difese", "Difese ed eccezioni",
         "le difese della controparte o della parte assistita: eccezione di inadempimento, caso fortuito, "
         "concorso del danneggiato, clausole del contratto", "qualificazione_alternativa"),
        ("termini", "Termini",
         "prescrizione, decadenza, termini di denuncia o di impugnazione, solo se i fatti li pongono", "procedura"),
    ],
    "amministrativo": [
        ("atto", "Atto o comportamento dell'amministrazione",
         "il provvedimento, il silenzio o il comportamento e l'autorità che lo ha adottato", "qualificazione"),
        ("vizio", "Vizi",
         "incompetenza, violazione di legge, eccesso di potere, difetto di motivazione o di istruttoria, "
         "inerzia", "qualificazione"),
        ("tutela", "Tutela",
         "il giudizio e i rimedi (annullamento, risarcimento, azione contro il silenzio, accesso, autotutela) "
         "e i termini per agire", "procedura"),
        ("esecuzione", "Esecuzione",
         "l'esecuzione di una sentenza e il giudizio di ottemperanza, solo se i fatti la pongono", "procedura"),
    ],
}
_GRID_FOR_AREA = {"penale": "penale", "civile": "civile", "lavoro": "civile", "altro": "civile",
                  "amministrativo": "amministrativo", "tributario": "amministrativo"}
_FACT_TYPE_TO_MODEL_KEY = {"soggetti": "parti", "condotta": "condotta", "mezzo": "condotta",
                           "intenzione": "indizi_intenzionali", "evento": "evento", "danno": "evento",
                           "condotta_successiva": "condotta_successiva", "procedura": "fase_procedurale"}


def _central_slot(frame: Optional[Dict[str, Any]]) -> List[tuple]:
    """CASE_FRAME_ENABLED: a first grid slot for the frame's central issues,
    still only pertinent when the facts pose one; [] without them."""
    names = [c["nome"] for c in (frame or {}).get("central_issue_candidates") or []]
    if not names:
        return []
    return [("questione_centrale", "Questione centrale",
             "l'istituto attorno a cui è costruito il caso, tra le questioni centrali candidate del QUADRO ("
             + "; ".join(names) + "), solo se i fatti lo pongono davvero", "qualificazione")]


def _numbered_sentences(text: str) -> List[str]:
    """The case's sentences, for call 1 to refer to by number. No split inside
    "art. 584" or "c.p." followed by a digit or a lower-case word."""
    parts = (" ".join(p.split()) for p in _SENTENCE_SPLIT_RE.split(text or ""))
    return [p for p in parts if len(p) > 1]


def _structured_facts(facts: str, request: str,
                      frame: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call 1: who, what, area and request, and the facts in neutral form, one
    or more per numbered sentence. None on any failure or without facts."""
    sentences = _numbered_sentences(_case_text(facts, request))
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    t0 = time.time()
    try:
        parsed = _chat_json([SystemMessage(content=_STRUCTURED_FACTS_SYSTEM),
                             HumanMessage(content=numbered + _frame_block(frame))],
                            max_tokens=1800, temperature=0)
    except Exception as exc:
        logger.warning("structured analysis, facts: failed: %s", exc)
        return None
    finally:
        vlog("structured_analysis_facts_llm", duration_ms=(time.time() - t0) * 1000)
    if not isinstance(parsed, dict):
        return None
    fatti = [f for f in (parsed.get("fatti") if isinstance(parsed.get("fatti"), list) else [])
             if isinstance(f, dict) and str(f.get("fatto") or "").strip()][:_STRUCTURED_MAX_FACTS]
    if not fatti:
        return None
    return {**parsed, "fatti": fatti}


def _structured_grid(extracted: Dict[str, Any], grid: List[tuple],
                     frame: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
    """Call 2: the grid's slots, each pertinent or not, with institutions in
    words for the pertinent ones. None on any failure."""
    parte = str(extracted.get("parte_assistita") or "").strip()
    posizione = str(extracted.get("posizione") or "").strip()
    fatti = "\n".join(f"- {str(f.get('fatto')).strip()}" for f in extracted["fatti"])
    griglia = "\n".join(f"- {slot}: {question}" for slot, _, question, _ in grid)
    human = (f"PARTE ASSISTITA: {parte}{f' ({posizione})' if posizione else ''}\n"
             f"RICHIESTA: {str(extracted.get('richiesta') or '').strip()}\n\nFATTI:\n{fatti}\n\nGRIGLIA:\n{griglia}"
             + _frame_block(frame))
    t0 = time.time()
    try:
        parsed = _chat_json([SystemMessage(content=_STRUCTURED_GRID_SYSTEM), HumanMessage(content=human)],
                            max_tokens=1800, temperature=0)
    except Exception as exc:
        logger.warning("structured analysis, grid: failed: %s", exc)
        return None
    finally:
        vlog("structured_analysis_grid_llm", duration_ms=(time.time() - t0) * 1000)
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else None


def _grid_questions(grid: List[tuple], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The pertinent slots as _analyse_case_v2's "questioni", in grid order.
    Unknown slot ids, non-pertinent slots and slots with nothing to search
    with are left out; never a model-given article number."""
    by_slot = {str(item.get("voce") or "").strip(): item for item in items}
    questions = []
    for slot, label, _, categoria in grid:
        item = by_slot.get(slot)
        if not item or item.get("pertinente") is not True:
            continue
        istituti = [{"nome": str(i.get("nome") or "").strip(), "codice": str(i.get("codice") or "").strip(),
                     "numero_indicativo": ""}
                    for i in (item.get("istituti") if isinstance(item.get("istituti"), list) else [])
                    if isinstance(i, dict) and str(i.get("nome") or "").strip()][:3]
        parole = _as_list(item.get("parole_chiave"))[:6]
        if not istituti and not parole:
            continue
        valutazione = " ".join(str(item.get("valutazione") or "").split())
        questions.append({"titolo": f"{label}: {valutazione}"[:220] if valutazione else label,
                          "categoria": categoria, "istituti": istituti, "parole_chiave": parole, "voce": slot})
    return questions[:_STRUCTURED_MAX_QUESTIONS]


def _analyse_case_structured(facts: str, lang: str, request: str = "",
                             frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The case analysis in two short calls (facts, then the area's grid),
    returned in _analyse_case_v2's shape. Falls back to _analyse_case_v2 if
    either call fails or no grid slot is pertinent."""
    extracted = _structured_facts(facts, request, frame)
    area = str((extracted or {}).get("area") or "").strip().lower()
    if extracted and area == "amministrativo" and _DAMAGES_RE.search(facts):
        area = "civile"
    grid = _central_slot(frame) + _STRUCTURED_GRIDS[_GRID_FOR_AREA.get(area, "civile")]
    items = _structured_grid(extracted, grid, frame) if extracted else None
    questions = _grid_questions(grid, items) if items else []
    vlog("structured_analysis", {"area": area, "facts": len((extracted or {}).get("fatti") or []),
                                 "pertinent": [q["voce"] for q in questions],
                                 "institutions": [i["nome"] for q in questions for i in q["istituti"]],
                                 "fallback": not questions})
    if not questions:
        logger.warning("structured analysis gave no pertinent issue, falling back to the v2 analysis")
        return _analyse_case_v2(facts, lang, request, **({"frame": frame} if frame else {}))
    fatti = [{"fatto": str(f.get("fatto")).strip(), "effetto": str(f.get("effetto") or "neutro").strip(),
              "perche": str(f.get("perche") or "").strip()} for f in extracted["fatti"]][:15]
    model: Dict[str, List[str]] = {}
    for f in extracted["fatti"]:
        key = _FACT_TYPE_TO_MODEL_KEY.get(str(f.get("tipo") or "").strip().lower())
        if key:
            model.setdefault(key, []).append(str(f.get("fatto")).strip())
    return {"parte_assistita": str(extracted.get("parte_assistita") or "").strip(),
            "posizione": str(extracted.get("posizione") or "").strip(), "area": area,
            "richiesta": str(extracted.get("richiesta") or "").strip(),
            "ricerca_fatti": _as_list(extracted.get("ricerca_fatti"))[:8], "questioni": questions,
            "fatti": fatti, "fatti_strutturati": {k: "; ".join(v) for k, v in model.items()}}


# --- CASE_FRAME_ENABLED ---------------------------------------------------------
#
# r13 (21 Oct), lawyer feedback on held-out drafts: the issue a case is built
# around was never identified (an attempt against mere preparatory acts; the
# execution of an administrative judgment and the nullity of acts eluding it),
# and the draft never answered the questions the case asked. One call, before
# the analysis, temperature 0, reads the case's frame: its facts, its procedural
# posture (in words and as one of _POSTURE_TYPES, which METHOD_CANDIDATES_ENABLED
# reads), its areas, its explicit questions close to verbatim, and the
# institutions the case could be built around, in words, never by number. The
# questions, the posture and the central issues go into the analysis prompt and
# are searched as their own queries by the fusion pool; the central issues are
# also resolved as institutions (_with_central_questions) and the questions are
# given to the draft (_parere_human). Fails open: no frame, nothing changes.
CASE_FRAME_FLAG = "CASE_FRAME_ENABLED"
_FRAME_MAX_TASKS = 6
_FRAME_MAX_CENTRAL = 3
_FRAME_MAX_AREAS = 3
_FRAME_MAX_FACTS = 20
_POSTURE_TYPES = (
    "precontenzioso", "giudizio_civile_primo_grado", "sentenza_civile_primo_grado", "sentenza_civile_appello",
    "esecuzione_civile", "indagini_preliminari", "giudizio_penale_primo_grado", "condanna_primo_grado",
    "sentenza_penale_appello", "provvedimento_amministrativo", "silenzio_amministrazione",
    "giudicato_amministrativo_non_eseguito", "altro",
)
_FRAME_AREAS = ("penale", "contratti", "responsabilita_civile", "locazione", "successioni", "diritti_reali",
                "famiglia", "consumo", "amministrativo", "lavoro", "tributario", "altro")

_CASE_FRAME_SYSTEM = (
    "Sei un avvocato italiano. Leggi il caso e restituisci SOLO un oggetto JSON con queste chiavi:\n"
    '- "facts": i fatti del caso uno per uno, in forma neutra e breve, senza valutazioni\n'
    '- "procedural_posture": in una frase, a che punto è la vicenda (per esempio "condanna in primo grado, '
    'termine per l\'appello in corso", "sentenza amministrativa passata in giudicato e non eseguita", '
    '"indagini preliminari", "controversia contrattuale prima del giudizio")\n'
    '- "posture_type": quella stessa fase, una tra ' + ", ".join(f'"{p}"' for p in _POSTURE_TYPES) + "\n"
    '- "areas": da 1 a 3 aree del diritto che la vicenda tocca, tra '
    + ", ".join(f'"{a}"' for a in _FRAME_AREAS) + "\n"
    '- "tasks": ogni domanda o richiesta esplicita che il caso rivolge a chi deve rispondere (per esempio '
    '"quale reato è configurabile", "quali rimedi ha il cliente", "redigere l\'atto di appello"), con parole '
    "il più possibile vicine al testo, una per voce; nessuna domanda che il testo non pone\n"
    '- "central_issue_candidates": da 1 a 3 istituti giuridici attorno a cui il caso è costruito, cioè quelli '
    "su cui la risposta si decide, anche se il testo non li nomina, ciascuno {\"nome\": il nome dell'istituto "
    "con le parole con cui è intitolato nel codice o nella legge, \"codice\": c.c., c.p.c., c.p., c.p.p., "
    "c.p.a., c.d.s., o il nome per intero di un'altra legge}\n"
    "Mai un numero di articolo. Non inventare fatti. Rispondi SOLO con il JSON."
)


def _case_frame(facts: str, request: str) -> Optional[Dict[str, Any]]:
    """The case's frame (see CASE_FRAME_ENABLED), normalised; None on any
    failure or when it holds neither questions nor central issues."""
    t0 = time.time()
    try:
        parsed = _chat_json([SystemMessage(content=_CASE_FRAME_SYSTEM),
                             HumanMessage(content=_case_text(facts, request))],
                            max_tokens=1500, temperature=0)
    except Exception as exc:
        logger.warning("case frame failed: %s", exc)
        return None
    finally:
        vlog("case_frame_llm", duration_ms=(time.time() - t0) * 1000)
    if not isinstance(parsed, dict):
        return None
    posture_type = str(parsed.get("posture_type") or "").strip().lower()
    central = []
    for item in parsed.get("central_issue_candidates") if isinstance(parsed.get("central_issue_candidates"),
                                                                     list) else []:
        nome = str(item.get("nome") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        codice = str(item.get("codice") or "").strip() if isinstance(item, dict) else ""
        if nome and not _CITED_ARTICLE_RE.search(nome):
            central.append({"nome": nome, "codice": codice})
    frame = {
        "facts": _as_list(parsed.get("facts") if isinstance(parsed.get("facts"), list) else [])[:_FRAME_MAX_FACTS],
        "procedural_posture": " ".join(str(parsed.get("procedural_posture") or "").split()),
        "posture_type": posture_type if posture_type in _POSTURE_TYPES else "altro",
        "areas": [a for a in (x.lower() for x in _as_list(parsed.get("areas"))) if a in _FRAME_AREAS][:_FRAME_MAX_AREAS],
        "tasks": list(dict.fromkeys(" ".join(t.split()) for t in _as_list(
            parsed.get("tasks") if isinstance(parsed.get("tasks"), list) else [])))[:_FRAME_MAX_TASKS],
        "central_issue_candidates": central[:_FRAME_MAX_CENTRAL],
    }
    vlog("case_frame", {k: frame[k] for k in ("procedural_posture", "posture_type", "areas", "tasks",
                                             "central_issue_candidates")})
    if not frame["tasks"] and not frame["central_issue_candidates"]:
        return None
    if feature_flags.enabled(FRAME_SANITY_FLAG):
        frame = _sane_frame(frame, _case_text(facts, request))
    return frame


# FRAME_SANITY_ENABLED (diagnose_long, 25 Oct): the frame's areas and posture set the scaffolding (method
# lists, checklist, the vote's area filter, the plan's area rule), and the model got them wrong on
# criminal cases: h6 (a "denuncia-querela" against a surgeon) -> responsabilita_civile, so the vote's
# filter dropped every c.p. article; n1 (usury, "difensore dell'imputato") -> contratti; h5/h9
# (convictions to appeal) -> sentenza_civile_primo_grado, civil appeal articles cited; p2 ("Quali reati
# sono configurabili?") -> precontenzioso. Deterministic signals in the text now override: a criminal
# proceeding or conviction, or an offence the text asks about, puts "penale" among the areas and, for
# a civil (or "altro") posture, a criminal one; the areas are then marked uncertain, and the vote's
# area filter drops no code (vote_pool.filter_areas). A frame with an administrative posture is left as
# it is: an exclusion from a tender for a director's conviction (n4) stays an administrative case.
FRAME_SANITY_FLAG = "FRAME_SANITY_ENABLED"
_CRIMINAL_PROCEEDING_RE = re.compile(
    r"\bimputat\w*|\bindagat\w*|\bquerel[a-zà-ù]*\b(?!\s+di\s+falso)|\bsporg\w*\s+denuncia|pubblico\s+ministero"
    r"|procura\s+della\s+repubblica|rinvio\s+a\s+giudizio|capo\s+d['’]\s?imputazione"
    r"|giudizio\s+(?:immediato|abbreviato|direttissimo)|decreto\s+penale|\barrest\w*|\breclusione\b"
    r"|\bpena\s+(?:detentiva|pecuniaria|di\s+giustizia)", re.IGNORECASE)
_CRIMINAL_CONVICTION_RE = re.compile(
    r"\bcondannat\w*\s+(?:per\b|alla\s+(?:pena|reclusione)|a\s+\d+\s+(?:anni|mesi))"
    r"|\bcondanna\s+(?:penale\b|[^.;]{0,40}\b(?:alla\s+(?:pena|reclusione)|per\s+(?:il\s+)?(?:reato|delitto)))"
    r"|\binflig\w*[^.;]{0,40}\bpen[ae]\b|\bgiudicat\w*\s+responsabil\w*[^.;]{0,80}\b(?:delitt|reat)"
    r"|\briconosciut\w*\s+colpevol", re.IGNORECASE)
_CRIMINAL_OFFENCE_RE = re.compile(
    r"\breat[oi]\b|\bdelitt[oi]\b|\bcontravvenzion\w*|\bartt?\.?\s*\d+(?:-\w+)?\s*,?\s*c\.?\s?p\.?(?:\s?p\.?)?(?!\s?[ac]\b)",
    re.IGNORECASE)
_CRIMINAL_QUESTION_RE = re.compile(
    r"\bqual[ie]?\s+(?:reat|delitt|fattispecie)|(?:reat|delitt)\w*[^.;?]{0,40}configurabil"
    r"|configurabil\w*[^.;?]{0,40}(?:reat|delitt)|responsabilit[àa]\s+penale|rilevanza\s+penale|profil\w*\s+penal",
    re.IGNORECASE)
_CRIMINAL_TRIAL_RE = re.compile(
    r"\bimputat\w*|rinvio\s+a\s+giudizio|dibattiment\w*|capo\s+d['’]\s?imputazione"
    r"|giudizio\s+(?:immediato|abbreviato|direttissimo)|decreto\s+che\s+dispone\s+il\s+giudizio", re.IGNORECASE)
_CRIMINAL_APPEAL_LEVEL_RE = re.compile(
    r"corte\s+d['’]\s?appello[^.;]{0,60}\b(?:conferm|condann|riform|rigett)\w*"
    r"|(?:conferm|condann)\w*[^.;]{0,40}\bin\s+(?:appello|secondo\s+grado)|sentenza\s+(?:di\s+secondo\s+grado|d['’]\s?appello)",
    re.IGNORECASE)


def _sane_frame(frame: Dict[str, Any], case_text: str) -> Dict[str, Any]:
    """The frame with FRAME_SANITY_ENABLED's overrides (see above); unchanged when none applies."""
    from . import method_candidates

    convicted = bool(_CRIMINAL_CONVICTION_RE.search(case_text))
    criminal = convicted or bool(_CRIMINAL_PROCEEDING_RE.search(case_text)) or (
        bool(_CRIMINAL_OFFENCE_RE.search(case_text)) and bool(_CRIMINAL_QUESTION_RE.search(case_text)))
    if not criminal or (frame.get("posture_type") or "") in method_candidates.ADMINISTRATIVE_POSTURES:
        return frame
    areas, posture, overrides = list(frame.get("areas") or []), frame.get("posture_type") or "altro", []
    if "penale" not in areas:
        areas = ["penale"] + areas[:_FRAME_MAX_AREAS - 1]
        overrides.append("area penale added")
    if posture == "altro" or posture in method_candidates.CIVIL_POSTURES:
        new = (("sentenza_penale_appello" if _CRIMINAL_APPEAL_LEVEL_RE.search(case_text) else "condanna_primo_grado")
               if convicted else "giudizio_penale_primo_grado" if _CRIMINAL_TRIAL_RE.search(case_text)
               else "indagini_preliminari")
        overrides.append(f"posture {posture} -> {new}")
        posture = new
    if not overrides:
        return frame
    vlog("case_frame_sanity", {"overrides": overrides, "areas": areas, "posture_type": posture})
    return {**frame, "areas": areas, "posture_type": posture, "areas_uncertain": True, "sanity": overrides}


def _frame_block(frame: Optional[Dict[str, Any]]) -> str:
    """The frame as a block appended to an analysis call's human message, or ""."""
    if not frame:
        return ""
    lines = ["\n\nQUADRO DEL CASO (estratto in precedenza, da verificare sui fatti):"]
    if frame.get("procedural_posture"):
        lines.append(f"Fase: {frame['procedural_posture']}")
    if frame.get("tasks"):
        lines.append("Quesiti posti dal caso:")
        lines += [f"- {t}" for t in frame["tasks"]]
    if frame.get("central_issue_candidates"):
        lines.append("Questioni centrali candidate:")
        lines += [f"- {c['nome']} ({c['codice']})" if c.get("codice") else f"- {c['nome']}"
                  for c in frame["central_issue_candidates"]]
    if frame.get("client_role"):             # FACT_REGIMES_ENABLED
        lines.append(f"Ruolo della parte assistita: {frame['client_role']}")
    if frame.get("qualifications"):          # FACT_REGIMES_ENABLED
        lines.append("Qualifiche delle parti dedotte dai fatti (con i regimi speciali che comportano):")
        lines.append(frame["qualifications"])
    lines.append("Le questioni devono comprendere le questioni centrali candidate che i fatti pongono davvero e "
                 "servire a rispondere a ciascun quesito, tenendo conto della fase.")
    return "\n".join(lines)


def _frame_queries(frame: Dict[str, Any]) -> List[str]:
    """The frame's own search queries: each question, the posture, each
    central issue -- deduplicated, in that order."""
    queries = list(frame.get("tasks") or []) + [frame.get("procedural_posture") or ""] + \
        [c["nome"] for c in frame.get("central_issue_candidates") or []]
    return list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))


def _analysis_issue_titles(analysis: Dict[str, Any]) -> List[str]:
    """The analysis' own issues as search queries (HYBRID_NARRATIVE_ENABLED): each question's title without
    the "Questione centrale:" label the frame's issues carry (those are searched as the frame's queries)."""
    titles = [str(q.get("titolo") or "").strip() for q in analysis.get("questioni") or [] if isinstance(q, dict)
              and q.get("voce") != "questione_centrale"]
    return list(dict.fromkeys(t for t in titles if t))


def _same_institution(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def _with_central_questions(analysis: Dict[str, Any], frame: Dict[str, Any]) -> Dict[str, Any]:
    """analysis with one question per central issue the analysis does not
    already name as an institution, first -- so _resolve_institution, the
    rulings search and the draft's issue list all see it. Only for the v2
    shape (questions with "istituti")."""
    named = [i.get("nome") or "" for q in analysis.get("questioni") or [] for i in q.get("istituti") or []]
    added = [{"titolo": f"Questione centrale: {c['nome']}", "categoria": "qualificazione",
              "istituti": [{"nome": c["nome"], "codice": c.get("codice") or "", "numero_indicativo": "",
                            **{k: c[k] for k in ("fonte", "ranges", "articolo") if c.get(k)}}],
              "parole_chiave": [], "voce": "questione_centrale"}
             for c in frame.get("central_issue_candidates") or []
             if not any(_same_institution(c["nome"], n) for n in named)]
    # STRICT_ARTICLES_ENABLED: an institution the analysis names itself takes the code table's article and code
    # (the boat case, 26 Oct: the analysis named "foro del consumatore" on its own, so the regime's art. 33 Cod.
    # cons. never reached the resolver and the answer cited art. 24 Cost.).
    tables = [c for c in frame.get("central_issue_candidates") or [] if isinstance(c, dict) and c.get("articolo")]
    questions = list(analysis.get("questioni") or [])
    if tables:
        questions = [{**q, "istituti": [_with_table_article(i, tables) for i in q["istituti"]]}
                     if isinstance(q.get("istituti"), list) else q for q in questions]
    if not added and questions == list(analysis.get("questioni") or []):
        return analysis
    return {**analysis, "questioni": added + questions}


def _with_table_article(istituto: Dict[str, Any], tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    table = next((c for c in tables if isinstance(istituto, dict)
                  and _same_institution(c["nome"], istituto.get("nome") or "")), None)
    if table is None or istituto.get("articolo"):
        return istituto
    return {**istituto, "articolo": table["articolo"], "codice": table.get("codice") or istituto.get("codice") or ""}


def _analyse_case_for_pipeline(facts: str, lang: str, request: str = "") -> Dict[str, Any]:
    """The analysis the strategy pipeline uses, by flag; shared with the
    diagnostic scripts so they see what the pipeline does. CASE_FRAME_ENABLED
    reads the case's frame first and gives it to whichever analysis runs; the
    result then carries it as "frame"."""
    from . import fact_signals
    signals = None
    if fact_signals.enabled() or fact_signals.regimes_enabled():
        frame, signals = _frame_and_fact_signals(facts, request)
    else:
        frame = _case_frame(facts, request) if feature_flags.enabled(CASE_FRAME_FLAG) else None
    extra = {"frame": frame} if frame else {}
    if not feature_flags.enabled(CASE_ANALYSIS_FLAG):
        analysis = _analyse_case(facts, lang, request, **extra)
    elif feature_flags.enabled(STRUCTURED_ANALYSIS_FLAG):
        analysis = _analyse_case_structured(facts, lang, request, **extra)
    else:
        analysis = _analyse_case_v2(facts, lang, request, **extra)
    if signals is not None:
        analysis = {**analysis, "fact_signals": signals}
    if not frame:
        return analysis
    if feature_flags.enabled(CASE_ANALYSIS_FLAG):
        analysis = _with_central_questions(analysis, frame)
    return {**analysis, "frame": frame}


def _frame_and_fact_signals(facts: str, request: str) -> tuple:
    """FACT_SIGNALS_ENABLED or FACT_REGIMES_ENABLED: (frame, signals) -- the frame (CASE_FRAME_ENABLED) and the
    typed facts read at once, then the fact rules fired within the frame's areas (FACT_SIGNALS_ENABLED); their
    institutions and the regimes of the parties' qualifications (FACT_REGIMES_ENABLED) join the frame's central
    issues (fact_signals.all_central), so the analysis takes them as questions, they are resolved (within the
    rule's article ranges: fact_signals.keep_resolved) and tagged _central; the qualifications go to the frame
    as "qualifications" (_frame_block). signals None on failure."""
    from concurrent.futures import ThreadPoolExecutor
    from . import fact_signals

    def read_facts():
        try:
            return fact_signals.extract_facts(_case_text(facts, request))
        except Exception as exc:
            logger.warning("fact signals failed, the case is read without them: %s", exc)
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        frame_job = pool.submit(_case_frame, facts, request) if feature_flags.enabled(CASE_FRAME_FLAG) else None
        read = pool.submit(read_facts).result()
        frame = frame_job.result() if frame_job else None
    signals = None
    if read is not None:
        try:
            signals = fact_signals.signals_from(read[0], read[1], (frame or {}).get("areas") or [],
                                                request=request)
        except Exception as exc:
            logger.warning("fact signals: rules not applied: %s", exc)
    if signals is not None and fact_signals.regimes_enabled():
        signals["posture"] = fact_signals.claim_posture(_case_text(facts, request), signals.get("facts"))
        frame = _claimant_frame(frame, signals["posture"], signals.get("facts"))
    added = fact_signals.all_central(signals)
    qualified = fact_signals.qualifications_block(signals)
    if qualified:
        frame = {**(frame or {}), "qualifications": qualified}
    if added:
        frame = dict(frame or {})
        have = [c.get("nome") or "" for c in frame.get("central_issue_candidates") or [] if isinstance(c, dict)]
        frame["central_issue_candidates"] = list(frame.get("central_issue_candidates") or []) + [
            c for c in added if not any(_same_institution(c["nome"], h) for h in have)]
    return frame, signals


# FACT_REGIMES_ENABLED (the lawyer's review of the pool case's answer: clients seeking damages got the
# "respond to a claim received" scaffolding): a claimant client's frame says so; a posture text about answering a
# claim is replaced when the case says no claim or act was received; no posture at all, with no proceeding, is the
# civil pre-litigation one.
_ANSWERING_RE = re.compile(r"ricev|rispond|rispost|pretesa\s+(?:avversaria|altrui|della controparte)|difender\w*\s+da",
                           re.IGNORECASE)


def _claimant_frame(frame: Optional[Dict[str, Any]], posture: Dict[str, Any],
                    facts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not frame or not posture.get("claimant") or posture.get("received"):
        return frame
    out = {**frame, "client_role": "attore: la parte assistita intende far valere una pretesa (risarcimento o azione)"}
    changed = []
    if _ANSWERING_RE.search(frame.get("procedural_posture") or ""):
        out["procedural_posture"] = ("Nessun giudizio avviato e nessuna pretesa ricevuta: la parte assistita intende "
                                     "far valere le proprie pretese.")
        changed.append("posture text")
    if (frame.get("posture_type") or "altro") == "altro" and not (facts or {}).get("proceedings") and any(
            a not in ("penale", "amministrativo") for a in frame.get("areas") or []):
        out["posture_type"] = "precontenzioso"
        changed.append("posture type")
    vlog("case_frame_claimant", {"why": posture.get("why"), "changed": changed})
    return out


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
    """'415bis' / '415-BIS' / '2051.1' / '21-novies' -> '415-bis' / '2051' / '21-nonies' (the
    canonical spelling the BE parser writes, article_suffix.py)."""
    return normalize_number(raw.lower().split(".")[0])


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
    t0 = time.time()
    try:
        chosen = _chat_json([SystemMessage(content=_ARTICLE_CHOICE_SYSTEM),
                            HumanMessage(content=f"CASO:\n{facts[:6000]}\n\nARTICOLI CANDIDATI:\n{listing}")],
                           max_tokens=200, temperature=0)
    except Exception as exc:
        logger.warning("article choice failed: %s", exc)
        return None
    finally:
        vlog("article_choice_llm", duration_ms=(time.time() - t0) * 1000)
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


# Which duplicate Section wins, and an article's real text when its "N.0.0" holds only the date its wording
# took effect: one rule for every path, in article_text.py.
_section_quality = article_text.section_quality


def _dedupe_sections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per distinct (document, section name), keeping the
    highest-_section_quality duplicate, in first-seen order. The corpus has
    exact duplicate Section nodes per name (same comment elsewhere in this
    file, e.g. _source_block) and Neo4j gives no ordering guarantee among
    them beyond what's asked for."""
    best: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for row in rows:
        key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
        if key not in best:
            order.append(key)
            best[key] = row
        elif _section_quality(row.get("s")) > _section_quality(best[key].get("s")):
            best[key] = row
    return [best[key] for key in order]


def _fetch_articles(session, refs: List[tuple]) -> List[Dict[str, Any]]:
    """The text of each article, read from its code by number: per document, the canonical row carrying the
    article's real text (article_text.article_rows -- "N.0.0" unless it holds only its date in force, else the
    bare "N", else the commi joined), else its dot fragments. Duplicates sharing a Section name are resolved
    by _section_quality, not left to Neo4j's unspecified order."""
    rows = []
    for num, abbr in refs:
        prefix = _CODE_PREFIXES.get(abbr)
        if not prefix:
            continue
        found = []
        # The canonical spelling, then legacy ones ("21-novies") of Sections
        # ingested before the BE normalised suffixes (article_suffix.py).
        for spelling in number_variants(num):
            raw = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
                "AND (s.name = $n OR s.name STARTS WITH $n + '.') "
                "RETURN d, s ORDER BY s.name LIMIT 80",
                prefix=prefix, n=spelling,
            ).data()
            by_doc: Dict[Any, List[Dict[str, Any]]] = {}
            for row in raw:
                by_doc.setdefault((row.get("d") or {}).get("id"), []).append(row)
            found = [r for doc_rows in by_doc.values() for r in article_text.article_rows(spelling, doc_rows)]
            if found:
                break
        for row in found[:6]:
            row["_article"] = (num, abbr)
            rows.append(row)
    return rows


# CASE_ANALYSIS_ENABLED: article resolution beyond the six traditional codes.
def _fetch_any(session, documents: List[Dict[str, Any]], refs: List[tuple]) -> List[Dict[str, Any]]:
    """Article rows for refs naming any document: the six traditional codes
    by the existing fast path (_fetch_articles, an index-friendly prefix
    match), anything else resolved against the real document names — same
    mechanism issue_spotting.py uses for a model-proposed law outside those
    six (Codice del Consumo, a numbered law)."""
    from .issue_spotting import resolve_code, fetch_article

    standard = [r for r in refs if r[1] in _CODE_PREFIXES]
    rows = _fetch_articles(session, standard)
    for number, name in refs:
        if name in _CODE_PREFIXES:
            continue
        doc = resolve_code(name, documents)
        if not doc:
            continue
        found = fetch_article(session, doc["id"], number)
        for row in found:
            row["_article"] = (number, doc["name"])
        rows.extend(found)
    return rows


def _issue_article_refs(session, doc_id: Optional[str], parole_chiave: List[str]) -> List[str]:
    """Article numbers found by full-text search on one issue's keywords,
    scoped to the institution's own document (doc_id) -- never across the
    whole corpus any more.

    Round 8 (19 Oct): an unscoped, corpus-wide version of this search
    returned wrong-code matches on prod for generic doctrinal names that
    don't literally resolve to one document ("causa effettiva" -> 23 c.c.,
    46 L. sul diritto internazionale privato; "dolo indiretto" -> 86 TUEL,
    90 Codice dei medicinali -- both about homicide in a c.p. case). When
    _resolve_institution's own "codice" doesn't resolve to a document,
    there was no document to scope the search to, and nothing then
    filtered out the wrong ones -- a global Lucene query (the terms are
    ORed together) always finds SOME top-10 hit somewhere in the corpus,
    whether or not it's actually about this institution. Unresolved is
    better than wrong, so a caller with no doc_id now gets [] instead of a
    corpus-wide guess.

    Every hit is also re-verified: at least half its stems must appear in
    the matched section's own title or text -- the same discipline
    article_by_title and _number_matches_institution already apply. A
    fulltext score alone is not proof a match is about this institution,
    only that it contains one of its words somewhere."""
    from .issue_spotting import _stems

    if not doc_id:
        return []
    words = [w for w in parole_chiave if isinstance(w, str)]
    terms = _lucene_query(" ".join(words))
    if not terms:
        return []
    stems: set = set()
    for w in words:
        stems |= _stems(w)
    try:
        found = session.run(
            "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
            "RETURN node.name AS sec, node.titolo_articolo AS titolo_articolo, node.title AS title, "
            "node.plain_text AS plain_text, score ORDER BY score DESC LIMIT 10",
            t=terms, doc_id=doc_id,
        ).data()
    except Exception as exc:
        logger.warning("issue search failed for %r: %s", terms, exc)
        return []
    needed = max(1, (len(stems) + 1) // 2)
    numbers: List[str] = []
    for row in found:
        m = re.match(r"^(\d+(?:-[a-z]+)?)(?:[._]|$)", (row.get("sec") or "").strip(), re.IGNORECASE)
        if not m:
            continue
        text_stems = _stems(" ".join(filter(None, [row.get("titolo_articolo"), row.get("title"),
                                                    row.get("plain_text")])))
        if stems and len(stems & text_stems) < needed:
            continue
        number = _article_num(m.group(1))
        if number not in numbers:
            numbers.append(number)
        if len(numbers) == 4:
            break
    return numbers


def _number_matches_institution(session, doc_id: str, number: str, nome: str) -> bool:
    """True if the article's own title, or (when it has none) its own
    operative text, contains every stem of the institution's name — the
    same test issue_spotting.article_by_title uses to find a number from a
    name, run in reverse to check a number the model (not the corpus)
    proposed. True when the institution has no usable words to check, or
    the number matches nothing in the document at all (fetch_article will
    find nothing either, so there is nothing to wrongly trust); False when
    the article exists but neither its title nor its text support the name.

    Round 3, 18 Oct: on prod this fell back to "no title to check, don't
    drop it" for the model's numero_indicativo on a document whose matching
    Section genuinely had no title set, and the model's number (1950 c.c.,
    "regresso del fideiussore") was accepted for an unrelated institution
    ("diritto di revoca del contratto di locazione") purely because there
    was nothing to check it against — exactly the unverified trust this
    function exists to prevent. Checking plain_text when title is empty
    closes that gap without needing every Section to have a title."""
    from .issue_spotting import _stems

    stems = _stems(nome)
    if not stems:
        return True
    rows = session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id = $doc_id "
        "AND (s.name = $n + '.0.0' OR s.name STARTS WITH $n + '.') "
        "RETURN s.title AS title, s.plain_text AS plain_text LIMIT 3",
        doc_id=doc_id, n=number,
    ).data()
    if not rows:
        return True
    for r in rows:
        title = (r.get("title") or "").strip()
        if title:
            if stems <= _stems(title):
                return True
        else:
            text = (r.get("plain_text") or "").strip()
            if text and stems <= _stems(text):
                return True
    return False


RESOLUTION_STRICT_FLAG = "RESOLUTION_STRICT_ENABLED"
# Words that make a rubrica generic when it has no other: many articles in many codes carry it, so a
# match on it alone says nothing about the article (RESOLUTION_STRICT_ENABLED).
_GENERIC_HEADING_WORDS = (
    "risarcimento danno danni prescrizione accessione accessioni disposizioni generali comuni definizioni "
    "definizione oggetto ambito applicazione effetti effetto termine termini competenza sanzioni sanzione "
    "obblighi obbligo norme norma applicabili rinvio decadenza spese casi modalità procedimento finalità "
    "principi principio regole regola diritto diritti disciplina")
_generic_stems_cache: Dict[str, set] = {}
STRICT_ARTICLES_FLAG = "STRICT_ARTICLES_ENABLED"
# STRICT_ARTICLES_ENABLED: words too common in rubriche to tie a keyword's article to an institution (art. 492
# c.c.'s rubrica is "Garanzia"): the generic ones above and these.
_STRICT_GENERIC_WORDS = "garanzia garanzie prestazione prestazioni"


def _strict_generic_stems() -> set:
    if "strict" not in _generic_stems_cache:
        _generic_stems_cache["strict"] = _stems_of(f"{_GENERIC_HEADING_WORDS} {_STRICT_GENERIC_WORDS}")
    return _generic_stems_cache["strict"]


def _stems_of(text: str) -> set:
    from .issue_spotting import _stems
    return _stems(text or "")


def _rows_heading(rows: List[Dict[str, Any]]) -> str:
    """The article's rubrica: titolo_articolo, else title, of its first row that has one."""
    for row in rows or []:
        s = row.get("s") or {}
        heading = " ".join(str(s.get("titolo_articolo") or s.get("title") or "").split())
        if heading:
            return heading
    return ""


def _generic_heading(heading: str) -> bool:
    """True when every content word of the rubrica is a generic one ("Risarcimento dei danni",
    "Prescrizione", "Accessioni"); False for an empty rubrica."""
    if "stems" not in _generic_stems_cache:
        _generic_stems_cache["stems"] = _stems_of(_GENERIC_HEADING_WORDS)
    stems = _stems_of(heading)
    return bool(stems) and stems <= _generic_stems_cache["stems"]


def _resolve_institution(session, documents: List[Dict[str, Any]], istituto: Dict[str, str],
                         parole_chiave: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Article rows for one institution ({nome, codice, numero_indicativo}
    from _analyse_case_v2), found by its name against the real rubriche —
    never by trusting numero_indicativo alone (see _analyse_case_v2's
    docstring for why).

    Order, within the document named by "codice": (1) titolo_articolo (the
    parser's official rubrica, set on ingestion — BE §7.1 — only present on
    documents already re-ingested under it), then (2) s.title (today's
    title, which may be LLM-generated and less reliable, but is all that
    exists for a document not yet re-ingested) — tried against "nome" AND,
    in turn, each of the issue's "parole_chiave" (round 3, 18 Oct: "nesso
    causale"/"causa effettiva" won't title-match a rubrica worded "Rapporto
    di causalità" or "Concorso di cause" even though they're the same
    institution — the model's own words for it often will, one of them,
    even when the institution's own name doesn't); (3, HEADING_INDEX_ENABLED
    only) semantic match against a dedicated heading embedding index
    (_heading_index_match, "nome" only) — catches an institution named with
    no shared words at all with its rubrica ("tentativo" for art. 56's
    "Delitto tentato": one extra title word over a single-stem name fails
    article_by_title's own tolerance rule, even though it's the right
    article); (4) full-text search on "nome" plus every "parole_chiave"
    together (_issue_article_refs) — the
    name alone often doesn't appear in the article's operative text (art.
    43's rubrica is "Elemento psicologico del reato", but naming the
    institution "elemento soggettivo" won't title-match it, and its text
    says "doloso"/"colposo", not "elemento" or "soggettivo"). Round 8, 19
    Oct: this step only ran, and its hits were only trusted, when "codice"
    resolved to a document and the hit matched its name — with no document
    resolved, it used to search the whole corpus unscoped and trust
    whatever a Lucene OR-of-words query happened to rank highest; it is now
    always scoped to that document (nothing is searched without one) and
    every hit is re-verified against the institution's own words before
    being trusted. (5) numero_indicativo, only if the article it names is
    itself titled or worded consistently with "nome" (_number_matches_
    institution) — a wrong guess with no textual support is dropped rather
    than fetched.

    Every outcome is logged: institution -> resolved article (or none) ->
    which of the four found it."""
    from .issue_spotting import resolve_code, article_by_title, fetch_article

    nome = istituto.get("nome", "")
    codice = istituto.get("codice", "")
    if not nome:
        return []

    # Round 5, item 3 (18 Oct): c3 ran 70s/189s/>400s(timeout) across three
    # otherwise-identical repeats, after the per-parola_chiave title-match
    # loop below (round 3, item 2) turned one unindexed CONTAINS scan per
    # institution into up to 2*(1+len(parole_chiave)). This timer, logged
    # per institution, is how to tell that cost apart from the LLM calls'
    # when VERBOSE_LOGGING is on, instead of guessing from a single total.
    t0 = time.time()

    abbr = _code_of(codice) if codice else None
    doc = None
    if abbr:
        prefix = _CODE_PREFIXES[abbr]
        doc = next((d for d in documents if (d.get("name") or "").startswith(prefix)), None)
    elif codice:
        doc = resolve_code(codice, documents)

    def _tag(rows: List[Dict[str, Any]], number: str, heading_match: bool = False) -> List[Dict[str, Any]]:
        for row in rows:
            row["_article"] = (number, abbr or (doc or {}).get("name", codice))
            if heading_match:
                # Round 9 (19 Oct): resolved by the article's own rubrica
                # (titolo_articolo/title), the strongest of the four paths
                # below -- the applicability check must not be able to drop
                # it on its own judgement alone (see _applicability_candidates).
                # Not reusing _source: that field already means other things
                # elsewhere (CROSSREF_SOURCE, CASE_ANALYSIS_SOURCE for
                # priority-trimming) and _select_articles_v2 overwrites it
                # on every institution row it adds.
                row["_heading_match"] = True
        return rows

    def _found(rows: List[Dict[str, Any]], number: str, reason: str) -> List[Dict[str, Any]]:
        vlog("case_analysis_institution", {"institution": nome, "resolved": f"{number} {abbr or codice}",
                                           "reason": reason}, duration_ms=(time.time() - t0) * 1000)
        return rows

    # RESOLUTION_STRICT_ENABLED: a match on a generic rubrica ("Risarcimento dei danni" is art. 81 c.c.'s,
    # on the broken promise of marriage) is trusted only when the model's own number names that article.
    strict = feature_flags.enabled(RESOLUTION_STRICT_FLAG)
    numero_indicativo = _article_num(istituto.get("numero_indicativo", "") or "")

    def _generic_alone(rows: List[Dict[str, Any]], number: str, path: str) -> bool:
        heading = _rows_heading(rows)
        if not strict or number == numero_indicativo or not _generic_heading(heading):
            return False
        vlog("case_analysis_institution_rejected", {"institution": nome, "article": f"{number} {abbr or codice}",
                                                    "path": path, "heading": heading, "why": "generic heading alone"})
        return True

    # STRICT_ARTICLES_ENABLED (the lawyer's review, 26 Oct): an article found by one of the issue's keywords,
    # not by the institution's own name, counts only if its rubrica shares a word with the institution beyond
    # the generic ones -- "garanzia delle cose vendute" was resolved to art. 492 c.c. ("Garanzia", the heir's)
    # by the keyword "garanzia", "risarcimento danni" and "lesioni personali" to art. 1917 c.c. by
    # "responsabilità civile". The full-text path has its own such check (RESOLUTION_STRICT_ENABLED).
    strict_articles = feature_flags.enabled(STRICT_ARTICLES_FLAG)

    def _keyword_unrelated(rows: List[Dict[str, Any]], number: str, term: str, path: str) -> bool:
        if not strict_articles or term == nome or number == numero_indicativo:
            return False
        heading = _rows_heading(rows)
        if (_stems_of(nome) & _stems_of(heading)) - _strict_generic_stems():
            return False
        vlog("case_analysis_institution_rejected", {"institution": nome, "article": f"{number} {abbr or codice}",
                                                    "path": f"{path}, via parola chiave {term!r}", "heading": heading,
                                                    "why": "keyword match shares no specific word with the institution"})
        return True

    # FACT_SIGNALS_ENABLED / STRICT_ARTICLES_ENABLED (the live answer of 26 Oct): a match on a generic rubrica word
    # alone ("usucapione", "prescrizione", "accessioni") counts only inside that word's chapter of the code -- never
    # art. 2689 c.c. ("Usucapione", the transcription of the judgment) for "usucapione di servitù apparente".
    from . import fact_signals
    generic_ranges = strict_articles or fact_signals.enabled()

    def _generic_word_out(rows: List[Dict[str, Any]], number: str, path: str) -> bool:
        if not generic_ranges:             # not even the model's own number: 2689 is a wrong guess too
            return False
        heading = _rows_heading(rows)
        word = fact_signals.generic_out_of_range(nome, abbr or "", number, heading)
        if not word:
            return False
        vlog("case_analysis_institution_rejected", {"institution": nome, "article": f"{number} {abbr or codice}",
                                                    "path": path, "heading": heading,
                                                    "why": f"generic rubrica word {word!r} outside its chapter"})
        return True

    # STRICT_ARTICLES_ENABLED: an institution a code table names the article of (fact_signals.REGIMES' "articolo":
    # "foro del consumatore" is art. 33 Cod. cons., whose rubrica is about unfair terms) is that article.
    table_number = _article_num(str(istituto.get("articolo") or ""))
    if strict_articles and table_number and doc:
        rows = fetch_article(session, doc["id"], table_number)
        if rows:
            return _found(_tag(rows, table_number, heading_match=True), table_number, "code table")

    if doc:
        for term in [nome, *(parole_chiave or [])]:
            if not isinstance(term, str) or not term.strip():
                continue
            for field in ("titolo_articolo", "title"):
                number = article_by_title(session, doc["id"], term, field=field)
                if number:
                    rows = fetch_article(session, doc["id"], number)
                    if rows and not _generic_alone(rows, number, field) and \
                            not _keyword_unrelated(rows, number, term, field) and \
                            not _generic_word_out(rows, number, field):
                        reason = field if term == nome else f"{field}, via parola chiave {term!r}"
                        # Only the official heading exempts a row from the
                        # applicability check: s.title is LLM-written on every
                        # document not yet re-ingested (BE, 16 Oct: "Estorsione"
                        # on art. 612 and on sub-sections of 346, 628, 630).
                        return _found(_tag(rows, number, heading_match=(field == "titolo_articolo")),
                                      number, reason)

    # HEADING_INDEX_ENABLED: "nome" only, not every parola_chiave -- one
    # embedding call per unresolved institution, not up to 1+len(parole_
    # chiave) of them (see _heading_index_match's docstring for what this
    # catches that the word-match loop above structurally cannot).
    if doc and feature_flags.enabled(HEADING_INDEX_FLAG):
        number = _heading_index_match(session, doc["id"], nome)
        if number:
            rows = fetch_article(session, doc["id"], number)
            if rows and not _generic_alone(rows, number, "heading index") and \
                    not _generic_word_out(rows, number, "heading index"):
                return _found(_tag(rows, number, heading_match=True), number,
                             "heading index, verified against its own words")

    numbers = _issue_article_refs(session, doc["id"] if doc else None, [nome, *(parole_chiave or [])])
    if numbers:
        rows = []
        for number in numbers[:2]:
            found = fetch_article(session, doc["id"], number)
            # RESOLUTION_STRICT_ENABLED: a full-text hit counts only if its rubrica shares a content word
            # with the institution (diagnose_long, 25 Oct: concorso -> 544-quinquies c.p., dolo -> 133/118).
            # STRICT_ARTICLES_ENABLED: nor is a shared generic word (the boat case, 26 Oct: "risarcimento danni"
            # -> art. 2767 c.c., the injured party's privilege on the insurer's indemnity).
            shared = _stems_of(nome) & _stems_of(_rows_heading(found))
            if strict_articles:
                shared -= _strict_generic_stems()
            if (strict or strict_articles) and found and not shared:
                vlog("case_analysis_institution_rejected", {
                    "institution": nome, "article": f"{number} {abbr or codice}", "path": "fulltext",
                    "heading": _rows_heading(found), "why": "rubrica shares no word with the institution"})
                continue
            if found and (_generic_alone(found, number, "fulltext") or _generic_word_out(found, number, "fulltext")):
                continue
            rows.extend(_tag(found, number))
        if rows:
            label = ", ".join(dict.fromkeys(r["_article"][0] for r in rows)) if strict else ", ".join(numbers[:2])
            return _found(rows, label, "fulltext, scoped to the institution's own document and "
                                       "verified against its own words")

    numero = _article_num(istituto.get("numero_indicativo", ""))
    if numero and doc and _number_matches_institution(session, doc["id"], numero, nome):
        rows = fetch_article(session, doc["id"], numero)
        if rows and not (generic_ranges and fact_signals.generic_out_of_range(nome, abbr or "", numero,
                                                                              _rows_heading(rows))):
            return _found(_tag(rows, numero), numero, "model's numero_indicativo, verified against its title")

    vlog("case_analysis_institution", {"institution": nome, "resolved": None, "reason": "not found"},
        duration_ms=(time.time() - t0) * 1000)
    return []


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


PLAN_THEN_WRITE_FLAG = "PLAN_THEN_WRITE_ENABLED"

# The corpus' rulings are Italgiure files ("snpen@s50@a2026@n00190@tS.clean.pdf",
# "snciv_s30_a2026_n12601_tO"): archive, section and sub-section, year, number, act type.
_ITALGIURE_RE = re.compile(r"(snpen|snciv)[@_]s(\w)(\w)[@_]a(\d{4})[@_]n(\d+)[@_]t([a-z])", re.IGNORECASE)
_ITALGIURE_SECTIONS = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII",
                       "F": "fer.", "L": "L.", "T": "T."}
_ITALGIURE_ACTS = {"S": "sent.", "O": "ord.", "I": "ord. interl.", "D": "decr."}


def _ruling_reference(row: Dict[str, Any]) -> str:
    """'Cass. pen., sez. V, sent. n. 190/2026' from the ruling's Italgiure file name, else ''.
    No date: the documents' date fields are often empty or odd (scripts/exp_ruling_cards.py)."""
    d = row.get("d") or {}
    for name in (d.get("file_name"), d.get("name")):
        m = _ITALGIURE_RE.search(str(name or ""))
        if not m:
            continue
        archive, section, sub, year, number, act = m.groups()
        court = "Cass. pen." if archive.lower() == "snpen" else "Cass. civ."
        if section.upper() == "U":
            chamber = "Sez. U."
        else:
            chamber = "sez. " + _ITALGIURE_SECTIONS.get(section.upper(), section.upper() + ".")
            chamber += f"-{sub.upper()}" if sub != "0" else ""
        act_label = _ITALGIURE_ACTS.get(act.upper())
        return f"{court}, {chamber}, " + (f"{act_label} " if act_label else "") + f"n. {int(number)}/{year}"
    return ""


def _ruling_label(row: Dict[str, Any]) -> str:
    """The name a ruling goes by in the prompts and the sources; with PLAN_THEN_WRITE_ENABLED its
    full reference when the file name gives one (rulings only with a full reference: the lawyer's
    review, 22 Oct), so every step names and matches it the same way."""
    if feature_flags.enabled(PLAN_THEN_WRITE_FLAG):
        reference = _ruling_reference(row)
        if reference:
            return reference
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


def _grouped_source_block(article_rows: List[Dict[str, Any]], questioni: List[Dict[str, Any]],
                          chars: int) -> str:
    """Like _source_block, but with one heading per legal issue, for rows
    _select_articles_v2 tagged with _question (their index in `questioni`).
    Rows with no _question (cited by the user, added by cross-reference) go
    in a trailing "Altri articoli" block. Falls back to one flat block when
    no row carries _question — the v1 path (_select_articles), unaffected."""
    if not any("_question" in r for r in article_rows):
        return _source_block(article_rows, chars)
    by_question: Dict[int, List[Dict[str, Any]]] = {}
    other: List[Dict[str, Any]] = []
    for row in article_rows:
        idx = row.get("_question")
        if idx is None:
            other.append(row)
        else:
            by_question.setdefault(idx, []).append(row)
    blocks = []
    for idx, q in enumerate(questioni):
        rows = by_question.get(idx)
        if not rows:
            continue
        title = q.get("titolo") or f"Questione {idx + 1}"
        blocks.append(f"### {title}\n{_source_block(rows, chars)}")
    if other:
        blocks.append(f"### Altri articoli\n{_source_block(other, chars)}")
    return "\n\n".join(blocks)


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
    if not area:
        # Unknown branch: nothing is "the other" one. Prod tests, 23-24 Oct: when the case
        # analysis failed (its fallback has area ""), every criminal ruling of a stalking case
        # was dropped here and the parere was written from 1 ruling, or none.
        return False
    codes = [code for text in texts for _, code in _refs_in(text)]
    penal = sum(code in _PENAL_CODES for code in codes)
    civil = sum(code in _CIVIL_CODES for code in codes)
    return civil > penal if area == "penale" else penal > civil


def _case_branch(analysis: Dict[str, Any]) -> str:
    """The analysis' area, else the case frame's ("penale" stays, "amministrativo" stays, any
    other frame area is civil), else "" (unknown)."""
    if analysis.get("area"):
        return analysis["area"]
    areas = (analysis.get("frame") or {}).get("areas") or []
    if "penale" in areas:
        return "penale"
    if areas:
        return "amministrativo" if all(a == "amministrativo" for a in areas) else "civile"
    return ""


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
        if _other_branch(texts, _case_branch(analysis)):
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


RULING_FIT_FLAG = "RULING_FIT_ENABLED"
# RULING_FIT_ENABLED (the lawyer's review, 26 Oct): the pool case cited a ruling on asylum for minors, the boat
# case one on a seller who hid the defects -- applied as if Marex had hidden them, which the case never says.
# The writer's rule (cite a ruling only where the case's facts meet its conditions) did not stop either.
# Round 2 (26 Oct): asked only whether a fact of the case meets the ruling's conditions, the check dropped every
# ruling of the boat case, the general rules with the wrong ones ("a letter interrupts the limitation period").
_RULING_FIT_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati un CASO e alcuni passi di pronunce, ciascuno con un "
    "identificativo [S1], [S2]... Per ogni pronuncia indica a che cosa può servire in un parere sul CASO:\n"
    "- \"fatti\": il principio si applica perché un fatto che il CASO descrive ne integra i presupposti; indica "
    "quel fatto con le parole del caso;\n"
    "- \"regola\": il principio enuncia una regola generale su una questione giuridica che il CASO pone (lo "
    "stesso istituto), senza presupporre fatti che il CASO non descrive;\n"
    "- \"passo\": il principio indica qualcosa che la parte assistita può fare ora per tutelarsi (un atto, un "
    "termine da interrompere, un giudice da adire); indica quel passo;\n"
    "- \"nessuno\": il principio riguarda un'altra questione giuridica, oppure presuppone fatti che il CASO non "
    "descrive (che una parte abbia nascosto, saputo o fatto qualcosa che il caso non dice).\n"
    "I fatti del giudizio deciso dalla pronuncia non sono fatti del caso: non supporre che nel caso sia accaduto "
    "ciò che è accaduto lì. Rispondi solo con un array JSON, un oggetto per pronuncia: {\"id\": \"S1\", "
    "\"uso\": \"fatti\", \"regola\", \"passo\" o \"nessuno\", \"fatto\": \"il fatto del CASO (per fatti) o il "
    "passo (per passo); vuoto altrimenti\", \"perche\": \"motivo in poche parole\"}."
)
# The live answer of 26 Oct: a ruling that denied the passage (the plot "non può essere considerato intercluso")
# was cited three times in support of a client claiming one. Each ruling's direction for the client's position:
# from its card (outcome, favoured party) when it has one, else in the same call.
_RULING_FIT_DIRECTION = (
    "\nPer ogni pronuncia indica anche \"verso\" rispetto alla POSIZIONE DELLA PARTE ASSISTITA: \"contro\" se "
    "l'esito (la domanda accolta o respinta, chi ha vinto) o il principio va contro ciò che la parte assistita "
    "sostiene o chiede -- nega il diritto che afferma, esclude un presupposto di cui ha bisogno, accoglie la pretesa "
    "che essa contrasta; \"a favore\" se lo sostiene; \"neutro\" altrimenti. Quando il passo riporta l'esito e la "
    "parte favorita della scheda, decidi da quelli. Aggiungi all'oggetto: \"verso\": \"a favore\", \"contro\" o "
    "\"neutro\"."
)
_ADVERSE_NOTE = ("ORIENTAMENTO SFAVOREVOLE alla parte assistita (esito o principio contrario alla sua posizione): "
                 "solo come argomento della controparte, nell'obiezione, con la replica; mai a sostegno.")
_RULING_FIT_NOTES = {
    "fatti": "Fatto del caso che ne integra i presupposti: {}.",
    "regola": "Uso nel parere: regola generale sulla questione, da citare come tale, senza attribuire al caso i "
              "fatti del giudizio deciso.",
    "passo": "Uso nel parere: passo che la parte assistita può compiere ora: {}.",
}


def _adverse_by_card(row: Dict[str, Any]) -> Optional[bool]:
    """True when the ruling's card puts it against the client (its role "sfavorevole"), False when for the
    client ("favorevole"), None without a card or a role (then the check decides)."""
    role = row.get("_card_role")
    return True if role == "sfavorevole" else False if role == "favorevole" else None


def _ruling_fit(facts: str, request: str, ruling_rows: List[Dict[str, Any]],
                analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """RULING_FIT_ENABLED: the rulings that can serve the parere -- a fact of the case meets the principle's
    conditions ("fatti"), it states a general rule on a question the case raises ("regola"), or a step the
    client can take now ("passo") -- each with its use as _fit_use and the fact or step as _fit_fact; one call
    for all. A ruling of no use ("nessuno") goes, and so does a "fatti" one whose fact is not the case's own (most
    of its longer words absent from the case: the boat case's hidden defects). No verdict or an unknown use, a
    failed call or the flag off: the row unchanged. Logged as case_analysis_ruling_fit.
    A kept ruling whose outcome or principle goes against the client's position is marked _fit_adverse (its card's
    role when it has one, else the call's "verso"): only the opponent's argument, with the reply."""
    if not ruling_rows or not feature_flags.enabled(RULING_FIT_FLAG):
        return ruling_rows
    party = _stance_party(analysis or {})
    case = _case_text(facts, request)
    texts = [r.get("_passage") or " ".join(((r.get("s") or {}).get("plain_text") or "").split())[:_PASSAGE_CHARS]
             for r in ruling_rows]
    labels = [_ruling_label(r) for r in ruling_rows]
    passages = "\n\n".join(f"[S{i}] {label}\n{text}" for i, (label, text) in enumerate(zip(labels, texts), 1))
    t0 = time.time()
    try:
        verdicts = _chat_json([SystemMessage(content=_RULING_FIT_SYSTEM + _RULING_FIT_DIRECTION),
                               HumanMessage(content=f"CASO:\n{case[:3000]}\n\nPOSIZIONE DELLA PARTE ASSISTITA: {party}"
                                                    f"\n\nPASSI DELLE PRONUNCE:\n{passages}")],
                              max_tokens=200 + 170 * len(ruling_rows), temperature=0)
    except Exception as exc:
        logger.warning("ruling fit check failed, rulings kept: %s", exc)
        return ruling_rows
    finally:
        vlog("case_analysis_ruling_fit_llm", duration_ms=(time.time() - t0) * 1000)
    by_id = {str(v.get("id") or "").strip().strip("[]").upper(): v for v in verdicts or [] if isinstance(v, dict)} \
        if isinstance(verdicts, list) else {}
    kept, log = [], {"kept": [], "dropped": []}
    for i, (row, label) in enumerate(zip(ruling_rows, labels), 1):
        verdict = by_id.get(f"S{i}") or {}
        use = str(verdict.get("uso") or "").strip().lower()
        fact = " ".join(str(verdict.get("fatto") or "").split())
        why = str(verdict.get("perche") or "")[:200]
        if use == "nessuno":
            log["dropped"].append({"ruling": label, "why": why})
        elif use == "fatti" and (not fact or not _found_in(fact, case)):
            log["dropped"].append({"ruling": label, "why": f"fact not in the case: {fact[:200]}"})
        else:
            by_card = _adverse_by_card(row)
            direction = str(verdict.get("verso") or "").strip().lower()
            adverse = by_card if by_card is not None else direction == "contro"
            how = "card" if by_card is not None else "check"
            entry = {"ruling": label, "use": use if use in _RULING_FIT_NOTES else "(no verdict)"}
            if use in _RULING_FIT_NOTES:
                row = {**row, "_fit_use": use, "_fit_fact": fact}
                entry["fact"] = fact[:200]
            if adverse:
                row = {**row, "_fit_adverse": True}
                entry.update(adverse=True, adverse_by=how, outcome=str(row.get("_card_esito") or "")[:120])
            kept.append(row)
            log["kept"].append(entry)
    log["adverse"] = [e["ruling"] for e in log["kept"] if e.get("adverse")]
    vlog("case_analysis_ruling_fit", log)
    return kept


def _with_fit_facts(ruling_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RULING_FIT_ENABLED: each kept ruling's passage led by what it serves for -- the fact of the case that meets
    its conditions, a general rule, or the step the client can take -- for the plan and the writer (after the
    principles, which read the passages alone)."""
    out = []
    for row in ruling_rows:
        use = row.get("_fit_use")
        notes = [_ADVERSE_NOTE] if row.get("_fit_adverse") else []
        if use in _RULING_FIT_NOTES and (use == "regola" or row.get("_fit_fact")) and not row.get("_fit_adverse"):
            notes.append(_RULING_FIT_NOTES[use].format((row.get("_fit_fact") or "").rstrip(".")))
        if notes:
            passage = row.get("_passage") or " ".join(((row.get("s") or {}).get("plain_text") or "").split())
            row = {**row, "_passage": f"{' '.join(notes)} {passage}".strip()}
        out.append(row)
    return out


def _wrong_code_mark(ref: tuple, known: set) -> str:
    """" [DA VERIFICARE: nelle fonti è l'art. N <code>]" when the sources hold
    this number only under other codes (the draft put it under the wrong one,
    e.g. 585 c.p. for 585 c.p.p.), else the plain mark."""
    others = sorted(code for number, code in known if number == ref[0] and code != ref[1])
    return (f" [DA VERIFICARE: nelle fonti è l'art. {ref[0]} {' / '.join(others)}]" if others
            else _UNVERIFIED)


def _mark_unverified(text: str, articles: List[tuple], sources_text: str, wrong_code: bool = False) -> str:
    """Flag article and decision numbers the retrieved sources do not contain,
    the first time each appears. With wrong_code (STRATEGY_DRAFT_RULES_ENABLED),
    an article whose number the sources hold only under another code says so."""
    known = {(n, a) for n, a in articles}
    flagged, out, pos = set(), [], 0
    for m in _CITED_ARTICLE_RE.finditer(text):
        ref = (_article_num(m.group(1)), _code_of(m.group(2)))
        if ref[1] in _CODE_PREFIXES and ref not in known and ref not in flagged:
            flagged.add(ref)
            out.append(text[pos:m.end()] + (_wrong_code_mark(ref, known) if wrong_code else _UNVERIFIED))
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
# Previewed on production (16 Oct, scripts/preview_strategy.py): asked for a
# concrete procedural section and a sentence range, the model stated them from
# memory and wrongly ("Tribunale" for an art. 584 case, which goes to the Corte
# d'assise; "no alternative procedures"; "libertà vigilata" as a precautionary
# measure; homicide as art. 578, 5-12 years), so section 5 stays as today and
# the rules forbid stating what the sources don't give. A second preview showed
# the list of penal points to "consider" (mental element, causation, alternative
# charges, mitigating circumstances) makes the model invent material: a fact the
# case doesn't contain ("emotional agitation"), art. 59 for a mitigating
# circumstance, art. 582 as an alternative "colposa" charge. Only the rules
# that restrict the draft stay.
STRATEGY_DRAFT_RULES_FLAG = "STRATEGY_DRAFT_RULES_ENABLED"
_PROCEDURAL_SECTION = (
    "**5. Profili processuali e operativi**: solo quelli pertinenti al caso, tra prescrizione o "
    "decadenza, condizioni di procedibilità, competenza, attività istruttorie, terzi da coinvolgere, "
    "possibilità di definizione stragiudiziale.\n"
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
    "- Nella sezione 5 indica che cosa va verificato e perché: non affermare competenze, riti, misure "
    "cautelari, termini o pene che le fonti non riportano.\n"
    # r13 (21 Oct), the lawyer's review of held-out drafts: the questions the
    # case asked were never answered directly; articles cited under the wrong
    # code (585 c.p. for the appeal term of 585 c.p.p.) or from another matter
    # (339 c.p. for the aggravating circumstances of robbery, which are 628 co.
    # 3); arguments that conceded contested elements of the charge; generic
    # sections (witnesses, experts, settlement); case law with no source.
    "- Se il caso pone domande o richieste esplicite (quelle in QUESITI DEL CASO, quando presenti), apri il "
    "parere con **Risposte ai quesiti**, prima della sezione 1: una risposta diretta e motivata a ciascuna, "
    "nell'ordine in cui è posta; le sezioni successive la sviluppano.\n"
    "- Ogni articolo che citi porta il suo codice o la sua legge (\"art. 585 c.p.p.\", mai \"art. 585\" da "
    "solo) e deve trovarsi in NORME o nei passi di GIURISPRUDENZA con quel codice: se lo stesso numero compare "
    "nelle fonti solo per un altro codice, non citarlo. Una circostanza, un termine o un rimedio si cita con "
    "l'articolo che lo disciplina per quel reato o quella pretesa, mai con un articolo di un'altra materia.\n"
    "- Non concedere mai un elemento contestato dell'accusa o della pretesa (il fatto, la condotta, l'elemento "
    "soggettivo, il nesso causale, la qualificazione, il danno): sostieni la tesi della parte assistita; una "
    "posizione che presuppone di concedere un elemento va presentata solo come tesi subordinata (\"in via "
    "subordinata, se si ritenesse provato...\").\n"
    "- Niente sezioni o paragrafi generici: testimoni, consulenti o esperti, accordi stragiudiziali, mediazione "
    "o altre iniziative si indicano solo se i fatti del caso o le fonti li rendono pertinenti, dicendo perché.\n"
    "- Senza una pronuncia presente in GIURISPRUDENZA non richiamare orientamenti o principi giurisprudenziali, "
    "nemmeno in termini generali: ometti il riferimento.\n"
)
# The procedural section under the draft rules: no list of items to fill in.
_PROCEDURAL_SECTION_RULES = (
    "**5. Profili processuali e operativi**: solo quelli che i fatti del caso o le fonti pongono (la fase in cui "
    "si trova la vicenda, gli atti da compiere e i loro termini); nessuna voce generica.\n"
)
_CASE_LAW_RULE = (
    "- Giurisprudenza: cita solo le pronunce presenti in GIURISPRUDENZA, con l'identificativo "
    "indicato tra parentesi quadre; senza fonte puoi richiamare un orientamento solo in termini "
    "generali, senza numeri né date. "
)
_CASE_LAW_RULE_STRICT = (
    "- Giurisprudenza: cita solo le pronunce presenti in GIURISPRUDENZA, con l'identificativo "
    "indicato tra parentesi quadre. "
)


def _parere_system(analysis: Dict[str, Any], lang: str) -> str:
    from .language import language_display_name
    party = analysis.get("parte_assistita") or "la parte assistita"
    position = f" ({analysis['posizione']})" if analysis.get("posizione") else ""
    draft_rules = feature_flags.enabled(STRATEGY_DRAFT_RULES_FLAG)
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
        + (_PROCEDURAL_SECTION_RULES if draft_rules else _PROCEDURAL_SECTION)
        + "**6. Conclusioni**: gli argomenti in ordine di forza e la condotta consigliata.\n\n"
        "REGOLE:\n"
        "- Usa ogni fatto elencato in ANALISI DEI FATTI e spiega perché aiuta o danneggia la parte "
        "assistita. Non aggiungere fatti, date, importi o atti che il caso non contiene: se un dato "
        "manca, indica che cosa va verificato o acquisito.\n"
        "- Norme: cita ogni articolo di NORME con il numero e il codice indicati tra parentesi quadre "
        "prima del suo testo, mai con un altro numero. Riporta tra virgolette solo testi presenti in "
        "NORME; per norme non presenti descrivine il contenuto senza virgolette.\n"
        + (_CASE_LAW_RULE_STRICT if draft_rules else _CASE_LAW_RULE)
        + "I PRINCIPI DALLA GIURISPRUDENZA sono ricavati da quei passi: "
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
    tasks = "\n".join(f"- {t}" for t in ((analysis.get("frame") or {}).get("tasks") or []))
    return (
        f"CASO:\n{_case_text(facts, request)}\n\n"
        + (f"QUESITI DEL CASO:\n{tasks}\n\n" if tasks else "")
        + (f"ANALISI DEI FATTI:\n{facts_list}\n\n" if facts_list else "")
        + (f"QUESTIONI GIURIDICHE:\n{issues}\n\n" if issues else "")
        + f"NORME (testo dalla banca dati):\n{articles or '(nessun testo recuperato)'}\n\n"
        + (f"PRINCIPI DALLA GIURISPRUDENZA (per questione):\n{principles}\n\n" if principles else "")
        + f"GIURISPRUDENZA (passi dalla banca dati):\n{rulings or '(nessuna pronuncia recuperata)'}\n\n"
        + "Redigi ora il parere."
    )


def _estimate_tokens(text: str) -> int:
    return int(len(text or "") / _CHARS_PER_TOKEN) + 1


def _protected_article_row(row: Dict[str, Any], named: set) -> bool:
    """Rows the prompt keeps whole until the last trimming step: named in the
    case, cross-referenced from a selected article, matched by their heading,
    or found by two or more independent signals."""
    from .nodes.cypher import CROSSREF_SOURCE
    return bool(row.get("_article") in named or row.get("_source") == CROSSREF_SOURCE
                or row.get("_heading_match") or row.get("_multi_signal") or row.get("_vote"))


def _protection_rank(row: Dict[str, Any], named: set) -> int:
    """How strongly a row is protected, for the last-resort trim: 0 named in the case,
    1 matched by its heading, 2 cross-referenced, 3 found by several signals or voted by
    similar rulings (VOTE_POOL_ENABLED), 4 none."""
    from .nodes.cypher import CROSSREF_SOURCE
    if row.get("_article") in named:
        return 0
    if row.get("_heading_match"):
        return 1
    if row.get("_source") == CROSSREF_SOURCE:
        return 2
    return 3 if row.get("_multi_signal") or row.get("_vote") else 4


_LAST_RESORT_CHARS = 500
# Each article given back a trimmed prompt's free room: step 3's size for the other articles.
_REFILL_CHARS = 500


def _refill_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Left-out rows in the order a trimmed prompt's free room goes back to them: those the
    applicability check judged applicable (or an alternative), then the method candidates of
    the case's area and posture, then the rest; the pipeline's order within each."""
    def priority(row: Dict[str, Any]) -> int:
        if row.get("_applicability") in ("applicabile", "alternativa"):
            return 0
        return 1 if row.get("_source") == METHOD_CANDIDATES_SOURCE else 2
    return sorted(rows, key=priority)


def _with_text_cut(row: Dict[str, Any], chars: int) -> Dict[str, Any]:
    """A copy of the row whose text, as _source_block reads it, is cut to chars."""
    s = row.get("s") or {}
    text = row.get("_passage") or " ".join((s.get("plain_text") or s.get("abstract") or "").split())
    return {**row, "_passage": text[:chars]}


# (ruling passage chars, rulings kept or None for all, chars of other articles
# with 0 = left out, chars of protected articles), mildest first. Step 0 is the
# prompt as it always was.
_PROMPT_TRIM_STEPS = (
    (_PASSAGE_CHARS + 100, None, _ARTICLE_CHARS, _ARTICLE_CHARS),
    (1200, None, _ARTICLE_CHARS, _ARTICLE_CHARS),
    (600, 5, _ARTICLE_CHARS, _ARTICLE_CHARS),
    (600, 5, 500, _ARTICLE_CHARS),
    (600, 5, 0, _ARTICLE_CHARS),
    (400, 3, 0, 800),
)


def _fit_parere_prompt(system: str, facts: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                       ruling_rows: List[Dict[str, Any]], request: str = "", principles: str = "",
                       budget: int = _DRAFT_INPUT_BUDGET) -> tuple:
    """(articles, rulings, human) for the draft call, trimmed step by step
    until the estimated prompt fits `budget`: the ruling passages first, then
    the text of the other articles, then those articles themselves, then the
    text of protected ones (_protected_article_row), and as a last resort the
    protected articles themselves, least protected first. When the other
    articles were all left out, the room still free goes back to them, one at
    a time and short (_refill_order). The case text and the instructions are
    never cut. A prompt that already fits is exactly the one built before
    this existed."""
    named = set(_refs_in(_case_text(facts, request)))
    questioni = analysis.get("questioni", [])
    for step, (ruling_chars, max_rulings, other_chars, kept_chars) in enumerate(_PROMPT_TRIM_STEPS):
        rows, left_out = [], []
        for row in article_rows:
            chars = kept_chars if _protected_article_row(row, named) else other_chars
            if not chars:
                left_out.append(row.get("_article"))
            else:
                rows.append(row if chars >= _ARTICLE_CHARS else _with_text_cut(row, chars))
        articles = _grouped_source_block(rows, questioni, _ARTICLE_CHARS)
        rulings = _source_block(ruling_rows if max_rulings is None else ruling_rows[:max_rulings], ruling_chars)
        human = _parere_human(facts, analysis, articles, rulings, request, principles)
        total = _estimate_tokens(system) + _estimate_tokens(human)
        if total <= budget or step == len(_PROMPT_TRIM_STEPS) - 1:
            break
    dropped_for_size = 0
    if total > budget:
        # Last resort, never sent over budget. Prod, 21 Oct: after the last step about 48
        # protected articles at 800 characters each still filled 12,936 estimated tokens
        # against a 9,100 budget, the draft call overflowed and the user got "Si è
        # verificato un errore nella generazione della strategia" (11 times, 20-23 Oct).
        # The protected rows are dropped one by one, least protected first (_protection_rank)
        # and, within a rank, latest first; the rest are cut to _LAST_RESORT_CHARS.
        step = len(_PROMPT_TRIM_STEPS)
        ranked = sorted(rows, key=lambda r: _protection_rank(r, named))
        while ranked and total > budget:
            left_out.append(ranked.pop().get("_article"))
            dropped_for_size += 1
            articles = _grouped_source_block([_with_text_cut(r, _LAST_RESORT_CHARS) for r in ranked],
                                             questioni, _ARTICLE_CHARS)
            human = _parere_human(facts, analysis, articles, rulings, request, principles)
            total = _estimate_tokens(system) + _estimate_tokens(human)
    refilled = 0
    if not other_chars and not dropped_for_size and total < budget:
        # Prod, 23 Oct (the PM's five test cases): from step 4 every unprotected article was
        # left out at once -- 84 to 109 of them -- and the prompts ended at 8,481-11,461 of
        # 13,000 tokens while 1218 c.c. (a delay without fault), 438/444 c.p.p. (the rites the
        # user asked about) and 1158/1168/1170 c.c. sat among the left out. The free room goes
        # back to them in _refill_order, one article at a time; one that does not fit is skipped.
        have = {r["_article"] for r in rows if r.get("_article")}
        for row in _refill_order([r for r in article_rows if not _protected_article_row(r, named)]):
            if not row.get("_article") or row["_article"] in have:
                continue
            trial = rows + [_with_text_cut(row, _REFILL_CHARS)]
            trial_articles = _grouped_source_block(trial, questioni, _ARTICLE_CHARS)
            trial_human = _parere_human(facts, analysis, trial_articles, rulings, request, principles)
            trial_total = _estimate_tokens(system) + _estimate_tokens(trial_human)
            if trial_total > budget:
                continue
            rows, articles, human, total = trial, trial_articles, trial_human, trial_total
            have.add(row["_article"])
            refilled += 1
        left_out = [a for a in left_out if a not in have]
    sizes = {"total": total, "budget": budget, "step": step, "instructions": _estimate_tokens(system),
             "case": _estimate_tokens(_case_text(facts, request)), "articles": _estimate_tokens(articles),
             "rulings": _estimate_tokens(rulings), "principles": _estimate_tokens(principles),
             "dropped_for_size": dropped_for_size, "refilled": refilled,
             "articles_left_out": [f"{a[0]} {a[1]}" for a in left_out if a]}
    vlog("case_analysis_prompt_tokens", sizes)
    if step:
        logger.info("draft prompt trimmed to fit the context: %s", sizes)
    return articles, rulings, human


def _is_context_overflow(exc: Exception) -> bool:
    return "maximum context length" in str(exc).lower()


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
    from . import source_tiers

    # HYBRID_NARRATIVE_ENABLED: the articles the rulings agree on are the map's hypotheses
    # (_select_articles_fusion), never a channel of their own here; their searches are skipped.
    hybrid = source_tiers.hybrid_enabled(analysis.get("route"))
    first_rulings = [] if hybrid else _search_case_law(session, analysis)
    similar = [] if hybrid else _search_similar_facts(session, analysis,
                                                      {(r.get("d") or {}).get("id") for r in first_rulings})
    counts = _articles_cited_in_rulings(
        session, [(r.get("d") or {}).get("id") for r in (similar + first_rulings)[:10]]) if not hybrid else {}
    similar_counts = _articles_cited_in_rulings(
        session, [(r.get("d") or {}).get("id") for r in similar]) if not hybrid else {}
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


# Caps how many articles ever reach the draft prompt. Not known to be the
# cause of anything today — checked (prod, 17 Oct): nothing truncates the
# article list itself, only _ARTICLE_CHARS per article's text — but
# _select_articles_v2's union can make article_rows larger than before, so
# this is the safety net asked for "if so, prioritise", sized generously
# (v1 alone could already reach "keep" (uncapped) + 6 chosen).
_MAX_DRAFT_ARTICLES = 24


def _article_priority(row: Dict[str, Any], named_in_case: set) -> int:
    """Lower sorts first = kept preferentially if _MAX_DRAFT_ARTICLES is
    ever exceeded: named by the user > cross-referenced from a selected
    article's own text > resolved from a named institution > everything
    else (ruling agreement, the model's own candidate choice)."""
    from .nodes.cypher import CROSSREF_SOURCE

    article = row.get("_article")
    if article and article in named_in_case:
        return 0
    if row.get("_source") == CROSSREF_SOURCE:
        return 1
    if row.get("_source") == CASE_ANALYSIS_SOURCE:
        return 2
    return 3


def _prioritise_article_rows(article_rows: List[Dict[str, Any]], named_in_case: set,
                             limit: int = _MAX_DRAFT_ARTICLES) -> List[Dict[str, Any]]:
    """article_rows, trimmed to `limit` by _article_priority when it would
    otherwise exceed it — stable within each tier (original order kept)."""
    if len(article_rows) <= limit:
        return article_rows
    ordered = sorted(enumerate(article_rows),
                     key=lambda pair: (_article_priority(pair[1], named_in_case), pair[0]))
    return [row for _, row in ordered[:limit]]


# --- CASE_POOL_FUSION_ENABLED: institution + raw-text + distilled-bullet search, ranked by fusion ---

_ARTICLE_SECTION_RE = re.compile(r"^(\d+)(-[a-z]+)?(?:\.|$)", re.IGNORECASE)

# Round 8, 19 Oct: "document_type = 'primary'" alone let a book through into
# a prod ranked pool ("176 Manuale di diritto processuale penale", its own
# section numbering mistaken for an article) -- its document_type property
# is apparently mislabelled on prod. Reject by name too, the same words
# issue_spotting._NOT_A_LAW rejects for resolve_code: a code or a law's name
# never contains them.
_NOT_A_PRIMARY_SOURCE_RE = re.compile(r"manuale|trattato|commentari|dispens|prontuario|formulario",
                                      re.IGNORECASE)

# FUSION_CLEAN_ENABLED (default false; only acts with CASE_POOL_FUSION_ENABLED):
# scripts/exp_distill.py's FUSION_CLEAN (round 11-13, 20 Oct: 35/47 @20, the
# best channel without HYDE's extra call) -- BASELINE's and DISTILL's text
# searches on primary documents only, minus everything that is not a code or a
# law by name: _NOT_A_PRIMARY_SOURCE_RE's books plus textbooks and the templates
# / contract models the 20 Oct briefing found in the pool (c3: "LOCAZIONE
# ABITATIVA DI NATURA TRANSITORIA"). Deliberately not "contratt" alone: "Codice
# dei contratti pubblici" is a law. The experiment imports this regex, so the
# two cannot drift apart.
FUSION_CLEAN_FLAG = "FUSION_CLEAN_ENABLED"
_NOT_A_LAW_RE = re.compile(
    _NOT_A_PRIMARY_SOURCE_RE.pattern
    + r"|fondamenti|lineamenti|istituzioni di|lezioni di|compendio|corso di diritto|parte generale|parte speciale"
    r"|fac.?simile|template|\bmodell[oi] d|schema di contratto|contratto tipo|scrittura privata|natura transitoria"
    r"|locazione abitativa",
    re.IGNORECASE,
)
# The index is asked for this many times more nodes than kept, so the primary-
# document filter still leaves enough (exp_distill.py's _SCOPED_OVERFETCH), and
# 3k rows are read so k refs survive the name filter.
_CLEAN_OVERFETCH = 5


def _row_to_fusion_ref(row: Dict[str, Any]) -> Optional[tuple]:
    """A Section row from fusion's own vector/fulltext search (the number
    isn't already known, unlike _fetch_articles'/_resolve_institution's
    callers) -> (number, code abbreviation or the document's own name,
    matching _fetch_any's tagging convention) -- or None for a row with no
    parseable article number, no document, or a document that isn't
    actually a code or a law (_NOT_A_PRIMARY_SOURCE_RE)."""
    s, d = row.get("s") or {}, row.get("d") or {}
    name, d_name = s.get("name") or "", d.get("name") or ""
    if not _ARTICLE_SECTION_RE.match(name) or not d_name or _NOT_A_PRIMARY_SOURCE_RE.search(d_name):
        return None
    number = _article_num(name)
    abbr = next((a for a, p in _CODE_PREFIXES.items() if d_name.startswith(p)), None)
    return (number, abbr or d_name)


def _fusion_vector_search(session, text: str, index: str, k: int) -> List[tuple]:
    try:
        embedding = list(embedding_model.embed_query(text))
    except Exception as exc:
        logger.warning("fusion: embedding failed: %s", exc)
        return []
    try:
        rows = session.run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
            "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'primary' "
            "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC",
            index=index, k=k, embedding=embedding,
        ).data()
    except Exception as exc:
        logger.warning("fusion: vector search on index %r failed (does it exist?): %s", index, exc)
        return []
    seen, refs = set(), []
    for row in rows:
        ref = _row_to_fusion_ref({"s": {"name": row.get("name")}, "d": {"name": row.get("doc_name")}})
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _fusion_fulltext_search(session, text: str, k: int) -> List[tuple]:
    from .lookups import _sanitize_fulltext_query
    q = _sanitize_fulltext_query(text)
    if not q:
        return []
    try:
        rows = session.run(
            "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $k}) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'primary' "
            "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC",
            q=q, k=k,
        ).data()
    except Exception as exc:
        logger.warning("fusion: fulltext search failed: %s", exc)
        return []
    seen, refs = set(), []
    for row in rows:
        ref = _row_to_fusion_ref({"s": {"name": row.get("name")}, "d": {"name": row.get("doc_name")}})
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _clean_refs(rows: List[Dict[str, Any]], k: int) -> List[tuple]:
    """rows (best first) -> the first k distinct refs whose document is a code
    or a law by name (_NOT_A_LAW_RE)."""
    seen, refs = set(), []
    for row in rows:
        if _NOT_A_LAW_RE.search(row.get("doc_name") or ""):
            continue
        ref = _row_to_fusion_ref({"s": {"name": row.get("name")}, "d": {"name": row.get("doc_name")}})
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
            if len(refs) == k:
                break
    return refs


def _fusion_clean_searches(session, text: str, index: str, k: int) -> tuple:
    """FUSION_CLEAN's (vector refs, fulltext refs) for `text`: 3k primary-
    document rows each, over-fetched from the index so the document filter
    still leaves them, then cut to k refs that are codes or laws by name --
    scripts/exp_distill.py's _clean_searches, on this module's refs."""
    from .lookups import _sanitize_fulltext_query

    vector = []
    try:
        embedding = list(embedding_model.embed_query(text))
        vector = session.run(
            "CALL db.index.vector.queryNodes($index, $fetch, $embedding) YIELD node, score "
            "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'primary' "
            "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC LIMIT $limit",
            index=index, fetch=3 * k * _CLEAN_OVERFETCH, limit=3 * k, embedding=embedding,
        ).data()
    except Exception as exc:
        logger.warning("fusion clean: vector search on index %r failed: %s", index, exc)
    fulltext = []
    q = _sanitize_fulltext_query(text)
    if q:
        try:
            fulltext = session.run(
                "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $fetch}) YIELD node, score "
                "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'primary' "
                "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC LIMIT $limit",
                q=q, fetch=3 * k * _CLEAN_OVERFETCH, limit=3 * k,
            ).data()
        except Exception as exc:
            logger.warning("fusion clean: fulltext search failed: %s", exc)
    return _clean_refs(vector, k), _clean_refs(fulltext, k)


def _fusion_best_rank_merge(*ref_lists: List[tuple]) -> List[tuple]:
    """Union of refs across the given ranked lists, ordered by the best
    (lowest) rank any one of them gives it.

    Round 8, 19 Oct: _fusion_text_search used to simply concatenate one
    list after the other (every vector hit, however weak, ranked ahead of
    every fulltext-only hit, however strong) instead of a true merge -- a
    diagnosable cause of prod's fusion pool not reproducing scripts/
    exp_distill.py's measured FUSION_BASELINE ranking. This is the same
    algorithm that script's own _merge_best_rank uses; exp_distill.py now
    imports this one directly instead of keeping its own copy, so the two
    can no longer drift apart the way they just did."""
    best = {}
    for ranked in ref_lists:
        for rank, ref in enumerate(ranked, start=1):
            if ref not in best or rank < best[ref]:
                best[ref] = rank
    return sorted(best, key=lambda ref: best[ref])


def _fusion_text_search(session, text: str, index: str, k: int) -> List[tuple]:
    """Best-rank merge (_fusion_best_rank_merge) of vector + fulltext refs
    for `text` -- the pool's "raw-text search" and "distilled-bullet
    search" sources both use this."""
    return _fusion_best_rank_merge(
        _fusion_vector_search(session, text, index, k),
        _fusion_fulltext_search(session, text, k),
    )


def _fusion_hits_and_best_rank(*ref_lists: List[tuple]) -> tuple:
    """({ref: how many of ref_lists contain it}, {ref: best (lowest) rank
    among those that do}) -- the computation _fusion_merge_by_hits ranks
    by, exposed separately so _select_articles_fusion can tell a ref found
    by two or more independent signals from one only a single list
    happened to surface (round 9, 19 Oct: those must survive the cap
    regardless of rank, and are exempt from the applicability check --
    see _FUSION_POOL_CAP and _applicability_candidates)."""
    hits, best_rank = {}, {}
    for refs in ref_lists:
        for rank, ref in enumerate(refs, start=1):
            hits[ref] = hits.get(ref, 0) + 1
            if ref not in best_rank or rank < best_rank[ref]:
                best_rank[ref] = rank
    return hits, best_rank


def _fusion_merge_by_hits(*ref_lists: List[tuple]) -> List[tuple]:
    """Union of refs across the given lists, ranked by (how many of them
    contain it, descending), then best rank among those that do --
    FUSION_BASELINE's ranking, validated on scripts/exp_distill.py (round
    6, 19 Oct): it beat every re-ranking method tried there, at zero added
    latency. Shared with that script's own _merge_by_hits (an alias to
    this function, not a separate copy) for the same reason as
    _fusion_best_rank_merge above."""
    hits, best_rank = _fusion_hits_and_best_rank(*ref_lists)
    return sorted(hits, key=lambda ref: (-hits[ref], best_rank[ref]))


def _fusion_distill_bullets(case_text: str) -> List[Dict[str, str]]:
    parsed = _chat_json([SystemMessage(content=_FUSION_DISTILL_SYSTEM), HumanMessage(content=case_text[:6000])],
                        max_tokens=700, temperature=0)
    if not isinstance(parsed, list):
        return []
    return [{"categoria": str(b.get("categoria") or ""), "testo": str(b.get("testo") or "").strip()}
            for b in parsed if isinstance(b, dict) and str(b.get("testo") or "").strip()][:8]


def _select_articles_fusion(session, driver, database: str, facts: str, request: str,
                            analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Candidate pool = CURRENT (_select_articles_v2's own full "selected":
    v1's code-search + _choose_articles union institution-resolved) union
    BASELINE (vector + fulltext search on the case text itself) union
    DISTILL (vector + fulltext search per distilled bullet, temperature 0,
    statutory vocabulary, no article numbers) -- ranked by fusion
    (_fusion_merge_by_hits) and capped at _FUSION_POOL_CAP. These are the
    exact three sources validated as FUSION_BASELINE on scripts/
    exp_distill.py (round 6, 19 Oct: 27/47 @8, 32/47 @12 of 47 expected
    articles across 12 cases).

    Round 8 (19 Oct): the first version of this function replaced CURRENT
    with institution-resolution alone, to avoid v1's own search +
    _choose_articles call -- a simplification exp_distill.py's own numbers
    never tested, and the actual cause of a prod parity bug (check_
    applicability.py --fusion on Sempronio ranked a textbook and three
    wrong-code institution matches into the top 12, where exp_distill.py's
    FUSION_BASELINE got Sempronio 5/5 @12). Calling _select_articles_v2
    here for CURRENT, instead of reimplementing its logic, is what keeps
    this from drifting the same way again: BASELINE and DISTILL also now
    go through _fusion_best_rank_merge/_fusion_merge_by_hits, which
    scripts/exp_distill.py imports directly rather than keeping its own
    copies. The one deliberate difference from what exp_distill.py
    measures is that BASELINE/DISTILL's own search (_fusion_vector_search/
    _fusion_fulltext_search) is restricted to primary sources and rejects
    book/manual names (_NOT_A_PRIMARY_SOURCE_RE) -- the same prod run's
    other finding; re-running the experiment is how to re-validate the
    recall numbers against that narrower pool.

    Articles named by the user's own text, whatever the rulings agree on
    (_select_articles_v2's own "keep"), and (round 9, 19 Oct) any article
    two or more of CURRENT/BASELINE/DISTILL independently found, are added
    on top of the cap, uncapped -- see _FUSION_POOL_CAP; those multi-signal
    rows are also tagged _multi_signal and, together with rows _resolve_
    institution found by the article's own rubrica (_heading_match), are
    exempt from the applicability check (_applicability_candidates): it may
    prune a single-signal text-search candidate, never a core qualification
    reached this way. Cross-reference broadening expands only from user-
    named and institution-resolved articles, not from the whole capped pool
    (round 8, 19 Oct: broadening from everything pulled in cross-references
    with no connection to what fusion specifically added, e.g. 1, 3, 444,
    450, 470 c.p.p. on Sempronio) -- the same discipline _select_articles_v2
    already applies to its own institution rows.

    r13 (21 Oct): FUSION_CLEAN_ENABLED runs BASELINE's and DISTILL's searches
    as exp_distill.py's FUSION_CLEAN (_fusion_clean_searches: primary
    documents over-fetched, then non-laws by name dropped); CASE_FRAME_ENABLED
    adds the frame's questions, posture and central issues as a fourth signal.

    Behind CASE_POOL_FUSION_ENABLED (requires CASE_ANALYSIS_ENABLED). Used
    by run_case_analysis_pipeline; returns the same keys _select_articles_v2
    does, so run_case_analysis_pipeline and scripts/check_applicability.py
    don't need to special-case which selection path ran."""
    from .doc_lookup import _fetch_doc_names
    from .nodes.retrieval import broaden_article_context

    v2 = _select_articles_v2(session, driver, database, facts, request, analysis)
    keep, named = v2["keep"], v2["named"]
    institutions_resolved = v2.get("institutions_resolved") or []
    heading_matched = set(v2.get("heading_matched") or [])
    central_matched = set(v2.get("central_matched") or [])
    current_refs = list(dict.fromkeys(v2["selected"]))

    documents = _fetch_doc_names(driver, database)
    case = _case_text(facts, request)
    clean = feature_flags.enabled(FUSION_CLEAN_FLAG)
    if clean:
        # exp_distill.py's FUSION_CLEAN exactly: BASELINE merges its two clean
        # lists by best rank, each DISTILL query keeps vector's refs then
        # fulltext's (its _retrieve_clean).
        def search(text):
            return list(dict.fromkeys(sum(_fusion_clean_searches(session, text, _CASE_POOL_VECTOR_INDEX,
                                                                 _FUSION_SEARCH_K), [])))
        baseline_refs = _fusion_best_rank_merge(*_fusion_clean_searches(session, case, _CASE_POOL_VECTOR_INDEX,
                                                                        _FUSION_SEARCH_K))
    else:
        def search(text):
            return _fusion_text_search(session, text, _CASE_POOL_VECTOR_INDEX, _FUSION_SEARCH_K)
        baseline_refs = search(case)

    t0 = time.time()
    try:
        bullets = _fusion_distill_bullets(case)
    except Exception as exc:
        logger.warning("fusion: distill bullets failed: %s", exc)
        bullets = []
    vlog("case_pool_fusion_distill_llm", duration_ms=(time.time() - t0) * 1000)
    joined = " ".join(b["testo"] for b in bullets)
    distill_lists = [search(b["testo"]) for b in bullets]
    if joined:
        distill_lists.append(search(joined))
    distill_refs = _fusion_merge_by_hits(*distill_lists) if distill_lists else []

    # CASE_FRAME_ENABLED: the case's explicit questions, its procedural posture
    # and its central issues, each searched as its own query -- a fourth
    # signal, so an article they find and another signal also finds survives
    # the cap like any multi-signal one.
    frame_queries = _frame_queries(analysis.get("frame") or {})
    frame_lists = [search(q) for q in frame_queries]
    frame_refs = _fusion_merge_by_hits(*frame_lists) if frame_lists else []
    # HYBRID_NARRATIVE_ENABLED (source_tiers.py): the rulings similar to the facts read first as a MAP. The
    # issues their cards decide and the analysis' own issues are searched in the primary documents as one
    # more signal; the articles they apply are hypotheses only (confirmed below), in place of the vote.
    from . import source_tiers
    hybrid = source_tiers.hybrid_enabled(analysis.get("route"))
    ruling_map, issue_refs = None, []
    if hybrid:
        ruling_map = source_tiers.ruling_map(session, driver, database, case, _CASE_POOL_VECTOR_INDEX, analysis)
        issue_queries = [q for q in dict.fromkeys(ruling_map["issues"] + _analysis_issue_titles(analysis))
                         if q and q not in frame_queries][:_HYBRID_ISSUE_QUERIES]
        issue_lists = [search(q) for q in issue_queries]
        issue_refs = _fusion_merge_by_hits(*issue_lists) if issue_lists else []
        vlog("hybrid_issue_queries", {"queries": issue_queries, "found": len(issue_refs)})
    vlog("case_pool_fusion_sources", {"clean": clean, "frame_queries": len(frame_queries),
                                      "baseline": len(baseline_refs), "distill": len(distill_refs),
                                      "frame": len(frame_refs), "issues": len(issue_refs)})

    signals = [current_refs, baseline_refs, distill_refs] + ([frame_refs] if frame_refs else []) \
        + ([issue_refs] if issue_refs else [])
    hits, best_rank = _fusion_hits_and_best_rank(*signals)
    fused = sorted(hits, key=lambda ref: (-hits[ref], best_rank[ref]))
    # VOTE_POOL_ENABLED: the articles the rulings most similar to the case's facts cite
    # (vote_pool.py), joined as VOTE_POOL_MODE says: appended after the cap, or merged into
    # the ranking before it. Never a signal of its own: a voted article is not multi-signal.
    from . import vote_pool
    voted, how = [], None
    vote_stats: Dict[tuple, Dict[str, int]] = {}
    if vote_pool.enabled() and not hybrid:
        voted, how = vote_pool.ruling_vote(session, driver, database, case, _CASE_POOL_VECTOR_INDEX,
                                           analysis, vote_stats), vote_pool.mode()
    # HYBRID_NARRATIVE_ENABLED: a map hypothesis is a candidate only if a primary channel found it (it then
    # passes the cap as found by two sources) or 3+ of the top similar rulings of the case's area apply it
    # (added after the cap); the fetch below says whether it exists in the primary corpus. Graded like any.
    map_channel: List[tuple] = []
    map_rulings: List[tuple] = []
    if hybrid and ruling_map and ruling_map["articles"]:
        channel_refs = set().union(*(set(refs) for refs in signals))
        confirmed, why = source_tiers.confirmed_hypotheses(ruling_map["articles"], channel_refs, ruling_map["stats"])
        map_channel = [ref for ref in confirmed if ref in channel_refs]
        map_rulings = [ref for ref in confirmed if ref not in channel_refs]
        vote_stats.update({ref: dict(ruling_map["stats"][ref]) for ref in confirmed if ref in ruling_map["stats"]})
        vlog("hybrid_hypotheses", why)
    # Exempt from the cap regardless of rank: found by two or more signals,
    # or resolved by the article's own rubrica (heading_matched) -- a
    # single-signal text-search hit is the only thing the cap should ever
    # bound. The cap only bounds how many of those fill the rest.
    not_keep = [ref for ref in fused if ref not in keep]
    if voted and how[0] == "rrf":
        not_keep = [ref for ref in vote_pool.weighted_rrf(not_keep, voted, how[1], how[2]) if ref not in keep]
    exempt_from_cap = {ref for ref in not_keep if hits.get(ref, 0) >= 2} | (heading_matched & set(not_keep)) \
        | (set(map_channel) & set(not_keep))
    survives_cap = [ref for ref in not_keep if ref in exempt_from_cap]
    fill = [ref for ref in not_keep if ref not in exempt_from_cap]
    capped = survives_cap + fill[:_FUSION_POOL_CAP]
    added = vote_pool.new_votes(voted, set(keep) | set(capped), how[1]) if voted and how[0] == "append" else []
    capped += added
    capped += [ref for ref in map_rulings if ref not in keep and ref not in capped]
    selected = keep + capped
    vote_tagged = vote_pool.tagged(voted, selected, added, how) if voted else set()
    if voted:
        vlog("case_pool_vote_cut", {"mode": ":".join(str(p) for p in how), "added": [f"{n} {c}" for n, c in added],
                                    "tagged": sorted(f"{n} {c}" for n, c in vote_tagged)})

    candidate_rows = _fetch_any(session, documents, selected)
    if hybrid and (map_channel or map_rulings):
        fetched = {row.get("_article") for row in candidate_rows}
        vlog("hybrid_hypotheses_fetched", {"not_in_primary_corpus": [f"{n} {c}" for n, c in map_channel + map_rulings
                                                                       if (n, c) not in fetched]})
    fused_rank = {ref: i for i, ref in enumerate(fused, 1)}
    signal_names = ["current", "baseline", "distill"] + (["frame"] if frame_refs else []) \
        + (["issues"] if issue_refs else [])
    signal_sets = [set(refs) for refs in signals]
    for row in candidate_rows:
        ref = row.get("_article")
        if ref in fused_rank:
            row["_fused_rank"] = fused_rank[ref]      # read by PLAN_KEEP_CHECKED under CHECK_GRADED
        # The grader's evidence (GRADER_EVIDENCE_ENABLED): which searches found it, how many similar rulings apply it.
        row["_signals"] = [name for name, refs in zip(signal_names, signal_sets) if ref in refs]
        if ref in vote_stats:
            row["_vote_stats"] = dict(vote_stats[ref])
        if hits.get(ref, 0) >= 2:
            row["_multi_signal"] = True
        if ref in heading_matched:
            # The _heading_match tag _resolve_institution set is on a row
            # object this function never sees (article_rows here are freshly
            # re-fetched by ref, not reused from v2's own rows) -- re-applied
            # by ref so the applicability exemption still reaches it.
            row["_heading_match"] = True
        if ref in central_matched:
            row["_central"] = True                    # resolved for a central issue (_central_questions)
        if ref in map_channel or ref in map_rulings:
            # A confirmed map hypothesis (HYBRID_NARRATIVE_ENABLED): the grader sees how many similar rulings
            # apply it; found by a channel too, it counts as found by two sources. No other protection.
            row["_map"] = True
            row["_signals"] = row["_signals"] + ["map"]
            if ref in map_channel:
                row["_multi_signal"] = True
        if ref in vote_tagged:
            # Voted by similar rulings (VOTE_POOL_ENABLED): still checked for applicability;
            # kept by the prompt's trimming like a multi-signal row; never removed by the plan check.
            row["_vote"] = True
            row["_vote_rank"] = voted.index(ref) + 1
    article_rows = sorted((r for r in candidate_rows if r.get("_article") in selected),
                          key=lambda r: selected.index(r["_article"]))
    have = {((r.get("d") or {}).get("id"), (r.get("s") or {}).get("name")) for r in article_rows}

    crossref_seed_refs = set(named) | set(institutions_resolved)
    seed_rows = [r for r in article_rows if r.get("_article") in crossref_seed_refs]
    if seed_rows:
        broadened = broaden_article_context(
            {"raw_result": seed_rows, "law_hint_doc_id": "", "user_id": "", "tenant_id": ""},
            driver, database,
        ).get("raw_result", seed_rows)
        abbr_of = {prefix: abbr for abbr, prefix in _CODE_PREFIXES.items()}
        for row in broadened:
            key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
            if key in have:
                continue
            have.add(key)
            if "_article" not in row:
                d_name = (row.get("d") or {}).get("name") or ""
                abbr = next((a for p, a in abbr_of.items() if d_name.startswith(p)), None)
                if abbr:
                    row["_article"] = (_article_num((row.get("s") or {}).get("name") or ""), abbr)
            article_rows.append(row)

    # v2's own base (_select_articles) already ran issue spotting on its
    # own candidates; reusing its "spotted" rows here (rather than calling
    # spotted_rows a second time on fusion's differently-built article_rows)
    # adds those articles at no extra LLM cost.
    spotted = v2.get("spotted") or []
    for row in spotted:
        key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
        if key not in have:
            have.add(key)
            article_rows.append(row)

    return {"first_rulings": v2["first_rulings"], "similar": v2["similar"], "keep": keep,
            "named": named, "candidates": fused, "chosen": capped, "selected": selected,
            "article_rows": article_rows, "spotted": spotted,
            "institutions_resolved": institutions_resolved}


def _select_articles_v2(session, driver, database: str, facts: str, request: str,
                        analysis: Dict[str, Any]) -> Dict[str, Any]:
    """v1's full selection (_select_articles — the user's own citations,
    ruling agreement, code search, the model's own choice among candidates,
    cross-reference broadening, issue spotting) UNION institution-resolved
    rows from analysis["questioni"][].istituti (_resolve_institution, any
    document, never trusting the model's own number alone).

    v2 used to REPLACE _select_articles's mechanism entirely with only the
    institution/parole_chiave path — on prod (17 Oct) this lost articles
    v1's other, redundant paths found that institutions alone didn't: 581,
    41 (ruling agreement / crossref, Sempronio), 628 (p3, rapina), 1495 (c2,
    vizi della cosa) — a net regression (1.36 avg vs. 1.64 flag-off) despite
    genuine gains (586, 589, 658 c.p.c.). Calling _select_articles first and
    adding to its result keeps everything v1 already finds, while still
    reaching institutions outside the six/four codes v1 is structurally
    limited to (Codice del Consumo, L. 392/1978, ...).

    Used when CASE_ANALYSIS_ENABLED. Returns the same keys as
    _select_articles, plus "institutions_resolved" (the articles only this
    path added, for diagnostics) — so run_case_analysis_pipeline and
    scripts/check_applicability.py don't need to special-case which one ran.
    """
    from .doc_lookup import _fetch_doc_names
    from .nodes.retrieval import broaden_article_context

    base = _select_articles(session, driver, database, facts, request, analysis)
    documents = _fetch_doc_names(driver, database)
    named_in_case = set(_refs_in(_case_text(facts, request)))

    have = {((r.get("d") or {}).get("id"), (r.get("s") or {}).get("name"))
            for r in base["article_rows"]}
    added: List[Dict[str, Any]] = []
    # Sequential, not concurrent: a neo4j Session isn't safe for concurrent
    # use from several threads, and these are DB-only round-trips (cheap)
    # — the one LLM call already happened, in _analyse_case_v2.
    central_questions = _central_questions(analysis)
    central_matched: List[tuple] = []
    for index, q in enumerate(analysis.get("questioni", [])):
        for istituto in q.get("istituti") or []:
            resolved = _resolve_institution(session, documents, istituto, q.get("parole_chiave"))
            if istituto.get("fonte"):          # FACT_SIGNALS_ENABLED: a fact rule's institution, within its ranges
                from . import fact_signals
                kept = fact_signals.keep_resolved(istituto, resolved)
                vlog("case_analysis_fact_institution", {"institution": istituto.get("nome"), "fonte": istituto["fonte"],
                                                         "kept": [_label_of(r) for r in kept],
                                                         "rejected": [_label_of(r) for r in resolved if r not in kept]})
                resolved = kept
            for row in resolved:
                if index in central_questions and row.get("_article"):
                    central_matched.append(row["_article"])
                key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
                if key in have:
                    continue
                have.add(key)
                row["_question"] = index
                row["_source"] = CASE_ANALYSIS_SOURCE
                added.append(row)

    article_rows = base["article_rows"] + added
    if added:
        # Broaden only from articles the user's own text names, or that v2
        # itself just resolved via an institution — not every v1 candidate.
        # Round 3, item 3 (18 Oct): broadening the full merged set re-expanded
        # articles v1 had already selected and broadened once itself (116,
        # 118 on Sempronio), pulling in generic cross-references a second
        # time (1, 2, 11 c.p.) as noise with no connection to what v2 added.
        seed_rows = [r for r in base["article_rows"] if r.get("_article") in named_in_case] + added
        broadened = broaden_article_context(
            {"raw_result": seed_rows, "law_hint_doc_id": "", "user_id": "", "tenant_id": ""},
            driver, database,
        ).get("raw_result", seed_rows)
        abbr_of = {prefix: abbr for abbr, prefix in _CODE_PREFIXES.items()}
        for row in broadened:
            key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
            if key in have:
                continue
            have.add(key)
            if "_article" not in row:
                d_name = (row.get("d") or {}).get("name") or ""
                abbr = next((a for p, a in abbr_of.items() if d_name.startswith(p)), None)
                if abbr:
                    row["_article"] = (_article_num((row.get("s") or {}).get("name") or ""), abbr)
            article_rows.append(row)

    article_rows = _prioritise_article_rows(article_rows, named_in_case)
    added_articles = [r["_article"] for r in added if r.get("_article")]
    # Which of those were resolved by the article's own rubrica
    # (_resolve_institution's _heading_match) -- _select_articles_fusion
    # re-derives this by ref, not by row object, since it re-fetches its
    # own article_rows from scratch (_fetch_any) rather than reusing these.
    heading_matched = list(dict.fromkeys(r["_article"] for r in added
                                         if r.get("_heading_match") and r.get("_article")))
    central_matched = list(dict.fromkeys(central_matched))
    for row in article_rows:
        if row.get("_article") in central_matched:
            row["_central"] = True
    return {**base, "selected": base["selected"] + added_articles, "heading_matched": heading_matched,
            "article_rows": article_rows, "institutions_resolved": added_articles, "central_matched": central_matched}


def _label_of(row: Dict[str, Any]) -> str:
    ref = row.get("_article") or ("", "")
    return f"{ref[0]} {ref[1]}".strip()


def _central_questions(analysis: Dict[str, Any]) -> set:
    """The indices of the analysis' questions about a central issue: the ones made from the frame's
    central issues (voce "questione_centrale"), and any whose institution is one of them. Their resolved
    articles are tagged _central (read by PLAN_KEEP_CHECKED under CHECK_GRADED)."""
    names = [c.get("nome") or "" for c in ((analysis or {}).get("frame") or {}).get("central_issue_candidates") or []
             if isinstance(c, dict)]
    return {i for i, q in enumerate((analysis or {}).get("questioni") or [])
            if isinstance(q, dict) and (q.get("voce") == "questione_centrale"
                                        or any(_same_institution(ist.get("nome") or "", n)
                                               for ist in q.get("istituti") or [] if isinstance(ist, dict)
                                               for n in names if n))}


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
    "procedura) scegli tra \"applicabile\" e \"incompatibile\". Un'alternativa deve corrispondere "
    "all'esito effettivo narrato nei fatti, non solo al tipo di condotta: se l'esito ha due forme "
    "alternative nei fatti e nel testo dell'articolo (morte o sopravvivenza della vittima, invalidità "
    "permanente o temporanea, danno o pericolo) e il testo presuppone quella che i fatti escludono, "
    "l'articolo non è un'alternativa, è \"incompatibile\" — anche se descrive lo stesso tipo di condotta. "
    "Esempio: se i fatti dicono che la vittima è morta, un articolo sulla lesione personale (che "
    "presuppone la sopravvivenza della vittima) non è un'alternativa all'omicidio, è incompatibile con "
    "quell'esito — scrivi il requisito in positivo (\"presuppone la sopravvivenza della vittima\"), non "
    "come negazione (\"non deve esserne derivata la morte\"): lo stesso requisito espresso con una "
    "negazione verrebbe trattato come il limite che un fatto esclude a favore dell'assistito, non "
    "come un'incompatibilità.\n"
    "- \"incompatibile\": un fatto preciso del caso contraddice un requisito scritto nel testo "
    "dell'articolo.\n"
    "Un articolo che pone un'esclusione, un limite o un'eccezione (\"non si può\", \"non è dovuta\", "
    "\"salvo che\") non è incompatibile quando i fatti escludono quel limite: è \"applicabile\", perché "
    "serve proprio a sostenere che il limite non opera.\n"
    "Usa \"incompatibile\" solo se puoi indicare quel requisito e quel fatto; in ogni dubbio usa "
    "\"applicabile\". Rispondi SOLO con un array JSON, un oggetto per articolo, nessun altro testo. Per un "
    "articolo \"applicabile\" o \"alternativa\" scrivi solo article e verdict; solo per un articolo "
    "\"incompatibile\" aggiungi requisito, fatto e reason (una frase):\n"
    '[{"article": "<come indicato>", "verdict": "applicabile"}, '
    '{"article": "<come indicato>", "verdict": "alternativa"}, '
    '{"article": "<come indicato>", "verdict": "incompatibile", "requisito": "<il requisito del testo>", '
    '"fatto": "<il fatto del caso>", "reason": "<una frase>"}]'
)
# 23 Oct: the check was sent every candidate in one call (87-89 on prod) and asked for 220
# answer tokens each, 19,580 against a 20,000-token context: it failed on 72 of about 260
# strategy answers since 20 Oct (40 "maximum context length", 32 "max_completion_tokens"),
# each time keeping every article. Now: at most _APPLICABILITY_MAX_CHECKED candidates, in
# the pipeline's order (the ones that can reach the draft), _APPLICABILITY_BATCH per call,
# up to _APPLICABILITY_WORKERS calls at once, and the long verdict only for the articles
# dropped -- the check took a median 26 s on prod, most of it writing reasons for articles kept.
_APPLICABILITY_MAX_CHECKED = 36
_APPLICABILITY_BATCH = 12
_APPLICABILITY_WORKERS = 3


def _applicability_key(label: str) -> str:
    """"Art. 116 c.p." / "116 c.p." -> "116 c.p.": the model doesn't always
    echo the label exactly."""
    return " ".join(re.sub(r"^\s*art(?:icolo)?\.?\s*", "", str(label), flags=re.IGNORECASE).lower().split())


def _applicability_candidates(facts: str, request: str, article_rows: List[Dict[str, Any]]) -> tuple:
    """({article: rows} to check, {label: text}). Exempt, never sent to the
    model: rows without an article tag, cross-reference rows (grounded in a
    chosen article's own text), the articles the user's own text names, and
    (round 9, 19 Oct) rows _resolve_institution found by the article's own
    rubrica (_heading_match) or that the fusion pool found via two or more
    independent signals (_multi_signal) -- the check may prune a single-
    signal text-search candidate, never a core qualification reached this
    way. Round 9's own prod diagnostic (sempronio_no_hint) found the model's
    "incompatibile" judgement wrong about art. 584 c.p. itself (a genuine
    qualification the facts raise, not a speculative candidate) -- the
    prompt already asks for a concrete requirement+fact before dropping
    anything, but that alone didn't stop it; this is the code-level backstop
    for articles strong enough that the check should never get the final say.

    19 Oct, later: _multi_signal is no longer exempt. On Sempronio (prod,
    check_applicability.py --fusion) the fusion pool's top three were 43, 593
    and 116, all multi-signal and so never checked; 116 (several agents) and
    593 (failing to help; Sempronio helped) are wrong for the facts, and the
    check drops them when it sees them. Searches agreeing is not evidence
    that an article applies; multi-signal rows still survive the cap. And an
    exemption now covers the whole article: 581, cross-referenced from 584's
    text, was still sent to the check through a second, untagged row."""
    from .nodes.cypher import CROSSREF_SOURCE

    named = set(_refs_in(_case_text(facts, request)))
    exempt = {row.get("_article") for row in article_rows
              if row.get("_article") and (row.get("_source") == CROSSREF_SOURCE or row.get("_article") in named
                                          or row.get("_heading_match"))}
    to_check: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows:
        article = row.get("_article")
        if not article or article in exempt:
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


def _applicability_batch(case: str, texts: Dict[str, str]) -> tuple:
    """(verdict list or None, error text) for one batch of articles."""
    human = (
        f"Fatti del caso:\n{case}\n\n"
        "Articoli candidati (testo reale dalla banca dati):\n"
        + "\n\n".join(f"[{label}]\n{text}" for label, text in texts.items())
    )
    # Round 8 (19 Oct): a fixed 1200-token cap truncated the JSON reply past 6-7
    # articles when every verdict carried requisito/fatto/reason. Only dropped
    # articles carry them now; the room still allows several drops per batch.
    max_tokens = 400 + 120 * len(texts)
    try:
        verdicts = _chat_json([SystemMessage(content=_APPLICABILITY_SYSTEM), HumanMessage(content=human)],
                              max_tokens=max_tokens, temperature=0)
    except Exception as exc:
        logger.warning("applicability check failed for a batch of %d articles, keeping them: %s", len(texts), exc)
        return None, str(exc)[:160]
    return (verdicts, "") if isinstance(verdicts, list) else (None, "reply is not a JSON list")


def _applicability_verdicts(facts: str, request: str, texts: Dict[str, str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """{label key: verdict} from the model, _APPLICABILITY_BATCH articles per call and up to
    _APPLICABILITY_WORKERS calls at once; a failed batch only leaves its own articles without
    a verdict (kept). None if every batch fails. CHECK_GRADED_ENABLED has its own: _graded_verdicts."""
    if not texts:
        return {}
    case = _case_text(facts, request)[:2000]
    labels = list(texts)
    batches = [{label: texts[label] for label in labels[i:i + _APPLICABILITY_BATCH]}
               for i in range(0, len(labels), _APPLICABILITY_BATCH)]
    t0 = time.time()
    if len(batches) == 1:
        results = [_applicability_batch(case, batches[0])]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(_APPLICABILITY_WORKERS, len(batches))) as pool:
            results = list(pool.map(lambda batch: _applicability_batch(case, batch), batches))
    errors = [error for verdicts, error in results if verdicts is None]
    log = {"checked": len(texts), "batches": len(batches), "failed_batches": len(errors)}
    if errors:
        log["error"] = errors[0]
    vlog("case_analysis_applicability", log, duration_ms=(time.time() - t0) * 1000)
    if len(errors) == len(results):
        return None
    return {_applicability_key(v["article"]): v for verdicts, _ in results for v in verdicts or []
            if isinstance(v, dict) and isinstance(v.get("article"), str)}


# A limit or an exclusion: facts that exclude it make the article an argument
# for the client, not incompatible. On production (15 Oct) the model dropped
# 1455 c.c. ("Il contratto non si può risolvere se l'inadempimento ... ha scarsa
# importanza") for an eviction case and 1491 c.c. ("Non è dovuta la garanzia se
# ...") for a hidden defect, rewording the limit as a plain requirement, so the
# article's own text decides: its first sentence is a negated rule with a
# condition ("non" early, then "se"). Requirements worded as limits count too.
_LIMIT_WORDING = re.compile(r"\b(?:non|né|salvo|tranne|eccett\w*)\b", re.IGNORECASE)
_LIMIT_RULE = re.compile(r"^[^.;]{0,40}\bnon\b[^.;]{0,80}\bse\b", re.IGNORECASE)


def _is_limit_rule(article_text: str) -> bool:
    return bool(_LIMIT_RULE.search(" ".join((article_text or "").split())[:300]))


def _drops_article(verdict: Optional[Dict[str, Any]], article_text: str = "") -> bool:
    """Only an "incompatibile" verdict that names both the requirement in the
    article's text and the contradicting fact drops an article, and never a
    limit or an exclusion (by the article's text or the requirement's wording)."""
    requisito = str((verdict or {}).get("requisito") or "").strip()
    return bool(verdict) and str(verdict.get("verdict", "")).strip().lower() == "incompatibile" \
        and bool(requisito) and bool(str(verdict.get("fatto") or "").strip()) \
        and not _LIMIT_WORDING.search(requisito) and not _is_limit_rule(article_text)


def _filter_applicable_articles(facts: str, request: str, article_rows: List[Dict[str, Any]],
                                analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The article rows without those the model shows to be incompatible with
    the facts, in their original order. Fails open: on any error, unparseable
    reply or missing verdict, the rows are kept -- except, with CHECK_PRIORITY_ENABLED,
    the articles left without a verdict while the check ran, and, with
    CHECKLIST_GATED_ENABLED, the method lists' candidates without one.
    CHECK_GRADED_ENABLED: a relevance grade instead of the verdict (_GRADED_SYSTEM), for up to
    _GRADED_MAX_CHECKED articles; grade 0 is removed, 1 stays as "marginale", 2 and 3 are "applicabile";
    each kept row carries _grade. An article still without a grade after the retry (_graded_verdicts)
    stays as "non valutato": never removed, never counted as passing the check."""
    from . import method_candidates

    priority = feature_flags.enabled(CHECK_PRIORITY_FLAG)
    gated = feature_flags.enabled(method_candidates.CHECKLIST_GATED_FLAG)
    graded = feature_flags.enabled(CHECK_GRADED_FLAG)
    to_check, texts = _applicability_candidates(facts, request, article_rows)
    if not texts and not (priority or gated):
        return article_rows
    if priority:
        texts = _check_order(texts, to_check)
    # Only the first _APPLICABILITY_MAX_CHECKED, in the pipeline's order: the later ones
    # rarely reach the draft (its budget holds some 25 articles at full text) and stay unchecked.
    # A graded call costs about a third of a verdict per article: twice the slots.
    limit = _GRADED_MAX_CHECKED if graded else _APPLICABILITY_MAX_CHECKED
    unchecked = len(texts) - limit
    unchecked_labels = list(texts)[limit:]
    if unchecked > 0:
        texts = dict(list(texts.items())[:limit])
    ungraded: List[str] = []
    if not texts:
        verdicts = {}
    elif graded and feature_flags.enabled(GRADER_EVIDENCE_FLAG):
        issues = _graded_issues(analysis)
        verdicts, ungraded = _graded_verdicts(
            _graded_case(facts, request, analysis, issues),
            _graded_texts(texts, to_check, _evidence_context(facts, request, analysis)),
            _graded_system(evidence=True, issues=bool(issues), examples=feature_flags.enabled(GRADER_EXAMPLES_FLAG)),
            issues)
    elif graded:
        verdicts, ungraded = _graded_verdicts(_graded_case(facts, request, analysis), _graded_texts(texts, to_check))
    else:
        verdicts = _applicability_verdicts(facts, request, texts)
    if verdicts is None:
        # Every batch failed: kept as before; the method lists' candidates are not (CHECKLIST_GATED).
        return _without_unjudged(article_rows, to_check, {}, set(), method_only=True) if gated else article_rows
    dropped_articles, dropped, grades = set(), [], {}
    for article in to_check:
        label = f"{article[0]} {article[1]}"
        verdict = verdicts.get(_applicability_key(label))
        if graded:
            grade = _grade(verdict)
            if grade == 0:
                dropped_articles.add(article)
                dropped.append({"article": label, "grade": 0, "requisito": "", "fatto": "",
                                "reason": str(verdict.get("reason") or "")[:200]})
            elif grade is not None:
                grades[article] = (grade, " ".join(str(verdict.get("issue") or "").split())[:120])
        elif _drops_article(verdict, texts.get(label, "")):
            dropped_articles.add(article)
            dropped.append({"article": label, "requisito": str(verdict.get("requisito"))[:200],
                            "fatto": str(verdict.get("fatto"))[:200], "reason": str(verdict.get("reason", ""))[:200]})
    if graded:
        shown = {label: f"grade {_grade(verdicts.get(_applicability_key(label)))}"
                 if _grade(verdicts.get(_applicability_key(label))) is not None
                 else "ungraded (kept)" if label in ungraded else "missing" for label in texts}
    else:
        shown = {label: (verdicts.get(_applicability_key(label)) or {}).get("verdict", "missing") for label in texts}
    log = {
        "checked": sorted(texts),
        "verdicts": shown,
        "dropped": dropped,
        "unchecked": max(unchecked, 0),
        "unchecked_labels": unchecked_labels,
        "max_checked": limit,
    }
    # The verdict travels with the rows checked and kept: a trimmed draft prompt gives its free
    # room to those judged applicable first (_refill_order).
    judged = {}
    if graded:
        judged = {article: "applicabile" if grade >= 2 else "marginale" for article, (grade, _) in grades.items()}
        # No grade even after the retry: kept for the writer, but neither passed nor forced into the plan.
        judged.update({article: "non valutato" for article in to_check if f"{article[0]} {article[1]}" in ungraded})
        log["ungraded"] = ungraded
        log["grades"] = {f"{a[0]} {a[1]}": g for a, (g, _) in grades.items()}
        log["issues"] = {f"{a[0]} {a[1]}": issue for a, (_, issue) in grades.items() if issue}
    else:
        for article in to_check:
            verdict = verdicts.get(_applicability_key(f"{article[0]} {article[1]}"))
            if verdict and article not in dropped_articles:
                judged[article] = str(verdict.get("verdict", "")).strip().lower()

    def tagged(row):
        article = row.get("_article")
        if article not in judged:
            return row
        extra = {"_grade": grades[article][0], "_grade_issue": grades[article][1]} if article in grades else {}
        return {**row, "_applicability": judged[article], **extra}
    rows = [tagged(r) for r in article_rows if r.get("_article") not in dropped_articles]
    if priority or gated:
        # CHECK_PRIORITY_ENABLED: an article left unchecked may not be cited; CHECKLIST_GATED_ENABLED: nor
        # may a method lists' candidate.
        kept = _without_unjudged(rows, to_check, judged, dropped_articles, method_only=not priority)
        log["removed_unchecked"] = sorted({f"{r['_article'][0]} {r['_article'][1]}" for r in rows if r not in kept})
        rows = kept
    vlog("case_analysis_applicability", log)
    return rows


# CHECK_PRIORITY_ENABLED (diagnose_long, 25 Oct: half of the pool past the 36 checked slots, and
# kept unchecked): the slots go first to the voted articles, then to those several signals found,
# then the rest in the pipeline's order, the method lists last (_check_rank); an article left
# unchecked is removed.
CHECK_PRIORITY_FLAG = "CHECK_PRIORITY_ENABLED"

# CHECK_GRADED_ENABLED (diagnose_long, 26 Oct, F1-F5 on: 75% of the 685 articles cited without being
# expected had been judged "applicabile" -- the yes/no check let almost everything through). A grade of
# relevance instead, from the rubrica and the first _GRADED_TEXT_CHARS of the text, with the case's
# questions and issues in view; grade 0 leaves the pool before the prompt, PLAN_KEEP_CHECKED keeps 2-3.
CHECK_GRADED_FLAG = "CHECK_GRADED_ENABLED"
_GRADED_TEXT_CHARS = 400
# Reply max_tokens: base, per article. Run 1206 (60 + 45 per article): a compact entry in Italian runs
# 45-60 tokens, so a batch of 12 was cut, and a cut array is no list: its 12 articles got no grade.
_GRADED_TOKENS = (120, 60)
_GRADED_MAX_CHECKED = 72
_GRADED_RETRY_BATCH = 6         # the ids still without a grade go once more, this many per call
_GRADED_SYSTEM = (
    "Sei un giurista. Ricevi i fatti di un caso, i suoi quesiti e le sue questioni, e alcuni articoli di legge "
    "candidati, ciascuno con un identificativo tra parentesi quadre ([A1], [A2], ...), la sua rubrica e "
    "l'inizio del testo. Per ogni articolo dai un voto di pertinenza al caso:\n"
    "- 3: disciplina la questione centrale, quella da cui dipende l'esito;\n"
    "- 2: disciplina un'altra questione del caso, o serve direttamente alla difesa o alla pretesa della parte "
    "assistita sui fatti dati;\n"
    "- 1: tocca il caso solo di lato (una regola generale che qui non decide nulla, un aspetto che i fatti non "
    "pongono);\n"
    "- 0: non riguarda questi fatti, o è di un'altra materia o di un'altra fase (un articolo civile in un caso "
    "solo penale, il termine di un giudizio che non c'è).\n"
    "Giudica da ciò che l'articolo dispone, non dal numero e non da una parola in comune con i fatti. Nel dubbio "
    "tra due voti scegli il più basso; solo per un articolo segnato \"Trovato da: più fonti\" (più ricerche "
    "indipendenti, o le sentenze su fatti simili, lo hanno trovato), nel dubbio tra 1 e 2 scegli 2.\n"
    "Rispondi SOLO con un array JSON compatto, esattamente un oggetto per ogni identificativo ricevuto, "
    "nell'ordine dato, nessuno escluso (anche quelli con voto 0): "
    '[{"a": "A1", "g": 0, "q": "<la questione che serve, in poche parole>", '
    '"r": "<perché, in al massimo 15 parole>"}]'
)


def _graded_batch(case: str, texts: Dict[str, str], retry: bool = False, system: str = _GRADED_SYSTEM,
                  issues: Optional[List[str]] = None) -> tuple:
    """({label: verdict} for the articles one graded call returned a grade for, problem). Each article
    is sent under a short id (A1, A2, ...) to echo back, and the reply is read object by object
    (_graded_reply), so a reply cut short still gives the grades it holds. problem is "" when every
    article got a grade, else what went wrong (error, how many missing, whether the reply was cut).
    retry only names the call for scripts/diagnose_long.py's dumps."""
    ids = {f"A{i}": label for i, label in enumerate(texts, 1)}
    human = (
        f"Fatti del caso:\n{case}\n\n"
        f"Articoli candidati (testo reale dalla banca dati), {len(ids)}: un oggetto per ciascuno degli "
        f"identificativi {', '.join(ids)}.\n\n"
        + "\n\n".join(f"[{key}] {label}\n{texts[label]}" for key, label in ids.items())
    )
    try:
        raw = _chat([SystemMessage(content=system), HumanMessage(content=human)],
                    max_tokens=_GRADED_TOKENS[0] + _GRADED_TOKENS[1] * len(ids), temperature=0)
    except Exception as exc:
        logger.warning("graded check failed for a batch of %d articles: %s", len(ids), exc)
        return {}, f"error: {str(exc)[:160]}"
    read = _graded_reply(raw, ids, issues)
    if len(read) == len(ids):
        return read, ""
    cut = not re.sub(r"`+\s*$", "", (raw or "").strip()).rstrip().endswith("]")
    return read, f"{len(ids) - len(read)} of {len(ids)} without a grade" + (" (reply cut)" if cut else "")


def _graded_reply(raw: str, ids: Dict[str, str], issues: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """{label: verdict} from a graded reply: each object whose id ("A3", or the label itself) is one
    sent and whose grade is 0-3; the first one per article counts. With the case's numbered issues
    (GRADER_EVIDENCE_ENABLED), an issue given as "Q2" is read as the second one's title."""
    by_key = {_applicability_key(label): label for label in ids.values()}
    read: Dict[str, Dict[str, Any]] = {}
    for item in _json_objects(raw):
        ref = str(item.get("a") or item.get("article") or item.get("id") or "").strip()
        short = re.match(r"\[?\s*(A\d+)\b", ref, re.IGNORECASE)
        label = ids.get(short.group(1).upper()) if short else by_key.get(_applicability_key(ref.strip("[] ")))
        grade = _grade({"grade": item.get("g", item.get("grade"))})
        if label and grade is not None and label not in read:
            issue = item.get("q") or item.get("issue") or ""
            number = re.match(r"\s*Q\s*(\d+)\b", str(issue or ""), re.IGNORECASE) if issues else None
            if number and 1 <= int(number.group(1)) <= len(issues):
                issue = issues[int(number.group(1)) - 1]
            read[label] = {"article": label, "grade": grade, "issue": issue,
                           "reason": item.get("r") or item.get("reason")}
    return read


def _json_objects(raw: str) -> List[Dict[str, Any]]:
    """Every JSON object in a reply, in order, also when the reply is cut in the middle of the array
    (the objects before the cut) or wraps the array in an object ({"articoli": [...]})."""
    import json
    text, decoder = raw or "", json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    i = text.find("{")
    while i >= 0:
        try:
            item, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(item, dict):
            lists = [v for v in item.values() if isinstance(v, list)]
            if lists and not any(k in item for k in ("a", "article", "id")):
                found += [x for v in lists for x in v if isinstance(x, dict)]
            else:
                found.append(item)
        i = text.find("{", end)
    return found


def _graded_verdicts(case: str, texts: Dict[str, str], system: str = _GRADED_SYSTEM,
                     issues: Optional[List[str]] = None) -> tuple:
    """({label key: verdict}, [labels left without a grade]) for the graded check: _APPLICABILITY_BATCH
    articles per call, up to _APPLICABILITY_WORKERS at once; then the articles still without a grade --
    skipped by the model, cut off, or in a failed call -- once more, _GRADED_RETRY_BATCH per call. Never
    fails as a whole: an article that still has no grade is returned among the ungraded."""
    from concurrent.futures import ThreadPoolExecutor

    def run(labels: List[str], size: int, retry: bool) -> List[tuple]:
        groups = [{label: texts[label] for label in labels[i:i + size]} for i in range(0, len(labels), size)]
        if len(groups) == 1:
            return [_graded_batch(case, groups[0], retry, system, issues)]
        with ThreadPoolExecutor(max_workers=min(_APPLICABILITY_WORKERS, len(groups))) as pool:
            return list(pool.map(lambda group: _graded_batch(case, group, retry, system, issues), groups))

    t0 = time.time()
    labels = list(texts)
    got: Dict[str, Dict[str, Any]] = {}
    first = run(labels, _APPLICABILITY_BATCH, False)
    for read, _ in first:
        got.update(read)
    missing = [label for label in labels if label not in got]
    second = run(missing, _GRADED_RETRY_BATCH, True) if missing else []
    for read, _ in second:
        got.update({label: v for label, v in read.items() if label not in got})
    ungraded = [label for label in labels if label not in got]
    problems = [problem for _, problem in first if problem]
    log = {"checked": len(labels), "batches": len(first), "failed_batches": sum(1 for read, p in first if p and not read),
           "incomplete_batches": sum(1 for read, p in first if p and read),
           "cut_replies": sum(1 for _, p in first + second if "(reply cut)" in p),
           "retried": len(missing), "retry_batches": len(second), "recovered": len(missing) - len(ungraded),
           "ungraded": ungraded}
    if problems:
        log["error"] = problems[0]
    vlog("case_analysis_applicability", log, duration_ms=(time.time() - t0) * 1000)
    return {_applicability_key(label): v for label, v in got.items()}, ungraded


def _grade(verdict: Optional[Dict[str, Any]]) -> Optional[int]:
    """The grade 0-3 of a graded verdict, or None when it has none."""
    match = re.match(r"\s*([0-3])\b", str((verdict or {}).get("grade", "")))
    return int(match.group(1)) if match else None


def _graded_texts(texts: Dict[str, str], to_check: Dict[tuple, List[Dict[str, Any]]],
                  evidence: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Each article as the graded check reads it: its rubrica and the first _GRADED_TEXT_CHARS of its text;
    one the vote or two or more channels found is marked "Trovato da: più fonti" (run 1302: 15 central
    articles graded 1, h7 319-321 c.p., c1 1218 c.c., n5 252 TUEL -- for these, a doubt between 1 and 2
    goes to 2)."""
    rows_of = {f"{a[0]} {a[1]}": rows for a, rows in to_check.items()}

    def entry(label: str, text: str) -> str:
        rows = rows_of.get(label, [])
        if evidence is not None:
            return (f"Rubrica: {_rows_heading(rows) or '-'}\n" + "".join(f"{line}\n" for line in _evidence_lines(rows, evidence))
                    + f"Testo: {text[:_GRADED_TEXT_CHARS]}")
        several = any(r.get("_vote") or r.get("_multi_signal") for r in rows)
        return (f"Rubrica: {_rows_heading(rows) or '-'}\n" + ("Trovato da: più fonti\n" if several else "")
                + f"Testo: {text[:_GRADED_TEXT_CHARS]}")
    return {label: entry(label, text) for label, text in texts.items()}


# GRADER_EVIDENCE_ENABLED (diagnose_long run 1415: 20 of 122 expected articles lost at NOT_IN_PLAN, 15 of
# them graded 1 although central): the grader sees, per article, what found it -- the searches, the similar
# rulings applying it, the central issue it was resolved for, whether its code is the case's area's -- and
# grades it per issue of the case (frame questions, central issues, analysis issues, numbered Q1, Q2 ...),
# by how directly it governs that issue, with worked examples. PLAN_KEEP_CHECKED then keeps up to 18.
GRADER_EVIDENCE_FLAG = "GRADER_EVIDENCE_ENABLED"
# The worked examples apart (grader_bench on run 1415: the evidence and the per-issue grade, v2, without them).
GRADER_EXAMPLES_FLAG = "GRADER_EXAMPLES_ENABLED"
_GRADED_MAX_ISSUES = 10
_SIGNAL_WORDS = {"current": "ricerca sull'analisi del caso", "baseline": "ricerca sul testo del caso",
                 "distill": "ricerca sui fatti essenziali", "frame": "ricerca sui quesiti e sulle questioni",
                 "issues": "ricerca sulle questioni del caso e delle sentenze simili",
                 "map": "sentenze su fatti simili"}


def _evidence_context(facts: str, request: str, analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What _evidence_lines needs of the case: the main codes of its area (None when unknown)."""
    from . import vote_pool
    return {"allowed": vote_pool.allowed_codes(vote_pool.filter_areas(analysis),
                                               vote_pool.named_codes(_case_text(facts, request)))}


def _evidence_lines(rows: List[Dict[str, Any]], evidence: Dict[str, Any]) -> List[str]:
    """The grader's evidence on one article, from its rows' tags: "Trovato da" (the searches, the similar
    rulings, the area's list, the central issue), "Sentenze simili" (how many apply it), "Materia"."""
    from . import vote_pool
    found = list(dict.fromkeys(_SIGNAL_WORDS[name] for r in rows for name in r.get("_signals") or []
                               if name in _SIGNAL_WORDS))
    if any(r.get("_vote") for r in rows):
        found.append("sentenze su fatti simili")
    if any(r.get("_central") for r in rows):
        found.append("istituto di una questione centrale")
    if rows and all(r.get("_source") == METHOD_CANDIDATES_SOURCE for r in rows):
        found.append("elenco degli articoli della materia")
    lines = [f"Trovato da: {'; '.join(found)}" + (" (più fonti)" if len(found) >= 2 else "")] if found else []
    stats = next((r["_vote_stats"] for r in rows if r.get("_vote_stats")), None)
    if stats and stats.get("rulings"):
        cards = (f"; {stats['carded']} delle {stats['cards']} con scheda lo applicano" if stats.get("cards") else "")
        lines.append(f"Sentenze simili: applicato da {stats['rulings']} delle {stats.get('similar') or '?'} più simili"
                     + cards)
    code = next((r["_article"][1] for r in rows if r.get("_article")), "")
    allowed = evidence.get("allowed")
    if allowed is not None and code in vote_pool.MAIN_CODES:
        lines.append("Materia: del caso" if code in allowed else "Materia: un'altra rispetto alle aree del caso")
    return lines


def _graded_issues(analysis: Optional[Dict[str, Any]]) -> List[str]:
    """The case's issues for the grader, numbered in this order: the frame's questions, its central
    issues, the analysis' issues (deduplicated, at most _GRADED_MAX_ISSUES)."""
    analysis = analysis or {}
    frame = analysis.get("frame") or {}
    items = [str(t).strip() for t in frame.get("tasks") or [] if str(t or "").strip()]
    items += [str(c.get("nome")).strip() for c in frame.get("central_issue_candidates") or []
              if isinstance(c, dict) and c.get("nome")]
    items += [re.sub(r"^Questione centrale:\s*", "", str(q.get("titolo"))).strip()
              for q in analysis.get("questioni") or [] if isinstance(q, dict) and q.get("titolo")]
    seen, out = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out[:_GRADED_MAX_ISSUES]


# Invented cases; no article in them is expected in h1-h11 or n1-n9 (checked against
# tests/eval_narrative_cases.json), so they set the scale without teaching a held-out answer.
_GRADED_EXAMPLES = (
    "ESEMPI (casi inventati, solo per il metro di giudizio):\n"
    "1. Penale. Durante una lite Tizio rompe con un bastone il vetro dell'auto del vicino, poi lo ripaga. Q1 "
    "qualificazione del fatto; Q2 attenuanti. Art. 635 c.p. -> 3 (Q1, il reato); art. 62 c.p. -> 2 (Q2, il danno "
    "risarcito prima del giudizio); art. 133 c.p. -> 1 (la misura della pena non decide nessuna questione); "
    "art. 2043 c.c. -> 0 (civile, il caso è solo penale).\n"
    "2. Penale, fase. Il cliente ha ricevuto l'avviso di conclusione delle indagini. Q1 che cosa può fare ora. "
    "Art. 415-bis c.p.p. -> 3 (Q1); art. 408 c.p.p. -> 2 (Q1, l'archiviazione da sollecitare); art. 585 c.p.p. "
    "-> 0 (i termini per impugnare una sentenza che non c'è).\n"
    "3. Penale, parte generale. Tizio, aggredito con un coltello, colpisce l'aggressore. Q1 se il fatto è "
    "giustificato. Art. 52 c.p. -> 3 (Q1: una norma generale che qui decide la questione); art. 55 c.p. -> 2 (Q1, "
    "l'eccesso colposo, se i fatti lo pongono); art. 62-bis c.p. -> 1 (le attenuanti generiche non decidono se il "
    "fatto è giustificato).\n"
    "4. Civile, contratti. Il compratore riceve merce difettosa e vuole sciogliere il contratto. Q1 risoluzione "
    "per inadempimento; Q2 garanzia per vizi. Art. 1453 c.c. -> 3 (Q1); art. 1455 c.c. -> 2 (Q1, l'importanza "
    "dell'inadempimento); art. 1495 c.c. -> 2 (Q2, i termini della denuncia); art. 1175 c.c. -> 1 (principio "
    "generale che qui non decide nulla); art. 2051 c.c. -> 0 (nessuna cosa in custodia).\n"
    "5. Civile, illecito. Un vaso cade da un balcone e danneggia un'auto in sosta. Q1 responsabilità del "
    "proprietario del vaso; Q2 il danno. Art. 2051 c.c. -> 3 (Q1, la custodia); art. 2056 c.c. -> 2 (Q2, la "
    "misura del risarcimento); art. 1490 c.c. -> 0 (garanzia nella vendita: nessuna vendita).\n"
    "6. Amministrativo. Il Comune non risponde da mesi a un'istanza di accesso agli atti. Q1 come ottenere i "
    "documenti. Art. 25 l. 241/1990 -> 3 (Q1, il rimedio contro il diniego o il silenzio sull'accesso); art. 116 "
    "c.p.a. -> 3 (Q1, il rito dell'accesso); art. 22 l. 241/1990 -> 2 (Q1, chi ha diritto di accesso); art. 41 "
    "c.p.a. -> 1 (la notifica di ogni ricorso: non decide nulla qui); art. 1453 c.c. -> 0 (nessun contratto con il "
    "Comune).\n"
    "7. Indizi. Un articolo dalla rubrica generica, applicato da 5 sentenze simili e della materia del caso, che "
    "disciplina un elemento di Q1 -> 2, non 1; lo stesso articolo trovato da una sola ricerca, senza un fatto che "
    "lo ponga -> 1.\n"
)


def _graded_system(evidence: bool = False, issues: bool = False, examples: bool = False) -> str:
    """The graded check's instructions: _GRADED_SYSTEM, or with GRADER_EVIDENCE_ENABLED's parts -- the
    evidence and its rule, the grade per numbered issue, the worked examples (scripts/grader_bench.py
    compares them one by one)."""
    if not (evidence or issues or examples):
        return _GRADED_SYSTEM
    head = ("Sei un giurista. Ricevi i fatti di un caso, " + ("le sue questioni numerate (Q1, Q2, ...), " if issues else
                                                              "i suoi quesiti e le sue questioni, ")
            + "e alcuni articoli di legge candidati, ciascuno con un identificativo tra parentesi quadre ([A1], "
            "[A2], ...), la sua rubrica, " + ("gli indizi raccolti dal sistema " if evidence else "")
            + "e l'inizio del testo.\n")
    if issues:
        scale = ("Per ogni articolo indica in \"q\" la questione che disciplina più direttamente e dai il voto "
                 "secondo quanto direttamente la disciplina:\n"
                 "- 3: la decide: è la norma da cui dipende la risposta a quella questione;\n"
                 "- 2: ne disciplina un elemento, o serve direttamente all'argomento della parte assistita su di "
                 "essa;\n"
                 "- 1: la tocca solo di lato (una regola generale che lì non decide nulla, un aspetto che i fatti "
                 "non pongono);\n"
                 "- 0: non disciplina nessuna questione del caso, o è di un'altra materia o di un'altra fase (un "
                 "articolo civile in un caso solo penale, il termine di un giudizio che non c'è); \"q\": \"\".\n")
    else:
        scale = _GRADED_SYSTEM[_GRADED_SYSTEM.index("Per ogni articolo"):_GRADED_SYSTEM.index("Giudica da")]
    judge = "Giudica da ciò che l'articolo dispone, non dal numero e non da una parola in comune con i fatti. "
    if evidence:
        doubt = ("Gli indizi (Trovato da, Sentenze simili, Materia) vanno pesati insieme al testo: un articolo "
                 "applicato da 3 o più sentenze simili e della materia del caso vale almeno 2, salvo che sia "
                 "chiaramente di un'altra materia o di un'altra fase. Per un articolo trovato da più fonti, nel "
                 "dubbio tra 1 e 2 scegli 2; per uno trovato da una sola ricerca, nel dubbio tra due voti scegli il "
                 "più basso.\n")
    else:
        doubt = ("Nel dubbio tra due voti scegli il più basso; solo per un articolo segnato \"Trovato da: più "
                 "fonti\", nel dubbio tra 1 e 2 scegli 2.\n")
    output = ("Rispondi SOLO con un array JSON compatto, esattamente un oggetto per ogni identificativo ricevuto, "
              "nell'ordine dato, nessuno escluso (anche quelli con voto 0): "
              + ('[{"a": "A1", "g": 0, "q": "Q2", "r": "<perché, in al massimo 15 parole>"}]' if issues else
                 '[{"a": "A1", "g": 0, "q": "<la questione che serve, in poche parole>", '
                 '"r": "<perché, in al massimo 15 parole>"}]'))
    return head + scale + judge + doubt + (_GRADED_EXAMPLES if examples else "") + output


def _graded_case(facts: str, request: str, analysis: Optional[Dict[str, Any]],
                 issues: Optional[List[str]] = None) -> str:
    """The case for the graded check: its text, then its questions, central issues and issues -- or, given
    the numbered issues (GRADER_EVIDENCE_ENABLED, _graded_issues), those."""
    if issues:
        return (_case_text(facts, request)[:2000] + "\n\nQuestioni del caso:\n"
                + "\n".join(f"Q{i}. {issue}" for i, issue in enumerate(issues, 1)))
    analysis = analysis or {}
    frame = analysis.get("frame") or {}
    parts = [_case_text(facts, request)[:2000]]
    tasks = [t for t in frame.get("tasks") or [] if t]
    central = [c.get("nome") for c in frame.get("central_issue_candidates") or [] if isinstance(c, dict) and c.get("nome")]
    issues = list(dict.fromkeys(str(q.get("titolo")).strip() for q in analysis.get("questioni") or []
                                if isinstance(q, dict) and q.get("titolo")))[:8]
    if tasks:
        parts.append("Quesiti:\n" + "\n".join(f"- {t}" for t in tasks))
    if central:
        parts.append("Questione centrale: " + "; ".join(central))
    if issues:
        parts.append("Questioni del caso:\n" + "\n".join(f"- {q}" for q in issues))
    return "\n\n".join(parts)


def _check_rank(rows: List[Dict[str, Any]]) -> int:
    """An article's place in the check's order, from all its rows (CHECK_PRIORITY_ENABLED): voted,
    then several signals, then the rest, the method lists last. The articles the user named, the
    cross-references and the rows found by their rubrica (_heading_match) are exempt from the check
    (_applicability_candidates), so they never take a slot."""
    if any(r.get("_vote") for r in rows):
        return 0
    if any(r.get("_multi_signal") for r in rows):
        return 1
    if all(r.get("_source") == METHOD_CANDIDATES_SOURCE for r in rows):
        return 3
    return 2


def _check_order(texts: Dict[str, str], to_check: Dict[tuple, List[Dict[str, Any]]]) -> Dict[str, str]:
    """texts reordered by _check_rank, the pipeline's order kept within each rank."""
    rank = {f"{a[0]} {a[1]}": _check_rank(rows) for a, rows in to_check.items()}
    order = sorted(enumerate(texts), key=lambda pair: (rank.get(pair[1], 2), pair[0]))
    return {label: texts[label] for _, label in order}


def _without_unjudged(rows: List[Dict[str, Any]], to_check: Dict[tuple, List[Dict[str, Any]]], judged: Dict[tuple, str],
                      dropped: set, method_only: bool) -> List[Dict[str, Any]]:
    """rows without the articles sent to the check that got no verdict (only the method lists'
    candidates among them when method_only). Exempt articles are never in to_check, so never removed."""
    def unjudged(row):
        article = row.get("_article")
        if article not in to_check or article in judged or article in dropped:
            return False
        return not method_only or all(r.get("_source") == METHOD_CANDIDATES_SOURCE for r in to_check[article])
    return [r for r in rows if not unjudged(r)]


# --- Draft quality (r14, 21 Oct: held-out run after the parser fix) ------------
#
# Three checks after the draft, each behind its own flag:
#   DRAFT_CITATIONS_ONLY_ENABLED  the sources returned are only the articles and
#       rulings the draft refers to (h1 listed the whole retrieval pool: TU
#       istruzione, codice della strada, 1117 c.c., 2043-2059...).
#   CLIENT_STANCE_CHECK_ENABLED   one call checks section 3 for arguments that
#       favour the other side (h1, for the pupil's parents, argued the teacher's
#       diligence); if any, section 3 is rewritten once with that list.
#   QUALIFICATION_CHECK_ENABLED   every cited article outside the case's area
#       and posture lists is checked, by its real heading, against the use the
#       draft makes of it (h7: 609-bis c.p., sexual violence, cited as a
#       mitigating circumstance in a corruption case); a mismatch is flagged.
DRAFT_CITATIONS_ONLY_FLAG = "DRAFT_CITATIONS_ONLY_ENABLED"
CLIENT_STANCE_CHECK_FLAG = "CLIENT_STANCE_CHECK_ENABLED"
QUALIFICATION_CHECK_FLAG = "QUALIFICATION_CHECK_ENABLED"

_REF_NUMBER = r"\d+(?:\s*-?\s*" + SUFFIX + r"(?![a-z]))?"
# "art. 56", "artt. 112 e 114", "articoli 2043, 2048 e 2059", "Articolo 593".
_ARTICLE_RUN_RE = re.compile(
    r"\b(?:artt?\.?|articol[oi])\s*(" + _REF_NUMBER + r"(?:\s*(?:,|e|ed)\s*" + _REF_NUMBER + r")*)",
    re.IGNORECASE,
)
_RUN_NUMBER_RE = re.compile(_REF_NUMBER, re.IGNORECASE)
# What may sit between the number and the code: a comma, "comma 3", "co. 2, n. 1", "del", "della".
_CODE_LEAD_RE = re.compile(
    r"^\s*,?\s*(?:(?:co\.|comma|commi)\s*\d+(?:\s*,?\s*(?:lett\.|lettera|n\.)\s*\w+)?\s*,?\s*)?"
    r"(?:del(?:la|lo|l['’])?\s+|d\.\s*)?",
    re.IGNORECASE,
)
# Codes written out in full, longest first.
_CODE_NAMES = [
    (r"codice di procedura penale", "c.p.p."), (r"codice di procedura civile", "c.p.c."),
    (r"codice del processo amministrativo", "c.p.a."), (r"codice della strada", "c.d.s."),
    (r"codice civile", "c.c."), (r"codice penale", "c.p."),
]
_DECISION_NUMBER_RES = (
    re.compile(r"\b(\d{2,6})\s*/\s*((?:19|20)\d{2})\b"),
    re.compile(r"\bn\.?\s*(\d{2,6})\s+(?:del|anno)\s+((?:19|20)\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{2,6})-((?:19|20)\d{2})\b"),
)


def _code_after(tail: str, documents: List[Dict[str, Any]]) -> Optional[str]:
    """The code a citation's tail names: a sigla or a code in full, else the
    name of one of `documents` (the retrieved non-standard laws) resolved the
    way issue spotting resolves a code -- or None."""
    from .issue_spotting import resolve_code

    rest = tail[_CODE_LEAD_RE.match(tail).end():]
    for pattern, abbr in _CODE_SPELLINGS:
        if re.match(rf"(?:{pattern})", rest, re.IGNORECASE):
            return abbr
    low = rest.lower()
    for name, abbr in _CODE_NAMES:
        if low.startswith(name):
            return abbr
    if documents:
        words = re.split(r"[\n;:()\[\]]", rest)[0].split()[:8]
        for n in range(len(words), 0, -1):
            doc = resolve_code(" ".join(words[:n]), documents)
            if doc:
                return doc["name"]
    return None


def _cited_articles(draft: str, known: set, documents: List[Dict[str, Any]]) -> List[tuple]:
    """Every (number, code) the draft cites, in order, deduplicated. A number
    with no code is attributed to the one known article with that number, if
    there is exactly one."""
    refs: List[tuple] = []
    for m in _ARTICLE_RUN_RE.finditer(draft):
        code = _code_after(draft[m.end():m.end() + 120], documents)
        for n in _RUN_NUMBER_RE.findall(m.group(1)):
            number = _article_num(n)
            if code:
                ref = (number, code)
            else:
                same = [k for k in known if k[0] == number]
                if len(same) != 1:
                    continue
                ref = same[0]
            if ref not in refs:
                refs.append(ref)
    return refs


def _decision_numbers(text: str) -> set:
    return {(m.group(1), m.group(2)) for regex in _DECISION_NUMBER_RES for m in regex.finditer(text or "")}


def _ruling_is_cited(label: str, draft_low: str, draft_numbers: set) -> bool:
    """A ruling (or any source without an article tag) is cited when the draft
    names it, or gives its decision or docket number."""
    norm = " ".join(label.lower().split())
    if norm and norm in draft_low:
        return True
    return bool(_decision_numbers(label) & draft_numbers)


def _draft_sources(draft: str, article_rows: List[Dict[str, Any]], ruling_rows: List[Dict[str, Any]],
                   documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows the draft actually refers to: articles it cites with their code,
    rulings and other sources it names or numbers. Every row of a cited article
    is kept (its commi)."""
    known = {r["_article"] for r in article_rows if r.get("_article")}
    row_docs = {(r.get("d") or {}).get("name") for r in article_rows if r.get("_article")
                and r["_article"][1] not in _CODE_PREFIXES}
    docs = [d for d in documents if d.get("name") in row_docs]
    cited = set(_cited_articles(draft, known, docs))
    draft_low = " ".join(draft.lower().split())
    draft_numbers = _decision_numbers(draft)
    kept = []
    for row in article_rows + ruling_rows:
        if row.get("_article"):
            if row["_article"] in cited:
                kept.append(row)
        elif _ruling_is_cited(_ruling_label(row), draft_low, draft_numbers):
            kept.append(row)
    vlog("case_analysis_draft_sources", {"cited": sorted(f"{n} {c}" for n, c in cited),
                                         "rows_in": len(article_rows) + len(ruling_rows), "rows_kept": len(kept)})
    return kept


_SECTION3_RE = re.compile(r"(?m)^[ \t#>*]*3\.\s*\**\s*Argomenti\b.*$")
# Section 4 by its prescribed title: arguments inside section 3 may be numbered "#### 4." too.
_SECTION4_RE = re.compile(r"(?m)^[ \t#>*]*4\.\s*\**\s*Punti\b")

_STANCE_SYSTEM = (
    "Sei un avvocato italiano che rivede la bozza di un collega. Ti vengono dati il caso, la parte per cui "
    "l'avvocato agisce e la sezione 3 della bozza, che dovrebbe contenere solo argomenti a favore di quella "
    "parte. Per ciascun argomento chiediti: se il giudice lo accogliesse, a chi gioverebbe? Elenca SOLO gli "
    "argomenti che, accolti, giovano alla controparte o indeboliscono la posizione della parte assistita (per "
    "esempio, per il danneggiato, un argomento che prova la diligenza del danneggiante; per l'imputato, uno che "
    "conferma un elemento dell'accusa). Un argomento difensivo o subordinato della parte assistita non va "
    'elencato. Restituisci SOLO un oggetto JSON: {"parte_assistita": stringa, "argomenti_contrari": [{"argomento": '
    'il titolo o la frase iniziale dell\'argomento, "perche": in una frase, a chi giova e perché}]}; lista vuota '
    "se non ce ne sono."
)


def _section_3_bounds(draft: str) -> Optional[tuple]:
    start = _SECTION3_RE.search(draft)
    if not start:
        return None
    end = _SECTION4_RE.search(draft, start.end())
    return (start.start(), end.start() if end else len(draft))


def _stance_party(analysis: Dict[str, Any]) -> str:
    party = str(analysis.get("parte_assistita") or "").strip() or "la parte assistita"
    position = str(analysis.get("posizione") or "").strip()
    return f"{party} ({position})" if position else party


def _opposing_arguments(facts: str, request: str, analysis: Dict[str, Any], section: str) -> List[Dict[str, str]]:
    """Section 3's arguments that favour the other side; [] on any failure."""
    tasks = "\n".join(f"- {t}" for t in ((analysis.get("frame") or {}).get("tasks") or []))
    human = (f"CASO:\n{_case_text(facts, request)[:4000]}\n\n"
             + (f"QUESITI:\n{tasks}\n\n" if tasks else "")
             + f"PARTE ASSISTITA: {_stance_party(analysis)}\n\nSEZIONE 3 DELLA BOZZA:\n{section[:6000]}")
    t0 = time.time()
    try:
        parsed = _chat_json([SystemMessage(content=_STANCE_SYSTEM), HumanMessage(content=human)],
                            max_tokens=900, temperature=0)
    except Exception as exc:
        logger.warning("client stance check failed: %s", exc)
        return []
    finally:
        vlog("case_analysis_stance_llm", duration_ms=(time.time() - t0) * 1000)
    items = parsed.get("argomenti_contrari") if isinstance(parsed, dict) else None
    return [{"argomento": str(i.get("argomento") or "").strip(), "perche": str(i.get("perche") or "").strip()}
            for i in (items if isinstance(items, list) else [])
            if isinstance(i, dict) and str(i.get("argomento") or "").strip()]


def _check_client_stance(draft: str, facts: str, request: str, analysis: Dict[str, Any],
                         system: str, human: str) -> str:
    """The draft with section 3 rewritten once when the check finds arguments
    for the other side in it; unchanged otherwise or on any failure."""
    bounds = _section_3_bounds(draft)
    if not bounds:
        vlog("case_analysis_stance", {"skipped": "no section 3"})
        return draft
    section = draft[bounds[0]:bounds[1]]
    opposing = _opposing_arguments(facts, request, analysis, section)
    vlog("case_analysis_stance", {"party": _stance_party(analysis), "opposing": opposing})
    if not opposing:
        return draft
    listing = "\n".join(f"- {o['argomento']}: {o['perche']}" for o in opposing)
    heading = draft[bounds[0]:].split("\n", 1)[0]
    rewrite = (
        f"{human}\n\nQuesta è la sezione 3 che hai scritto:\n{section}\n\n"
        f"Una revisione ha trovato che questi argomenti giovano alla controparte, non a {_stance_party(analysis)}:\n"
        f"{listing}\n\nRiscrivi SOLO la sezione 3, con la stessa intestazione ({heading.strip()}): ogni argomento "
        "deve escludere, ridurre o contrastare la responsabilità o la pretesa contro la parte assistita, o "
        "sostenerne la pretesa. Usa gli stessi fatti, norme e pronunce delle fonti; un fatto che favorisce la "
        "controparte va trattato come obiezione da superare, non come argomento. Restituisci solo la sezione 3."
    )
    try:
        new_section = _call_chat([SystemMessage(content=system), HumanMessage(content=rewrite)], max_tokens=2000)
    except Exception as exc:
        logger.warning("client stance rewrite failed, draft kept: %s", exc)
        return draft
    new_section = (new_section or "").strip()
    if not _SECTION3_RE.match(new_section):
        logger.warning("client stance rewrite did not return a section 3, draft kept")
        return draft
    return draft[:bounds[0]] + new_section + "\n\n" + draft[bounds[1]:]


_QUALIFICATION_SYSTEM = (
    "Sei un giurista. Per ciascun articolo citato in una bozza di parere ti vengono dati la sua rubrica reale "
    "(il titolo ufficiale nel codice) e la frase della bozza che lo cita. Decidi se la rubrica è compatibile con "
    "l'uso che la frase ne fa: un articolo sulla violenza sessuale citato come circostanza attenuante in un caso "
    "di corruzione è incompatibile; un articolo generale (concorso di persone, circostanze, elemento psicologico) "
    "citato per quell'istituto è compatibile. Nel dubbio è compatibile. Restituisci SOLO un array JSON di oggetti "
    '{"articolo": l\'identificativo dato, "compatibile": true o false, "perche": una frase}.'
)


# FLAGGED_REMOVE_ENABLED (diagnose_long run 1415, a sample of 20 rubrica-incompatible citations: about 9
# real junk -- 757, 681 c.c., 600-bis c.p., 119/1/135 c.p.a., 35 c.p.c., a maxims collection cited as a law --
# most already marked by this check, which only marked them): a citation the check finds incompatible is
# removed, unless the pipeline has its own evidence for the article (_protected_refs: graded 2-3, forced,
# voted, two or more channels, central, named in the case); then the mark stays, or goes when the check
# calls it a rubrica-wording mismatch ("tipo": "formulazione"). And an article of a document that is not a
# law is never cited as one (_citation_hygiene).
FLAGGED_REMOVE_FLAG = "FLAGGED_REMOVE_ENABLED"
_QUALIFICATION_SYSTEM_TYPED = _QUALIFICATION_SYSTEM.replace(
    '"perche": una frase}.',
    '"perche": una frase, "tipo": solo per un articolo incompatibile, "materia" se disciplina un\'altra '
    'materia o un altro istituto rispetto all\'uso della frase, "formulazione" se riguarda lo stesso istituto '
    'o uno vicino e solo la rubrica è formulata diversamente dalla frase}.')
# WRITER_RULES_ENABLED (the lawyer's review of the pool case's answer): an article about the inverse situation
# (the damage a party causes cited for the damage that party suffers) is never right for the sentence; the check
# reads the article's opening text too, and such a citation is removed (unless the case names the article).
_QUALIFICATION_DIRECTION = (
    " Leggi anche l'inizio del testo: è incompatibile anche un articolo che disciplina la situazione inversa a "
    "quella della frase (il danno cagionato da un soggetto citato per il danno subito da quel soggetto; la "
    "responsabilità del debitore citata per quella del creditore; l'obbligo del conduttore citato per quello del "
    "locatore): \"tipo\": \"direzione\".")
_LAW_DOCUMENT_TYPES = {"primary", "legislation"}
_CCNL_CASE_RE = re.compile(r"\bc\.?\s*c\.?\s*n\.?\s*l\.?(?!\w)|contratt\w*\s+collettiv", re.IGNORECASE)


def _citation_context(draft: str, ref: tuple, documents: Optional[List[Dict[str, Any]]] = None) -> str:
    """The line of the draft where it first cites ref (any suffix spelling), cut to 400 chars. With
    documents (FLAGGED_REMOVE_ENABLED, and scripts/diagnose_long.py): the sentence that cites ref itself,
    by number and code (a line citing 2053 c.c. is not 1982's, "658 Codice Civile" is not 658 c.p.c.),
    its whole line when the sentence is under 120 characters."""
    if documents is not None:
        from . import plan_then_write as ptw
        for known in (None, {ref}):        # by its code first; a bare number only if no line names the code
            for start, end, refs in ptw._citations(draft, documents, known):
                if any(ptw._same_ref(r, ref) for r in refs):
                    s, e = ptw._sentence_span(draft, start, end)
                    if e - s >= 120:
                        return draft[s:e].strip()[:400]
                    a, b = draft.rfind("\n", 0, start) + 1, draft.find("\n", end)
                    return draft[a:b if b != -1 else len(draft)].strip()[:400]
        return ""
    number = re.escape(ref[0]).replace(r"\-", r"\s*-?\s*")
    m = re.search(rf"\b(?:artt?\.?|articol[oi])\s*(?:[\d\s,e-]*?)\b{number}\b", draft, re.IGNORECASE)
    if not m:
        return ""
    start = draft.rfind("\n", 0, m.start()) + 1
    end = draft.find("\n", m.end())
    return draft[start:end if end != -1 else len(draft)].strip()[:400]


def _check_qualifications(draft: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                          session, documents: List[Dict[str, Any]], case_text: str = "") -> str:
    """The draft with each cited article whose real heading does not fit the use
    the draft makes of it flagged at its first citation. Articles of the case's
    area/posture lists are trusted; the others are checked in one call.
    FLAGGED_REMOVE_ENABLED: such a citation is removed instead (plan_then_write.remove_citations),
    unless _protected_refs has the article; a protected one keeps its mark, or loses it when the
    check calls the mismatch only the rubrica's wording."""
    from . import method_candidates

    remove = feature_flags.enabled(FLAGGED_REMOVE_FLAG)
    from . import plan_then_write
    direction = plan_then_write._writer_rules()

    trusted = set(method_candidates.candidate_refs(*method_candidates.case_lists(analysis)))
    known = {r["_article"] for r in article_rows if r.get("_article")}
    cited = [ref for ref in _cited_articles(draft, known, documents) if ref not in trusted]
    if not cited:
        return draft
    headings = {}
    rows_by_ref = {}
    for row in article_rows:
        if row.get("_article"):
            rows_by_ref.setdefault(row["_article"], row)
    missing = [ref for ref in cited if ref not in rows_by_ref]
    if missing:
        try:
            for row in _fetch_any(session, documents, missing):
                rows_by_ref.setdefault(row.get("_article"), row)
        except Exception as exc:
            logger.warning("qualification check: heading fetch failed: %s", exc)
    for ref in cited:
        s = (rows_by_ref.get(ref) or {}).get("s") or {}
        heading = (s.get("titolo_articolo") or s.get("title") or "").strip()
        context = _citation_context(draft, ref, documents if remove or direction else None)
        if heading and context:
            headings[f"{ref[0]} {ref[1]}"] = (heading, context, ref)
    if not headings:
        return draft
    opening = {label: " ".join(str(((rows_by_ref.get(ref) or {}).get("s") or {}).get("plain_text") or "").split())[:300]
               for label, (_, _, ref) in headings.items()} if direction else {}
    human = "\n\n".join(f"[{label}]\nRubrica: {heading}" + (f"\nTesto: {opening[label]}" if opening.get(label) else "")
                        + f"\nFrase: {context}" for label, (heading, context, _) in headings.items())
    system = _QUALIFICATION_SYSTEM_TYPED if remove or direction else _QUALIFICATION_SYSTEM
    if direction:
        system = system.replace(" Nel dubbio è compatibile.", _QUALIFICATION_DIRECTION + " Nel dubbio è compatibile.")
    t0 = time.time()
    try:
        verdicts = _chat_json([SystemMessage(content=system), HumanMessage(content=human)],
                              max_tokens=max(600, (110 if remove or direction else 90) * len(headings)), temperature=0)
    except Exception as exc:
        logger.warning("qualification check failed, draft kept: %s", exc)
        return draft
    finally:
        vlog("case_analysis_qualification_llm", duration_ms=(time.time() - t0) * 1000)
    if not isinstance(verdicts, list):
        return draft
    flagged = []
    protected = _protected_refs(article_rows, _named_refs(case_text, documents)) if remove else {}
    named = _named_refs(case_text, documents) if direction else []
    to_remove, kept, note_dropped, inverse = [], {}, [], []
    # STRICT_ARTICLES_ENABLED (round 3): a flagged article kept only by the pipeline's own evidence (graded,
    # forced, voted, channels, central -- not named in the case, not a procedural tool) gets a second opinion
    # on its full text; "it governs no question of the case" removes it (_second_opinion).
    second_opinion = remove and feature_flags.enabled(STRICT_ARTICLES_FLAG)
    deferred: List[tuple] = []
    for v in verdicts:
        if not isinstance(v, dict) or v.get("compatibile") is not False:
            continue
        entry = headings.get(_applicability_key(str(v.get("articolo") or "")))
        entry = entry or next((e for label, e in headings.items()
                               if _applicability_key(label) == _applicability_key(str(v.get("articolo") or ""))), None)
        if entry:
            heading, _, ref = entry
            label = f"{ref[0]} {ref[1]}"
            flagged.append(label)
            if direction and str(v.get("tipo") or "").strip().lower() == "direzione" and \
                    _protection(ref, {n: "" for n in named}) is None:
                inverse.append(label)
                to_remove.append(ref)          # the inverse situation: removed even when protected
                continue
            if remove:
                why = _protection(ref, protected)
                if why is None:
                    to_remove.append(ref)
                    continue
                if second_opinion and why not in _SECOND_OPINION_EXEMPT:
                    deferred.append((ref, heading, why, str(v.get("tipo") or "").strip().lower()))
                    continue
                kept[label] = why
                if str(v.get("tipo") or "").strip().lower() == "formulazione":
                    note_dropped.append(label)
                    continue
            draft = _flag_citation(draft, ref, f" [DA VERIFICARE: l'art. {ref[0]} {ref[1]} riguarda «{heading}»]")
    governs = _second_opinion(case_text, [ref for ref, *_ in deferred], rows_by_ref) if deferred else {}
    for ref, heading, why, kind in deferred:
        label = f"{ref[0]} {ref[1]}"
        if governs.get(ref) is False:
            to_remove.append(ref)
            continue
        kept[label] = why
        if kind == "formulazione":
            note_dropped.append(label)
            continue
        draft = _flag_citation(draft, ref, f" [DA VERIFICARE: l'art. {ref[0]} {ref[1]} riguarda «{heading}»]")
    log = {"checked": sorted(headings), "flagged": flagged}
    if deferred:
        log["second_opinion"] = {f"{n} {c}": governs.get((n, c)) for (n, c), *_ in deferred}
    if direction:
        log["inverse"] = inverse
    if remove or to_remove:
        draft, removal = plan_then_write.remove_citations(draft, to_remove, documents, known)
        log.update({"removed": [f"{n} {c}" for n, c in to_remove], "kept": kept, "note_dropped": note_dropped})
        vlog("case_analysis_flagged_removed", {"removed": removal, "count": len(to_remove)})
    vlog("case_analysis_qualification", log)
    return draft


_SECOND_OPINION_EXEMPT = {"named in the case", "procedural tool for the facts"}
_SECOND_OPINION_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati un caso e il testo integrale di un articolo di legge. Decidi se "
    "l'articolo disciplina almeno una delle questioni giuridiche che il caso pone, anche come regola generale "
    "applicabile ai suoi fatti; non basta che usi parole del caso o che riguardi un istituto vicino. Nel dubbio "
    'rispondi true. Restituisci SOLO un oggetto JSON {"disciplina": true o false, "perche": una frase}.')
_SECOND_OPINION_CHARS = 3000


def _second_opinion(case_text: str, refs: List[tuple], rows_by_ref: Dict[tuple, Dict[str, Any]]) -> Dict[tuple, Any]:
    """STRICT_ARTICLES_ENABLED: {article: True / False / None} -- one call per article, in parallel, the case and
    the article's full text: does it govern any question of the case? None (kept as before) without a text, on a
    failed call or an unreadable reply. Logged as case_analysis_second_opinion."""
    from concurrent.futures import ThreadPoolExecutor

    def ask(ref):
        text = " ".join(str(((rows_by_ref.get(ref) or {}).get("s") or {}).get("plain_text") or "").split())
        if not text:
            return ref, None, "no text"
        try:
            reply = _chat_json([SystemMessage(content=_SECOND_OPINION_SYSTEM),
                                HumanMessage(content=f"CASO:\n{(case_text or '')[:3000]}\n\nARTICOLO: art. {ref[0]} "
                                                     f"{ref[1]}\n{text[:_SECOND_OPINION_CHARS]}")],
                               max_tokens=200, temperature=0)
        except Exception as exc:
            return ref, None, f"failed: {type(exc).__name__}"
        if not isinstance(reply, dict) or not isinstance(reply.get("disciplina"), bool):
            return ref, None, "unreadable"
        return ref, reply["disciplina"], str(reply.get("perche") or "")[:200]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(6, len(refs)) or 1) as pool:
        answers = list(pool.map(ask, refs))
    vlog("case_analysis_second_opinion", {"answers": [{"article": f"{r[0]} {r[1]}", "governs": g, "why": w}
                                                      for r, g, w in answers]},
         duration_ms=(time.time() - t0) * 1000)
    return {ref: governs for ref, governs, _ in answers}


def _named_refs(case_text: str, documents: List[Dict[str, Any]]) -> List[tuple]:
    """The articles the case's own text cites, by code or by a law's name."""
    return list(dict.fromkeys(_refs_in(case_text) + _cited_articles(case_text or "", set(), documents)))


def _protected_refs(article_rows: List[Dict[str, Any]], named: List[tuple]) -> Dict[tuple, str]:
    """{article: why} the pipeline has its own evidence for, so FLAGGED_REMOVE_ENABLED never removes it:
    named in the case, forced into the plan (PLAN_KEEP_CHECKED with CHECK_GRADED), graded 2-3, voted by
    similar rulings, found by two or more channels, resolved for a central issue."""
    from . import plan_then_write
    out: Dict[tuple, str] = {ref: "named in the case" for ref in named}
    if plan_then_write._discipline():
        for ref, why in plan_then_write._forced(article_rows).items():
            out.setdefault(ref, f"forced ({why})")
    for row in article_rows:
        ref = row.get("_article")
        if not ref:
            continue
        grade = row.get("_grade")
        why = (f"grade {grade}" if isinstance(grade, int) and grade >= 2 else "voted" if row.get("_vote")
               else "two or more channels" if row.get("_multi_signal") else "central issue" if row.get("_central")
               else "procedural tool for the facts" if row.get("_procedural_tool") else None)
        if why:
            out.setdefault(ref, why)
    return out


def _protection(ref: tuple, protected: Dict[tuple, str]) -> Optional[str]:
    from . import plan_then_write
    return next((why for other, why in protected.items() if plan_then_write._same_ref(ref, other)), None)


def _citation_hygiene(draft: str, session, documents: List[Dict[str, Any]], article_rows: List[Dict[str, Any]],
                      case_text: str) -> str:
    """FLAGGED_REMOVE_ENABLED: the draft without the citations of articles of a document that is not a law
    (document_type other than primary/legislation: maxims collections, books, contract models; a CCNL
    unless the case speaks of one), never one the case names. Logged as case_analysis_citation_hygiene."""
    from . import plan_then_write
    known = {r["_article"] for r in article_rows if r.get("_article")}
    cited = _cited_articles(draft, known, documents)
    names = sorted({code for _, code in cited if code not in _CODE_PREFIXES})
    if not names:
        return draft
    try:
        types = {r["name"]: {t for t in r["types"] if t} for r in session.run(
            "MATCH (d:Document) WHERE d.name IN $names RETURN d.name AS name, "
            "collect(DISTINCT d.document_type) AS types", names=names).data()}
    except Exception as exc:
        logger.warning("citation hygiene: document types not read, draft kept: %s", exc)
        return draft
    named = _named_refs(case_text, documents)
    ccnl_case = bool(_CCNL_CASE_RE.search(case_text or ""))
    bad = [ref for ref in cited if ref[1] in types and types[ref[1]] and not types[ref[1]] & _LAW_DOCUMENT_TYPES
           and not ("ccnl" in types[ref[1]] and ccnl_case) and _protection(ref, {n: "" for n in named}) is None]
    if not bad:
        return draft
    draft, removal = plan_then_write.remove_citations(draft, bad, documents, known)
    vlog("case_analysis_citation_hygiene", {"removed": [f"{n} {c}" for n, c in bad],
                                            "types": {c: sorted(types[c]) for _, c in bad}, "log": removal})
    return draft


def _flag_citation(draft: str, ref: tuple, mark: str) -> str:
    """mark after ref's first citation, replacing a plain [DA VERIFICARE] already there."""
    for m in _CITED_ARTICLE_RE.finditer(draft):
        if (_article_num(m.group(1)), _code_of(m.group(2))) != ref:
            continue
        end = m.end()
        if draft.startswith(_UNVERIFIED, end):
            return draft[:end] + mark + draft[end + len(_UNVERIFIED):]
        if draft.startswith(" [DA VERIFICARE", end):
            return draft
        return draft[:end] + mark + draft[end:]
    return draft


# METHOD_CANDIDATES_ENABLED (default false; acts only with APPLICABILITY_CHECK_
# ENABLED, which is what filters them): the method articles of the case's areas
# and the procedural articles of its posture (src/rag/method_candidates.py,
# a draft for the lawyer to validate) are added as candidates to whatever the
# selection picked. They are tagged METHOD_CANDIDATES_SOURCE, never exempt from
# the check, never protected from the prompt's trimming, and never seeds for
# cross-reference broadening.
METHOD_CANDIDATES_FLAG = "METHOD_CANDIDATES_ENABLED"
METHOD_CANDIDATES_SOURCE = "method_candidates"


def _add_method_candidates(session, driver, database: str, analysis: Dict[str, Any],
                           picked: Dict[str, Any]) -> Dict[str, Any]:
    """picked with the area and posture candidates that are not already among
    its article rows appended to "article_rows" and "selected"; unchanged with
    the flag off, without the applicability check, or on any failure."""
    if not feature_flags.enabled(METHOD_CANDIDATES_FLAG):
        return picked
    if not feature_flags.enabled("APPLICABILITY_CHECK_ENABLED"):
        logger.warning("%s needs APPLICABILITY_CHECK_ENABLED to filter its candidates: nothing added",
                       METHOD_CANDIDATES_FLAG)
        return picked
    from . import method_candidates
    from .doc_lookup import _fetch_doc_names

    # CHECKLIST_GATED_ENABLED: only the lists of the case's verified area x posture (case_lists).
    areas, posture = method_candidates.case_lists(analysis)
    refs = method_candidates.candidate_refs(areas, posture)
    have = {r.get("_article") for r in picked.get("article_rows") or [] if r.get("_article")}
    wanted = [ref for ref in refs if ref not in have]
    if not wanted:
        return picked
    try:
        rows = _fetch_any(session, _fetch_doc_names(driver, database), wanted)
    except Exception as exc:
        logger.warning("method candidates: fetch failed, nothing added: %s", exc)
        return picked
    for row in rows:
        row["_source"] = METHOD_CANDIDATES_SOURCE
    added = list(dict.fromkeys(r["_article"] for r in rows if r.get("_article")))
    vlog("case_analysis_method_candidates", {"areas": areas, "posture": posture,
                                             "added": [f"{n} {c}" for n, c in added],
                                             "not_found": [f"{n} {c}" for n, c in wanted if (n, c) not in added]})
    return {**picked, "article_rows": list(picked.get("article_rows") or []) + rows,
            "selected": list(picked.get("selected") or []) + [r for r in added if r not in picked.get("selected", [])],
            "method_candidates": added}


def _regime_rows(session, driver, database: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """FACT_REGIMES_ENABLED: the article rows of the procedural tools the facts call for (fact_signals.
    procedural_tools), tagged _procedural_tool; and, in place, each legal term's article and text
    (fact_signals.attach_term_sources). [] with the flag off, without signals, or on any failure."""
    from . import fact_signals
    signals = analysis.get("fact_signals")
    if not fact_signals.regimes_enabled() or not signals:
        return []
    try:
        from .doc_lookup import _fetch_doc_names
        documents = _fetch_doc_names(driver, database)
        fact_signals.attach_term_sources(session, documents, signals)
        tools = fact_signals.regime_refs(signals)
        rows = _fetch_any(session, documents, list(tools)) if tools else []
    except Exception as exc:
        logger.warning("fact regimes: tools and terms not read: %s", exc)
        return []
    for row in rows:
        row["_procedural_tool"] = tools.get(row.get("_article"), True)
        row["_source"] = row.get("_source") or "procedural_tool"
    vlog("case_analysis_fact_regimes", {
        "qualifications": signals.get("qualifications") or [], "regimes": [r["nome"] for r in signals.get("regimes") or []],
        "tools": [t["id"] for t in signals.get("tools") or []],
        "damages": [d["id"] for d in signals.get("damages") or []],
        "multi_roles": signals.get("multi_roles") or [], "posture": signals.get("posture") or {},
        "tool_articles": sorted({f"{r['_article'][0]} {r['_article'][1]}" for r in rows if r.get("_article")}),
        "terms": [t.get("text") for t in signals.get("terms") or []],
        "term_refs": {t["id"]: t.get("ref") for t in signals.get("terms") or []}})
    return rows


def _with_regime_rows(article_rows: List[Dict[str, Any]], regime_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """article_rows with the procedural tools' rows (_regime_rows) after the applicability check, which never
    removes them: an article already there is tagged, any other appended."""
    have = {r["_article"]: r for r in article_rows if r.get("_article")}
    out = list(article_rows)
    for row in regime_rows:
        ref = row.get("_article")
        if ref in have:
            have[ref]["_procedural_tool"] = row["_procedural_tool"]
        else:
            out.append(row)
            have[ref] = row
    return out


def _support_rulings(facts: str, request: str, article_rows: List[Dict[str, Any]],
                     analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """HYBRID_NARRATIVE_ENABLED: the rulings the writer gets are those citing at least one selected article
    (Cassation boilerplate excluded), ranked by similarity to the facts x articles shared
    (source_tiers.support_rulings), each with its best passage (_ruling_passages: the case's branch only) and,
    when carded, its card (ruling_cards.annotate). [] on any failure: RULING_FIT then has nothing to judge."""
    import neo4j
    from . import ruling_cards, source_tiers
    from .main import driver, NEO4J_DATABASE

    refs = source_tiers.refs_of(article_rows)
    try:
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            support = source_tiers.support_rulings(session, refs, _case_text(facts, request), _MAX_RULINGS,
                                                   weight_shared=True, stage="narrative")
            rows = _ruling_passages(session, support, refs, analysis)
    except Exception as exc:
        logger.warning("hybrid: support rulings failed, none given: %s", exc)
        return []
    store = ruling_cards.store()
    carded = {}
    if store is not None:
        ids = {(r.get("d") or {}).get("id") for r in rows}
        carded = {c["id"]: c for c in store.cards if c["id"] in ids}
    vlog("hybrid_support_rulings", {"articles": [f"{n} {c}" for n, c in refs], "found": len(support),
                                    "kept": len(rows), "carded": len(carded)})
    return ruling_cards.annotate(rows, carded)


def _log_rulings_cited(draft: str, ruling_rows: List[Dict[str, Any]]) -> None:
    """Which rulings given to the writer the draft cites (scripts/diagnose_long.py reads their CITES edges
    against the articles the answer names)."""
    draft_low, numbers = " ".join((draft or "").lower().split()), _decision_numbers(draft or "")
    vlog("case_analysis_rulings_cited", {"rulings": [
        {"id": (r.get("d") or {}).get("id"), "label": _ruling_label(r),
         "cited": _ruling_is_cited(_ruling_label(r), draft_low, numbers), "source": r.get("_source") or ""}
        for r in ruling_rows]})


def run_case_analysis_pipeline(facts: str, session_lang: str = "it", request: str = "") -> dict:
    """A reasoned parere on a case: facts described in the message, or an
    uploaded document (`facts`) with the user's message as `request`.

    Returns {"draft": str, "citations": list} like run_defensive_pipeline.
    """
    import neo4j
    from .main import driver, NEO4J_DATABASE

    pipeline_t0 = time.time()
    case_analysis_v2 = feature_flags.enabled(CASE_ANALYSIS_FLAG)
    pool_fusion = case_analysis_v2 and feature_flags.enabled(_CASE_POOL_FUSION_FLAG)
    analysis = _analyse_case_for_pipeline(facts, session_lang, request)
    # ROUTER_ENABLED: the request's route, confirmed by the case frame; HYBRID_NARRATIVE_ENABLED acts on a
    # narrative only (source_tiers.py). Without the router the strategy pipeline is the narrative path.
    from . import query_router, source_tiers
    route = query_router.route_for_pipeline(_case_text(facts, request), analysis)
    analysis = {**analysis, "route": route["route"]}
    hybrid = source_tiers.hybrid_enabled(route["route"])
    vlog("case_analysis_route", {"route": route["route"], "reasons": route.get("reasons"), "hybrid": hybrid})
    article_rows, ruling_rows, candidates, selected, regime_rows = [], [], [], [], []
    retrieval_t0 = time.time()
    try:
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            select_fn = _select_articles_fusion if pool_fusion else (
                _select_articles_v2 if case_analysis_v2 else _select_articles)
            picked = select_fn(session, driver, NEO4J_DATABASE, facts, request, analysis)
            picked = _add_method_candidates(session, driver, NEO4J_DATABASE, analysis, picked)
            regime_rows = _regime_rows(session, driver, NEO4J_DATABASE, analysis)
            first_rulings, similar = picked["first_rulings"], picked["similar"]
            candidates, selected, article_rows = picked["candidates"], picked["selected"], picked["article_rows"]
            if hybrid:
                # The authority tier: the articles come from primary documents only, whatever channel found
                # them (logged by type; none other may be left: source_tier_hits' non_primary_after).
                article_rows = source_tiers.keep_types(session, article_rows, source_tiers.AUTHORITY_TYPES,
                                                       "authority")
                regime_rows = source_tiers.keep_types(session, regime_rows, source_tiers.AUTHORITY_TYPES,
                                                      "authority_regimes")
            try:
                filled = article_text.fill_texts(session, article_rows + regime_rows)
                if filled:
                    vlog("case_analysis_in_force_only_filled", {"sections": filled})
            except Exception as exc:
                logger.warning("articles holding only their date in force not filled: %s", exc)
            # HYBRID_NARRATIVE_ENABLED: the rulings come after the applicability check, from the articles it
            # kept (_support_rulings), not from searches of their own.
            if not hybrid:
                # Second search with the verified article numbers: rulings on the
                # article itself, not on whatever else the keywords match.
                second = _search_case_law(session, analysis, numbers=[n for n, _ in selected[:4]],
                                          exclude_docs={(r.get("d") or {}).get("id") for r in similar})
                reading = _reading_order(similar, second, first_rulings)
                # RULING_CARDS_ENABLED: the rulings the precomputed cards rank best for the case are
                # read first, each passage led by the card's outcome, side and quote (ruling_cards.py).
                from . import ruling_cards
                carded: Dict[str, Dict[str, Any]] = {}
                if ruling_cards.enabled():
                    searched = len(reading)
                    card_rows, carded = ruling_cards.card_rulings(session, driver, NEO4J_DATABASE,
                                                                  _case_text(facts, request), analysis, picked)
                    reading = ruling_cards.lead_with(card_rows, reading)
                ruling_rows = ruling_cards.annotate(_ruling_passages(session, reading, selected, analysis), carded)
                if carded:
                    vlog("case_analysis_ruling_cards_reading", {
                        "searched": searched, "card_rows": len(carded), "reading": len(reading),
                        "kept": len(ruling_rows),
                        "cards_kept": sum((r.get("d") or {}).get("id") in carded for r in ruling_rows)})
    except Exception as exc:
        logger.warning("case analysis retrieval failed: %s", exc)
    vlog("case_analysis_retrieval", {"v2": case_analysis_v2, "pool_fusion": pool_fusion,
                                     "articles": len(article_rows), "rulings": len(ruling_rows),
                                     "article_refs": sorted({f"{r['_article'][0]} {r['_article'][1]}"
                                                             for r in article_rows if r.get("_article")})},
        duration_ms=(time.time() - retrieval_t0) * 1000)

    # Per-article applicability (round 3, item 3): off unless
    # APPLICABILITY_CHECK_ENABLED. Runs after article_rows/ruling_rows are
    # final, before both the citations and the draft prompt use them.
    if feature_flags.enabled("APPLICABILITY_CHECK_ENABLED"):
        article_rows = _filter_applicable_articles(facts, request, article_rows, analysis)
    if regime_rows:
        article_rows = _with_regime_rows(article_rows, regime_rows)
    voted_kept = sorted({f"{r['_article'][0]} {r['_article'][1]}" for r in article_rows
                         if r.get("_vote") and r.get("_article")})
    if voted_kept:
        vlog("case_pool_vote_kept", {"kept": voted_kept})   # voted articles past the applicability check
    if hybrid:
        ruling_rows = _support_rulings(facts, request, article_rows, analysis)

    ruling_rows = _ruling_fit(facts, request, ruling_rows, analysis)
    principles = _extract_principles(facts, analysis, ruling_rows, request)
    ruling_rows = _with_fit_facts(ruling_rows)
    system = _parere_system(analysis, session_lang)
    articles, rulings, human = _fit_parere_prompt(system, facts, analysis, article_rows, ruling_rows,
                                                  request, principles)
    # PLAN_THEN_WRITE_ENABLED: a plan, then the sections from it (plan_then_write.py); None on
    # any failure, and the parere is written in one call as before.
    planned = None
    if feature_flags.enabled(PLAN_THEN_WRITE_FLAG):
        from . import plan_then_write
        documents_for_plan: List[Dict[str, Any]] = []
        if plan_then_write._discipline():
            try:
                from .doc_lookup import _fetch_doc_names
                documents_for_plan = _fetch_doc_names(driver, NEO4J_DATABASE)
            except Exception as exc:
                logger.warning("plan-then-write: no document names, citations read by code only: %s", exc)
        planned = plan_then_write.write_parere(facts, request, analysis, article_rows, ruling_rows,
                                               articles, rulings, principles, session_lang, documents_for_plan)
    if planned:
        draft = planned
    else:
        draft_t0 = time.time()
        try:
            draft = _call_chat([SystemMessage(content=system), HumanMessage(content=human)],
                               max_tokens=_PARERE_TOKENS)
        except Exception as exc:
            # The estimate was still too low for this text: once more, smaller,
            # rather than an error for a size problem.
            if not _is_context_overflow(exc):
                raise
            logger.warning("draft prompt over the context, retrying once with a smaller one: %s", str(exc)[:200])
            articles, rulings, human = _fit_parere_prompt(system, facts, analysis, article_rows, ruling_rows,
                                                          request, principles, budget=_DRAFT_INPUT_BUDGET * 7 // 10)
            draft = _call_chat([SystemMessage(content=system), HumanMessage(content=human)],
                               max_tokens=_PARERE_TOKENS)
        vlog("case_analysis_draft_llm", duration_ms=(time.time() - draft_t0) * 1000)
    # A planned parere had its arguments checked for the client's side before it was written.
    if feature_flags.enabled(CLIENT_STANCE_CHECK_FLAG) and not planned:
        draft = _check_client_stance(draft, facts, request, analysis, system, human)
    from . import plan_then_write as _ptw
    writer_rules = _ptw._writer_rules()
    if writer_rules and not planned:
        draft = _ptw.review_parere(draft, _case_text(facts, request), article_rows, analysis)
    found = sorted({row["_article"] for row in article_rows if row.get("_article")})
    draft = _mark_unverified(draft, found, articles + "\n" + rulings,
                             wrong_code=feature_flags.enabled(STRATEGY_DRAFT_RULES_FLAG))
    citations_only = feature_flags.enabled(DRAFT_CITATIONS_ONLY_FLAG)
    qualification = feature_flags.enabled(QUALIFICATION_CHECK_FLAG)
    flagged_remove = feature_flags.enabled(FLAGGED_REMOVE_FLAG)
    documents: List[Dict[str, Any]] = []
    if citations_only or qualification or flagged_remove:
        try:
            from .doc_lookup import _fetch_doc_names
            documents = _fetch_doc_names(driver, NEO4J_DATABASE)
            if qualification or flagged_remove:
                with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
                    if flagged_remove:
                        draft = _citation_hygiene(draft, session, documents, article_rows, _case_text(facts, request))
                    if qualification:
                        draft = _check_qualifications(draft, analysis, article_rows, session, documents,
                                                      _case_text(facts, request))
        except Exception as exc:
            logger.warning("draft checks after the draft failed, draft kept: %s", exc)

    arguments_at = "**4." if planned else "**3."      # where the client's arguments start
    # WRITER_RULES_ENABLED: a parere is complete in itself, no "deep-dive" links under it (the lawyer's review).
    topics = [] if writer_rules else _extract_deepdive_topics(
        draft[draft.find(arguments_at):] if arguments_at in draft else draft, {})
    if topics:
        draft += "\n\n" + _format_deepdive_suggestion(topics)
    if ruling_rows or _UNVERIFIED.rstrip("]") in draft:
        draft += (
            "\n\n⚠️ Le sentenze citate provengono dal corpus documentale; i riferimenti segnati "
            "[DA VERIFICARE] non sono stati trovati nelle fonti consultate. Verificare tutto prima dell'uso."
        )
    # A row whose text lived only in `abstract` (special/dottrina books, e.g.
    # "Codice Penale Commentato") fails _extract_citations' length gate and
    # comes back with sections: [] — a citation naming a source with nothing
    # in it. Drop it here rather than let it reach the user.
    _log_rulings_cited(draft, ruling_rows)
    source_rows = (_draft_sources(draft, article_rows, ruling_rows, documents) if citations_only
                   else article_rows + ruling_rows)
    citations = [c for c in _rows_to_sources(source_rows) if c.get("sections")]
    vlog("case_analysis_pipeline", {"v2": case_analysis_v2, "pool_fusion": pool_fusion},
        duration_ms=(time.time() - pipeline_t0) * 1000)
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
