#!/usr/bin/env python3
"""v60h (10): every user-facing Italian string Code has drafted since the interactive flow began, in one table for the
PO (key, text, where it appears) -- docs/copy_review.md. Re-runnable; reads the source only.

  1. src/rag/interaction_copy.py: each constant (and each entry of a dict of texts) whose comment says Code drafted
     it -- «my draft» / «my drafts», or a version tag after the PO's v54 texts (v55 … v60h) with no «PO» in it; the
     PO's own texts (v54, verbatim; «the PO's text», «the PO's words») are left out. Where: the functions that read it
     (C.KEY in src/rag), and the copy's own section (round 1, the questions, round 2, the answer, closing).
  2. The user-facing strings Code drafted outside the copy file (EXTRA: §4's procedural lines, the closed-list
     mediazione and negoziazione texts, the check's and the read-back's marks): every string literal of 12 characters or
     more, with a space, in the named constants and functions.
  3. v60i (9): the texts of the closed lists kept as data (DATA: the labour dispute's procedural facts; v62: the posture's
     acts and goals, the procedural section's blocks by stage; v63: the general-part concepts' articles), each with
     its reference.

  python scripts/copy_export.py                    # docs/copy_review.md
  python scripts/copy_export.py --out /tmp/copy.md
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "src" / "rag" / "interaction_copy.py"
SRC = ROOT / "src" / "rag"
_DRAFT_RE = re.compile(r"\bmy drafts?\b", re.IGNORECASE)
_PO_RE = re.compile(r"\bPO\b(?!\s+review)")
_VERSION_RE = re.compile(r"\bv(5[5-9]|6\d)[a-z]?\b")
# (module, name, where): constants or functions of other modules whose strings reach the user
EXTRA = (
    ("render_computed", "listed_mediation", "Risposta, «Profili processuali»: la mediazione obbligatoria (lista chiusa)"),
    ("render_computed", "negotiation_of", "Risposta, «Profili processuali»: la negoziazione assistita"),
    ("render_computed", "NEGOTIATION_TITLE", "Risposta, «Profili processuali»: titolo del paragrafo"),
    ("render_computed", "PROCEDURE_TITLE", "Risposta, «Profili processuali»: titolo del paragrafo dei reati"),
    ("render_computed", "_fact_lines", "Risposta, «Profili processuali»: «Procedibilità e competenza»"),
    ("procedural_facts", "QUERELA_DEFAULT", "Risposta, «Procedibilità e competenza»: termine di querela"),
    ("procedural_facts", "REMISSIONE_DEFAULT", "Risposta, «Procedibilità e competenza»: remissione"),
    ("procedural_facts", "UFFICIO_DEFAULT", "Risposta, «Procedibilità e competenza»: procedibilità"),
    ("procedural_facts", "competenza", "Risposta, «Procedibilità e competenza»: competenza"),
    ("procedural_facts", "extract", "Risposta, «Procedibilità e competenza»: procedibilità e remissione"),
    ("record_check", "check", "Risposta, ogni sezione: segno su una frase da verificare", r"^\s*\(da verificare.*\)$"),
    ("readback", "UNVERIFIED", "Risposta: segno su un fatto della parte non trovato nel caso"),
    # v70: the lawyer's method's own texts (LAWYER_METHOD)
    ("lawyer_method", "facts_section", "Metodo, «Fatti»: certi, ipotizzabili, mancanti"),
    ("lawyer_method", "changes", "Metodo, «Fatti»: che cosa cambierebbe un fatto mancante"),
    ("lawyer_method", "subsumption_rows", "Metodo, «Sussunzione»: il fatto di un elemento"),
    ("lawyer_method", "subsumption_section", "Metodo, «Sussunzione»: le tabelle e le circostanze"),
    ("lawyer_method", "NO_AGGRAVANTE", "Metodo, «Sussunzione»: nessuna aggravante integrata"),
    ("lawyer_method", "AGGRAVANTI_CHECKED", "Metodo, «Sussunzione»: le aggravanti verificate"),
    ("lawyer_method", "_assessment", "Metodo, «Conclusioni»: il perché dell'esito probabile"),     # v79: the body
    ("lawyer_method", "HYPOTHESIS_LABEL", "Metodo, «Sintesi» e «Conclusioni»: l'addebito ricavato dalla prima lettura"),
    ("lawyer_method", "NO_CHARGE_WHY", "Metodo, «Conclusioni»: nessun addebito né domanda nel quesito"),
    ("lawyer_method", "TRANSAZIONE_ARGUMENT", "Metodo, «Soluzioni a confronto»: l'argomento della transazione del "
                                              "convenuto (v79.8)"),
    ("lawyer_method", "PENAL_ROUTE_LINE", "Metodo, «Sintesi»: la via penale di un caso civile (v79.8)"),
    ("lawyer_method", "PENAL_ROUTE_CONCLUSION", "Metodo, «Conclusioni»: la via penale di un caso civile (v79.8)"),
    ("lawyer_method", "PENAL_ROUTE_UNKNOWN", "Metodo, la via penale: procedibilità non letta (v79.8)"),
    ("lawyer_method", "NO_ACTION_LINE", "Risposta, la nota del metodo non applicato: nessuna norma d'azione (le parole "
                                        "del PO, v79.8.6; v79.10.1: mai linea principale; v79.12.5, voce 2: in "
                                        "«Soluzioni a confronto» quando le dieci sezioni restano)"),
    ("lawyer_method", "NO_ACTION_EXPIRED_LINE", "Metodo, «Sintesi»: ogni domanda preclusa da un termine decorso "
                                                "(v79.12.4, voce 7; v79.12.5, voce 2: le dieci sezioni restano)"),
    # v79.12.5 (item 2): no admitted claim -- the ten sections kept
    ("lawyer_method", "CORPUS_LINE", "Metodo, «Sintesi»: nessuna domanda ammessa, o un istituto senza modulo né norme "
                                     "lette della sua famiglia (le parole del PO, v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_HEAD", "Metodo, «Soluzioni a confronto»: nessuna linea e nessuna previsione, senza "
                                       "domanda ammessa (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_EXPIRED_WHY", "Metodo, «Soluzioni a confronto»: il perché, ogni domanda preclusa da un "
                                              "termine (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_READ", "Metodo, «Soluzioni a confronto»: titolo dell'elenco delle norme lette "
                                       "(v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_NONE_READ", "Metodo, «Soluzioni a confronto»: nessuna norma letta (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_ARTICLE_WHY", "Metodo, «Soluzioni a confronto»: perché una norma letta non dà una "
                                              "linea al cliente (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_QUESTION", "Metodo, «Questione giuridica»: senza domanda ammessa (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_OBJECTIONS", "Metodo, «Obiezioni e risposte»: senza linea proposta (v79.12.5, voce 2)"),
    ("lawyer_method", "NO_CLAIM_CONCLUSION", "Metodo, «Conclusioni»: senza linea né previsione (v79.12.5, voce 2)"),
    ("lawyer_method", "MISSING_LAWS_LINE", "Metodo, «Sintesi» e «Conclusioni»: la disciplina speciale assente "
     "dall'archivio (le parole del PO, v79.9)"),
    ("lawyer_method", "NO_ACTION_SUBJECT", "Risposta, la nota senza norma d'azione: nessun istituto nominato (v79.8.5)"),
    ("lawyer_method", "CONTAINMENT_LINE", "Metodo, «Soluzioni a confronto»: la linea di contenimento dell'imputato "
                                          "(v79.7: sempre offerta, ultima)"),
    ("lawyer_method", "ASSESSMENT_LINES", "Metodo, «Conclusioni»: esito probabile, grado di certezza, obiettivo"),
    ("lawyer_method", "RITE_PRECLUDED", "Metodo, «Profili processuali»: un rito alternativo precluso dalla pena del "
                                        "reato contestato (v79.10.3: mai sotto la strategia)"),
    ("lawyer_method", "RITE_TO_VERIFY_LINE", "Metodo, «Soluzioni a confronto» e «Profili processuali»: un rito con la "
                                             "pena edittale non letta (v79.10.3)"),
    ("lawyer_method", "RITI_PRECLUDED_HEAD", "Metodo, «Profili processuali»: i riti preclusi dalla pena (v79.10.3)"),
    ("lawyer_method", "RITI_PRECLUDED_TITLE", "Metodo, materiale dello scrittore: i riti da non proporre (v79.10.3)"),
    ("lawyer_method", "RITE_POINT", "Metodo, «Punti da verificare»: la pena edittale che decide un rito (v79.10.3)"),
    ("lawyer_method", "BEST_CASE", "Metodo, il patteggiamento: il caso più favorevole della pena in concreto (v79.10.7)"),
    ("lawyer_method", "BEST_CASE_ATTENUANTI", "Metodo, il patteggiamento: le attenuanti contate nel caso più "
                                              "favorevole (v79.10.7)"),
    ("lawyer_method", "BEST_CASE_NONE", "Metodo, il patteggiamento: nessuna attenuante integrata o da provare "
                                        "(v79.10.7)"),
    ("lawyer_method", "TEMPORAL_LINE", "Metodo, «Profili processuali»: la disciplina applicabile"),
    ("lawyer_method", "TEMPORAL_UNKNOWN", "Metodo, «Profili processuali»: la data dei fatti non indicata"),
    ("lawyer_method", "TEMPORAL_ABROGATED", "Metodo, «Profili processuali»: norme abrogate"),
    ("lawyer_method", "TEMPORAL_POINTS", "Metodo, «Punti da verificare»: la disciplina applicabile"),
    ("lawyer_method", "temporal", "Metodo, «Profili processuali»: le riforme successive ai fatti"),
    ("lawyer_method", "fallback_thesis", "Metodo, «Soluzioni a confronto»: la tesi di ripiego"),
    ("lawyer_method", "solutions_section", "Metodo, «Soluzioni a confronto»: la tabella e la strategia"),
    ("lawyer_method", "OBJ_PROVEN", "Metodo, «Obiezioni e risposte»: un elemento provato dell'addebito"),
    ("lawyer_method", "OBJ_OPEN", "Metodo, «Obiezioni e risposte»: un elemento da provare dell'addebito"),
    ("lawyer_method", "OBJ_DENIED", "Metodo, «Obiezioni e risposte»: un elemento del cliente da provare"),
    ("lawyer_method", "NO_ANSWER", "Metodo, «Obiezioni e risposte»: risposta da costruire"),
    ("lawyer_method", "objections_section", "Metodo, «Obiezioni e risposte»: nessuna obiezione"),
    ("lawyer_method", "orientations_section", "Metodo, «Orientamenti giurisprudenziali»"),
    ("lawyer_method", "question_section", "Metodo, «Questione giuridica»: dal registro"),
    ("lawyer_method", "points_section", "Metodo, «Punti da verificare»: nessun punto"),
    ("lawyer_method", "compose", "Metodo: sezioni vuote"),
    ("lawyer_method", "ANALYSIS_TITLE", "Metodo, «Norme e analisi per elementi»: titolo delle tabelle"),
    ("lawyer_method", "RISK_LINE", "Metodo, «Conclusioni e valutazione del rischio»: rischio residuo"),
    ("lawyer_method", "STEPS_TITLE", "Metodo, «Conclusioni e valutazione del rischio»: passi successivi"),
    ("lawyer_method", "steps_text", "Metodo, «Conclusioni e valutazione del rischio»: lo scopo di un passo"),
    ("lawyer_method", "summary_section", "Metodo, «Sintesi della soluzione»: senza sintesi del modello"),
    ("lawyer_method", "CLAIM_MARK", "Metodo: una circostanza invocata senza i suoi requisiti"),
    ("lawyer_method", "BARRED_LABEL", "Metodo, «Soluzioni a confronto»: una tesi che nega un elemento integrato"),
    ("lawyer_method", "CANDIDATE_TEXTS", "Metodo, «Soluzioni a confronto»: le tesi candidate del codice"),
    ("lawyer_method", "conclusions_from_selection", "Metodo, «Conclusioni»: dalla tesi scelta"),
    ("lawyer_method", "steps_from_selection", "Metodo, «Conclusioni»: i passi successivi dalla tesi scelta"),
    ("lawyer_method", "STEP_TEXTS", "Metodo, «Conclusioni»: i passi dagli elementi della tesi scelta (v73)"),
    ("lawyer_method", "CODE_ARGUMENTS", "Metodo, «Soluzioni a confronto»: l'argomento delle tesi che non contestano "
                                        "elementi (v74)"),
    ("lawyer_method", "CHARGED_PROSE", "Metodo, «Norme e analisi per elementi»: il paragrafo della norma contestata "
                                       "dalle righe di sussunzione (v75b)"),
    ("lawyer_method", "NOT_APPLIED", "Risposta: il metodo dell'avvocato non applicato (v76)"),
    ("lawyer_method", "NOT_APPLIED_CLIENT", "Risposta, più clienti: il metodo non applicato alla posizione di uno di "
                                            "loro, gli altri con le sezioni del metodo (v79.10.1)"),
    ("lawyer_method", "NOT_APPLIED_WHY", "Risposta: perché il metodo non è stato applicato (v76)"),
    ("lawyer_method", "REDUCTION_WHY", "Metodo, «Conclusioni»: l'esito probabile della linea di riduzione, sul suo "
                                       "articolo (v79.10.2)"),
    ("issue_records", "NOT_FATTISPECIE", "Registro: una norma di parte generale o processuale (v74, nel log)"),
    ("interactive_flow", "DECOMPOSE_WHY", "Registro: perché una norma ha un registro scomposto (v74, nel log)"),
    ("lawyer_method", "thesis_objections", "Metodo, «Obiezioni e risposte»: dalla tesi scelta"),
    ("lawyer_method", "MAIN_LINE", "Metodo, «Conclusioni»: la linea principale"),
    ("lawyer_method", "SUB_LINE", "Metodo, «Conclusioni»: la linea subordinata"),
    ("lawyer_method", "BARRED_LINE", "Metodo, «Conclusioni»: le tesi non sostenibili"),
    ("lawyer_method", "UNVERIFIABLE_LABEL", "Metodo, «Soluzioni a confronto»: una tesi che non dichiara gli elementi"),
    ("lawyer_method", "INTEGRATED_NOTE", "Metodo, «Soluzioni a confronto»: una tesi minoritaria che contesta un "
                                         "elemento integrato"),
    ("lawyer_method", "orientations_section", "Metodo, «Orientamenti rilevanti»: righe del codice"),
    ("lawyer_method", "settle_objective", "Metodo, «Conclusioni»: una tesi sostenibile alternativa"),
    # v79.12.7 (items 1, 3, 4, 5): the penal lines (src/rag/penal_lines.py; the tables in data/penal_lines.json)
    ("lawyer_method", "JURISPRUDENTIAL_ROW", "Metodo, «Analisi per elementi»: un elemento di fonte giurisprudenziale "
                                             "(v79.12.7, voce 1)"),
    ("penal_lines", "SOURCE_NOTE", "Metodo: la nota dell'avvocato come fonte di un elemento (v79.12.7, voce 1)"),
    ("penal_lines", "SOURCE_SCHEMA", "Metodo: la scheda rivista come fonte di un elemento (v79.12.7, voce 1)"),
    ("penal_lines", "INFERRED_WHY", "Registro: perché un elemento di fonte giurisprudenziale è da provare "
                                    "(v79.12.7, voce 1)"),
    ("penal_lines", "SOURCE_LINE", "Metodo, «Soluzioni a confronto»: la fonte dell'elemento contestato dalla linea "
                                   "principale (v79.12.7, voce 1)"),
    ("penal_lines", "SOURCE_PASSAGE_LINE", "Metodo, «Soluzioni a confronto»: la pronuncia e il suo passaggio sotto la "
                                           "linea principale (v79.12.7, voce 1)"),
    ("penal_lines", "POST_FACT_LINE", "Metodo, «Soluzioni a confronto»: il soccorso narrato come elemento favorevole "
                                      "successivo al fatto (v79.12.7, voce 3)"),
    ("penal_lines", "REQUALIFY_TEXT", "Metodo, «Soluzioni a confronto»: la riqualificazione con l'elemento che "
                                      "differisce e i fatti (v79.12.7, voce 4)"),
    ("penal_lines", "CAUSATION_CONTRO", "Metodo, «Soluzioni a confronto»: il contro della difesa sull'art. 41 su una "
                                        "conseguenza ordinaria (v79.12.7, voce 5)"),
    ("penal_lines", "CAUSATION_REPLY", "Metodo, «Obiezioni e risposte»: la risposta sull'elemento soggettivo, mai "
                                       "sulla negazione del nesso (v79.12.7, voce 5)"),
    # v79.12.7 (items 2, 6): the attenuanti from the facts, the riti computed
    ("penal_lines", "CIRC_FACT_LINE", "Metodo, «Soluzioni a confronto»: un'attenuante dai fatti narrati (v79.12.7, "
                                      "voce 2)"),
    ("penal_lines", "CIRC_STEP_LINE", "Metodo, «Soluzioni a confronto»: l'attenuante del 62 n. 6 legata al passo "
                                      "concreto (v79.12.7, voce 2)"),
    ("penal_lines", "CIRC_POINT", "Metodo, «Soluzioni a confronto» e «Punti da verificare»: il punto legato "
                                  "all'attenuante (v79.12.7, voce 2)"),
    ("penal_lines", "CIRC_TO_VERIFY_LINE", "Metodo, «Soluzioni a confronto»: un'attenuante senza fatti, da verificare "
                                           "(v79.12.7, voce 2)"),
    ("penal_lines", "CIRC_EXCLUDED_LINE", "Metodo, «Soluzioni a confronto»: un'attenuante esclusa da una frase del caso "
                                          "(v79.12.7, voce 2)"),
    ("penal_lines", "POST_FACT_SUPPORT", "Metodo, «Soluzioni a confronto»: il soccorso a sostegno delle generiche "
                                         "(v79.12.7, voce 2)"),
    ("penal_lines", "BILANCIAMENTO_LINE", "Metodo, «Soluzioni a confronto»: il bilanciamento dell'art. 69 c.p. con "
                                          "un'aggravante nel registro (v79.12.7, voce 2)"),
    ("penal_lines", "CIRC_UNEXCLUDED", "Risposta: una circostanza detta esclusa che i fatti non escludono, riscritta "
                                       "(v79.12.7, voce 2)"),
    ("penal_lines", "CIRC_UNEXCLUDED_OPEN", "Risposta: i requisiti aperti della circostanza riscritta (v79.12.7, voce "
                                            "2)"),
    ("penal_lines", "CIRC_NAMES", "Risposta: il tipo di una circostanza riscritta (v79.12.7, voce 2)"),
    ("lawyer_method", "COMP_BASE", "Metodo, i riti: il calcolo, la base (v79.12.7, voce 6)"),
    ("lawyer_method", "COMP_NONE", "Metodo, i riti: il calcolo senza attenuanti (v79.12.7, voce 6)"),
    ("lawyer_method", "COMP_ATTENUANTE", "Metodo, i riti: il calcolo, la diminuzione per un'attenuante (v79.12.7, "
                                         "voce 6)"),
    ("lawyer_method", "COMP_FLOOR", "Metodo, i riti: il calcolo, il limite dell'art. 67 c.p. (v79.12.7, voce 6)"),
    ("lawyer_method", "COMP_RITE", "Metodo, i riti: il calcolo, la diminuzione per il rito (v79.12.7, voce 6)"),
    ("lawyer_method", "COMP_FURTHER", "Metodo, i riti: il calcolo, l'ulteriore sesto dell'abbreviato (v79.12.7, voce "
                                      "6)"),
    ("lawyer_method", "COMP_LIMIT", "Metodo, i riti: il calcolo, la prova dei cinque anni (v79.12.7, voce 6)"),
    ("lawyer_method", "CONSENT_LINE", "Metodo, i riti: il consenso del pubblico ministero (v79.12.7, voce 6)"),
    ("lawyer_method", "CONSENT_UNLIKELY_LINE", "Metodo, i riti: il consenso del pubblico ministero improbabile "
                                              "(v79.12.7, voce 6)"),
    ("lawyer_method", "FRACTION_WORDS", "Metodo, i riti: le frazioni in parole (v79.12.7, voce 6)"),
    # v79.12.7 (item 7): the main line once per section
    ("lawyer_method", "STRATEGY_MAIN_IN_TABLE", "Metodo, «Soluzioni a confronto»: la strategia rinvia alla tesi "
                                                "principale della tabella (v79.12.7, voce 7)"),
    ("lawyer_method", "MAIN_LINE_SEE_ANSWER", "Metodo, «Risposte ai quesiti»: una risposta successiva rinvia alla prima "
                                              "che indica la linea principale (v79.12.7, voce 7)"),
)

# (data file, where): the closed lists' texts kept as data -- the title and each fact's text with its reference
DATA = (
    ("labour_procedure.json", "Risposta, «Profili processuali»: controversia di lavoro (lista chiusa)"),
    # v62: the posture's act and goal (the plan's main line), the procedural section's blocks by stage
    ("posture_acts.json", "Piano, linea principale: l'atto e l'obiettivo per la posizione del cliente"),
    ("stage_procedure.json", "Risposta, «Profili processuali»: i blocchi per fase (lista chiusa)"),
    # v63: the general-part concepts a «cenni» subject rests on, with their articles (doctrine, for the lawyer)
    ("general_concepts.json", "Risposta, «Cenni» e «Norme applicate ai fatti»: le norme di parte generale di un concetto"),
    # v70: the lawyer's method's tables Code drafted -- the reforms, the litisconsorzio, the stragiudiziale options, the
    # circumstances' elements, the riti alternativi (his own texts in the same file -- sections, fasi, rules, modules --
    # are his, verbatim from docs/lawyer/metodo.md: not listed)
    ("lawyer_method.json", "Metodo dell'avvocato (LAWYER_METHOD): le tabelle di Code"),
    # v74: the fact nouns the answer takes from the case; v75: the remedies and eccezioni by matter
    ("fact_terms.json", "Risposta: i nomi dei fatti presi dal caso (classi di atti e oggetti)"),
    ("remedies.json", "Metodo, «Soluzioni a confronto»: rimedi dell'attore ed eccezioni del convenuto per materia"),
    # v79.12.7: the penal lines' printed texts -- the jurisprudential elements, the riqualificazione's elements
    ("penal_lines.json", "Metodo, «Analisi per elementi» e «Soluzioni a confronto»: elementi di fonte "
                         "giurisprudenziale, riqualificazioni e attenuanti dai fatti (v79.12.7)"),
)


def _comments(source: str):
    """({line: full-line comment}, {line: inline comment}) of the source."""
    full, inline = {}, {}
    lines = source.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            n = tok.start[0]
            (full if lines[n - 1].lstrip().startswith("#") else inline)[n] = tok.string.lstrip("#").strip()
    return full, inline


def _above(comments, line):
    """The contiguous block of full-line comments right above a line, top first."""
    block, n = [], line - 1
    while n in comments:
        block.insert(0, comments[n])
        n -= 1
    return " ".join(block)


def _group(comments, starts, line, key):
    """The full-line comment block that heads the run of assignments a line is in (code lines between, no blank line):
    it holds for this key when the run's first key shares this key's first word (ANGLE_STATED_CHOSEN, ANGLE_STATED)."""
    n = line - 1
    first = key
    while n >= 1 and n not in comments and n in starts:
        if starts[n]:
            first = starts[n]
        n -= 1
    if n not in comments:
        return ""
    return _above(comments, n + 1) if first.split("_")[0] == key.split("_")[0] else ""


def _section(comments, line):
    """The copy's section a line falls in («--- round 1 ---» and the like)."""
    best = ""
    for n in sorted(comments):
        if n > line:
            break
        m = re.match(r"-{3}\s*(.+?)\s*-{3,}", comments[n])
        if m:
            best = m.group(1)
    return best


def _strings(node):
    """The string texts a value holds (a string, a tuple / dict of them), each once."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [s for e in node.elts for s in _strings(e)]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in node.values)]
    return []


def _drafted(note: str) -> bool:
    if _DRAFT_RE.search(note):
        return True
    return bool(_VERSION_RE.search(note)) and not _PO_RE.search(note)


_USES = {}


def _uses():
    """{key: [module.function, …]} of every C.<KEY> in src/rag, each file read and parsed once."""
    if _USES:
        return _USES
    for path in sorted(SRC.glob("*.py")):
        if path.name == "interaction_copy.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "C." not in text:
            continue
        spans = sorted(((fn.lineno, fn.end_lineno, fn.name) for fn in ast.parse(text).body
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))), key=lambda x: x[0])
        for n, line in enumerate(text.splitlines(), 1):
            for key in re.findall(r"\bC\.([A-Z][A-Z0-9_]+)\b", line):
                fn = next((name for a, b, name in spans if a <= n <= b), "<module>")
                where = f"{path.stem}.{fn}"
                if where not in _USES.setdefault(key, []):
                    _USES[key].append(where)
    return _USES


def _users(key: str):
    """module.function of every use of C.<key> in src/rag."""
    return list(_uses().get(key, []))


_PLACES = {"round 1": "Primo turno", "the questions: one template per family (a family with no template is not asked)":
           "Primo turno, «Da chiarire»", "round 2": "Secondo turno / dopo una risposta", "the answer": "Risposta (parere)",
           "closing and follow-ups": "Chiusura e seguiti"}


def copy_rows():
    source = COPY.read_text(encoding="utf-8")
    comments, inline = _comments(source)
    tree = ast.parse(source)
    starts = {}                 # {line: the key assigned there, "" on a continuation line} of the module's code lines
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            for n in range(node.lineno, node.end_lineno + 1):
                starts[n] = node.targets[0].id if n == node.lineno else ""
    rows = []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        key = node.targets[0].id
        if not key.isupper():
            continue
        note = " ".join(x for x in [_above(comments, node.lineno) or _group(comments, starts, node.lineno, key)]
                        + [inline.get(n, "") for n in range(node.lineno, node.end_lineno + 1)] if x)
        place = _PLACES.get(_section(comments, node.lineno), _section(comments, node.lineno))
        where = "; ".join([place] + _users(key))
        if isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                label = ast.literal_eval(k) if k is not None else "?"
                own = " ".join(x for x in (_above(comments, k.lineno), inline.get(v.end_lineno, "")) if x)
                if _PO_RE.search(own) or not (_drafted(own) or _drafted(note)):
                    continue
                texts = _strings(v)
                for i, text in enumerate(texts):
                    rows.append((f"{key}[{label!s}]" + (f"#{i + 1}" if len(texts) > 1 else ""), text, where))
            continue
        if _drafted(note):
            texts = _strings(node.value)
            for i, text in enumerate(texts):
                rows.append((key + (f"#{i + 1}" if len(texts) > 1 else ""), text, where))
    return rows


def extra_rows():
    rows = []
    for module, name, where, *keep in EXTRA:
        path = SRC / f"{module}.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node.name
            if target != name:
                continue
            found = []
            root = node.value if isinstance(node, ast.Assign) else node
            inner = {id(v) for sub in ast.walk(root) if isinstance(sub, ast.JoinedStr) for v in sub.values}
            body = getattr(root, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                inner.add(id(body[0].value))            # the docstring
            for sub in ast.walk(root):
                if not isinstance(sub, (ast.Constant, ast.JoinedStr)) or id(sub) in inner:
                    continue
                for text in _strings(sub):
                    if text in found:
                        continue
                    if keep and not re.search(keep[0], text):
                        continue
                    if isinstance(node, ast.Assign) or (len(text) >= 12 and " " in text
                                                        and re.search(r"[a-zà-ù]{2,}[ :,.']+[a-zà-ù]", text)):
                        found.append(text)
            for i, s in enumerate(found):
                rows.append((f"{module}.{name}" + (f"#{i + 1}" if len(found) > 1 else ""), s, where))
    return rows


def data_rows():
    import json
    rows = []
    for name, where in DATA:
        try:
            data = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stem = name.rsplit(".", 1)[0]
        if data.get("title"):
            rows.append((f"data/{stem}.title", data["title"], where))
        for fact in data.get("facts") or []:
            text = " ".join(str(fact.get("text") or "").split())
            if text:
                rows.append((f"data/{stem}[{fact.get('id')}]", text + (f" ({fact['ref']})" if fact.get("ref") else ""),
                             where))
        for entry in data.get("entries") or []:          # v62: posture_acts -- per cell its act and its goal
            cell = f"{'|'.join(entry.get('rito') or [])}/{'|'.join(entry.get('role') or [])}/" \
                   f"{'|'.join(entry.get('stage') or [])}"
            for key in ("act", "goal"):
                if entry.get(key):
                    rows.append((f"data/{stem}[{cell}].{key}", entry[key], where))
            for i, line in enumerate(entry.get("subordinate") or [], 1):     # v70: the imputato's subordinate line
                rows.append((f"data/{stem}[{cell}].subordinate.{i}", line, where))
        for concept in data.get("concepts") or []:       # v63: general_concepts -- per concept its articles
            articles = ", ".join(str(a) for a in concept.get("articles") or [])
            if concept.get("name") and articles:
                rows.append((f"data/{stem}[{concept['name']}]", f"artt. {articles} {concept.get('code') or ''}".strip(),
                             where))
        for matter in data.get("matters") or []:         # v75: remedies -- per matter its remedies and eccezioni
            if not isinstance(matter, dict):
                continue
            for kind in ("remedies", "eccezioni"):
                for r in matter.get(kind) or []:
                    rows.append((f"data/{stem}[{matter['id']}].{kind}", f"{r['label']} ({r['articles']})"
                                 + (f" -- richiede: {r['requires']}" if r.get("requires") else ""), where))
        for cls in data.get("classes") or []:             # v74: fact_terms -- per class its acts, each with its forms
            if not isinstance(cls, dict):
                continue
            for i, member in enumerate(cls.get("members") or [], 1):
                rows.append((f"data/{stem}[{cls['id']}].{i}", ", ".join(f["form"] for f in member), where))
        for el in data.get("jurisprudential_elements") or []:     # v79.12.7: penal_lines -- the printed texts
            if isinstance(el, dict) and el.get("element"):
                rows.append((f"data/{stem}.jurisprudential_elements[{el.get('id')}]",
                             f"{el['element']} ({', '.join(el.get('articles') or [])})", where))
        for r in data.get("requalify_elements") or []:
            if isinstance(r, dict):
                for key in ("charge_element", "alternative_element"):
                    if r.get(key):
                        rows.append((f"data/{stem}.requalify_elements[{r.get('id')}].{key}", r[key], where))
        for t in data.get("circumstance_triggers") or []:     # v79.12.7 (item 2): the attenuanti from the facts
            if not isinstance(t, dict):
                continue
            base = f"data/{stem}.circumstance_triggers[{t.get('id')}]"
            rows.append((base, f"{t.get('name')} ({t.get('article')})", where))
            for key in ("point", "condition", "to_verify"):
                if t.get(key):
                    rows.append((f"{base}.{key}", t[key], where))
            for st in t.get("steps") or []:
                if isinstance(st, dict) and st.get("name"):
                    rows.append((f"{base}.steps", st["name"], where))
        if "jurisprudential_elements" in data:
            continue
        if data.get("accusatory"):
            rows.append((f"data/{stem}.accusatory", ", ".join(data["accusatory"]), where))
        if "matters" in data or "classes" in data:
            continue
        if "sections" in data:                           # v70: lawyer_method -- his method's texts, all of them
            rows += method_rows(data, stem, where)
            continue
        for family, stages in data.items():              # v62: stage_procedure -- per family and stage its blocks
            if family.startswith("_") or not isinstance(stages, dict):
                continue
            for stage, blocks in stages.items():
                for block in blocks if isinstance(blocks, list) else []:
                    roles = f" ({', '.join(block['roles'])})" if block.get("roles") else ""
                    base = f"data/{stem}[{family}/{stage}/{block.get('title')}{roles}]"
                    rows.append((f"{base}.title", block.get("title") or "", where))
                    for i, line in enumerate(block.get("lines") or [], 1):
                        rows.append((f"{base}.{i}", line, where))
                    for key in ("date_flag", "date_line"):
                        if block.get(key):
                            rows.append((f"{base}.{key}", block[key], where))
    return rows


def method_rows(data, stem, where):
    """v70: data/lawyer_method.json -- Code's drafted tables only: the reforms, the litisconsorzio lines, the
    stragiudiziale options, the circumstances and their elements, the riti alternativi (the lawyer's own texts in the
    file are his, not drafts)."""
    rows = [(f"data/{stem}.reforms[{r['id']}]", f"{r['name']}, dal {r['from']}: {r['note']}", where)
             for r in data.get("reforms") or []]
    rows += [(f"data/{stem}.litisconsorzio[{m['matter']}]", m["line"], where) for m in data.get("litisconsorzio") or []]
    rows += [(f"data/{stem}.stragiudiziale[{k}]", v, where) for k, v in (data.get("stragiudiziale") or {}).items()]
    for c in data.get("circumstances") or []:
        rows.append((f"data/{stem}.circumstances[{c['label']}]", f"{c['kind']}: {c['name']}", where))
        rows += [(f"data/{stem}.circumstances[{c['label']}].{i}", e, where) for i, e in enumerate(c.get("elements") or [], 1)]
    rows += [(f"data/{stem}.riti_alternativi[{r['id']}]", r["line"], where) for r in data.get("riti_alternativi") or []]
    for r in data.get("riti_alternativi") or []:           # v79.10.3 (C1): a listed offence's line, the pena unknown
        rows += [(f"data/{stem}.riti_alternativi[{r['id']}].{k}", r[k], where)       # v79.10.7: the possible lines
                 for k in ("listed_line", "to_verify", "possible_line", "possible_pecuniary_line", "provisional",
                           "computed_line")                                  # v79.12.7 (item 6)
                 if r.get(k)]
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "↵ ")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(ROOT / "docs" / "copy_review.md"))
    args = parser.parse_args(argv)
    rows = copy_rows() + extra_rows() + data_rows()
    lines = ["# Copy review: the strings Code drafted for the interactive flow", "",
             "Generated by `scripts/copy_export.py` (re-run after any copy change). One row per user-facing Italian "
             "string Code drafted since the interactive flow began (v55 on): the copy file's own entries "
             "(`src/rag/interaction_copy.py`, edit there), the few written elsewhere (key `module.function`) and the "
             "closed lists kept as data (key `data/<file>`, edit the file). "
             "`{…}` marks a slot the code fills. Which strings are Code's is read from the copy file's comments: an "
             "entry tagged «my draft», or with a version after the PO's v54 texts (v55 on) and no «PO», is listed; an "
             "untagged entry is taken as the PO's v54 text and left out.", "",
             f"{len(rows)} strings.", "",
             "| key | text | where it appears |", "|---|---|---|"]
    lines += [f"| `{k}` | {_cell(t)} | {_cell(w)} |" for k, t, w in rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} strings; {out}")


if __name__ == "__main__":
    sys.exit(main())
