"""INTERACTIVE_FLOW: every text the interactive rounds show, in one place, for the PO to edit without touching the logic.

Rules for editing:
  - each value is a Python string; {slot} marks are filled from the computed facts (the slots each template takes are
    listed beside it) -- keep them;
  - a sentence of a template whose slot cannot be filled is left out (fill_parts), the rest is shown as written; a
    single-line template whose slot cannot be filled is not shown at all (fill);
  - nothing here is rephrased by a model: what is written here is what the user reads;
  - a question family (QUESTIONS) with no template is never asked; an objection (OBJECTIONS, OBJECTIONS_BY_QUESTION)
    or a condition (CONDITIONS) with no template is never written. Deleting an entry switches it off.
v54: the PO's texts (8 Nov), verbatim; the slot names are the code's.
"""

import re
import string
from typing import Any, Optional

# --- round 1 ----------------------------------------------------------------------------------------------------------
ROUND1_OPENING = ("Ho letto il caso di {client}. Prima di scrivere il parere, vorrei concordare con te su cosa concentrarlo "
                  "e chiarire alcuni punti che il testo non dice.")            # client
TOPICS_TITLE = "**Argomenti individuati**"
TOPIC_LINE = "{n}. **{title}** (artt. {articles}) — {decides}."              # n, title, articles, decides
TOPIC_LINE_SHORT = "{n}. **{title}** (artt. {articles})."                    # the same, when nothing says what it decides
TOPIC_POINT = " Da chiarire: {point}."                                       # point: added when there is one
# v55 (my draft, pending PO review): what a topic of a rule to verify decides -- never written as a fixed conclusion
DECIDES_TO_VERIFY = "da verificare: {decides}"                               # decides
NO_TOPICS = "Nessun argomento individuato con sicurezza: descrivi l'angolo che ti interessa."
OTHER_PROFILES_TITLE = "**Altri profili** (non trattati, salvo che tu lo chieda)"
OTHER_PROFILE_LINE = "- {capo} (artt. {articles})"                          # capo (the anchors' Capo), articles
OTHER_PROFILE_LINE_TITLE = "- {title} (artt. {articles})"                   # title, articles: no Capo read
FACTS_INTRO = "Ecco i fatti e i termini che ho ricavato dal caso:"
# the events, as sentences in chronological order, joined by "; " and closed by "."
EVENT_IN = "nel {date} {what}"                                               # date (a year, a month), what
EVENT_ON = "il {date} {what}"                                                # date (a day), what
EVENT_START = "all'inizio del {year} {what}"                                 # year, what: an event dated «inizio del»
# the computed terms, one sentence each: by term (usucapione: every usucapione term) and status; a term with no entry
# keeps its computed line (TERM_LINE)
TERM_TEXTS = {
    ("possessoria", "DECORSO"): "L'azione possessoria è ormai preclusa (termine annuale dal {year}).",          # year
    ("non_uso_servitu", "NON DECORSO"): "L'estinzione per non uso non è maturata (scadrebbe nel {year}).",      # year
    ("usucapione", "DA VERIFICARE"): "L'usucapione va verificata: il caso non dice da quando dura il possesso.",
    ("usucapione", "INCERTO"): "L'usucapione va verificata: il caso non dice da quando dura il possesso.",
    ("usucapione", "NON VERIFICABILE"): "L'usucapione va verificata: il caso non dice da quando dura il possesso.",
}
TERM_LINE = "{text}"                                                         # text: a term with no TERM_TEXTS entry
PARTIES_LINE = "Parti: {parties}."                                           # parties
PARTY = "{name} ({role})"                                                    # name, role
PARTY_NO_ROLE = "{name}"                                                     # name
PARTY_CLIENT = "{party} — cliente"                                           # party
NO_FACTS = "Nessun fatto estratto."
QUESTIONS_TITLE = "**Da chiarire**"
QUESTION_LINE = "{n}. {text}"                                                # n, text
QUESTION_REF = "{short} (domanda {n})"                                       # short, n: a question named in a topic line
INSTRUCTION = ("Dimmi su quale argomento vuoi che mi concentri (il numero, oppure descrivimi l'angolo che ti interessa), "
               "rispondi alle domande che puoi, oppure scrivi \"procedi\" e parto dalla tesi principale.")

# --- the questions: one template per family (a family with no template is not asked) -------------------------------
QUESTIONS = {
    # possession.since of the client's side; who (the client), what (the way, with its article), land (with its article)
    "possesso_inizio": ("Per valutare l'usucapione: da quando {who} o i suoi danti causa usano {what} per raggiungere "
                        "{land}? Il caso non indica l'anno in cui è iniziato."),
    # works_or_signs.visible / before_separation; work (the works, as the facts name them)
    "opera_separazione": ("Per la destinazione del padre di famiglia: quando i fondi furono separati, esistevano già opere "
                          "visibili e permanenti ({work}) destinate al passaggio?"),
    # property_history.ended_by and its year; owner (the original owner)
    "atto_separazione": ("Con quale atto e in quale anno {owner} ha separato i fondi? Da questo dipende anche se il passaggio "
                         "coattivo sarebbe dovuto senza indennità (art. 1054 c.c.)."),
    # access.landlocked of the client's land (a «sì» is an access: not landlocked); client
    "accesso_alternativo": "Il fondo di {client} ha un altro accesso alla via pubblica, anche se scomodo o insufficiente?",
    # the date of the notice a contract term runs from or is interrupted by; counterparty
    "data_messa_in_mora": "In quale data {counterparty} è stata messa in mora o ha ricevuto la contestazione?",
    # the date of the fact a prescription or querela term runs from
    "data_fatto": "Quando è avvenuto il fatto? Serve per i termini di prescrizione e di querela.",
    # v55, the PO's text: the act's notification, while a TAR term is open
    "atto_notifica": ("In quale data è stato notificato (o pubblicato) il provvedimento che intendi impugnare? Da questa "
                      "data decorre il termine per il ricorso al TAR."),
    # v55 (my draft, pending PO review): the cautionary order's execution, for a misure cautelari topic (topic_tables)
    "data_misura": ("In quale data è stata eseguita o notificata l'ordinanza cautelare? Da questa data decorrono i dieci "
                    "giorni per la richiesta di riesame (art. 309 c.p.p.)."),
}
# how a topic's «da chiarire» point names an open question
QUESTION_SHORT = {
    "possesso_inizio": "inizio del possesso",
    "opera_separazione": "opere visibili alla separazione dei fondi",
    "atto_separazione": "atto e anno della separazione dei fondi",
    "accesso_alternativo": "altro accesso alla via pubblica",
    "data_messa_in_mora": "data della messa in mora",
    "data_fatto": "data del fatto",
    "atto_notifica": "data di notifica del provvedimento",
    "data_misura": "data dell'ordinanza cautelare",
}

# --- round 2 ----------------------------------------------------------------------------------------------------------
ROUND2 = ("Riassumo come imposterò il parere. Tratterò: {topics}. Fatti su cui mi baso: {facts}. Resta da chiarire: "
          "{points}; dove non abbiamo la risposta, lo segnalerò come punto da provare. Parti da convenire: {parties}. "
          "Va bene così, o vuoi correggere qualcosa?")                       # topics, facts, points, parties
CONFIRM_LINE = "Va bene così, o vuoi correggere qualcosa?"
LIST_JOIN = "; "

# the changes a reply makes, as the round-2 facts list them
CHANGE_LINE = "{text}"                                                       # text
CHANGE_RULE_ON = "{rule}: ora applicabile"                                   # rule (its institution)
CHANGE_RULE_OFF = "{rule}: non più applicabile"                              # rule
CHANGE_RULE_TEXT = "{rule}: conclusione aggiornata"                          # rule
CHANGE_TERM = "{term}: {before} → {after}"                                   # term (its label), before, after
CHANGE_PROCEDURAL = "profili processuali ricalcolati"
CHANGE_PARTY_ADDED = "parte aggiunta: {name}"                                # name
CHANGE_PARTY_ROLE = "{name}: ruolo {before} → {after}"                       # name, before, after
CHANGE_DATE = "data di «{what}»: {before} → {after}"                         # what, before, after
CHANGE_EVENT_ADDED = "evento aggiunto: {date} — {what}"                      # date, what
ADDITION_ANSWER = "{question} {answer}"                                      # question, answer: a precisazione

# --- the answer -------------------------------------------------------------------------------------------------------
# the requisites' statuses, from the fact fields (never the writer's)
STATUS_PROVATO = "provato"
STATUS_DA_PROVARE = "da provare"
STATUS_NON_RISULTA = "non risulta"
# §2 «Norme applicate ai fatti»: one heading per article, its commi's status lines, then the writer's application
NORM_HEADING_RUBRICA = "**Art. {number} {code} — {rubrica}**"                # number, code, rubrica
NORM_HEADING = "**Art. {number} {code}**"                                    # number, code
COMMA_LABEL = "co. {k}"                                                      # k
COMMA_LABEL_ONLY = "testo"
REQUISITE_LINE = "- {comma}, requisito «{requisite}»: {status}."             # comma, requisite, status
COMMA_LINE = "- {comma}: {status}."                                          # comma, status: a comma with no requisite
# the procedural section's mediation paragraph
MEDIATION = ("Mediazione obbligatoria (art. 5 d.lgs. 28/2010): il suo mancato esperimento comporta l'improcedibilità "
             "della domanda.")
SECTION_TITLES = ("Risposta e strategia", "Norme applicate ai fatti", "Obiezioni e repliche", "Profili processuali",
                  "Conclusioni")
OTHER_PROFILES_SECTION = "Altri profili"
OTHER_PROFILES_BODY = ("Per le norme seguenti i fatti accertati non provano ancora alcun requisito ({articles}): restano "
                       "profili da sviluppare se le prove li confermeranno.")   # articles
OTHER_PROFILES_ITEM = "- {sentence}"                                         # sentence
TERM_FIXED = "Quanto al termine: {text}."                                    # text: a computed term's own line

# the objections' titles (each paragraph of «Obiezioni e repliche» opens with one, in bold)
OBJECTIONS = {
    "tolleranza": "La controparte potrà obiettare che il passaggio avveniva per mera tolleranza.",
    "ritardo": "La controparte potrà obiettare che il ritardo nell'agire dimostra l'abbandono del diritto.",
    "cortili_giardini": ("La controparte potrà obiettare che il passaggio inciderebbe su cortili o giardini esenti "
                         "(art. 1051 c.c.)."),
}
# the objection an open question raises, by its family
OBJECTIONS_BY_QUESTION = {
    "possesso_inizio": "La controparte potrà obiettare che non è provato quando è iniziato il possesso.",
    "opera_separazione": "La controparte potrà obiettare che le opere non erano visibili al momento della separazione.",
}
OBJECTION_TITLE = "**{n}) {text}**"                                          # n, text

# a requisite «da provare»: the words that open every sentence relying on it, by family
CONDITIONS = {
    "atto_separazione": "Se l'interclusione deriva dall'atto di separazione, ",
    "opera_separazione": "Se le opere risultano esistenti alla separazione, ",
    "possesso_inizio": "Se il possesso ventennale risulta compiuto, ",
}
CONDITIONS_JOIN = " e "

# --- closing and follow-ups -------------------------------------------------------------------------------------------
CLOSING = "Se vuoi, posso approfondire: {topics}. Oppure dimmi un fatto nuovo e aggiorno il parere."    # topics
CLOSING_TOPIC = "{n}. {title}"                                               # n, title
NEW_FACT = "Ho aggiornato i fatti:\n{changes}\n\nVuoi che rigeneri la risposta su «{titles}»? (sì / no)"  # changes, titles
NEW_FACT_NONE = "precisazione acquisita"
UNCLEAR = "Non ho capito la risposta. {next}"                                # next: INSTRUCTION or CONFIRM_LINE
NO_TOPIC_TO_ANSWER = "Nessun argomento da trattare: indica il numero di uno degli argomenti."
ANSWER_FAILED = "Non sono riuscito a scrivere la risposta su questo argomento. Riprova, oppure scegli un altro argomento."


def _slots(template: str) -> list:
    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def _given(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def fill(template: Optional[str], **slots: Any) -> Optional[str]:
    """The template with its slots filled, or None: no template, or a slot it names left empty or not given."""
    if not template:
        return None
    try:
        names = _slots(template)
        if not all(_given(slots.get(n)) for n in names):
            return None
        return template.format(**{n: slots[n] for n in names})
    except (KeyError, IndexError, ValueError):
        return None


def fill_parts(template: Optional[str], **slots: Any) -> Optional[str]:
    """The template sentence by sentence: a sentence whose slot cannot be filled is left out, the others as written; None
    when nothing is left."""
    if not template:
        return None
    sentences = re.split(r"(?<=[.?!])\s+(?=[A-ZÀ-Ü\"«])", template)
    kept = [k for k in (fill(s, **slots) for s in sentences) if k]
    return " ".join(kept) or None


def capital(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
