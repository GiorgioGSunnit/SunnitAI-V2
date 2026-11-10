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
# v60: the question the user's message asks (interactive_flow.apply_stated), with a topic matching it (listed first) or none
ANGLE_STATED_CHOSEN = ("Mi chiedi: «{angle}». L'argomento che vi risponde è il primo della lista: se mi chiedi il parere, "
                       "parto da lì.")                                         # angle
ANGLE_STATED = "Mi chiedi: «{angle}». Ne terrò conto nel parere."                                             # angle
TOPIC_LINE = "{n}. **{title}** (artt. {articles}) — {decides}."              # n, title, articles, decides
TOPIC_LINE_SHORT = "{n}. **{title}** (artt. {articles})."                    # the same, when nothing says what it decides
TOPIC_POINT = " Da chiarire: {point}."                                       # point: added when there is one
# v60 (A, B, my draft): the first read's hypotheses under the topic that anchors them; those no topic anchors
HYPOTHESIS_LINE = "   - linea {line}: {claim}"                               # line (principale, subordinata, controparte), claim
HYPOTHESIS_LINE_PARTY = "   - linea {line} per {party}: {claim}"             # v60e (my draft): the client still open
UNANCHORED_TITLE = "**Ipotesi da verificare** (nessuna norma trovata le sostiene ancora)"   # v60 (B, my draft)
UNANCHORED_LINE = "- {claim}"                                                # v60 (B, my draft): claim
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
FORMER_OWNER = {"o": "ex proprietario", "a": "ex proprietaria"}             # v56 (my draft): a party who sold
NO_FACTS = "Nessun fatto estratto."
QUESTIONS_TITLE = "**Da chiarire**"
QUESTION_LINE = "{n}. {text}"                                                # n, text
QUESTION_FROM_READ = "Il caso non dice {point}: puoi precisarlo?"            # v60c (my draft): point, the first read's
QUESTION_REF = "{short} (domanda {n})"                                       # short, n: a question named in a topic line
# v60 (H): what the user can do, in plain words -- never a string to type
INSTRUCTION = ("Puoi dirmi su quale argomento concentrarmi, anche descrivendolo con parole tue, rispondere alle domande che "
               "puoi, aggiungere o correggere un fatto, oppure chiedermi subito il parere: in quel caso parto dal primo "
               "argomento.")

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
    # v59 (my draft): the same question where no topic is criminal -- no querela
    "data_fatto_civile": "Quando è avvenuto il fatto? Serve per il termine di prescrizione.",
    # v55, the PO's text: the act's notification, while a TAR term is open
    "atto_notifica": ("In quale data è stato notificato (o pubblicato) il provvedimento che intendi impugnare? Da questa "
                      "data decorre il termine per il ricorso al TAR."),
    # v57 (my draft): the judgment's date, for the ottemperanza's ten years (art. 114, co. 1, c.p.a.)
    "data_giudicato": ("In quale data la sentenza è passata in giudicato? Da questa data decorrono i dieci anni per "
                       "l'azione di ottemperanza (art. 114, co. 1, c.p.a.)."),
    # v57 (my draft): two parties could be the client and the case does not say which (fact_signals.one_client)
    "parte_assistita": "Chi assisti in questo caso: {names}?",
    # v56 (my drafts, pending PO review): art. 52 co. 2 c.p.'s requisites, for a legittima difesa topic (topic_tables)
    "arma_detenuta": "L'arma (o il mezzo) usata per difendersi era legittimamente detenuta da {client}?",
    "desistenza": "Al momento della reazione l'aggressore aveva desistito (si era fermato o stava fuggendo)?",
    "pericolo_aggressione": "Al momento della reazione c'era ancora un pericolo concreto di aggressione per {client} o per altri?",
    # v56 (my draft): the act eluding a judgment, for an ottemperanza topic -- art. 31 co. 4 c.p.a.'s 180 days
    "data_atto_elusivo": ("In quale data è stato adottato o comunicato il provvedimento che ritieni in contrasto con la "
                          "sentenza? Da questa data decorrono i 180 giorni per farne valere la nullità (art. 31, co. 4, "
                          "c.p.a.)."),
    # v55 (my draft, pending PO review): the cautionary order's execution, for a misure cautelari topic (topic_tables)
    "data_misura": ("In quale data è stata eseguita o notificata l'ordinanza cautelare? Da questa data decorrono i dieci "
                    "giorni per la richiesta di riesame (art. 309 c.p.p.)."),
}
CLIENT_JOIN = " o "                                                          # v57: the possible clients
LIST_COMMA = ", "
DEFENDER = "chi si è difeso"                                                 # the client slot when no name is known
# how a topic's «da chiarire» point names an open question
QUESTION_SHORT = {
    "possesso_inizio": "inizio del possesso",
    "opera_separazione": "opere visibili alla separazione dei fondi",
    "atto_separazione": "atto e anno della separazione dei fondi",
    "accesso_alternativo": "altro accesso alla via pubblica",
    "data_messa_in_mora": "data della messa in mora",
    "data_fatto": "data del fatto",
    "atto_notifica": "data di notifica del provvedimento",
    "parte_assistita": "parte assistita",
    "data_giudicato": "data del giudicato",
    "data_misura": "data dell'ordinanza cautelare",
    "arma_detenuta": "arma legittimamente detenuta",
    "desistenza": "desistenza dell'aggressore",
    "pericolo_aggressione": "pericolo di aggressione",
    "data_atto_elusivo": "data del provvedimento elusivo",
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
CHANGE_TERM = "{term}: ora {after} (prima {before})"                          # v60e: no arrows; term, before, after
CHANGE_PROCEDURAL = "profili processuali ricalcolati"
CHANGE_PARTY_ADDED = "parte aggiunta: {name}"                                # name
CHANGE_PARTY_ROLE = "{name}: ora {after} (prima {before})"                       # name, before, after
CHANGE_DATE = "data di «{what}»: ora {after} (prima {before})"                         # what, before, after
CHANGE_EVENT_ADDED = "evento aggiunto: {date} — {what}"                      # date, what
CHANGE_CLIENT = "parte assistita: {name}"                                    # v60d (my draft): name
ADDITION_ANSWER = "{question} {answer}"                                      # question, answer: a precisazione
# v60e (my draft): after a reply that answers questions, the restatement -- two sentences (the client; what was answered,
# in the answers' words), then what comes next; never the questions' text, arrows or repeats
RESTATE_CLIENT = "Imposto il parere per {client}."                             # client
RESTATE_NO_CLIENT = "Il parere esaminerà la posizione delle parti, senza assumerne una."
RESTATE_ANSWERS = "Tengo conto di quanto mi hai precisato: {items}."            # items
RESTATE_ITEM = "{point}: {answer}"                                             # point (the question's short label), answer
# v60h (8, my draft): a yes/no answer restated as the fact it settles -- (sì, no) -- by question family; a first read's
# question restates its own point as a statement (interactive_flow.stated_fact)
RESTATE_YES_NO = {
    "opera_separazione": ("alla separazione dei fondi esistevano già opere visibili e permanenti destinate al passaggio",
                          "alla separazione dei fondi non esistevano opere visibili e permanenti destinate al passaggio"),
    "accesso_alternativo": ("il fondo ha un altro accesso alla via pubblica",
                            "il fondo non ha altro accesso alla via pubblica"),
    "arma_detenuta": ("l'arma era legittimamente detenuta", "l'arma non era legittimamente detenuta"),
    "desistenza": ("al momento della reazione l'aggressore aveva desistito",
                   "al momento della reazione l'aggressore non aveva desistito"),
    "pericolo_aggressione": ("al momento della reazione c'era ancora un pericolo concreto di aggressione",
                             "al momento della reazione non c'era più un pericolo concreto di aggressione"),
}
RESTATE_NEXT_TOPICS = "Tratterò: {topics}. Va bene così, o vuoi correggere qualcosa?"     # topics
RESTATE_NEXT_ANSWERED = "Vuoi che riscriva la risposta su «{titles}» tenendone conto?"     # titles

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
# v58 (the PO's words): the planned articles §2 does not treat -- no text read, or nothing to say on the facts
OTHER_NORMS_LINE = "Altre norme del tema non trattate: {articles}."              # articles
OTHER_NORMS_ITEM = "art. {number} {code}"                                      # number, code
OTHER_NORMS_JOIN = ", "
# v60d (my draft): a §2 block whose gloss says the article does not apply, one item in one line
NOT_APPLICABLE_LINE = "Norme considerate e non applicabili: {articles}."         # articles
NOT_APPLICABLE_ITEM = "art. {number} {code}"                                     # number, code
# v60h (6, my draft): a sanction article, one line under the obligation article it sanctions (never a block of its own)
SANCTION_LINE_RUBRICA = "- Sanzione: art. {number} {code} — {rubrica}."           # number, code, rubrica
SANCTION_LINE = "- Sanzione: art. {number} {code}."                              # v60h (6, my draft): number, code
MEDIATION = ("Mediazione obbligatoria (art. 5 d.lgs. 28/2010): il suo mancato esperimento comporta l'improcedibilità "
             "della domanda.")
# v60 (C): §4 «Procedibilità e competenza», one line per offence from the article's own text (procedural_facts) -- the
# generic «procedibile d'ufficio; competenza: tribunale» is never written; a part with no fact is left out
PROCEDURE_HEAD = "{name} (art. {number} c.p.)"                               # name (the rubrica), number
PROCEDURE_PROCEDIBILITA = "procedibile {text}"                               # text
PROCEDURE_QUERELA = "querela entro {term}, {remissione}"                     # term, remissione
PROCEDURE_COMPETENZA = "competenza: {text}"                                  # text
PROCEDURE_PENA = "pena: {text}"                                              # text
PROCEDURE_JOIN = "; "
# v60h (1, my draft): a querela offence's d'ufficio conditions, each with its record status, and the line's verdict
PROCEDURE_CONDITIONAL = "procedibile a querela della persona offesa ({comma}); d'ufficio {conditions}"   # comma, conditions
PROCEDURE_CONDITION = "{condition} ({where})"                    # v60h (1, my draft): condition, where (comma: status)
PROCEDURE_CONDITION_JOIN = "; oppure "                                      # v60h (1, my draft)
PROCEDURE_VERDICT = " — esito: {verdict}"                                    # v60h (1, my draft): verdict
PROCEDURE_VERDICTS = {"provato": "procedibile d'ufficio (condizione provata)",           # v60h (1, my draft)
                      "non risulta": "procedibile a querela (nessuna condizione d'ufficio risulta)",
                      "da provare": "a querela, d'ufficio se una condizione è provata"}
PROCEDURE_UNKNOWN = "procedibilità e competenza da verificare (non lette dal testo dell'articolo)"
PROCEDURE_FIXED = "Quanto ai profili processuali: {text}."                   # text: a writer sentence contradicting them
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
# v56 (my draft, pending PO review): a round-1 question left unanswered, for the writer and in the answer
UNKNOWN_TITLE = "FATTI NON NOTI (domande del primo turno senza risposta)"
UNKNOWN_LINE = "- {point}: non noto ({question})"                            # point, question
UNKNOWN_RULE = "Non affermarli e non dedurli: dove servono, scrivi che vanno verificati."
UNKNOWN_MARK = " (da verificare: {point})"                                  # point: the questions' short labels

# v60 (D, my draft): the client neither stated nor answered when the answer is asked for
CLIENT_ASSUMED = ("Non mi hai indicato chi assisti: imposto il parere per {name} ({role}), la parte che il caso indica "
                  "come destinataria dell'azione. Se assisti un'altra parte, dimmelo e lo riscrivo.")   # name, role
CLIENT_UNKNOWN = ("Non mi hai indicato chi assisti e il caso non lo fa capire: il parere esamina la posizione delle parti "
                  "senza assumerne una. Dimmi chi assisti e lo riscrivo dal suo punto di vista.")

# --- closing and follow-ups -------------------------------------------------------------------------------------------
CLOSING = ("Se vuoi, posso approfondire anche: {topics}. Puoi anche darmi un fatto nuovo, e aggiorno il parere, o "
           "chiedermi il documento.")                                        # topics
CLOSING_TOPIC = "{n}. {title}"                                               # n, title
NEW_FACT = "Ho aggiornato i fatti:\n{changes}\n\nVuoi che riscriva la risposta su «{titles}» tenendone conto?"  # changes, titles
NEW_FACT_NONE = "precisazione acquisita"
UNCLEAR = "Non sono sicuro di aver capito. {next}"                         # next: the CLARIFY_* question
NO_TOPIC_TO_ANSWER = "Non trovo l'argomento da trattare: dimmi quale degli argomenti ti interessa."
# v60 (H, my draft): the one clarifying question, by round; the answer to a question on how the exchange works
CLARIFY_TOPICS = "Vuoi che mi concentri su uno degli argomenti elencati, o che scriva subito il parere partendo dal primo?"
CLARIFY_CONFIRM = "Vuoi che scriva il parere così impostato, o c'è qualcosa da correggere?"
CLARIFY_ANSWERED = "Vuoi approfondire un altro argomento, aggiungere un fatto o avere il documento?"
FLOW_HELP = (                                                                # v60 (H, my draft)
    "Funziona così: ti propongo gli argomenti del caso e le domande sui fatti che mancano. Puoi scegliere uno o "
    "più argomenti, anche descrivendoli, rispondere alle domande, aggiungere o correggere fatti, chiedermi un "
    "profilo che non ho elencato, oppure chiedermi il parere; dopo il parere puoi approfondire un altro "
    "argomento o avere il documento.")
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
