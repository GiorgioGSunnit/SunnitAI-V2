"""v60g (d): the interactive answer's writer rules -- one de-conflicted set, the record as the writer's source.

The writer reads the per-article record (issue_records: each requisite's status, the case's sentences for and against
it, what is missing) instead of re-deriving it, under these rules. The audit's conflicts (artefact C of the v60f
investigation) and how each is resolved:

  P1  The writer rules forbade calling a norm «non applicabile» while v60d turned such glosses into «Norme considerate e
      non applicabili». Resolved: «non applicabile» is the record's verdict (issue_records.verdict: a requisite of
      every comma «non risulta», v60h: or the distinctive element «non risulta»), never a reading of the gloss. v60h
      (5): such a norm is not in the SCHEDE -- the code lists it in «Norme considerate e non applicabili», one line,
      never a block --; of the norms in the SCHEDE the writer never says it (the check removes it: record_check).
  P2  «Cita una norma solo dove sostiene un argomento» against «per ciascuna norma … un paragrafo». Resolved by section:
      in «Norme applicate ai fatti» every norm of the SCHEDE has its paragraph (applicable, to prove or not applicable);
      in the other sections a norm is cited only where it supports an argument.
  P3  The status: «riporta quello stato» (rules text) against «non ripeterlo e non contraddirlo» (§2), «non risulta»
      never explained. Resolved: the status lines are the code's, from the record, above each paragraph; the paragraph
      neither restates nor contradicts them and gives, per requisite, the fact that proves it (provato), the proof to
      acquire (da provare) or the fact that excludes it (non risulta) -- the three statuses defined below.
  P4  «Tratta solo l'argomento indicato» against a plan made without the topic and case-wide material. Resolved: the
      plan's issues are the chosen topics (v60f c); the case-wide blocks (conclusions, terms, the procedural section)
      are context, used only where they bear on the chosen topics, never a new issue.
  P5  Not an item: the audit's numbering skipped it (P1-P4, P6, P9-P11).
  P6  «Non affermare fatti non noti» against «diritti già acquisiti … prima il loro riconoscimento» and «parti da
      convenire, mai eventuali», neither filtered by the unknowns. Resolved: a right is already acquired only when a
      fact rule the case states fired (plan_then_write.acquired_rights(stated_only)); a party to sue is named when the
      case names it, else it is «da individuare»; a fact is asserted only from a sentence of the case (as the case
      gives it: a fact, an allegation, a hypothesis), else it is what is missing.
  P9  (v60f) one list of sections: the writer is told the answer's five.
  v60i (1, 2): the SCHEDE hold only the norms a §2 block is due to (issue_records.block_due: «applicabile», «argomento
      del cliente», «da provare» with a fact of the case for it or, v60j, with a missing fact an open round-1 question
      asks); the others are the code's lines, each with its
      reason. «argomento del cliente» -- the counterparty's norm whose elements fail -- is written as the client's
      argument against it, the one norm of the SCHEDE whose paragraph says it does not apply.
  v60q: the statuses are the code's -- the writer never uses their words as ordinary words (STATUS_WORDS_RULE; the
      status_words step rewrites the ones it still writes); a party's state of mind is argued, never missing
      (STATE_OF_MIND_RULE); single shot, an article is a reference, never its quoted text (REFERENCES_RULE, *_REFERENCES).

The set replaces plan_then_write._WRITER_RULES' first rule (P1/P2) for the interactive answer; its other rules (the
burden of proof, no conduct the case does not state, rulings, terms, the client's side, the institution rules) stand.
"""

from typing import Any, Dict

STATUS_DEFINITIONS = (
    "«provato»: una frase del caso lo dice (riprendila con le parole del caso); «da provare»: nessuna frase del caso "
    "lo dice: indica la prova da acquisire (MANCA); "
    "«non risulta»: il caso dice il contrario (il fatto CONTRO): per quel requisito la norma non si applica; «non "
    "pertinente»: il caso dice il contrario per un comma che non è quello su cui la norma si regge -- quel comma non "
    "riguarda il caso, non esclude e non prova nulla.")

# v60q (my drafts): the statuses are the code's -- the writer states facts and arguments; a party's state of mind is
# argued from the facts, never a missing fact nor a question
STATUS_WORDS_RULE = (
    "- Gli stati («provato», «da provare», «non risulta», «da verificare») li scrive il codice nelle righe di stato: nel "
    "tuo testo non usarli mai come parole comuni (mai «non è da provare che…», «è provato che…», «non risulta che…»); "
    "scrivi i fatti e gli argomenti. Sola eccezione: la formula «da verificare: il caso non dice …», dove è richiesta.\n")
STATE_OF_MIND_RULE = (
    "- Lo stato soggettivo di una parte (se ha agito deliberatamente, intenzionalmente, consapevolmente, in mala fede, "
    "con dolo) non è mai un fatto mancante né una domanda: argomentalo dai fatti del caso (condotte, tempi, "
    "comunicazioni).\n")

# v60s (4, my draft): §1 and the conclusions assert only the norms the record allows -- «applicabile», or «da provare»
# with a proven element (v60t: and the main line's anchor, with its own status)
ASSERTABLE_RULE = (" Fonda la sezione solo su queste norme: {articles}; ogni altra norma nominala solo "
                   "come «da valutare».")
NONE_ASSERTABLE_RULE = (" Nessuna norma è ancora applicabile sui fatti, né ha un requisito provato: nominale solo come "
                        "«da valutare».")
# v68 (my draft): the client's own articles (weapons) not yet assertable -- argued with what remains to prove, never
# with the «da valutare» formula
WEAPON_TO_PROVE_RULE = (" Le norme {articles} sono del cliente: presentale come domande o eccezioni da sostenere, "
                        "dicendo quale requisito resta da provare, senza la formula «da valutare».")

# v68 (my draft): an article the user named whose block the writer left empty -- a second call for it alone
NAMED_RETRY_NOTE = (" Tratta solo l'{article}, che l'utente ha nominato: un solo paragrafo, aperto da «[1]», su che cosa "
                    "dispone la norma e come si applica ai fatti del caso.")

# v68 (my draft): five or more requested blocks -- the lead is a summary, the questions have their own sections
COLLAPSED_LEAD_NOTE = (" I quesiti seguenti hanno ciascuno una sezione propria, scritta a parte: {questions}. Qui solo la "
                       "sintesi della risposta e la strategia, in uno-due paragrafi, senza ripetere le risposte ai "
                       "quesiti.")

# v62 (my draft): an article whose requisites are not counted (c.p. Libro I, procedural, general, definitional) -- its
# §2 paragraph is an application, no requisite proven or to prove
APPLICATION_MARK = "applicazione: norma generale o processuale, nessun requisito da contare"
APPLICATION_RULE = ("Per le norme segnate «applicazione»: un paragrafo che dice che cosa dispone la norma e come opera nel "
                    "caso; non elencare requisiti provati o da provare e non scrivere un esito.")
# v65 (my draft): for a defending client (imputato, convenuto) an application paragraph reads from the defence too
APPLICATION_MARK_DEFENCE = ("applicazione, dalla parte della difesa: che cosa sostiene la difesa e che cosa resta da "
                            "verificare")
APPLICATION_RULE_DEFENCE = ("Per le norme segnate «applicazione, dalla parte della difesa»: un paragrafo che dice che cosa "
                            "dispone la norma, che cosa ne ricava la difesa sui fatti del caso e che cosa resta da "
                            "verificare; mai una conclusione contro la parte assistita (che la norma la condanna, che "
                            "l'aggravante o l'elemento sussiste), né requisiti provati o un esito.")
# v62 (my draft): the charge or the claim against the client, read from the defence
THREAT_MARK = {"penale": "norma dell'accusa, da contestare: gli elementi letti dalla difesa",
               "civile": "norma della domanda avversaria, da contestare: gli elementi letti dalla difesa"}
THREAT_RULE = {
    "penale": ("Per le norme segnate «norma dell'accusa»: scrivi dalla parte della difesa. Per ogni elemento da "
               "contestare indica il punto contestabile sui fatti; per ogni elemento non provato rileva che l'accusa "
               "non lo prova; non scrivere che la norma è applicabile né quanti requisiti sono provati. Contesta solo "
               "gli elementi elencati sotto la norma: nessun altro elemento, requisito o atteggiamento psicologico."),
    "civile": ("Per le norme segnate «norma della domanda avversaria»: scrivi dalla parte del convenuto. Per ogni "
               "elemento da contestare indica il punto contestabile sui fatti; per ogni elemento non provato rileva che "
               "la controparte non lo prova; non scrivere che la norma è applicabile né quanti requisiti sono provati. "
               "Contesta solo gli elementi elencati sotto la norma: nessun altro elemento, requisito o atteggiamento "
               "psicologico."),
}
# v62 (my draft): the requested blocks -- a «cenni» section on an institution, the alternative qualifications
CENNI_INSTRUCTION = ("SEZIONE: Cenni su «{subject}». Un paragrafo breve, di quattro-sei frasi, sull'istituto: che cos'è e "
                     "quali sono i suoi elementi.")
CENNI_GROUNDED = (" Fondalo sul testo e sulla rubrica delle NORME fornite, citate come riferimento («art. N codice — "
                  "rubrica») senza riportarne il testo tra virgolette; nessun altro articolo. Non applicarle ai fatti "
                  "del caso, non scrivere requisiti provati o da provare, non citare pronunce.")
CENNI_UNGROUNDED = (" Nessuna norma è fornita per questo istituto: non citare articoli né pronunce, non applicarlo ai "
                    "fatti del caso.")
ALTERNATIVES_INSTRUCTION = ("SEZIONE: Qualificazioni alternative. Per ciascuna qualificazione di QUALIFICAZIONI "
                            "ALTERNATIVE, nell'ordine, una riga nuova aperta dal suo numero tra parentesi quadre e "
                            "dall'esito -- «SÌ» se si adatta ai fatti del caso, «NO» se non si adatta -- poi una sola "
                            "frase sul perché («[1] SÌ: ...», «[2] NO: ...»). Non ripetere il numero né la rubrica "
                            "dell'articolo (li scrive il codice), non aggiungere altre qualificazioni, non citare "
                            "pronunce.")

# v65 (my draft): a listed question of the user, its own section -- answered from the record and the store only
QUESITO_INSTRUCTION = ("SEZIONE: {question} Rispondi a questa domanda, in uno-tre paragrafi, solo con il materiale "
                       "fornito: le SCHEDE DELLE NORME (lo stato di ogni requisito è quello delle schede) e i fatti "
                       "processuali che il codice ha calcolato. Se il materiale non basta, dillo e indica che cosa va "
                       "verificato. Cita le norme come riferimento («art. N codice»), senza riportarne il testo; non "
                       "citare pronunce; non ripetere la domanda.")

# v79.16.0 (C2): the re-ask of one question whose reviewed answer kept fewer than two sentences -- the verified
# affermazioni di diritto and the question's own articles are the only law it may state
REASK_RULE = (" Usa SOLO le AFFERMAZIONI DI DIRITTO VERIFICATE e gli ARTICOLI DEL QUESITO riportati sotto: nessun'altra "
              "regola di diritto. Se non bastano, scrivi che il materiale non consente di rispondere.")

# v62 (my draft): §1 and the conclusions -- the charge or the claim is what the client contests, never asserted
THREAT_ASSERT_RULE = (" Le norme {articles} sono l'addebito o la domanda contro {who}: presentale come da contestare, "
                      "mai come applicabili.")
# v63 (my draft): §1 and the conclusions contest only the threat's own elements, by name -- never one the norm lacks
THREAT_ELEMENTS_RULE = (" Di queste norme contesta solo gli elementi del registro, chiamandoli per nome: {items}. Nessun "
                        "altro elemento, requisito o atteggiamento psicologico: ciò che non è in questo elenco la norma "
                        "non lo richiede.")
THREAT_ELEMENTS_ITEM = "art. {article}: {elements}"
# v65 (my draft): an alternative the first read finds no fact for stays out of §1
UNFOUNDED_ALTERNATIVES_RULE = (" Non proporre in questa sezione le qualificazioni alternative che nessun fatto del caso "
                               "sostiene ({articles}): sono trattate nella sezione delle qualificazioni alternative.")
THREAT_ELEMENTS_NONE = "art. {article}: nessun elemento nel registro, non contestarne altri"

# v60r (6, my draft): an element no fact of the case touches is the code's «da provare» -- never narrated as missing
NO_SIGNAL_RULE = (
    "- Un requisito segnato «nessun fatto del caso a favore o contro» resta da provare: non scrivere che manca né che il "
    "caso non lo dice; argomenta sui fatti che ci sono.\n")

RECORD_RULES = (
    "- Le SCHEDE DELLE NORME sono il tuo materiale: per ogni norma lo stato di ciascun requisito, i fatti del caso a "
    "favore e contro, che cosa manca. Non ricavarli di nuovo, non cambiarli. Stati: " + STATUS_DEFINITIONS + "\n"
    "- Nelle SCHEDE ci sono solo le norme con esito «applicabile», «argomento del cliente» o «da provare» con almeno un "
    "fatto del caso a sostegno o con un fatto mancante che una domanda aperta chiede (indica quale fatto manca): le "
    "altre le elenca il codice in una riga, con il motivo («Norme considerate e non "
    "applicabili», «Altre norme del tema non trattate»); non commentarle. Di una norma delle SCHEDE con esito "
    "«applicabile» o «da provare» non dire mai che non si applica (né «non pertinente», né «non rilevante»).\n"
    "- Una norma con esito «argomento del cliente» è quella su cui si fonda la controparte e di cui un elemento non "
    "risulta: scrivila come argomento del cliente -- la controparte può invocarla, ma il fatto CONTRO esclude "
    "quell'elemento --, mai come una norma «non applicabile» senza dire chi la invoca e perché non regge.\n"
    "- Nella sezione Norme applicate ai fatti ogni norma delle SCHEDE ha il suo paragrafo; nelle altre sezioni cita una "
    "norma solo dove sostiene un argomento.\n"
    "- Le questioni sono gli argomenti scelti: le CONCLUSIONI, i TERMINI e i profili processuali sono contesto, da "
    "usare solo dove riguardano questi argomenti, mai una questione nuova.\n"
    "- Un fatto si afferma solo se è in una frase del caso, come il caso lo dà (un'allegazione resta un'allegazione, "
    "un'ipotesi un'ipotesi); altrimenti è ciò che manca e va provato. Un diritto è già acquisito solo se le CONCLUSIONI "
    "lo dicono; altrimenti è da provare.\n"
    "- Non scrivere mai i numeri delle frasi del caso (S1, S2…) né quelli dei requisiti (R1…).\n"
    + STATUS_WORDS_RULE + STATE_OF_MIND_RULE + NO_SIGNAL_RULE)

DEFENDANTS_RULE = ("Sono ricavate dai fatti: indica per nome quelle che il caso nomina; una parte che il caso non "
                   "identifica è «da individuare», mai «eventuale».")

RULES_TEXT = ("Enuncia la regola di ciascuna norma riportando il comma tra «…» con il suo numero («co. N») o con parole "
              "tue che restano dentro quel comma. Lo stato di ogni requisito è quello delle SCHEDE DELLE NORME: non "
              "cambiarlo e non aggiungere requisiti.")

NORMS_INSTRUCTION = (
    "SEZIONE: Norme applicate ai fatti. Per ciascuna norma delle SCHEDE DELLE NORME, nell'ordine, un paragrafo aperto dal "
    "numero della norma tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»), senza titolo: la regola, "
    "riportando il comma tra «…» con il suo numero o restando dentro le sue parole; poi l'applicazione ai fatti, "
    "requisito per requisito come la SCHEDA la dà: per uno «provato» il fatto del caso che lo prova; per uno «da "
    "provare» la prova da acquisire; per uno «non risulta» il fatto che lo esclude; per una norma con esito «argomento "
    "del cliente», prima la pretesa della controparte, poi perché non regge. Lo stato di ogni requisito è già "
    "scritto sopra il paragrafo: non ripeterlo e non contraddirlo. Solo le norme delle SCHEDE: le altre norme "
    "dell'argomento hanno già il loro titolo o la loro riga e non vanno commentate.")

# v60q (single shot, my drafts): an article is a reference, never its quoted text -- §2 cites it «art. N codice — rubrica»
# with the requisites' status lines (the code's), the writer's application under it; the text goes to the response's
# article citations
REFERENCES_RULE = (
    "- Le norme si citano come riferimento («art. N codice»), mai riportandone il testo: nessun comma tra «…», nessuna "
    "frase copiata dalla norma, nemmeno con «(co. N)». Il testo lo apre chi legge; tu scrivi come la norma si applica "
    "ai fatti.\n")
RULES_TEXT_REFERENCES = ("La regola di ciascuna norma la ricavi dal suo testo (TESTI DELLE NORME), ma non riportarlo, né "
                         "tra «…» né parola per parola: cita la norma come «art. N codice» e applicala ai fatti. Lo stato "
                         "di ogni requisito è quello delle SCHEDE DELLE NORME: non cambiarlo e non aggiungere requisiti.")
NORMS_INSTRUCTION_REFERENCES = (
    "SEZIONE: Norme applicate ai fatti. Per ciascuna norma delle SCHEDE DELLE NORME, nell'ordine, un paragrafo aperto dal "
    "numero della norma tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»), senza titolo e senza riportare il "
    "testo della norma: come si applica ai fatti, requisito per requisito come la SCHEDA la dà -- per un requisito "
    "soddisfatto il fatto del caso che lo prova; per uno ancora aperto la prova da acquisire; per uno escluso il fatto "
    "che lo esclude; per una norma con esito «argomento del cliente», prima la pretesa della controparte, poi perché non "
    "regge. Lo stato di ogni requisito è già scritto sopra il paragrafo: non ripeterlo e non contraddirlo. Solo le norme "
    "delle SCHEDE: le altre norme dell'argomento hanno già il loro titolo o la loro riga e non vanno commentate.")

SCOPE_NOTE = (" Le questioni sono gli argomenti scelti; il resto del materiale è contesto, da usare solo dove li "
              "riguarda.")


def system_rules(party: str, references: bool = False) -> str:
    """The interactive writer's rules for the system prompt: RECORD_RULES (v60q, references -- single shot: with
    REFERENCES_RULE), then plan_then_write._WRITER_RULES without its first rule (P1/P2, replaced)."""
    from .plan_then_write import _WRITER_RULES
    rest = _WRITER_RULES.format(party=party or "la parte assistita")
    first_end = rest.find("\n- ")
    return RECORD_RULES + (REFERENCES_RULE if references else "") + (rest[first_end + 1:] if first_end >= 0 else rest)


def as_log() -> Dict[str, Any]:
    return {"resolved": ["P1", "P2", "P3", "P4", "P6"], "chars": len(RECORD_RULES) + len(NORMS_INSTRUCTION)}
