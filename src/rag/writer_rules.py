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

The set replaces plan_then_write._WRITER_RULES' first rule (P1/P2) for the interactive answer; its other rules (the
burden of proof, no conduct the case does not state, rulings, terms, the client's side, the institution rules) stand.
"""

from typing import Any, Dict

STATUS_DEFINITIONS = (
    "«provato»: una frase del caso, narrata come fatto, lo dice (riprendila con le parole del caso); «da provare»: il "
    "caso non lo stabilisce (nessun fatto, o solo un'allegazione o un'ipotesi): indica la prova da acquisire (MANCA); "
    "«non risulta»: il caso dice il contrario (il fatto CONTRO): per quel requisito la norma non si applica.")

RECORD_RULES = (
    "- Le SCHEDE DELLE NORME sono il tuo materiale: per ogni norma lo stato di ciascun requisito, i fatti del caso a "
    "favore e contro, che cosa manca. Non ricavarli di nuovo, non cambiarli. Stati: " + STATUS_DEFINITIONS + "\n"
    "- Le norme con esito «non applicabile» non sono nelle SCHEDE: le elenca il codice in una riga («Norme considerate e "
    "non applicabili»); non commentarle. Di ogni norma delle SCHEDE non dire mai che non si applica (né «non "
    "pertinente», né «non rilevante»).\n"
    "- Nella sezione Norme applicate ai fatti ogni norma delle SCHEDE ha il suo paragrafo; nelle altre sezioni cita una "
    "norma solo dove sostiene un argomento.\n"
    "- Le questioni sono gli argomenti scelti: le CONCLUSIONI, i TERMINI e i profili processuali sono contesto, da "
    "usare solo dove riguardano questi argomenti, mai una questione nuova.\n"
    "- Un fatto si afferma solo se è in una frase del caso, come il caso lo dà (un'allegazione resta un'allegazione, "
    "un'ipotesi un'ipotesi); altrimenti è ciò che manca e va provato. Un diritto è già acquisito solo se le CONCLUSIONI "
    "lo dicono; altrimenti è da provare.\n"
    "- Non scrivere mai i numeri delle frasi del caso (S1, S2…) né quelli dei requisiti (R1…).\n")

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
    "provare» la prova da acquisire; per uno «non risulta» il fatto che lo esclude. Lo stato di ogni requisito è già "
    "scritto sopra il paragrafo: non ripeterlo e non contraddirlo. Solo le norme delle SCHEDE: le altre norme "
    "dell'argomento hanno già il loro titolo o la loro riga e non vanno commentate.")

SCOPE_NOTE = (" Le questioni sono gli argomenti scelti; il resto del materiale è contesto, da usare solo dove li "
              "riguarda.")


def system_rules(party: str) -> str:
    """The interactive writer's rules for the system prompt: RECORD_RULES, then plan_then_write._WRITER_RULES without
    its first rule (P1/P2, replaced)."""
    from .plan_then_write import _WRITER_RULES
    rest = _WRITER_RULES.format(party=party or "la parte assistita")
    first_end = rest.find("\n- ")
    return RECORD_RULES + (rest[first_end + 1:] if first_end >= 0 else rest)


def as_log() -> Dict[str, Any]:
    return {"resolved": ["P1", "P2", "P3", "P4", "P6"], "chars": len(RECORD_RULES) + len(NORMS_INSTRUCTION)}
