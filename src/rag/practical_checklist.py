"""PLAN_THEN_WRITE_ENABLED: the practical checks of a case's posture and area, for the
parere's "Profili processuali e termini" section (plan_then_write.py).

The lawyer's review of the held-out drafts (22 Oct) found the practical content missing:
deadlines, appeal grounds, who to sue, damages, subordinate requests. The PM's test cases
(23 Oct) asked for it directly ("Qual è la modalità processuale migliore") and got
"processo ordinario". These lists are deterministic, keyed like method_candidates.py
(the case frame's own "posture_type" and areas); the writer uses an item only where the
case supports it, and a deadline that needs a date the case lacks becomes a check to make.

DRAFT, pending the lawyer's validation, like method_candidates.py.
"""

from typing import Dict, Iterable, List

# posture -> what to verify at that stage
POSTURE_CHECKS: Dict[str, List[str]] = {
    "precontenzioso": [
        "Prescrizione del diritto (ordinaria di dieci anni, art. 2946 c.c.; cinque anni per il danno "
        "extracontrattuale, art. 2947 c.c.) e atti che la interrompono (art. 2943 c.c.); eventuali decadenze "
        "proprie del contratto o della materia.",
        "Se la parte assistita riceve una pretesa: risposta scritta entro il termine indicato, contestando ciò che "
        "non è dovuto, con le eccezioni che ha (inadempimento della controparte, prescrizione, riduzione).",
        "Se è la parte assistita a pretendere: messa in mora (art. 1219 c.c.) o diffida ad adempiere (art. 1454 "
        "c.c.); decreto ingiuntivo per un credito provato per iscritto (art. 633 c.p.c.).",
        "Condizioni di procedibilità prima della causa: se la materia richiede la mediazione (art. 5 d.lgs. "
        "28/2010) o la negoziazione assistita.",
        "Consulenza tecnica preventiva per accertare i fatti e tentare la conciliazione (art. 696-bis c.p.c.).",
        "Chi è la controparte da convenire o a cui rivolgersi, ed eventuali altri obbligati o assicuratori.",
        "Domande o eccezioni in via subordinata (per esempio, la riduzione di un importo).",
    ],
    "giudizio_civile_primo_grado": [
        "Preclusioni del giudizio in corso: termini per le memorie, le domande nuove e le richieste di prova.",
        "Onere della prova (art. 2697 c.c.): che cosa deve provare ciascuna parte e con quali mezzi.",
        "Chiamata in causa di terzi o di garanti (assicuratori, corresponsabili).",
        "Domande ed eccezioni in via subordinata.",
    ],
    "sentenza_civile_primo_grado": [
        "Termine per l'appello: trenta giorni dalla notificazione della sentenza (art. 325 c.p.c.), altrimenti "
        "sei mesi dalla pubblicazione (art. 327 c.p.c.).",
        "Motivi specifici a pena di inammissibilità (art. 342 c.p.c.); filtro di inammissibilità (art. 348-bis "
        "c.p.c.).",
        "Limiti alle domande, eccezioni e prove nuove in appello (art. 345 c.p.c.).",
        "Sospensione dell'efficacia esecutiva della sentenza, se ne ricorrono i presupposti.",
    ],
    "sentenza_civile_appello": [
        "Termine per il ricorso per cassazione: sessanta giorni dalla notificazione (art. 325 c.p.c.), "
        "altrimenti sei mesi dalla pubblicazione (art. 327 c.p.c.).",
        "Motivi solo tra quelli dell'art. 360 c.p.c. e contenuto del ricorso (art. 366 c.p.c.), a pena di "
        "inammissibilità.",
    ],
    "esecuzione_civile": [
        "Titolo esecutivo e precetto (artt. 474 e 480 c.p.c.).",
        "Opposizione all'esecuzione (art. 615 c.p.c.) o agli atti esecutivi, entro venti giorni (art. 617 "
        "c.p.c.).",
    ],
    "indagini_preliminari": [
        "Avviso di conclusione delle indagini (art. 415-bis c.p.p.): entro venti giorni la difesa può presentare "
        "memorie e documenti, chiedere nuove indagini o di essere interrogata.",
        "L'archiviazione la chiede il pubblico ministero (art. 408 c.p.p.): la difesa può solo sollecitarla con "
        "memorie e argomenti.",
        "Riti alternativi e loro limiti, scelti dall'indagato: giudizio abbreviato (art. 438 c.p.p., con la "
        "riduzione della pena); applicazione della pena su richiesta (art. 444 c.p.p., con limiti ed esclusioni "
        "per alcuni reati, tra cui la violenza sessuale, da verificare); sospensione del procedimento con messa "
        "alla prova (art. 464-bis c.p.p., solo per i reati entro i limiti di pena previsti).",
        "Procedibilità: reato a querela o d'ufficio; termine della querela e possibilità di remissione.",
        "Misure cautelari in corso e relativi rimedi; incidente probatorio quando una prova rischia di "
        "disperdersi o la persona offesa è vulnerabile.",
        "Qualificazione giuridica del fatto contestato e possibile riqualificazione in un reato meno grave.",
    ],
    "giudizio_penale_primo_grado": [
        "Riti alternativi ancora possibili e loro termini (artt. 438, 444, 464-bis c.p.p.).",
        "Proscioglimento immediato per una causa evidente (art. 129 c.p.p.).",
        "Regola di giudizio: condanna solo se la colpevolezza è provata al di là di ogni ragionevole dubbio "
        "(art. 533 c.p.p.), altrimenti assoluzione (art. 530 c.p.p.).",
        "Riqualificazione del fatto, circostanze attenuanti, particolare tenuità del fatto (art. 131-bis c.p., "
        "solo entro i limiti di pena previsti), sospensione condizionale della pena.",
    ],
    "condanna_primo_grado": [
        "Termine per l'appello (art. 585 c.p.p.) e motivi specifici a pena di inammissibilità (art. 581 "
        "c.p.p.).",
        "Divieto di reformatio in peius sull'appello del solo imputato (art. 597 c.p.p.); rinnovazione "
        "dell'istruzione (art. 603 c.p.p.).",
    ],
    "sentenza_penale_appello": [
        "Ricorso per cassazione: termine (art. 585 c.p.p.) e motivi solo tra quelli dell'art. 606 c.p.p., a "
        "pena di inammissibilità (art. 591 c.p.p.).",
    ],
    "provvedimento_amministrativo": [
        "Termine di impugnazione: sessanta giorni dalla notificazione, comunicazione o piena conoscenza (artt. "
        "29 e 41 c.p.a.), con notifica ad almeno un controinteressato.",
        "Domanda cautelare (art. 55 c.p.a.) se il provvedimento causa un pregiudizio grave e irreparabile.",
        "Azione di risarcimento entro centoventi giorni (art. 30 c.p.a.).",
    ],
    "silenzio_amministrazione": [
        "Azione avverso il silenzio (artt. 31 e 117 c.p.a.), finché dura l'inadempimento e comunque entro un "
        "anno dalla scadenza del termine per provvedere.",
        "Termine del procedimento (art. 2 l. 241/1990) e sua scadenza.",
    ],
    "giudicato_amministrativo_non_eseguito": [
        "Ricorso per l'ottemperanza (artt. 112-114 c.p.a.) entro il termine di prescrizione dell'actio iudicati "
        "(art. 114 c.p.a.); nomina di un commissario ad acta e penalità di mora.",
    ],
}

# area -> what to verify in that matter
AREA_CHECKS: Dict[str, List[str]] = {
    "penale": [
        "Elemento soggettivo (dolo o colpa) e nesso causale: che cosa l'accusa deve provare.",
        "Reato tentato o consumato (art. 56 c.p.) e desistenza volontaria, se l'azione non si è compiuta.",
        "Concorso di persone (art. 110 c.p.): il ruolo di ciascuno e la responsabilità per il reato diverso "
        "voluto da uno solo dei concorrenti (art. 116 c.p.).",
        "Circostanze aggravanti contestate e attenuanti applicabili (artt. 61, 62, 62-bis c.p.); particolare "
        "tenuità del fatto (art. 131-bis c.p.), solo entro i limiti di pena previsti.",
    ],
    "contratti": [
        "Inadempimento imputabile o no (art. 1218 c.c.): causa esterna, impossibilità sopravvenuta, prova "
        "liberatoria del debitore.",
        "Clausole del contratto e loro limiti: penale e sua riduzione (artt. 1382 e 1384 c.c.), interpretazione "
        "secondo buona fede (artt. 1366 e 1375 c.c.).",
        "Rimedi: risoluzione e importanza dell'inadempimento (artt. 1453 e 1455 c.c.), eccezione "
        "d'inadempimento (art. 1460 c.c.).",
        "Danno risarcibile e concorso del creditore (artt. 1223 e 1227 c.c.); prescrizione (art. 2946 c.c.).",
    ],
    "responsabilita_civile": [
        "Titolo della responsabilità: contrattuale (art. 1218 c.c., anche da contatto sociale) o "
        "extracontrattuale (art. 2043 c.c.), con il diverso onere della prova e la diversa prescrizione "
        "(artt. 2946 e 2947 c.c.).",
        "Chi risponde: l'autore, chi ha la custodia della cosa (art. 2051 c.c.), chi esercita un'attività "
        "pericolosa (art. 2050 c.c.), l'ente o il datore di lavoro (art. 2049 c.c.), l'assicuratore.",
        "Voci di danno, patrimoniale e non patrimoniale (art. 2059 c.c.), e loro prova.",
        "Concorso del danneggiato (art. 1227 c.c.).",
    ],
    "diritti_reali": [
        "Titolo del diritto: contratto, usucapione (art. 1158 c.c.), destinazione del padre di famiglia per le "
        "servitù apparenti (art. 1062 c.c.), sentenza.",
        "Tutela del possesso: reintegrazione o manutenzione entro un anno dallo spoglio o dalla turbativa "
        "(artt. 1168 e 1170 c.c.).",
        "Azioni a tutela del diritto: confessoria della servitù (art. 1079 c.c.), negatoria (art. 949 c.c.), "
        "rivendicazione (art. 948 c.c.).",
        "Mediazione obbligatoria prima della causa nelle controversie in materia di diritti reali (art. 5 "
        "d.lgs. 28/2010).",
    ],
    "locazione": [
        "Rito delle locazioni (art. 447-bis c.p.c.) e procedimento per convalida di sfratto (artt. 657 e 658 "
        "c.p.c.).",
        "Inadempimento e sua importanza (artt. 1453 e 1455 c.c.); obblighi del locatore e del conduttore "
        "(artt. 1575 e 1587 c.c.).",
        "Mediazione obbligatoria prima della causa in materia di locazione (art. 5 d.lgs. 28/2010), esclusa "
        "per il procedimento di convalida.",
    ],
    "successioni": [
        "Accettazione o rinuncia all'eredità e loro termini (artt. 470, 480 e 519 c.c.); accettazione con "
        "beneficio d'inventario (art. 484 c.c.).",
        "Mediazione obbligatoria prima della causa in materia di successioni (art. 5 d.lgs. 28/2010).",
    ],
    "famiglia": [
        "Interesse del minore e affidamento (artt. 337-bis e 337-ter c.c.); mantenimento (art. 316-bis c.c.).",
    ],
    "consumo": [
        "Clausole vessatorie nei contratti con il consumatore (artt. 33-36 Codice del Consumo).",
        "Garanzia legale di conformità e termini per far valere il difetto (artt. 128 ss. Codice del Consumo).",
    ],
    "amministrativo": [
        "Vizi del provvedimento: motivazione (art. 3 l. 241/1990), preavviso di rigetto (art. 10-bis l. "
        "241/1990), vizi non invalidanti (art. 21-octies l. 241/1990).",
        "Autotutela: annullamento d'ufficio e suoi termini (art. 21-nonies l. 241/1990).",
    ],
}


def checks_for(areas: Iterable[str], posture_type: str = "") -> List[str]:
    """The posture's checks, then each area's, without repeats."""
    checks = list(POSTURE_CHECKS.get(posture_type or "", []))
    for area in areas:
        checks += [c for c in AREA_CHECKS.get(area, []) if c not in checks]
    return checks
