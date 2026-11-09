"""TOPIC_TABLES (default false; v55, drafts pending the lawyer's validation): the criminal and administrative institutions
the fact rules and REGIME_CANDIDATES do not hold -- anchors and questions only, never a conclusion.

Each entry: "family" (penale / amministrativo: a case of another family never reads it, fact_signals.family), "name"
(the topic's title), "triggers" (patterns searched, as REGIME_CANDIDATES', in the case's issue and institution names --
fact_signals.case_issue_names), "refs" (its articles, the anchor first), "anchors" ([(pattern, ref, title)]: a name
matching the pattern moves that article first and gives the topic that title -- "furto in abitazione" anchors on 624-bis
c.p.) and "questions" (the interactive flow's question families -- interaction_copy.QUESTIONS -- a term of the topic runs
from a date the case may not give; a family with no template is not asked).

topic_step reads them as institutions of source "tabella", after the regimes: the anchor its first article, the topic's
articles its own then the anchor's Sezione and links (topic_step.articles). scripts/interactive_review.py --tables-report
replays them offline on a run's records.
"""

import re
from typing import Any, Dict, List, Optional

from . import feature_flags

FLAG = "TOPIC_TABLES"
_L241 = "Legge sul Procedimento Amministrativo 2026"
_APPALTI = "Codice degli Appalti 2026"
_TUEL = "Testo unico degli enti locali 2026"

TABLES: List[Dict[str, Any]] = [
    # --- penale -----------------------------------------------------------------------------------------------------
    {"id": "rapina", "family": "penale", "name": "rapina", "triggers": [r"rapin\w*"],
     "refs": [("628", "c.p.")], "questions": []},
    # furto: procedibile a querela salvo le aggravanti (624 co. 3, 625): the date of the fact runs the querela's term
    {"id": "furto", "family": "penale", "name": "furto",
     "triggers": [r"furt\w*", r"scipp\w*", r"borseggi\w*", r"sottrazion\w* di (?:cose|beni)"],
     "refs": [("624", "c.p."), ("624-bis", "c.p."), ("625", "c.p."), ("61", "c.p.")],
     "anchors": [(r"abitazion|strapp|scipp|dimora", ("624-bis", "c.p."), "furto in abitazione o con strappo"),
                 (r"aggravat", ("625", "c.p."), "furto aggravato")],
     "questions": ["data_fatto"]},
    {"id": "truffa", "family": "penale", "name": "truffa", "triggers": [r"truff\w*", r"raggir\w*", r"artifiz\w*"],
     "refs": [("640", "c.p.")], "questions": ["data_fatto"]},
    {"id": "corruzione", "family": "penale", "name": "corruzione",
     "triggers": [r"corruzion\w*", r"corrott\w*", r"corruttor\w*", r"tangent\w*"],
     "refs": [("318", "c.p."), ("319", "c.p."), ("319-ter", "c.p."), ("320", "c.p."), ("321", "c.p."), ("322", "c.p.")],
     "anchors": [(r"contrari\w* ai doveri|atto contrario|propria", ("319", "c.p."),
                  "corruzione per un atto contrario ai doveri d'ufficio"),
                 (r"atti giudiziari", ("319-ter", "c.p."), "corruzione in atti giudiziari"),
                 (r"incaricat\w* di (?:un )?pubblico servizio", ("320", "c.p."),
                  "corruzione di persona incaricata di un pubblico servizio")],
     "questions": []},
    {"id": "concorso_persone", "family": "penale", "name": "concorso di persone nel reato",
     "triggers": [r"concors\w* (?:di persone|nel reato|esterno|morale|materiale|eventuale|anomalo)", r"concorrent\w* nel",
                  r"complic\w*"],
     "refs": [("110", "c.p."), ("112", "c.p."), ("114", "c.p."), ("116", "c.p."), ("117", "c.p.")],
     "anchors": [(r"reat\w* propri\w*|qualit\w* personal|117", ("117", "c.p."),
                  "concorso dell'estraneo nel reato proprio"),
                 (r"anomal|divers\w* da quello voluto", ("116", "c.p."), "reato diverso da quello voluto")],
     "questions": []},
    # v56 (v55-p1): the defence and its excess, with the contested offence (575 the intentional, 589 the negligent killing);
    # the questions are 52 co. 2's requisites -- a weapon lawfully held, no desistance, a danger of aggression
    {"id": "legittima_difesa", "family": "penale", "name": "legittima difesa ed eccesso colposo",
     "triggers": [r"legittima difesa", r"difesa legittima", r"eccesso colpos\w*", r"eccesso di difesa",
                  r"difesa (?:domiciliare|personale|putativa)", r"scriminant\w*"],
     "refs": [("52", "c.p."), ("55", "c.p."), ("575", "c.p."), ("589", "c.p.")],
     "anchors": [(r"eccesso", ("55", "c.p."), "eccesso colposo di legittima difesa")],
     "questions": ["arma_detenuta", "desistenza", "pericolo_aggressione"], "gate": "reacted"},
    {"id": "tentativo", "family": "penale", "name": "delitto tentato", "triggers": [r"tentativ\w*", r"tentat[oai]\b"],
     "refs": [("56", "c.p.")], "questions": []},
    # lesioni: 582 (lievissime) and 590 a querela: the date of the fact runs the querela's term
    {"id": "lesioni", "family": "penale", "name": "lesione personale",
     "triggers": [r"lesion\w* personal\w*", r"lesion\w*", r"percoss\w*"],
     "refs": [("582", "c.p."), ("583", "c.p."), ("585", "c.p."), ("590", "c.p.")],
     "anchors": [(r"colpos|stradal|sanitari|medic", ("590", "c.p."), "lesioni personali colpose"),
                 (r"grav", ("583", "c.p."), "lesioni personali aggravate")],
     "questions": ["data_fatto"]},
    {"id": "associazione_mafiosa", "family": "penale", "name": "associazione di tipo mafioso",
     "triggers": [r"mafios\w*", r"stampo mafios\w*", r"\bmafi\w*", r"camorr\w*", r"ndrangh\w*", r"\bcosc[ah]e?\b",
                  r"416-bis"],
     "refs": [("416-bis", "c.p."), ("416", "c.p."), ("110", "c.p.")], "questions": []},
    # misure cautelari: the riesame runs ten days from the ordinanza's execution or notification (art. 309 c.p.p.)
    {"id": "misure_cautelari", "family": "penale", "name": "misure cautelari personali",
     "triggers": [r"misur\w* cautelar\w*", r"custodia cautelare", r"arresti domiciliari", r"ordinanz\w* cautelar\w*",
                  r"riesame", r"esigenz\w* cautelar\w*"],
     "refs": [("274", "c.p.p."), ("273", "c.p.p."), ("275", "c.p.p."), ("285", "c.p.p."), ("309", "c.p.p.")],
     "anchors": [(r"riesame", ("309", "c.p.p."), "riesame della misura cautelare"),
                 (r"custodia cautelare|carcere", ("285", "c.p.p."), "custodia cautelare in carcere")],
     "questions": ["data_misura"]},
    # prescrizione del reato: it runs from the day of the consummation (art. 158 c.p.)
    {"id": "prescrizione_reato", "family": "penale", "name": "prescrizione del reato",
     "triggers": [r"prescrizion\w* del reato", r"estinzion\w* del reato", r"prescrizion\w*"],
     "refs": [("157", "c.p."), ("158", "c.p."), ("159", "c.p."), ("160", "c.p."), ("161", "c.p.")],
     "questions": ["data_fatto"]},
    # --- amministrativo ---------------------------------------------------------------------------------------------
    {"id": "motivazione", "family": "amministrativo", "name": "motivazione del provvedimento",
     "triggers": [r"motivazion\w*", r"difetto di istruttori\w*"],
     "refs": [("3", _L241), ("21-octies", _L241)], "questions": []},
    {"id": "partecipazione", "family": "amministrativo", "name": "comunicazione di avvio del procedimento",
     "triggers": [r"avvio del procedimento", r"comunicazion\w* (?:di )?avvio", r"partecipazion\w* al procedimento",
                  r"contraddittorio procedimentale"],
     "refs": [("7", _L241), ("8", _L241), ("10", _L241), ("21-octies", _L241)], "questions": []},
    {"id": "preavviso_rigetto", "family": "amministrativo", "name": "preavviso di rigetto",
     "triggers": [r"preavviso di rigetto", r"motivi ostativi", r"10-bis", r"diniego"],
     "refs": [("10-bis", _L241), ("21-octies", _L241)], "questions": []},
    {"id": "autotutela", "family": "amministrativo", "name": "autotutela",
     "triggers": [r"autotutela", r"annullament\w* d['’]ufficio", r"revoc\w*", r"ritiro del provvediment\w*"],
     "refs": [("21-nonies", _L241), ("21-quinquies", _L241), ("21-octies", _L241)],
     "anchors": [(r"revoc", ("21-quinquies", _L241), "revoca del provvedimento"),
                 (r"annullament\w* d['’]ufficio|autotutela", ("21-nonies", _L241), "annullamento d'ufficio")],
     "questions": []},
    {"id": "invalidita_provvedimento", "family": "amministrativo", "name": "invalidità del provvedimento",
     "triggers": [r"annullabil\w*", r"illegittim\w*", r"violazion\w* di legge", r"eccesso di potere", r"incompetenz\w*",
                  r"nullit\w* del provvediment\w*", r"vizi\w* del provvediment\w*"],
     "refs": [("21-octies", _L241), ("21-septies", _L241)],
     "anchors": [(r"nullit", ("21-septies", _L241), "nullità del provvedimento")],
     "questions": []},
    # ricorso al TAR: sixty days from the notification, communication or full knowledge of the act (artt. 29 and 41)
    {"id": "ricorso_tar", "family": "amministrativo", "name": "ricorso al TAR per l'annullamento",
     "triggers": [r"\btar\b", r"tribunale amministrativo", r"ricors\w* giurisdizional\w*",
                  r"annullament\w* (?:del|dell['’]|di) (?:provvediment|att)", r"impugnazion\w* (?:del|dell['’]|di)"
                  r" (?:provvediment|att)", r"silenzi\w*", r"risarciment\w* (?:del danno )?(?:da|per) (?:provvediment|"
                  r"lesione di interess)", r"interess\w* legittim\w*"],
     "refs": [("29", "c.p.a."), ("41", "c.p.a."), ("30", "c.p.a."), ("31", "c.p.a.")],
     "anchors": [(r"silenzi|inerzi", ("31", "c.p.a."), "ricorso avverso il silenzio"),
                 (r"risarciment", ("30", "c.p.a."), "azione di condanna al risarcimento")],
     "questions": ["atto_notifica"]},
    # v56 (v55-h10): the judgment's execution and its elusion -- the act eluding it is null (art. 21-septies l. 241/1990),
    # the nullity asked within 180 days (art. 31 co. 4 c.p.a.): the act's date, by data_atto_elusivo
    {"id": "ottemperanza", "family": "amministrativo", "name": "ottemperanza ed elusione del giudicato",
     "triggers": [r"ottemperanz\w*", r"elusion\w*", r"elus\w* (?:del |il )?giudicat\w*", r"violazion\w* del giudicat\w*",
                  r"giudicat\w*", r"esecuzion\w* (?:della|di|delle) sentenz\w*", r"commissari\w* ad acta",
                  r"sentenz\w* del (?:tar|consiglio di stato)"],
     "refs": [("112", "c.p.a."), ("114", "c.p.a."), ("21-septies", _L241), ("31", "c.p.a.")],
     "anchors": [(r"elusion|elus|violazion\w* del giudicat|nullit", ("21-septies", _L241),
                  "elusione del giudicato (nullità del provvedimento)")],
     "questions": ["data_atto_elusivo", "data_giudicato"]},
    # appalti: the causes of exclusion; the special rite (art. 120 c.p.a., thirty days from the communication of the award
    # or the exclusion) is the ricorso_tar_appalti term's topic, never this one's article (a one-article term topic would
    # take this one in: topic_step's overlap)
    {"id": "appalti", "family": "amministrativo", "name": "esclusione dalla gara",
     "triggers": [r"appalt\w*", r"\bgara\b", r"aggiudicazion\w*", r"contratt\w* pubblic\w*", r"esclusion\w*"],
     "refs": [("94", _APPALTI), ("95", _APPALTI), ("96", _APPALTI), ("98", _APPALTI)],
     "anchors": [(r"illecit\w* professional|grave illecito", ("98", _APPALTI), "illecito professionale grave"),
                 (r"esclusion\w* automatic", ("94", _APPALTI), "esclusione automatica dalla gara")],
     "questions": ["atto_notifica"]},
    {"id": "dissesto", "family": "amministrativo", "name": "dissesto dell'ente locale",
     "triggers": [r"dissest\w*", r"\btuel\b", r"organo straordinario di liquidazione", r"liquidazion\w* dell['’]ente"],
     "refs": [("252", _TUEL), ("248", _TUEL), ("254", _TUEL)], "questions": []},
]


# v57 (J: v56-p1 had no procedural section once the civil tools were gated out): the criminal offences' procedural
# facts, for render_computed's «Procedibilità e competenza» -- drafts pending the lawyer: (name, procedibilità, competenza)
CRIMINAL_PROCEDURE = {
    ("575", "c.p."): ("omicidio", "d'ufficio", "Corte d'assise (art. 5 c.p.p.)"),
    ("589", "c.p."): ("omicidio colposo", "d'ufficio", ""),
    ("628", "c.p."): ("rapina", "d'ufficio", ""),
    ("624", "c.p."): ("furto", "a querela della persona offesa, d'ufficio con le aggravanti (art. 624, co. 3)", ""),
    ("640", "c.p."): ("truffa", "a querela della persona offesa, d'ufficio con le aggravanti (art. 640, co. 3)", ""),
    ("582", "c.p."): ("lesione personale", "a querela della persona offesa, d'ufficio nei casi aggravati (art. 582, co. 2)",
                      ""),
}


def offences(analysis: Optional[Dict[str, Any]], areas: Optional[List[str]] = None) -> List[tuple]:
    """The CRIMINAL_PROCEDURE offences the case's tables hold (matches, whatever TOPIC_TABLES: the procedural section
    reads them), in the table's order."""
    refs = [tuple(r) for t in matches(analysis, areas) for r in t["refs"]]
    return [r for r in CRIMINAL_PROCEDURE if r in refs]


# v58 (v57-h5: 52, 53, 55 c.p. on the word «armi»): a table read only when the facts hold its situation -- "reacted":
# an aggression, an intrusion or a threat, and someone reacting to it
_AGGRESSION = r"aggred|aggression|aggressor|minacc|intrus|ladr|rapinator|irromp|introdott\w* (?:in casa|nell['’]abitazione)|scassin"
_REACTION = r"reag|si difend|difender|spar|colp[iì]|feris|ferit|uccid|uccis|accoltell|respin|immobilizz"


def _texts(analysis: Optional[Dict[str, Any]]) -> str:
    from . import fact_signals
    facts = ((analysis or {}).get("fact_signals") or {}).get("facts") or {}
    frame = (analysis or {}).get("frame") or {}
    return fact_signals._low(" ".join([fact_signals._texts(facts, ("events", "injuries", "proceedings"))]
                                      + [str(x) for x in frame.get("facts") or []]))


def gate_open(gate: str, analysis: Optional[Dict[str, Any]]) -> bool:
    if gate == "reacted":
        text = _texts(analysis)
        return bool(re.search(_AGGRESSION, text) and re.search(_REACTION, text))
    return True


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def matches(analysis: Optional[Dict[str, Any]], areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """[{id, name, refs, questions, matched: [(trigger, the name it is in)]}] of the TABLES the case's issue and
    institution names call for, within the families of its areas (none known: every family); the anchor and the title
    from the first "anchors" pattern a matching name holds ("table_name": the table's own name, topic_step.named_by_table)."""
    from . import fact_signals
    families = {fact_signals.family(a) for a in areas or []} - {""}
    names = fact_signals.case_issue_names(analysis)
    out = []
    for table in TABLES:
        if families and table["family"] not in families:
            continue
        if table.get("gate") and not gate_open(table["gate"], analysis):
            continue
        matched = [(m.group(0), name) for name in names for t in table["triggers"]
                   for m in [re.search(rf"\b{t}", fact_signals._low(name))] if m]
        if not matched:
            continue
        refs, title = [tuple(r) for r in table["refs"]], table["name"]
        hit = [n for _, n in matched]
        for pattern, ref, named in table.get("anchors") or []:
            if any(re.search(pattern, fact_signals._low(n)) for n in hit):
                refs = [tuple(ref)] + [r for r in refs if r != tuple(ref)]
                title = named
                break
        out.append({"id": table["id"], "name": title, "refs": refs, "questions": list(table.get("questions") or []),
                    "matched": list(dict.fromkeys(matched)), "table_name": table["name"]})
    return out
