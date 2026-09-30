"""Labelled requests for template selection.

Each case: (request, acceptable catalog indices, new-template index or None).
Indices refer to the server catalog order (catalog_enriched_server.json) and to
the sorted list of the 103 converted templates.

kind
  found     - the catalog has the right template (one or several acceptable)
  near      - the exact template is one of the 103 new ones, but the catalog has
              a generic one a lawyer could reasonably start from
  missing   - only the new template fits; today the right answer is "we don't
              have it", after the upload it is the new template
  unrelated - nothing fits, before or after the upload
"""

FOUND = [
    ("Scrivi un ricorso ex art. 700 c.p.c.", [66, 67, 65]),
    ("Scrivi Ricorso ex 700 cpc", [66, 67, 65]),
    ("Mi serve un atto di precetto per una sentenza civile", [97, 430]),
    ("Prepara un'intimazione di sfratto per morosità dell'inquilino", [124, 435]),
    ("Redigi una querela per stalking", [445, 264]),
    ("Ho bisogno di una comparsa di costituzione e risposta", [8, 27, 24]),
    ("Opposizione a decreto ingiuntivo", [112]),
    ("ricorso per decreto ingiuntivo per un credito non pagato", [113, 403, 454, 455]),
    ("Contratto di locazione 4+4 a canone libero", [384, 401]),
    ("contratto di affitto per studenti universitari fuori sede", [382]),
    ("Lettera di licenziamento per giusta causa", [390]),
    ("devo impugnare il licenziamento del mio cliente", [389, 459, 466, 462]),
    ("lettera di dimissioni volontarie", [458]),
    ("Ricorso al TAR contro un provvedimento del comune", [306, 397]),
    ("Istanza di accesso civico generalizzato FOIA", [366, 361]),
    ("ricorso contro il silenzio della pubblica amministrazione", [369, 368]),
    ("Ricorso per ottemperanza a una sentenza del TAR non eseguita", [351]),
    ("Atto di appello contro la sentenza civile di primo grado", [82]),
    ("Ricorso per cassazione penale", [189, 483, 245]),
    ("Richiesta di patteggiamento", [210, 211]),
    ("istanza di sospensione del procedimento con messa alla prova", [213, 222]),
    ("Opposizione alla richiesta di archiviazione del PM", [266]),
    ("Costituzione di parte civile nel processo penale", [269, 440]),
    ("Nomina del difensore di fiducia", [141, 263]),
    ("richiesta di riesame contro il sequestro preventivo", [191, 192, 283]),
    ("domanda di riparazione per ingiusta detenzione", [438]),
    ("ricorso contro una multa dell'autovelox", [428, 476, 477, 478]),
    ("Ricorso al Prefetto contro un verbale del codice della strada", [478, 476, 477]),
    ("Clausola arbitrale da inserire in un contratto", [450, 451]),
    ("domanda di ammissione al passivo della liquidazione giudiziale", [479]),
    ("Contratto di franchising", [391]),
    ("Verbale di assemblea condominiale", [393]),
    ("Lettera di diffida e messa in mora per un pagamento", [394]),
    ("Ricorso per separazione consensuale", [409, 115]),
    ("Ricorso per divorzio congiunto", [408, 473, 115]),
    ("Atto di citazione per usucapione di un terreno", [410]),
    ("citazione per divisione ereditaria tra fratelli", [413]),
    ("istanza di nomina del consulente tecnico d'ufficio", [38]),
    ("Memoria ex art. 183 cpc", [25]),
    ("Istanza di sequestro conservativo", [68]),
    ("Istanza di ammissione al patrocinio a spese dello Stato", [422, 475]),
    ("Ricorso per equa riparazione per irragionevole durata del processo, legge Pinto", [128]),
    ("Accordo transattivo per chiudere una vertenza di lavoro", [469, 471, 470]),
    ("Revoca delle dimissioni", [463]),
    ("Istanza di liberazione anticipata del detenuto", [260]),
]

# (request, acceptable catalog indices - empty when nothing is a sensible start,
#  index of the new template that fits exactly)
NEW = [
    ("Scrivi un contratto di riporto di titoli", [], 39),
    ("Mi serve una clausola di bring along per lo statuto della società", [], 22),
    ("Contratto di comodato d'uso di un impianto fotovoltaico", [], 26),
    ("Verbale del consiglio di amministrazione che nomina il comitato esecutivo", [], 96),
    ("Contratto di anticipazione bancaria con pegno su merci", [], 2),
    ("Atto di costituzione di pegno da parte di un terzo", [], 13),
    ("Verbale di assemblea straordinaria di spa per sostituire i liquidatori", [], 98),
    ("Contratto di cessione del credito pro solvendo", [], 36),
    ("Contratto di appalto per lavori di ristrutturazione con superbonus", [], 35),
    ("Lettera al datore di lavoro per comunicare il congedo parentale", [], 33),
    ("Domanda di congedo straordinario per assistere il genitore disabile", [], 45),
    ("Ricorso per ricusazione del giudice", [], 91),
    ("Reclamo contro il diniego di omologa degli accordi di ristrutturazione dei debiti", [], 78),
    ("Ricorso contro la cartella di pagamento dell'Agenzia delle Entrate", [], 84),
    ("Memoria di costituzione dell'ufficio contro il ricorso sull'avviso di accertamento basato su indagini bancarie", [], 67),
    ("Istanza del custode per la liquidazione del compenso nell'espropriazione immobiliare", [], 54),
    ("Contratto di vendita di spazi pubblicitari su un sito web", [], 40),
    ("Verbale di consegna e accettazione delle opere realizzate dall'appaltatore", [], 99),
    ("Lettera di assunzione con informativa privacy per il dipendente", [], 64),
    ("Clausola elastica nel contratto di lavoro part-time", [], 24),
    ("Istanza di revoca della confisca di prevenzione", [], 57),
    ("Ricorso per la risoluzione del contratto di affitto agrario per morosità", [], 90),
    ("Istanza di rimessione in termini del contumace", [], 59),
    # near: a generic catalog template is a reasonable starting point
    ("Atto di citazione per impugnare una delibera dell'assemblea condominiale", [419, 9, 23], 8),
    ("Atto di citazione per risarcimento danni da immissioni rumorose del vicino", [411, 9, 23], 10),
    ("Atto di precetto su titolo provvisoriamente esecutivo", [97, 430], 14),
    ("Invito all'appaltatore a eliminare i difetti dell'opera", [394], 50),
    ("Denuncia-querela per mancato versamento dell'assegno di mantenimento", [262, 264], 41),
    ("Dichiarazione di costituzione dell'ente nel processo ai sensi del d.lgs. 231/2001", [279], 42),
    ("Eccezione di difetto di giurisdizione nel processo penale", [202], 47),
    ("Ricorso per cassazione contro la sentenza di patteggiamento", [189, 483, 245], 88),
    ("Memoria difensiva del convenuto nel rito del lavoro", [461], 70),
    ("Nomina a difensore del responsabile civile con procura speciale", [147, 444, 141, 263], 72),
]

UNRELATED = [
    "Scrivi un testamento olografo",
    "Prepara un contratto di mutuo ipotecario",
    "Mi serve l'atto costitutivo di una srl",
    "Regolamento aziendale sull'uso dei dispositivi informatici",
    "Contratto di agenzia commerciale con esclusiva di zona",
    "Atto di donazione di un immobile al figlio",
    "Privacy policy per un sito e-commerce",
    "Contratto di sponsorizzazione sportiva",
    "Ricorso per l'adozione di un minore",
    "Statuto di un'associazione sportiva dilettantistica",
    "Patto parasociale di prelazione tra soci",
]
