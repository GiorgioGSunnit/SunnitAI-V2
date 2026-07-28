# TO_VERIFY — valori da verificare contro le fonti ufficiali

Checklist dei valori numerici verificati contro la fonte ufficiale. Stato al
2026-07-24 (pass di verifica delle fonti primarie).

Legenda: [x] verificato contro fonte primaria e caricato; [~] valore reale
caricato ma in attesa della spunta finale di un umano contro l'artefatto
primario; [ ] da fare.

Nota: verifica della FONTE (il valore corrisponde all'artefatto ufficiale) e
verifica dell'IMPLEMENTAZIONE (i test passano) sono distinte. Un test verde
NON implica che un valore sia verificato contro la fonte.

## parameters/legal_it/legal_interest_rates.yml — VERIFICATO

- [x] Serie completa 2021-2026, ciascun anno con decreto MEF e URL diretto G.U.:
      - 2021 = 0,01% — DM 11/12/2020, G.U. 310 del 15/12/2020 (20A06997)
      - 2022 = 1,25% — DM 13/12/2021, G.U. 297 del 15/12/2021 (21A07417)
      - 2023 = 5,00% — DM 13/12/2022, G.U. 292 del 15/12/2022 (22A07140)
      - 2024 = 2,50% — DM 29/11/2023, G.U. 288 del 11/12/2023 (23A06669)
      - 2025 = 2,00% — DM 10/12/2024, G.U. 294 del 16/12/2024 (24A06721)
      - 2026 = 1,60% — DM 10/12/2025, G.U. 289 del 13/12/2025 (25A06705)
      Ogni citazione ha `url` diretto e `last_verified_at`.

## parameters/legal_it/foi_indices.yml — PARZIALE

- [x] Modello dati esteso con `base_year` per ogni voce e `base_links` per il
      raccordo tra basi (2025->2015 = 1,214, fonte ISTAT). La strategia rilinka
      le basi diverse prima di dividere (mai divisione cross-base implicita).
- [x] Giugno 2026 = 102,8 (base 2025), fonte primaria ISTAT — ultimo mese
      pubblicato al 2026-07-24 (luglio 2026 non ancora pubblicato).
- [~] 2021 gen-giu (102,9 / 103,0 / 103,3 / 103,7 / 103,6 / 103,8), base 2015 —
      valori coerenti con la serie ISTAT (fonte secondaria rivaluta.it),
      `verified: false` in attesa della spunta contro l'artefatto ISTAT.
- [ ] Ingestione in blocco della serie storica MENSILE completa ISTAT (tutti i
      mesi necessari, entrambe le basi) — attualmente solo i mesi sopra sono
      caricati; i mesi assenti falliscono in modo strutturato.
- [ ] Verificare la convenzione del mese finale contro https://rivaluta.istat.it.
      Nota di policy: il coefficiente resta a precisione piena (decisione
      confermata) — il calcolatore ISTAT arrotonda a 3 decimali (es. 1208,00 vs
      1208,12 su 1000 EUR marzo 2021 -> giugno 2026).

## parameters/legal_it/dm55_compensi.yml — VERIFICATO (fonte)

- [x] I 24 valori (sei scaglioni fino a 520.000 EUR x quattro fasi) della
      Tabella 2 (DM 147/2022) verificati contro l'allegato ufficiale in G.U. —
      `verified: true`. I valori non vanno modificati.
- [x] Regola del rimborso spese generali +15% (art. 2 DM 55/2014) verificata.
- [x] Codice redazionale G.U. del DM 147/2022 corretto in `22G00157` (era
      erroneamente `22G00156`) in entrambi i file (pack e parametri).
- [x] Limiti di adeguamento (art. 4, come modificato dal DM 147/2022) corretti:
      aumento max 50% (era erroneamente 80%), riduzione max 50%; RIMOSSA la
      speciale riduzione fino al 70% per la sola fase istruttoria. Aggiornati
      strategia, validazione, messaggi d'errore, metadata del pack e test.
- [x] CPA (+4%) e IVA (+22%) resi input espliciti (`applica_cpa`,
      `applica_iva`, default true). Non sono più assunzioni universali
      nascoste: l'ordine è compenso -> +15% spese generali -> +4% CPA (se
      applica_cpa) -> +22% IVA (se applica_iva); flag, aliquote e importi
      intermedi compaiono nel risultato e nell'audit; il default (quando
      l'input è omesso) è registrato come assunzione. Nessuna inferenza da
      linguaggio naturale.
- [ ] Cause oltre 520.000 EUR: NON supportate finché la maggiorazione
      discrezionale "fino al 30%" (art. 6) non sia modellata deliberatamente;
      la strategia restituisce un errore "fascia non supportata" (decisione
      confermata).

## Modulo penale (BOZZE 0.1-draft) — VALIDAZIONE LEGALE RICHIESTA

Le cornici e le multe sono trascritte da fonti/norme ma NON validate da un
penalista; la meccanica (artt. 63-66, 69, 442) è testata, il contenuto
giuridico no. Ogni calcolatore emette il warning `draft_not_validated`.

- [ ] Cornice `furto_aggravato_draft` (art. 625: 2-6 anni + multa 927-1.500):
      verificare contro il testo vigente; NON copre art. 624-bis (furto in
      abitazione, cornice autonoma 4-7 anni) né le pluriaggravate.
- [ ] Cornice `rapina_aggravata_draft` (art. 628 co. 3: 7-20 anni): figura
      trascritta dalle asserzioni già presenti nel codice, da verificare —
      le pene della rapina sono state più volte riformate (L. 36/2019); multa
      del co. 3 NON calcolata; co. 4 (pluriaggravata) non coperto.
- [ ] Multa di `furto_pena_draft` (154-516) e `rapina_pena_draft` (927-2.500):
      riportata SOLO nella cornice edittale base (art. 24 c.p.), non adeguata
      per circostanze, tetti art. 66 n. 3 o rito. Modellare le regole
      specifiche della pena pecuniaria richiede validazione legale.
- [ ] Catalogo delle circostanze: ancora conteggi generici (±1/3), non la
      qualificazione giuridica gated su validazione di un penalista.

## Parità simulazione/produzione (match ambiguo) — ALLINEATA PARZIALMENTE

- [x] Su match AMBIGUO **forte** (tutti i candidati a pari punteggio superano
      da soli la soglia di auto-routing) la produzione ora chiede all'utente di
      scegliere, come già faceva la simulazione. La scelta si accetta per
      numero, per `calculator_id` o per nome visualizzato ed è persistita in
      `pending_calculation` (fase `choose_calculator`). Test:
      `tests/test_calculation_route.py::test_strong_ambiguity_asks_the_user_to_choose`
      e `::test_a_choice_is_accepted_by_number_id_or_name`.
- [ ] Su match ambiguo **debole** (pari punteggio sotto la soglia) la
      produzione continua a instradare verso la RAG normale, di proposito:
      chiedere lì trasformerebbe qualunque menzione incidentale di un tema
      giuridico in un menu. Evidenza:
      `::test_weak_ambiguity_still_falls_back_to_normal_rag_without_prompting`.
      Se il prodotto vuole chiedere anche lì, serve una decisione sulla UX.

## Comparatori (confronto polizze / gas e luce) — DECISIONI DI BUSINESS APERTE

L'aritmetica dei comparatori è esatta e verificata dai test; la **validità del
modello di punteggio** no. Restano da decidere con il business:

- [ ] Pesi delle componenti. Attuali (dimostrativi): polizze 0,50 costo / 0,25
      coperture / 0,15 condizioni / 0,05 servizi / 0,05 recensioni; gas e luce
      0,60 costo / 0,20 condizioni / 0,10 servizi / 0,10 recensioni. La somma
      esatta a 1 è imposta dal validatore, i valori no.
- [ ] Punti delle componenti `points`/`rules` (es. kasko 20, infortuni 15,
      franchigia oltre 300 euro -10, penale di recesso -0,5 punti per euro):
      scelti per la demo, mai concordati.
- [ ] `massimale` delle polizze: raccolto ma NON usato nel punteggio. Serve una
      scala verificata (per fascia di massimale? rapporto col premio?) prima di
      convertirlo in punti; inventarne una falserebbe il confronto.
- [ ] Tolleranza di parità: attuale `0.50` punti su 100 in entrambi i pack,
      configurabile per pack (`formula.tie_tolerance`). Da confermare che sia
      la soglia oltre la quale il business considera una differenza reale.
- [ ] Gas e luce: il costo annuo esclude IVA, accise, oneri di sistema,
      trasporto/distribuzione/misura, fasce F1/F2/F3 e le variazioni dopo il
      primo anno. Includerli richiede parametri ARERA verificati e versionati
      per data, che la piattaforma non ha: finché non ci sono, l'esclusione
      resta dichiarata in `exclusions` e nei warning.
- [ ] Eta del conducente e storico sinistri sono raccolti, validati e tracciati
      ma non incidono sul punteggio (identici per tutte le offerte). Se il
      business vuole usarli servirà un modello di rischio vero, non un malus
      applicato in egual misura a tutti i candidati.

## Test locali — deterministici, senza rete/LLM, DB temporaneo

- [x] `test_compensi_dm55.py` — valori reali Tabella 2; limiti ±50% (accettato/
      rifiutato al confine), ex-70% istruttoria rifiutato, confini di scaglione
      ±1 cent, oltre 520.000 non supportato, tutte le combinazioni CPA/IVA.
- [x] `tests/corpus/scenarios.yml` + `test_corpus.py` — corpus deterministico
      (engine_verified / local_api_verified / integration_simulated).
- [x] `test_api.py` — contratto API locale, incluso il nuovo flag CPA/IVA.
- [x] `test_parity.py` — evidenza parità lato simulazione.
- [x] `test_rivalutazione_istat.py` / `test_rivalutazione_interessi_1712.py` —
      fixture controllato (`support.py`) + casi FOI stessa-base e cross-base.
- [x] `test_extraction_regressions.py` — difetti di estrazione/routing #1-#8.
