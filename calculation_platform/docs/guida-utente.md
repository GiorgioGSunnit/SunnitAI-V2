# Guida utente della Calculation Platform

## 1. Introduzione

La Calculation Platform è un motore di calcolo deterministico. Il chiamante seleziona un calcolatore e fornisce gli input; il motore valida i dati, risolve gli eventuali parametri versionati per data e applica la strategia dichiarata nel relativo formula pack YAML.

I calcoli numerici sono eseguiti in `Decimal`. Le formule sono dichiarative e valutate dal codice della piattaforma: nessun LLM esegue i calcoli. Nei payload JSON i valori `Decimal` sono serializzati come stringhe esatte, per esempio `"11060.00"`, e non come numeri in virgola mobile.

Una risposta riuscita contiene, oltre al `result`:

- `citations`: riferimenti normativi o ufficiali;
- `warnings`: limiti, esclusioni, dati non verificati o altre cautele;
- `assumptions`: assunzioni dichiarate dal calcolatore e valori predefiniti applicati;
- `defaults_applied`: gli stessi valori predefiniti in forma leggibile dalle macchine, uno per voce, con il percorso dell'input (`{"path": "polizze[0].franchigia", "value": "0"}`);
- `exclusions`: cio che il calcolatore dichiara esplicitamente di NON coprire, come lista strutturata;
- `steps`: passaggi numerati della derivazione;
- `inputs_used`, `parameters_used` e `derived_values`: dati effettivamente usati e relativa provenienza.

Il chiamante deve sempre controllare `status`, `warnings` e `assumptions`, non soltanto il valore in `result`.

## 2. Avvio del servizio

Dalla radice del repository:

```bash
.venv/bin/uvicorn calculation_platform.app.main:app --reload --port 8802
```

URL base:

```text
http://localhost:8802
```

La documentazione Swagger interattiva è disponibile su:

```text
http://localhost:8802/docs
```

## 3. Endpoint principali

| Metodo e percorso | Uso |
|---|---|
| `GET /calculators` | Restituisce il catalogo sintetico dei calcolatori: `id`, nome, categoria, descrizione, parole chiave e alias. |
| `GET /calculators/{id}` | Restituisce la definizione completa del calcolatore, inclusi input, parametri, formula, esempi e il flag `requires_period`. Un ID sconosciuto produce HTTP 404. |
| `POST /match` | Esegue un abbinamento deterministico per parole chiave e alias. Accetta `{"query": "..."}` e restituisce stato, candidati ordinati, termini riconosciuti e input ancora necessari; non esegue il calcolo. |
| `POST /calculate` | Valida ed esegue il calcolo, assegna un `request_id` se assente e tenta di salvarne richiesta e risultato nello storico. Gli errori applicativi sono restituiti nel `CalculationResult` con `status: "error"`. |
| `GET /tool-schemas` | Genera gli schemi di function calling per tutti i calcolatori. Ogni elemento contiene nome del tool, `calculator_id`, descrizione e `input_schema`. |
| `GET /calculations` | Elenca i calcoli recenti. Il parametro query `limit` vale 50 per impostazione predefinita ed è limitato a un massimo di 200. |
| `GET /calculations/{id}` | Restituisce un calcolo salvato con riepilogo, richiesta originale e risultato completo. Un ID assente produce HTTP 404. |
| `GET /calculations/{id}/report` | Restituisce un report HTML stampabile del calcolo salvato. |
| `POST /calculations/{id}/replay` | Riesegue la richiesta salvata senza creare un nuovo record e restituisce `stored_result`, `replayed_result` e `matches`. Il confronto copre risultato, passaggi, input e parametri usati, valori derivati, stato ed errori. |

## 4. Come si effettua un calcolo

Inviare una richiesta JSON a `POST /calculate`.

### Campi della richiesta

| Campo | Obbligatorio | Significato |
|---|---:|---|
| `calculator_id` | sì | ID esatto del calcolatore, ricavabile da `GET /calculators`. |
| `inputs` | dipende dal calcolatore | Oggetto con gli input dichiarati nel formula pack. I tipi supportati dalle definizioni correnti sono `decimal`, `integer`, `boolean`, `string`, `date` ISO `YYYY-MM-DD`, `string_list` e `object_list` (lista di oggetti, usata dai comparatori: ogni elemento e validato contro gli `item_fields` dichiarati). |
| `tax_year` | no | Anno d'imposta. Per la risoluzione dei parametri viene convertito nella data del 31 dicembre dell'anno, salvo che sia presente `as_of_date`. |
| `as_of_date` | no | Data ISO `YYYY-MM-DD` usata per selezionare i parametri versionati. Ha precedenza su `tax_year`. Se entrambi mancano, la piattaforma usa la data corrente. |
| `period` | solo se richiesto | Oggetto `{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}`. Attualmente è obbligatorio soltanto per `legal_it.legal_interest`. |
| `caller_supplied_values` | no | Override espliciti dei parametri, indicizzati con il nome dichiarato dal calcolatore. La precedenza sulla tabella è applicata dai calcolatori che usano il resolver generale e, in modo esplicito, da `legal_it.legal_interest`; `parameters_used` ne registra l'origine `caller_supplied`. Le strategie specialistiche FOI, DM 55 e ravvedimento non leggono questo campo nel codice corrente. |
| `confirm_assumptions` | no | `false` predefinito. Dichiara che il chiamante ha visto e accettato le assunzioni e i default che la richiesta produrra. Non modifica alcun numero e non rimuove alcuna assunzione: nei confronti sposta soltanto `result.comparison.provisional_status` da `provisional_unconfirmed` a `confirmed_with_assumptions`. |
| `options` | no | Oggetto con `explain` (predefinito `true`), `rounding` (predefinito `null`) e `require_sources` (predefinito `false`). Nel codice corrente questi valori sono accettati dallo schema ma non sono letti dal motore e non modificano il calcolo. |

Lo schema accetta inoltre `request_id`, `jurisdiction` e `tenant_id`. `request_id` è facoltativo e viene generato dall'endpoint se manca. `jurisdiction` non è usato dal flusso di calcolo corrente; `tenant_id` è riservato al futuro supporto multi-tenant e non è applicato.

### Esempio completo: IRPEF lorda 2026

Il formula pack `legal_it.irpef` dichiara l'esempio di un reddito imponibile di 42.000 euro per il 2026, con risultato atteso pari a 11.060,00 euro.

Richiesta:

```json
{
  "request_id": "guida-irpef-2026",
  "calculator_id": "legal_it.irpef",
  "inputs": {
    "taxable_income": 42000
  },
  "tax_year": 2026
}
```

Forma completa della risposta riuscita:

```json
{
  "request_id": "guida-irpef-2026",
  "calculator_id": "legal_it.irpef",
  "status": "success",
  "result": {
    "gross_tax": "11060.00"
  },
  "formula_used": "legal_it.irpef",
  "formula_version": "1",
  "raw_inputs": {
    "taxable_income": 42000
  },
  "inputs_used": {
    "taxable_income": "42000"
  },
  "parameters_used": {
    "brackets": {
      "name": "brackets",
      "value": [
        {
          "up_to": 28000,
          "rate": 0.23
        },
        {
          "up_to": 50000,
          "rate": 0.33
        },
        {
          "up_to": null,
          "rate": 0.43
        }
      ],
      "origin": "parameter_store",
      "parameter_id": "legal_it.irpef_brackets",
      "source": "Legge 30 dicembre 2025, n. 199 — Legge di Bilancio 2026 (G.U. Serie Generale n. 301 del 30/12/2025, S.O. n. 42)",
      "effective_from": "2026-01-01",
      "effective_to": "2026-12-31",
      "official": true,
      "last_verified_at": null,
      "citations": [
        {
          "reference": "Legge 30 dicembre 2025, n. 199 (Legge di Bilancio 2026) — riduzione della seconda aliquota IRPEF dal 35% al 33% dal periodo d'imposta 2026",
          "source_name": "Gazzetta Ufficiale Serie Generale n. 301 del 30/12/2025, Supplemento Ordinario n. 42",
          "publisher": "Gazzetta Ufficiale della Repubblica Italiana",
          "publication_date": "2025-12-30",
          "url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:2025-12-30;199",
          "official": true
        }
      ]
    }
  },
  "date_resolution": {
    "as_of_date": "2026-12-31",
    "source": "derived_from_tax_year",
    "tax_year": 2026
  },
  "derived_values": {},
  "steps": [
    {
      "step": 1,
      "type": "bracket",
      "bracket_up_to": "28000",
      "rate": "0.23",
      "taxable_in_bracket": "28000",
      "tax_in_bracket": "6440.00"
    },
    {
      "step": 2,
      "type": "bracket",
      "bracket_up_to": "50000",
      "rate": "0.33",
      "taxable_in_bracket": "14000",
      "tax_in_bracket": "4620.00"
    }
  ],
  "citations": [
    {
      "reference": "Art. 11 D.P.R. 22 dicembre 1986, n. 917 (TUIR) — Determinazione dell'imposta",
      "source_name": "Testo Unico delle Imposte sui Redditi",
      "publisher": "Normattiva — Istituto Poligrafico e Zecca dello Stato",
      "publication_date": "1986-12-22",
      "url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.del.presidente.della.repubblica:1986-12-22;917",
      "official": true
    },
    {
      "reference": "Legge 30 dicembre 2025, n. 199 (Legge di Bilancio 2026) — riduzione della seconda aliquota dal 35% al 33%",
      "source_name": "Gazzetta Ufficiale Serie Generale n. 301 del 30/12/2025, Supplemento Ordinario n. 42",
      "publisher": "Gazzetta Ufficiale della Repubblica Italiana",
      "publication_date": "2025-12-30",
      "url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:2025-12-30;199",
      "official": true
    }
  ],
  "warnings": [
    {
      "code": "definition",
      "message": "This calculates gross national IRPEF only."
    },
    {
      "code": "definition",
      "message": "It does not include deductions, detrazioni, regional surcharge, municipal surcharge, tax credits, or special regimes."
    },
    {
      "code": "parameter_verification_missing",
      "message": "Il parametro ufficiale 'legal_it.irpef_brackets' non ha una verifica automatica registrata; ricontrollare la fonte prima dell'uso operativo."
    }
  ],
  "assumptions": [
    {
      "code": "definition",
      "message": "Assumes the taxable_income figure supplied is already net of any deductions the caller wants applied — this calculator does not compute deductions itself."
    }
  ],
  "errors": []
}
```

Lettura della risposta:

- `status` distingue il successo dall'errore applicativo.
- `result.gross_tax` è il risultato finale. È una stringa decimale esatta: il client deve elaborarla con un tipo decimale, non con `float`.
- `raw_inputs` conserva quanto inviato; `inputs_used` mostra i valori dopo coercizione e applicazione dei default.
- `parameters_used` documenta valore, origine, periodo di efficacia e citazioni della tabella applicata.
- `citations` contiene le fonti dichiarate dal calcolatore; le citazioni dentro `parameters_used` documentano invece lo specifico parametro risolto.
- `warnings` non rende fallito il calcolo, ma segnala limiti sostanziali o di verifica. Nell'esempio l'IRPEF è soltanto lorda nazionale e la tabella ufficiale non ha un timbro `last_verified_at`.
- `assumptions` esplicita le ipotesi del modello e gli eventuali default applicati.
- `steps` ricostruisce il calcolo: 28.000 × 23% = 6.440 e 14.000 × 33% = 4.620.

## 5. Catalogo dei calcolatori

La colonna “Periodo” indica se è richiesto il campo di primo livello `period`; le date dichiarate come normali input non cambiano tale flag.

| ID | Nome | Input obbligatori | Periodo | Copertura dei parametri |
|---|---|---|:---:|---|
| `business.invoice_total` | Invoice total (net + VAT - discount) | `net_amount` (`decimal`); `vat_rate` (`decimal`, rate) | no | Nessuna tabella usata: l'aliquota IVA è fornita dal chiamante. La tabella `business/vat_rates.yml` è riservata a uso futuro. |
| `business.loan_payment` | Loan monthly payment (amortized) | `principal` (`decimal`); `annual_rate` (`decimal`, rate); `months` (`integer`, mesi) | no | Nessun parametro versionato. |
| `business.confronto_gas_luce` | Confronto offerte gas e luce (costo annuo + punteggio 0-100) | `consumo_annuo_kwh` (`decimal`, kWh); `consumo_annuo_smc` (`decimal`, Smc); `offerte` (`object_list`, minimo 2, con `fornitore`, `prezzo_kwh_luce`, `prezzo_smc_gas` obbligatori per ciascuna offerta) | no | Nessun parametro versionato: prezzi e consumi sono forniti dal chiamante. Pesi dimostrativi. |
| `legal_it.compensi_dm55` | Compensi forensi (DM 55/2014, giudizi ordinari di cognizione) | `valore_causa` (`decimal`, EUR); `fasi` (`string_list`) | no | Tabella efficace dal 23/10/2022, ma solo lo scaglione 26.000,01–52.000 contiene valori sintetici; gli altri sono stub nulli. |
| `legal_it.contributo_unificato_civile` | Contributo unificato (processo civile ordinario) | Nessuno nello schema; operativamente serve `valore_causa` oppure `valore_indeterminabile: true`, salvo esenzione | no | Scaglioni efficaci dal 01/01/2013, senza data finale dichiarata. |
| `legal_it.furto_pena_draft` | [BOZZA] Cornice edittale — furto (art. 624 c.p.) | `aggravanti_comuni` (`integer`); `attenuanti_comuni` (`integer`) | no | Nessuna tabella; formula pack `0.1-draft`, non validato legalmente. Riporta anche la multa base (154-516 euro), non adeguata. |
| `legal_it.furto_aggravato_draft` | [BOZZA] Cornice edittale — furto aggravato (art. 625 c.p.) | `aggravanti_comuni` (`integer`); `attenuanti_comuni` (`integer`) | no | Nessuna tabella; formula pack `0.1-draft`, non validato legalmente. Cornice 2-6 anni + multa base 927-1.500 euro; l'aggravante va confermata da un legale. |
| `legal_it.imu` | IMU — imposta municipale sugli immobili | `rendita_catastale` (`decimal`, EUR); `moltiplicatore` (`decimal`); `aliquota` (`decimal`, rate) | no | Nessuna tabella: moltiplicatore e aliquota sono forniti dal chiamante. |
| `legal_it.inps_contributions` | Contributi previdenziali INPS (quota lavoratore e datore) | `retribuzione_lorda` (`decimal`, EUR); `aliquota_lavoratore` (`decimal`, rate); `aliquota_datore_lavoro` (`decimal`, rate) | no | Nessuna tabella: le aliquote sono fornite dal chiamante. |
| `legal_it.irpef` | IRPEF lorda (imposta sul reddito delle persone fisiche) | `taxable_income` (`decimal`, EUR) | no | Scaglioni 2024–2025 (23%/35%/43%) e 2026 (23%/33%/43%). |
| `legal_it.late_payment_interest` | Interessi di mora nelle transazioni commerciali (D.Lgs. 231/2002) | `capitale` (`decimal`, EUR); `tasso_riferimento_bce` (`decimal`, rate); `giorni` (`integer`, giorni) | no | Nessuna tabella: il tasso BCE del semestre è fornito dal chiamante. |
| `legal_it.legal_interest` | Interessi legali (simple interest at the legal rate) | `capital` (`decimal`, EUR) | sì | Saggi legali per 2024, 2025 e 2026; l'intero periodo richiesto deve essere coperto. |
| `legal_it.notice_indemnity` | Indennità sostitutiva del mancato preavviso | `retribuzione_mensile_globale` (`decimal`, EUR); `mesi_preavviso` (`decimal`, mesi) | no | Nessun parametro versionato; i mesi derivano dal CCNL e sono forniti dal chiamante. |
| `legal_it.omicidio_pena_draft` | [BOZZA] Cornice edittale — omicidio volontario (art. 575 c.p.) | `aggravanti_comuni` (`integer`); `attenuanti_comuni` (`integer`) | no | Nessuna tabella; formula pack `0.1-draft`, non validato legalmente. |
| `legal_it.rapina_pena_draft` | [BOZZA] Cornice edittale — rapina (art. 628 c.p.) | `aggravanti_comuni` (`integer`); `attenuanti_comuni` (`integer`) | no | Nessuna tabella; formula pack `0.1-draft`, non validato legalmente. Riporta anche la multa base (927-2.500 euro), non adeguata. |
| `legal_it.rapina_aggravata_draft` | [BOZZA] Cornice edittale — rapina aggravata (art. 628, co. 3 c.p.) | `aggravanti_comuni` (`integer`); `attenuanti_comuni` (`integer`) | no | Nessuna tabella; formula pack `0.1-draft`, non validato legalmente. Cornice 7-20 anni (da verificare); l'aggravante va confermata da un legale. |
| `legal_it.ravvedimento_operoso` | Ravvedimento operoso (omesso o tardivo versamento) | `tributo_non_versato` (`decimal`, EUR); `scadenza_originaria` (`date`); `data_pagamento` (`date`) | no | Regime sanzionatorio dal 01/09/2024; saggi legali disponibili per 2024–2026. Il periodo di ritardo deve essere interamente coperto. |
| `legal_it.registration_tax_leases` | Imposta di registro sui contratti di locazione | `annual_rent` (`decimal`, EUR); `years` (`decimal`, anni); `first_registration` (`boolean`) | no | Aliquota e minimo efficaci dal 26/04/1986, senza data finale dichiarata. |
| `legal_it.rivalutazione_interessi_1712` | Rivalutazione + interessi su debiti di valore (Cass. SS.UU. 1712/1995) | `importo` (`decimal`, EUR); `data_iniziale` (`date`); `data_finale` (`date`) | no | FOI disponibile solo per 2024-11, 2024-12, 2025-12 e 2026-02, tutti valori sintetici; saggi legali 2024–2026. |
| `legal_it.rivalutazione_istat` | Rivalutazione monetaria (indice ISTAT FOI) | `importo` (`decimal`, EUR); `data_iniziale` (`date`); `data_finale` (`date`) | no | FOI disponibile solo per 2024-11, 2024-12, 2025-12 e 2026-02, tutti valori sintetici. |
| `legal_it.termini_processuali_civili` | Termini processuali civili (computo a giorni) | `data_decorrenza` (`date`); `giorni` (`integer`) | no | Calendario delle festività nazionali dal 01/01/2025 al 31/12/2027. |
| `legal_it.tfr` | TFR — quota annua maturata e rivalutazione del fondo | `retribuzione_lorda_annua` (`decimal`, EUR) | no | Nessuna tabella: la variazione ISTAT è un input facoltativo del chiamante, con default 0. |

## 6. Confronti tra offerte (calcolatori comparatori)

`business.confronto_gas_luce` produce una **classifica**: a ciascuna offerta
viene assegnato un punteggio
0-100 composto da componenti pesate che sommano esattamente a 1.

L'aritmetica e esatta e verificabile passo per passo; il **modello di
punteggio** (quali componenti, con quali pesi) e ancora dimostrativo e va
concordato con il business. Sono due cose diverse e il risultato le tiene
separate: non trattare il punteggio come una misura oggettiva di mercato.

### Punteggio di costo relativo alla migliore offerta

Il costo e valutato rispetto alla **migliore** offerta del gruppo, non alla
peggiore: l'offerta piu economica vale 100 e le altre valgono
`costo_migliore / costo_offerta * 100`. Di conseguenza aggiungere un'offerta
piu cara non modifica i punteggi gia calcolati ne il loro ordine.

Casi limite risolti senza divisioni per zero: se tutti i costi sono zero
tutte le offerte valgono 100; se il minimo e zero e altre offerte hanno un
costo reale, le offerte a costo zero valgono 100 e le altre 0.

### Parita sostanziale

`result.comparison` accompagna sempre `ranking` e `best` (mantenuti per
compatibilita):

| Campo | Significato |
|---|---|
| `decision_status` | `clear_winner` oppure `effective_tie`. |
| `best_candidates` | Tutte le offerte entro `tie_tolerance` dal punteggio massimo **esatto**. |
| `score_gap` | Distacco tra la prima e la seconda, calcolato sui totali esatti e arrotondato solo per la visualizzazione. |
| `tie_tolerance` | Soglia dichiarata dal formula pack, predefinita `0.50`. |
| `provisional` | `true` se un default e stato applicato a un campo che incide sul punteggio. |
| `provisional_status` | `none`, `provisional_unconfirmed` oppure `confirmed_with_assumptions`. |
| `scoring_completeness` | Frazione 0-1 dei campi valutati effettivamente forniti dal chiamante. |
| `scoring_defaults_applied` | I soli default che incidono sul punteggio, con `path` e `value`. |
| `cost_basis` | Quale variabile derivata rappresenta il costo, per mostrarlo prima del punteggio. |

Quando `decision_status` e `effective_tie` la piattaforma emette un warning
esplicito e **nessuna offerta puo essere presentata come la migliore**: la
differenza rientra nel rumore del modello di punteggio.

### Dati assunti e conferma

`defaults_applied` elenca ogni default applicato in forma leggibile dalle
macchine (`{"path": "polizze[0].franchigia", "value": "0"}`), inclusi i campi
annidati delle singole offerte. Ogni voce della classifica porta inoltre un
blocco `data_quality` che distingue tre stati diversi:

- `provided_fields` — il chiamante ha dichiarato il valore (anche `false` o `0`);
- `assumed_fields` — la piattaforma ha applicato un default;
- `unknown_fields` — nessuno ha fornito il dato e non esiste un default.

Un `false` esplicito e un `false` assunto non sono la stessa affermazione e non
vengono mai confusi. I campi dichiarati ma non usati da alcuna componente non
riducono `scoring_completeness`.

`confirm_assumptions: true` nella richiesta registra soltanto che il chiamante
ha preso atto delle assunzioni: il risultato resta `provisional`, le assunzioni
restano tutte nel payload e i numeri non cambiano.

### Raccolta incrementale delle offerte

Nel percorso di produzione le offerte sono raccolte **una per messaggio**, con
fasi esplicite (dati comuni, offerte, riepilogo, conferma) conservate nello
stato della sessione. Le offerte gia raccolte non vengono ricostruite a ogni
turno: ripetere un'offerta col suo nome la corregge, "rimuovi <nome>" la
elimina, e il massimo accettato e 20 offerte.

### Cosa NON e incluso

Ogni comparatore dichiara le proprie esclusioni in `exclusions`, restituito da
`/calculate`, dai calcoli salvati, dal replay, dal report HTML e dalle risposte
del chatbot.

- Gas e luce: il costo annuo NON e l'importo della bolletta. Sono esclusi IVA,
  accise, oneri di sistema, trasporto/distribuzione/misura, fasce orarie
  F1/F2/F3 e le variazioni di prezzo dopo il primo anno. Non sono stati
  aggiunti perche la piattaforma non dispone di parametri verificati: stimarli
  produrrebbe un numero preciso e sbagliato.

## 7. Dati preliminari e coperture

### Dati non utilizzabili operativamente

- `legal_it.furto_pena_draft`, `legal_it.omicidio_pena_draft` e `legal_it.rapina_pena_draft` sono bozze dimostrative, versione `0.1-draft`, dichiarate come non validate legalmente. Non devono essere trattate come strumenti professionali o previsioni della sentenza.
- `parameters/legal_it/foi_indices.yml` contiene soltanto quattro mesi, tutti marcati `placeholder: true` e `verified: false`: novembre 2024, dicembre 2024, dicembre 2025 e febbraio 2026. I numeri sono sintetici e non sono dati ISTAT reali. I calcolatori `legal_it.rivalutazione_istat` e `legal_it.rivalutazione_interessi_1712` producono avvisi espliciti quando li usano; un mese assente produce `parameter_unresolved`, senza interpolazione.
- `parameters/legal_it/dm55_compensi.yml` è anch'esso un segnaposto non verificato. Soltanto lo scaglione 26.000,01–52.000 euro contiene valori medi sintetici per le quattro fasi; tutti gli altri scaglioni hanno `valore_medio: null` e producono `parameter_unresolved`. Anche un risultato nello scaglione popolato reca un avviso che lo dichiara non utilizzabile operativamente.

### Finestre temporali effettive

- IRPEF: anni d'imposta 2024, 2025 e 2026.
- Interessi legali: 2024, 2025 e 2026.
- Ravvedimento: tabella sanzionatoria per violazioni dal 1° settembre 2024; la componente interessi resta limitata ai saggi 2024–2026.
- Festività nazionali per i termini processuali: 2025–2027. Se il parametro è risolto ma il termine coinvolge date esterne alla copertura, la strategia emette un warning di verifica manuale; se la data usata per risolvere la tabella è fuori dalla sua efficacia, il calcolo fallisce.
- Contributo unificato: tabella dal 1° gennaio 2013 senza termine finale dichiarato.
- Imposta di registro sulle locazioni: aliquota e minimo dal 26 aprile 1986 senza termine finale dichiarato.
- DM 55: tabella formalmente efficace dal 23 ottobre 2022, ma con i limiti sintetici sopra indicati.

Le tabelle ufficiali presenti non riportano attualmente `last_verified_at`. Quando un parametro ufficiale proveniente dallo store è incluso in `parameters_used`, il motore aggiunge un warning `parameter_verification_missing`. Una tabella senza valore efficace per la data richiesta causa normalmente `parameter_unresolved`; per i calcoli a segmenti, anche una copertura soltanto parziale del periodo viene rifiutata.

Controllare sempre `warnings`, `assumptions`, `date_resolution` e `parameters_used` prima di usare un risultato in attività legali, fiscali o amministrative.

## 8. Gestione degli errori

Con una richiesta top-level valida, `POST /calculate` converte gli errori del motore in un `CalculationResult`. L'endpoint risponde HTTP 200 anche quando il calcolo ha `status: "error"`; il client deve quindi leggere il campo `status`.

Forma generale:

```json
{
  "request_id": "id-della-richiesta",
  "calculator_id": "legal_it.irpef",
  "status": "error",
  "result": {},
  "formula_used": null,
  "formula_version": null,
  "raw_inputs": {},
  "inputs_used": {},
  "parameters_used": {},
  "date_resolution": null,
  "derived_values": {},
  "steps": [],
  "citations": [],
  "warnings": [],
  "assumptions": [],
  "errors": [
    {
      "code": "input_invalid",
      "message": "Messaggio leggibile",
      "details": {}
    }
  ]
}
```

Per input obbligatori mancanti, `details.missing_inputs` elenca tutti i nomi mancanti e `details.missing` fornisce la specifica utilizzabile dal client per chiedere i dati:

```json
{
  "request_id": "esempio-errore-irpef",
  "calculator_id": "legal_it.irpef",
  "status": "error",
  "errors": [
    {
      "code": "input_invalid",
      "message": "Missing required input(s): taxable_income",
      "details": {
        "missing_inputs": [
          "taxable_income"
        ],
        "missing": [
          {
            "name": "taxable_income",
            "type": "decimal",
            "required": true,
            "description": "Reddito imponibile IRPEF",
            "unit": "EUR",
            "min_value": 0
          }
        ]
      }
    }
  ]
}
```

Anche il `period` mancante è rappresentato in `details.missing_inputs` e in `details.missing`, con i due campi `start_date` ed `end_date`.

Codici da gestire:

| Codice | Significato e azione |
|---|---|
| `input_invalid` | Input mancante, non convertibile, fuori dai limiti o combinazione non valida. Usare `details.input`, `details.missing_inputs` e `details.missing` senza analizzare il testo di `message`. |
| `parameter_unresolved` | Parametro sconosciuto, data/mese non coperto, tabella incompleta o copertura parziale di un periodo. Correggere data/anno oppure aggiornare la tabella; non sostituire automaticamente un valore vicino. |
| `calculator_not_found` | `calculator_id` sconosciuto. `details.available` contiene gli ID caricati. |
| `calculator_not_applicable` | Il calcolatore non è applicabile alla data risolta. `details` indica limiti di applicabilità, data usata e relativa origine. |
| `strategy_execution_failed` | Errore semantico del calcolo, per esempio data iniziale successiva a quella finale o combinazione non supportata. Leggere i dettagli specifici. |
| `definition_invalid` | Formula pack non valido rilevato durante il caricamento del registro; è un problema di configurazione del servizio, non un dato correggibile dal chiamante. |
| `platform_error` | Errore generico della gerarchia della piattaforma. |

Un JSON che non rispetta nemmeno lo schema Pydantic della richiesta può invece essere respinto da FastAPI con HTTP 422, prima dell'esecuzione del motore. Gli HTTP 404 degli endpoint di dettaglio e storico usano il normale campo FastAPI `detail`, non la forma `CalculationResult`.
