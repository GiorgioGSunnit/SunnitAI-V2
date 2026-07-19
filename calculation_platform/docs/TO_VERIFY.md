# TO_VERIFY — valori da verificare contro le fonti ufficiali

Checklist unica di ogni valore numerico che un umano deve verificare contro
la fonte ufficiale prima dell'uso operativo del modulo. Dopo la verifica:
aggiornare il valore se necessario, impostare `verified: true` /
`placeholder: false` dove presenti, stampare `last_verified_at`, e
aggiornare i valori attesi dei golden test corrispondenti (cercare
`verified_against: TODO` nei test e nei pack).

## parameters/legal_it/legal_interest_rates.yml

- [ ] Saggio legale 2024 = 2,50% — Decreto MEF 29/11/2023, G.U. Serie
      Generale n. 288 dell'11/12/2023.
- [ ] Saggio legale 2025 = 2,00% — Decreto MEF 10/12/2024, G.U. Serie
      Generale n. 294 del 16/12/2024.
- [ ] Saggio legale 2026 = 1,60% — Decreto MEF 10/12/2025, G.U. Serie
      Generale n. 289 del 13/12/2025 (verificare numero e data esatti della
      G.U. e la percentuale).

## parameters/legal_it/irpef_brackets.yml

- [ ] Scaglioni 2024–2025: 23% fino a 28.000, 35% fino a 50.000, 43% oltre —
      art. 1 D.Lgs. 216/2023.
- [ ] Scaglioni 2026: aliquota intermedia ridotta al 33% (28.000–50.000) —
      Legge di Bilancio 2026 (Legge 30/12/2025 n. 199): verificare numero
      legge, G.U. e che la riduzione 35%→33% sia effettivamente in vigore
      per il periodo d'imposta 2026 con questi estremi di scaglione.

## parameters/legal_it/contributo_unificato.yml

- [ ] Tutte le fasce di valore e gli importi del contributo unificato —
      art. 13 DPR 115/2002 nel testo VIGENTE (verificare su Normattiva; gli
      importi sono stati piu volte rimodulati). Verificare anche le
      maggiorazioni per appello (+50%) e cassazione (+100%).

## parameters/legal_it/foi_indices.yml — SEGNAPOSTO

- [ ] TUTTI i mesi presenti sono segnaposto sintetici (`placeholder: true`):
      2024-11 = 100.0, 2024-12 = 100.5, 2025-12 = 102.0, 2026-02 = 102.5.
      NON sono dati ISTAT.
- [ ] Caricare in blocco la serie storica ufficiale ISTAT "FOI senza
      tabacchi" (indici mensili, base corrente) dal sito ISTAT, sostituendo
      i segnaposto e completando ogni mese necessario ai calcoli.
- [ ] Verificare la convenzione del mese finale contro il calcolatore
      ufficiale https://rivaluta.istat.it (stesso coefficiente tra due mesi
      campione).

## parameters/legal_it/dm55_compensi.yml — SEGNAPOSTO

- [ ] Scaglione 26.000,01–52.000: i quattro valori medi (studio 2000,
      introduttiva 1500, istruttoria 5000, decisionale 3500) sono SINTETICI.
      Sostituire con i valori medi della Tabella 2 (giudizi ordinari di
      cognizione innanzi al Tribunale) del DM 55/2014 come sostituita dal
      DM 147/2022 (G.U. n. 236 dell'08/10/2022).
- [ ] Popolare gli altri scaglioni (attualmente stub con
      `valore_medio: null`): fino a 1.100; 1.100,01–5.200; 5.200,01–26.000;
      52.000,01–260.000; 260.000,01–520.000; oltre 520.000 — per tutte e
      quattro le fasi.
- [ ] Verificare i limiti di aumento/riduzione codificati nella strategia
      (`app/strategies/dm55_fees.py`): aumento max 80%, riduzione max 50%,
      riduzione fino al 70% per la sola fase istruttoria — art. 4 DM 55/2014
      come modificato dal DM 147/2022.
- [ ] Verificare la catena accessori: +15% rimborso spese generali (art. 2
      DM 55/2014), +4% CPA, +22% IVA (aliquota ordinaria vigente).

## Golden test da riverificare dopo il caricamento dei dati reali

Ogni docstring contiene il calcolo a mano completo e la riga
`verified_against: TODO (official calculator/source check pending)`.

- [ ] `tests/test_rivalutazione_istat.py::test_rivalutazione_golden_case_with_placeholder_months`
      — ricalcolare con gli indici FOI reali e confrontare con
      rivaluta.istat.it.
- [ ] `tests/test_rivalutazione_interessi_1712.py::test_single_year_slice`
      e `::test_multi_year_with_rate_change_and_leap_year` — ricalcolare
      con gli indici FOI reali; la struttura (slice, giorni, divisori
      365/366, criterio della media) resta invariata.
- [ ] `tests/test_compensi_dm55.py::test_full_chain_multi_fase_golden_case`
      e `::test_single_fase_chain` — ricalcolare con i valori medi
      ministeriali reali.
- [ ] Gli `examples` nei pack `rivalutazione_istat.yml`,
      `rivalutazione_interessi_1712.yml`, `compensi_dm55.yml` (stessi numeri
      dei golden test).
