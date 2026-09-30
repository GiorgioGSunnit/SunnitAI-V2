# DEJURE templates → chatbot catalog

Scripts that turn the DEJURE PDF formulari into fill-ready Word templates and
catalog entries for `/opt/chatbot/data/system_templates/`. They run on a
Windows PC; only `enrich_run.py` needs the model server.

Working files (logs, name maps, the rename/drop plan, model results) live
**outside the repo**, in `C:\Users\anton\Downloads\catalog_enrich` — override
with the `DEJURE_DATA` environment variable. Source PDFs and DOCX folders are
under `C:\Users\anton\Downloads\downloads\downloads`.

## Conversion rule (agreed with the product owner, Sept 2026)

Keep the document exactly as the PDF, turn blanks (`....`, `…`, `[…]`, `___`)
into `[DA COMPILARE]`, remove only the DEJURE header and footer (title line,
"Autori:", collection label, logo, copyright line). Notes stay as plain text
with their `[n]` numbers. Nothing else is removed.

## Pipeline, in order

| Step | Script | In → out |
|---|---|---|
| Convert | `convert_dejure_templates.py` | PDF → DOCX (pdf2docx + layout repairs). The full batch was converted by a colleague's version of this script, which adds table handling. |
| Fix 1 | `fix_converted_all.py` | `converted_all` → `converted_all_fixed`: spaces lost by pdf2docx (after accents, at italic runs; added only where the PDF has them), zip-garbled filenames, split/double accents, empty "Autori:" lines. Log `fix_converted_all.csv` maps new → original names. |
| Fix 2 | `fix_titles_headings.py` | → `converted_all_final`: removes the DEJURE header title line, puts back the act's own heading where the converter deleted it as a duplicate. |
| Fix 3 | `fix_manual.py` | in place: three files the general steps cannot repair (Il Piano genitoriale signature section, two headings). Re-run after any re-run of step 2. |
| Mislabels | `check_pdf_titles.py` → `classify_mismatch.py` → `resolve_mismatch.py` → `list_redownload.py` | Finds PDFs whose content is a different form than their name; writes `mismatch_plan.csv` (rename / drop) and the list of forms never downloaded. |
| Catalog 1 | `prepare_catalog_inputs.py` | Everything that needs no model: which files go in, label, area, filename, sections. → `enrich_inputs.json` |
| Catalog 2 | `enrich_run.py` | Model calls only (description, category, up to 30 fields). Standard library only, resumable: re-run the same command to continue. → `results.jsonl` |
| Catalog 3 | `build_catalog.py --cap 30` | Production catalog + fields fix + corrected pilot files − replaced entries + new entries. Stages `C:\tplstage\docs\` and `C:\tplstage\catalog_enriched.json`, writes `review_nuovi_modelli.csv`. |

Pilot files (the first 103) go through fix 1 and 2 with `pilot_pdfmap.py` and the
`FIX_SRC / FIX_DST / FIX_LOG / FIX_PDFMAP` variables (see the scripts' docstrings).

## Checks

- `verify_converted_all.py` (`VERIFY_FIXED=final` for the final folder): opens, words vs PDF, header/footer, logos, blanks, accents.
- `glued_any.py`: words glued without a space, confirmed against the PDF's word pairs.
- `heading_check.py` (`HEADING_DIR=converted_all_final`): act headings lost.
- `find_lost_passages.py --all 20`: PDF passages truly missing, header excluded.
- `word_check.ps1`: opens every file in real Word (read-only). Word must be closed.

## Decisions baked in

- The 103 pilot titles are skipped (already live); 2 original entries are replaced
  (`REPLACE_OLD`), 10 same-title forms from other areas are kept with the area in
  the label (`KEEP_BOTH`) — both in `prepare_catalog_inputs.py`.
- Agrarian and condominium/lease collections default to civil procedure; their
  letters, contracts and clauses go to "Contratti e Atti Stragiudiziali".
- Field limit 30: signature fields are dropped first, then the list is cut.
