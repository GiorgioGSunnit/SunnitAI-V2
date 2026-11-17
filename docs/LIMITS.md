# SunnitAI-V2: what it does not do, or does only partly

This page is for the product owner, the lawyer who scores the answers, and the engineers. It lists what the system
does not do, or does only in part, so its answers are read correctly. Every limit below comes from the code or its
comments. The reference `path:NAME` points to the constant, function or copy text that sets it. If a number is
given, it is the constant's value in this checkout.

Some behaviour depends on feature flags read from the server's `.env` (`src/rag/feature_flags.py`). This page
describes the case-first single shot (`CASE_FIRST_SINGLE_SHOT`). Where the lawyer's method (`LAWYER_METHOD`) matters,
the page says so. Its default in the code is false, and the nightly needs it on (`scripts/nightly_sim.py` docstring).

---

## 1. What the system answers and what it refuses

The intake reads every request once and picks one route (`src/rag/intake.py:ROUTES`, `route_of`):

| deliverable | route | what the user gets |
|---|---|---|
| `parere` | `case_first` | the single-shot parere (`src/rag/interactive_flow.py:single_shot`) |
| `atto` | `generation` (or `document` when an upload is to be filled) | a drafted act |
| `revisione_documento` | `document`, or `rag` without an upload | an analysis of the upload, or the plain answer |
| `cenni`, `quesito`, `calcolo`, `altro` | `rag` | the plain RAG answer, not a parere |
| `ricerca_giurisprudenza` | `out_of_scope` | a fixed refusal |

- **Case-law research is refused.** A request for sentenze, massime or orientamenti goes to
  `out_of_scope` (`src/rag/intake.py:ROUTES`). The user gets `src/rag/interaction_copy.py:CASE_LAW_OUT_OF_SCOPE`:
  «La ricerca di giurisprudenza (sentenze, massime, orientamenti) non rientra tra le richieste che gestisco…». The
  rule does not fire when the same message also asks for a parere or a defence
  (`src/rag/intake.py:deliverable_rule`).
- **Tax and insolvency matters are outside the corpus.** `src/rag/intake.py:OUT_OF_CORPUS` = `("tributario",
  "fallimentare")`. A parere on one of them gets `src/rag/interaction_copy.py:OUT_OF_CORPUS_MATTER`: «La materia
  {materia} non rientra nelle fonti che consulto…». The matter's name comes from `OUT_OF_CORPUS_MATTERS`, either
  «tributaria» or «fallimentare (crisi d'impresa e insolvenza)». The reply is built in
  `src/chatbot/api.py:_out_of_scope_reply`.
  - The refusal fires only when the *rules* found the rito (confidence `rule`). If the model found it, the request is
    not refused and follows the normal route (`src/rag/intake.py:route_of`).
  - A *question* or a *calculation* on a tax matter is not refused. It gets the plain answer, whose calculation gate
    reaches the tax calculators (`src/rag/intake.py:route_of` docstring). Read such an answer knowing that the
    matter is outside the corpus for a parere.
- **Short questions get a plain answer, not a parere.** A single question of at most
  `src/rag/intake.py:QUESTION_MAX_WORDS` = 35 words with no facts is a `quesito`. A message naming an article in at
  most `ARTICLE_ONLY_MAX_WORDS` = 15 words is read as a request to explain it. Both get the RAG answer, which does
  not have the parere's sections or checks.
- **When the intake is unsure.** Rules run first, then one model call fills only the fields the rules left empty.
  Anything still empty takes a default: parere for a narrated case, `pre_filing` for the stage, and `sconosciuto`
  for role and rito (`src/rag/intake.py` docstring). A wrong default changes the route or the posture of the answer.
- **The document after an answer** reuses the last answer's text as the draft. It is never a fresh draft
  (`src/chatbot/api.py:_interactive_document`). It is headed «BOZZA DOCUMENTO (richiede revisione da parte
  dell'avvocato)».

## 2. Corpus limits

- **The answer cites only what the graph holds.** Articles come from retrieval over the Neo4j graph. A follow-up
  message reads no corpus: the parere is updated from the analysis cached in the session
  (`src/rag/interactive_flow.py` docstring, item 1). That session state expires after
  `src/rag/interactive_flow.py:TTL_HOURS` = 24.
- **Special laws need a domain signal.** The special-law codes are listed in `data/special_laws.json`: codice della
  strada, codice del consumo, the testi unici (TUB, TUF, edilizia, immigrazione, stupefacenti, sicurezza sul
  lavoro, enti locali, TUIR), d.lgs. 231/2001, d.lgs. 11/2010, and the regulations listed in the next bullet. An
  article from one of these laws becomes a candidate only when the case shows its domain signal or the user names
  the article (`src/rag/special_laws.py:gate`). If the case is relevant but the facts do not trigger the signal, the
  law is left out. The file itself says its list and expressions are a draft and that the corpus's document names
  must be checked on the server (`data/special_laws.json:_about`).
- **Regulations are excluded unless listed or named.** The EU and Banca d'Italia regulations in
  `data/special_laws.json` are DORA, GDPR, Reg. 692/2014, Direttiva 94/25/CE, the Regolamento generale and Provv.
  2022/1917 of the Banca d'Italia. Each needs its own entry and signal. A regulation with no entry never reaches
  grading unless the user names it (`data/special_laws.json:_about`, v75).
- **The labour rito's articles need the labour rito.** The c.p.c. articles of the labour rito (409-447) are used only
  when the case's rito is «lavoro», or when the user names them (`src/rag/special_laws.py:labour_gate`).
- **Some laws are known to be missing.** `data/ingestion_tier1.json` lists governing laws the answers need that the
  corpus lacks:
  - D.Lgs. 11/2010 on payment services (phishing, card and transfer fraud), status «da ingerire»
  - D.L. 11/2009 (ammonimento del questore, atti persecutori), status «da verificare nel corpus: se assente, da
    ingerire»

  For labour, d.lgs. 66/2003 and l. 300/1970 had no primary document on 28 Oct. Their standard articles (Statuto
  artt. 7, 18; d.lgs. 66/2003 artt. 3, 5, 10; d.lgs. 23/2015 art. 3; d.lgs. 81/2015 artt. 19, 21, 42) are cited by
  reference and rubrica, never with their text (`src/rag/ccnl_path.py:FRAMEWORK_REFERENCES`).
- **What happens with a law or article not in the graph:**
  - A hypothesis that no retrieved article supports is listed as «Ipotesi da verificare (nessuna norma trovata le
    sostiene ancora)» and never asserted (`src/rag/interaction_copy.py:UNANCHORED_TITLE`).
  - An article whose text was not read gets no commi in `article_citations`, and it is not listed at all if it has no
    rows either (`src/rag/interactive_flow.py:article_citations`). Its terms read «testo non letto, termini da
    verificare» (`src/rag/interaction_copy.py:STORE_TERM_UNREAD`). Procedibilità and competenza read «da verificare
    (non lette dal testo dell'articolo)» (`PROCEDURE_UNKNOWN`).
  - Under the lawyer's method, if the main article's text was not retrieved, the method is not applied and the
    answer says so: «il testo dell'art. … non è stato recuperato»
    (`src/rag/lawyer_method.py:NOT_APPLIED`, `NOT_APPLIED_WHY["unread"]`).

## 3. Answer limits

### Caps

| what | constant | value |
|---|---|---|
| topics fully covered by a single-shot parere | `src/rag/interactive_flow.py:SINGLE_SHOT_TOPICS` | 5 (at least `SINGLE_SHOT_TOPICS_MIN` = 3 where as many exist) |
| articles in the single-shot answer | `src/rag/interactive_flow.py:SINGLE_SHOT_ARTICLES` | 15 |
| norm blocks written in the body | `src/rag/interactive_flow.py:BODY_BLOCKS_MAX` | 7 |
| «Altri profili» lines | `src/rag/interactive_flow.py:OTHER_PROFILES_MAX` | 5 |
| «Punti da precisare» block | `src/rag/interactive_flow.py:OPEN_POINTS_MAX` | 0 (no block) |
| alternative qualifications searched | `src/rag/interactive_flow.py:ALTERNATIVES_MAX` | 3 |
| requisites linked to the facts per topic | `src/rag/issue_records.py:LINK_REQUISITES_MAX` | 36 |
| method facts: certi / ipotizzabili / mancanti | `src/rag/lawyer_method.py:FACTS_MAX` | 12 / 8 / 8 |
| objections (method) | `src/rag/lawyer_method.py:OBJECTIONS` | 3 |
| «Sintesi della soluzione» (method) | `src/rag/lawyer_method.py:SUMMARY_LINES` | 5 sentences |

What the caps mean for the reader:

- **Topics past the cap.** They appear only as one line each under «Altri profili (non trattati, salvo che tu lo
  chieda)» (`src/rag/interactive_flow.py:covered_topics`, `src/rag/interaction_copy.py:OTHER_PROFILES_TITLE`). They
  are not analysed.
- **Norms past the seven blocks.** They are not discussed in the text. They appear in the article citations with the
  reason «non trattata nel testo: il parere svolge le norme più rilevanti, fino a sette»
  (`src/rag/interaction_copy.py:LINE_REASON_CAPPED`).
- **Articles past the single-shot cap.** They appear as «tema «…», oltre il limite di {cap} articoli»
  (`LINE_REASON_CAP`).
- **Which norms win a block.** The ranking is fixed (`src/rag/interactive_flow.py:block_rank`): the main anchor first,
  then «applicabile», then «da provare» with an element proven, then articles the user named, then the rest. A
  relevant norm can be pushed out by the cap.

### «da verificare»

- **Missing facts.** The parere never asks the user a question in its text. A fact the case does not state, where
  it matters, is written as «da verificare: il caso non dice …» (`src/rag/interaction_copy.py:OPEN_POINT`,
  `OPEN_POINTS_WRITER_RULE`).
- **Open legal characterisations** carry the status «da verificare» (`QUALIFIED_STATUS`). The parere argues them
  rather than asking for them.
- **Ambiguous terms.** When a comma gives two numbers for the same act, the comma is quoted, no number is chosen,
  and the decorrenza is «da verificare» (`src/rag/interaction_copy.py:AMBIGUOUS_TERM`, `STORE_TERM_LINE`).
- **Under the lawyer's method,** every «da verificare» marker is moved out of the body into the last section, «Punti
  da verificare» (`src/rag/lawyer_method.py:collect_points`). An unproven circumstance is marked «da verificare: i
  requisiti dell'art. …» (`CLAIM_MARK`).

A «da verificare» point is an admitted gap. It is not an error to be corrected later by the system.

### Orientamenti giurisprudenziali

These rules apply under the lawyer's method (`src/rag/lawyer_method.py:orientations_section`).

- The «orientamenti» section is written by the code, not the model. For each controverted point it gives either a
  *retrieved* ruling (label, document id, massima quoted) or the single line «orientamento da verificare»
  (`ORIENTATION_TO_VERIFY`). In the second case the point also goes to «Punti da verificare».
- A ruling cited in the text that is not among the retrieved ones is replaced by «orientamento da verificare»
  (`unretrieved_rulings_out`, `ruling_ok`).
- A writer sentence attributing a view to the case law («giurisprudenza prevalente / minoritaria / consolidata», «la
  Cassazione ritiene», «Sezioni Unite») is removed (`_ORIENTATION_CLAIM_RE`, `orientation_claims_out`).
- As a result, the parere never says what the prevailing case law is unless a retrieved document shows it. The
  rulings available are those in the graph, so coverage depends on the ingested batches.

### Theses (lawyer's method)

- A thesis that contests an element the record marks «integrato» is allowed only if it is declared «minoritaria»,
  with realismo «basso». Otherwise it is barred.
- A thesis declaring no element cannot be verified and never leads.
- A minority thesis leads only when no sustainable non-minority line exists.

(`src/rag/lawyer_method.py` docstring, `soluzioni`; `coherent_solutions`.)

### No quoted article text in the body

- In the single shot an article is a *reference*, never its quoted text
  (`src/rag/interactive_flow.py`, `references = single_mode(state)` and `quoted`).
- The commi go to `article_citations`, which the frontend shows on demand
  (`src/rag/interactive_flow.py:article_citations`).
- A reader who wants the text of the norm must open the citations. The body paraphrases the norm.

### Assumptions about the client

- If the user does not say whom they assist, the parere is written for the party the case puts the action against
  (the convenuto or the one a carico), never simply the first name in the text (`src/rag/interactive_flow.py:settle_client`).
- The first line says so: «Il parere è reso nell'interesse di {client}; se assisti un'altra parte, dimmelo»
  (`src/rag/interaction_copy.py:SINGLE_SHOT_CLIENT`, `SINGLE_SHOT_CLIENT_ROLE`).
- If no party can be picked, the parere examines the parties' positions without taking one
  (`CLIENT_UNKNOWN`).
- Always check that first line. A wrong client turns the whole strategy around.

## 4. Time and model limits

- **Target time.** The single-shot target is `src/rag/interactive_flow.py:SINGLE_SHOT_TARGET_S` = 240 s. It is only
  logged against (`single_shot_timings`, `over_target`). Nothing stops or shortens an answer that runs over.
- **Front-end cut-off.** Production goes through a Vercel rewrite, which cancels a request after 120 s of silence.
  With `CHAT_KEEPALIVE_ENABLED`, the API sends a space after `_KEEPALIVE_GRACE_SECONDS` = 10 and then every
  `_KEEPALIVE_INTERVAL_SECONDS` = 20 (`src/chatbot/api.py`). Without the flag, an answer longer than 120 s shows the
  user an error, even though the server finishes it. A failure after streaming has started returns the generic
  `_LATE_ERROR_ANSWER`.
- **Per-step time limits.** When a limit expires, that step's output is dropped or kept partial, and the answer
  still goes out:

  | step | limit | what happens on timeout |
  |---|---|---|
  | each model call | `src/rag/ai_chat.py:_chat_kwargs["request_timeout"]` = 240 s (default `max_tokens` 1500) | — |
  | first read of the case | `src/rag/case_read.py:READ_SECONDS` = 20 s | the first call's read stands; the parties' top-up is skipped |
  | linking requisites to facts | `src/rag/issue_records.py:LINK_SECONDS` = 30 s | the record stays as it was |
  | distinctive elements | `DISTINCTIVE_SECONDS` = 45 s | — |
  | circumstances (method) | `src/rag/lawyer_method.py:CIRCUMSTANCES_SECONDS` = 30 s | — |

- **When the case-first flow fails.** If `interactive_flow.single_shot` raises, `_interactive_start` returns `None`.
  The API then runs the older defensive pipeline (`src/chatbot/api.py`, case_first branch:
  `run_defensive_pipeline`), headed «BOZZA STRATEGIA DIFENSIVA». That answer:
  - has none of the case-first structure, caps or checks above
  - carries no version footer
  - is not saved to the held-out set, because the footer and the save happen only in
    `src/chatbot/api.py:_interactive_reply`

  If that pipeline fails too, the user gets «Si è verificato un errore nella generazione della strategia. Riprova.»
- **Method set aside.** Under the lawyer's method, when the record offers no grounded line or the main article cannot
  be broken into elements or was not read, the parere falls back to the ordinary structure. It says so in a note
  (`src/rag/lawyer_method.py:NOT_APPLIED`).
- **Model dependency.** Every model call goes to one OpenAI-compatible endpoint (`ChatOpenAI`), set by
  `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env` (`src/rag/ai_chat.py:_chat_kwargs`,
  `env.production.template`). Temperature is 0. Answers change with the model behind that endpoint: a model change
  is a new baseline. Thinking is on only for the thesis selection and the writer under `LLM_THINKING=per_call`
  (`src/rag/ai_chat.py:per_call`). The thesis call takes the best of `src/rag/lawyer_method.py:SAMPLES` = 3 replies.

## 5. Measurement limits (v77)

- **The nightly measures structure, not legal substance.** `scripts/nightly_sim.py` runs five cells
  (`tests/data/nightly_cells.json`): civile/attore/pre_filing, civile/convenuto/primo_grado,
  penale/imputato/primo_grado, penale/persona_offesa/pre_filing, lavoro/attore/pre_filing. For each it checks:
  - the intake's reading of the cell
  - the ten sections present and non-empty
  - the tables, the stragiudiziale option, three objections and the assessment lines
  - no «da verificare» outside §10
  - the version footer

  (`src/rag/lawyer_method.py:structure_checks`, `scripts/nightly_sim.py:footer_check`.) A green nightly means the
  parere has the right *shape*. It says nothing about whether the law is applied correctly.
- **The stage-list diff against v59** (`scripts/stage_diff.py`) compares the procedural section's blocks and articles
  only, without a model or database. On the server's copy, which is not a git checkout, it is *skipped*, not failed
  (`scripts/nightly_sim.py:stage_check`).
- **Substance is measured only by the lawyer's scores on the held-out set.**
  - Every case-first answer served (live or test copy) is saved to `/opt/chatbot/heldout/<date>_<id>.json`
    (`src/chatbot/heldout.py`).
  - `scripts/heldout_export.py` hands N unscored answers to the lawyer (default `-n 10`).
  - `scripts/heldout_score.py` reads back his seven section scores, 0-5 each (`src/chatbot/heldout.py:SECTIONS`,
    `SCALE`).

  The limits of this measure:
  - few cases, scored by one lawyer
  - a mean per version (`src/chatbot/heldout.py:version_means`)
  - a case scored again replaces its earlier score (`latest_scores`)
  - a block with no voto is not recorded
  - no rule may be tuned on these cases (module docstring)
- **A version's mean is comparable only with enough scored cases.** The table shows the case count beside each mean
  (`src/chatbot/heldout.py:means_table`). The code sets no minimum, so read the count before comparing two versions.
- **The 31-case sim is retired as a measure.** `scripts/diagnose_long.py:TUNING_IDS` and `HELDOUT_IDS` are fixtures
  for unit tests: no batch, no recall, no acceptance. `--interactive-sim` without `--only` or `--file` refuses to run
  (`scripts/diagnose_long.py` docstring and argument check).
- **The replay tools cannot show method changes.** `scripts/render_replay.py` re-renders a stored transcript through
  today's code with no model call, keeping the writer's paragraphs as written. It covers round 1, round 2, §2 and §4
  of the older layout. `scripts/interactive_review.py` checks an `--interactive-sim` report against the interactive
  rules, with no model and no database. Neither uses `lawyer_method` or `issue_records`. So a change to the method,
  the thesis selection, the writer or the record checks does not show in a replay.

## 6. Operational limits

- **Two copies on the server.**
  - `/opt/chatbot` is live.
  - `/root/v38` is the test copy, run by `scripts/serve_test.sh` on a spare port 8030-8049. It refuses to run from
    `/opt/chatbot` and refuses port 8010.

  The test copy reads the live sessions, logos and templates (`src/state_paths.py` docstring). Its answers also go
  into the held-out set, marked `source: test` (`src/chatbot/heldout.py:source`). Filter with
  `heldout_export.py --source live` when only live answers should be scored.
- **Promote never deletes in `data/`.** `scripts/deploy.sh` and `scripts/promote.sh` sync one directory at a time:
  `src/ scripts/ docs/ tests/` mirrored with `--delete`, `data/` added to and updated only. So a data file removed
  from the repository stays on the server until someone deletes it by hand. `scripts/deploy_exclude.txt` (env
  files, `state/`, `heldout/`, logs, `diagnose_reports/`, the server-built `data/institutions_*.json`, the article
  cards, the uploads, the runtime files `data/` used to hold) is never copied and never deleted.
  `data/private_documents/` (uploads) stays in `data/` and is copied into every promote snapshot.
- **Version footer.** Each case-first answer ends with «SunnitAI <version>», read from the `VERSION` file written by
  `scripts/release.sh` (`src/version.py`). A copy without that file reports `unknown`, and its answers cannot be told
  apart in the held-out table.
- **Held-out answers contain client case text.** Each record holds:
  - the prompt
  - the uploaded case text when the case was a file
  - the intake and the answer

  (`src/chatbot/heldout.py:record`.) The export pack quotes the case text, cut at
  `scripts/heldout_export.py:CASE_TEXT_CHARS` = 20000 characters. These files are sensitive and stay on the server.
  Send the pack to the lawyer only by the agreed channel, and do not commit packs or `scores.jsonl` to the
  repository. Off the server, nothing is saved unless `HELDOUT_DIR` is set (`src/chatbot/heldout.py:folder`).
