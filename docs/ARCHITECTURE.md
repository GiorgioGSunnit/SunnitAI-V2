# SunnitAI-V2 architecture

SunnitAI-V2 is a FastAPI service (`src/chatbot/api.py`) that answers Italian lawyers in Italian. Its knowledge is a
Neo4j graph of Italian codes, laws and rulings. It reaches an OpenAI-compatible LLM endpoint configured in `.env`.
Users, tenants, conversations and billing are stored in Postgres. Most traffic goes through `POST /api/chat`. Each
request is first read once by the **intake** (`src/rag/intake.py`), which picks one of five routes. The main route is
**case-first**: a narrated case gets one complete *parere* from `src/rag/interactive_flow.py`. With `LAWYER_METHOD` on,
that parere follows the lawyer's ten-section method (`src/rag/lawyer_method.py`). The other routes draft *atti*, review
uploaded documents, answer plain questions through the LangGraph RAG pipeline (`src/rag/main.py`), or return a fixed
out-of-scope message. Two copies run on the server: `/opt/chatbot` (live) and `/root/v38` (test). Quality is measured
by two things: a five-cell nightly checks structure and regressions, and the lawyer scores a held-out set of real
answers for substance.

```
POST /api/chat ──> api.chat ──(keep-alive when CHAT_KEEPALIVE_ENABLED)──> api._chat_impl
   │
   ├─ open case-first session? ── api._interactive_document  (DOCX of the last answer)
   │                           └─ api._interactive_follow_up ──> interactive_flow.follow_up ──> _single_update
   ├─ deepdive link ──> defensive_generation.generate_deepdive_analysis
   ├─ document branch (logged-in user, file attached or named)
   │     intake(message + file text) ──> CASE_FIRST ──> api._interactive_upload_case / _interactive_start
   │                                 ──> compare | analyse | generate (fill upload) | defensive_draft
   └─ intake(message) = intake.request_intake ──> route
         CASE_FIRST   ──> api._interactive_start ──> interactive_flow.single_shot (or .start, rounds)
                          (failure ──> defensive_generation.run_defensive_pipeline)
         OUT_OF_SCOPE ──> api._out_of_scope_reply (interaction_copy)
         GENERATION   ──> template choice ──> api._run_generation_sync ──> DOCX
         RAG/DOCUMENT ──> ChatBot.chat ──> rag.main.run (LangGraph over Neo4j)
   case-first answer ──> api._interactive_reply: footer «SunnitAI <version>», heldout.save, session saved
```

## 1. The request path of `POST /api/chat`

### 1.1 Entry

- `api.chat`: when `CHAT_KEEPALIVE_ENABLED` is on, `_chat_impl` runs as a task with its own DB session. If no reply
  is ready after a grace period, the response streams spaces (`api._keepalive_body`) and then the JSON.
- `api._chat_impl`: checks the tenant's product access, then creates a session if the request has no `session_id`.
  `_intake_box` caches the intake so that it is read at most once per request (`api._request_intake`).

### 1.2 The intake (`src/rag/intake.py`)

`intake.request_intake` is the single entry point. The API, `interactive_flow.take_intake` and the sims all call it.
It wraps `safe_detect`, which wraps `detect`; it passes the last `HISTORY_TURNS` messages and logs a
`[VERBOSE:intake]` line (`intake.log_view`).

`intake.detect` runs deterministic rules first: `deliverable_rule`, `rito_rule`, `client_rule`, `stage_rule` and
`prior_decisions`. Then it makes one bounded model call (`ask_model`), only for fields the rules left empty. Anything
still empty gets a default. The output fields are:
- `deliverable`: one of `parere`, `atto`, `cenni`, `quesito`, `revisione_documento`, `calcolo`,
  `ricerca_giurisprudenza`, `altro`.
- `rito`, `role`, `stage`.
- `client`, `facts`, `case_in` (`upload` or `message`), `requested_blocks`, `named_articles`.
- `confidence` (rule, model or default) and `why`, per field.
- `route`.

`intake.route_of` maps the deliverable to a route through `ROUTES`, with these exceptions:

| Route | When |
|---|---|
| `case_first` | `parere` |
| `generation` | `atto`. If the upload is the document to fill, the route is `document` instead. |
| `document` | `revisione_documento` with an upload. Without an upload, the route is `rag`. |
| `rag` | `cenni`, `quesito`, `calcolo`, `altro` |
| `out_of_scope` | `ricerca_giurisprudenza`, or a `parere` whose rito is `tributario` or `fallimentare` (`OUT_OF_CORPUS`) and was set by a rule |

`intake.doc_intent` turns the intake into the document branch's own intent: `analyse`, `generate`, `defensive_draft`
or `rag`. If `detect` raises, `safe_detect` falls back to defaults: the old narrative check decides between `parere`
and `altro`.

### 1.3 How the route is chosen in `_chat_impl`

The checks run in this order:

1. **Open case-first session** (only when `api._interactive_on`, i.e. `interactive_flow.turns_on`: the flags
   `INTERACTIVE_FLOW` or `CASE_FIRST_SINGLE_SHOT`).
   - `_interactive_document`: the user asks for the document after an answer. The last answer, without its footer,
     becomes a DOCX through `_persist_generated_docx`.
   - `_interactive_follow_up`: the session's anchors hold a live state (`interactive_flow.active`). The message goes
     to `interactive_flow.follow_up`.
   - Otherwise the intake is read on the message. If the request carries a document (`_expects_upload`), the intake
     waits for the file's text instead. If the route is `CASE_FIRST`, `_interactive_start` runs.
2. **Deepdive** (`deepdive:` / `approfondisci:` prefix): `defensive_generation.generate_deepdive_analysis`.
3. **Document branch** (logged-in user; files matched by id, by name or by reference):
   - A correction branch handles "correggi …" on a generated document.
   - The intake is read with the primary file's text (`_upload_text`).
   - Two or more files are compared only when the message has a comparison verb **and** the route is `DOCUMENT`.
     Otherwise only one file is used.
   - With one file, `CASE_FIRST` goes to `_interactive_upload_case` when `case_in == "upload"` (the file is the case,
     the message the angle), and to `_interactive_start` otherwise.
   - Otherwise `doc_intent` picks one of three paths: `generate` fills the uploaded document; `analyse` reviews it with
     two best-effort RAG lookups plus one LLM call; `defensive_draft` runs `defensive_generation.run_defensive_pipeline`
     on the upload.
   - Anything else falls through.
4. **The intake's route**:
   - `CASE_FIRST`: `_interactive_start`, unless it already ran on this request. If it fails (returns None),
     `run_defensive_pipeline` answers on the message text ("BOZZA STRATEGIA DIFENSIVA").
   - `OUT_OF_SCOPE`: `_out_of_scope_reply`. The text is `interaction_copy.CASE_LAW_OUT_OF_SCOPE` or
     `OUT_OF_CORPUS_MATTER`.
   - `GENERATION`: see 1.5.
   - Anything else (`RAG`, or `DOCUMENT` with no file found): `chatbot.chat` (`ChatBot.chat` in
     `src/chatbot/session.py`), which runs `rag.main.run`.

### 1.4 The case-first flow

`api._interactive_start` calls `interactive_flow.entry_args`: the case is the upload's text if there is one, otherwise
the message. It then runs in an executor:
- `interactive_flow.single_shot`, when `CASE_FIRST_SINGLE_SHOT` is on: one complete parere;
- `interactive_flow.start` otherwise: round 1 of the multi-turn rounds.

`api._interactive_reply` stores the state in the session's anchors (`interactive_flow.store`, key `STATE_KEY`, TTL
`TTL_HOURS`). It records both messages and returns `ChatResponse` with `interaction` and `article_citations`.

`interactive_flow.single_shot` runs these steps:
1. `analyse_case`. Started in parallel: the first read (`case_read.start`, when `INTERACTIVE_CASE_FIRST` or single
   shot is on). Then the pipeline's analysis (`defensive_generation._analyse_case_for_pipeline`), retrieval and grading,
   and the topics built from graded rows (`topic_groups`). Each stage is timed by `stage_meter.Meter`.
2. `state["mode"] = "single_shot"`. An optional `case_id` names the thesis log.
3. `take_intake`: stores the intake in `state["intake"]`, resolved against the analysis (`intake.resolve`). It
   records `intake_source`: `request` if the API passed the intake, `flow` if it was read here.
4. `stated` and `apply_stated`: the client and the question named in the user's own words. The client becomes
   `parte_assistita`; the question becomes the angle that orders the topics.
5. `prepare_side`: if no client was stated, picks the client the answer will assume. That client's side orders the
   topics (`reside`).
6. `covered_topics`: at most `SINGLE_SHOT_TOPICS` (5) topics, ranked by: an article the user named, then the main-line
   hypothesis, then a row graded 3. It covers at least `SINGLE_SHOT_TOPICS_MIN` when that many exist.
7. `write_answer(state, numbers)`: runs inside `lawyer_method.scope()`. Inside it, `_write_answer` does the following:
   - grades the chosen topics' remaining rows (`grade_chosen`);
   - `settle_client`, then `posture.of`;
   - links the records (`link_records`);
   - writes the plan and the sections, renders the computed procedural block, and runs the checks (see 3);
   - when `method_mode`, recomposes the answer into the lawyer's ten sections (see 2).
   Topics not covered get one line each under «Altri profili» (`single_profiles`).
8. `article_citations`: the cited articles with their commi, rubrica and record rows, for the frontend.
9. The answer gets the version footer (`version.with_footer`). `state["last_answer"]` keeps the text without it.

A later message in a single-shot session goes through `follow_up`, then `_single_update`:
- A flow question gets the help text.
- A request for the document returns `document_request`.
- Anything else updates the state (`apply_reply`, `match_angle`) and rewrites the parere from the cached analysis.
  There is no new retrieval. Changed parts are marked «(aggiornato)» (`mark_updated`), and the result has
  `kind = "regenerate"` with the footer.

In rounds mode (`start`, `round1`, `round2`), `follow_up` parses numbers, «procedi», answers and corrections
(`parse_reply`), and writes topic answers with `write_answer` (`kind` `answer` / `deepen` / `regenerate`).

### 1.5 The other routes, in brief

- **Generation of atti**: `classify_document_type`, else `classify_system_template` (catalog picker: several
  candidates mean a "Quale variante preferisci?" reply, none a clarification). `_run_generation_sync` drafts the act.
  When the intake saw facts, the case is first read into a short record (`intake.case_record` via `case_read.read`).
  For a known user the draft is saved as DOCX (`_persist_generated_docx`). `src/rag/document_generation.py` and
  `src/rag/template_selection.py` back this route; the standalone endpoints are `/api/generate` and
  `/api/generate/download`.
- **Document review**: the `analyse` intent in the document branch. It extracts the text (capped by
  `utils.document.document_char_limit`), runs RAG lookups to ground the review in the corpus, then makes one LLM call. Several files
  go through `_run_comparison_sync`.
- **Plain RAG**: `ChatBot.chat` → `rag.main.run`, a LangGraph pipeline. Its nodes: `calculation_gate` /
  `calculation_node` (`src/rag/calculation.py`), `decompose_query`, Cypher generation and execution over Neo4j
  (`src/rag/nodes/`), and `synthesize_answer`, which is wrapped by `source_tiers.support_node`.
- **Out of scope**: fixed text from `src/rag/interaction_copy.py`, with no retrieval.

## 2. The lawyer's method (`src/rag/lawyer_method.py`, flag `LAWYER_METHOD`)

`interactive_flow.method_mode` is true only when all of these hold: `LAWYER_METHOD` is on, the state is single-shot,
the state has records, and the method was not refused. Under it, `_write_answer` does the following:
- `lawyer_method.begin(rito, areas)` sets a contextvar, so every writer call carries his rules (`writer_rules_text`):
  REGOLE DI ONESTÀ, FASE 7, STILE, and the modules for the case's rito and areas.
- `interactive_flow.method_inputs` calls `lawyer_method.prepare` (context), `method_plan` (thesis selection) and
  `objections_plan`.
- If the main thesis is a placeholder (`placeholder_main`), the method is refused. The standard five-section body
  (`interaction_copy.SECTION_TITLES`) is written instead, prefixed by `not_applied_note`.
- Otherwise `lawyer_method.compose` lays our checked sections out in his ten sections.

**The ten sections** (titles from `data/lawyer_method.json` → `sections`; keys are the code's):

| # | Key | Title |
|---|---|---|
| 1 | `sintesi` | Sintesi della soluzione (max 5 righe) |
| 2 | `fatti` | Fatti: certi / ipotizzabili / mancanti (`facts_split`) |
| 3 | `questione` | Questione giuridica |
| 4 | `norme` | Norme e analisi per elementi (element tables: `subsumption_table`; circumstances: `circumstance_records`) |
| 5 | `orientamenti` | Orientamenti rilevanti: only rulings found in the retrieved context (`ruling_ok`), otherwise «orientamento da verificare» |
| 6 | `soluzioni` | Soluzioni a confronto (table: tesi, pro, contro, realismo; plus the stragiudiziale option) |
| 7 | `obiezioni` | Obiezioni e risposte (three) |
| 8 | `processuali` | Profili processuali (computed block; `temporal_line`, `litisconsorzio_lines`) |
| 9 | `conclusioni` | Conclusioni e valutazione del rischio (`assessment`: esito probabile, grado di certezza, obiettivo realistico) |
| 10 | `verificare` | Punti da verificare: every «da verificare» marker is moved here (`collect_points`) |

**Thesis selection**: the code chooses the theses, and the model only argues them.
- `thesis_candidates` lists the candidates from the record: per-element contests, riqualificazioni, scriminanti /
  attenuanti, the containment line, and the role-based lines from `data/remedies.json`.
- `method_plan` sends one request asking for `SAMPLES` (3) replies at temperature 0.7, with thinking on when
  `LLM_THINKING=per_call`. `score_sample` scores each reply and the best is kept. `clean_method` drops any thesis the
  model added.
- `coherent_solutions` (through `selection`) bars a thesis that contests an «integrato» element, unless it is
  declared minoritaria.
- The summary, conclusions and next steps follow from that selection (`conclusions_from_selection`,
  `steps_from_selection`).
- `final_check` answers his CONTROLLO FINALE from the code's own checks and logs the result.
- Every thesis call is logged to `<state dir>/logs/thesis_<case>.json` (`thesis_log`; `THESIS_LOG_DIR` overrides the folder).
  `scripts/thesis_replay.py` replays it.

**`structure_checks(answer, cell)`** is the nightly's measure of an answer's shape. It checks:
- all ten sections are present, non-empty and in order;
- §1 has at most 5 sentences;
- the element table and the solutions table are present, and there is no «requisiti provati N su M» line;
- the stragiudiziale option is there, and there are three objections;
- §9 has its esito / grado / obiettivo lines;
- no «da verificare» appears outside §10;
- orientations are rendered by the code;
- checks that depend on the cell: mediazione / ATP / 700 for a civil claimant before filing, riti alternativi for an
  imputato, and the ratione temporis line.

**Data and documents**:
- `docs/lawyer/metodo.md` is the lawyer's text. `scripts/lawyer_method_import.py` copies it verbatim into
  `data/lawyer_method.json`: ruolo, fasi, sections, stile, honesty_rules, controllo_finale and modules.
  `tests/test_v70.py` checks that the two stay in sync. The tables that follow in the JSON (reforms, litisconsorzio,
  stragiudiziale, circumstances, riti_alternativi) are Code's drafts for review.
- `docs/lawyer/note_inquadramento.md` holds his per-article notes. `scripts/fill_procedural_store.py --notes` imports
  them onto the Section nodes; `src/rag/article_notes.py` reads them and falls back to the file. They reach the
  records, the norms writer and the thesis call.
- `docs/lawyer/critica_penale_v2.md` and `docs/lawyer/critiques/` hold his critique. It defines the seven score
  sections used by the held-out set (`heldout.SECTIONS`).

## 3. Module map of `src/rag/`

**Intake & routing**
- `intake.py`: the per-request intake and its route (section 1.2).
- `query_router.py`: older router (`ROUTER_ENABLED`); `party_kinds` is used by the intake.
- `calculation.py`: HTTP bridge to the calculation platform; the gate inside the RAG graph.
- `language.py`: session language (Italian by default).
- `doc_lookup.py`: document-name lookup and reference extraction.

**Case reading & facts**
- `case_read.py`: the first read (one bounded call): client, goal, facts, unknowns, hypotheses.
- `fact_signals.py` (+ `fact_signal_rules.yaml`): the typed decisive facts, terms, parties, procedural tools.
- `fact_issues.py`, `fact_resolver.py`: a fact's institution, and its resolution to articles.
- `fact_terms.py` (`data/fact_terms.json`): fact nouns taken from the case's own words.
- `posture.py` (`data/posture_acts.json`): rito / role / stage → posture and its consumers.
- `procedural_facts.py`: procedibilità, terms and amounts read from the article's text, never from the model.
- `party_names.py`, `special_laws.py` (`data/special_laws.json`), `sentence_split.py`.

**Retrieval & grading**
- `main.py` + `nodes/` (retrieval, cypher, synthesis, routing) + `graph_nodes.py`: the LangGraph RAG pipeline.
- `lookups.py`, `lookup_indexes.py`, `embedding_config.py`, `reranker.py`: Neo4j B-tree, full-text and vector search.
- `defensive_generation.py`: the case-analysis pipeline. `interactive_flow` imports its analysis, pool and grading.
- `source_tiers.py`, `evidence_tiers.py`, `pool_gates.py`, `vote_pool.py`, `edge_expand.py`, `method_candidates.py`,
  `ruling_cards.py`, `article_relations.py`: pool building and filters, each behind its flag.
- `article_text.py`: one article's text, read the same way on every path.

**Topics & records**
- `topic_step.py`, `topic_groups.py`, `topic_tables.py`: the case's topics from the graded rows.
- `issue_records.py`: the per-article record (requisites, the case's sentences for and against).
- `cenni.py` (`data/general_concepts.json`), `article_notes.py`.

**Writing & rendering**
- `interactive_flow.py`: the case-first flow and `write_answer`.
- `lawyer_method.py`: the ten-section method (section 2).
- `plan_then_write.py`: plan, then sections. `writer_rules.py`: the writer's rules.
- `rule_quote.py`: rules quoted from the comma.
- `render_computed.py` (`data/stage_procedure.json`), `practical_checklist.py`: the procedural section from templates.
- `anchored_writing.py`, `comma_material.py`, `section_fit.py`, `article_cards.py`: what the writer receives.
- `interaction_copy.py`: every fixed text the flow shows.
- `document_generation.py`, `template_selection.py`: atti.
- `ai_chat.py`, `llm_reply.py`, `prompts.py`: model clients and reply parsing.

**Checks & guards**
- `record_check.py`: the final check against the record.
- `readback.py`: removes contradictory claims.
- `answer_ledger.py`, `citation_ledger.py`: every removal is recorded.
- `repetition_guard.py`, `text_tidy.py`: remove repeated sentences, repair text defects.
- `citations.py`, `citation_closure.py`, `article_codes.py`: citation checks.
- `grounded_review.py`, `answer_invariants.py`, `plain_grounding.py`: grounding checks.
- `protected_sources.py`: protected sources (doctrine books, CCNL) are never reproduced.
- `role_fit.py`, `stage_meter.py` (timings per stage), `verbose_logger.py` (`[VERBOSE:*]` lines).

**Data files under `data/`**
- `lawyer_method.json`, `remedies.json`: the method.
- `posture_acts.json`, `stage_procedure.json`, `labour_procedure.json`, `mediation_matters.json`: posture and
  procedure.
- `special_laws.json`, `ingestion_tier1.json`, `general_concepts.json`, `fact_terms.json`: corpus scope and vocabulary.

## 4. Storage & services

- **Neo4j**: the graph of codes, laws and rulings (`Document`-`CONTAINS`->`Section` nodes).
  Configured by `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` and `NEO4J_DATABASE`. The driver is opened in
  `src/rag/main.py`.
- **Postgres**: tenants, users and their profiles and settings, conversations, user documents and subscriptions
  (`src/db/models.py`). `DATABASE_URL` is read by `src/db/base.py`; migrations live in `src/db/migrations`
  (`alembic.ini`). Stripe billing is in `src/chatbot/billing.py`.
- **Sessions**: `ChatBot` in `src/chatbot/session.py`. Postgres (`conversations`) is the primary read store and is
  loaded on demand. `sessions.json` is still written (`_save_sessions`) but never read back. It lives in the state
  folder (`src/state_paths.py`: `STATE_DIR`, else `/opt/chatbot/state` on the server, else `<checkout>/state`;
  `SESSIONS_FILE` overrides). The case-first state lives in the session's anchors.
- **LLM**: any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`), with embeddings at
  `EMBEDDING_BASE_URL`. Both are configured in `src/rag/ai_chat.py` from `.env` (template: `env.production.template`).
- **Calculation platform** (`calculation_platform/`): a separate HTTP service, systemd `calc-platform` on port 8802
  (`deploy.sh`). It is reached at `CALC_PLATFORM_URL` (default `http://localhost:8802`) from `src/rag/calculation.py`.
- **Logs**:
  - the journal's `[VERBOSE:<event>]` lines (`verbose_logger.vlog`);
  - `<state dir>/logs/thesis_<case>.json`, one per thesis call (`THESIS_LOG_DIR` overrides);
  - `<state dir>/logs/cypher_queries.log` (`cypher_logger.py`; `CYPHER_LOG_PATH` overrides). Before v77 these were
    `logs/` and `data/logs/`; `scripts/promote.sh` moves them once.
- **Feature flags**: environment booleans read on every call (`src/rag/feature_flags.py`). A change needs a restart.

## 5. Measurement & tooling

- `scripts/nightly_sim.py`: five posture cells from `tests/data/nightly_cells.json` (civile/attore/pre_filing,
  civile/convenuto/primo_grado, penale/imputato/primo_grado, penale/persona_offesa/pre_filing,
  lavoro/attore/pre_filing). For each cell it:
  1. runs the intake and checks it read the cell's rito, role and stage;
  2. writes the parere through `interactive_flow.request_case_first`, the same intake and flow as the API;
  3. runs `lawyer_method.structure_checks` on it, plus `version_footer`.
  It outputs one row per cell and one column per check. Checks that passed in `--baseline` and fail now are listed
  under «Regressions». The run exits 1 on any failure. `--answers DIR` checks saved answers offline. The report records
  the VERSION it ran, and ends with the lawyer's mean scores per version.
- `scripts/stage_diff.py`: the procedural blocks rendered by the v59 tree (`git archive`, default ref `9422fd7`)
  compared with today's for the default cell. It makes no model or DB calls, and exits 1 when a block or an article
  was lost.
- `scripts/diagnose_long.py --interactive-sim`: the 31 eval cases (`TUNING_IDS` + `HELDOUT_IDS` defined in the script,
  cases in `tests/eval_narrative_cases.json`). **As of v77 this is diagnosis only, not an acceptance measure**: no
  rule is validated against it.
- **Acceptance** has two parts. Structure and regressions are checked by the nightly. Substance is checked by the
  held-out set, which the lawyer scores (section 6).
- Other tools: `scripts/intake_check.py` (intake on sample requests; `--rules-only`), `scripts/thesis_replay.py`, and
  `scripts/critique_input.py` with `scripts/rubric_score.py`.

## 6. v77 delivery: version, footer, held-out set, deploy

- **VERSION**: a file at the repo root containing `<semver>+<git short sha>`. `scripts/release.sh` writes it, then
  commits and tags `v<semver>`. `src/version.py` reads it once per process (`current`); with no file the value is
  `unknown`.
- **`GET /api/version`** (`api.version_check`) returns `{"version": ...}`.
- **Footer**: every case-first answer ends with one line, `SunnitAI <version>` (`version.with_footer`, which is
  idempotent). The footer is added in three places: `interactive_flow.single_shot`, the regenerate path
  (`_single_update`), and `api._interactive_reply`, for kinds `answer`, `deepen` and `regenerate`. It is stripped
  (`version.strip_footer`) before the answer becomes a DOCX in `api._interactive_document`.
- **Held-out set**: `src/chatbot/heldout.py` `save` is called from `api._interactive_reply`. It writes every
  case-first answer served by the live copy (`/opt/chatbot`) or the test copy (`/root/v38`) to
  `/opt/chatbot/heldout/<date>_<id>.json`. Each record holds the prompt, the uploaded case text if any, the intake,
  the answer, `article_citations`, the version, `source` (live/test) and the timings. `HELDOUT_DIR` overrides the
  folder. With no `/opt/chatbot` nothing is saved, and a failed save never affects the answer.
- **Scoring**:
  - `scripts/heldout_export.py -n N --out pack.md` exports N unscored answers, each with a blank score block covering
    the lawyer's seven sections (0-5).
  - `scripts/heldout_score.py pack_filled.md` reads the filled pack and appends to `heldout/scores.jsonl`.
  - `heldout.means_table` builds the table of means per version.
- **Deploy**: `scripts/deploy.sh` copies the code from the Mac to `/root/v38`. `scripts/promote.sh`, run on the server,
  copies `/root/v38` to `/opt/chatbot` (snapshot, per-directory rsync with `data/` never `--delete`, boolean flags
  from `.env.v45`, import check, restart). It leaves the state folder alone. See `docs/RUNBOOK.md`.

## 7. The two server copies

| | `/opt/chatbot` (live) | `/root/v38` (test) |
|---|---|---|
| Run by | systemd service `chatbot` (port 8000 in `deploy.sh`) | `scripts/serve_test.sh start [PORT]` |
| Env | `/opt/chatbot/.env` | `.env.v45`, plus `INTERACTIVE_FLOW`, `TOPIC_STEP`, `TOPIC_TABLES`, `INTERACTIVE_CASE_FIRST` forced on |
| Ports | the service's | first free port in 8030-8049 |
| Held-out `source` | `live` | `test` |

- `serve_test.sh` refuses to run from `/opt/chatbot` and refuses port 8010. It never touches the `chatbot` service or
  overrides LLM settings. It binds to `127.0.0.1` by default (reach it with `ssh -L`). `stop` and `status` manage it.
- The test copy shares the live state folder (`/opt/chatbot/state`) and the held-out folder.
- **Never use port 8010.**

## Where to look next

- `docs/FLAGS.md`: the feature flags.
- `docs/RULES.md`: the rules.
- `docs/LIMITS.md`: the known limits.
- `docs/RUNBOOK.md`: release, deploy and promote.
