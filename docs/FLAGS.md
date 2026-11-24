# Environment flags and settings

Every environment variable the server code (`src/`) reads, plus those read by the operator scripts
`scripts/nightly_sim.py`, `scripts/diagnose_long.py` and `scripts/serve_test.sh`.

## How flags are read

- Settings come from the process environment. The `.env` file is loaded by python-dotenv when the modules are first
  imported (`src/rag/ai_chat.py`, `src/rag/embedding_config.py`, `src/utils/db_client.py`; `src/main.py` loads
  `<project root>/.env`). python-dotenv does not override a variable that is already in the environment, so a value
  exported before start (for example by `scripts/serve_test.sh`) takes precedence over `.env`.
- **A change to `.env` needs a restart.** `feature_flags.enabled()` calls `os.getenv` on every call, but `.env` is
  read only at startup. Several non-boolean settings (marked "import" below) are read once, when their module is
  imported.
- **Boolean flags** go through `src/rag/feature_flags.py:enabled(name, default=False)`:
  - the value is stripped and lower-cased;
  - `1`, `true`, `yes`, `on` mean on, and `0`, `false`, `no`, `off` mean off;
  - an unset or empty variable gives the default;
  - any other value (a typo, for example) logs a warning (`NAME='...' is not true/false: using <default>`) and falls
    back to the default, so a typo never switches a feature on.
  - Every boolean flag defaults to `false` except `LEGACY_LAW_HINT_FALLBACK` and `FOOTER_IN_BODY` (v79.12.7), which
    default to `true`.
- Three switches do **not** use `feature_flags.enabled()` and accept fewer values:
  - `RERANKER_ENABLED` and `VERBOSE_LOGGING` are on only for `true` (any case), and are read at import.
  - `STRIPE_AUTOMATIC_TAX_ENABLED` is on for `1`/`true`/`yes`/`on`; any other non-empty value means off.
- `ROLE_FIT` also accepts `shadow` (see below).
- The test suite starts every test with every boolean flag unset. `tests/conftest.py:_flags()` builds its `FLAGS` set
  by scanning `src/rag/**/*.py` for `enabled("NAME"` / `_enabled("NAME"` and for module constants matching
  `^[A-Z_]*FLAG[A-Z_]* = "NAME"`. That finds 90 flags, and all of them are listed below. Four boolean flags are
  outside that scan, and the suite does not unset them: `CHAT_KEEPALIVE_ENABLED` (it lives in `src/chatbot/`),
  `FACT_ISSUES_RESOLVER_V2_ENABLED` (its constant `RESOLVER_V2_FLAG` contains a digit),
  `RULING_CARDS_CIVIL_STRONG_ONLY` (its constant is `CIVIL_STRONG_ENV`) and `STRIPE_AUTOMATIC_TAX_ENABLED` (it does
  not use `feature_flags`).

"Default" below always means the default in the code, never the value on any server. Paths are relative to the
repository root. `dg` means `src/rag/defensive_generation.py`, and `ptw` means `src/rag/plan_then_write.py`.

---

## 1. Case-first flow (the parere)

These are the narrative-case path served by `src/rag/interactive_flow.py` and its single-shot parere.

| Name | Default | Read in | What it does |
|---|---|---|---|
| `INTERACTIVE_FLOW` | false | `src/rag/interactive_flow.py:enabled` (constant `FLAG`) | Answers a narrative case without a document in rounds (topics and questions, then the frame, then the answer on the chosen topic). |
| `INTERACTIVE_CASE_FIRST` | false | `src/rag/interactive_flow.py:case_first_enabled` (`CASE_FIRST_FLAG`) | Round 1 reads the case before anchoring (`case_read`) and grades the pool in two rounds. Topics are graded rows grouped by Sezione/Capo. Implied by `CASE_FIRST_SINGLE_SHOT`. |
| `CASE_FIRST_SINGLE_SHOT` | false | `src/rag/interactive_flow.py:single_shot_enabled` (`SINGLE_SHOT_FLAG`) | One message in, one complete parere out. Round 1 runs inside it and is never shown. When on, this is the narrative path whatever `INTERACTIVE_FLOW` says (`turns_on`). |
| `TOPIC_STEP` | false | `src/rag/topic_step.py:enabled` | After retrieval, builds 3-6 topics (lawyer's title, at most 8 anchor articles, the computed facts each one touches) from the questioni, the fired regimes and the central issues. |
| `TOPIC_TABLES` | false | `src/rag/topic_tables.py:enabled` | `topic_step` reads the draft criminal and administrative institution tables (`topic_tables.TABLES`). They supply anchors and question families only. |
| `LAWYER_METHOD` | false | `src/rag/lawyer_method.py:enabled` | Writes the single-shot parere by the lawyer's method (`data/lawyer_method.json`) in his ten-section layout. |
| `LLM_THINKING` | unset | `src/rag/ai_chat.py` (import: `_thinking_kwargs`, `_announce_thinking`), `ai_chat.py:per_call`, `src/rag/lawyer_method.py:thesis_log` | Unset or any other value: nothing is sent (the model's default). `false`/`0`/`no`/`off`: reasoning is disabled on every call. `per_call`/`per-call`/`call`: reasoning is off except on the thesis-selection and writer calls. |
| `LLM_REASONING_EFFORT` | unset | `src/rag/ai_chat.py:reasoning_effort`, `thinking_body` (via `thinking_bind`); `lawyer_method.py:thesis_log` | v78.1.1: when set, a call with thinking on (`LLM_THINKING=per_call` and a call that asks) sends it as `chat_template_kwargs["reasoning_effort"]` beside `enable_thinking`. A model whose template ignores the key behaves as before. `--reasoning-effort` on `scripts/thesis_replay.py` and `scripts/nightly_sim.py` overrides it for that run; the thesis log records it. |
| `THESIS_DETERMINISTIC` | unset | `src/rag/lawyer_method.py:sampling`, `deterministic` | v79: when `1`/`true`, the thesis call sends one sample at temperature 0 instead of best-of-3. Use it for version-to-version comparison. `--deterministic` on `scripts/nightly_sim.py` sets it for that run. `scripts/thesis_replay.py --deterministic` does the same for a replay. The thesis log records `deterministic`, `n` and `temperature`. |
| `RULINGS_PARTY_NAMES` | `redact` | `src/rag/rulings.py:redact_names` | v79 (F2): `redact` replaces the parties' names in a quoted passage with «[omissis]»; `keep` leaves them. The default follows the spec («unless told otherwise»), pending the PO's decision. |
| `LLM_THINKING_EXTRA_TOKENS` | 2000 | `src/rag/ai_chat.py` (import, `THINKING_EXTRA_TOKENS`) | Extra `max_tokens` added to a call that reasons under `LLM_THINKING=per_call`. |
| `OPEN_POINTS_MAX` | 0 | `src/rag/interactive_flow.py:open_points_max` (`OPEN_POINTS_ENV`) | Number of «Punti da precisare» the single-shot answer asks for. 0 means no block (every missing fact is marked «da verificare» in place). |
| `GRADE_PREFILTER_N` | 50 | `dg:grade_prefilter_n` (`GRADE_PREFILTER_ENV`) | Number of candidates, by embedding similarity, that the case-first round-1 grading keeps. |
| `HYPOTHESIS_ANCHOR_MIN` | 0.75 | `src/rag/topic_groups.py:anchor_min` (`ANCHOR_MIN_ENV`) | Minimum embedding similarity for a case-read hypothesis to anchor on a topic's Sezione/Capo title. |
| `THESIS_LOG_DIR` | unset, which means `<state dir>/logs` | `src/rag/lawyer_method.py:thesis_log` (`THESIS_LOG_ENV`) | Folder for `thesis_<case>.json` (the exact input and output of the thesis call, for replay). |
| `HELDOUT_DIR` | see section 5 | `src/chatbot/heldout.py:folder` | Folder where every case-first answer is saved for the held-out set. |
| `FOOTER_IN_BODY` | **true** | `src/rag/interactive_flow.py:footer_in_body` (read by `with_version`, in `single_shot`, `_single_update` and `src/chatbot/api.py:_interactive_reply`; `scripts/nightly_sim.py:footer_check`) | v79.12.7 (item 8): the case-first answer's body ends with the footer `SunnitAI <version>`. The version is also the response's metadata in every case: `version` in the reply payload (`ChatResponse.version`, `null` on any other reply), in the single-shot and regenerate results, and in the session message's metadata. **Turn it off (`FOOTER_IN_BODY=false`) once Antonelo's frontend renders the `version` field outside the text**: the body then carries no footer (one already there is stripped). The nightly's `version_footer` check reads the metadata, and with the flag off it wants no footer in the body. |

The following are **code constants, not environment variables**: no `.env` line changes them, and changing them
needs a code edit. All are in `src/rag/interactive_flow.py`:

| Constant | Value | Meaning |
|---|---|---|
| `SINGLE_SHOT_ARTICLES` | 15 | Article cap of the single-shot answer |
| `SINGLE_SHOT_TOPICS` | 5 | Topics the single-shot answer covers in full; the rest get one line each |
| `SINGLE_SHOT_TOPICS_MIN` | 3 | Topics covered at minimum, where that many exist |
| `SINGLE_SHOT_TARGET_S` | 240 | Time target in seconds; logged against, flags `over_target` |
| `TOPIC_ARTICLES` | 8 | Article cap of a round-based topic answer |
| `MAX_QUESTIONS` | 5 | Questions asked in round 1 |
| `OPEN_POINTS_MAX` | 0 | Fallback for the `OPEN_POINTS_MAX` env var |
| `ROUND1_GRADE_ROUNDS`, `RESERVE_MAX` | 2, 12 | Round-1 grading rounds, ungraded rows kept for the answer step |

---

## 2. Boolean flags by area

All default to `false` unless stated otherwise.

### 2.1 Routing, chat and retrieval graph

| Name | Default | Read in | What it does |
|---|---|---|---|
| `ROUTER_ENABLED` | false | `src/rag/query_router.py:enabled` (`ROUTER_FLAG`), `src/chatbot/api.py:_chat_impl` | Classifies each chat request as narrative, direct, ccnl or document, and routes it. |
| `CHAT_KEEPALIVE_ENABLED` | false | `src/chatbot/api.py:chat` (`_KEEPALIVE_FLAG`) | Streams `/api/chat` replies that are not ready after 10 s (a space every 20 s, then the JSON) so a proxy's 120 s silence limit does not cut them. |
| `PRIMARY_FIRST_ENABLED` | false | `src/rag/source_tiers.py:primary_first_enabled` | On a direct question, the graph's article nodes keep only primary and special documents. Rulings join at synthesis only if they cite an answer article. |
| `CCNL_PATH_ENABLED` | false | `src/rag/ccnl_path.py:enabled` | Answers a ccnl-routed turn from the collective agreement in force at the case's date (sector index, version, sections). |
| `LEGACY_LAW_HINT_FALLBACK` | **true** | `src/rag/nodes/retrieval.py:_law_hint_fallback` | Keeps the legacy guessed document when the intent classifier named none. Off: a query that names no article searches every document. |
| `SCOPED_DOTTRINA_ENABLED` | false | `src/rag/nodes/retrieval.py:context_retrieval`, `:dottrina_search` | With a resolved law hint, searches that law's own commentary only, when it has one. |
| `ISSUE_SPOTTING_ENABLED` | false | `src/rag/issue_spotting.py:issue_spotting` (`ISSUE_SPOTTING_FLAG`), `dg:_select_articles` | One call names the institutions at stake in a narrative question. Their articles are fetched by rubrica or number, then checked. |

### 2.2 Case analysis and facts

| Name | Default | Read in | What it does |
|---|---|---|---|
| `CASE_ANALYSIS_ENABLED` | false | `dg:_analyse_case_for_pipeline`, `dg:_case_analysis_pipeline` (`CASE_ANALYSIS_FLAG`) | Structured fact model plus a per-issue exact fetch/search over any document (`_analyse_case_v2`/`_select_articles_v2`). |
| `STRUCTURED_ANALYSIS_ENABLED` | false | `dg:_analyse_case_for_pipeline` | Case analysis in two calls (neutral facts, then a fixed per-area grid). Needs `CASE_ANALYSIS_ENABLED`. |
| `CASE_FRAME_ENABLED` | false | `dg:_analyse_case_for_pipeline`, `dg:_frame_and_fact_signals` | One call before the analysis reads the case frame (facts, posture, areas, questions, central issues). |
| `FRAME_SANITY_ENABLED` | false | `dg:_case_frame`, `src/rag/pool_gates.py:gate_areas`, `src/rag/vote_pool.py:filter_areas` | A criminal proceeding or offence in the text forces "penale" into the frame's areas and posture. Uncertain areas filter no code. |
| `ISSUE_CHECKLISTS_ENABLED` | false | `dg:_analyse_case_v2` | Appends general per-area issue checklists (methods, never article numbers) to the analysis prompt. |
| `FACT_SIGNALS_ENABLED` | false | `src/rag/fact_signals.py:enabled` | One call extracts typed facts; code computes the legal terms from dates and fires `fact_signal_rules.yaml`. |
| `FACT_REGIMES_ENABLED` | false | `src/rag/fact_signals.py:regimes_enabled` | Typed facts qualify the parties (consumer, public body, public official), and their special regimes join the central issues. |
| `REGIME_CANDIDATES` | false | `src/rag/fact_signals.py:regime_candidates_enabled` | Adds regime and procedural-tool candidate articles, filtered by `client_role` conditions. Draft pending the lawyer. |
| `ROLE_BLOCKS_ENABLED` | false | `src/rag/fact_signals.py:fit_to_role` | The criminal profile and posture checks follow the client's side (accused or offended party). |
| `PARTIES_ENFORCED` | false | `src/rag/fact_signals.py:parties_line` | With more than one party to sue, the conclusions name them all with art. 102 c.p.c. |
| `TERM_EXPIRY_TEXT` | false | `src/rag/fact_signals.py:_term_trace` | Writes a servitude's non-use term with its end ("scade nel 2036"), not a bare year. |
| `FACT_ISSUES_ENABLED` | false | `src/rag/fact_issues.py:enabled` | One call names, per decisive fact, the institutions it raises (at most 8). They are added to the central issues. |
| `FACT_ISSUES_EVERY_FACT_ENABLED` | false | `src/rag/fact_issues.py:every_fact` | With `FACT_ISSUES_ENABLED`: at least one institution per decisive fact, with the cap scaled to the case (8-16). |
| `FACT_ISSUES_RESOLVER_V2_ENABLED` | false | `src/rag/fact_issues.py:resolver_v2` (`RESOLVER_V2_FLAG`) | With `FACT_ISSUES_ENABLED`: institutions named as precisely as a code heading. A generic name is refused. Not in `conftest.FLAGS`. |
| `HEADING_INDEX_ENABLED` | false | `dg:_resolve_institution` | Resolves an institution against a vector index of article headings when the word match fails. Needs `scripts/build_heading_index.py`. |
| `RESOLUTION_STRICT_ENABLED` | false | `dg:_resolve_institution` | No resolution through a generic rubrica alone. A full-text resolution needs a shared word with the institution. |
| `STRICT_ARTICLES_ENABLED` | false | `dg:_resolve_institution`, `dg:_check_qualifications`, `src/rag/fact_signals.py:qualifications_block` | An article found only by keyword or full text counts only if its rubrica shares a non-generic word with the institution. |

### 2.3 Candidate pool (retrieval for the strategy)

| Name | Default | Read in | What it does |
|---|---|---|---|
| `CASE_POOL_FUSION_ENABLED` | false | `dg:_case_analysis_pipeline` (`_CASE_POOL_FUSION_FLAG`) | Adds a fused, ranked pool (current selection, raw-text search and distilled-bullet search) on top of `_select_articles_v2`. Needs `CASE_ANALYSIS_ENABLED`. |
| `FUSION_CLEAN_ENABLED` | false | `dg:_fusion_search_fn`, `dg:_select_articles_fusion`, `dg:prefetch_case_branches` | Fusion searches on primary documents only, minus books, templates and contract models. |
| `VOTE_POOL_ENABLED` | false | `src/rag/vote_pool.py:enabled` | Joins to the pool the articles cited by the rulings most similar to the facts, IDF-weighted. |
| `HYBRID_NARRATIVE_ENABLED` | false | `src/rag/source_tiers.py:hybrid_enabled` (`HYBRID_FLAG`) | Similar rulings are read as a map of issues and hypotheses, searched in primary documents in place of the vote. |
| `ARTICLE_RELATIONS_ENABLED` | false | `src/rag/article_relations.py:enabled` | Adds up to 6 related c.c. articles from `ARTICLE_RELATIONS_FILE` after the cap. |
| `EDGE_EXPAND` | false | `src/rag/edge_expand.py:enabled` | Adds one-hop neighbours (same Sezione, cites, cited by) of the prompt's and grade-3 articles. Needs the `REFERS_TO` graph. |
| `POOL_GATE_AREA` | false | `src/rag/pool_gates.py:area_enabled` (`AREA_FLAG`) | Drops a main-code candidate that the case's areas do not allow, unless it was named or found by two channels. |
| `POOL_GATE_LINK` | false | `src/rag/pool_gates.py:link_enabled` (`LINK_FLAG`) | Drops a single-channel row that is not in the Capo of, or `REFERS_TO`-linked to, a stronger row. |
| `METHOD_CANDIDATES_ENABLED` | false | `dg:_add_method_candidates` | Adds the method and procedural articles of the case's areas and posture as candidates. Acts only with `APPLICABILITY_CHECK_ENABLED`. |
| `CHECKLIST_GATED_ENABLED` | false | `src/rag/method_candidates.py:case_lists` | Method lists and the practical checklist only for the verified area × posture. Their candidates must pass the check. |
| `RULING_CARDS_ENABLED` | false | `src/rag/ruling_cards.py:enabled` | Chooses the strategy's rulings from the precomputed ruling cards (`RULING_CARDS_FILES`). |
| `RULING_CARDS_CIVIL_STRONG_ONLY` | false | `src/rag/ruling_cards.py:civil_settings` (`CIVIL_STRONG_ENV`) | Civil cases: ruling cards only from strong pool anchors (no keyword-hit anchors). Not in `conftest.FLAGS`. |
| `RULING_FIT_ENABLED` | false | `dg:_ruling_fit` | Before the plan, one call per ruling asks what it can serve for (fatti / regola / passo / nessuno). "nessuno" drops it. |

### 2.4 Applicability check and grading

| Name | Default | Read in | What it does |
|---|---|---|---|
| `APPLICABILITY_CHECK_ENABLED` | false | `dg:_case_analysis_pipeline`, `dg:_add_method_candidates`, `dg:_edge_expand` | Per-article applicability check against the facts, before citations and the draft prompt. |
| `CHECK_PRIORITY_ENABLED` | false | `dg:_filter_applicable_articles` | Check slots go to voted, then multi-signal, then other articles. An unchecked article is removed. |
| `CHECK_GRADED_ENABLED` | false | `dg:_filter_applicable_articles` | The check grades each article 0-3 with the issue it serves. Grade 0 leaves the pool. |
| `GRADER_EVIDENCE_ENABLED` | false | `dg:_filter_applicable_articles.grade_these` | The grader sees what found each article and grades per numbered issue. |
| `GRADER_EXAMPLES_ENABLED` | false | `dg:_filter_applicable_articles.grade_these` | Adds worked examples to the evidence grader's prompt. |
| `GRADER_VOTE` | false | `dg:_graded_verdicts` | Each article is graded alone by three calls; it is applicable when two or more votes give at least 2. |
| `GRADER_FIXED_ORDER` | false | `dg:_graded_verdicts` | Graded-check batches ordered by article id, whatever the retrieval order. |
| `GRADE_ALL_PROMPT` | false | `dg:_filter_applicable_articles` | Grades the articles that will reach the draft prompt (at most 120 per case), not only the first slots. |

### 2.5 Plan and writing

| Name | Default | Read in | What it does |
|---|---|---|---|
| `PLAN_THEN_WRITE_ENABLED` | false | `dg:_case_analysis_pipeline`, `dg:_ruling_label` | Builds the parere as a checked plan, then writes six sections in parallel. Falls back to the one-call parere. |
| `PLAN_KEEP_APPLICABLE_ENABLED` | false | `ptw:make_plan`, `ptw:_keep_all` | The plan keeps every article the applicability check let through (up to 7 issues). |
| `PLAN_KEEP_CHECKED_ENABLED` | false | `ptw:make_plan`, `ptw:_keep_all`, `ptw:_discipline` | Every article judged applicable or alternative is in the plan. Takes precedence over `PLAN_KEEP_APPLICABLE_ENABLED`. |
| `PLAN_RULE_INCLUDE` | false | `ptw:rule_include` | Adds to the plan, by rule and without a call, prompt articles that no issue holds (grade 3, and grade 2 with a signal). |
| `PLAN_PATCH_REDO` | false | `ptw:make_plan` | When the plan review only finds unanswered questions, asks for those answers alone instead of redoing the plan. |
| `PLAN_CAP` | false | `ptw:make_plan` | The writer's plan holds at most `PLAN_CAP_MAX` substantive articles. |
| `ROLE_FIT` | false | `src/rag/role_fit.py:mode` | `true`: one call per plan article asks whom it works for, and counterparty norms are tagged. `shadow`: asked and logged only. |
| `NORME_SCOPE` | false | `ptw:norme_scope_enabled` | "Norme e giurisprudenza" writes only the primary norms (at most 8). |
| `NORME_SPLIT` | false | `ptw:norme_split` | With more than 8 plan articles, "Norme e giurisprudenza" is written in two concurrent halves. |
| `RULE_QUOTE` | false | `src/rag/rule_quote.py:enabled` | Per plan article, the commi that work for the client and their requisites, matched to the parties' roles. |
| `RENDER_COMPUTED` | false | `src/rag/render_computed.py:enabled` | "Profili processuali e termini" is rendered from the computed block by template, never written by the model. |
| `STRATEGY_DRAFT_RULES_ENABLED` | false | `dg:_case_analysis_pipeline`, `dg:_parere_system` | Adds the lawyer's draft rules to the parere prompt (defend, do not restate the charge; full procedural section). |
| `WRITER_RULES_ENABLED` | false | `ptw:_writer_rules` | Writer rules (cite only in support, no invented facts, word budget per section) plus a review. |
| `FACT_MARK_INLINE_ENABLED` | false | `ptw:review_section` (`FACT_MARK_FLAG`) | Writes the "fact not in the case" marker into the answer. Off: flagged sentences are only logged. Acts with `WRITER_RULES_ENABLED`. |
| `REVIEW_FIXES_ENABLED` | false | `ptw:_review_fixes`; `src/rag/fact_signals.py:all_central`, `:procedural_tools`, `:qualify_parties`, `:terms_block` | Fixes from the lawyer's 26 Oct review (decisive facts, every procedural item, no unstated certainties). |
| `COMMA_MATERIAL_ENABLED` | false | `src/rag/comma_material.py:enabled` | The writer receives an article as its best-matching commi "(co. K)", not as its first N characters. |
| `ARTICLE_CARDS_ENABLED` | false | `src/rag/article_cards.py:enabled` | The writer reads an article's card (`ARTICLE_CARDS_FILE`) in place of an over-budget text. |
| `ANCHORED_WRITING_ENABLED` | false | `src/rag/anchored_writing.py:enabled` | The plan names numbered passages [A1]…, and each rule sentence must end with its anchor. Failing sentences are rewritten. |
| `ANCHOR_REWRITE_CAP` | false | `src/rag/anchored_writing.py:_anchor_rewrite` | Anchored rewrites in calls of at most 2,500 max_tokens (9 sentences), 4 at a time. |
| `WRITER_LOOP_GUARD` | false | `ptw:_loop_guard` | Cuts a section reply that repeats one block, and never rewrites a looped section at 1.5× tokens. |

### 2.6 Review guards (after writing)

| Name | Default | Read in | What it does |
|---|---|---|---|
| `CLIENT_STANCE_CHECK_ENABLED` | false | `dg:_case_analysis_pipeline` | One call checks section 3 for arguments that favour the other side, and rewrites it once if it finds any. |
| `QUALIFICATION_CHECK_ENABLED` | false | `dg:_case_analysis_pipeline` | Checks each cited article outside the case's lists against its real heading. A mismatch is stamped `[DA VERIFICARE …]`. |
| `RUBRICA_TEXT_CHECK` | false | `dg:_check_qualifications` | No `[DA VERIFICARE]` stamp when the citing clause's key terms occur in the article's text. |
| `FLAGGED_REMOVE_ENABLED` | false | `dg:_case_analysis_pipeline`, `dg:_check_qualifications`, `ptw:_after_checks` | Removes a citation that the qualification check finds incompatible, unless the pipeline has its own evidence for it. |
| `CITATION_CLOSURE_ENABLED` | false | `src/rag/citation_closure.py:enabled` | Reads and checks every article cited outside the material. A compatible one becomes a source; otherwise the citation is removed. |
| `CLOSURE_MATERIAL_ONLY` | false | `src/rag/citation_closure.py:material_only` | Closure material is the pool, tools, terms and case articles only. Any other citation is removed without a read. |
| `PLAN_ARTICLES_COVER_ENABLED` | false | `ptw:covering` (`COVER_FLAG`) | A plan article graded 2 or higher that no section cites gets one or two sentences in its issue's section. |
| `GROUNDED_EXCLUSION` | false | `ptw:grounded_exclusion` | Accepts a grounded "article A does not apply" sentence in plan cover and in the writer review. |
| `GROUNDED_REVIEW_ENABLED` | false | `src/rag/grounded_review.py:enabled` | Judges each section's rule/holding sentences against that section's own material and rewrites the unsupported ones. |
| `GROUNDED_REVIEW_STRICT_ENABLED` | false | `src/rag/grounded_review.py:strict` (`STRICT_FLAG`) | With grounded review: never fetches an article outside the checked material; the sentence is rewritten or dropped. |
| `ANSWER_INVARIANTS_ENABLED` | false | `src/rag/answer_invariants.py:enabled` | General invariant checks (I1…) on every strategy answer after writing, each logged as found, fixed or remaining. |
| `TOOL_CITES_PROTECTED` | false | `ptw:protected_tool_refs` | Sentences citing a procedural tool's or a computed term's article are trimmed, never dropped, by the later checks. |
| `USUCAPIONE_CONDITIONAL` | false | `ptw:usucapione_asserted`, `ptw:usucapione_conditional`, `ptw:write_sections` | With the possession's start unknown, an asserted usucapione is rewritten in the conditional. |
| `DRAFT_CITATIONS_ONLY_ENABLED` | false | `dg:_case_analysis_pipeline` | The sources returned with a parere are only the articles and rulings its text cites. |

### 2.7 Protected sources

| Name | Default | Read in | What it does |
|---|---|---|---|
| `PROTECTED_PARAPHRASE_ENABLED` | false | `src/rag/protected_sources.py:enabled`; used in `src/chatbot/api.py` | Protected sources are never reproduced on any answer path (special and ccnl documents, configured names, documents with a reproduction notice). |
| `PROTECTED_DOCUMENT_NAMES` | unset | `src/rag/protected_sources.py:protected_names` (`NAMES_ENV`) | Exact document names to protect, separated by `;` or newlines. |
| `PROTECTED_DOCUMENTS_FILE` | unset | `src/rag/protected_sources.py:protected_names` (`NAMES_FILE_ENV`) | File of document names to protect, one per line, `#` for comments. Re-read when it changes. |

---

## 3. Non-boolean settings

### 3.1 Chat model (LLM)

`LLM_THINKING` and `LLM_THINKING_EXTRA_TOKENS` are in section 1. Temperature (0), `max_tokens` (1500) and the request
timeout (240 s) are hard-coded in `src/rag/ai_chat.py:_chat_kwargs`.

| Name | Default | Read in | What it does |
|---|---|---|---|
| `LLM_BASE_URL` | unset (the OpenAI API) | `src/rag/ai_chat.py` (import), `src/rag/doc_lookup.py:_classify_query_intent`, `src/preprocessing/parallel_llm_extractor.py:_build_chains`, `src/chatbot/api.py:_debug_check_sync` | Base URL of the OpenAI-compatible chat server. |
| `LLM_MODEL` | `OPENAI_MODEL`, else `nemotron-2-30B-A3B` | `src/rag/ai_chat.py` (import), `src/rag/doc_lookup.py:_classify_query_intent` (default `""`), `src/chatbot/api.py:_debug_check_sync` | Chat model name. |
| `LLM_API_KEY` | `OPENAI_API_KEY` | `src/rag/ai_chat.py` (import), `src/rag/doc_lookup.py`, `src/rag/reranker.py:rerank_results`, `src/rag/template_selection.py:_embed`, `src/preprocessing/*`, `src/chatbot/api.py:_debug_check_sync` | API key for the chat server. It is also the reranker's key and the embedding key's fallback. |
| `OPENAI_MODEL` | `nemotron-2-30B-A3B` | `src/rag/ai_chat.py` (import) | Legacy fallback for `LLM_MODEL`. |
| `OPENAI_API_KEY` | unset | `src/rag/ai_chat.py`, `src/rag/template_selection.py:_embed`, `src/preprocessing/generate_embeddings.py`, `src/preprocessing/parallel_llm_extractor.py` | Legacy fallback for `LLM_API_KEY` and `EMBEDDING_API_KEY`. When set, it also makes `EMBEDDING_PROVIDER=auto` pick OpenAI. |

### 3.2 Embeddings and vector indexes

| Name | Default | Read in | What it does |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | `auto` (`local` in `generate_embeddings._add_doc_prefix`) | `src/rag/ai_chat.py` (import), `src/preprocessing/generate_embeddings.py` | `local` (HuggingFace), `openai`, or `auto`: OpenAI client if `EMBEDDING_BASE_URL` or `OPENAI_API_KEY` is set, else local. |
| `EMBEDDING_BASE_URL` | unset | `src/rag/ai_chat.py` (import), `src/rag/template_selection.py:_embed`, `src/preprocessing/generate_embeddings.py` | Base URL of the embedding server. |
| `EMBEDDING_API_KEY` | `LLM_API_KEY`, else `OPENAI_API_KEY` | same as above | API key for the embedding server. |
| `EMBEDDING_MODEL` | OpenAI: `OPENAI_EMBEDDING_MODEL`, else `text-embedding-3-small`; local: `BAAI/bge-large-en-v1.5`; template selection: none | `src/rag/ai_chat.py:_init_openai_embeddings`, `:_init_local_embeddings`, `src/rag/template_selection.py:_embed`, `src/preprocessing/generate_embeddings.py` | Embedding model name. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | `src/rag/ai_chat.py:_init_openai_embeddings`, `src/preprocessing/generate_embeddings.py` | Legacy fallback for `EMBEDDING_MODEL`. |
| `EMBEDDING_DIMENSIONS` | 4096 | `src/preprocessing/schema/schema.py`, `src/preprocessing/generate_embeddings.py` (import) | Vector size of the indexes (ingestion). |
| `EMBEDDING_QUERY_MODE` | `token_ids` | `src/rag/embedding_config.py` (import, `_choice`) | `token_ids` (legacy) or `text` (plain text with the Qwen query instruction). An invalid value falls back to the default. |
| `EMBEDDING_INDEX_SUFFIX` | `""` | `src/rag/embedding_config.py` (import, `_identifier`) | Suffix added to the migrated vector indexes' names (for example `_v2`). |
| `EMBEDDING_PROPERTY` | `embedding` | `src/rag/embedding_config.py` (import, `_identifier`) | Node property holding the vector for the document-scoped search. |
| `EMBEDDING_MIN_SCORE_SECTION` | 0.55 | `src/rag/embedding_config.py` (import, `_score`) | Text-mode minimum score for `section_embeddings`. |
| `EMBEDDING_MIN_SCORE_COMMENTATO` | 0.55 | `src/rag/embedding_config.py` (import) | Text-mode minimum score for `commentato_section_embeddings`. |
| `EMBEDDING_MIN_SCORE_FISCALITA` | 0.55 | `src/rag/embedding_config.py` (import) | Text-mode minimum score for `fiscalita_section_embeddings`. |
| `CASE_POOL_VECTOR_INDEX` | `section_embeddings_v2` | `dg` (import) | Vector index the fusion pool searches. |
| `HEADING_VECTOR_INDEX` | `section_heading_embeddings` | `dg` (import) | Vector index used by `HEADING_INDEX_ENABLED`. |

### 3.3 Reranker

| Name | Default | Read in | What it does |
|---|---|---|---|
| `RERANKER_ENABLED` | `false` (only `true` turns it on) | `src/rag/reranker.py` (import) | Reranks retrieval results through the reranker service. |
| `RERANKER_URL` | `http://217.160.8.129:8002/v1/rerank` | `src/rag/reranker.py` (import) | Reranker endpoint. |
| `RERANKER_MODEL` | `reranker` | `src/rag/reranker.py:rerank_results` | Model name sent to the reranker. |

### 3.4 Neo4j

| Name | Default | Read in | What it does |
|---|---|---|---|
| `NEO4J_URI` | none (`bolt://localhost:7687` in `src/cypher_cli.py` and `generate_embeddings`) | `src/rag/main.py` (import), `src/utils/db_client.py:DBClient.__init__`, `src/chatbot/api.py:_debug_check_sync`, `src/cypher_cli.py`, `src/preprocessing/*` | Neo4j connection URI. |
| `NEO4J_USER` | none (`neo4j` in the CLI and `generate_embeddings`) | same as above | Neo4j user. |
| `NEO4J_PASSWORD` | none | same as above | Neo4j password. |
| `NEO4J_DATABASE` | `neo4j` | `src/rag/main.py` (import), `src/cypher_cli.py`, `src/preprocessing/main.py`, `src/preprocessing/generate_embeddings.py` | Neo4j database name. |

### 3.5 PostgreSQL, auth and sessions

| Name | Default | Read in | What it does |
|---|---|---|---|
| `DATABASE_URL` | a local PostgreSQL URL hard-coded in `src/db/base.py` (it contains credentials, so it is not reproduced here) | `src/db/base.py` (import), `src/db/migrations/env.py` | SQLAlchemy database URL. |
| `MIGRATION_DATABASE_URL` | unset, which means `DATABASE_URL` | `src/db/migrations/env.py` | Database URL for Alembic migrations. |
| `PG_HOST` | `127.0.0.1` | `src/chatbot/user_store.py:_get_pool` | User store host. |
| `PG_PORT` | 5432 | `src/chatbot/user_store.py:_get_pool` | User store port. |
| `PG_DATABASE` | `astrea` | `src/chatbot/user_store.py:_get_pool` | User store database. |
| `PG_USER` | `astrea_admin` | `src/chatbot/user_store.py:_get_pool` | User store user. |
| `PG_PASSWORD` | `""` | `src/chatbot/user_store.py:_get_pool` | User store password. |
| `JWT_SECRET_KEY` | `change-me-in-production` | `src/chatbot/auth.py` (import) | JWT signing key. **Must be set in any deployment.** |
| `JWT_EXPIRE_MINUTES` | 10080 (7 days) | `src/chatbot/auth.py` (import) | Access-token lifetime. |
| `FALLBACK_USER_EMAIL` | `admin@studiorossi.it` | `src/chatbot/session.py` (import) | User email assumed when the frontend sends none (temporary). |
| `SESSIONS_FILE` | `<state dir>/sessions.json` | `src/chatbot/session.py` (import) | Sessions store file. |

### 3.6 Billing (Stripe)

| Name | Default | Read in | What it does |
|---|---|---|---|
| `STRIPE_SECRET_KEY` | none (required) | `src/chatbot/billing.py:get_stripe_client` (`require_stripe_env`) | Stripe API key. Missing or placeholder raises `BillingConfigError`. |
| `STRIPE_WEBHOOK_SECRET` | none (required) | `src/chatbot/billing.py:get_webhook_secret` | Stripe webhook signing secret. |
| `STRIPE_PRICE_PLUS_SINGLE_MONTHLY` | none (required for that plan) | `src/chatbot/billing.py` (`PLAN_PRICE_ENV`, `require_stripe_env`) | Price id of the plan `plus-single-monthly`. |
| `STRIPE_PRICE_PLUS_SINGLE_ANNUAL` | none (required for that plan) | `src/chatbot/billing.py` (`PLAN_PRICE_ENV`) | Price id of the plan `plus-single-annual`. |
| `STRIPE_PRICE_PLUS_MULTIUSER` | none (required for that plan) | `src/chatbot/billing.py` (`PLAN_PRICE_ENV`) | Price id of the plan `plus-multiuser`. |
| `STRIPE_TRIAL_DAYS` | 7 (minimum 0) | `src/chatbot/billing.py:get_trial_days` | Trial length. A non-integer raises `BillingConfigError`. |
| `STRIPE_MULTIUSER_MIN_SEATS` | 3 (minimum 3) | `src/chatbot/billing.py:get_multiuser_min_seats` | Minimum seats of the multi-user plan. |
| `STRIPE_AUTOMATIC_TAX_ENABLED` | false | `src/chatbot/billing.py:get_automatic_tax_enabled` (`stripe_bool_env`) | Stripe automatic tax. On for `1`/`true`/`yes`/`on`. |

### 3.7 External services, files and paths

| Name | Default | Read in | What it does |
|---|---|---|---|
| `CALC_PLATFORM_URL` | `http://localhost:8802` | `src/rag/calculation.py:PlatformClient.__init__` | Base URL of the separate calculation service. |
| `SYSTEM_TEMPLATES_CATALOG_PATH` | `<state dir>/system_templates/catalog_enriched.json` | `src/rag/document_generation.py` (import), `src/rag/template_selection.py` (import) | System templates catalog. Its folder is also the default folder of the template vectors. |
| `TEMPLATE_VECTORS_PATH` | `<catalog folder>/template_vectors.npy` | `src/rag/template_selection.py` (import) | Template embedding matrix. |
| `TEMPLATE_VECTORS_INDEX` | `<catalog folder>/template_index.json` | `src/rag/template_selection.py` (import) | Template vector index (row → template). |
| `ARTICLE_CARDS_FILE` | `<state dir>/article_cards.jsonl` | `src/rag/article_cards.py:_store` (`FILE_ENV`) | Article cards file for `ARTICLE_CARDS_ENABLED`. Re-read when it changes. |
| `ARTICLE_RELATIONS_FILE` | `""` (none: the feature does nothing) | `src/rag/article_relations.py:store` (`FILE_ENV`) | Relations file built by `scripts/maint_relations.py`. |
| `RULING_CARDS_FILES` | `""` | `src/rag/ruling_cards.py:_paths` (`FILES_ENV`) | Comma-separated JSONL ruling-card files. |
| `VOTE_POOL_IDF_FILE` | `""` (counts are computed in-process, and the vote is skipped until they are ready) | `src/rag/vote_pool.py:idf_table` (`IDF_FILE_ENV`); `scripts/diagnose_long.py:_vote_idf_file` (reads and sets it) | Precomputed IDF counts for the ruling vote. |
| `FACT_SIGNALS_TODAY` | unset | `src/rag/fact_signals.py:reply_today`, `:present` (`TODAY_ENV`) | `YYYY-MM-DD` used as the case's present when the case states none (tests, benches). Never the real today in `present`. |

### 3.8 Logging

| Name | Default | Read in | What it does |
|---|---|---|---|
| `VERBOSE_LOGGING` | `false` (only `true` turns it on) | `src/rag/verbose_logger.py` (import) | Prints the `[VERBOSE:…]` pipeline records (`vlog`). |
| `CYPHER_LOG_PATH` | `<state dir>/logs/cypher_queries.log` | `src/rag/cypher_logger.py:_log_path` | Cypher query log file. |
| `CYPHER_LOG_MAX_BYTES` | 5242880 (5 MiB) | `src/rag/cypher_logger.py:_max_bytes` | Rotation size of the Cypher log. |
| `CHATBOT_PROJECT_ROOT` | unset, which means the nearest folder holding `pyproject.toml` | `src/rag/cypher_logger.py:_project_root` | Deploy directory used by the Cypher logger. |
| `LOGLEVEL` | `INFO` | `src/preprocessing/main.py` (import) | Log level of the ingestion pipeline. |
| `THESIS_LOG_DIR` | `<state dir>/logs` | `src/rag/lawyer_method.py:thesis_log` | See section 1. |

### 3.9 Numeric caps, targets and modes

| Name | Default | Read in | What it does |
|---|---|---|---|
| `DOCUMENT_TEXT_CHARS` | 40000 | `src/utils/document.py:document_char_limit` | Characters of an uploaded document sent to the model in one call. |
| `SYNTHESIS_DATA_CHARS` | 9000 | `src/rag/nodes/synthesis.py:_synthesis_data_chars` | Characters of retrieved law that the chat answer is written from. |
| `HISTORY_CHAR_BUDGET` | 24000 | `src/chatbot/session.py` (import) | Characters of conversation history one answer may carry. |
| `MODEL_CONTEXT_TOKENS` | 32768 | `src/rag/section_fit.py:context_tokens` (`CONTEXT_ENV`) | Model context that section prompts are fitted into. |
| `PLAN_CAP_MAX` | 12 | `ptw:plan_cap_max` | Substantive articles in the writer's plan under `PLAN_CAP` (minimum 1). |
| `GRADED_MAX_CHECKED` | 0 (no limit) | `dg:_graded_max_checked` | Article slots of the graded check, for an A/B. 0 or invalid means every article. |
| `GRADER_VOTE_WORKERS` | 8 | `dg:_grader_vote_workers` | Concurrent calls under `GRADER_VOTE` (minimum 1). |
| `GRADER_VOTE_TEMPERATURE` | 0 | `dg:_grader_vote_temperature` | Temperature of the `GRADER_VOTE` calls. |
| `VOTE_POOL_MODE` | `append:6` | `src/rag/vote_pool.py:mode` (`MODE_ENV`) | How voted articles join the pool: `append:N` or `rrf:W_POOL:W_VOTE`. An invalid value falls back to the default. |
| `VOTE_POOL_K` | 60 | `src/rag/vote_pool.py:ruling_vote` (`K_ENV`) | Rulings' sections per channel in the vote. |
| `RULING_CARDS_CIVIL_SLOTS` | 3 (clamped 0-5) | `src/rag/ruling_cards.py:civil_settings` (`CIVIL_SLOTS_ENV`) | Ruling-card slots for a civil case. |
| `GROUNDED_REVIEW_POLICY` | `rewrite` | `src/rag/grounded_review.py:policy` (`POLICY_VAR`) | `rewrite` or `abstain` (an unsupported sentence is replaced by a "Non ho trovato…" line). Any other value means `rewrite`. |
| `CCNL_SECTION_ORDER` | unset (auto) | `src/rag/ccnl_path.py:group_ordered` (`ORDER_ENV`) | Forces the key used to order a CCNL's Sections (one of `ORDER_CANDIDATES`). |
| `TEMPLATE_SELECT_FLOOR` | 0.62 | `src/rag/template_selection.py` (import) | Minimum similarity for a template match. |
| `TEMPLATE_SELECT_MARGIN` | 0.10 | `src/rag/template_selection.py` (import) | Margin the best template needs over the next one. |
| `TEMPLATE_SELECT_WINDOW` | 0.05 | `src/rag/template_selection.py` (import) | Similarity window of the templates offered together. |

### 3.10 Operator-script variables

| Name | Default | Read in | What it does |
|---|---|---|---|
| `ENV_FILE` | `.env.v45` | `scripts/serve_test.sh` | Env file whose `KEY=VALUE` lines the test server exports. |
| `PY` | `venv/bin/python`, else `/opt/chatbot/venv/bin/python` | `scripts/serve_test.sh` | Python interpreter for uvicorn. |
| `HOST` | `127.0.0.1` | `scripts/serve_test.sh` | Bind address of the test server. |

`scripts/nightly_sim.py` takes `--env` (default `.env`, loaded with python-dotenv) and `--heldout-dir`, which falls back
to `HELDOUT_DIR` through `heldout.folder()`. `scripts/diagnose_long.py` takes `--env` (default `./.env`) and
`--set NAME=VALUE` (repeatable; it overrides the environment). Its module-level `FLAGS` list is only recorded in each
run's output.

---

## 4. Process notes

- `ROLE_FIT=shadow` asks the role-fit question and logs the answer, but tags nothing and adds no writer rule.
- `LLM_THINKING` is not boolean. `true` sends nothing (the model's default), the same as leaving it unset.
- `CASE_FIRST_SINGLE_SHOT=true` implies `INTERACTIVE_CASE_FIRST` (`case_first_enabled`) and opens the narrative path
  even with `INTERACTIVE_FLOW=false` (`turns_on`).

---

## 5. v77 additions

| Name | Default | Read in | What it does |
|---|---|---|---|
| `HELDOUT_DIR` | unset: `/opt/chatbot/heldout` when `/opt/chatbot` exists, else none (nothing is saved) | `src/chatbot/heldout.py:folder` (`ENV`); used by `src/chatbot/api.py` (`heldout.save`), `scripts/nightly_sim.py`, `scripts/heldout_export.py`, `scripts/heldout_score.py` | Folder of the held-out set. Every case-first answer is saved as `<date>_<id>.json`, and the lawyer's scores go in `scores.jsonl`. A failed save is logged and never touches the answer. |
| `STATE_DIR` | unset: `/opt/chatbot/state` when `/opt/chatbot` exists, else `<checkout>/state` | `src/state_paths.py:state_dir` | Runtime state folder: sessions, tenant logos, system templates, article cards and logs. Each of these keeps its own override (`SESSIONS_FILE`, `SYSTEM_TEMPLATES_CATALOG_PATH`, `ARTICLE_CARDS_FILE`, `CYPHER_LOG_PATH`, `THESIS_LOG_DIR`). |

**`VERSION` is a file, not a flag.** `src/version.py:current()` reads the first line of `<repository root>/VERSION`
(`<semver>+<git short sha>`, written by `scripts/release.sh`) once per process. Without the file it returns `unknown`.
`/api/version` returns it, every case-first answer ends with the footer `SunnitAI <version>` (since v79.12.7 while
`FOOTER_IN_BODY` is on; the reply's `version` field carries it either way), and the held-out records and the nightly
report carry it. No environment variable changes it.

---

## 5b. v78 addition

| Name | Default | Read in | What it does |
|---|---|---|---|
| `RULINGS_RECORD` | false | `src/rag/rulings.py:enabled` (`FLAG`) | The Cassazione rulings into the record (`src/rag/rulings.py`): the rulings citing each record article, with estremi parsed from the file name, ranked, a quoted «passaggio della motivazione» picked by one linking call, computed orientations, §5 rendered by the code with the fixed corpus line, the lawyer's pronunce di riferimento (`src/rag/landmark_rulings.py`). The writer cites only the registry. The thesis and objections calls receive it, and a prevailing line against a contest lowers its realismo to basso. With it on, `select:v1_case_law` (`defensive_generation._select_articles`), the REFERS_TO neighbours in a topic's articles (`topic_step.articles_split`) and `EDGE_EXPAND` stop feeding the pool. Keep it off until `scripts/rulings_parse_rate.py` has been reported and `data/ruling_sections.json` is confirmed. |

## 5c. v79.11 addition

| Name | Default | Read in | What it does |
|---|---|---|---|
| `ELEMENT_SCHEMAS` | false | `src/rag/element_schemas.py:enabled` (`FLAG`) | Element schemas (`docs/RULES.md` 11.5). An article with a usable schema (`element_schemas.for_row`: the generated line of the same raw-text hash with no blocking check, or the lawyer's reviewed one) is decomposed into the schema's elements (`issue_records._requisites`); every other article runs the text path unchanged. The record keeps its path (`rec["element_path"]`, `state["element_paths"]`, log `interactive_element_path`) and its schema (`rec["schema"]`). The schema's tipo joins the general-part gate (`issue_records.decomposed_by_gate(..., schema=)`): it never overrides the hard invariants, and only a reviewed schema gates an article out. The schema's kinds select the grading rule (`issue_records.element_rule`: the mental check, a condition's parts, a termine's date), and only ever lower a status. The admission runs in shadow only (`lawyer_method.schema_shadow`, log `lawyer_method_schema_admission`). A schema's rapporto names the institution module only when the patterns find none. The counters are neutral with it off. |
| `ARTICLE_SCHEMAS_FILE` | `data/article_schemas.jsonl` | `src/rag/element_schemas.py:schema_file` (`FILE_ENV`) | The generated schemas (`scripts/article_cards_bulk.py --schemas`), one line per article keyed (document, article), the last line of a key winning. A relative path is read from the repository root. Read at call time and re-read when it changes. Never deployed over the server's own (`scripts/deploy_exclude.txt`). |
| `ARTICLE_SCHEMAS_REVIEWED_FILE` | `data/article_schemas_reviewed.jsonl` | `src/rag/element_schemas.py:reviewed_file` (`REVIEWED_ENV`) | The lawyer's overlay (`scripts/schema_show.py --edit`, `--import-notes`), keyed by label. A reviewed schema wins over the generated one while its source hash is the article's. Tracked in git and never written by the generator. |

## 5c.1 v79.12.3 additions (no new environment variable)

No new flag either. With `ELEMENT_SCHEMAS` on, an offence's procedibilità also reads the element schema's
«procedibilita» block (`src/rag/procedural_facts.py:with_schema`). The store's own value wins a conflict unless the
schema is reviewed, or the store's d'ufficio is only the general rule and the schema reads the querela in the text's
own words. With the flag off, the procedural facts are exactly the store's.

The element schemas' checker maps the vocabulary's synonyms from `data/schema_vocabulary.json`
(`src/rag/element_schemas.py:VOCABULARY_FILE`, re-read when it changes; a missing file maps nothing). That path is fixed,
not an environment variable. The new options belong to the operator scripts:

| Option | Default | Read in | What it does |
|---|---|---|---|
| `--regen-list FILE` | none | `scripts/schema_show.py --recheck` | Writes the (doc, article) keys that still need regenerating as JSONL (`{"doc", "article", "label", "reason", "detail"}`). These are an unusable line (by its first blocking check) and a line with no blocking check read with no element, every one dropped by the repair or none written (`no_elements`, not counted as usable after). A label the lawyer reviewed is never listed. The file is written with `--dry-run` too. |
| `--only-keys FILE` | none | `scripts/article_cards_bulk.py --schemas` | Only these keys (the `--regen-list` JSONL, or `doc<TAB>article` lines), each made again (`asked`). A reviewed label is never made. |
| `--regen-prompt` | off | `scripts/article_cards_bulk.py --schemas` | Also makes again the lines whose prompt alone changed (`prompt`). Without it they are kept and counted (`prompt_kept`), so a new `SCHEMA_SYSTEM` never remakes the whole file unasked. |

v79.12.5 (79.12.3 item 7: standalone surnames after an article, rule «article») adds no flag either: it reads the role list from
`data/redaction_stoplist.json` «role_nouns» (`src/rag/rulings.py:REDACTION_STOPLIST_FILE`, a fixed path), and
`RULINGS_PARTY_NAMES=keep` still leaves every name, this one too.

v79.12.5 item 0 (the element schemas' logic pruner and vocabulary) adds no flag or option: the new synonyms and the
«ops» map are in `data/schema_vocabulary.json` (`src/rag/element_schemas.py:VOCABULARY_FILE`, a fixed path), and a
`--regen-list` key now also carries `"unknown"` (the line's values no vocabulary maps) when it has any.

v79.12.5 items 2, 6, 10 and 11 add no flag: the no-claim sections, the role rules, the counterparty's qualification
and the consumer code's scope, the scope defaults (`src/rag/topic_groups.py:GENERAL_CLAUSES` and the stage table,
`data/stage_procedure.json`), the closed list kept in §8 and the main-outside rule are unconditional (the method's parts
under `LAWYER_METHOD`, as before); the nightly's two-turn cell is a «followup» key in `tests/data/nightly_cells.json`,
and `--answers DIR` reads its second answer from `DIR/<cell id>.followup.md`.

v79.12.5 items 8 and 9 add no flag: the citation table is `data/corpus_expected.json`'s archive names
(`src/rag/corpus_laws.py:EXPECTED_FILE`, a fixed path), the 168-bis draft status stays in `data/lawyer_method.json`
«status», and the ledger and the rulings' date fixes are unconditional (`RULINGS_RECORD` still decides whether §5
prints rulings at all).

v79.12.7 items 1, 3, 4 and 5 add no flag: the jurisprudential elements, the absorbed titles, the riqualificazione's
facts and the art. 41 cap are unconditional under `LAWYER_METHOD` (penal postures only), their tables in
`data/penal_lines.json` (`src/rag/penal_lines.py:DATA_FILE`, a fixed path). With `ELEMENT_SCHEMAS` on, a reviewed
schema's element carrying a jurisprudential element is used as is (tagged, never copied); off, the note
(`article_notes`) and the kept ruling passages (`RULINGS_RECORD`) feed it.

v79.12.7 items 2 and 6 add no flag: the attenuanti from the facts (`data/penal_lines.json`
«circumstance_triggers», «bilanciamento», «exclusion_patterns») and the riti's printed computation
(`data/lawyer_method.json` «riti_alternativi»: «computation», «computed_line», «possible_line») are unconditional under
`LAWYER_METHOD` (the imputato before the sentence, penal postures only, where the riti were already printed).

v79.12.7 item 7 adds no flag: the text residue's tables are `data/text_hygiene.json` (`src/rag/text_hygiene.py:
DATA_FILE`, a fixed path) -- the stance marks, the draft-status wording, the procedural acts, the cenni's rule verbs
and thresholds, the date forms --, and its checks are unconditional: the answers block's and the cenni's under
`LAWYER_METHOD` (compose), the procedural acts' certainty in every record (`issue_records.certainty`), the last pass
over every case-first answer (`interactive_flow.text_residue`). Item 8 adds `FOOTER_IN_BODY` (section 1).

v79.12.7's review fixes add no flag: the new tables (`data/penal_lines.json` «denial_patterns», «not_by_offender_patterns»,
the riqualificazione's and the art. 41 triggers' negation patterns, «step_exclude_patterns», «rite_prose»;
`data/text_hygiene.json` «procedural_act_modals», «code_context_patterns», «cenni_absolutes») are read by the same
unconditional checks as above.

## 5d. v79.10 addition (addendum A5)

| Name | Default | Read in | What it does |
|---|---|---|---|
| `FAMILY_GATE_SHADOW_GRADE` | false | `src/rag/interactive_flow.py:family_shadow_grade` (`FAMILY_SHADOW_GRADE_FLAG`) | The family gate's alarm (`docs/RULES.md` 11.6). After round 1's grading, the retrieval rows the family gate dropped (distinct articles, at most 40, by retrieval order) are graded by the same grader in their own isolated call, and the result is only recorded: `state["family_gate"]["dropped_graded"]` {doc: {graded_relevant, articles}}; nothing else reads it. Off: a proxy count, never graded ({doc: {dropped_articles, articles, proxy: true}}, «stima (non graduati)»). `scripts/nightly_sim.py` sets it to true for its run (`--no-family-shadow-grade` skips it). It costs one more grading call per case when a document was dropped, so leave it off on the live copy. |

## 5e. v79.13.0 addition

| Name | Default | Read in | What it does |
|---|---|---|---|
| `APPLICABILITY_FIT` | false | `src/rag/applicability.py:enabled` (`FLAG`) | The applicability judgment (`docs/RULES.md` 11.12). For every admitted candidate norm the model judges whether the provision governs the case's facts (fit si / no / dipende, a perché quoting the article and naming fact ids, a condizione for dipende, an ambito), in batches of at most 8 per call (`data/applicability.json` «batch_size»), up to 4 at once («concurrency»), thinking off, temperature 0, within a 20 s budget («deadline_s»); the code verifies every word of each judgment against the article's text, the facts and the reviewed note before it has any effect -- an unverified judgment leaves the norm on its current path. Runs once per answer in `src/rag/interactive_flow.py:_write_answer` (`applicability_fit`, through `decompose_records`' `fit` hook: after the family gate and the cap, before the decomposition; the multi-client and follow-up answers too). Effects: «no» removes the norm from the records, the topics, the candidates and §4 (`state["fit_removed"]`, read as dead by every later step and answer; a last pass drops any sentence still citing it; put back and judged again when the user later names it or the facts change), unless the user names it or the text charges it (its special-law alias too) -- then kept and printed as the defence's contest (the prose answer too); «dipende» keeps it, its touched elements «da provare» whatever any later grading says, the condizione in §10 as the user's question. Off: nothing runs, the output is 79.12.7's; turned off after an answer that ran it, its effects are undone (`interactive_flow.fit_clear`) and never read. One more model call per 8 candidates, the step capped at 20 s from its start. `scripts/nightly_sim.py` sets it to true for its cells (`--no-applicability-fit` skips it). v79.15.0 (item 3, not behind a new flag): the judgment is verified by ids, the string matcher removed -- the article goes as its numbered periods (A<art>.<n>), a «si» cites them («cita_norma»), a «no» and a «dipende» cite the facts («cita_fatti», S-ids; a system assumption is no fact); a missing or unknown id leaves the norm on its path (`fit_unverified_ids`); a «no» never removes the verified frame's addebito/domanda article nor a user-named one (`fit_protected_kept`). |

## 5f. v79.14.0 addition

| Name | Default | Read in | What it does |
|---|---|---|---|
| `CASE_FRAME` | false | `src/rag/case_frame.py:enabled` (`FLAG`) | The case frame (`docs/RULES.md` 11.13). One model call at the top of the pipeline reads the case frame from the request's full text -- parti (nome, ruolo_nel_fatto, lato cliente / controparte / terzo / autorità), cliente, ruolo, rito, stadio (each with a quoted perché), addebito_o_domanda (articolo or null, descrizione, frase), istituto, quesiti --, thinking off, temperature 0, a JSON reply, within 15 s (`data/case_frame.json` «deadline_s»); the code verifies every field against the text with the 79.13 fit checker's quote match (`applicability.passage_at`) before it counts: a field that fails is dropped (`frame_unverified.<field>`) and intake's and posture's rules fill it (`frame_fallback.<field>`); a cliente or a ruolo with no quote is never accepted, a party listed as controparte is never the client, a penal rito's charge article is a penal provision or is dropped (`charge_rito_dropped`). The verified frame is bound across the pipeline -- intake's rito, role, stage, client and questions (confidence `frame`), intake.resolve, posture.of, posture.choose_charge, the stated / settled / multiple clients, the first read's client and institution, the method's header and thesis call --: a later step that disagrees is logged `frame_conflict` with its name and loses; a verified «sconosciuto» role prints the honesty line. Counters on the delivered answer: `frame_conflict`, `role_flip`, `charge_rito_mismatch` (each must be 0), `frame_conflict_attempts`. A failed call, a reply that is no JSON object or a call over 15 s leaves every field to the rules (`frame_failed`), never a crash. With `APPLICABILITY_FIT` also on, the verified frame (`state["case_frame"]`) goes to the fit judgment as context (`case_frame.fit_context`, `interactive_flow.applicability_fit`). No call for a request no case pipeline reads (a plain question with no facts: `intake._frame_wanted`). Off: nothing runs, the output is 79.13.0's. Not `CASE_FRAME_ENABLED` (the older frame of `dg._case_frame`, another path). `scripts/nightly_sim.py` sets it to true for its cells (`--no-case-frame` skips it) and prints the frame per cell; `scripts/frame_timing.py` times the call. |

## 5g. v79.15.0 addition

| Name | Default | Read in | What it does |
|---|---|---|---|
| `ANALYSIS_STEP` | false | `src/rag/analysis_step.py:enabled` (`FLAG`) | The verified analysis step (`docs/RULES.md` 11.17). One model call between the case frame and rendering (thinking off, temperature 0, a JSON reply, within 40 s for the whole step, `data/analysis_step.json` «deadline_s») decides the leading and the subordinate line and every schema element's esito (provato / contestato / da_provare / escluso), each claim citing ids the code verifies: the case's sentences S1..Sn (the system's assumptions never sent) and the fitted norms' numbered periods A<art>.<n> (`applicability.article_ids`, the fit's own numbering). An unknown id voids its claim; an element not verbatim is dropped; «provato» resting only on a non-client party's or an authority's assertion becomes «contestato» with the party; «escluso» needs an S-id; a line on a remedy / general-part article (`data/remedy_articles.json`; 2043 c.c. only on a contractual frame) or missing the frame's verified addebito/domanda article is re-asked once (at most two calls), then «da determinare» or the frame's article with every element «da_provare»; a law statement with no verifying A-id of the article it names, or not grounded in the periods it cites, is dropped; «provato» downgraded whenever any asserting phrase of a cited sentence is another party's or an authority's, even beside the client's own words. A failed, late or non-JSON call: nothing from the step. `analysis_step.run(state)` computes, returns and logs («analysis_step»: prompt_hash, status, seconds, calls, counters, the verified lines); the flow runs it once per answer after `applicability_fit` (`interactive_flow.analysis_step_run`) and, under `LAWYER_METHOD`, §1, §4's element tables (the Esito column), §6, §9, every «x su y» count and the answers' line come from the verified analysis only (remedy / general-part articles named in §4 under the line, no table, no count; an undetermined line printed «da determinare (linea non verificata)»; a Q&A or §4 sentence stating what a norm provides with no verified affermazione_di_diritto removed: `qa_sentences_dropped`, `norms_sentences_dropped`); the old engine still runs, logged «legacy_plan» and compared (`legacy_disagrees_line`, `legacy_disagrees_esito`), never rendered -- its no-claim verdict writes no section either. Off: `run` returns status «off», nothing is read or called, the output is 79.14.3's (with the fit by ids, 79.15.0 item 3, which is not behind this flag). We set it true in `.env.v45`. |

## 5h. Local additions, 22 Nov 2026 (not released, set in no copy)

| Name | Default | Read in | What it does |
|---|---|---|---|
| `FACT_RECORD` | false | `src/rag/fact_record.py:enabled` (`FLAG`) | The fact record of the main user query: a temporary JSON built once per case beside round 1 (`fact_record.start` in `interactive_flow.analyse_case`), kept in the state as «fact_record». The model returns the text's elements in order, each with the exact words it is made of, its kind (fatto / richiesta / istruzione / estraneo) and, for a fact, its status and who alleges it; the code verifies (a quote not in the text is dropped; a fact's certainty is the more cautious of the model's and the rule's; a speaker must be in the text; what the model leaves out comes back by the rule). `issue_records.sentences(facts=)` numbers the record's facts S1.. in place of the rule's split, uncut and with no ceiling on their number; requests, instructions and extraneous text (a pasted earlier answer) are never a sentence. A failed or late step: no record, the rule's sentences as with the flag off. Prompt and numbers: `data/fact_record.json` (deadline 60 s, pieces of about 2,000 characters read in parallel). Logged «fact_record». Off: nothing is called, the output is unchanged. |
| `USER_TEXT_UNCAPPED` | false | `src/rag/user_limits.py:enabled` (`FLAG`) | A test switch: the fixed cuts on the user's own text are lifted (`user_limits.cap` / `most` at each site: the case text 12,000 characters, the numbered sentences 260 characters and 80 in number, the grader's 2,000, the distill's and the candidate choice's 6,000, the fact issues' 8,000, the opposing-arguments check's 4,000, the plan patches' 6,000, the rulings link's 1,500, the writer's quoted sentences 160, a question's 300, the frame's 40,000, §2's 220). Not lifted: the text sent to the embedding model (4,000), §2's number of facts, any cut on text the model or the code wrote. With it on a very long case can overflow a call's context: the helper retries with a smaller answer budget or the step fails and is logged. Off: every cut as before. |

## 5i. v79.16.0 (23 Nov 2026): the fact record drives every stage; questions drive retrieval; Q&A assembly; one opinion

| Name | Default | Read in | What it does |
|---|---|---|---|
| `QUESTION_RETRIEVAL` | false | `src/rag/nodes/retrieval.py:question_retrieval_enabled` (`QUESTION_RETRIEVAL_FLAG`) | B: each verified quesito of the case frame (none: each numbered question of the user's text, `intake.question_items`) is one more retrieval query through the retriever the fact queries use (`defensive_generation._fusion_text_search`, the same index and k), its rows fetched as any candidate's and merged into the candidate set with origin «quesito:n» (`retrieve_for_quesiti`, `merge_quesito_rows`, called from `interactive_flow._retrieve` after the first read's search, graded past the cap as its hits are). They pass the gates, the grading and the applicability fit unchanged; an article a quesito alone brought is offered first to that question's writer call (`interactive_flow.quesito_articles_block`). Counters `retrieval_quesiti_queries`, `retrieval_quesiti_added` (labels), `fit_kept_from_quesiti`; log line «retrieval_quesiti» with each quesito's top-3. The fact-driven queries, their ranking and k untouched. We set it true in `.env.v45`. |

Under the existing flags: `FACT_RECORD` now gates A -- with the record «ok» every consumer of the case's sentences reads
its facts (the analysis step's input keeps each sentence's F-id and speaker; `analysis_step.record_provenance`: a
«provato» stands on a fact stated accertato, or alleged by the client, or alleged by the other party against its own
interest, else contestato naming the party -- counters `analysis_provenance_from_record` / `analysis_provenance_fallback`,
the text heuristic only without the record; `fact_record_fallback` = 1 when the record is absent or partial and the
rule's split stands in; §2 from the record (`analysis_step.record_facts_split`: certi / ipotizzabili by the record's
stato, «Fatti mancanti» the verified lines' open and contested elements, never «Nessun elemento resta da provare» while
the tables hold one); §10 the same open elements and the record's uncertain items (`fact_record.uncertain_items`), §4's
prose not scanned; the F-ids in the «analysis_step» log line (`facts_cited`), the facts' text on the page).
`ANALYSIS_STEP` now also gates C and D -- C1: the questions are the user's numbered items whole (`intake.question_items`,
read first by `intake.questions` / `quesiti`, the frame's verified quesiti mapped to the whole item; counter
`qa_questions_detected`); C2: a writer's answer the review leaves under two sentences is re-asked once with the verified
affermazioni and the quesito's articles as its only law (`interactive_flow.reask_question`, `writer_rules.REASK_RULE`),
still empty: «Il corpus non contiene informazioni sufficienti per rispondere a questo quesito.» (counters
`qa_answer_reasked`, `qa_answer_insufficient`); C3: a replacement statement must cite an article the answer itself cited
(`ProseReview(cited_scope=…)`, else dropped); D1: every writer call carries the one verified context block
(`analysis_step.context_block`, `state["verified_context"]`); D2: code bridges open §3, §5, §6, §9 (`question_bridge`,
`rulings_bridge`, `SOLUTIONS_BRIDGE`, the §4 counts first in `conclusions_text`); D3: a sentence printed once across the
answers and §1..§10 (`dedup_sections`, counter `sections_dedup_dropped`), §5 only on the lines', the quesiti's and the
remedies' articles (`rulings_view`, counter `rulings_off_line_dropped`); D4: no §4 prose on a remedy article
(`remedy_prose_out`). Measurement only: `line_table_weaker` = 1 when the subordinate line holds more provato elements
than the leading one.

## 5j. v79.16.1 (24 Nov 2026): fit by ids only; fit verdicts feed the analysis; pertinence; the quesiti gate; §8 by rito

No new flag. Under the existing ones: (1) `QUESTION_RETRIEVAL` -- the gate's decision is logged on every case under the
plain event name `retrieval_quesiti` (79.16.0 wrote `interactive_retrieval_quesiti`, which a grep for the plain name
missed): `queries`, `source` (frame | numbered_items | questions | none) and, with no search, `reason` (flag_off |
not_case_first | no_questions); the sources in order are the frame's verified quesiti, the text's numbered items
(`intake.question_items`), the text's explicit questions (`intake.questions` -- the server's cases whose questions sat
inside the narrative), none. (2) `APPLICABILITY_FIT` -- a judgment is verified by ids only (`applicability.verify`): a «si»
and a «dipende» cite the article's periods and the facts, a «no» the facts, every id existing; the perché, the ambito and
the condizione are recorded and logged, never verified (the content-word, one-sentence, quantity and polarity checks are
gone: they discarded true verdicts for a word). The prompt's schema asks a «si» for its facts too (prompt hash
5b06ff45a132 → 677c05970c73). Counters `fit_si_verified`, `fit_dipende_verified` beside `fit_no_verified` and
`fit_unverified_ids`; each stored judgment carries `cita_norma` / `cita_fatti`. (3) `ANALYSIS_STEP` -- each norm of the
step's input carries its fit verdict, shown in the prompt's article block as «fit: si (verificato) — <perché>» (the system
text unchanged, hash 472423ea32d3); a line stands only on the allowed tier (`analysis_step.line_tier`: the verified «si»,
else the verified «dipende», else the unverified «si» / «dipende», never a «no»); counters `line_from_fit_si`,
`line_from_fit_dipende`, `line_from_unverified`. A procedural, temporal or remedial element is never the element in play
(`procedural_element`: the schema's kind, else `data/analysis_step.json` «procedural_patterns» on the element's text); a
pick on one is refused and re-asked once with the allowed E-ids (`analysis_line_procedural_rejected`). (4) a «provato» or
an «escluso» needs one cited fact sharing a content lemma with the element's text (`pertinent`: four-letter words cut to
five characters, «pertinence_stopwords» and the parties' names out); none: «da_provare», logged in the step's line
(`no_pertinence`) and counted (`analysis_provato_no_pertinence`); a «contestato» is unchanged. (5) §8's blocks are
selected by the verified frame's rito (`lawyer_method.rito_blocks`: the civil blocks never in a penal opinion, the penal
ones never in a civil or labour one, a «Termini» line by its code; counter `sections_rito_blocks_dropped`); the
procedural facts of the charge article are read from every row of the answer, not the chosen refs' rows alone.

## 5k. v79.16.2 (24 Nov 2026, hotfix): two 79.16.1 rules back to log-only; the fallbacks on a failed step

No new flag; under `ANALYSIS_STEP`. Server evidence on 79.16.1: 640 c.p. elements proved by facts such as «si presenta
come intermediario autorizzato» were downgraded to «da provare» by the pertinence lemma check (legal element words never
occur in plain facts); a penal case that led on 646 c.p. in 79.16.0 printed «da determinare» because the unverified 646
was refused beside a verified article and the re-ask timed out. (1) Pertinence (`analysis_step.pertinent`) never changes a
verdict: the esito stays as the model gave it after provenance; the check is computed, counted
(`analysis_provato_no_pertinence`) and logged in the step's line (`no_pertinence`, the element and the cited facts).
(2) The fit tier is a preference, never a gate: `line_tier` lists every allowed article in the order verified «si» →
verified «dipende» → unverified or unjudged «si» / «dipende» (never a «no», never a remedy); the article block of the
prompt tells the model each norm's rank («Preferenza per la linea (prima il fit si verificato, poi dipende verificato, poi
non verificato): 1/2» -- the human message; the system text unchanged, hash 472423ea32d3) and the re-ask's allowed list is
in that order, each article tagged («[fit si verificato]», «[fit si non verificato]», «[fit non giudicato]»); `_verify_line`
refuses a fit «no» and a procedural element only; a chosen line on an unverified or unjudged norm counts
`line_from_unverified` = 1 and nothing else. (3) A first call that failed, timed out or answered no JSON object runs
`finish` on no reply (`empty_checked`): the frame's charge (`addebito del quadro, linea non verificata`), else the rubrica
fallback, else «da determinare», in that order, on every path; `analysis_fallback_skipped_reason` says which
(frame_charge_used, "" when the rubrica fallback ran, no_named_charge, no_rubrica_match; step_failed only for an input
that cannot be read); the step's status stays failed / timeout / malformed and `analysis_failed` = 1, the view and the
rendering read the fallback line. The section dedup (`dedup_sections`) no longer splits a sentence after a code or article
abbreviation («art. 646 c.p. (addebito del quadro, …)» is one sentence): 79.16.0 dropped §1's label as the Q&A line's
duplicate.

## 5l. v79.17.0 (24 Nov 2026): the pretesa anchors every rito; the frame asks instead of guessing; questions bind
## articles; prose from verified material only; the sections from one object

No new flag. (1) `CASE_FRAME` -- the frame's `addebito_o_domanda` field is the pretesa of every rito: {tipo (accusa | domanda
| ricorso | impugnazione | opposizione | istanza), contenuto, contro, rito, articolo, frase}; the prompt asks it as «pretesa»
(hash 01439962c7c9 → 44d4b100df83; an «addebito_o_domanda» reply is still read). Sources in order:
the text's explicit words (verified by quote), the verified quesiti, the text's last request (`case_frame.derive_pretesa`:
`derivata` true, fonte quesiti | ultima_richiesta, counter `frame_pretesa_derived`; `frame_pretesa_verified` when the
model's held). Ask, don't guess: cliente, ruolo or pretesa unverified and the text naming no side (`case_frame.must_ask`,
`side_named`: no «assisto …», no «il mio cliente …») → the flow asks ONE question («Per chi è reso il parere e che cosa
chiede?», data `ask_side`) and stops (`interactive_flow.ask_side`, reply kind `ask_side`, counter `frame_asked_side`); the
next message runs the case again as one text (`answer_after_ask`), never a second question. The text names the side but the
pretesa is derived → §1 opens with «Pretesa dedotta dai quesiti: «…».». (3) `APPLICABILITY_FIT` -- the fit's human message
carries «PRETESA» and the numbered «QUESITI» (Q1…); every verdict adds «governa» (stored `governs`: ["pretesa"] and/or
["Qn"], [] = context only; absent = nothing said), hash 677c05970c73 → c60a2f78634c; counters `fit_governs_pretesa`,
`fit_governs_quesiti`, `fit_context_only`; a question's retrieval articles are kept for its answer only when the verdict
governs it (`interactive_flow.quesiti_articles`). (2) the binding on the pretesa (`interactive_flow.binding_pretesa`,
`analysis_step.governs_sets`): with a verified «si» / «dipende» governing the pretesa, the leading line's candidates are
those articles, the subordinate's those and the quesiti's; a context-only norm is never a line (`analysis_line_context_rejected`),
a quesito-only one never the leading line (`analysis_line_quesito_only`); the article block tells the model «governa: …»;
counters `binding_pretesa_applied`, `binding_pretesa_excluded`. (4) `ANALYSIS_STEP` -- with the step ok, every Q&A, §4 and
§7 writer call receives ONE block (`verified_material.block`): the pretesa, the client and its side, the verified lines and
esiti, the section's articles with their cited periods as ids ([A1385cc.1]), the facts as ids ([S3]); it must cite the ids
inline, the code renders them. The post-processing of that prose is one rule (`verified_material.review_*`): a sentence
citing an article outside the fitted set (or an excluded one) or an id that does not exist is dropped -- no replacement,
no dependent-sentence drop, no lemma matching, no rule-statement check on those sites (counters as before:
`qa_sentences_dropped`, `prose_unfitted_article_dropped`, `prose_excluded_article_dropped`, new `prose_unknown_id_dropped`;
`qa_sentences_replaced` is 0 by construction). An answer losing two or more sentences is written once more with the same
material; a fragment or an article's text is never an answer (`qa_answer_fragment`, then the insufficient line). The
«Affermazioni di diritto verificate» block is the log's only (`compose` log «statements_block»). (5) one object
(`analysis_step.attach_object`: pretesa, client, side, lines, esiti, articles by governs, record facts, open items): §1
opens with the pretesa line, §3 with «La questione è se <la pretesa> sia fondata sull'art. X, posto che …», §9 adds
«Rischio per <cliente>: favorevole | incerto | sfavorevole» (`risk_word`, computed from the leading article's esiti, read
the other way round for a client resisting the pretesa), §10 lists the frame's unverified fields, §5's rulings only on
the lines' and the governs' articles; «Il materiale non indica obiezioni» prints only when the record holds no allegation
by a non-client party (`lawyer_method.record_allegations`). `data/institution_laws.json`: every entry carries `family`
and `tags`; the lavoro, amministrativo, tributario, famiglia and fallimentare sets added (data only). `scripts/replay_cases.py`
reports the pretesa, `asked`, the frame and binding counters; `data/replay_cases/batch_2/` holds six more synthetic cases.

## 5m. v79.17.1 (24 Nov 2026), items 6-9: the questions through the family gate with the frame's rito; no-element
## articles context only; the replay's seconds per stage; the pretesa's binding always has a set; a code outside the rito

No new flag. (6) `QUESTION_RETRIEVAL` + `CASE_FRAME` -- the questions' rows pass the same family gate as the facts' rows
before they join the candidates (`interactive_flow.gate_quesito_rows`, source «quesiti»): an article outside the case's
families is not retrieved for a quesito and never enters «by_quesito» (counter `retrieval_quesiti_gated`; the log line
`retrieval_quesiti` lists `gated`). The gate reads the frame's verified rito (`case_frame.verified_rito`, carried as
`analysis["frame_rito"]`; a rule-filled or unknown rito restricts nothing): `data/special_laws.json` «rito_families» maps
each rito to the special-law families it admits beside the codes (`special_laws.rito_families`, `family_gate.context(rito=)`,
`decide` → `rito_refused`, why «fuori dal rito <rito> (<famiglia>)» -- after the user's naming, before the rules and the
doors; the codes and an unknown family never gated). The facts' rows pass the same rule (counter `family_rito_dropped`,
`family_gate.rito_dropped`). The table is Code's draft for the lawyer's review (data only). (7) `ANALYSIS_STEP` -- a fitted
article with no schema element is context only: not in the prompt's article block (`analysis_step.listed_norms`; the remedy
set stays listed, marked), in no line list (`line_tier`), refused as a line (`_verify_line`: «nessun elemento dello schema,
solo contesto»), counted once from the input (`candidates_no_elements_excluded`, log `context_only_no_elements`). The
system prompts are unchanged (analysis_step 472423ea32d3, applicability c60a2f78634c, case_frame 44d4b100df83). (8)
`scripts/replay_cases.py` reports `stages` per case -- frame, record, extraction, retrieval, fit, analysis, write, total
seconds as the flow measured them (`stage_seconds`), in the table («stages (s)») and the diff («seconds.<stage>»); no
optimisation. (9) `APPLICABILITY_FIT` + `CASE_FRAME` -- the binding on the pretesa always has a set: after the fit, when no
verified verdict governs the pretesa (`fit_governs_pretesa` 0), one more call (`applicability.pretesa_recall`: the pretesa
alone and the fitted articles -- the verified «si» / «dipende» -- by their ids; its own prompt `pretesa_recall.prompt_system`
in `data/applicability.json`, hashed apart, `recall_prompt_hash` 1a584033916f; its own budget `deadline_s` 12 after the
fit's 20) answers the ids that govern it, possibly none; the code keeps the ids it gave and drops the rest (`recall_verify`),
and the kept articles get «pretesa» in their verdict's governs (`governs_recalled`), so `binding_pretesa` («recalled»), the
analysis step and the writer read one set. Counters `fit_pretesa_recall` (the call made), `fit_pretesa_recall_empty` (the
verified answer empty: the binding stays unapplied, as before); log `applicability_pretesa_recall`; the replay's row
«binding / recall». `fit_governs_pretesa` stays the first pass's count. Also `ANALYSIS_STEP` -- an article whose code is
outside the frame's rito (`data/analysis_step.json` «rito_codes»: cp, cpp for the penal riti; cc, cpc for the civil ones;
`analysis_step.outside_rito`, the frame's rito valore else intake, as `penal_frame`) is never the leading line and is a
subordinate line only when the fit tagged it as governing the pretesa or a quesito (`governs_tagged`); otherwise context (the
1439 c.c. case of the persona offesa). Listed in the prompt's block (the step reads it), out of the leading line's list,
in the subordinate's only when tagged (`line_tier`, the re-ask's «[solo linea subordinata]»), refused as a line otherwise
(`_verify_line`, counter `analysis_line_outside_rito_rejected`; log `outside_rito`, `rito`). A rito not in the table, no
rito, a special law or the Costituzione: nothing restricted. The system prompts of the fit and the step are unchanged.

## 5n. v79.17.2 (24 Nov 2026): the governs-pretesa set is substantive and always recalled; the leading line in the set,
## always; the honesty line

No new flag. (1) `APPLICABILITY_FIT` + `CASE_FRAME` -- the candidates for the governs-pretesa set (a verdict's tag or the
recall's answer) are substantive articles only: at least one schema element (the record's requisites), not in the remedy
set (`data/remedy_articles.json`, `when_contractual` with a contractual frame), not of a procedural code
(`data/applicability.json` «procedural_codes»: cpc, cpp, cpa) -- `applicability.substantive`. The recall always runs when a
pretesa is given, over the substantive articles the fit did not remove (a verified «si» / «dipende», an unverified verdict,
an unjudged norm -- never a verified «no»: an unverified verdict is no reason to call the corpus silent), and its answer is
united with the verdicts' «pretesa» tags on substantive articles (`applicability.apply_recall`). It runs after the records
are decomposed (`interactive_flow.pretesa_recall_step`, after `decompose_records`: a named or charged article gets its
requisites there, after the fit), and the binding is computed again there (`binding_pretesa`: «enforced» -- a pretesa was
given; «nonsubstantive_tagged» -- a tag the set refused; counters `binding_set_size`, `binding_empty_honesty`); its seconds
go into `fit_seconds`. (2) `ANALYSIS_STEP` -- the step reads the flow's set (`make_input(binding=)`, `inputs`;
`governs_sets` «enforced»: the set among the step's norms, possibly empty, applied whatever the norms' tags): the leading
line must be in it, always -- an article outside it, a norm the fit never judged, a quesito-governed article are refused
(`_verify_line`, counter `analysis_line_outside_set_rejected`); the subordinate may be the set's or a quesito-governed
article; the frame's charge and the rubrica fallback pick inside the set only (`finish`, `rubrica_fallback(allowed=)`). An
empty set: no line and no fallback pick -- the honesty line «Il corpus non copre la pretesa: nessun articolo applicabile
la governa.» (`HONESTY_LINE`, my draft) printed in §1 before the lines (`summary_text`), the leading line «da determinare»,
`analysis_fallback_skipped_reason` «binding_empty», counter `analysis_binding_empty_honesty`. Why 79.17.1 accepted 1225 c.c.
and 2043 c.c. outside the set while refusing 1454 c.c.: the step recomputed the set from its own norms' tags and treated an
unjudged norm as allowed; in caparra the set's only article (1385 c.c.) was not among the step's norms and in
infiltrazioni it (1218 c.c.) was in the remedy set, so the step saw no binding and picked freely; in opposizione the set's
645 c.p.c. was among the norms, the binding held and 1454 c.c. (a quesito's) was refused as the leading line. The replay's
row carries «honesty line» and the counters. The system prompts are unchanged (analysis_step 472423ea32d3, applicability
c60a2f78634c, the recall 1a584033916f).

## 5o. v79.17.3 (24 Nov 2026): the set is the union of the tags, the recall and the fit's verified «si»; the pretesa's
## own article; the honesty line only on an empty union

No new flag. (1) `interactive_flow.binding_pretesa`: the governs-pretesa set is the union of the verdicts' «pretesa» tags,
the recall's answer and every substantive article the fit verified «si» (the verified «dipende» ones when there is no
«si») -- the fit already judged these, the recall only adds, never decides alone; «reasons» {label: tag | recall |
fit_si | fit_dipende | pretesa} in the log and the replay's row. (2) The honesty line (79.17.2) only when that union is
empty: no substantive article fits at all. (3) The pretesa's own article (`case_frame.charge_label`) enters the set when
its verdict is not a verified «no» and it has elements (`has_elements`, set by `applicability.apply_recall`); a
frame-named article the fit rejects with a verified «no» is dropped from the pretesa before the recall
(`pretesa_recall_step`: `addebito_o_domanda.articolo` → None, `articolo_scartato` keeps it, the fit's pretesa text
recomputed; log `pretesa_article_dropped`, counter `frame_pretesa_article_dropped`) -- the 640 c.p. for lesioni colpose
case. (4) Everything else of 79.17.2 stays: the set enforced on the leading line, a procedural or no-element article
never a line (an article in the set with no admissible element gives «da determinare» without the honesty line). The
system prompts are unchanged.

## 5p. v79.18.0 (24 Nov 2026): strategy as a verified step; set tiers; set fallback

No new flag: the strategy step runs under `ANALYSIS_STEP`, after the analysis step and the rulings record, in place of the
objections call (`interactive_flow.method_objections`: with a strategy input the old `lawyer_method.objections_plan` is
not called). (A1) `interactive_flow.binding_pretesa`: the governs-pretesa set has tiers -- «tiers» {label: si_verificato
| dipende_verificato | non_verificato}: every substantive verified «si», then the verified «dipende», then the unverified
«si» / «dipende» («fit_non_verificato»: present, untrusted), in that order; the honesty line only when no substantive
article has any «si» / «dipende» verdict (counter `binding_set_unverified`). (A2) `analysis_step.set_fallback`: the
model's line failed twice and the set is not empty -- the set's best-tier article leads (stato `insieme_governante`, every
element «da_provare», tesi on its rubrica; counter `analysis_line_fallback_set`, fallback reason `set_line_used`); never
«da determinare» with a non-empty set. (B) `src/rag/strategy_step.py`, `data/strategy_step.json` (deadline 40 s, one
re-ask; prompt hash logged): the input is ids and verbatim texts only -- the pretesa and the side, the set with its tiers
and each article's elements with their verified esiti and S-ids (`state_input`, from `state["analysis_norms"]`,
`binding_pretesa`, the analysis view), the facts, the counterparty's allegations, the registry's rulings on those
articles as G-ids (`rulings_for`), the quesiti; the reply's linee (2-3: articolo, eid, tipo fondare | contestare |
riqualificare | eccepire, rapporto, perche_prima, cita_fatti, cita_pronunce, obiezione with risposta_cita_fatti or
«nessuna risposta nei fatti»), passi, and (item 12) passi_immediati, contro_difesa (one per allegation, the code adds
«senza risposta» for a missing one), verifiche_documentali (id-cited or dropped). Verified by code: ids exist, the
article is in the set, the eid is its, «contestare» only on contestato / da_provare, «fondare» only on provato (else
rejected, re-asked once). forza is the code's, by side: the defence's the contestato + da_provare count on the line's
article; a claimant's (item 10) the eid's esito first, then fewer further conditions in the same comma, then the provato
count. Ranking: a linea above one with strictly greater forza only citing a G-id, else reordered by forza
(`strategy_rank_reordered`, both orders logged; ties keep the model's order); the first ranked is the principale.
Rendering (`lawyer_method.compose`, the data's labels): §1 the ranked lines (`summary_text` with `lines`), §6 in the order
passi immediati -> linee per forza (the table, forza «provato x / contestato y / da provare z») -> risposta ai quesiti
(the fit's Q tags, else the quesito's own search) -> contro-difesa -> verifiche documentali, §7 the obiezioni with the
cited facts as written or «senza risposta nei fatti narrati», §9 the risk word from the principale's forza by the
data's thresholds (`risk`, by side) and §1's lines, §10 the passi (an element a passo names is not asked again). The
step failing, timing out or verifying nothing: the analysis step's rendering as before. Counters (block_counters):
strategy_calls, strategy_reasked, strategy_lines, strategy_rank_reordered, strategy_line_rejected,
strategy_objections_unanswered, strategy_rulings_cited, strategy_seconds, strategy_failed, strategy_bad_id,
strategy_passi_added, strategy_contro_difesa_added, strategy_entries_uncited; `state["strategy"]` and the replay's
«strategy» row (ranked articles + forza + the reordered flag), the «strategy» stage in the seconds. (Item 11)
`analysis_step.degree_grade`: an element with a degree word (data `degree_words`, by grade) is «provato» only when a
cited fact holds the same or a stronger degree, else «da_provare» (`analysis_degree_downgraded`, logged
«degree_downgraded»). (Item 13) `data/mediation_matters.json`: «appalto» among the art. 5, co. 1 matters beside the
contratti d'opera (post-Cartabia, flagged `da_confermare` for the lawyer), out of the facoltativa list;
`fact_signals.mediation` reads the matters from the data (`mediation_subjects`), the typed contract's listed matter
decides first, never a clause word; with the article's text in the corpus only a matter the text names. A record fact
the client stated is certo in §2, never «allegazione», and the client is never the party «che afferma» of a contestato.
The analysis, fit, frame and recall prompts are unchanged.

## 5q. v79.18.1 (24 Nov 2026): section C of 79.18.0 -- the penal defence review

No new flag. C1 `strategy_step.defence_rank` / `forza`: the defence's order is (a) an element contestato by the client's
own facts (a cited fact accertato or stated by the client: `mark_own`), (b) an element da_provare where a ruling the
linea cites (its `cita_pronunce` or `condizione`) sets a condition the facts do not show, (c) a bare da_provare, then the
open count -- never «most da_provare». C2 a linea's `condizione`: a G-id attached to its eid, verified by id and on the
line's article, printed beside the element. C3 a penal defence (`penal_defence`: the frame's rito penal, side difesa)
needs a «riqualificare» linea on another article of the set when there is one (`alternative_articles`): missing, the
re-ask asks for it, then the code prints «nessuna qualificazione alternativa tra le norme applicabili (art. …)» in §6
(`strategy_riqualificazione_missing`); a penal pretesa adds the standing quesito «qualificazione alternativa del fatto»
(`retrieval.STANDING_PENAL_QUESITO`) to the questions searched, through the family gate (`retrieval_quesiti_standing`).
C4 an obiezione's risposta declares the eids / G-ids it rests on (`risposta_cita_elementi`); an element whose esito does
not fit the side (defence: contestato / da_provare / escluso; claimant: provato) is a mismatch -- re-asked once, then
printed «senza risposta nei fatti narrati» (`strategy_objection_mismatch`). C5 a passo must name a concrete proof to
acquire (data `passi_words`) and not restate its element (`restatement_overlap`): else re-asked once, then the code's
label (`strategy_passi_not_concrete`). C6 `lawyer_method.rites_block`: §8 prints the three riti (abbreviato,
patteggiamento, messa alla prova) each with its status word (ammesso / possibile / da verificare / escluso) and the
reason, the rendered rito lines removed (`rite_lines_out`); `state["rites_block"]` for the replay. C7 the frame's
unverified fields are logged (`log["frame_unverified"]`) and no longer printed in §10; machine timestamps are removed
from the body (`timestamps_out`, `timestamps_dropped`); a point reference prints the point's article (`collect_points`
with `article_refs`, `point_article`) and never «v. punto N»; the accusing authority's charge sentence is a
«contestazione dell'accusa» (`analysis_step.is_authority`, `party_label`: §2's label, the table's esito, §10's question,
§7's objection, the strategy's contro-difesa). C8 `lawyer_method.rulings_view` with the verified view: a ruling without
a passage is not listed (`rulings_no_passage_dropped`); a ruling on an article other than the lines' (40, 41, 43 c.p.,
a remedy) only when its passage cites the line's article number or an element's word (`ruling_on_line`,
`rulings_general_part_dropped`). C9 `verified_material.rule_without_id`: a sentence attributing a rule (a qa_statements
verb) to article X with no A-id of X is dropped (`prose_rule_without_id_dropped`). The strategy prompt changed (hash
logged); the analysis, fit, frame and recall prompts are unchanged.

## 5r. v79.18.2 (24 Nov 2026): the side's re-ask; the frame-named article's set entry; the retrieval's sub-stages; the matter gate demotes; the «alternativa» tier

No new flag. Item 1 the risk thresholds are data by side (`data/strategy_step.json` «risk»; the rules of 5s replaced
79.18.2's share thresholds). Item 2 `strategy_step.side_of`: the strategy's side is the
analysis step's from the verified frame; with the frame's ruolo unverified it is the one the pretesa's tipo implies
(data «side_by_tipo»: accusa / opposizione -> difesa, domanda / ricorso / impugnazione -> attore), logged as
`side_source`; a reply whose every linea is rejected for the side's rules («contestare» on a non-open element,
«fondare» on a non-provato one: `_side_mismatch`) is re-asked once with the side stated explicitly (`side_statement`,
data «reask_side» / «side_rules», the re-ask's `{lato}` line; `strategy_side_reasked`). Item 3
`interactive_flow.binding_pretesa`: the frame-named article enters the set with the «pretesa» reason only on a verified
«si» / «dipende» (never on «≠ no»; an unverified verdict leaves it to the unverified tier, no verdict leaves it out).
Item 4 the fusion's sub-steps are marked on round 1's meter (`defensive_generation._select_articles_fusion`:
select:current, select:searches -- the baseline / distill / frame searches, their embedding and graph calls --,
select:vote, select:fetch, select:crossref, select:spotted); `state["timings"]["round1_counts"]` keeps each sub-stage's
llm / embed / graph counts beside `round1_stages`; the replay row (`replay_cases.retrieval_stages`) names the slowest
sub-stages with their calls («retrieval stages (s)»). Item 5 `topic_groups.matter_gate` demotes, never drops, once the
case spans more than one area: the areas are the case's matters plus those the frame's pretesa article touches
(`analysis["frame_charge"]`, passed as `touched`) or a quesito's rows touch (`_quesito`; a row only a quesito brought
opens a further area instead of a case matter); in each further area the top `MATTER_AREA_TOP_K` (3) hits by retrieval
rank stay tagged `_demoted` and rank after every other row (`interactive_flow._rank_of`: the grade cap and the ranked
list), the rest go as before (`state["matter_gate"]`: areas, demoted; counters `topic_filter_areas`,
`topic_filter_demoted`). Item 6 a verified «no» never removes an article of the same Titolo / Capo of the code as the
charge or the pretesa's article (`topic_groups.same_code_unit`: `CODE_UNITS` and the contracts' `capo_of`;
`applicability.effects(..., alternative=)`, `kept_alternative`, the judgment marked `alternative`, `fit_no_kept_alternative`):
the binding lists it as `alternative` with the tier «alternativa» (`TIER_ALTERNATIVA`, `binding_set_alternative`), never in
the set's `principale`; the strategy input carries it (`alternative: True`, labels.alternativa) and admits a
«riqualificare» linea on it only (`_verify_line`), counted among `alternative_articles` for C3, and never the principale
whatever its forza (`finish`, `strategy_alternativa_demoted`). The strategy prompt changed (hash logged); the analysis,
fit, frame and recall prompts are unchanged.

## 5s. v79.18.3 (24 Nov 2026): item 1 of 79.18.2 -- the risk thresholds

No new flag. `strategy_step.risk_of` (data `risk`, by side; `risk_word` reads what `finish` stores as `analysis["risk"]`:
word, capped, why). Claimant (attore, neutro): favorevole = the principale's element provato AND (`provati_min` (2) or more
provato elements on its article OR a G-cite on the line: `cita_pronunce` or `condizione`); sfavorevole = the principale's
element da_provare / escluso; else incerto (provato without that support, or contestato). Defence: favorevole = the
principale's element contestato by the client's own facts (`forza.fatti_propri`: a cited fact accertato or the client's)
AND a G-cite; sfavorevole = the charge's elements all provato and none contestato; else incerto (contestato or da_provare
with a G-cite but no own fact, an own fact without a G-cite, a bare da_provare, an escluso element). Any side: a set built
only from unverified tiers (`analysis["set_non_verificato"]`: every non-alternative article of the input at the
non_verificato tier -- a recall-empty set) or a one-element article (`cap.elementi_min`) caps a favorevole at incerto
(`strategy_risk_capped`; the replay's strategy row carries `risk_why` and `risk_capped`). The strategy prompt is unchanged.

## 6. Which copy sets what

This section covers which file sets the flags in each copy. Only the code defaults are documented here; check the
files on the server for the values actually set.

| Copy | Environment source |
|---|---|
| Live, `/opt/chatbot` | `/opt/chatbot/.env`, loaded at import (see "How flags are read"). |
| Test copy, `/root/v38`, through `scripts/serve_test.sh` | Exports every `KEY=VALUE` line of `.env.v45` (or `ENV_FILE`), then also exports `INTERACTIVE_FLOW=true`, `TOPIC_STEP=true`, `TOPIC_TABLES=true`, `INTERACTIVE_CASE_FIRST=true`. Because these are exported before Python starts, they win over any `.env` that python-dotenv also finds. No LLM setting is overridden: the model is the one in `.env.v45`. The script refuses to run from `/opt/chatbot*` and refuses port 8010. |
| `scripts/promote.sh` (live ← test) | Copies the boolean flags of `/root/v38/.env.v45` into `/opt/chatbot/.env`. That means lines `KEY=true\|false\|yes\|no\|on\|off` whose key does not start with `LLM_`, `EMBEDDING_`, `NEO4J_` or `STRIPE_`. It never copies LLM settings. Each change is printed (`scripts/promote.sh` step 3). |
| `scripts/nightly_sim.py` | Loads `--env` (default `.env`). The docstring requires the server's environment with `LAWYER_METHOD=true`, `CASE_FIRST_SINGLE_SHOT=true` and `LLM_THINKING=per_call` (except in `--answers` mode, which needs no model and no database). |
| `scripts/diagnose_long.py` | Loads `--env`, then applies `--set NAME=VALUE`. `--interactive-sim` defaults `INTERACTIVE_FLOW=true` and `INTERACTIVE_CASE_FIRST=true`. `--interactive-sim --single-shot` defaults `CASE_FIRST_SINGLE_SHOT=true` and `INTERACTIVE_CASE_FIRST=true`. A `--set` of the same name wins. With `VOTE_POOL_ENABLED` and no readable `VOTE_POOL_IDF_FILE`, it computes the counts once and sets `VOTE_POOL_IDF_FILE` for its child processes. |
| Tests (`tests/conftest.py`) | Every name in `FLAGS` (the 90 from `_flags()`) is unset before each test. A test that needs a flag sets it with `monkeypatch.setenv`. |
