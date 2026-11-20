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
