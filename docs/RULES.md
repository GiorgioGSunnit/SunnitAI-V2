# Pipeline rules catalogue (v59 – v79)

This is a catalogue of the deterministic rules the case-first pipeline applies, from v59 to v76b. Its readers are the
engineers who maintain the pipeline. Check it before you add a rule, so that you don't add one that already exists or
one that contradicts an existing rule.

## What a "rule" is here

A rule is a deterministic behaviour of the pipeline: a gate, a cap, an ordering, a status, a rewrite or a removal. The
code applies it in the same way to every case. Rules are tagged in comments and docstrings with the version that
introduced them (`v59`, `v60`, `v60b` … `v60t`, `v61` … `v76`, `v76b`). Sub-items are written `v60h (5)` or `v60f (P9)`:
the number or letter is the item of that version's change list. When a later version changes a rule, its tag stands
next to the older one in the same comment, for example "v60h … (v60t: gone …)". This catalogue lists each rule under
the version that introduced it and names the later amendments in the same bullet. Many entries in
`src/rag/interaction_copy.py` and `src/rag/writer_rules.py` are tagged "my draft". They are copy, which means fixed
wording for the parere or for the writer prompt, and they belong to the rules that use them.

Each bullet reads: what the rule does — version tag — `path:function`. A `function` can be a module constant or a data
file when that is where the rule lives. Several functions defined inside `src/rag/interactive_flow.py:_write_answer`
or `_reviewed` are cited by their enclosing function.

## Acceptance policy (from v77)

The 31 eval cases are **not** a measure for rules. They are listed in `tests/eval_narrative_cases.json`, and the ids
are `TUNING_IDS` and `HELDOUT_IDS` in `scripts/diagnose_long.py`. They are unit-test fixtures only:
`scripts/diagnose_long.py` is a diagnosis and runs no batch by default. A new rule is accepted on two things:

- **Structure and regressions:** the five-cell nightly, `scripts/nightly_sim.py`. It runs one case per posture cell:
  civile/attore/pre_filing, civile/convenuto/primo_grado, penale/imputato/primo_grado, penale/persona_offesa/pre_filing
  and lavoro/attore/pre_filing. For each cell it checks the intake, the parere's structure
  (`lawyer_method.structure_checks`) and the version footer. A check that passed yesterday and fails today is a
  regression.
- **Substance:** the lawyer's scores on the held-out set. The set is defined in `src/chatbot/heldout.py`. The pack is
  exported with `scripts/heldout_export.py` and scored with `scripts/heldout_score.py`.

v77 itself adds no pipeline rule. It covers delivery and measurement only: the `VERSION` file and `src/version.py`,
the «SunnitAI <version>» footer, `GET /api/version`, saving to the held-out set, and the acceptance change above.

No `v60a` tag appears anywhere in `src/` or `data/`.

---

## 1. Intake & routing

- Intake reads the request once, before any route decision. Deterministic rules run first, and a single model call fills only the fields the rules leave empty — v61 — `src/rag/intake.py:detect`
- The intake is always logged, whatever `VERBOSE_LOGGING` says — v61 — `src/rag/verbose_logger.py:ALWAYS_EVENTS`
- Routing by deliverable: parere goes to the case-first single shot; atto goes to document generation; cenni, quesito, calcolo and altro go to the plain RAG answer — v61 — `src/rag/intake.py:route_of`
- Case-law research (ricerca_giurisprudenza), and a parere on an out-of-corpus matter (tributario or fallimentare, with the rito read by rule), get the out-of-scope copy — v61 — `src/rag/intake.py:route_of`, `src/chatbot/api.py:_out_of_scope_reply`
- Short-topic rule: a message of 8 words or fewer that names a legal topic, gives no facts of the user's own and asks for no strategy is a question, whatever the classifier says. The rule moved here from the API — v61 — `src/rag/intake.py:topic_question`, `src/chatbot/api.py:_is_topic_question`
- An atto requested on a narrated case is drafted with the case-first first read as its context (`case_record`) — v61 — `src/chatbot/api.py:_run_generation_sync`, `src/rag/document_generation.py:generate_document`
- The intake is kept in the state, and every later stage reads it. Its default role and rito are filled from the analysis — v61 — `src/rag/interactive_flow.py:take_intake`, `src/rag/intake.py:resolve`
- `state["intake_source"]` is «request» when the intake was read before the flow and «flow» when a caller skipped it. The API and the sims share one function — v73 — `src/rag/intake.py:request_intake`, `src/rag/interactive_flow.py:request_case_first`, `entry_args`
- The questions the message lists after a lead-in («si chiede di valutare: 1. … 2. …») each become a «quesito» block, in the message's order — v65 — `src/rag/intake.py:quesiti`
- Without a lead-in, two or more question sentences (ending in «?») are each a quesito, in order — v69 — `src/rag/intake.py:question_sentences`
- A narrative that ends on one question of law is a quesito. Since v68 this holds whether or not the question names an article — v65, v68 — `src/rag/intake.py:point_of_law_question`, `deliverable_rule`
- A single question that asks for a course of action («come impugnare …») still asks for the case's analysis — v68 — `src/rag/intake.py:question_sentences`
- When the text has no date to count from, the quesito gets the term and its article but no number. A quesito or a cenni is never answered by a calculator counting from today — v69 — `src/rag/intake.py:deliverable_rule`, `decorrenza_date`, `src/rag/calculation.py:calculation_gate`
- The articles the message names are kept with the verb around each one and its use: «weapon» when the client invokes the article, «threat» when it is used against the client. Since v68 the verb's own clause decides — v65, v68 — `src/rag/intake.py:named_articles`
- The rito is «lavoro» only when the text describes an employment relationship. The «lavori» of an appalto or an «impresa edile» do not count — v69 — `src/rag/intake.py:rito_rule`, `employment`, `take`
- Role and stage are read from current-status sentences first («attualmente persona offesa»). A future or hypothetical mention never counts («potrà costituirsi parte civile») — v65 — `src/rag/intake.py:_current_role`, `stage_rule`, `hypothetical`, `client_rule`
- A persona offesa who has joined the trial is the parte civile — v62 — `src/rag/intake.py:client_rule`
- A charge sentence («il PM contesta …», «imputato di … ai sensi dell'art. N», «chiede la condanna …») becomes a «contestazione» entry together with the articles it names — v71 — `src/rag/intake.py:prior_decisions`
- An abbreviation never ends a sentence («il Sig. Caio», «Avv. Bianchi», «art. 640»). Every splitter of the case's text rejoins the pieces — v75 — `src/rag/sentence_split.py:rejoin`, `src/rag/intake.py:sentences`
- The intake's deliverable (cenni, quesito) travels in the agent state, and the plain answer's grounding reads it — v68 — `src/rag/main.py:AgentState`, `src/chatbot/session.py:chat`
- A plain cenni or quesito whose retrieved context holds no ruling gets the no-rulings line. Every article it cites must be among the retrieved sections — v68 — `src/rag/plain_grounding.py:NO_RULINGS_NOTE`, `src/rag/nodes/synthesis.py:_citation_source_blocks`
- A plain cenni stays inside the area its subject names (code and unit, Titolo first). Its articles are read from the graph when retrieval missed them, and its notes come from the same code — v69 — `src/rag/plain_grounding.py:cenni_area`, `cenni_restrict`, `src/rag/nodes/synthesis.py:synthesize_answer`
- A cenni never closes with a question — v69 — `src/rag/plain_grounding.py:no_closing_question`
- A plain quesito gets the store's terms for its procedural articles — v69 — `src/rag/plain_grounding.py:store_lines`
- Single shot: one message in, one complete parere out. Round 1 runs inside it — v60k — `src/rag/interactive_flow.py:single_shot`, `case_first_enabled`
- A message after a single-shot parere (answers, a new fact, «approfondisci X», «e se…») produces the parere again, with the changed parts marked «(aggiornato)» — v60k — `src/rag/interactive_flow.py:_single_update`, `mark_updated`
- What a reply can do inside the interaction is read without the model. An unclear reply gets one clarifying question, never a fresh round 1 — v60 (H) — `src/rag/interactive_flow.py:follow_up`, `round1`

## 2. Case reading & facts

- The first read of the case runs beside retrieval, as one bounded model call — v60 (A) — `src/rag/interactive_flow.py:analyse_case`, `src/rag/case_read.py:read`
- The institutions found by the first read are searched before grading: first their rubricated name, then their own wording, then the content nouns of the hypothesis — v60d, v60e — `src/rag/interactive_flow.py:first_read_search`, `claim_nouns`
- The first read's alternative qualifications are searched too, up to `ALTERNATIVES_MAX` = 3 — v60g (7) — `src/rag/interactive_flow.py:read_institutions`
- The first read has three search channels («rubrica», «testo», «indice»). The channel that brought each anchor is logged — v60h (4) — `src/rag/fact_resolver.py:resolve`, `src/rag/interactive_flow.py:anchor_channel`
- The rubrica is tried first, as an exact match, then a prefix, then a root. All identical rubriche are kept, up to `EXACT_KEPT` = 8 — v60n, v60o — `src/rag/fact_resolver.py:resolve`
- A rubrica match needs `RUBRICA_MIN_WORDS` content words of the institution's name. A generic legal word never matches on its own — v65 — `src/rag/fact_resolver.py:rubrica_admits`, `index_channel`
- A procedural ask («sequestro», «querela», «costituirsi parte civile») is searched through the concepts table — v75 — `src/rag/interactive_flow.py:first_read_search`, `src/rag/cenni.py:procedural_refs`
- While the client is unknown, the read gives every named party's lines. A top-up call asks again for the parties left without lines, but only when `TOPUP_MIN_SECONDS` of budget remain — v60g (4) — `src/rag/case_read.py:missing_parties`, `topup_read`
- Once the client is settled, its side drives the topics: the client's own lines are attached to them — v60e, v60g (4) — `src/rag/interactive_flow.py:reside`, `src/rag/case_read.py:side`
- The client can be stated in the first person («assisto X», «difendo l'imputato»), by role, or as the lawyer of a role («sono l'avvocato dell'inquilino») — v60d, v60i (8) — `src/rag/interactive_flow.py:apply_stated`, `role_party`, `reply_stated`
- A deceased party is never offered as the client and never named as a party to sue — v60d — `src/rag/interactive_flow.py:deceased`, `client_question`, `src/rag/render_computed.py:blocks`
- The writer receives the case's relationship sentences and who has died — v60d — `src/rag/interactive_flow.py:relationships`, `relationships_block`
- Relative dates («2 giorni fa», «ieri») are resolved against today. Vague periods («circa tre mesi fa») become ranges. Accents and hour counts are handled too — v60d, v60g (6), v60h (7) — `src/rag/fact_signals.py:parse_date`, `relative_period`, `relative_text`, `src/rag/interactive_flow.py:relative_periods`
- The user's own message (the request, a question, an instruction) is never a case sentence for a tool, term or fact trigger — v60i (4) — `src/rag/fact_signals.py:case_only`, `user_message`
- A listed ask and its lead-in («In particolare:») are the user's sentences, never facts — v75 — `src/rag/fact_signals.py:case_only`, `src/rag/issue_records.py:_asked`, `src/rag/interactive_flow.py:case_sentences`
- Each computed term needs its triggering fact stated in a sentence of the case or of a precisazione — v60h (3) — `src/rag/interactive_flow.py:trigger_terms`, `src/rag/fact_signals.py:signals_from`
- Possession also includes using a way or a thing over time («usa», «transita», «da anni»). The usucapione's own fact is the possession ad usucapionem — v60h, v60i (5) — `src/rag/fact_signals.py:_JUDGMENT_FACT`
- The case is split into numbered sentences, each with a certainty: accertato, allegato, ipotizzato or assunto. Precisazioni are appended — v60f — `src/rag/issue_records.py:sentences`, `certainty`
- An attributed hypothesis («se fosse») is ipotizzato. Each attributed sentence gets its speaker, and the counterparty's version is never support for the client — v60t — `src/rag/issue_records.py:certainty`, `settle`, `src/rag/interactive_flow.py:case_sentences`, `src/rag/posture.py:mark_sides`
- «provato» needs a supporting sentence that is the case's own words, verbatim. v60h's condition that the sentence must name the element's nouns is withdrawn — v60t (1) — `src/rag/issue_records.py:verbatim`, `grounded`, `settle`
- A status needs a sentence: when a fact derived without a case sentence holding it, the status stays «da provare» — v76 — `src/rag/issue_records.py:settle`, `_trigger`
- The method's facts come from the narrative only. The ask, the candidate's instructions and an article's text are never facts — v71 — `src/rag/lawyer_method.py:narrative`, `facts_split`
- Roles are not names. A party named only by role is written in lower case, with the articled preposition («dell'inquilino», never «di il proprietario») — v60h (9), v60i (7) — `src/rag/party_names.py:lower_roles`, `contract_roles`
- No question is asked that the case already answers. Generic templates rank after the specific ones — v60k — `src/rag/interactive_flow.py:questions`, `asked_in_case`
- A point that asks for a legal characterisation is never a question, because settling it is the parere's job — v60l — `src/rag/interactive_flow.py:qualification`, `with_read_questions`
- A point about a party's state of mind is never a question. The parere argues it from the facts — v60q — `src/rag/interactive_flow.py:state_of_mind`, `with_read_questions`
- What the first read finds the case does not say is asked after the rules' and the tables' questions. It keeps `READ_QUESTIONS_MIN` places, is deduplicated, and every drop is logged with its reason — v60c (5), v60d — `src/rag/interactive_flow.py:with_read_questions`
- A yes/no answer is restated as the fact it settles, never as the question's text — v60h (8) — `src/rag/interactive_flow.py:_apply_correction`, `apply_reply`
- After a reply that answers questions, the restatement is two prose sentences: the client, then what was answered — v60e — `src/rag/interactive_flow.py:restatement`
- Wording of the date question: a case where no topic is criminal is never asked about querela — v59 — `src/rag/interactive_flow.py:question_wording`
- A term line (usucapione, negatoria, rivendicazione) is carried only by a topic of its own matter — v59 — `src/rag/interactive_flow.py:carried_terms`

## 3. Posture, role & stage

- The posture is built once per answer. Rito and role come from the intake when its rules or the model read them, otherwise from the existing logic. The stage is the intake's — v62 — `src/rag/posture.py:of`
- The posture cell (rito, role, stage) gives the main line's act and goal from `data/posture_acts.json`. A cell with no entry keeps today's plan — v62 — `src/rag/posture.py:entry_for`, `apply_plan`
- The entry's «subordinate» lines are always added to the plan: the imputato's riti alternativi and sentencing — v70 — `src/rag/posture.py:apply_plan`
- Each record gets a side. It is a «threat» when the client is imputato or convenuto and the article is the charge or the claim (named in the charge sentences or relied on by the other side). Otherwise it is a «weapon» — v62 — `src/rag/posture.py:mark_sides`
- The client's own alternative qualification is a weapon, never the charge. `rec["charged"]` marks the charge whoever the client is — v63 — `src/rag/posture.py:mark_sides`, `alternative_rows`
- An article the user names takes the side its verb gives, whatever the client's role — v65 — `src/rag/posture.py:mark_sides`
- Charge order: the intake's charge first. The in-flow fallback reads the charge sentences as the intake does — v71, v73 — `src/rag/posture.py:intake_charges`, `charge_labels`
- When a civil claim names no article, the claim is the first read's principal hypothesis. Since v76, for the convenuto it is the hypothesis that matches the pleaded claim (petitum and causa petendi) — v75, v76 — `src/rag/posture.py:claim_from_read`, `pleaded_claim`
- The counterparty's label is the opposite of the client's role (attore ↔ convenuto, imputato ↔ persona offesa / parte civile) — v65 — `src/rag/posture.py:counterparty_label`, `src/rag/interactive_flow.py:intake_labels`
- The label of the party the action is directed at comes from the frame's area (penale: querelato or indagato; civile: convenuto) — v60r (5) — `src/rag/interactive_flow.py:frame_party_label`
- The single-shot parere's first line names the client: «Il parere è reso nell'interesse di X; se assisti un'altra parte…» when the client is assumed, and the frame's or role's label when the client is known — v60k, v60s (7) — `src/rag/interactive_flow.py:settle_client`, `known_client_line`
- A convenuto that the extraction writes as «controparte» is never the assumed client — v60s (7) — `src/rag/interactive_flow.py:_assumed_party`
- A client who is assumed is said first in the answer. After a reply that answered a question, «Non mi hai indicato…» is never written — v60 (D), v60d — `src/rag/interactive_flow.py:_write_answer`, `settle_client`
- The intake's rito family, never the frame's area, is the key of the ratione temporis, stragiudiziale, mediazione and riti lines — v75 — `src/rag/posture.py:intake_family`, `src/rag/lawyer_method.py:_family_key`
- The threats are read in penal words («dall'accusa») for a penal posture. The client defends when the posture's role is imputato or convenuto — v62, v65 — `src/rag/interactive_flow.py:penal_posture`, `defending_posture`
- §4 by stage: after the decision there is never a procedibilità line. At the trial and in the indagini, the stage's closed list follows today's blocks — v62, v65 — `src/rag/render_computed.py:criminal_procedure`, `blocks`, `OPEN_STAGES`
- The parties and damages blocks are only for civile / attore / pre_filing — v62 — `src/rag/render_computed.py:parties_allowed`
- The pre-filing list (mediazione, ATP, 700) is restored for the civile and famiglia riti only. A block whose tool the section already shows is left out — v70 (10) — `src/rag/render_computed.py:blocks`, `stage_blocks`, `PRE_FILING_RITI`
- In the stage blocks, a charge is not a prior decision (it gives no {date}) — v71 — `src/rag/render_computed.py:stage_blocks`
- Mediazione and the pre-filing list follow the rito the intake read — v75 — `src/rag/render_computed.py:mediation_rito`
- «Procedibilità e competenza», «Profilo penale» and the querela term appear only when the intake read a penal rito — v75b — `src/rag/render_computed.py:penal_rito`, `rito_terms`
- The imputato's riti alternativi apply to a penal intake rito before the sentence, each gated on the charged offence's pena — v70, v75 — `src/rag/lawyer_method.py:riti_lines`
- For an imputato charged with a d'ufficio offence, the stragiudiziale option has no remissione and no 162-ter. When the intake read no rito, there is no line from another rito's table — v71, v75 — `src/rag/lawyer_method.py:stragiudiziale_line`

## 4. Retrieval, grading & demotion

- Round 1 runs a graded check on the embedding prefilter's top N over `ROUND1_GRADE_ROUNDS` expansion rounds. The answer step grades the rest of the chosen topics — v60 (G) — `src/rag/interactive_flow.py:_retrieve`, `_write_answer`
- Every graph query, chat call and embedding call is counted per stage. Since v60h (11) each chat call is recorded with its seconds and tokens — v60, v60h (11) — `src/rag/stage_meter.py:Meter`, `call`, `src/rag/ai_chat.py:_stage_call`
- The graph reads and fusion branches run in parallel, and the case-only branches start beside the analysis — v60b — `src/rag/defensive_generation.py:prefetch_case_branches`, `searches`, `src/rag/topic_step.py:build`
- The first read's own finds are graded outside the prefilter's cap. The prefilter counts as «cutting» when more than `CUTTING_SHARE` = 0.8 of what it kept graded 2 or higher — v60d — `src/rag/defensive_generation.py:_filter_applicable_articles`, `src/rag/interactive_flow.py:grading_breadth`
- Every kept candidate is graded (graded == kept, checked as an invariant). An exempt candidate (user-named, the rubrica's own) is graded but never removed — v60o — `src/rag/interactive_flow.py:grading_invariant`, `src/rag/defensive_generation.py:grade_these`
- Grading runs in one pool, priority labels first. Calls, wall time and the longest call are logged — v60p — `src/rag/defensive_generation.py:_grade_kept`, `vote`
- The anchors of the fired rules and tables are candidates, graded like any other — v60n — `src/rag/interactive_flow.py:rule_anchor_rows`
- Area-code priority: an article of a procedural code anchors only a procedural institution — v60n — `src/rag/topic_groups.py:may_anchor`, `src/rag/interactive_flow.py:procedural_institutions`
- The c.p.a. is used only in an administrative frame — v60i (3) — `src/rag/interactive_flow.py:area_codes`
- Code-area gate: an article of a code outside the frame's area is used only when the first read, a rule or the user names it. The others become «altri profili», one per Capo, never graded and never a topic — v60q — `src/rag/interactive_flow.py:_retrieve`, `src/rag/topic_groups.py:outside_profiles`
- Capo-scope gate for c.c. Libro IV Titolo III: a Capo is admitted only for the case's own contract type — v60q — `src/rag/topic_groups.py:gate_topics`
- A general-part article (c.p. Libro I) is never a grading candidate, except when the user names it — v64 — `src/rag/cenni.py:general_part_gate`, `general_part`
- An article of a special law needs the law's domain signal in the first read, except when the user names it. Since v74 this holds on every route. Since v75 a regulation is also recognised by its number — v65, v74, v75 — `src/rag/special_laws.py:gate`, `regulation`, `data/special_laws.json`
- Articles 409-447 c.p.c. are admitted only when the case's rito is «lavoro» — v69 — `src/rag/special_laws.py:labour_gate`, `labour_article`
- Each graded article's distinctive element is linked before the topics are built — v60g (1) — `src/rag/issue_records.py:distinctive`
- Round 1 demotes an article to grade 1 only when the extracted facts exclude its distinctive element or a narrated sentence contradicts it. Silence never demotes. Each demotion is logged with its reason — v60m — `src/rag/issue_records.py:demote_contradicted`
- The distinctive-element link runs to completion in smaller calls (`DISTINCTIVE_SECONDS` = 45) and logs what it left unassessed — v60h (5) — `src/rag/issue_records.py:DISTINCTIVE_SECONDS`
- An abrogated article (its text is the abrogation) is never a topic's article, planned or written — v60c — `src/rag/interactive_flow.py:prune_abrogated`, `src/rag/article_text.py:row_abrogated`
- A row whose rubrica belongs to another article's text, confirmed by a neighbouring row, is never a topic's article — v60e — `src/rag/article_text.py:row_misaligned`, `shifted`, `src/rag/topic_groups.py:_best_rows`
- In the rubrica check, one stem shared with the rubrica or the first comma clears a citation — v60c — `src/rag/defensive_generation.py:one_stem_clears`
- A fired rule fires only on a case whose areas include the rule's area. A rule family is asked only within its area — v60d — `src/rag/interactive_flow.py:rules_in_area`, `family_fits`
- With the case-first flow, the criminal content of §4 comes only from the procedural store, for the topics' offences — v60d — `src/rag/render_computed.py:store_only`

## 5. Topics & coverage

- Topics are built from the graded rows. The topics of rules and tables attach to them — v60 (B) — `src/rag/interactive_flow.py:_retrieve`, `src/rag/topic_groups.py:attach_rules`
- A hypothesis is anchored on the graded rows' rubriche by stems first, then on Sezione / Capo titles by similarity, and by words when the embedding fails — v60c — `src/rag/topic_groups.py:_stem_anchor`, `anchor_by_similarity`, `src/rag/interactive_flow.py:attach_side`
- A hypothesis that no row anchors is listed as «Ipotesi da verificare», never asserted. What was searched for it is logged — v60 (B), v60e — `src/rag/interactive_flow.py:round1`, `attach_side`
- The anchor of a rule or table that was not graded is listed, but never planned or written — v60c — `src/rag/topic_groups.py:attach_rules`
- Topic order: code weight, then number of graded rows, then the main-line hypothesis, then the user's angle, then best grade. Since v60d a topic from the case area's code comes first — v60c, v60d — `src/rag/topic_groups.py:order`, `area_first`, `mark_area`
- Every identified offence is in a topic — v60c — `src/rag/interactive_flow.py:analyse_case`
- «procedi» with no topic chosen takes the immediate answer — v60c — `src/rag/interactive_flow.py:default_choice`
- A graded topic is reduced to its articles graded 2 or higher, and §4's terms follow those articles — v60g (5) — `src/rag/topic_step.py:graded_topics`
- A term is carried by a topic of its own matter. A topic that only holds a term is dropped while other topics are listed — v59 — `src/rag/topic_step.py:carries_term`, `term_topics_out`, `_term_only`
- A table names a topic when half or more of its questions are already the topic's — v59 — `src/rag/topic_step.py:named_by_table`
- The Capo neighbours of a matched article stay matched, never left bare as «non trattate» — v59 — `src/rag/topic_step.py:_merge`
- The single-shot answer covers at most `SINGLE_SHOT_TOPICS` = 5 topics and at least `SINGLE_SHOT_TOPICS_MIN` = 3 when that many exist — v60n, v60q — `src/rag/interactive_flow.py:covered_topics`
- Each topic not covered gets one line under «Altri profili», up to `OTHER_PROFILES_MAX` = 5 lines, best by score. This applies only when the method is off — v60r (2), v60n — `src/rag/interactive_flow.py:single_profiles`, `_write_answer`
- An «Altri profili» article must share its code and Libro with a covered topic. Since v76 c.c. Libro IV is split by its Titoli — v68, v76 — `src/rag/interactive_flow.py:single_profiles`, `profile_book`, `PROFILE_BOOKS`
- Articles are capped by weight, with each topic's matched articles first — v60f (A4) — `src/rag/interactive_flow.py:capped_refs`
- The single-shot article cap is `SINGLE_SHOT_ARTICLES`. Whatever the cap leaves out is named once, with its first topic, in «Altre norme del tema non trattate» — v60k, v60q — `src/rag/interactive_flow.py:_write_answer`
- The answer's main line is the topic that holds the first read's main-line hypothesis, otherwise the first topic — v60t (2) — `src/rag/interactive_flow.py:main_line_topic`
- An issue titled after an offence unit that has no offence article left after the qualification check is dropped together with all its articles — v60s (1) — `src/rag/interactive_flow.py:offence_issue`, `surviving_issues`
- The plan argues the chosen topics as fixed issues. It never selects, adds or regroups issues — v60f (c) — `src/rag/interactive_flow.py:plan_inputs`, `src/rag/plan_then_write.py:make_plan`
- A party the case does not identify is «da individuare» in the plan — v60g (P6) — `src/rag/plan_then_write.py:_plan_human`

## 6. Records, elements & the lawyer's method

### 6.1 Per-article records

- The per-article record is the intermediate representation between round 1 and the answer. It is built without a model call. The link uses one bounded call per chosen topic, run in parallel — v60f — `src/rag/issue_records.py:build`, `link`, `src/rag/interactive_flow.py:build_records`, `link_records`
- Settle: the facts decide first. Otherwise a narrated sentence that contradicts an element makes it «non risulta», and one that supports it makes it «provato». An allegation alone leaves it «da provare». While an answer is being written there is one status source — v60g (d) — `src/rag/issue_records.py:settle`, `recorded_status`, `statuses_from`
- The d'ufficio conditions of a querela offence are record elements of their own (P1…). They are linked and settled like requisites but are never part of the verdict — v60h (1) — `src/rag/issue_records.py:_conditions`, `elements`
- A distinctive element that «non risulta» makes the verdict «non applicabile» — v60h (5) — `src/rag/issue_records.py:verdict`, `seed`
- «non applicabile» comes only from an unmet element of the anchor comma. An unmet element of another comma makes that comma «non pertinente» — v60r (3) — `src/rag/issue_records.py:verdict`, `element_status`, `unmet_in_anchor`
- The anchor comma is co. 1 unless the topic names another one — v60s (2) — `src/rag/issue_records.py:anchor_comma`, `build`
- An article the counterparty relies on whose elements fail gets the verdict «argomento del cliente», never «non applicabile» — v60i (2) — `src/rag/issue_records.py:verdict`, `src/rag/interactive_flow.py:role_verdicts`
- An element with no fact signal is «da provare» and is never narrated as missing — v60r (6) — `src/rag/issue_records.py:no_signal`, `NO_SIGNAL_NOTE`
- Element grading standard: a fact the case narrates as having happened supports its element — v60s (3) — `src/rag/issue_records.py:recorded_qualified`
- Requisites are counted («N su M») only for a fattispecie. A rule on time limits or proof is never a fattispecie (v76), and neither is a rule that modifies, extends or administers other articles (v76b) — v62, v76, v76b — `src/rag/issue_records.py:counted`, `_DEFINITIONAL_RE`
- Only a fattispecie or a cause di giustificazione is decomposed into elements, on every route. Artt. 50-54 c.p. are graded but are never the charge, a threat or a claim — v74 — `src/rag/issue_records.py:decomposed_by_gate`, `justification`, `decomposed`
- With the method on, every record is split into elements as the charge is, and the method reasons only on decomposed fattispecie — v74 — `src/rag/interactive_flow.py:decompose_records`, `src/rag/lawyer_method.py:method_records`
- When the decisive charge or claim cannot be read or split, the method is not applied, and the answer says so — v76 — `src/rag/interactive_flow.py:decompose_records`, `src/rag/lawyer_method.py:NOT_APPLIED_WHY`
- The d'ufficio conditions of decomposed records are graded by the link like the topics' conditions — v76 — `src/rag/interactive_flow.py:decompose_records`
- A threat article's elements are read from the defence («elemento da contestare», «non provato dall'accusa»), and its single status sentence is never «applicabile, N su N» — v62 — `src/rag/issue_records.py:defence_items`, `threat_status`
- A threat's elements are given by name, and elements that are «non pertinente» are left out — v63 — `src/rag/issue_records.py:threat_elements`
- An element counts as contradicted when it is «non risulta» in its anchor comma or contradicted by a sentence — v65 — `src/rag/issue_records.py:contradicted`
- An article is assertable when it is «applicabile», or «da provare» with at least one element «provato» — v60s (4) — `src/rag/issue_records.py:assertable`
- An article whose requisites are not counted, or one brought in by a cenni or another general road, gets an «applicazione» paragraph — v63, v64 — `src/rag/issue_records.py:applied`

### 6.2 Elements from the article's text

- A clause about the penalty or a consequence, or an aggravante, is never a requisite line — v59 — `src/rag/rule_quote.py:requisite_kept`
- A comma that extends other articles' penalties («Nei casi previsti dagli articoli …») has its own line. Since v76b its first sentence decides — v59, v76b — `src/rag/rule_quote.py:refers_penalties`, `offence_elements`, `article_rules`
- A condition set on the penalty («qualora l'offerta non sia accettata») is an element — v59 — `src/rag/rule_quote.py:offence_elements`
- The subjective element named in an offence's clause («cagiona per colpa») is an element — v74 — `src/rag/rule_quote.py:offence_elements`
- A civil fattispecie told in one sentence («Qualunque fatto doloso o colposo …») is split into elements — v75 — `src/rag/rule_quote.py:civil_elements`
- General penal splitter: the clause after «chiunque», wherever the penalty stands, cut at means and purpose markers. Subject heads are matched as whole words — v76 — `src/rag/rule_quote.py:penal_clauses`, `_subjective_split`
- A guarantee sentence («Il venditore è tenuto a garantire …») yields elements — v76 — `src/rag/rule_quote.py:civil_elements`
- A clausola di riserva or a default clause is never an element, and the leading riserva is cut — v76b — `src/rag/rule_quote.py:element_kept`, `without_riserva`
- A qualified subject with its own relative clause is a reato proprio, and its quality is an element — v76b — `src/rag/rule_quote.py:qualified_subject`, `civil_elements`
- Pieces are cut before their means, purpose and cause markers. A civil comma takes the first split that gives two or more elements — v76b — `src/rag/rule_quote.py:split_markers`, `civil_split`, `subject_clause`
- A special law's penal provision is split like an offence — v76b — `src/rag/rule_quote.py:penal_provision`, `article_rules`
- A noun-phrase condition («in caso di …») ends where the main clause begins. «se la» works as a connector — v76, v76b — `src/rag/rule_quote.py:requisites`, `_CONNECTOR_RE`
- A record's single condition counts as an element — v76b — `src/rag/issue_records.py:_requisites`
- A comma added later («1-bis», «3-ter») is its own comma, in its own order — v67 — `src/rag/comma_material.py:comma_labels`, `comma_order`, `src/rag/rule_quote.py:row_commi`

### 6.3 The lawyer's method (`LAWYER_METHOD`)

- With records present, the single-shot parere is written by the lawyer's method in his ten sections. The data is his words, verbatim — v70 — `src/rag/lawyer_method.py`, `src/rag/interactive_flow.py:method_mode`, `data/lawyer_method.json`
- His rules (REGOLE DI ONESTÀ, FASE 7, STILE, the case's modules) go into every writer call while the method is active — v70 — `src/rag/lawyer_method.py:writer_rules_text`, `current_rules`, `src/rag/plan_then_write.py:write_sections`
- `LLM_THINKING=per_call` turns reasoning on only for the thesis selection and the writer, and both durations are logged — v70 (3) — `src/rag/ai_chat.py:thinking_bind`, `per_call`
- Fase 1 splits the facts into certi, ipotizzabili and mancanti. A missing fact must name the element it would change, otherwise it is left out — v70 — `src/rag/lawyer_method.py:facts_split`
- §4 holds a subsumption table per article (elemento / fatto che lo integra / esito) and no «N su M» status line — v70 — `src/rag/lawyer_method.py:subsumption_table`, `src/rag/interactive_flow.py:status_sentence`
- Circumstances become records graded against the sentences, and the aggravante check always runs for the charged offence (`NO_AGGRAVANTE` when none applies) — v70 — `src/rag/lawyer_method.py:circumstance_records`, `aggravante_note`
- The procedural section gets the ratione temporis line and the litisconsorzio lines — v70 — `src/rag/lawyer_method.py:temporal`, `litisconsorzio_lines`
- Thesis coherence is checked through the element ids each thesis declares, never through its words. A thesis that declares no element is unverifiable. A thesis that contests an «integrato» element is barred unless it is declared minoritaria (realismo basso). A minority thesis leads only when there is no sustainable majority line — v71 — `src/rag/lawyer_method.py:coherent_solutions`, `barred`
- With no charge on the record, the outcome is never «esito pieno». When every element is integrato, it is «parziale» if a sustainable alternative exists, otherwise containment — v71 — `src/rag/lawyer_method.py:assessment`, `settle_objective`
- A sentence promising an outcome the record does not give is removed — v71 — `src/rag/lawyer_method.py:coherent_text`
- A thesis log records the exact input of the thesis call — v71 — `src/rag/lawyer_method.py:thesis_log`, `src/rag/interactive_flow.py:single_shot`
- The code enumerates the theses: per-element contests, riqualificazioni with a sound record, scriminanti and attenuanti backed by facts, and the containment line. The model only argues them — v72 — `src/rag/lawyer_method.py:thesis_candidates`, `apply_valuations`
- The thesis call is best of `SAMPLES` = 3, scored by the code: every candidate valued and none added. `METHOD_TOKENS` = 3200 — v72 — `src/rag/lawyer_method.py:score_sample`, `method_plan`
- The summary, the conclusions and the next steps follow from the code's selection, never from the model's plan — v72 — `src/rag/lawyer_method.py:conclusions_from_selection`, `summary_section`, `compose`
- There are always three objections, with `thesis_objections` as the last resort. Objections are accepted in any shape the model gives — v72 — `src/rag/lawyer_method.py:thesis_objections`, `clean_objections`
- The lawyer's «note di inquadramento» (store first, then his file) reach the records, the norms writer and the thesis call — v72 — `src/rag/article_notes.py`, `src/rag/lawyer_method.py:attach_notes`, `src/rag/interactive_flow.py:method_notes`
- Objections are written in their own call after the selection, for the selected line. An answer that argues a barred thesis is dropped — v73 — `src/rag/lawyer_method.py:objections_plan`
- The next steps come only from the selected line's own elements: proof to acquire or document for each element it rests on, and the counterparty's proof to verify for each element it contests — v73 — `src/rag/lawyer_method.py:steps_from_selection`
- The model argues only contests and requalifications (`MODEL_ARGUED`). The code writes the argument of every other candidate from the record — v74 — `src/rag/lawyer_method.py:code_argument`, `apply_valuations`, `score_sample`
- Arguments are bound to their declarations: a sentence of pro, contro or argument that negates an element beyond the thesis's «contesta» is removed — v74 — `src/rag/lawyer_method.py:bind_arguments`, `negation_flags`
- The main threat is a fattispecie only — v74 — `src/rag/lawyer_method.py:main_threat`
- A placeholder main line («Tesi da costruire …») is a failure state and is never printed. The method refuses the answer and the standard body is written — v75 — `src/rag/lawyer_method.py:placeholder_main`, `fallback_thesis`, `end`, `src/rag/interactive_flow.py:_write_answer`
- The method never steps aside silently: the answer says why it was not applied — v76 — `src/rag/lawyer_method.py:NOT_APPLIED`, `not_applied_note`, `src/rag/interactive_flow.py:_write_answer`
- Candidate sets depend on the role. The victim's set covers the claim and the costituzione di parte civile, never the offender's circumstances. Attenuanti are for the offender only. The convenuto gets a riconvenzionale and the eccezioni of the claim's matter. The attore gets the remedies of the matter — v75 — `src/rag/lawyer_method.py:thesis_candidates`, `_victim_candidates`, `matter_of`, `data/remedies.json`
- An eccezione's argument is what it needs, taken from the table — v75 — `src/rag/lawyer_method.py:code_argument`
- The attore's claim with nothing proven yet becomes «Fondare la domanda … provando …». A riconvenzionale or a user-named claim to prove gets the same treatment — v75b, v76 — `src/rag/lawyer_method.py:CANDIDATE_TEXTS`, `CODE_ARGUMENTS`, `thesis_candidates`
- A candidate is a contest or requalification only when its record is assertable and not contradicted. The record decides — v75b — `src/rag/lawyer_method.py:thesis_candidates`
- The paragraph of a charged article is written from its sussunzione rows only — v75b — `src/rag/lawyer_method.py:charged_prose`, `charged_blocks`
- An objection's reply sentence that negates an element the main line does not contest is removed — v75b — `src/rag/lawyer_method.py:bind_replies`, `objections_plan`
- Ranking among eligible theses: realism first, then proven elements rested on, then candidate order. The model's own pick counts only on the old path — v76 — `src/rag/lawyer_method.py:coherent_solutions`
- A civil role's fallback line comes from the remedies table. The containment line is penal only — v76 — `src/rag/lawyer_method.py:fallback_thesis`
- A thesis the model added without a candidate id is never displayed — v76 — `src/rag/lawyer_method.py:compose`
- The stragiudiziale option is the table's line only, and the model's version is dropped — v76 — `src/rag/lawyer_method.py:solutions_section`
- A penal line is never offered under a civil intake rito — v76 — `src/rag/lawyer_method.py:thesis_candidates`

## 7. Rendering & the parere's sections

- The answer has one list of sections, with one numbering and one set of titles, and the writer is told those five sections. The read-only computed block takes its real name — v60f (P9) — `src/rag/interactive_flow.py:sections_note`, `src/rag/render_computed.py:read_only_block`
- While an answer is written, the writer reads the SCHEDE (the records), and every status is the record's — v60g (d) — `src/rag/interactive_flow.py:record_mode`, `_jobs`, `src/rag/issue_records.py:writer_block`, `src/rag/writer_rules.py`
- §2 block due: a §2 block is written only for an article that is «applicabile», «argomento del cliente», or «da provare» with a case sentence supporting it. Every other article becomes one line with its reason — v60i (1) — `src/rag/issue_records.py:block_due`, `line_reason`, `src/rag/interactive_flow.py:record_lines`
- A block is also due for a «da provare» article whose missing fact is asked by an open round-1 question — v60j — `src/rag/issue_records.py:asked`, `block_due`, `src/rag/interactive_flow.py:open_questions`
- The anchor of each covered topic earns its §2 block. In single shot only the main line's anchor does, and a «da provare» article earns a block only with an element «provato» — v60p, v60t (2) — `src/rag/interactive_flow.py:anchor_blocks`, `src/rag/issue_records.py:block_due`, `proven`
- The main line's anchor is always written first, with its honest status — v60t (2) — `src/rag/interactive_flow.py:norms`
- An article the user names always has its block — v65 — `src/rag/interactive_flow.py:norms`, `general_rows`, `user_named`
- A named article that the writer left empty gets a second call for that article alone — v68 — `src/rag/interactive_flow.py:named_retry`
- At most `BODY_BLOCKS_MAX` = 7 blocks are written, ranked main anchor, then «applicabile», then «da provare» with an element. The rest become rows of the citations («non trattata nel testo …») — v69 — `src/rag/interactive_flow.py:block_rank`, `glossed_refs`, `norms`
- §2 has one status sentence per article. A non-fattispecie or an applied article gets none. Under the method §4's table replaces it — v62, v70 — `src/rag/interactive_flow.py:status_sentence`
- Single shot: the record's rows (requisite and comma status lines) never appear in the body. They go to the article citations only — v60r (1) — `src/rag/interactive_flow.py:ledger_out`
- An «applicazione» paragraph that touches neither the client nor `APPLICATION_FACT_STEMS` of the case leaves the body — v64 — `src/rag/interactive_flow.py:grounded_application`, `ledger_out`
- A §2 line that is an article's label alone is dropped — v60s (5) — `src/rag/interactive_flow.py:_bare_article_line`
- A sanction article never gets a block of its own: it is a line under its obligation's block, or it goes in «non trattate» — v60h (6) — `src/rag/interactive_flow.py:is_sanction`, `assemble_norms`, `bucket_articles`
- A gloss that declares its article inapplicable becomes one item of «Norme considerate e non applicabili», never a block. The phrasings come from live answers — v60d, v60e, v60i (6) — `src/rag/interactive_flow.py:declares_inapplicable`, `norms_final`
- An article the record marks «non applicabile» is one item of the closing line with its reason, never a block — v60g, v60h (5), v60i (1) — `src/rag/interactive_flow.py:norms_final`, `folded_norms`
- Each planned article falls into exactly one bucket (scritto, non applicabile, …), recorded in its record — v60f — `src/rag/interactive_flow.py:bucket_articles`, `record_answer`
- A matched article that the writer skips keeps its block with its comma line, never «non trattate» — v60 — `src/rag/interactive_flow.py:norm_blocks`, `assemble_norms`
- A comma line appears only where the comma names a fact — v59 — `src/rag/interactive_flow.py:norm_blocks`
- Final pass on the answer as the user reads it: bare §2 headings go into the closing line, orphan fragments are removed, repeated conditional markers appear once, and scaffold bullets are removed — v59 — `src/rag/interactive_flow.py:answer_final`, `_orphans_out`, `_markers_once`, `_scaffold_bullets_out`
- A fact the case does not state is written «da verificare: il caso non dice …» where it counts — v60k — `src/rag/interactive_flow.py:shown_status`, `conditional_on_unknown`, `open_point_fact`
- There is no «Punti da precisare» block by default (`OPEN_POINTS_MAX` = 0) — v60q — `src/rag/interactive_flow.py:open_points`, `open_points_max`
- A legal characterisation left open reads «da verificare» on §2's status line, never as a question — v60l — `src/rag/issue_records.py:points_from`, `src/rag/interactive_flow.py:unknown_block`
- After «il caso non dice» the text continues in lower case, except for names and acronyms — v60l — `src/rag/interactive_flow.py:lowered_start`
- Consecutive «(da verificare…)» markers are merged into one — v60r (4) — `src/rag/interactive_flow.py:_write_answer`
- «non verificabile» never appears in the answer. «da verificare» takes its place — v65 — `src/rag/interactive_flow.py:banned_words_out`, `src/rag/answer_invariants.py:_STATUS_WORDS`, `src/rag/fact_signals.py:_term_entry`
- A writer's paragraph that starts with «(» or a lowercase word is removed — v64 — `src/rag/interactive_flow.py:fragment_paragraphs`
- §1 and the conclusions assert only assertable articles, plus the main anchor with its own status. They never assert a cenni or general-road article (v63, v64) or a weapon (v68) — v60s (4), v60t — `src/rag/interactive_flow.py:asserted_only`, `ASSERTING_TITLES`
- The charge or claim is contested in §1 and §5, never asserted. Only the record's elements are named — v62, v63 — `src/rag/interactive_flow.py:material`, `threat_elements_rule`, `src/rag/writer_rules.py:THREAT_ASSERT_RULE`
- For a defending client, an application paragraph is also read from the defence — v65 — `src/rag/writer_rules.py:APPLICATION_RULE_DEFENCE`
- An alternative with no fact of the case behind it is never in §1 — v65 — `src/rag/interactive_flow.py:material`, `src/rag/writer_rules.py:THREAT_ELEMENTS_NONE`
- The client's own articles that are not yet assertable are argued together with what remains to prove, never «da valutare» — v68 — `src/rag/interactive_flow.py:material`, `src/rag/writer_rules.py:WEAPON_TO_PROVE_RULE`
- The requested blocks come first, in the order asked, and the standard body follows them — v62, v68 — `src/rag/interactive_flow.py:requested_sections`
- With five or more requested blocks the body is short: «Sintesi e strategia» and «Norme» — v68 — `src/rag/interactive_flow.py:collapsed`, `collapsed_body`
- A quesito's section heading is the question itself, and the quesito is answered only from the record and the store — v65 — `src/rag/interactive_flow.py:quesito_title`, `quesito_material`
- Alternatives section: per alternative with an article, «art. N — rubrica» plus one sentence. Since v64 it is written before §4 and its «SÌ/NO» verdicts decide §4's lines — v63, v64 — `src/rag/interactive_flow.py:alternatives_text`, `write_alternatives`, `parse_alternatives`
- A «SÌ» stands only on a record with no contradicted element. An alternative with no facts behind it is a «NO» — v65 — `src/rag/interactive_flow.py:sound_alternative`, `parse_alternatives`
- Cenni subjects are anchored on their articles only when the match is confident: general concepts table, then the institutions index, then the rubrica. Each anchored article becomes an «applicazione» paragraph — v63 — `src/rag/cenni.py:resolve`, `data/general_concepts.json`, `src/rag/interactive_flow.py:cenni_rows`, `general_applications`
- The general part enters only through its own roads: a cenni, the kind of a charged element, the posture table's sanctions, or a user-named article — v64 — `src/rag/cenni.py:element_refs`, `src/rag/interactive_flow.py:general_rows`, `general_part_out`

### 7.1 The procedural section (§4)

- Identical d'ufficio lines are merged into one. While the date of the fact is unknown, the section says what runs from it — v59 — `src/rag/render_computed.py:procedure_lines`
- §4's terms are those of the listed topics only, and mediazione follows the topics' matter — v59 — `src/rag/render_computed.py:blocks`, `parties_allowed`
- There is one line per offence, built from the article's own procedural facts and never written by the model — v60 (C) — `src/rag/procedural_facts.py:extract`, `src/rag/render_computed.py:_fact_lines`
- The chosen topics' offences are in the criminal procedure whatever the frame's area — v60 — `src/rag/render_computed.py:criminal_procedure`
- The procedibilità line names each d'ufficio condition with its record status, plus the line's verdict — v60h (1) — `src/rag/procedural_facts.py:line`, `ufficio_verdict`, `conditions`
- Negoziazione assistita replaces mediazione where the case's own facts call for it. A tool needs its triggering case sentence — v60e — `src/rag/render_computed.py:negotiation_of`, `_triggered_tools`
- The conditions of procedibilità are closed lists keyed by matter. A matter the negoziazione's list excludes never gets it — v60h (2) — `src/rag/render_computed.py:mediation_of`, `_negotiation_excluded`, `data/mediation_matters.json`
- The case's own matter (contract type and area) decides mediazione, never an article the parere writes — v60q — `src/rag/render_computed.py:case_matter`
- The trascrizione della domanda (art. 2653 c.c.) needs its own fact — v60i (5) — `src/rag/render_computed.py:negotiation_of`
- The labour dispute's procedural facts come from a closed list — v60i (9) — `src/rag/render_computed.py:labour_of`, `data/labour_procedure.json`
- A term or trascrizione line can also be triggered by the parere's own analysis, and each line logs the trigger that produced it — v60p — `src/rag/render_computed.py:own_trigger`, `own_term_lines`, `own_trascrizione`, `src/rag/interactive_flow.py:own_analysis`
- The querela term appears only where it applies. It never appears at the trial (v63) or once a charge is stated (v64). Prescrizione is always shown — v63, v64 — `src/rag/render_computed.py:criminal_procedure`, `procedure_lines`
- When a charge is stated, the table's offences are limited to the charge's and the fitting alternatives' — v63, v64 — `src/rag/render_computed.py:criminal_procedure`, `src/rag/posture.py:procedure_offences`
- A computed term line whose reference has no article number is dropped. A special law's term line needs its domain signal — v65 — `src/rag/render_computed.py:_term_line`, `blocks`, `src/rag/special_laws.py:term_allowed`
- §4's terms come from the store wherever it has them, with their decorrenza. An ambiguous term is quoted, and an unread article reads «testo non letto». The stage list carries no number of its own — v66 — `src/rag/render_computed.py:store_term_line`, `stage_blocks`, `store_articles`, `src/rag/interactive_flow.py:store_terms_for`
- A term or amount that §1 or the conclusions state for an article must match the store. A term the law states ambiguously is never rewritten — v65, v66 — `src/rag/interactive_flow.py:store_numbers_out`, `store_numbers`

### 7.2 The procedural store (`procedural_facts`)

- Competenza: art. 5 c.p.p. (Corte d'assise) decides first, then the pena — v62 — `src/rag/procedural_facts.py:pena`, `court`
- Every term of every comma is kept with its subject (the act it is the term of), including those of the procedural codes — v65 — `src/rag/procedural_facts.py:termini`, `term_text`
- The decorrenze use the law's own words. All readers go through one accessor. Two numbers for one act in one comma mark the term ambiguous — v66 — `src/rag/procedural_facts.py:decorrenze`, `term_lines`, `terms_for`, `marked_ambiguous`
- The act that owns a number is read leftwards from its marker. «E'» reads as «È». A list item ends at its «;» — v67 — `src/rag/procedural_facts.py:owner`, `acts`, `termini`
- A term counted back from an event («almeno settanta giorni prima dell'udienza») is a minimum. Two numbers that run from two different events are two terms — v69 — `src/rag/procedural_facts.py:termini`, `marked_ambiguous`
- The giudice di pace is matched by the exact article (612-bis is not 612) — v76 — `src/rag/procedural_facts.py:competenza`, `PEACE_JUDGE`
- The corpus markers in an article's text («(abrogato)», footnotes) are stripped — v60s (6) — `src/rag/procedural_facts.py:clean_facts`

## 8. Citations & article_citations

- One reading of an answer's citations serves every step, including codes named with spaces — v60e — `src/rag/citations.py`, `src/rag/interactive_flow.py:ref_documents`, `src/rag/readback.py:_cited`
- A document name that names a normative act counts as a citation's code — v60h — `src/rag/citations.py:_SIGLA_RE`
- A shared tolerant parser («art. N», «art N», «art.N», «articolo N») is used for intake, the charge, named articles and the rubrica check — v72 — `src/rag/citations.py:refs_in`, `article_refs`, `src/rag/intake.py:named_articles`, `src/rag/posture.py:sentence_articles`
- Single shot: an article is a reference, never its quoted text. §2 cites it as «art. N codice — rubrica» — v60q — `src/rag/interactive_flow.py:material`, `src/rag/writer_rules.py:NORMS_INSTRUCTION`
- `article_citations` gives the frontend the articles the parere cites. Since v60r it also carries the record's rows — v60q, v60r (1) — `src/rag/interactive_flow.py:article_citations`, `src/chatbot/api.py:_protected_references_only`
- The articles the answer applies are protected from tidy's removal — v60f (X7) — `src/rag/interactive_flow.py:_write_answer`
- Only the retrieved rulings, with document id and massima, may be cited. Any other ruling becomes «orientamento da verificare», and the orientations section is rendered by code — v70, v71 — `src/rag/lawyer_method.py:ruling_ok`, `unretrieved_rulings_out`, `orientations_section`, `src/rag/interactive_flow.py:method_inputs`
- A writer's paragraph mark «[N]» never counts as part of a sentence — v60f (X4) — `src/rag/plan_then_write.py:_sentence_span`, `_sentence_spans`
- «Codice della Strada 2026» keys as «Codice della strada» — v60e — `src/rag/plan_then_write.py:_code_key`
- An institution cited with another article's number has its citation replaced by the institution's own article. If no article is found, the sentence is dropped — v68 — `src/rag/record_check.py:institution_article`, `_cited_instead`, `check`
- In requested sections, the rubrica rule applies: «istituto (art. N)». The record's rubrica is used when the rows hold none — v69 — `src/rag/interactive_flow.py:rubrica_out`, `src/rag/record_check.py:_institution`
- The sources that clear an institution named for an article are tried in this order: rubrica, the code's index, the text, then the record's names — v60j — `src/rag/record_check.py:_cleared`, `_institution`
- A citation of a comma, number or letter the article does not have is dropped — v65 — `src/rag/record_check.py:phantom_item`, `article_items`, `src/rag/interactive_flow.py:phantom_commi_out`

## 9. Checks & guards

- The answer ledger records every sentence a later step removes, together with the step that removed it — v60f — `src/rag/answer_ledger.py`, `src/rag/plan_then_write.py:_step`
- A deferring ledger lets no step delete: their removals are put back as proposals — v60g (e) — `src/rag/answer_ledger.py`
- A rewrite is never counted as a removal — v60g, v60q — `src/rag/answer_ledger.py:record`, `WRITING_STEPS`
- `record_check` is the one check. It runs last, against the record, and it is the only step that removes. A proposal that no rule confirms is rejected — v60g (e) — `src/rag/record_check.py:check`, `src/rag/interactive_flow.py:_reviewed`
- `record_check` rules:
  - `esito_contraddetto`: stating an «argomento del cliente» applies contradicts it — v60i (2) — `src/rag/record_check.py:_verdict_contradicted`
  - `norma_non_applicabile`: a §2 paragraph under a «non applicabile» article — v60h (5) — `src/rag/record_check.py:check`
  - `applicazione_non_chiesta`: a paragraph on an article with no block due — v60i (1) — `src/rag/record_check.py:check`
  - `testo_della_norma`: single shot, the sentence reproduces an article's text — v60q — `src/rag/record_check.py:_quote_outside`, `_reproduces`
  - `regola_fuori_testo`: single shot, a rule sentence judged outside its comma — v60q — `src/rag/record_check.py:check`
  - `stato_soggettivo`: a counterparty's state of mind given as an unstated fact. The «il caso non dice» clause is cut and the rest is kept — v60q — `src/rag/record_check.py:_state_of_mind_cut`
  - `requisito_non_registrato`: §1 or the conclusions state a requirement of a threat that names none of its recorded elements. The writer review's verdict decides when it ran — v64, v65 — `src/rag/record_check.py:_unrecorded_requirement`, `_flagged`
  - `alternativa_senza_fatti`: an alternative with no facts behind it cited in §1 — v65 — `src/rag/record_check.py:check`
  - `domanda_nel_parere`: single shot, a question inside the parere's body — v60k — `src/rag/record_check.py:check`
  - `comma_inesistente`: since v65 this also covers numbers and letters — v65 — `src/rag/record_check.py:phantom_item`
  - `istituto_rubrica` (mark only): an institution no source clears gets «(da verificare: art. N, rubrica «…»)» — v60g (3), v60j — `src/rag/record_check.py:_institution`
- The code's own closing lines of §2 are never read as writer sentences — v60q — `src/rag/record_check.py:_code_line`
- A mark is placed before the full stop, and a sentence that ends on an abbreviation keeps its dot — v60s — `src/rag/record_check.py:_mark`
- Read-back: of a claim and its negation on the same article, the one the record supports is kept. A fact found in no source is marked «(non risulta dal caso)» — v60 (E) — `src/rag/readback.py:read_back`, `contradictions`, `unsupported_facts`
- A sentence stating what an article provides that shares no stem with its text is dropped. Claims are compared across paragraphs. Numbers inside citations are not facts. An amount derived from the case's amounts by one operation is marked «(calcolato)» — v60c — `src/rag/readback.py:ungrounded`, `contradictions_across`, `unsupported_facts`
- Places of offices and centuries are not persons — v60c — `src/rag/readback.py:_NOT_NAMES`
- A party's act or event must map to a sentence of the case — v60g (2) — `src/rag/readback.py:read_back`, `src/rag/interactive_flow.py:party_names`
- Statuses are the code's words: the writer never uses them as ordinary words — v60q — `src/rag/interactive_flow.py:term_consistency`, `src/rag/writer_rules.py:STATUS_DEFINITIONS`
- Facts are named with the case's words. An act noun from the same class (a «pugno» for the case's «schiaffo») is replaced, or the sentence is dropped. A word that prejudges the accused is replaced by the client's name — v74 — `src/rag/fact_terms.py:apply`, `data/fact_terms.json`, `src/rag/interactive_flow.py:fact_wording_inputs`
- Semantic requirement review: the writer review checks every norm that has a record (since v68, the client's own too) against the record's elements — v65, v68 — `src/rag/plan_then_write.py:_review_call`, `review_threats`, `src/rag/interactive_flow.py:requirement_context`
- A requested section with no sentence stating a requirement that a record holds is checked too — v68 — `src/rag/interactive_flow.py:requirement_out`
- Negation review element by element: an argument may not negate an element its thesis does not contest. When the review call fails, a word test is used — v74 — `src/rag/plan_then_write.py:negation_review`, `src/rag/record_check.py:negated_elements`
- The writer's procedural sentences are checked against the article's text — v60 (C) — `src/rag/procedural_facts.py:consistency`, `src/rag/interactive_flow.py:_reviewed`
- No remissione or 162-ter for a d'ufficio offence — v71 — `src/rag/lawyer_method.py:procedibility_out`
- A writer sentence attributing an orientation to the case law is removed, because orientations are the code's — v71 — `src/rag/lawyer_method.py:orientation_claims_out`
- His CONTROLLO FINALE is answered from the code's own checks and logged — v72 — `src/rag/lawyer_method.py:final_check`
- Every «da verificare» marker is moved to §10 as a whole sentence (never a clause), and duplicates are merged. Since v75b §10 holds only open questions, an orientamento to verify stays in §5, and no sentence means no point — v74, v75b — `src/rag/lawyer_method.py:collect_points`, `point_sentence`, `same_point`
- The nightly's yardstick for an answer's shape: the ten sections, the tables, three objections, the assessment, no «da verificare» outside §10, and cell-specific blocks — v70 — `src/rag/lawyer_method.py:structure_checks`
- A section name never appears in the text, because the writer names real sections — v60f (P9) — `src/rag/interactive_flow.py:cosmetic_fixes`, `answer_final`
- A §2 status line written by the code is not a fragment — v60k — `src/rag/text_tidy.py:_FRAGMENT_START_RE`

---

## 10. Rulings in the record (v78, `RULINGS_RECORD`)

- Per record article, the interpretation sections that CITES its Section, grouped by ruling, linked ones only (`cites_linked_at`); found / linked / kept logged per article — v78 — `src/rag/rulings.py:build`.
- Estremi from the file name only (`snpen|snciv`, section code, year, number, act); the chamber from `data/ruling_sections.json` (provisional); `date_enacted` else «data n.d.»; `document_number` never used — v78 — `rulings.py:parse_estremi`, `render_estremi`.
- The abstract only ranks: noun overlap with the elements in dispute, then the linking call's pertinenza (0 dropped); order Sezioni Unite, score, date; 6 graded per article, of which 3 printed, 8 per answer round robin — v78 — `rulings.py:_order_key`, `round_robin`.
- The printed passage is the sentence the linking call names among the ruling's own candidate sentences, verbatim, as «passaggio della motivazione»; none → estremi only; never «massima» — v78 — `rulings.py:link`, `ruling_line`.
- Orientamento per controverted point (an element a candidate thesis contests): ≥ 3 graded passages (of up to 6 per article, printed or not) with a majority → prevalente / minoritario with counts; else «orientamento da verificare» with the rulings listed; §5 ends with the corpus line — v78 — `rulings.py:orientation`, `section_text`.
- The writer's «Non citare pronunce» becomes «cita solo le pronunce del registro, con gli estremi indicati» (method mode); the prose check keeps only the registry's and the lawyer's rulings — v78 — `interactive_flow.py:_jobs`, `rulings.py:citable`, `lawyer_method.py:unretrieved_rulings_out`.
- The thesis and the objections calls receive the registry; a prevailing line for an element's integration lowers every thesis contesting it to «basso» — v78 — `lawyer_method.py:method_material`, `objections_material`, `rulings.py:cap_realism`.
- The lawyer's pronunce di riferimento (his file, imported onto the article's Section) printed in §5 only when written — v78 — `src/rag/landmark_rulings.py`.
- No case-law search into the v1 pool, no REFERS_TO neighbours into a topic's articles, no EDGE_EXPAND — v78 — `defensive_generation.py:_select_articles`, `topic_step.py:articles_split`, `edge_expand.py:enabled`.

### 10.1 v78b (the PO's review of the first answers)

- A quotable sentence states a rule (richiede, si configura, è integrato, non è necessario, sussiste …), holds the graded element's nouns and none of the procedural markers (motivo di ricorso, Corte territoriale, ricorrente, lamenta, addebito mosso, dep., Rv. …); the call's pick is checked again on its element — v78b — `rulings.py:quotable`, `candidate_sentences`, `link`.
- A citing abstract that does not share the article's rubrica nouns (both of «lesione personale») drops the ruling for that article (`off_rubrica`) — v78b — `rulings.py:on_rubrica`.
- Sez. 7 and the ordinanze rank after every other ruling (Sezioni Unite still first) — v78b — `rulings.py:demoted`, `_order_key`.
- A registry ruling is cited only in a sentence about its own article and on its graded element (its nouns present; a ruling with no passage, nowhere): otherwise its parenthesis, else its sentence, is removed — v78b — `lawyer_method.py:rulings_scope_out`.
- §6 and §7 (built from the plan after record_check) are reviewed: a sentence citing an article outside the answer's scope (records, circumstances, notes, the checked norms and procedure, the code-written lines, the case) is removed; an objection left empty is removed and the fallback fills §7 — v78b — `lawyer_method.py:article_scope`, `articles_out`, `compose`.
- A reported claim («apprendeva che», «aveva negato», «sostenendo che») is «allegato», never narrated fact: it can never make an element «non risulta» (v60t) — v78b — `issue_records.py:_ALLEGED_RE`, `interactive_flow.py:_SAYS_RE`.
- A victim's (persona offesa, parte civile) outcome is the querela's offence, the record marked «charged», whatever the main topic — v78b — `lawyer_method.py:assessment`.
- A record no case sentence touches (every element «da provare», nothing supporting or contradicting) leaves the method's records, unless it is the charge or the claim, the counterparty's, named by the user, or the only one left — v78b — `lawyer_method.py:method_records`, `fact_free`, `kept_without_facts`.

## 11. v79 (measurement and grounded statements)

Each rule below has synthetic tests only. None of them was tuned on a real or evaluation case.

- **79.0, A1.** A candidate the code enumerated is verifiable by construction, even when it declares no element (an eccezione, a remedy with nothing proven yet). Only a thesis of the model's own that declares no element is «unverifiable». — `lawyer_method.py:coherent_solutions`.
- **79.0, A1.** The reply's «valutazioni» are read in either shape the model gives: an object keyed by id, or a list of objects carrying their id. Decorated keys («[T1]», «T1 - …») count as that candidate. — `lawyer_method.py:valuations`, `candidate_key`.
- **79.0, A2.** The thesis selection's counts are logged in `lawyer_method_plan`, in the thesis log and in the state: offered, parsed, valued, barred, unverifiable, minoritaria, chosen, and whether the main line is the fallback the selection inserted. The v78 realism cap runs inside `method_plan`, before the counts. — `lawyer_method.py:selection_counts`.
- **79.0, A3.** The reply kept is the one with the highest score. Ties go to the smallest sha1 of the parsed reply, and both are logged. `THESIS_DETERMINISTIC` sends one sample at temperature 0. — `lawyer_method.py:choose`, `sampling`.
- **79.1, B1.** A rule statement says what a norm requires, permits, excludes or punishes, or when it applies (richiede, si configura, è integrato, non è necessario, prevede, punisce, si applica, esclude, decorre …). It stays only when it traces to one of three sources: an element of the record of the article it names, a store field of that article (procedibilità, termini, pena, competenza), or a registry ruling's quoted passage on it. A statement naming no article may trace to any of the answer's sources. Otherwise it is removed and logged with the sentence and the reason (`rule_statements_removed`). This applies to every written section, including §6's cells and §7's objections and replies. Code-written text is built from those sources and is not checked. — `src/rag/rule_statements.py`, `lawyer_method.py:compose`.
- **79.1, B2.** An objection the checks empty is dropped, and so is an objection whose reply they empty (`objections_dropped`). The standard fallback fills §7 back to three. — `lawyer_method.py:compose`, `objections_section`.
- **79.1, B2/D3.** The candidates' articles are in the answer's scope, so a sentence citing one is never removed for scope. — `lawyer_method.py:article_scope`.
- **79.1, B3.** The nightly's `grounded_output` check reads the delivered parere, minus §5, the tables and the code-written sentences. It must find no rule statement without a source and no article outside the scope. A parere the method did not write fails the check, with the reason «no sources». — `lawyer_method.py:output_check`, `scripts/nightly_sim.py:grounded_check`.
- **79.2, C1.** The first charged article stays the charge even when the first read lists it among the alternatives (it is no alternative). A second charged article that is also the client's alternative keeps the old behaviour: it is the client's line. — `posture.py:mark_sides`.
- **79.2, C2.** The charge wordings live in `data/charge_wordings.json`, not in code. v79 adds «arrestato (in flagranza) per», «giudicato responsabile», «condannato per», «rinviato a giudizio per», «imputato del delitto di» and «tratto in arresto per». — `posture.py:charge_re`.
- **79.2, C3.** A charge citing an article of a law in `data/special_laws.json` («art. 73 d.P.R. 309/1990», «art. 73, comma 5, del d.P.R. n. 309 del 1990», «art. 55 del d.lgs. 81/2008») makes a charge record, just as a code article does. Its label is mapped to the store document's article (`state["charge_aliases"]`). — `posture.py:special_cites`, `interactive_flow.py:special_law_aliases`, `decompose_records`.
- **79.2, C4.** A criminal case that states no charge takes it from the first read's main hypothesis, the same way a civil claim is derived. §1, §3 and §9 label it «ipotesi di reato (addebito non indicato nel quesito)». The reader never sees «il registro non contiene …»; with no charge at all, the reason is given in the reader's own words. — `posture.py:claim_from_read`, `lawyer_method.py:assessment`, `HYPOTHESIS_LABEL`, `NO_CHARGE_WHY`.
- **79.2, C5.** The model's «convenuto» is accepted only when the text shows someone suing, serving or prosecuting the client (citazione, causa, notifica, decreto ingiuntivo, sfratto, imputato …). Otherwise the role is «attore», and so is the default that `intake.resolve` derives. — `intake.py:sued`, `detect`, `resolve`.
- **79.2, C5.** «Assisto un/una/il/la/l' <noun>» names the client's role by that noun. — `interactive_flow.py:_CLIENT_NOUN_RE`, `stated`.
- **79.2, C5.** When the method writes a parere without being told whom the user assists, it writes from the derived role and says so in one line (`CLIENT_DERIVED`). It never says «Non mi hai indicato chi assisti» and then writes anyway. — `interactive_flow.py:settle_client`, `derived_role`.
- **79.3, D1.** A general-part defence article (c.p. 40, 41, 43, 50–56, 83, 131-bis) enters the record in a penal case in two ways: the first read names its concept (`data/general_concepts.json`), or the retrieval ranks it among the top 10 before the general-part gate drops it. It is then split into elements. With an element proven and none excluded, it is a defence line among the candidates (kind «defence», argued by the code; 50–54 stay «scriminante»). It is never a fattispecie, never the charge and never a subsumption table. Off that route the gate drops it as before. — `issue_records.py:DEFENCE_ARTICLES`, `decomposed_by_gate(defence=)`, `interactive_flow.py:defence_ranked`, `defence_wanted`, `lawyer_method.py:thesis_candidates`.
- **79.3, D2.** The retrieval's top 10 graded fattispecie (grade ≥ 2) that are not in the record are decomposed and linked as candidates (`ranked_candidate`), never as records. The answer neither prints nor assesses them. With a proven element and none excluded, one becomes a riqualificazione (offender) or a claim (claimant). Otherwise it is dropped silently (`ranked_dropped` in the log). — `interactive_flow.py:ranked_labels`, `decompose_records`, `lawyer_method.py:method_records`, `prepare`.
- **79.3, D3.** No citation check removes a sentence citing an article of the record or of the candidates: neither the writer's `record_check`, nor compose's scope check. — `interactive_flow.py:check_context`, `lawyer_method.py:candidate_articles`, `article_scope`.
- **79.3, D4.** The nightly's thesis-selection table adds `defence_lines`, `defence_gate_dropped` (criminal cells, with the articles listed) and `ranked_candidates`. — `scripts/nightly_sim.py:counts_lines`.
- **79.4, E1.** The forecast (§1, §9) is computed on the chosen main line's article: its own (riqualificazione, claim), else the article of the elements it contests or rests on, if it is a fattispecie of the record. A victim's or a parte civile's forecast stays on the querela's offence. A line on no fattispecie (the containment, a general-part defence) keeps the charge's forecast. — `lawyer_method.py:forecast`, `main_article`, `assessment(article=)`.
- **79.4, E2.** No internal wording reaches the reader. Element codes appear only inside tables: elsewhere a code's parenthesis is dropped and a bare code becomes its element's words. «L'accusa» and «aggravante» appear only in a criminal matter (`criminal`: the penal family, or no family and a criminal role); otherwise «la controparte», and a sentence on an aggravante is removed. The «registro» never appears: the reader-facing strings are reworded, §5's title is «Pronunce sulle norme applicate», and a sentence mentioning it is removed. The nightly's `reader_wording` check counts none. — `lawyer_method.py:reader_wording`, `wording_hits`, `output_check`, `scripts/nightly_sim.py:wording_check`.
- **79.4, E3.** Compose is the last check on the method's path. Each of its removals (prose checks, scope, coherence, objections, wording) enters the answer's ledger as «removed» with step `compose:<check>`. The writer's early checks stay «proposed» and record_check's removals «removed». The nightly's `ledger_removed_absent` check verifies that every sentence logged as removed (25 characters or more) is absent from the delivered parere. — `lawyer_method.py:compose` (`tracked`), `interactive_flow.py:merge_removals`, `scripts/nightly_sim.py:ledger_check`.
- **79.4, E4.** Prescription and decadenza are never a claim or a candidate. This covers c.c. 2934–2969 and any rubrica on prescrizione or decadenza. The remedies table's «eccezione di prescrizione» is no longer offered either. — `issue_records.py:prescription`, `counted`, `lawyer_method.py:prescription_cited`.
- **79.5, F1.** Each registry label the writer emits («[G1]», «(G1)», «G1») is replaced in the final text by the ruling's reference: «Cass. pen., Sez. 5, n. 1234/2025, 12.03.2025», with the date only when it is held. A label that names no registry ruling drops its sentence (`registry_labels_out` in the log). — `rulings.py:labels_out`, `reference`, `lawyer_method.py:compose`.
- **79.5, F2.** A quoted passage carries no PDF stamp («Corte di Cassazione - copia non ufficiale» and its page count) and no page count. A sentence starting inside a reference (lower case, a number, «n.», «Rv.», «Sez.», punctuation) is never a candidate passage. Party names are redacted as «[omissis]» in §5, in the registry the calls receive, and in the citable list: a name after a title (sig., dott., avv. …) and a run of capitalised words that names no office, court or act. `RULINGS_PARTY_NAMES=keep` leaves the names; the default is pending the PO's decision. — `rulings.py:strip_stamp`, `starts_clean`, `redact_names`, `printed`.
- **79.5, F3.** The nightly's `rulings_printed` check, live and offline: no «G<digit>» label, no stamp text, and no ruling cited without its head (a «Sez. …» or «n. N/YYYY» with no «Cass.» before it; an act's number such as «d.lgs. n. 81/2008» is not a ruling). — `rulings.py:printed_problems`, `scripts/nightly_sim.py:printed_check`.

### 11.1 v79.6

- **E4, restated.** Prescription and decadenza articles are never a claim or a candidate built on them, for an attore or for a convenuto's riconvenzionale. The convenuto's eccezione di prescrizione (remedies table) stays a candidate. — `issue_records.py:prescription`, `counted`.
- **F2, restated.** Party-name redaction stays the default. Every redaction is logged (verbose log `rulings_redaction` and `<state>/logs/redactions.jsonl`, with the words replaced). The stoplist is `data/redaction_stoplist.json` and grows from that log. — `rulings.py:redact_names`, `not_names`, `log_redaction`.
- **1.** Intake extracts every explicit question, in order (`questions`): a sentence ending in «?», or «chiedo se / vorrei sapere / quale … sia / è possibile …». The parere opens with «Risposte ai quesiti» before §1: one numbered answer per question, two to four sentences. The answer comes from the writer's own section on that question (checked as any prose, Block B), from §8 for a procedural question (rito, termini, impugnazione, competenza …), or otherwise from the selection (the main line and the forecast). With nothing grounded the answer is «Non risulta dagli atti / dalle norme in registro.» §9 no longer appends the quesiti. — `intake.py:questions`, `lawyer_method.py:answers_block`, `answers_of`.
- **2.** The first read returns `clienti`. With two or more clients (from the read, or «Sono il difensore di X e di Y»), each client gets its own posture, its records' sides, a charge marked on the clauses that name it (`posture.client_text`), its own candidates and its own thesis and objections calls. The rulings record is shared. The parere prints one §1 per client («Nell'interesse di …»); any other section is shared when identical, with a note saying so, and per client where it differs. Cost per extra client: four calls (thesis, argument binding, objections, reply binding), five with circumstances to grade. — `case_read.py:clean`, `interactive_flow.py:clients_of`, `method_inputs_clients`, `lawyer_method.py:compose_clients`.
- **3.** An article is repealed when its text is a repeal reference alone, a repeal note opens its text, or a status property of its node says so. Repealed articles are pruned before charge marking (`mark_sides`, `decompose_records`: never a charge, a line, an alternative or a candidate) and before the general-part re-entry. A repealed article the user names is mentioned only as «abrogato», with the repealing law (`repeal_note`). `scripts/repeal_audit.py` measures the database. — `article_text.py:repeal_kind`, `repeal_note`, `status_repealed`, `row_abrogated`, `interactive_flow.py:abrogated_labels`.
- **4.** In decomposition, elements joined by «o / od / ovvero / oppure», and categories listed as alternatives, are one element with branches (`rule_quote.join_branches`, `branches`). The penal and civil splitters never split inside a branch list, and a means or purpose clause stays its own element. A branch element may be twice the length limit. Grading: «provato» if any branch is, «non risulta» only if every branch is, else «da provare» (`issue_records.branch_combine`, `settle_branches`, from the nouns of the sentences the link found). `demote_contradicted` never demotes for one branch contradicted while another holds. The link call is told the same rule, and the sussunzione prints the branch that applies.
- **5.** A role nobody stated is derived from the text. Resisting a claim, a payment demand, a penalty or a charge gives the matter's defending role (imputato, convenuto); bringing or wanting to bring one gives the claiming role (persona offesa, attore). An entity the lawyer names («difensore della società», «assisto l'impresa») takes the side the text gives that entity. With neither, the claiming role is a reading, stated in one line (`ROLE_READING`). An unknown role never gets claimant or defendant candidates. C5 holds: the model's «convenuto» needs an action against the client. — `intake.py:derive_role`, `resolve`, `interactive_flow.py:role_reading_line`, `lawyer_method.py:thesis_candidates`.
- **6.** A keyword hit is a row the first read's institution search reached by a rubrica that lacks one of the institution's own words. It is dropped before grading when its matter (the code's Libro; c.c. Libri IV and VI by Titolo) is none of the case's: the matters of the multi-signal rows, the whole-name hits, the articles kept by name or rule, and c.p. Libro I alongside any c.p. matter. — `topic_groups.py:matter_of`, `keyword_hit`, `matter_gate`.
- **7.** The nightly adds `questions_answered`, `alternative_elements`, per-client thesis counts, and a sixth synthetic cell with two co-defendants. Every earlier check stays. — `scripts/nightly_sim.py`.

### 11.2 v79.7

- **1. Every candidate valued.** The thesis call values every candidate, keyed by its id. The code-argued ones (the containment line, claims, remedies, eccezioni, attenuanti, scriminanti, the victim's set) get only the model's realismo; their pro, contro and argument stay the code's (v74). One the code rates «basso» (nothing proven yet, an eccezione with no presupposto) is at most «medio»; a minority one stays «basso». The reply's keys are read as `T1`, `[T1]`, `T 1`, `T01`, «Tesi 1», «tesi_T1», «candidata 1», «1», the candidate's own text, a list in the candidates' order when no item names one, a list of one-key objects, or top-level keys. `valued` counts every candidate with a realismo. Before, it counted only the model-argued ones, and the prompt told the model «non valutarla» for the rest. — `lawyer_method.py:candidate_key`, `valuations`, `apply_valuations`, `score_sample`, `selection_counts`.
- **2. The imputato's enumeration is never empty.** The order is: the per-element contests in the elements' order (minoritaria where the element is integrato), the riqualificazione into each alternative the first read listed (offered with its elements open, never with one excluded), the scriminanti and defences, the attenuanti, and the containment line last. The containment line is always offered. When the charge is not fully integrato it rests on no element and is «medio» (`CODE_ARGUMENTS["contenimento_open"]`). With no charge at all its text is `CONTAINMENT_LINE`. Past the cap the minority contests go first and the containment line never goes. A charge record no one marked «threat» still gives its contests (the charged records, else the assessment's article), and the counts say where the contests came from (`threats_from`) and why it was unmarked (`no_threat`: no_charge / no_record / no_elements / unmarked). A fact-free alternative is kept as a candidate and never printed. — `lawyer_method.py:thesis_candidates`, `_no_threat_why`, `method_records`.
- **3. Block D operating.** A general-part defence enters the record:
  - when the first read or the case's own words name its concept;
  - when the retrieval ranks it among its top-ranked, by its own ranking and before any gate;
  - for 40, 41 and 43 c.p., whenever a charge is recorded.

  Each record carries its route (`defence_by`). A defence named by its concept is a line even with its elements still to prove (none excluded), and it is never dropped as «no fact». The retrieval's top-ranked articles that no gate dropped and that are not in the record are graded as candidates (`ranked_candidate`); a dropped article's stub counts as not in the record. Each top-ranked article's fate is traced for the nightly (`ranked_trace`: in record / candidate / dropped, with the reason). — `interactive_flow.py:retrieval_top`, `track_gate`, `defence_ranked`, `ranked_labels`, `defence_wanted`, `decompose_records`, `ranked_trace`.
- **4. The passage picker.** The abstract is on the article when one word begins with the rubrica's head noun by stem («lesion-», «maltratt-», «furt-»); v78 asked for every rubrica noun verbatim. The rule-verb list adds the massima forms («è configurabile», «deve essere ritenuto», «è sufficiente», «non basta», «si perfeziona» …), to be checked against the corpus with `scripts/rulings_filters.py`. The procedural exclusion stays. A sentence is on an element only when it shares one of the element's own nouns, not merely a generic one (`GENERIC_ELEMENT_NOUNS`: «profitto», «delitto», «denaro» …). Naming the article no longer adds to a sentence's score. Per article, the rulings record counts what each filter dropped (`FILTERS`). — `rulings.py:head_stem`, `on_rubrica`, `element_noun_hit`, `sentence_filter`, `candidate_sentences`, `build`.
- **Small.**
  - `--baseline` takes a file or a folder (its `nightly.json`, else its newest `.json`) and fails before any cell runs when it cannot be read.
  - The digital signature block («Firmato Da: … Emesso Da: … Serial#: …») is stripped from passages, and the printed check flags it.
  - The custody rule (`danno_cosa_in_custodia`) never fires on works entrusted to a contractor («affida … il rifacimento», an appalto).

### 11.3 v79.8

- **A. Latency.** Target: ≤ 240 s single-client, ≤ 330 s two-client.
  - A1: the matter gate before the applicability grading drops every retrieval hit outside the case's matters, a vector hit as well as a keyword one. A multi-signal row, a first-read whole-name hit and a kept article stay. (`topic_groups.matter_gate(every=True)`)
  - A2: at most `GRADE_CAP` (30) articles are graded per case. The articles the user names, the anchors and the first read's institution articles are always inside the cap; the others follow by retrieval rank. (`interactive_flow.grade_cap`, logged as `grade_funnel`)
  - A3: the thesis call takes one sample. `score_sample` gave every reply the same score, so best-of-3 only spent two calls; it returns only with a scorer that discriminates.
  - A4: the clients' objections calls (each with its reply binding) run concurrently (`interactive_flow.concurrently`, `method_objections`). The rubrica stamp still follows the writer review, because it reads the text that review produces.
  - A5: `case_counters` (graded_articles, samples_used, llm_calls, total_s, funnel) goes into the `single_shot_timings` line and into the nightly's counters table.
- **B. Claims that confer an action.**
  - B1: a claim («Fondare la domanda sull'art. X») or a riconvenzionale is built only on an article a rule admits:
    - an action of the institution module the first read names (`data/institution_modules.json`);
    - an action of a `remedies.json` matter (`actions`);
    - general obligations law (1218, 1453, 2043).

    A limiting rule (mora, prevedibilità, misura del danno, 1455, 1460 …), the other party's duty and a definitional or procedural article never are. Every candidate records the rule that admitted it (`admitted_by`), and each refusal is logged (`claims_refused`).
  - B2: the module's articles come first, in its order (2053 before 2051). 1218, 1453 and 2043 are subordinate: offered only when the module proves nothing. With a module active, nothing outside it is offered.
  - B3: a convenuto gets contests, eccezioni, a riconvenzionale and a transazione, never a claim.
  - B4: the counters are `unadmitted_candidates` (must be 0) and `main_outside_institution`. — `lawyer_method.py:claim_admission`, `active_modules`, `same_article`, `main_outside`.
- **C. Grading from facts.**
  - C1: only a fatto certo integrates an element (`issue_records.certain_fact`: grounded, stated as accertato). An allegazione, a counterparty's statement («sostiene», «afferma», «apprendeva che», «aveva negato») and the client's own claim letter («con diffida … chiedeva») leave it «da provare», saying so. §2's «fatti certi» read the same predicate, so §2 and §4 agree by construction. This reverses v60t's «grounded, an allegation too».
  - C2: a decomposed clause that states no condition is dropped at decomposition, its reason on the record (`fragments`) and in the log (`element_fragment`): a bare subject («il debitore», «uno dei contraenti»), a connective («con cui si conviene che»), a saving clause («salvo in ogni caso …»). — `rule_quote.fragment_why`.
  - C3: a d'ufficio condition the store cut at its connective is joined again (`rejoin_conditions`). A condition joined by «quando / se / qualora / purché / e» is «provato» only when the certain facts hold every part (`condition_parts`); otherwise «da provare», naming the missing part.
  - C4: the counters are `uncertain_integrated` (must be 0), `fragments_dropped` per article, `conditions_half` (must be 0) and `conditions_rejoined`. — `interactive_flow.grading_counters`.
- **D. Commi are fattispecie.**
  - D1: a comma after the first that states a special case («nei casi previsti …»), a presumption («si presume …», «si considera …») or an aggravation («la pena è aumentata …») has its elements marked `special` (`issue_records.special_comma`). The method's records split them into their own record, labelled `«2051 c.c. co. 2»` and carrying `comma_of` and its kind in the rubrica (`split_special`). The base rule's record keeps only its own elements, so its missing facts list only those. A fact-free special-comma record is v78's noise, as any record.
  - D2: the counter is `special_comma_records`.

## By version

| Tag | What it introduced |
|---|---|
| v59 | Fixes from the v58 review: terms carried by their matter's topic, table-named topics, Capo neighbours kept, requisite and penalty clauses, comma lines, the final answer pass |
| v60 | The case-first method (the PO's A–H): first read (A), topics from graded rows (B), procedural facts from the text (C), assumed client said first (D), read-back (E), round-1 grading limits (G), interaction intents (H); stage meter |
| v60b | Round-1 latency: graph reads, fusion branches and searches run in parallel |
| v60c | Abrogated articles out, hypothesis anchoring by stems or similarity, topic order, first-read questions, read-back grounding |
| v60d | First-read institutions searched and always graded, rules only in the case's areas, deceased never the client, relative dates, store-only §4 |
| v60e | Rubricated names first, misaligned rubriche out, client's side drives topics, restatement, one citation reader, negoziazione |
| v60f | The per-article record and the case's sentences, the answer ledger, fixed plan issues, one section list |
| v60g | The record as the writer's source (writer rules), `record_check` as the one removing check, distinctive elements, party top-up, vague periods |
| v60h | d'ufficio conditions as elements, mediation closed lists, sanction lines, roles are not names, per-call metering, term triggers |
| v60i | Block due / closing lines with reasons, «argomento del cliente», user message never a trigger, role words, articled prepositions, labour list |
| v60j | Open-question «da verificare» earns a block, institution clearing sources |
| v60k | Single shot: one parere per message, «il caso non dice …», updates marked «(aggiornato)», question defects |
| v60l | Legal characterisations never asked, lowercase after «il caso non dice» |
| v60m | Demotion only on excluded or contradicted distinctive elements |
| v60n | Single-shot topic cap, «Altri profili», rule anchors graded, area-code priority, rubrica first |
| v60o | Every candidate the prefilter keeps is graded (graded == kept); exempt candidates graded, never removed; identical rubriche kept |
| v60p | Covered topics' anchors earn blocks, grading in one pool, §4 triggered by the parere's own analysis |
| v60q | References not quoted text, `article_citations`, code-area and Capo gates, no «Punti da precisare», state of mind never asked, status words are the code's |
| v60r | Record rows out of the body, «Altri profili» cap, anchor-comma verdicts and «non pertinente», frame party label, no-signal «da provare» |
| v60s | Offence issues pruned, anchor comma named by the topic, assertable-only §1/§4, bare article lines out, known client line |
| v60t | Verbatim grounding for «provato», attributed sentences and the counterparty's version, main line and its anchor, asserting sections by title |
| v61 | Intake: read once per request, decides the route; out-of-scope copy; short-topic rule moved into it; atto drafted with the case record |
| v62 | Posture (rito, role, stage): act and goal table, threats and weapons, counted fattispecie, stage-driven §4, requested blocks in order |
| v63 | Cenni anchored on articles, alternatives section, threat elements by name, querela term only where it applies |
| v64 | General part never graded (enters only by its roads), alternatives before §4 with SÌ/NO, unrecorded requirement removal, fragment paragraphs |
| v65 | Quesiti and named articles with their verb, current-status sentences, special-law gate, user-named blocks, phantom items, store numbers in §1, no «non verificabile» |
| v66 | §4's terms from the procedural store through one accessor, decorrenze, ambiguous terms never rewritten |
| v67 | Term-owning acts in the store, «1-bis» commas as their own commas |
| v68 | Named retry, five-or-more requested blocks give a short body, weapons argued, «Altri profili» by code book, mis-cited institutions, cenni/quesito grounding |
| v69 | Body cap of seven blocks, multi-question quesiti, labour rito and c.p.c. 409-447 gate, no-date quesiti, plain cenni area |
| v70 | The lawyer's method (`LAWYER_METHOD`): ten sections, his rules in every writer call, subsumption tables, per-call thinking, pre-filing list restored, the nightly |
| v71 | Thesis coherence by declared element ids, retrieved rulings only, outcome rules, charge from intake, thesis log |
| v72 | Theses enumerated by the code, best of 3, conclusions and steps from the selection, three objections, notes di inquadramento, shared citation parser |
| v73 | Objections after the selection, steps from the selected line, `request_case_first` and `intake_source` |
| v74 | Every record decomposed (fattispecie gate), code-argued candidates, argument binding and negation review, `fact_terms`, whole-sentence points |
| v75 | Placeholder thesis refused, role-based candidate sets and remedies table, claim from the first read, intake rito as the tables' key, abbreviations, regulations |
| v75b | Charged prose from the sussunzione rows, attore's open claim, objection replies bound, penal-rito-only §4 lines, §10 open questions only |
| v76 | Method never silent, status needs a sentence, general penal splitter, ranking by realism and proof, civil fallback, pleaded claim, c.c. Libro IV Titoli, giudice di pace by article |
| v76b | Element splitting: riserva and default clauses, qualified subjects, means and purpose markers, civil split, special-law penal provisions, non-fattispecie rubriche |
| v77 | No pipeline rule: version footer and `/api/version`, held-out set, acceptance on the nightly plus the lawyer's held-out scores |
| v79.6 | Questions answered first, several clients, repealed articles, alternative elements, role derived, keyword hits by matter |
| v79 | Measurement (selection counts, deterministic choice and mode, code candidates never unverifiable); grounded rule statements; then charge and role, defences, forecast and wording, rulings as printed (79.2–79.5) |
