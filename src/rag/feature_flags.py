"""On/off switches for retrieval and answer behaviour, read from the environment.

Read on every call (.env is loaded at startup, so a change needs a restart).
Only true/false values count (1/0, yes/no, on/off also accepted); anything
else falls back to the default, so a typo never switches a feature on.

  LEGACY_LAW_HINT_FALLBACK     true:  today's document lock (round 3, item 1)
  ISSUE_SPOTTING_ENABLED       false (item 2)
  APPLICABILITY_CHECK_ENABLED  false (item 3)
  STRATEGY_DRAFT_RULES_ENABLED false (item 4)
  SCOPED_DOTTRINA_ENABLED      false (item 5)
  CASE_ANALYSIS_ENABLED        false: structured fact model + per-issue exact
                                fetch/search in run_case_analysis_pipeline,
                                any document (not just the six codes) —
                                replaces _analyse_case/_select_articles with
                                _analyse_case_v2/_select_articles_v2 there.
                                Not yet wired into the rag route's graph.
  CASE_POOL_FUSION_ENABLED     false (requires CASE_ANALYSIS_ENABLED): adds
                                a ranked pool on top of _select_articles_v2's
                                own selection — that selection (CURRENT)
                                union raw-text search (BASELINE) union
                                distilled-bullet search (DISTILL), ranked by
                                fusion and capped — see
                                _select_articles_fusion.
  HEADING_INDEX_ENABLED        false: resolves an institution against a
                                dedicated vector index over every primary
                                Section's titolo_articolo, by meaning
                                rather than shared words, when the word/
                                stem title-match fails — see
                                _heading_index_match and scripts/
                                build_heading_index.py (not built here;
                                that script has the exact CREATE VECTOR
                                INDEX statement and a size/time estimate).
  STRUCTURED_ANALYSIS_ENABLED  false (requires CASE_ANALYSIS_ENABLED): the case
                                analysis in two short calls instead of one --
                                the facts sentence by sentence, in neutral
                                form, then a fixed grid for the case's area
                                (penale / civile / amministrativo), each slot
                                pertinent or not, with the institution in
                                words for the pertinent ones. Same output
                                shape as _analyse_case_v2; falls back to it on
                                any failure. See _analyse_case_structured.
  ISSUE_CHECKLISTS_ENABLED     false: appends general issue checklists, by
                                area (penale, obbligazioni e contratti,
                                diritti reali e possesso, successioni,
                                diritto amministrativo), to _analyse_case_v2's
                                own prompt — methods to check for, never
                                article numbers, and only ones the facts
                                actually raise. Targets the generalisation
                                gap the held-out set found: the model's own
                                "questioni" never raising an issue at all,
                                so there was nothing for retrieval to search
                                on. See _ISSUE_CHECKLISTS_BLOCK.
  FUSION_CLEAN_ENABLED         false (acts only with CASE_POOL_FUSION_ENABLED):
                                the fusion pool's raw-text and bullet
                                searches on primary documents only, minus
                                books, templates and contract models by name
                                -- scripts/exp_distill.py's FUSION_CLEAN.
  CASE_FRAME_ENABLED           false: one call before the case analysis
                                reads the case's frame (facts, procedural
                                posture, areas, explicit questions, central
                                issues); fed to the analysis, searched as
                                its own queries by the fusion pool, the
                                questions given to the draft. See _case_frame.
  METHOD_CANDIDATES_ENABLED    false (acts only with APPLICABILITY_CHECK_
                                ENABLED): the method articles of the case's
                                areas and the procedural articles of its
                                posture (src/rag/method_candidates.py) added
                                as candidates, then filtered by the check.
  DRAFT_CITATIONS_ONLY_ENABLED false: the sources returned with a parere are
                                only the articles and rulings its text cites
                                (with their code / by name or number), not
                                every retrieved candidate. See _draft_sources.
  CLIENT_STANCE_CHECK_ENABLED  false: one call (temperature 0) checks section
                                3 for arguments that favour the other side;
                                if any, section 3 is rewritten once with that
                                list. See _check_client_stance.
  CHAT_KEEPALIVE_ENABLED       false: /api/chat replies not ready after 10 s
                                are streamed (a space every 20 s, then the
                                JSON) so Vercel's 120-second limit on a
                                silent server never cancels a long strategy.
                                See src/chatbot/api.py, chat().
  PLAN_THEN_WRITE_ENABLED      false: the parere as a plan (client, central
                                question, per issue articles, rulings by full
                                reference, argument, objection, rebuttal),
                                checked for the client's side, then six
                                sections written in parallel from it, with the
                                practical checks of the case's posture and
                                area; rulings named by their full reference
                                ("Cass. pen., sez. V, sent. n. 190/2026").
                                Falls back to the one-call parere on any
                                failure. See src/rag/plan_then_write.py.
  RULING_CARDS_ENABLED         false: the strategy's rulings chosen first from
                                the precomputed ruling cards in the files of
                                RULING_CARDS_FILES (comma-separated JSONL,
                                scripts/exp_ruling_cards.py), by the case's
                                central articles, keywords and the client's
                                side; each passage led by the card's outcome,
                                side and verified quote. See ruling_cards.py.
  PLAN_KEEP_APPLICABLE_ENABLED false (acts with PLAN_THEN_WRITE_ENABLED): the plan
                                keeps every article the applicability check let
                                through (in the case's areas): placed under an
                                issue or in a last "Altre norme rilevanti" one,
                                never removed by the plan check, up to 7 issues,
                                each discussed by name in sections 3 and 4.
                                See plan_then_write.py.
  VOTE_POOL_ENABLED            false (acts in the fusion pool, CASE_POOL_FUSION_
                                ENABLED): the articles cited by the rulings most
                                similar to the case's facts, IDF-weighted (the
                                cards' applied ones where a ruling has a card),
                                joined to the pool as VOTE_POOL_MODE says
                                (append:N or rrf:W_POOL:W_VOTE); VOTE_POOL_K
                                rulings' sections per channel, VOTE_POOL_IDF_
                                FILE for the counts. Tagged _vote: checked for
                                applicability, never removed by the plan check.
                                See vote_pool.py.
  QUALIFICATION_CHECK_ENABLED  false: every cited article outside the case's
                                area/posture lists is checked, by its real
                                heading, against the use the draft makes of
                                it; a mismatch is flagged [DA VERIFICARE: l'art.
                                N riguarda «...»]. See _check_qualifications.

The five below came from scripts/diagnose_long.py's run of 25 Oct (33 long cases):
  CHECKLIST_GATED_ENABLED      false: the method lists and the practical checklist
                                only for the case's verified area x posture (a
                                posture of another family than the areas' is
                                dropped); the lists' candidates pass the
                                applicability check or go; a checklist item only
                                if every article it names went through the check;
                                mediation steps only before a civil suit. See
                                method_candidates.case_lists, practical_checklist.
                                gated_checks.
  FRAME_SANITY_ENABLED         false: a criminal proceeding, conviction or an offence
                                the text asks about puts "penale" among the frame's
                                areas and turns a civil or "altro" posture into a
                                criminal one; the areas are then uncertain and the
                                vote's area filter drops no code. See _sane_frame.
  RESOLUTION_STRICT_ENABLED    false: an institution is not resolved through a
                                generic rubrica alone ("Risarcimento dei danni",
                                "Prescrizione") unless the model's own number names
                                that article; a full-text resolution counts only
                                when the rubrica shares a word with the institution.
                                See _resolve_institution.
  CHECK_PRIORITY_ENABLED       false: the applicability check's 36 slots go to the
                                voted articles, then multi-signal ones, then the
                                rest, the method lists last; an article left
                                unchecked is removed. See _filter_applicable_articles.
  PLAN_KEEP_CHECKED_ENABLED    false (acts with PLAN_THEN_WRITE_ENABLED): every
                                article the check judged applicabile or
                                alternativa is in the plan, grouped by issue, up
                                to 7 issues; never removed by the plan check. Over
                                PLAN_KEEP_APPLICABLE_ENABLED when both are on. With
                                CHECK_GRADED_ENABLED: only grades 2-3 and the
                                articles resolved for a central issue, each followed
                                by the cross-references its text cites, at most 14,
                                none forced into a catch-all issue (_forced); the
                                sections get each forced article with its issue, to
                                cite it only there, and lose any citation their own
                                material lacks and the case does not name (the
                                sentence rewritten without it, or dropped:
                                _cite_only_material). See plan_then_write.
  CHECK_GRADED_ENABLED         false (acts with APPLICABILITY_CHECK_ENABLED): the
                                check grades each article 0-3 (0 unrelated or
                                another area, 1 tangential, 2 an issue of the case,
                                3 its central question) with the issue it serves,
                                from its rubrica and first 400 characters, for up
                                to 72 articles (GRADED_MAX_CHECKED sets another
                                number) (one found by the vote or by two or
                                more channels: a doubt between 1 and 2 goes to 2);
                                grade 0 leaves the pool before the prompt. Articles without a grade (skipped, a cut
                                reply, a failed call) go once more in smaller
                                batches; still none, they stay as "non valutato":
                                not removed, not counted as passing, not kept by
                                the plan. See _GRADED_SYSTEM, _graded_verdicts.
  FLAGGED_REMOVE_ENABLED       false: after the draft, a citation QUALIFICATION_CHECK_
                                ENABLED finds incompatible with the article's rubrica
                                is removed (the sentence rewritten without it, or
                                dropped; a heading "**Art. N ...**" with its paragraph),
                                unless the pipeline has its own evidence for the
                                article (named in the case, forced into the plan,
                                graded 2-3, voted, two or more channels, central):
                                then it stays, its mark dropped when the check calls
                                the mismatch only the rubrica's wording. And an
                                article of a document that is not a law (maxims,
                                books, contract models; a CCNL unless the case is
                                about one) is not cited. See _check_qualifications,
                                _citation_hygiene. A removed citation takes its whole
                                sentence (never a rewriting, which left fragments: the
                                live answer of 26 Oct), here and in the sections' own
                                citation check; then a "Questa disposizione ..." with
                                no norm before it under the same heading, an "Inoltre,
                                conferma che ..." with no subject and an empty tail
                                ("..., in conformità.") go too (drop_dangling, also on
                                every section with WRITER_RULES_ENABLED or the graded
                                discipline; logged as case_analysis_dangling).
  GRADER_EVIDENCE_ENABLED      false (acts with CHECK_GRADED_ENABLED): the grader sees
                                per article what found it (the searches, how many
                                similar rulings apply it, a central issue, the area),
                                with the rule "applied by 3+ similar rulings and of
                                the case's area: at least 2", grades it per numbered
                                issue of the case by how directly it governs it (the
                                worked examples only with GRADER_EXAMPLES_ENABLED, off);
                                PLAN_KEEP_CHECKED forces up to 18,
                                offered to the plan by issue. scripts/grader_bench.py
                                compares the prompt versions on a run's saved pools.
  FACT_SIGNALS_ENABLED         false: one call reads the case's facts in a fixed
                                schema (parties and roles, property history,
                                possession, visible works, dated events, injuries,
                                money, contracts, proceedings, questions); the code
                                computes the legal terms from the dates (possessory
                                year, usucapione, prescrizione, querela, appeals,
                                TAR, guarantees) and fires the general rules of
                                src/rag/fact_signal_rules.yaml within the frame's
                                areas, whose institutions join the frame's central
                                issues and resolve only within each rule's article
                                ranges (mandatory for every rule), to an article whose rubrica shares a word
                                with them; the terms go to the plan and the
                                procedural section as facts, run to the case's own
                                present date (never the real today). See
                                src/rag/fact_signals.py, scripts/export_fact_rules.py,
                                scripts/fact_signals_bench.py.
  WRITER_RULES_ENABLED         false (acts with PLAN_THEN_WRITE_ENABLED; the review also
                                on a one-call parere): the sections cite an article only
                                where it supports an argument (never "every plan
                                article", never "non pertinente/non applicabile"); the
                                burden of proof only where the norm allocates it, in
                                its words; no conduct, knowledge or intent the case does
                                not state; facts favourable when they ground the client's
                                claim; a word budget per section, a cut section written
                                again or trimmed; no deep-dive links. After writing: a
                                sentence citing an article only to say it does not
                                apply, or a burden template whose norm allocates
                                nothing, is dropped; a sentence asserting a party's
                                conduct, knowledge or intent the case does not hold is
                                logged (arguments, quotes of the case, of a norm or a
                                ruling, citations and evidence lists never count), and
                                marked "[fatto non indicato nel caso: da verificare]"
                                only with FACT_MARK_INLINE_ENABLED (default off); one
                                proposing an
                                action against the client or a co-client is dropped
                                (their negligence is only the opponent's defence). A
                                party with several roles is examined for each. The plan
                                check and the qualification check (which reads the
                                article's opening text) reject an article about the
                                inverse situation; the qualification check removes it
                                even when protected, unless the case names it. Logged
                                as case_analysis_writer_rules. See plan_then_write.
  FACT_REGIMES_ENABLED         false: the typed facts (FACT_SIGNALS' extraction, run
                                also with that flag off) qualify the parties (a natural
                                person contracting for private use with a business is a
                                consumer; a public body; a public official), whose
                                special regimes (consumer guarantee, consumer's forum,
                                unfair terms; the notion of public official) join the
                                central issues; every applicable term of the term table
                                is stated with its article, DA VERIFICARE when the case
                                gives no dates; the accertamento tecnico preventivo
                                (696/696-bis c.p.c.) for defects or evidence that may be
                                lost and the art. 700 c.p.c. for safety or urgency are
                                offered to the plan and the procedural section, with
                                their texts. A person's death or injury by negligence
                                brings the criminal route (589/590 c.p., querela, parte
                                civile, 2947 co. 3 c.c.) also in a civil case; a death,
                                the heads of damage (iure proprio, iure hereditatis,
                                the insurers); a party with several activities, each
                                role. The procedural section treats each block it gets
                                (terms, tools, criminal route, heads of damage; the measures of art. 614-bis
                                c.p.c. when the client asks for a sanction or a penalty
                                against who does not comply) in its
                                own paragraph, is written once more if it leaves one
                                out; offered vs written logged as
                                case_analysis_material_use. A claimant client (seeks damages or an action)
                                with no claim received gets a claimant frame and no
                                "respond to a claim" checklist item. See
                                fact_signals.qualify_parties, procedural_tools,
                                claim_posture.
  STRICT_ARTICLES_ENABLED      false (the lawyer's review, 26 Oct): an institution's
                                article found by one of the issue's keywords, not by its
                                own name or number, or by full text, counts only if its
                                rubrica shares a word with the institution beyond the
                                generic ones (garanzia -> 492 c.c., responsabilità
                                civile -> 1917 c.c., risarcimento -> 2767 c.c.
                                refused); an institution a code table names the article
                                of is that article, also when the analysis names it
                                itself (foro del consumatore -> art. 33 Cod. cons., with
                                its note to the writer). Round 3: "prestazione" is a
                                generic word too (2116 c.c.); with FLAGGED_REMOVE_ENABLED,
                                a citation the qualification check flags but the pipeline's
                                evidence keeps (graded, forced, voted, channels, central --
                                not named in the case, not a procedural tool) gets one call
                                on the case and the article's full text: governing no
                                question of the case, it is removed (_second_opinion).
                                With this flag or FACT_SIGNALS_ENABLED, an institution
                                matching an article only on a generic rubrica word
                                ("usucapione", "prescrizione", "accessioni") resolves
                                only inside that word's chapter of the code: never art.
                                2689 c.c. for usucapione (fact_signals.
                                GENERIC_RUBRICA_RANGES).
  RULING_FIT_ENABLED           false (the lawyer's review, 26 Oct): before the plan, one
                                call asks what each ruling can serve for: a fact the
                                case describes meets its conditions ("fatti", which
                                fact), a general rule on a question the case raises
                                ("regola"), a step the client can take now ("passo"),
                                or nothing ("nessuno": another question, or facts the
                                case does not describe). Nothing, or a "fact" not in the
                                case's words, and the ruling goes; each kept one
                                reaches the plan and the writer led by its use. Logged
                                as case_analysis_ruling_fit. See _ruling_fit. A kept
                                ruling whose outcome or principle goes against the
                                client's position (its card's role when carded, else
                                "verso" in the same call) is adverse: out of the plan's
                                supporting rulings into the issue's "pronunce della
                                controparte", cited only in the objection with the
                                reply; a written sentence citing it with no objection
                                or reply word is dropped (case_analysis_adverse_rulings).
  REVIEW_FIXES_ENABLED         false (the lawyer's review, 26 Oct; acts with
                                PLAN_THEN_WRITE_ENABLED and FACT_REGIMES_ENABLED): the
                                inquadramento names the facts that decide the case
                                instead of labelling each one; the procedural section
                                writes every item of each block (iure hereditatis);
                                after the review, a sentence stating as certain what the
                                case does not say about a deadline, a report, a party's
                                proof, knowledge or concealment, or how soon a death
                                came, is rewritten as a condition
                                (case_analysis_unsupported_claims); a sale whose seller
                                stepped in to repair gets the recognition of the defect
                                (1495 co. 2, 2944 c.c.) as a central issue with art.
                                1495 c.c. and as a note on the terms; no art. 700 c.p.c.
                                after a death. Round 3: the consumer read from wider
                                private-use words (diporto, svago, vacanze, ospiti ...),
                                and with no purpose extracted from those words in the
                                events or contracts of a natural person whose occupation
                                has nothing to do with the purchase; the rewrite never
                                touches a sentence stating a condition or restating a
                                cited article's text; a criminal route still missing after
                                the rewrite gets its standard paragraph. See plan_then_write.
                                rewrite_unsupported, standard_blocks, fact_signals.
                                _recognition_note, _private_cues. Round 4 (the live
                                answer of 26 Oct): a right the facts show already
                                acquired (a fact rule marked acquired_right, or the
                                analysis) is argued first, its recognition and
                                protection; a constitutive remedy (passaggio coattivo)
                                last, "In via subordinata"; each explicit question of
                                the request has its own answer and treatment (the plan
                                made once more when one lacks it, the answers section
                                once more when it leaves one out); issues on the same
                                governing article or with near-identical titles merged;
                                an evidence list identical to an earlier one replaced by
                                a reference. See order_for_acquired, explicit_questions,
                                merge_duplicate_issues, dedupe_evidence.

  ANSWER_INVARIANTS_ENABLED    false (27 Oct): one general layer of checks on every strategy
                                answer after its sections are written, each logged found /
                                fixed / remaining (case_analysis_invariants): I1 every cited
                                article is checked material (graded 2-3, user-named, an
                                offered procedural tool, a computed term's article), else its
                                sentence goes; I2 every citation compared with its article's
                                rubrica and text, a mismatch removes the sentence (graded 3 or
                                user-named: the clause, else a "(da verificare: ...)" mark);
                                I3 an institution of the chapter table (usucapione 1158-1167
                                and 1061, servitù 1027-1099, possesso 1140-1172, passaggio
                                coattivo 1051-1055, ...) resolves only inside its chapter, the
                                model's number too, else the best rubrica inside it (fact
                                rules too); I4 a cited ruling shares a substantive article
                                with the answer by CITES, has a substantive holding, and if
                                adverse sits only in an objection or reply; I5 no internal
                                tokens (status labels, ids, placeholders, "1158.0.0"), the
                                terms given to the writer in plain Italian; I6 one call per
                                section rewrites ungrammatical sentences, refused if a
                                citation number changes; I7 a duration stated must be in the
                                term table, the case or the cited article's text. See
                                src/rag/answer_invariants.py.

Evidence tiers (27 Oct, src/rag/evidence_tiers.py; no flag of their own): computed in the fusion pool
(CASE_POOL_FUSION_ENABLED) and carried on each row -- T1 named in the case, resolved from a fired fact rule (or a
procedural tool the facts call for), exact rubrica match of an analysis institution; T2 applied by 3+ similar rulings
or found by 2+ channels; T3 one channel. Inside each step's own flag: every capacity limit (the fusion cap, the
checked slots, the plan's forced list -- now 18 --, the prompt's trimming and refill) orders by tier, grade, fusion
rank and never drops a T1/T2 candidate (past a limit it stays, ungraded or unforced); a T1/T2 article is removed
(grade 0, incompatibile, qualification mismatch, second opinion) only by two agreeing judgements or a contradicting
fact found in the case, else demoted (never forced into the plan, still the writer's); the forced list orders by
grade, tier, fusion rank, article id. Logged as evidence_tiers, evidence_cutoff, evidence_removal.

The three below are the source-tiered retrieval (26 Oct): the AUTHORITY tier is the primary
documents (codes, laws), the INTERPRETATION tier the rulings, filtered by what the authority found.
  ROUTER_ENABLED               false: each chat request is classified narrative (a fact pattern:
                                several sentences, two or more parties, a sequence of events,
                                and a request for action, an opinion or a parere), direct (a
                                question about a norm, an institution, a term or a procedure),
                                ccnl (a collective agreement, or an employment question that
                                needs one: its own path with CCNL_PATH_ENABLED, else tagged,
                                today's path) or document (an upload: its
                                own path) -- by signals first, one short LLM call only for a
                                tie; logged with its reasons (route_chosen, chat_route). The
                                intents that decided before keep precedence (defensive draft,
                                generation, a calculation pending or asked for); a "rag"
                                request routed narrative goes to the strategy pipeline. In the
                                pipeline the case frame must confirm a narrative (two or more
                                parties and events), else it is treated as direct. See
                                src/rag/query_router.py.
  PRIMARY_FIRST_ENABLED        false: on a direct question (all graph turns without the
                                router), every article-finding node of the chat's graph keeps
                                only primary and special documents' rows, each node's hits
                                logged by type (source_tier_hits, nothing else may be left);
                                the rulings join only at the synthesis, those whose Sections
                                CITE an article the answer is built on (Cassation boilerplate
                                excluded), by similarity to the question, no vote. Scored by
                                tests/eval_narrative.py --direct. See src/rag/source_tiers.py.
  HYBRID_NARRATIVE_ENABLED     false (acts in the fusion pool, CASE_POOL_FUSION_ENABLED; on a
                                request routed narrative, every strategy run without the
                                router): the rulings similar to the facts are read first as a
                                map -- the issues their cards decide and the analysis' issues
                                searched in the primary documents as one more signal, the
                                articles they apply only hypotheses, in place of the vote: a
                                hypothesis is a candidate when it exists in the primary corpus
                                and a primary channel finds it (it then passes the cap) or 3+
                                of the top similar rulings of the case's area apply it; graded
                                like any article. Every article row is from a primary document
                                (source_tier_hits "authority"). The writer's rulings are those
                                citing at least one article kept by the check, by similarity
                                to the facts x articles shared, then RULING_FIT. See
                                source_tiers.ruling_map, _support_rulings.

  ARTICLE_RELATIONS_ENABLED    false (relations agent v0, 28 Oct; acts in the fusion pool): reads
                                ARTICLE_RELATIONS_FILE, which scripts/maint_relations.py builds
                                from the civil ruling cards, the CITES edges and the Codice
                                Civile's texts (it never writes the graph): after the cap, up to
                                6 c.c. articles related to the named, resolved and first 5 fused
                                candidates (applied together by the same rulings, cited
                                together, named by the article's text), and in a civil case up
                                to 4 whose rulings' profile matches the case. Never a signal,
                                never exempt: graded like any candidate; tagged _related with
                                why; logged as article_relations_added. Measured with
                                scripts/relations_compare.py. See src/rag/article_relations.py.
  CCNL_PATH_ENABLED            false (28 Oct, after scripts/audit_ccnl.py): a turn the router routes
                                ccnl (decided by its signals, with or without ROUTER_ENABLED; a
                                pending turn or a calculation asked keeps the graph) is answered from
                                the agreement: the contract from the sector index (517 sectors, common
                                names, a variant the text does not settle used as an assumption to
                                confirm), the version in force at the case's date (the Testo
                                Consolidato covering it, else the latest CCNL; the later accordi /
                                verbali / rinnovi, a signed one over an ipotesi), the clauses inside
                                those documents only (v2 cosine + lexical, grown to their neighbouring
                                Sections, cited by "Art. N - Title"; pay rows read, never inferred;
                                the provider's watermark stripped), the primary labour law of the
                                topic and the rulings citing it. Paraphrase only: a passage copied
                                from a clause is rewritten once, then dropped; an amount not in the
                                material goes with its sentence. Falls back to the graph when no
                                contract is found or on any failure. Logged as ccnl_route,
                                ccnl_contract, ccnl_version, ccnl_clauses, ccnl_framework, ccnl_path.
                                Scored by tests/eval_narrative.py --ccnl. See src/rag/ccnl_path.py.
  PROTECTED_PARAPHRASE_ENABLED false (29 Oct): protected sources are never reproduced, on every answer path (the
                                chat's synthesis, the strategy writer, the CCNL path, document generation and
                                analysis). Protected: document_type special and ccnl, the documents named in
                                PROTECTED_DOCUMENT_NAMES / PROTECTED_DOCUMENTS_FILE (exact names; candidates
                                from scripts/protected_candidates.py), and any non-primary document whose text
                                carries a reproduction notice -- each row marked at retrieval. The writer gets
                                them tagged with a rule (paraphrase, cite by reference, never quote; short
                                figures as data); every answer is post-checked against the protected texts used
                                (10+ consecutive words or 3-shingle Jaccard >= 0.5: rewritten once keeping its
                                numbers, else dropped); notices stripped everywhere; the sources shown (chat
                                response, saved session, section endpoint) give a protected document's reference
                                only, primary law keeps its text. Counted with the flag off too (logged
                                protected_paraphrase / protected_sources_shown), reported by tests/
                                eval_narrative.py --direct / --ccnl, scripts/diagnose_long.py and scripts/
                                dryrun_strategy.py. See src/rag/protected_sources.py.
  COMMA_MATERIAL_ENABLED       false (29 Oct, after scripts/fetch_audit.py on run 0812: granularity the largest
                                measurable defect): an article reaches the writer as its commi ("N.k.0" Sections,
                                else its text split), labelled "(co. K)", not as its first N characters: within
                                the same per-article budget the commi whose words best match the issue it serves
                                (the analysis question before the plan, the plan issue after), the left-out ones
                                listed, a comma the case or the plan names always given -- read before the
                                sections when the row lacks it (one read, no model call). Logged per writer call
                                as comma_material (which commi of each article were given), comma_material_plan,
                                comma_material_fetched. Measured by scripts/fetch_audit.py ("comma not given /
                                beyond the truncation", target near 0). See src/rag/comma_material.py.
  GROUNDED_REVIEW_ENABLED      false (30 Oct, after scripts/fetch_audit.py on run 0812, judged: 28% of the
                                rule-sentences not in the writer's material, 27% of the cited claims on an
                                article never given): after the sections, each section's rule / institution /
                                procedure / holding sentences judged against that section's own material
                                (temperature 0, batches, fetch_audit's own judge). Not supported: an article it
                                cites that the material lacks is read (one read) and the sentence judged again
                                with it; otherwise rewritten from the material when it supports a version, else
                                as a pointer ("va verificata la disciplina dell'art. ...") keeping every
                                citation; dropped only as filler citing nothing. Supported sentences are never
                                touched. Logged as grounded_review. See src/rag/grounded_review.py.
  ROLE_BLOCKS_ENABLED          false (30 Oct, fetch_audit's role mechanism on run 0812): the criminal profile of a
                                personal harm and the criminal postures' checks follow the client's side -- for
                                the accused, defences, procedibility, the offended party as parte civile against
                                the client, the civil exposure and art. 2044 c.c.; for the offended party, the
                                querela, the parte civile (art. 76 c.p.p.) and the offended party's rights in the
                                proceedings. Unknown side: unchanged. See fact_signals.fit_to_role and
                                practical_checklist.VICTIM_POSTURE_CHECKS.
"""

import logging
import os

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    word = value.strip().lower()
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    logger.warning("%s=%r is not true/false: using %s", name, value, default)
    return default
