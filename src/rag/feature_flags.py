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
                                CHECK_GRADED_ENABLED: only grades 2-3, at most 12 by
                                grade then fusion rank, and none forced into a
                                catch-all issue. See plan_then_write._group_checked.
  CHECK_GRADED_ENABLED         false (acts with APPLICABILITY_CHECK_ENABLED): the
                                check grades each article 0-3 (0 unrelated or
                                another area, 1 tangential, 2 an issue of the case,
                                3 its central question) with the issue it serves,
                                from its rubrica and first 400 characters; grade 0
                                leaves the pool before the prompt. See _GRADED_SYSTEM.
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
