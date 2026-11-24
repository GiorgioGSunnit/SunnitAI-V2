"""v79.15.0 (item 1): the verified analysis step -- one model call between the case frame and rendering decides the
leading and the subordinate line and every schema element's esito; the code verifies every citation before anything
counts. The old engine (lawyer_method's schema matcher -> thesis generator -> score) no longer decides when the flag
is on: build B (the rendering section at the end of this module; interactive_flow.analysis_step_run,
lawyer_method.compose / answers_block / legacy_plan) prints §1, §4's element tables, §6, §9, every «x su y» count and
the answers' line from the verified analysis only, and logs the old engine's plan («legacy_plan») without printing it.

Behind ANALYSIS_STEP (default off: run() returns {"status": "off"}, nothing is called, no output changes). The prompt
and the word lists are data/analysis_step.json (DATA_FILE); the remedy / general-part set is data/remedy_articles.json
(REMEDY_FILE). The prompt is hashed (prompt_hash) and the hash logged with every run.

The input (make_input; inputs(state) reads it from the flow's state):
  - the verified frame -- only its verified fields (case_frame.held's «frame»: a field that failed verification is not
    in it): cliente, ruolo, rito, stadio, addebito_o_domanda (articolo, descrizione), istituto, parti (nome, lato,
    ruolo_nel_fatto);
  - the fitted norms -- the records left after applicability_fit (not dropped, not removed by a verified «no»), each
    with its rubrica, its schema elements (the record's requisites: what §4 tables; ELEMENT_SCHEMAS on, the schema's
    own) and its text as numbered periods (applicability.article_ids -- the fit's own numbering, one splitter: the
    commi, each cut at a full stop before a capital or an opening mark, abbreviations never ending one, numbered
    1..n across the article: «A640cp.1»); a norm of the remedy set marked so in the prompt;
  - the case's sentences numbered S1..Sn exactly as the flow numbers them (issue_records.sentences, the state's
    «sentences»: the case, then the client's precisazioni, then the system's assumptions) -- the system's assumptions
    (source «sistema») never sent and never valid ids (an assumption is no fact); their numbers unchanged;
  - the user's questions (the frame's quesiti, else interactive_flow.asked_questions).

The call (call_once): thinking off, temperature 0, a JSON reply (applicability.request_kwargs' body, this step's
max_tokens), on a worker thread waited at most the step's remaining budget -- «deadline_s» (40 s) from the step's
start, the re-ask included (the applicability batching/deadline pattern: a call not back in time is abandoned, its
pool shut down without waiting). At most 2 calls: the first, and one re-ask when the first's verification rejects a
line (any reason below), leaves out the leading line, or misses the frame's charge -- the one re-ask carries every
reason, one per line («reask»). The first call failed, not back in time or no JSON object: status failed / timeout /
malformed, analysis None -- nothing from the step is printed. The re-ask failing: the first reply's verification
stands, with the fallbacks.

The verification (verify, pure code, no model):
  - every S-id exists among the sentences sent; every A-id exists among the periods of the article it is cited for
    (a line's: its own article; a law statement's: any fitted article); a claim citing an id that does not exist is
    void (a line rejected, an element's verdict «da_provare», a law statement dropped) -- counted analysis_bad_sid /
    analysis_bad_aid;
  - elemento (and a line's elemento_in_gioco) is the E-id of one of the article's schema elements (v79.15.1: by id,
    existence only -- element_of); else the verdict is dropped (analysis_bad_eid) or the line rejected;
  - esito «provato» needs at least one cited S-id whose sentence is not a reported assertion of a party other than
    the client (reported_by): parte_che_afferma names a party that is not the client, or the sentence carries 79.14.3's
    alleged_by (interactive_flow.alleging_party: another party's allegation), or ANY of its asserting phrases
    (_subject_spans: the words right before an assertion verb -- data «assertion_verbs», whole words, any case -- in the
    same clause; the words after «secondo» / «a detta di» … -- data «speaker_prefixes» -- or after «per» at the
    sentence's start -- «speaker_prefixes_initial»; a passive's agent, «accusato da …») names as its speaker (speaker:
    a subject's last mention, a phrase after a marker its first) a non-client party of frame.parti (its name -- the
    client never by a word he shares with another party, a tie read as the other party -- or a content word of its
    ruolo_nel_fatto) or an authority (data «authority_words»: accusa, Procura, PM, investigatori), whatever the
    client says in the same sentence. The client's own statement (every asserting phrase naming the client, his side
    -- cliente, assistito, difesa -- or his procedural role: interactive_flow.client_roles) is never one. Only such sentences -> «contestato» with the party
    (analysis_provato_downgraded); no valid S-id -> «da_provare» (analysis_provato_unsupported);
  - esito «escluso» needs an S-id (else «da_provare», analysis_escluso_unsupported); «contestato» likewise
    (analysis_contestato_unsupported); an esito outside the four -> «da_provare» (analysis_esito_invalid); an element
    of a line's article the reply leaves out -> «da_provare» (analysis_element_missing): every schema element of a line's
    article gets one verdict, in the schema's order;
  - a line (linea_principale, linea_subordinata) is rejected (analysis_line_rejected, the reason re-asked) when its
    articolo is not a fitted norm, is in the remedy / general-part set (data/remedy_articles.json; 2043 c.c. only when
    the verified frame is contractual -- is_contractual; the frame's own verified addebito/domanda article never), its
    elemento_in_gioco is no E-id of it (or, for a claimant, an exception / limit element), its tesi is empty, it cites
    no id, or an id it cites does not exist. After the re-ask: a rejected or missing leading line is the rubrica
    fallback's (v79.15.1), else «da determinare» (LINE_UNDETERMINED, analysis_line_undetermined) -- the old engine's
    line is never its substitute; a rejected subordinate line is none;
  - the frame's verified addebito/domanda article (case_frame.charge_label) must be the leading or the subordinate
    line's article: else, after the re-ask (none when the article is not among the fitted norms -- the model cannot
    cite it), analysis_frame_charge_missing = 1 and the frame's article is the leading line (stato
    «addebito_del_quadro», no tesi: nothing generated) with every schema element «da_provare»; the model's verified
    leading line (else its subordinate one) becomes the subordinate line;
  - affermazioni_di_diritto: each kept only with at least one A-id, every A-id existing, every article its text names
    a fitted norm with one of its own periods cited (law_statements_off_article), and its tokens grounded in the cited
    periods (data «law_statements.overlap», law_statements_ungrounded) (law_statements_kept); else dropped, never
    printed (law_statements_dropped).
Nothing unverified is in the result; no text is generated by the code beyond the labels.

  run(state, call=None, inp=None, started=None)   the step: {"status", "error", "analysis", "counters", "prompt_hash",
                                                  "seconds", "calls", "reasons", "rejected"}; logs «analysis_step»
  analyse(inp, call=None, started=None)           the same on a ready input (tests; build B)
  make_input(frame, norms, sentences, questions, client, client_roles)   the input
  verify(reply, inp)                              one reply checked (pure)
  finish(checked, inp)                            the verified analysis, the fallbacks applied

Counters: analysis_calls (<= 2), analysis_reasked, analysis_failed (the first call failed, timed out or answered no
JSON), analysis_timeouts, analysis_line_rejected, analysis_line_undetermined, analysis_frame_charge_missing,
analysis_bad_sid, analysis_bad_aid, analysis_bad_eid, analysis_element_missing, analysis_esito_invalid,
analysis_provato_downgraded, analysis_provato_unsupported, analysis_escluso_unsupported,
analysis_contestato_unsupported, law_statements_kept, law_statements_dropped, law_statements_off_article,
law_statements_ungrounded, analysis_seconds; v79.15.1: analysis_line_exception, analysis_line_fallback_rubrica,
objections_from_allegations, objections_from_exceptions, analysis_line_kept_first.

v79.15.1 (brief items 1-3):
  - elements by id: each fitted norm's schema elements go to the model numbered E1..En per article, written
    «A<token>:E<n>» (the token of the A-ids: applicability.article_token -- «A1385cc:E2»); elemento_in_gioco and
    elementi[].elemento are such ids (a bare «E<n>» read on the article the line or the item names). Verified by
    existence only (element_of): an id that is no element of that article voids the claim -- the line rejected, the
    verdict not read -- counted analysis_bad_eid (replaces analysis_element_not_verbatim; the verbatim text check is
    gone). The element's text printed is always the schema's (the input's), never the model's.
  - the re-ask lists the articles a line may stand on -- the fitted norms not in the remedy set (the frame's own
    charge kept, as make_input marks them) -- as «A<token>» with their E-ids (for a claimant, only the constitutive
    ones) (allowed_text).
  - after the re-ask (merge_replies), each line kind the second reply's verified line, else the first reply's
    (analysis_line_kept_first: a line verified on attempt 1, the re-ask asked for a defect elsewhere and its own line
    of that kind rejected); the element verdicts read from the second reply first, then the first's;
  - then the fallbacks in this order: (1) the frame's verified charge article missing from the lines ->
    it leads (79.15.0's analysis_frame_charge_missing: a verified article beats a name); (2) else no verified leading
    line -> the frame's verified named charges / claims (addebito_o_domanda.descrizione, istituto.testo, then the older
    frame's central_issue_candidates whose every content word is in the case's sentences: inputs) matched to the fitted
    norms' rubriche exactly (fold: accents and case, whitespace collapsed, a final . ; : out -- no fuzzy match), the
    remedy set left out; the first name that matches, its first norm -> the leading line (stato FALLBACK_LINE «rubrica
    del quadro», every element «da_provare», tesi FALLBACK_TESI: «<rubrica>: elementi da provare dall'accusa» penal,
    «<rubrica>: elementi da provare» otherwise; analysis_line_fallback_rubrica = 1); (3) else «da determinare».
  - claimant lines: the verified frame says the client brings the claim (claimant(): ruolo valore/intake in data
    «claimant.roles»; with no such role verified and no defendant role, the client party's ruolo_nel_fatto holding a
    word of «claimant.party_words») -> a line's elemento_in_gioco must be a constitutive element: an element the
    schema's own kind flags (data «exception.schema_kinds» / «exception.special_kinds» on the record's schema_kind /
    special) or, the element carrying no schema kind, the keyword list («exception.keywords», on the element's text and
    on the words before it in its clause of the article) flags as an exception or a limit is refused
    (analysis_line_exception), re-asked in the one re-ask with the constitutive E-ids listed; the element's
    «eccezione_fonte» says which source decided («schema» | «parole_chiave»).
  - objections (finish -> analysis «obiezioni»): one per non-client allegation (a cited sentence reported_by another
    party) behind a «contestato» element of a line article, its elements, the client's own cited sentences and the
    verified status (objections_from_allegations); for a claimant, every exception element the model raised -- put as
    elemento_in_gioco, or given a verdict other than «da_provare» -- not already in an allegation's objection
    (objections_from_exceptions). Rendered by objection_items: labels and the case's sentences only.

v79.15.1 (brief item 4, build B): the prose replaced, not deleted -- ProseReview (the rendering section's end) reviews
the Q&A, §4's written prose and §7's model-written objections and answers sentence by sentence, before the answer-wide
checks (lawyer_method.compose / answers_block): a sentence citing an article the binding excluded or the fit removed
-> dropped (prose_excluded_article_dropped); one citing an article outside the fitted set (prose_scope: the input's
norms and the remedies §4 names) the user did not name -> dropped (prose_unfitted_article_dropped); a statement of what
a norm provides with no verified counterpart -> the first verified affermazione_di_diritto on its article, verbatim
(qa_sentences_replaced), else dropped (qa_sentences_dropped / norms_sentences_dropped / objections_sentences_dropped);
after any drop -- the review's or the checks' (orphans) -- the next sentence opening with an anaphoric connective
(data «prose_review.anaphoric_openers») goes too (qa_sentences_dropped_dependent).

v79.15.2 (hotfix): one canonical article key (art_parts / art_key: number with its suffix + code, lower-case, no space or
punctuation -- «A640cp», «640cp», «640 c.p.», «art. 640 c.p.», «Art. 640 cp» -> «640cp»; «640 c.p.p.» -> «640cpp») for
every article comparison here: same_article, resolve (the exact id token, then the key, then a bare number), the E-id
prefix (parse_eid, element_of), the remedy set (is_remedy), the contractual frame, the frame's charge, the prose
review's fitted / named / excluded sets, the bare-number citations and the Q&A counterpart. The logged and printed
articolo stays the fitted norm's label. analysis_fallback_skipped_reason (a string counter, SKIP_*: line_verified,
frame_charge_used, no_named_charge, no_rubrica_match, step_failed, no_input; "" when the rubrica fallback ran) in the
counters, the analysis («fallback_skipped_reason») and the «analysis_step» line.

v79.16.2 (hotfix): two 79.16.1 rules back to log-only -- pertinence never downgrades a verdict (counted and logged,
analysis_provato_no_pertinence, «no_pertinence»); the fit tier is a preference told to the model in the article block
(«preferenza per la linea: n/m») and in the re-ask's allowed list, never a rejection (_verify_line refuses a fit «no» and a
procedural element only; an unverified or unjudged chosen line: line_from_unverified = 1); a failed, timed-out or
malformed first call runs finish on no reply (empty_checked), so the frame's charge, the rubrica fallback and «da
determinare» apply in that order on every path, the reason logged (step_failed only for an input that cannot be read).

v79.15.3 (hotfix, three rendering fixes; no prompt change, no new logic in the model call). The brief reverses «no
text is generated by the code beyond the labels» for the tesi and §7:
  1. code_tesi (in finish): a verified line's printed tesi built from the verified esito of its elemento_in_gioco and
     the client's side (data «rendering.tesi»); the model's tesi kept as «tesi_modello», logged only (the
     «analysis_step» line's tesi_modello=). §1, §6, §9 print the code's sentence (line_text, solutions_text).
  2. client_side: the client's side (attore | difesa | neutro) from the verified frame only -- the brief's sets (data
     «claimant») on the verified ruolo, then on the client party's verified ruolo_nel_fatto; «claimant» is its attore.
  3. objection_items: §7's entries in one plain form (data «rendering.obiezioni»); asserted_unproven: a §7 answer
     sentence asserting an element integrato / provato that the verified table holds da_provare / contestato dropped
     by ProseReview (objections_sentences_dropped).
  (review) asserted_unproven reads clause by clause: a negation counts only before the verb in its clause (explicit
  word forms, so «restituzione» / «contesto» negate nothing), emphatic phrases («senza alcun dubbio») are no negation,
  a period reporting another party's claim («La Procura afferma che…», «Per l'accusa…») asserts nothing; code_tesi
  never quotes a counterparty's allegation as the {fatti} of any esito; the side sets carry the plural forms.
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait as _wait
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .feature_flags import enabled as _flag_enabled

logger = logging.getLogger(__name__)

FLAG = "ANALYSIS_STEP"
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DATA_FILE = os.path.join(_DATA, "analysis_step.json")
REMEDY_FILE = os.path.join(_DATA, "remedy_articles.json")
COUNTERS = ("analysis_calls", "analysis_reasked", "analysis_failed", "analysis_timeouts", "analysis_line_rejected",
            "analysis_line_undetermined", "analysis_frame_charge_missing", "analysis_bad_sid", "analysis_bad_aid",
            "analysis_bad_eid", "analysis_element_missing", "analysis_esito_invalid",
            "analysis_provato_downgraded", "analysis_provato_unsupported", "analysis_escluso_unsupported",
            "analysis_contestato_unsupported", "law_statements_kept", "law_statements_dropped",
            "law_statements_off_article", "law_statements_ungrounded", "analysis_seconds",
            # v79.15.1
            "analysis_line_exception", "analysis_line_fallback_rubrica", "objections_from_allegations",
            "objections_from_exceptions", "analysis_line_kept_first",
            # v79.15.2 (item 2): a string, the reason code (SKIP_*), "" when the fallback ran
            "analysis_fallback_skipped_reason",
            # v79.16.0 (A2): a «provato» decided by the record's stato / chi, or by the text heuristic (no record);
            # the measurement line_table_weaker: the subordinate line holds more provato elements than the leading one
            "analysis_provenance_from_record", "analysis_provenance_fallback", "line_table_weaker",
            # v79.16.1: the line's fit tier (item 3), a procedural element refused as the element in play (item 3),
            # a «provato» / «escluso» with no pertinent cited fact (item 4)
            "line_from_fit_si", "line_from_fit_dipende", "line_from_unverified", "analysis_line_procedural_rejected",
            "analysis_provato_no_pertinence",
            "analysis_line_context_rejected", "analysis_line_quesito_only",        # v79.17.0 (item 2)
            # v79.17.1 (item 7): the fitted articles with no schema element -- context only, never a line candidate,
            # not in the prompt's article block
            "candidates_no_elements_excluded",
            # v79.17.1 (item 9): a line on an article whose code is outside the frame's rito (1439 c.c. in a penal case)
            # refused -- as the leading line always, as the subordinate unless the fit tagged it (outside_rito)
            "analysis_line_outside_rito_rejected",
            # v79.17.2 (item 2): a leading line outside the governs-pretesa set (a subordinate outside the set and the
            # quesiti's) refused; the binding enforced with an empty set: the honesty line, no fallback pick
            "analysis_line_outside_set_rejected", "analysis_binding_empty_honesty")
PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO = "provato", "contestato", "da_provare", "escluso"
ESITI_ALL = (PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO)
OK, FAILED, MALFORMED, TIMEOUT, NO_INPUT, OFF = "ok", "failed", "malformed", "timeout", "no_input", "off"
LINE_KINDS = ("linea_principale", "linea_subordinata")
VERIFIED, FRAME_LINE, UNDETERMINED = "verificata", "addebito_del_quadro", "da_determinare"     # a line's «stato»
FALLBACK_LINE = "rubrica_del_quadro"        # v79.15.1 (item 2): the line found by the frame's named charge's rubrica
# my drafts (v79.15.0): the labels build B prints -- the leading line the step could not verify (§1), never a text
LINE_UNDETERMINED = "da determinare"
LINE_UNVERIFIED_LABEL = "linea non verificata"
# v79.15.1 (item 2, the brief's wording): the rubrica fallback's tesi -- penal / every other rito
FALLBACK_TESI = {"penale": "{rubrica}: elementi da provare dall'accusa", "civile": "{rubrica}: elementi da provare"}
SCHEMA, KEYWORDS = "schema", "parole_chiave"        # which source flagged an element as an exception or a limit
# v79.15.2 (item 2): why the rubrica fallback did not run -- a string counter (block_counters, the analysis_step line):
# "" when it ran (analysis_line_fallback_rubrica = 1)
SKIPPED_REASON = "analysis_fallback_skipped_reason"
SKIP_LINE_VERIFIED = "line_verified"            # a verified leading line (the re-ask's, or the first reply's kept)
SKIP_FRAME_CHARGE = "frame_charge_used"         # the frame's verified charge article missing from the lines: it leads
SKIP_NO_NAMED_CHARGE = "no_named_charge"        # no verified leading line, and the frame names no charge or claim
SKIP_NO_RUBRICA_MATCH = "no_rubrica_match"      # a named charge / claim, but no fitted rubrica equal to it
SKIP_STEP_FAILED = "step_failed"                # the input unread (v79.16.2: a failed call runs the fallbacks instead)
SKIP_NO_INPUT = "no_input"                      # no fitted norm or no sentence: nothing called
SKIP_BINDING_EMPTY = "binding_empty"            # v79.17.2 (item 1): the set is empty -- the honesty line, no pick
HONESTY_LINE = "Il corpus non copre la pretesa: nessun articolo applicabile la governa."    # v79.17.2 (item 1), my draft
STRING_COUNTERS = (SKIPPED_REASON,)


def enabled() -> bool:
    return _flag_enabled(FLAG)


@lru_cache(maxsize=1)
def config() -> Dict[str, Any]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def remedy_config() -> Dict[str, Any]:
    with open(REMEDY_FILE, encoding="utf-8") as f:
        return json.load(f)


def prompt_system() -> str:
    return "\n".join(config()["prompt_system"])


def prompt_hash() -> str:
    return hashlib.sha256(prompt_system().encode("utf-8")).hexdigest()[:12]


def _log(event: str, data: Dict[str, Any]) -> None:
    try:
        from .verbose_logger import vlog
        vlog(event, data)
    except Exception:           # noqa: BLE001 -- the log never fails the step
        pass


def _ap():
    from . import applicability
    return applicability


def _es():
    from . import element_schemas
    return element_schemas


def _cf():
    from . import case_frame
    return case_frame


def _flat(value: Any) -> str:
    return " ".join(str(value or "").split())


def empty_counters() -> Dict[str, Any]:
    return {k: ("" if k in STRING_COUNTERS else 0) for k in COUNTERS}


# --- articles: one canonical key, the remedy set, the contractual frame ----------------------------------------------
# v79.15.2 (hotfix, item 1): every article comparison of this module goes through ONE key (art_key) -- the reply's
# «articolo», an E-id's prefix, the remedy set, the fitted set, the frame's charge, the user-named articles, the re-ask's
# allowed list, the rubrica fallback, the objections and the law statements. 79.15.1 compared the reply's «articolo»
# with the display label («640 c.p.») by element_schemas.norm_label / applicability.article_key, which read the A-id
# form the prompt itself sends («A640cp», «640cp») as the number «640cp» with no code: the line was rejected
# («articolo non tra le norme applicabili: 640cp»). The displayed and logged «articolo» stays the label after resolve.

def _suffix_alt() -> str:
    from .comma_material import SUFFIXES
    return "|".join(sorted(SUFFIXES, key=len, reverse=True))


@lru_cache(maxsize=1)
def _code_names() -> Dict[str, str]:
    """data «article_key.codes»: a code's compact spelling -> its key («codicepenale» -> «cp»)."""
    return {str(k): str(v) for k, v in ((config().get("article_key") or {}).get("codes") or {}).items()}


@lru_cache(maxsize=1)
def _law_prefixes() -> List[Tuple[str, str]]:
    """data «article_key.law_prefixes»: a law's spelled-out kind before its number -> its sigla («legge» -> «l»),
    longest first."""
    items = ((config().get("article_key") or {}).get("law_prefixes") or {}).items()
    return sorted(((str(k), str(v)) for k, v in items), key=lambda kv: -len(kv[0]))


def art_parts(raw: Any) -> Tuple[str, str]:
    """v79.15.2: (number, code) of an article however written -- the number with its suffix, no space or dash
    («62bis»), the code compact, lower-case, no space or punctuation («cp», «cpp», «cc», «cpc», «l2411990»; a code's
    document name by its sigla: «Codice Penale 2026» -> «cp»). Accepted: «640 c.p.», «art. 640 c.p.», «Art. 640 cp»,
    «articolo 640 c.p. (Truffa)», «640cp», «A640cp» (the A-id's article), «A640cp:E1» (an E-id's article), «62-bis
    c.p.», «62 bis c.p.», «A62biscp»; a comma's or a numero's label read as its article («52 c.p. co. 2», «A52cpco2»,
    «62 n. 2 c.p.»). Review: a period id («A640cp.1») read as its article; a preposition before the code («640 del
    codice penale») skipped; a law's own number kept («1 L. n. 241/1990» -> «l2411990», never «l1990»: only a numero
    of the article -- «n. 2» not followed by «/» or a digit -- is dropped; the «n» of «L. n. 241» dropped in both
    the label and its id token, «ln2411990» -> «l2411990»). ("", "") when it names no article number; a bare number
    -> (number, "")."""
    s = _id_strip(raw)
    m = re.match(rf"^(\d+)\s*(?:-\s*)?((?:{_suffix_alt()}))?(.*)$", s)
    if not m:
        return "", ""
    number = m.group(1) + (m.group(2) or "")
    rest = re.sub(r"(?:^|\s)(?:co\.?|comma)\s*\d+\w*", " ", m.group(3) or "")
    rest = re.sub(r"(?:^|\s)n\.\s*\d+(?![\d/])", " ", rest)     # a numero («62 n. 2 c.p.»), not «L. n. 241/1990»
    rest = re.sub(r"^\s*(?:del|dello|della|dei|di)\s+", "", rest)  # «640 del codice penale»
    try:
        named = _es()._code(" ".join(rest.split())) if rest.strip() else ""
    except Exception:           # noqa: BLE001 -- the spelling as written
        named = rest
    code = re.sub(r"[^a-z0-9]", "", fold(named))
    code = re.sub(r"^n\d+(?=[a-z])", "", code)                  # «62n2cp»: a numero's token
    code = re.sub(r"(?<=[a-z])co\d+[a-z]*$", "", code)          # «52cpco2»: a comma's token
    code = re.sub(r"(?<=[a-z])n(?=\d)", "", code)               # «ln2411990» -> «l2411990» (a law's «n.»)
    for long, short in _law_prefixes():                         # «legge2411990» -> «l2411990»
        if code.startswith(long) and code[len(long):len(long) + 1].isdigit():
            code = short + code[len(long):]
            break
    return number, _code_names().get(code, code)


def _id_strip(raw: Any) -> str:
    """The article's writing without its wrapping: folded, a rubrica in brackets, «art.»/«articolo», an E-id's «:E<n>»,
    the ids' leading «A», and (review) a period id's «.<n>» («A640cp.1» -> «640cp») removed."""
    s = fold(raw)
    s = re.sub(r"\([^)]*\)", " ", s).strip(" \t[]«»\"'.,;:")
    s = re.sub(r"^art(?:icol[oi]|t)?\.?\s*", "", s)
    s = re.sub(r"\s*:\s*e\d+$", "", s)                         # an E-id: its article
    s = re.sub(r"^a(?=\d)", "", s)                              # the ids' «A» before the article
    if re.fullmatch(r"\d+[a-z0-9]*\.\d+", s):                   # a period id: its article
        s = s.rsplit(".", 1)[0]
    return s


def art_key(raw: Any) -> str:
    """v79.15.2: the one canonical key of an article: number (with its suffix) + code, no space or punctuation,
    lower-case -- «640 c.p.», «art. 640 c.p.», «Art. 640 cp», «640cp», «A640cp» -> «640cp»; «62-bis c.p.» -> «62biscp»;
    «1667 c.c.» -> «1667cc»; «640 c.p.p.» -> «640cpp» (never «640cp»). "" when it names no article number."""
    num, code = art_parts(raw)
    return f"{num}{code}" if num else ""


def _id_token(raw: Any) -> str:
    """The raw article as its ids spell it (applicability.article_token of the label: «640 c.p.» -> «640cp»; «A640cp»
    -> «640cp»; «art. 52 c.p. co. 2» -> «52cpco2») -- resolve's exact step, before the key."""
    return _ap().article_token(_id_strip(raw))


def same_article(a: Any, b: Any) -> bool:
    """Two writings of one article: the same art_key (v79.15.2: «A640cp», «640cp», «640 c.p.», «art. 640 c.p.» are one
    article; «640 c.p.p.» another)."""
    ka = art_key(a)
    return bool(ka) and ka == art_key(b)


def resolve(raw: Any, labels: Sequence[str]) -> Optional[str]:
    """The fitted norm's label an «articolo» of the reply names, in this order: the label whose id token is the raw's
    exactly (a comma's own label, «52 c.p. co. 2», before its article's), the label with the same art_key, a bare number
    when one norm has it; None when it names none. The label -- the display form -- is what is kept and logged."""
    key = art_key(raw)
    if not key:
        return None
    token = _id_token(raw)
    for label in labels:
        if token and _ap().article_token(label) == token:
            return label
    for label in labels:
        if art_key(label) == key:
            return label
    num, code = art_parts(raw)
    if not code:
        hits = [x for x in labels if art_parts(x)[0] == num]
        if len(hits) == 1:
            return hits[0]
    return None


def _number_key(number: Any) -> Optional[Tuple[int, int]]:
    """(the number, its suffix's rank: none 0, bis 1, ter 2 …) of «62-bis», «62 bis», «62bis»; None if not one."""
    from .comma_material import SUFFIXES
    m = re.match(r"^\s*(\d+)\s*(?:-\s*|\s+)?([a-z]+)?\s*$", str(number or "").lower())
    if not m:
        return None
    suffix = m.group(2) or ""
    if suffix and suffix not in SUFFIXES:
        return None
    return int(m.group(1)), (SUFFIXES.index(suffix) + 1 if suffix else 0)


def _in_list(number: str, entries: Sequence[Any]) -> bool:
    key = _number_key(number)
    if key is None:
        return False
    for e in entries or []:
        if isinstance(e, dict):
            lo, hi = _number_key(e.get("from")), _number_key(e.get("to"))
            if lo is not None and hi is not None and lo <= key <= hi:
                return True
        elif _number_key(e) == key:
            return True
    return False


def is_contractual(frame: Optional[Dict[str, Any]]) -> bool:
    """The verified frame says the relationship is contractual: its rito not penal (case_frame.PENAL_RITI, the frame's
    value or its intake value) and either its addebito_o_domanda article a c.c. article inside the data's «cc_ranges»
    (obligations and contracts, 1173-2042) or a word of its addebito_o_domanda (descrizione, frase) or istituto (testo)
    beginning with one of the data's «stems» (contratt-, inadempi-, …). No frame, or a penal one: not contractual."""
    f = frame if isinstance(frame, dict) else {}
    if not f:
        return False
    rito = f.get("rito") or {}
    if rito.get("valore") in _cf().PENAL_RITI or rito.get("intake") in _cf().PENAL_RITI:
        return False
    cfg = remedy_config().get("contractual") or {}
    charge = f.get("addebito_o_domanda") or {}
    if charge.get("articolo"):
        num, code = art_parts(charge["articolo"])            # v79.15.2: the one key
        if code == "cc" and _in_list(num, cfg.get("cc_ranges") or []):
            return True
    words = " ".join(_flat(x) for x in (charge.get("descrizione"), charge.get("frase"),
                                       (f.get("istituto") or {}).get("testo"))).lower()
    return any(re.search(rf"\b{re.escape(stem)}", words) for stem in cfg.get("stems") or [])


def is_remedy(label: Any, contractual: bool = False) -> bool:
    """The article is in the remedy / general-part set (data/remedy_articles.json «articles»; «when_contractual» only
    when the frame is contractual). v79.15.2: the article read by art_parts -- «A41cp», «1453cc» are in it too."""
    num, code = art_parts(label)
    cfg = remedy_config()
    if _in_list(num, (cfg.get("articles") or {}).get(code) or []):
        return True
    return contractual and _in_list(num, (cfg.get("when_contractual") or {}).get(code) or [])


# --- the input --------------------------------------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "«": '"', "»": '"', "„": '"'})


def verbatim_key(text: Any) -> str:
    """An element's text as compared verbatim: whitespace collapsed, quote marks normalised (’ ‘ ` -> ', “ ” « » „ ->
    ") and one enclosing pair of " out; nothing else changed (case, accents, punctuation as written)."""
    s = _flat(str(text or "").translate(_QUOTE_MAP))
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def fold(text: Any) -> str:
    """v79.15.1: a text compared without accents and case, whitespace collapsed (the rubrica match, the keywords)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", _flat(str(text or "").replace("’", "'")).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def rubrica_key(text: Any) -> str:
    """v79.15.1 (item 2): a rubrica or a named charge as the exact match reads it -- fold(), a final . ; : out."""
    return fold(text).strip(" .;:")


def element_id(label: Any, n: int) -> str:
    """v79.15.1 (item 1): the n-th schema element of the article as the model cites it: «A<token>:E<n>» (the A-ids'
    token, applicability.article_token: «A1385cc:E2»)."""
    return f"A{_ap().article_token(label)}:E{n}"


_EID_RE = re.compile(r"^(?:(.+?)\s*:\s*)?E(\d+)$", re.IGNORECASE)


def _eid_split(value: Any) -> Optional[Tuple[str, int]]:
    """(the article prefix as written, "" for a bare «E<n>»; n) of an E-id as a reply writes it (quotes, brackets and
    spaces around it out); None when it is not one -- v79.15.2: the prefix any writing of an article (art_parts:
    «A640cp», «640cp», «640 c.p.», «art. 640 c.p.»), never a text."""
    if not isinstance(value, (str, int)):
        return None
    s = str(value).strip().strip("[]()«»\"'.,;")
    m = _EID_RE.match(s)
    if not m:
        return None
    prefix = (m.group(1) or "").strip()
    if prefix and not art_key(prefix):
        return None
    return prefix, int(m.group(2))


def parse_eid(value: Any) -> Optional[Tuple[str, int]]:
    """(the article's art_key or "" for a bare «E<n>», n) of an E-id as a reply writes it; None when it is not one.
    «A640cp:E2», «640cp:E2», «640 c.p.:E2» -> ("640cp", 2)."""
    got = _eid_split(value)
    return None if got is None else (art_key(got[0]) if got[0] else "", got[1])


@lru_cache(maxsize=1)
def _exception_re() -> "re.Pattern[str]":
    words = "|".join(f"(?:{fold(x)})" for x in (config().get("exception") or {}).get("keywords") or []) or r"(?!x)x"
    return re.compile(rf"(?<![a-z0-9])(?:{words})", re.IGNORECASE)


def _clause_lead(element: str, periods: Sequence[Dict[str, str]]) -> str:
    """The words before the element in its clause of the article: the first period holding the element's text (folded;
    else its first four words), from the period's start or its last ; or : before the element, to the element."""
    key = fold(element)
    probes = [key] + ([" ".join(key.split()[:4])] if len(key.split()) > 4 else [])
    for probe in probes:
        for p in periods or []:
            text = fold(p.get("text"))
            at = text.find(probe)
            if at >= 0:
                head = text[:at]
                cut = max(head.rfind(";"), head.rfind(":"))
                return head[cut + 1:] if cut >= 0 else head
    return ""


def exception_of(element: Dict[str, Any], periods: Sequence[Dict[str, str]] = ()) -> Tuple[bool, str]:
    """v79.15.1 (item 3): (the element is an exception or a limit, the source that decided). The schema's own kind when
    the element carries one (schema_kind / special: data «exception.schema_kinds», «exception.special_kinds») ->
    SCHEMA; else the keyword list (data «exception.keywords»: eccezione, salvo, a meno che, non ha diritto, non è
    dovuto, se non prova, decadenza, denunzia …, accent- and case-folded, from a word's start) on the element's text
    and on the words before it in its clause of the article (_clause_lead) -> KEYWORDS."""
    cfg = config().get("exception") or {}
    kind, special = _flat(element.get("schema_kind")).lower(), _flat(element.get("special")).lower()
    if kind or special:
        hit = kind in [str(x).lower() for x in cfg.get("schema_kinds") or []] or \
            special in [str(x).lower() for x in cfg.get("special_kinds") or []]
        return hit, SCHEMA
    text = element.get("testo") or ""
    hit = bool(_exception_re().search(fold(text)) or _exception_re().search(_clause_lead(text, periods)))
    return hit, KEYWORDS


SIDE_CLAIMANT, SIDE_DEFENCE, SIDE_NEUTRAL = "attore", "difesa", "neutro"


def _role_hits(text: Any, entries: Sequence[Any]) -> List[str]:
    """The entries (whole words or phrases, accent- and case-folded) the text holds; a «/» or «_» read as a space."""
    def words(s: Any) -> str:
        return " ".join(re.sub(r"[^a-z']+", " ", fold(str(s or "").replace("/", " ").replace("_", " "))).split())
    low = f" {words(text)} "
    return [str(x) for x in entries or [] if words(x) and f" {words(x)} " in low]


def client_side(frame: Optional[Dict[str, Any]], client: str = "") -> Tuple[str, str]:
    """v79.15.3 (item 2): (the client's side, why) from the VERIFIED frame only -- SIDE_CLAIMANT, SIDE_DEFENCE or
    SIDE_NEUTRAL (data «claimant», whole words, accent- and case-folded):
      1. the verified ruolo (valore and intake: «imputato/indagato» read as both words): a word of «claimant.roles»
         -> claimant, of «claimant.defendant_roles» -> defence (the procedural role decides first: a committente
         «convenuto» is the defence);
      2. else the client's party in frame.parti (lato «cliente», or the client's name) -- its verified ruolo_nel_fatto
         (case_frame voids an ungrounded one): only claimant words -> claimant, only defence words -> defence;
         «claimant.worker» words (lavoratore) are a claimant's only when the claim is the worker's: the verified rito
         one of «worker.riti» (lavoro) and the frame verifying an addebito_o_domanda (articolo or descrizione);
      3. else (no verified role, no side word, or both sides' words) -> neutral («why» empty).
    79.15.1's claimant() read «roles» (attore, ricorrente) on the ruolo only and «party_words» on the party only
    after no defendant ruolo; the brief's sets now read on both."""
    f = frame if isinstance(frame, dict) else {}
    cfg = config().get("claimant") or {}
    pro, contra = cfg.get("roles") or [], cfg.get("defendant_roles") or []
    role = f.get("ruolo") if isinstance(f.get("ruolo"), dict) else {}
    said = " ".join(str(role.get(k) or "") for k in ("valore", "intake"))
    hit = _role_hits(said, contra)
    if hit:
        return SIDE_DEFENCE, f"ruolo: {hit[0]}"
    hit = _role_hits(said, pro)
    if hit:
        return SIDE_CLAIMANT, f"ruolo: {hit[0]}"
    worker = cfg.get("worker") or {}
    rito = f.get("rito") if isinstance(f.get("rito"), dict) else {}
    claim = f.get("addebito_o_domanda") if isinstance(f.get("addebito_o_domanda"), dict) else {}
    theirs = str(rito.get("valore") or "") in [str(x) for x in worker.get("riti") or []] and (
        not worker.get("needs_domanda") or bool(_flat(claim.get("articolo")) or _flat(claim.get("descrizione"))))
    ups: List[str] = []
    downs: List[str] = []
    for p in f.get("parti") or []:
        if not isinstance(p, dict):
            continue
        if not (p.get("lato") == "cliente" or (client and _cf().same_party(client, p.get("nome")))):
            continue
        said = p.get("ruolo_nel_fatto")
        ups += _role_hits(said, pro) + (_role_hits(said, worker.get("words") or []) if theirs else [])
        downs += _role_hits(said, contra)
    if ups and not downs:
        return SIDE_CLAIMANT, f"ruolo_nel_fatto: {ups[0]}"
    if downs and not ups:
        return SIDE_DEFENCE, f"ruolo_nel_fatto: {downs[0]}"
    return SIDE_NEUTRAL, ""


def claimant(frame: Optional[Dict[str, Any]], client: str = "") -> bool:
    """v79.15.1 (item 3): the verified frame says the client brings the claim. v79.15.3 (item 2): client_side's
    claimant (the brief's sets on the verified ruolo, then on the client party's verified ruolo_nel_fatto); a
    defence or a neutral side -> False (an unverified role: neither, as before)."""
    return client_side(frame, client)[0] == SIDE_CLAIMANT


def penal_frame(frame: Optional[Dict[str, Any]], label: Any = "") -> bool:
    """The verified rito is penal (case_frame.PENAL_RITI, valore or intake); no rito verified: the article a penal
    provision (case_frame.is_penal)."""
    rito = (frame or {}).get("rito") or {}
    if rito.get("valore") or rito.get("intake"):
        return rito.get("valore") in _cf().PENAL_RITI or rito.get("intake") in _cf().PENAL_RITI
    try:
        return bool(label) and _cf().is_penal(_cf().parse_article(label))
    except Exception:           # noqa: BLE001
        return False


def named_issues(frame: Optional[Dict[str, Any]], older: Sequence[str] = ()) -> List[str]:
    """v79.15.1 (item 2): the frame's verified named charges / claims, in order: addebito_o_domanda.descrizione,
    istituto.testo (both verified by the 79.14 frame: words in the text), then `older` (the older frame's
    central_issue_candidates, already checked by inputs); each once."""
    f = frame if isinstance(frame, dict) else {}
    out: List[str] = []
    for x in [(f.get("addebito_o_domanda") or {}).get("descrizione"), (f.get("istituto") or {}).get("testo"),
              *list(older or [])]:
        if _flat(x) and rubrica_key(x) not in [rubrica_key(y) for y in out]:
            out.append(_flat(x))
    return out


def make_input(frame: Optional[Dict[str, Any]], norms: Sequence[Dict[str, Any]], sentences: Sequence[Dict[str, Any]],
               questions: Sequence[str] = (), client: str = "", client_roles: Sequence[str] = (),
               older_issues: Sequence[str] = (), binding: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The step's input. frame: the verified frame's fields (only verified ones; {} none); norms: [{article, text,
    rubrica, elements: [{id, testo, schema_kind?, special?}]}] -- the fitted norms; sentences: [{id, text, certainty,
    source, alleged_by?}] numbered as the flow numbers them (a «sistema» one left out, the numbers kept); questions: the
    user's; client: the client's name (the frame's, else the flow's); client_roles: his procedural role words;
    older_issues: the older frame's central_issue_candidates' names the caller checked (v79.15.1, the rubrica
    fallback). v79.15.1: each element gets its E-id («eid», element_id), «eccezione» and «eccezione_fonte»
    (exception_of); the input carries «claimant» and «issues» (named_issues). v79.15.3: «side» / «side_why»
    (client_side: attore | difesa | neutro, from the verified frame only); «claimant» is side == attore. v79.17.2
    (item 1): binding -- the flow's binding on the pretesa (interactive_flow.binding_pretesa: enforced, principale);
    carried as «binding» {"enforced", "pretesa": its articles}; None: the sets are read from the norms' governs."""
    f = dict(frame or {})
    contractual = is_contractual(f)
    charge = _cf().charge_label({"frame": f}) if (f.get("addebito_o_domanda") or {}).get("articolo") else ""
    out_norms = []
    for n in norms or []:
        label = str(n.get("article") or "").strip()
        if not label:
            continue
        periods = _ap().article_ids(label, n.get("text"))
        elements = []
        for e in n.get("elements") or []:
            if not isinstance(e, dict) or not _flat(e.get("testo") or e.get("text")):
                continue
            el = {"id": str(e.get("id") or ""), "testo": _flat(e.get("testo") or e.get("text")),
                  "eid": element_id(label, len(elements) + 1)}
            for k in ("schema_kind", "special"):
                if _flat(e.get(k)):
                    el[k] = _flat(e.get(k))
            el["eccezione"], el["eccezione_fonte"] = exception_of(el, periods)
            elements.append(el)
        out_norms.append({"article": label, "rubrica": _flat(n.get("rubrica")), "text": str(n.get("text") or ""),
                          "elements": elements, "periods": periods,
                          "remedy": is_remedy(label, contractual) and not (charge and same_article(label, charge)),
                          # v79.16.1 (item 3): the fit's verdict (si | no | dipende | None: no judgment), verified or not
                          "fit": _canon_fit_value(n.get("fit")), "fit_verified": bool(n.get("fit_verified")),
                          "fit_perche": _flat(n.get("fit_perche")),
                          # v79.17.0 (item 2): what the verified verdict says the norm decides -- a list («pretesa»,
                          # «Q<n>»; [] context only) when the fit was asked, None when it was not
                          "governs": list(n["governs"]) if isinstance(n.get("governs"), (list, tuple)) else None})
    sents = [{"id": str(s.get("id")), "text": _flat(s.get("text")), "certainty": s.get("certainty"),
              "source": s.get("source"), "alleged_by": str(s.get("alleged_by") or ""),
              "fact": str(s.get("fact") or ""), "said_by": str(s.get("said_by") or "")}   # v79.16.0 (A2): the record's
             for s in sentences or [] if isinstance(s, dict) and s.get("id") and _flat(s.get("text"))
             and s.get("source") != "sistema"]
    client = _flat(client or (f.get("cliente") or {}).get("nome"))
    side, side_why = client_side(f, client)         # v79.15.3 (item 2): the verified side; «claimant» its claimant
    return {"frame": f, "norms": out_norms, "sentences": sents, "questions": [_flat(q) for q in questions or [] if
                                                                              _flat(q)],
            "client": client, "client_roles": [str(r) for r in client_roles or [] if str(r or "").strip()],
            "charge": charge, "contractual": contractual, "claimant": side == SIDE_CLAIMANT,
            "side": side, "side_why": side_why, "issues": named_issues(f, older_issues),
            "binding": {"enforced": bool((binding or {}).get("enforced")),
                        "pretesa": [str(x) for x in (binding or {}).get("principale") or []]} if binding else None}


def inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """The input as the flow's state holds it, after applicability_fit: the held frame's verified fields
    (case_frame.held -- CASE_FRAME on), the records not dropped and not removed by a verified «no»
    (interactive_flow.fit_removed_keys) with their row's text (applicability.norm_input), the state's sentences
    (interactive_flow.case_sentences when none), the frame's quesiti (else asked_questions), the client (the frame's,
    else the state's) and his role words (client_roles)."""
    from . import interactive_flow as flow
    cf, ap = _cf(), _ap()
    held = cf.held(state)
    frame = cf._frame(held) if held else {}
    rows = flow._restored_rows(state.get("rows") or [])
    have = {flow._key(r["_article"]): r for r in rows if r.get("_article")}
    removed = flow.fit_removed_keys(state)
    fits = {flow._key(flow._parse_label(str(j.get("article")))): j
            for j in ((state.get("applicability") or {}).get("judgments") or []) if isinstance(j, dict) and j.get("article")}
    norms = []
    for label, rec in (state.get("records") or {}).items():
        if not isinstance(rec, dict) or rec.get("dropped"):
            continue
        key = flow._key(flow._parse_label(label))
        if key in removed:
            continue
        n = ap.norm_input(label, have.get(key), rec, rows)
        kinds = {str(r.get("id")): r for r in rec.get("requisites") or [] if isinstance(r, dict) and r.get("id")}
        n["elements"] = [dict(e, **{k: kinds[str(e.get("id"))][k] for k in ("schema_kind", "special")
                                    if (kinds.get(str(e.get("id"))) or {}).get(k)})
                         for e in n.get("elements") or []]          # v79.15.1: the schema's own element kind
        j = fits.get(flow._key(flow._parse_label(label)))             # v79.16.1 (item 3): the fit's verdict
        if j is not None:
            n["fit"], n["fit_verified"], n["fit_perche"] = j.get("fit"), bool(j.get("verified")), j.get("perche") or ""
            if (state.get("applicability") or {}).get("governs_asked") and isinstance(j.get("governs"), list) and (
                    (j.get("verified") and j.get("fit") in (FIT_SI, FIT_DIPENDE))         # v79.17.0 (item 2)
                    or j.get("governs_recalled")):             # v79.17.2 (item 1): the recall's, verified or not
                n["governs"] = list(j["governs"])
        norms.append(n)
    sents = state.get("sentences") or flow.case_sentences(state)
    questions = list(frame.get("quesiti") or []) or flow.asked_questions(state)
    client = (frame.get("cliente") or {}).get("nome") or (state.get("client") or {}).get("name") or ""
    try:
        roles = flow.client_roles(state)
    except Exception:           # noqa: BLE001 -- no role read: the client's name and side words only
        roles = []
    return make_input(frame, norms, sents, questions, client, roles, older_issues(state, sents),
                      binding=state.get("binding_pretesa") if isinstance(state.get("binding_pretesa"), dict) else None)


def older_issues(state: Dict[str, Any], sentences: Sequence[Dict[str, Any]] = ()) -> List[str]:
    """v79.15.1 (item 2): the older frame's (CASE_FRAME_ENABLED, defensive_generation: state["analysis"]["frame"])
    central_issue_candidates' names -- that frame verifies no field, so a name counts only when every content word of
    it is in the case's sentences (case_frame.ungrounded, the 79.14 frame's own check of a descrizione); the system's
    assumptions are no text."""
    old = ((state.get("analysis") or {}).get("frame") or {}) if isinstance(state.get("analysis"), dict) else {}
    names = [_flat(c.get("nome") if isinstance(c, dict) else c) for c in old.get("central_issue_candidates") or []]
    text = " ".join(_flat(s.get("text")) for s in sentences or []
                    if isinstance(s, dict) and s.get("source") != "sistema")
    out = []
    for name in names:
        try:
            if name and text and not _cf().ungrounded(name, text):
                out.append(name)
        except Exception:       # noqa: BLE001 -- an unreadable name is no name
            continue
    return out


# --- the prompt -------------------------------------------------------------------------------------------------------

def _frame_lines(f: Dict[str, Any]) -> List[str]:
    labels = config()["labels"]
    lines = []
    if f.get("cliente"):
        lines.append(f"{labels['cliente']}: {f['cliente'].get('nome')}")
    for k in ("ruolo", "rito", "stadio"):
        if f.get(k):
            lines.append(f"{labels[k]}: {f[k].get('valore')}")
    if f.get("addebito_o_domanda"):
        a = f["addebito_o_domanda"]
        what = "; ".join(x for x in (a.get("articolo") or "", a.get("descrizione") or "") if x) or _flat(a.get("frase"))
        lines.append(f"{labels['addebito_o_domanda']}: {what}")
    if f.get("istituto"):
        lines.append(f"{labels['istituto']}: {f['istituto'].get('testo')}")
    parti = [p for p in f.get("parti") or [] if isinstance(p, dict) and p.get("nome")]
    if parti:
        lines.append(f"{labels['parti']}:")
        lines += [f"- {p['nome']} ({p.get('lato') or '-'}" + (f"; {p['ruolo_nel_fatto']}" if p.get("ruolo_nel_fatto")
                                                             else "") + ")" for p in parti]
    return lines


LABEL_LINE_RANK = "Preferenza per la linea (prima il fit si verificato, poi dipende verificato, poi non verificato)"
LABEL_GOVERNS = "governa"               # v79.17.0 (item 3): the article block's governs tags
LABEL_CONTEXT_ONLY = "(solo contesto: né la pretesa né un quesito)"


def _norm_block(n: Dict[str, Any], rank: Optional[Tuple[int, int]] = None) -> str:
    labels = config()["labels"]
    head = [f"ARTICOLO: {n['article']}" + (f" ({config()['remedy_mark']})" if n.get("remedy") else ""),
            f"Rubrica: {n.get('rubrica') or labels['none']}"]
    if n.get("fit"):                    # v79.16.1 (item 3): «fit: si — <perché>» (verificato: the ids held)
        head.append(f"fit: {n['fit']}" + (" (verificato)" if n.get("fit_verified") else "")
                    + (f" — {n['fit_perche']}" if n.get("fit_perche") else ""))
    if rank:                            # v79.16.2 (item 2): the tier told as a preference order, never a gate
        head.append(f"{labels.get('line_rank') or LABEL_LINE_RANK}: {rank[0]}/{rank[1]}")
    if isinstance(n.get("governs"), list):   # v79.17.0 (item 3): what the fit says the norm decides
        head.append(f"{labels.get('governs') or LABEL_GOVERNS}: " + (", ".join(n["governs"]) or LABEL_CONTEXT_ONLY))
    head.append(f"{labels['elements']}:")
    head += [f"- [{e['eid']}] {e['testo']}" for e in n.get("elements") or []] or [f"- {labels['none']}"]
    head.append(f"{labels['periods']}:")
    body = "\n".join(f"[{p['id']}] {p['text']}" for p in n.get("periods") or [])
    cap = int(config().get("max_article_chars") or 6000)
    if len(body) > cap:
        logger.info("analysis_step: %s periods cut at %d characters in the prompt", n["article"], cap)
        body = body[:cap] + " […]"
    return "\n".join(head) + "\n" + body


def build_messages(inp: Dict[str, Any]) -> List[Any]:
    """[system, human]: the verified frame's fields, the facts (S-id, label, text), the fitted norms (elements,
    numbered periods, the remedy mark), the user's questions."""
    from langchain_core.messages import HumanMessage, SystemMessage
    labels = config()["labels"]
    ap = _ap()
    parts = []
    frame = _frame_lines(inp.get("frame") or {})
    if frame:
        parts.append("\n".join([labels["frame"] + ":"] + frame))
    parts.append(f"{labels['facts']}:\n" + "\n".join(f"{s['id']} [{ap.fact_label(s)}] {s['text']}"
                                                      for s in inp.get("sentences") or []))
    tier, allowed = line_tier(inp)      # v79.16.2 (item 2): each norm's rank in the preference order (fit on)
    ranks = {} if tier == TIER_ALL else {a: (i + 1, len(allowed)) for i, a in enumerate(allowed)}
    parts.append(f"{labels['norms']}:\n\n" + "\n\n".join(_norm_block(n, ranks.get(n["article"]))
                                                            for n in listed_norms(inp)))      # v79.17.1 (item 7)
    qs = inp.get("questions") or []
    parts.append(f"{labels['questions']}:\n" + ("\n".join(f"- {q}" for q in qs) if qs else labels["none"]))
    return [SystemMessage(content=prompt_system()), HumanMessage(content="\n\n".join(parts))]


def has_elements(norm: Dict[str, Any]) -> bool:
    """v79.17.1 (item 7): the norm carries at least one schema element (the record's requisites, make_input)."""
    return bool(norm.get("elements"))


def listed_norms(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.17.1 (item 7): the norms the prompt's article block lists -- those with at least one element, and the remedy
    set (marked, never a candidate: listed for its periods, the remedies section's); a non-remedy article with no
    element is context only (no_element_articles): not listed, never offered, never a line."""
    return [n for n in inp.get("norms") or [] if has_elements(n) or n.get("remedy")]


def no_element_articles(inp: Dict[str, Any]) -> List[str]:
    """v79.17.1 (item 7): the fitted articles with no schema element (the remedy set aside: never a candidate anyway)
    -- counted candidates_no_elements_excluded, logged «context_only_no_elements»."""
    return [n["article"] for n in inp.get("norms") or [] if not has_elements(n) and not n.get("remedy")]


@lru_cache(maxsize=1)
def _rito_codes() -> Dict[str, frozenset]:
    """data «rito_codes»: the frame's rito -> the article_key code keys of its family."""
    return {str(k): frozenset(str(c) for c in v) for k, v in (config().get("rito_codes") or {}).items()
            if isinstance(v, (list, tuple))}


def frame_rito(frame: Optional[Dict[str, Any]]) -> str:
    """The frame's rito (valore, else intake -- as penal_frame reads it); "" with none."""
    rito = (frame or {}).get("rito") if isinstance((frame or {}).get("rito"), dict) else {}
    return str(rito.get("valore") or rito.get("intake") or "")


def outside_rito(norm: Dict[str, Any], inp: Dict[str, Any]) -> str:
    """v79.17.1 (item 9): the rito whose family the article's code is outside of (1439 c.c. in a penal case), "" when
    nothing restricts it: no rito, a rito not in the table, a code not in the table (a special law, the Costituzione),
    or the rito's own code."""
    table = _rito_codes()
    rito = frame_rito(inp.get("frame"))
    own = table.get(rito)
    if not own:
        return ""
    code = art_parts(norm.get("article"))[1]
    known = frozenset().union(*table.values())
    return rito if code and code in known and code not in own else ""


def governs_tagged(norm: Dict[str, Any]) -> bool:
    """The fit tagged the norm as governing the pretesa or a quesito (a governs list with a token)."""
    return isinstance(norm.get("governs"), list) and bool(norm["governs"])


def outside_rito_articles(inp: Dict[str, Any]) -> List[str]:
    """v79.17.1 (item 9): the fitted non-remedy articles whose code is outside the frame's rito (logged)."""
    return [n["article"] for n in inp.get("norms") or [] if not n.get("remedy") and outside_rito(n, inp)]


def line_elements(norm: Dict[str, Any], inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.15.1: the elements a line on this norm may be played on -- every one; for a claimant, the constitutive
    ones only (exception_of false). v79.16.1 (item 3): never a procedural, temporal or remedial element
    (procedural_element)."""
    els = [e for e in norm.get("elements") or [] if not procedural_element(e)]
    return [e for e in els if not e.get("eccezione")] if inp.get("claimant") else els


# --- v79.16.1 (item 3): the fit's verdicts feed the step -----------------------------------------------------------------
FIT_SI, FIT_NO, FIT_DIPENDE = "si", "no", "dipende"
GOVERNS_PRETESA = "pretesa"             # v79.17.0 (item 2): the governs token of the pretesa (applicability.GOVERNS_PRETESA)
TIER_SI, TIER_DIPENDE, TIER_UNVERIFIED, TIER_ALL = "fit_si", "fit_dipende", "unverified", "all"
PROCEDURAL_KINDS = ("termine", "competenza", "onere_della_prova", "procedibilità", "quantificazione")


def _canon_fit_value(value: Any) -> Optional[str]:
    v = str(value or "").strip().lower()
    return v if v in (FIT_SI, FIT_NO, FIT_DIPENDE) else None


def governs_sets(inp: Dict[str, Any]) -> Dict[str, Any]:
    """v79.17.0 (item 2): the binding on the pretesa as the step reads it from the norms' «governs» -- {"applied": a
    norm governs the pretesa, "pretesa": its articles, "quesiti": the articles governing a quesito only, "context":
    the verified norms governing neither (never a line)}; applied False when no norm carries a governs list. v79.17.2
    (items 1-2): with inp["binding"] enforced (the fit was given a pretesa) the pretesa set is the flow's (binding_pretesa
    .principale, among this input's norms) -- possibly empty -- and applied is True whatever the norms' tags: the
    leading line must be in it, nothing unjudged stands in for it."""
    norms = [n for n in inp.get("norms") or [] if not n.get("remedy")]
    with_gov = [n for n in norms if isinstance(n.get("governs"), list)]
    binding = inp.get("binding") if isinstance(inp.get("binding"), dict) else None
    enforced = bool(binding and binding.get("enforced"))
    out = {"applied": False, "pretesa": [], "quesiti": [], "context": [], "enforced": enforced}
    if not with_gov and not enforced:
        return out
    for n in with_gov:
        gov = n["governs"]
        if GOVERNS_PRETESA in gov:
            out["pretesa"].append(n["article"])
        elif any(str(x).startswith("Q") for x in gov):
            out["quesiti"].append(n["article"])
        else:
            out["context"].append(n["article"])
    if enforced:                    # v79.17.2 (items 1-2): the flow's set, among this input's norms, is the set
        labels = {n["article"] for n in norms}
        out["pretesa"] = [x for x in binding.get("pretesa") or [] if x in labels]
        out["quesiti"] = [x for x in out["quesiti"] if x not in out["pretesa"]]
        out["context"] = [x for x in out["context"] if x not in out["pretesa"]]
    out["applied"] = bool(out["pretesa"]) or enforced
    return out


def line_tier(inp: Dict[str, Any], kind: str = "linea_principale") -> Tuple[str, List[str]]:
    """(the best tier present, the articles a line may stand on in preference order): the verified «si», then the
    verified «dipende», then the unverified «si» / «dipende» and the norms the fit never judged; never a «no», never a
    remedy. v79.16.2 (item 2): a preference, not a gate -- the list holds every tier, in that order; _verify_line refuses
    a «no» only. With no fit judgment on any norm (the fit off): every non-remedy norm (TIER_ALL). v79.17.0 (item 2):
    with the binding on the pretesa applied (governs_sets), the leading line's list holds the pretesa's articles only,
    the subordinate's those and the quesiti's; a context-only norm is in neither. v79.17.1 (item 7): a norm with no
    schema element is in no list (context only). v79.17.1 (item 9): a norm whose code is outside the frame's rito
    (outside_rito) is never in the leading line's list and in the subordinate's only when the fit tagged it as governing
    the pretesa or a quesito (governs_tagged)."""
    norms = [n for n in inp.get("norms") or [] if not n.get("remedy") and has_elements(n)]    # v79.17.1 (item 7)
    # v79.17.1 (item 9): a code outside the frame's rito -- never the leading line; the subordinate only when tagged
    norms = [n for n in norms if not outside_rito(n, inp) or (kind != "linea_principale" and governs_tagged(n))]
    gs = governs_sets(inp)
    if gs["applied"]:               # the pretesa's articles (the subordinate: the quesiti's too); a norm the fit never
        allowed_g = set(gs["pretesa"]) | (set(gs["quesiti"]) if kind != "linea_principale" else set())    # judged stays
        norms = [n for n in norms if n["article"] in allowed_g       # -- unless the binding is enforced (v79.17.2)
                 or (not gs.get("enforced") and not isinstance(n.get("governs"), list))]
    if not any(n.get("fit") for n in inp.get("norms") or []):
        return TIER_ALL, [n["article"] for n in norms]
    verified = lambda n, fit: n.get("fit") == fit and bool(n.get("fit_verified"))    # noqa: E731
    top, allowed = TIER_UNVERIFIED, []
    for tier, keep in ((TIER_SI, lambda n: verified(n, FIT_SI)),
                       (TIER_DIPENDE, lambda n: verified(n, FIT_DIPENDE)),
                       (TIER_UNVERIFIED, lambda n: n.get("fit") != FIT_NO and not verified(n, FIT_SI)
                        and not verified(n, FIT_DIPENDE))):
        mine = [n["article"] for n in norms if keep(n)]
        if mine and not allowed:
            top = tier
        allowed += mine
    return top, allowed


def tier_tag(norm: Dict[str, Any]) -> str:
    """v79.16.2 (item 2): the norm's fit as the re-ask's allowed list tags it -- «fit si verificato», «fit dipende
    verificato», «fit si non verificato», «fit non giudicato»; v79.17.0: «; governa: pretesa, Q2» when the fit said."""
    fit = norm.get("fit")
    tag = "fit non giudicato" if not fit else f"fit {fit} " + ("verificato" if norm.get("fit_verified") else "non verificato")
    if isinstance(norm.get("governs"), list):
        tag += f"; {LABEL_GOVERNS}: " + (", ".join(norm["governs"]) or "solo contesto")
    return tag


def line_fit_counter(norm: Dict[str, Any]) -> str:
    """The counter a verified line on this norm adds to: line_from_fit_si / line_from_fit_dipende (verified), else
    line_from_unverified (v79.16.2: an unverified or unjudged fit alike; finish adds none with the fit off)."""
    fit = norm.get("fit")
    if norm.get("fit_verified") and fit == FIT_SI:
        return "line_from_fit_si"
    if norm.get("fit_verified") and fit == FIT_DIPENDE:
        return "line_from_fit_dipende"
    return "line_from_unverified"


@lru_cache(maxsize=1)
def _procedural_re() -> "re.Pattern[str]":
    pats = [str(x) for x in config().get("procedural_patterns") or [] if str(x or "").strip()]
    return re.compile("|".join(f"(?:{p})" for p in pats) if pats else r"(?!x)x", re.IGNORECASE)


def procedural_element(element: Dict[str, Any]) -> bool:
    """An element whose text is procedural, temporal or remedial -- never the element in play: by the schema's own kind
    when it has one (PROCEDURAL_KINDS), else by the data's pattern list («procedural_patterns», data/analysis_step.json)
    on the element's text alone."""
    kind = _flat(element.get("schema_kind")).lower()
    if kind:
        return kind in PROCEDURAL_KINDS
    if _flat(element.get("special")).lower() == "procedibilità":
        return True
    return bool(_procedural_re().search(_flat(element.get("testo") or element.get("elemento"))))


def allowed_text(inp: Dict[str, Any]) -> str:
    """v79.15.1 (item 2): the re-ask's list of the articles a line may stand on -- the fitted norms not in the remedy
    set (the frame's own charge kept: make_input never marks it), each with at least one usable element -- one per
    line: «- A<token> (art. <label>): <E-ids>» (a claimant's: the constitutive E-ids only)."""
    rows = []
    tier, allowed = line_tier(inp, "linea_subordinata")     # v79.16.2 (item 2): every tier, in the preference order,
    norms = {n["article"]: n for n in inp.get("norms") or []}  # each tagged; v79.17.0: the subordinate's wider list
    gs = governs_sets(inp)
    for label in allowed:
        n = norms[label]
        ids = [e["eid"] for e in line_elements(n, inp)]
        if n.get("remedy") or not ids:
            continue
        tag = "" if tier == TIER_ALL else f" [{tier_tag(n)}]"
        if (gs["applied"] and label in gs["quesiti"]) or outside_rito(n, inp):    # v79.17.0 (item 2): a quesito's
            tag += " [solo linea subordinata]"      # article, subordinate only; v79.17.1 (item 9): a code outside the rito
        rows.append(f"- A{_ap().article_token(n['article'])} (art. {n['article']}): {', '.join(ids)}{tag}")
    return "\n".join(rows) or f"- {config()['labels']['none']}"


def reask_messages(messages: List[Any], reply: str, reasons: Sequence[str],
                   inp: Optional[Dict[str, Any]] = None) -> List[Any]:
    """The re-ask: the first exchange, then the rejection reasons (data «reask»); v79.15.1: the allowed articles with
    their E-ids («{allowed}», allowed_text) and, for a claimant, data «reask_claimant» («{claimant}»)."""
    from langchain_core.messages import AIMessage, HumanMessage
    inp = inp or {}
    out = []
    for line in config()["reask"]:
        if "{claimant}" in line:
            if inp.get("claimant"):
                out.append(config().get("reask_claimant") or "")
            continue
        out.append(line.replace("{reasons}", "\n".join(f"- {r}" for r in reasons))
                   .replace("{allowed}", allowed_text(inp)))
    return list(messages) + [AIMessage(content=str(reply or "")), HumanMessage(content="\n".join(out))]


def request_kwargs() -> Dict[str, Any]:
    """The call's bind kwargs: the fit's (temperature 0, thinking off -- applicability.request_kwargs) with this
    step's reply budget."""
    out = dict(_ap().request_kwargs())
    out["max_tokens"] = int(config().get("max_tokens") or 3000)
    return out


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model (ai_chat.chat_model) bound to the kwargs; the reply without its reasoning, the call on the
    request's stage meter -- as applicability._default_call."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = ai_chat.chat_model.bind(**kwargs).invoke(messages)
    ai_chat._stage_call("analysis_step", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def parse_reply(raw: Any) -> Dict[str, Any]:
    """The reply's object; ValueError when it holds no JSON object."""
    from .llm_reply import parse_json_reply
    data = parse_json_reply(raw if isinstance(raw, str) else str(raw or ""))
    if not isinstance(data, dict):
        raise ValueError("no JSON object in the reply")
    return data


def call_once(messages: List[Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]],
              budget: float) -> Dict[str, Any]:
    """One call on a worker thread, waited at most budget seconds: {"status": ok | failed | timeout | malformed,
    "reply" (the text), "data" (the parsed object), "seconds", "error"}. A call not back in time is abandoned (its pool
    shut down without waiting); never raises."""
    started = time.time()
    if budget <= 0:
        return {"status": TIMEOUT, "reply": "", "data": None, "seconds": 0.0, "error": "nessun tempo residuo"}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis_step")
    try:
        fut = pool.submit(contextvars.copy_context().run, call or _default_call, messages, request_kwargs())
        done, _ = _wait([fut], timeout=budget)
        if fut not in done:
            return {"status": TIMEOUT, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"oltre {round(budget, 1)} s"}
        try:
            reply = fut.result()
        except Exception as exc:    # noqa: BLE001 -- nothing from the step
            return {"status": FAILED, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    try:
        data = parse_reply(reply)
    except Exception as exc:        # noqa: BLE001
        return {"status": MALFORMED, "reply": str(reply or "")[:2000], "data": None,
                "seconds": round(time.time() - started, 3), "error": str(exc)[:300]}
    return {"status": OK, "reply": str(reply or ""), "data": data, "seconds": round(time.time() - started, 3),
            "error": ""}


# --- who asserts a sentence -------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _verb_re() -> "re.Pattern[str]":
    verbs = "|".join(f"(?:{v})" for v in config().get("assertion_verbs") or [])
    return re.compile(rf"(?<![\w'’])(?:{verbs})(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _prefix_re() -> "re.Pattern[str]":
    """A speaker prefix: data «speaker_prefixes» anywhere after a space or a mark, «speaker_prefixes_initial» («Per la
    Procura, …») only at the sentence's start or after , ; : -- the phrase after it, to the next mark."""
    def alt(key: str) -> str:
        return "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in config().get(key) or []) or r"(?!x)x"
    return re.compile(rf"(?:(?:^|[\s,;:(])(?:{alt('speaker_prefixes')})|(?:^|[,;:])\s*(?:{alt('speaker_prefixes_initial')}))"
                      rf"\s+([^,;:.()]+)", re.IGNORECASE)


_CLAUSE_CUT_RE = re.compile(r"[,;:()«»\"“”]|\b(?:che|mentre|ma|però|perché|poiché|quando|dove|cui|se)\b",
                            re.IGNORECASE)
_SUBJECT_WORDS = 8
_TRAIL = {"ha", "hanno", "aveva", "avevano", "avrebbe", "avrebbero", "abbia", "abbiano", "è", "sono", "era", "erano",
          "non", "anche", "pure", "inoltre", "invece", "poi", "già", "sempre", "infatti", "ora", "oggi", "si"}
# a passive's agent: an assertion verb's participle followed by «da» («è accusato da Giulia Neri di …»); the agent's
# phrase ends at the next mark or at the first «di» / «per» / «a» / «in» / «con» that follows its first word
_PARTICIPLE_RE = re.compile(r"(?:at|ut|it|tt|st)[oaie]$", re.IGNORECASE)
_AGENT_RE = re.compile(r"^\s+(dall['’]|(?:dallo|dalla|dalle|dagli|dai|dal|da)\s)\s*([^,;:.()]+)", re.IGNORECASE)
_AGENT_ARTICLE = {"dal": "il ", "dallo": "lo ", "dalla": "la ", "dai": "i ", "dagli": "gli ", "dalle": "le ",
                  "dall'": "l'", "dall’": "l’"}         # «dalla Procura» -> «la Procura»
_AGENT_END_RE = re.compile(r"\s(?:di|per|a|ai|in|con|nei|nel|nella)\s", re.IGNORECASE)
SUBJECT, AFTER = "subject", "after"     # a phrase before its verb (the speaker its last mention) / after a marker (first)
CLIENT, OTHER = "client", "other"       # who a mention names: the client / another party or an authority


def _words(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", str(text or "").lower())


def _subject_spans(text: str) -> List[Tuple[int, str, str]]:
    """(position, phrase, SUBJECT | AFTER) of every phrase the sentence names as asserting something, in order: after a
    speaker prefix («Secondo la società, …», AFTER); the words right before each assertion verb in its clause (SUBJECT:
    a clause boundary -- a punctuation mark or «che», «mentre», «ma», … -- cuts them; an inciso between two commas
    skipped, «X, con una lettera, sostiene»; a verb coordinated with the assertion verb, «ammette e dichiara», and a
    trailing auxiliary or adverb left out; at most _SUBJECT_WORDS words); a passive's agent after the verb's participle
    («accusato da Giulia Neri», AFTER)."""
    s = str(text or "")
    found: List[Tuple[int, str, str]] = [(m.start(), m.group(1), AFTER) for m in _prefix_re().finditer(s)]
    for m in _verb_re().finditer(s):
        parts = _CLAUSE_CUT_RE.split(s[:m.start()])
        tail = parts[-1]
        if not tail.strip() and len(parts) >= 3:     # «X, <inciso>, sostiene»: the subject is before the inciso
            tail = parts[-3]
        words = tail.split()
        if len(words) >= 2 and words[-1].lower() in ("e", "ed"):      # «ammette e dichiara»
            words = words[:-2]
        while words and words[-1].lower().strip(".,") in _TRAIL:
            words = words[:-1]
        phrase = " ".join(words[-_SUBJECT_WORDS:]).strip()
        if re.search(r"[^\W\d_]{2,}", phrase):
            found.append((m.start(), phrase, SUBJECT))
        agent = _AGENT_RE.match(s[m.end():]) if _PARTICIPLE_RE.search(m.group(0)) else None
        if agent:
            phrase = _AGENT_ARTICLE.get(agent.group(1).strip().lower(), "") + \
                _AGENT_END_RE.split(" " + agent.group(2).strip() + " ", maxsplit=1)[0].strip()
            phrase = " ".join(phrase.split()[:_SUBJECT_WORDS])
            if re.search(r"[^\W\d_]{2,}", phrase):
                found.append((m.end(), phrase, AFTER))
    return sorted(found)


def subjects(text: str) -> List[str]:
    """The phrases a sentence names as asserting it, in order (_subject_spans)."""
    return [p for _, p, _k in _subject_spans(text)]


def _parties(inp: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """(name, is the client) of frame.parti, the input's client added when no party is his."""
    cf = _cf()
    out = [(str(p["nome"]), p.get("lato") == "cliente") for p in (inp.get("frame") or {}).get("parti") or []
           if isinstance(p, dict) and _flat(p.get("nome"))]
    client = _flat(inp.get("client"))
    if client and not any(cf.same_party(client, n) for n, _ in out):
        out.append((client, True))
    elif client:
        out = [(n, c or cf.same_party(client, n)) for n, c in out]
    return out


def _mentions(phrase: str, inp: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """(position, CLIENT | OTHER, the label) of every party or authority a phrase names, in order:
      - an authority word (data «authority_words») -> OTHER, labelled by the word as written with the article before it
        («la Procura», «Gli investigatori»);
      - a run of consecutive name words (case_frame name tokens: accent-folded, lowercased) -> the parties whose
        distinctive words (case_frame._distinctive) are among the run's, a party whose matched words are a strict subset
        of another's left out («Giulia Sarti» names Giulia Sarti, not Paolo Sarti); the client alone -> CLIENT;
        any other party left -> OTHER, its name (a tie between the client and another party -- «Sarti» with both
        named Sarti -- is OTHER: never read as the client's own statement);
      - a content word (4 letters or more) of a non-client party's ruolo_nel_fatto not in the client's -> OTHER;
      - the client's side (interactive_flow.CLIENT_SIDE_STEMS: cliente, assistito, difesa) or procedural role
        (_speaks_as_role on client_roles) -> CLIENT."""
    from . import interactive_flow as flow
    cf = _cf()
    raw = str(phrase or "")
    toks = [(m.start(), m.group(0)) for m in re.finditer(r"[^\W\d_]+", raw)]
    folded = [(pos, (cf.name_tokens(w) or [""])[0]) for pos, w in toks]
    out: List[Tuple[int, str, str]] = []
    low = raw.lower()
    for w in config().get("authority_words") or []:
        w = str(w).lower()
        pat = rf"(?<![\w.]){re.escape(w)}(?![\w])" if "." in w else rf"(?<![\w]){re.escape(w).replace(r'\ ', r'\s+')}\b"
        for m in re.finditer(pat, low):
            det = re.search(r"(?:^|\s)((?:il|lo|la|i|gli|le)\s+|l['’])$", raw[:m.start()], re.IGNORECASE)
            start = m.start() - len(det.group(1)) if det else m.start()
            out.append((start, OTHER, raw[start:m.end()].strip()))
    parties = _parties(inp)
    own = [(name, client, cf._distinctive(name)) for name, client in parties]
    name_words = set().union(*[d for _, _, d in own]) if own else set()
    i = 0
    while i < len(folded):
        if folded[i][1] not in name_words:
            i += 1
            continue
        j = i
        while j < len(folded) and folded[j][1] in name_words:
            j += 1
        run = {w for _, w in folded[i:j]}
        hit = [(name, client, d & run) for name, client, d in own if d & run]
        hit = [h for h in hit if not any(h[2] < o[2] for o in hit)]
        others = [h for h in hit if not h[1]]
        out.append((folded[i][0], OTHER, others[0][0]) if others else (folded[i][0], CLIENT, hit[0][0]))
        i = j
    client_roles = {w for name, client in parties if client for p in (inp.get("frame") or {}).get("parti") or []
                    if isinstance(p, dict) and p.get("nome") == name for w in _words(p.get("ruolo_nel_fatto"))}
    for p in (inp.get("frame") or {}).get("parti") or []:
        if not isinstance(p, dict) or not p.get("nome") or p.get("lato") == "cliente":
            continue
        role = {w for w in _words(p.get("ruolo_nel_fatto")) if len(w) >= 4} - client_roles
        out += [(pos, OTHER, str(p["nome"])) for pos, w in toks if w.lower() in role]
    for pos, w in toks:
        if any(w.lower().startswith(stem) for stem in flow.CLIENT_SIDE_STEMS) or (
                inp.get("client_roles") and flow._speaks_as_role(w, inp["client_roles"])):
            out.append((pos, CLIENT, w))
    if inp.get("client_roles") and flow._speaks_as_role(raw, inp["client_roles"]) and \
            not any(k == CLIENT for _, k, _l in out):          # a role of several words: where its phrase starts
        out.append((0, CLIENT, raw))
    return sorted(out, key=lambda x: x[0])


def speaker(phrase: str, inp: Dict[str, Any], last: bool = True) -> Tuple[str, str]:
    """(CLIENT | OTHER | "", the label) of who a phrase names as speaking: its last mention (a subject before its verb:
    «La difesa di Giulia Neri» is Giulia Neri, «Il difensore di Paolo Sarti» the client) or, last=False, its first (a
    phrase after a marker: «Per la Procura Paolo Sarti ha …» is the Procura); ("", "") none."""
    found = _mentions(phrase, inp)
    if not found:
        return "", ""
    _pos, kind, label = found[-1] if last else found[0]
    return kind, label


def _is_client(phrase: str, inp: Dict[str, Any]) -> bool:
    """The phrase names the client as the one speaking (speaker: CLIENT)."""
    return speaker(phrase, inp)[0] == CLIENT


def _other_party(phrase: str, inp: Dict[str, Any]) -> str:
    """The non-client party or authority the phrase names as speaking (speaker: OTHER), "" none."""
    kind, label = speaker(phrase, inp)
    return label if kind == OTHER else ""


def asserting_party(text: str, inp: Dict[str, Any]) -> str:
    """The non-client party or authority a sentence reports as asserting something: the first of its asserting phrases
    (_subject_spans) whose speaker is not the client -- one is enough, whatever the client says in the same sentence
    («Paolo Sarti nega, ma la Procura sostiene …» is the Procura's assertion); "" when none."""
    for _pos, phrase, kind in _subject_spans(text):
        who, label = speaker(phrase, inp, last=(kind == SUBJECT))
        if who == OTHER:
            return label
    return ""


# v79.16.0 (A2): the record as provenance -- the sentences carry the record's stato («certainty») and chi («said_by»,
# fact_record.verify: a speaker only for an allegation, only a name in the text). A «provato» stands on one cited fact
# stated accertato, or alleged by the client, or alleged by the other party against its own interest (the element
# burdens that party: a claimant client's exceptions, a defending client's constitutive elements); else it is
# contestato, the allegation's party named. The text heuristic (reported_by) only where the record is absent.
ACCERTATO_, ALLEGATO_ = "accertato", "allegato"


def burdens_other(element: Dict[str, Any], inp: Dict[str, Any]) -> bool:
    """The party other than the client bears the element: a claimant client -- its exceptions; a defending client --
    the constitutive elements; the side unknown -- never."""
    side = inp.get("side") or (SIDE_CLAIMANT if inp.get("claimant") else SIDE_NEUTRAL)
    if side == SIDE_CLAIMANT:
        return bool(element.get("eccezione"))
    if side == SIDE_DEFENCE:
        return not element.get("eccezione")
    return False


def record_provenance(ids: Sequence[str], m: Dict[str, Any], inp: Dict[str, Any], element: Dict[str, Any]
                      ) -> Tuple[Optional[str], bool]:
    """(the party whose allegation a «provato» rests on -- None: it stands --, whether the record decided). The record
    absent from the cited sentences (none carries «fact»): (None, False), the caller's heuristic decides."""
    sents = [m["sents"][i] for i in ids if i in m["sents"]]
    if not any(s.get("fact") for s in sents):
        return None, False
    alleged: List[str] = []
    for s in sents:
        if s.get("certainty") == ACCERTATO_:
            return None, True
        if s.get("certainty") != ALLEGATO_:
            continue
        chi = str(s.get("said_by") or s.get("alleged_by") or "")
        kind, label = speaker(chi, inp) if chi else ("", "")
        if chi and kind == CLIENT:
            return None, True
        if chi and kind == OTHER and burdens_other(element, inp):
            return None, True                   # an admission against its own interest
        alleged.append(label or chi)
    return next((w for w in alleged if w), ""), True


# --- v79.16.1 (item 4): a «provato» / «escluso» needs a pertinent fact -----------------------------------------------------
_LEMMA_RE = re.compile(r"[a-zà-ü]{4,}")
STEM_CHARS = 5


def _lemmas(text: Any, stop: set) -> set:
    """The content lemmas of a text: its words of four letters or more, accents folded, not in `stop`, each cut to
    STEM_CHARS (simple stemming: «appropriato» / «appropria» -> «appro»)."""
    return {w[:STEM_CHARS] for w in _LEMMA_RE.findall(fold(text)) if w not in stop and w[:STEM_CHARS] not in stop}


@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    words = set()
    try:
        cfg = _ap().config()
        for key in ("connective_words", "question_words", "generic_norm_words", "omittable_words", "article_words"):
            words |= {fold(x) for x in cfg.get(key) or []}
    except Exception:           # noqa: BLE001 -- the data's own list only
        pass
    words |= {fold(x) for x in config().get("pertinence_stopwords") or []}
    return frozenset(words | {w[:STEM_CHARS] for w in words})


def party_words(inp: Dict[str, Any]) -> set:
    """The parties' names' words (the frame's parti, the client), folded: never a content lemma."""
    out = set()
    names = [inp.get("client") or ""] + [p.get("nome") or "" for p in (inp.get("frame") or {}).get("parti") or []
                                         if isinstance(p, dict)]
    for name in names:
        out |= {w for w in _LEMMA_RE.findall(fold(name))}
    return out | {w[:STEM_CHARS] for w in out}


def pertinent(element: Dict[str, Any], sentences: Sequence[Dict[str, Any]], inp: Dict[str, Any]) -> bool:
    """At least one cited sentence shares a content lemma with the element's text (the schema text), stopwords and the
    parties' names out. An element with no content lemma of its own: pertinent (nothing to compare)."""
    stop = set(_stopwords()) | party_words(inp)
    want = _lemmas(element.get("testo") or element.get("elemento"), stop)
    if not want:
        return True
    return any(want & _lemmas(s.get("text"), stop) for s in sentences or [])


def reported_by(sentence: Dict[str, Any], inp: Dict[str, Any], parte: str = "") -> str:
    """The party other than the client whose reported assertion the sentence is, "" when it is not one: parte (the
    reply's parte_che_afferma) naming a non-client party; 79.14.3's mark (alleged_by: interactive_flow.alleging_party);
    the sentence's asserting subject (asserting_party)."""
    if _flat(parte) and not _is_client(parte, inp):
        return _flat(parte)
    if sentence.get("alleged_by"):
        return str(sentence["alleged_by"])
    return asserting_party(str(sentence.get("text") or ""), inp)


# --- the verification -------------------------------------------------------------------------------------------------

def _ids(value: Any) -> List[str]:
    return _ap().id_list(value)


def _maps(inp: Dict[str, Any]) -> Dict[str, Any]:
    norms = {n["article"]: n for n in inp.get("norms") or []}
    return {"norms": norms, "labels": list(norms), "sents": {s["id"]: s for s in inp.get("sentences") or []},
            "periods": {label: {p["id"]: p for p in n.get("periods") or []} for label, n in norms.items()},
            "all_periods": {p["id"]: label for label, n in norms.items() for p in n.get("periods") or []}}


def element_of(norm: Dict[str, Any], value: Any, labels: Optional[Sequence[str]] = None
               ) -> Optional[Dict[str, Any]]:
    """v79.15.1 (item 1): the norm's schema element an E-id names -- «A<token>:E<n>» naming this norm, or a bare
    «E<n>» -- by existence only; None otherwise (a text, another article's id, a number past the last element).
    v79.15.2: the prefix read by the one key -- with `labels` (the fitted norms) it must resolve to this norm (resolve:
    a comma's own label first), else same_article."""
    got = _eid_split(value)
    if got is None:
        return None
    prefix, n = got
    article = norm.get("article")
    if prefix:
        mine = resolve(prefix, labels) == article if labels else (
            _id_token(prefix) == _ap().article_token(article) or same_article(prefix, article))
        if not mine:
            return None
    els = norm.get("elements") or []
    return els[n - 1] if 1 <= n <= len(els) else None


def _eid_article(value: Any, m: Dict[str, Any]) -> Optional[str]:
    """The fitted norm an E-id's prefix names (resolve, the one key); None for a bare or unknown one."""
    got = _eid_split(value)
    return resolve(got[0], m["labels"]) if got and got[0] else None


def _verify_line(raw: Any, inp: Dict[str, Any], m: Dict[str, Any], counters: Dict[str, Any],
                 kind: str = "linea_principale") -> Tuple[Optional[Dict[str, Any]], str]:
    """(the verified line, "") or (None, why it is rejected)."""
    if not isinstance(raw, dict):
        return None, "non è un oggetto"
    label = resolve(raw.get("articolo"), m["labels"])
    if label is None:
        return None, f"articolo non tra le norme applicabili: {_flat(raw.get('articolo'))[:60] or '(vuoto)'}"
    norm = m["norms"][label]
    if norm.get("remedy"):
        return None, f"art. {label}: rimedio o parte generale, mai una linea"
    if not has_elements(norm):              # v79.17.1 (item 7): no schema element -> context only, never a line
        return None, (f"art. {label}: nessun elemento dello schema, solo contesto, mai una linea: ammessi "
                      f"{', '.join(f'art. {x}' for x in line_tier(inp, kind)[1]) or config()['labels']['none']}")
    if norm.get("fit") == FIT_NO:           # v79.16.1 (item 3): never a «no»; v79.16.2 (item 2): an unverified or
        return None, f"art. {label}: fit «no», mai una linea"     # unjudged norm is allowed (a preference, not a gate)
    rito_out = outside_rito(norm, inp)      # v79.17.1 (item 9): a code outside the frame's rito (1439 c.c. in a penal
    if rito_out and (kind == "linea_principale" or not governs_tagged(norm)):    # case): subordinate only, and only
        counters["analysis_line_outside_rito_rejected"] += 1                     # when the fit tagged it
        ok = ", ".join(f"art. {x}" for x in line_tier(inp, kind)[1]) or config()["labels"]["none"]
        if kind == "linea_principale":
            return None, (f"art. {label}: codice fuori dal rito {rito_out}, mai linea principale"
                          + (", al più subordinata" if governs_tagged(norm) else ", solo contesto") + f": ammessi {ok}")
        return None, (f"art. {label}: codice fuori dal rito {rito_out} e il fit non lo riferisce né alla pretesa né a "
                      f"un quesito: solo contesto, mai una linea: ammessi {ok}")
    gs = governs_sets(inp)                  # v79.17.0 (item 2): the binding on the pretesa -- a context-only norm is
    if gs.get("enforced"):                  # v79.17.2 (item 2): the leading line in the set, always; the subordinate
        allowed_set = set(gs["pretesa"]) | (set(gs["quesiti"]) if kind != "linea_principale" else set())   # in the
        if label not in allowed_set and not (gs["applied"] and label in gs["context"]) \
                and not (gs["applied"] and kind == "linea_principale" and label in gs["quesiti"]):   # set or the quesiti's
            counters["analysis_line_outside_set_rejected"] += 1
            if kind == "linea_principale" and not gs["pretesa"]:
                return None, f"art. {label}: {HONESTY_LINE[:-1]}: nessuna linea principale"
            return None, (f"art. {label}: fuori dall'insieme che governa la pretesa"
                          + (" e dai quesiti" if kind != "linea_principale" else "")
                          + f": ammessi {', '.join(f'art. {x}' for x in sorted(allowed_set, key=m['labels'].index)) or config()['labels']['none']}")
    if gs["applied"] and label in gs["context"]:            # never a line, a quesito's only the subordinate
        counters["analysis_line_context_rejected"] += 1
        return None, (f"art. {label}: solo contesto (il fit non lo riferisce né alla pretesa né a un quesito), mai una "
                      f"linea: ammessi {', '.join(f'art. {x}' for x in gs['pretesa'] + gs['quesiti'])}")
    if gs["applied"] and kind == "linea_principale" and label in gs["quesiti"]:
        counters["analysis_line_quesito_only"] += 1
        return None, (f"art. {label}: governa solo un quesito, ammesso come linea subordinata, non come principale: "
                      f"principale tra {', '.join(f'art. {x}' for x in gs['pretesa'])}")
    element = element_of(norm, raw.get("elemento_in_gioco"), m["labels"])
    if element is None:
        counters["analysis_bad_eid"] += 1
        said = _flat(raw.get("elemento_in_gioco"))[:40] or "(vuoto)"
        return None, (f"art. {label}: elemento_in_gioco «{said}» non è l'id di un elemento dello schema di questo "
                      f"articolo (A{_ap().article_token(label)}:E1…E{len(norm.get('elements') or [])})")
    if procedural_element(element):         # v79.16.1 (item 3): a procedural / temporal / remedial clause never in play
        counters["analysis_line_procedural_rejected"] += 1
        ok = ", ".join(e["eid"] for e in line_elements(norm, inp)) or config()["labels"]["none"]
        return None, (f"art. {label}: elemento_in_gioco {element['eid']} «{_flat(element.get('testo'))[:60]}» è una "
                      f"clausola processuale, temporale o rimediale, non un elemento della fattispecie: ammessi {ok}")
    if inp.get("claimant") and element.get("eccezione"):
        counters["analysis_line_exception"] += 1
        ok = ", ".join(e["eid"] for e in line_elements(norm, inp)) or config()["labels"]["none"]
        return None, (f"art. {label}: elemento_in_gioco {element['eid']} è un'eccezione o un limite "
                      f"({element.get('eccezione_fonte')}), non un elemento costitutivo della domanda del cliente: "
                      f"elementi costitutivi {ok}")
    sf, sn = _ids(raw.get("cita_fatti")), _ids(raw.get("cita_norma"))
    bad_f = [i for i in sf if i not in m["sents"]]
    bad_n = [i for i in sn if i not in m["periods"][label]]
    counters["analysis_bad_sid"] += len(bad_f)
    counters["analysis_bad_aid"] += len(bad_n)
    if bad_f:
        return None, f"art. {label}: cita_fatti inesistenti: {', '.join(bad_f[:6])}"
    if bad_n:
        return None, f"art. {label}: cita_norma non dell'articolo: {', '.join(bad_n[:6])}"
    if not sf and not sn:
        return None, f"art. {label}: nessun id citato"
    tesi = _flat(raw.get("tesi"))
    if not tesi:
        return None, f"art. {label}: tesi mancante"
    return {"articolo": label, "elemento_in_gioco": element["testo"], "elemento_id": element["id"],
            "eid": element["eid"], "tesi": tesi, "cita_fatti": sf, "cita_norma": sn, "stato": VERIFIED,
            "fit": norm.get("fit"), "fit_verified": bool(norm.get("fit_verified"))}, ""


def verify(reply: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """One reply checked (pure): {"lines": {kind: verified line or None}, "rejected": [{kind, articolo, why}],
    "reasons": [the re-ask's], "charge_missing", "reply": the reply, "counters"}. The elements and the law statements
    are verified in finish (on the final lines' articles)."""
    counters = empty_counters()
    m = _maps(inp)
    lines: Dict[str, Optional[Dict[str, Any]]] = {}
    rejected: List[Dict[str, Any]] = []
    reasons: List[str] = []
    raised: List[Dict[str, Any]] = []       # v79.15.1 (item 3): a claimant's exception element put as a line
    for kind in LINE_KINDS:
        raw = reply.get(kind) if isinstance(reply, dict) else None
        if raw in (None, "", {}, []):
            lines[kind] = None
            if kind == "linea_principale":
                reasons.append("linea_principale mancante")
            continue
        label = resolve(raw.get("articolo"), m["labels"]) if isinstance(raw, dict) else None
        if inp.get("claimant") and label:
            el = element_of(m["norms"][label], raw.get("elemento_in_gioco"), m["labels"])
            if el and el.get("eccezione"):
                raised.append({"articolo": label, "eid": el["eid"], "id": el["id"]})
        line, why = _verify_line(raw, inp, m, counters, kind)      # v79.17.0 (item 2): the kind decides the set
        lines[kind] = line
        if why:
            counters["analysis_line_rejected"] += 1
            # v79.15.2: the display label once resolved («640 c.p.» for «640cp»), the reply's text only when it names
            # no fitted norm
            rejected.append({"kind": kind, "articolo": label or _flat((raw or {}).get("articolo")
                                                                      if isinstance(raw, dict) else ""), "why": why})
            reasons.append(f"{kind}: {why}")
    charge = inp.get("charge") or ""
    missing = bool(charge) and not any(x and same_article(x["articolo"], charge) for x in lines.values())
    if missing:
        reasons.append(f"l'addebito o la domanda del quadro (art. {charge}) deve essere la linea principale o la "
                       "subordinata")
    return {"lines": lines, "rejected": rejected, "reasons": reasons, "charge_missing": missing,
            "reply": reply if isinstance(reply, dict) else {}, "counters": counters, "raised": raised}


def _reply_elements(reply: Dict[str, Any]) -> List[Tuple[Any, Dict[str, Any]]]:
    """The reply's element verdicts as (the «articolo» it names or None, the item): a list, or an object keyed by
    article."""
    raw = reply.get("elementi")
    out: List[Tuple[Any, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for art, items in raw.items():
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict):
                    out.append((it.get("articolo") or art, it))
    elif isinstance(raw, list):
        out = [(it.get("articolo"), it) for it in raw if isinstance(it, dict)]
    return out


def _item_target(art: Any, it: Dict[str, Any], m: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """v79.15.1: (the fitted norm, its element) an element verdict names -- the E-id's own article (its token) first,
    else the article the item names; (None, None) when the id names no element of it."""
    label = _eid_article(it.get("elemento"), m) or (resolve(art, m["labels"]) if art else None)
    if label is None:
        return None, None
    return label, element_of(m["norms"][label], it.get("elemento"), m["labels"])


def _canon_esito(value: Any) -> Optional[str]:
    v = _flat(value).lower()
    v = (config().get("esiti_synonyms") or {}).get(v, v)
    return v if v in config()["esiti"] else None


def _verify_element(item: Dict[str, Any], element: Dict[str, str], inp: Dict[str, Any], m: Dict[str, Any],
                    counters: Dict[str, Any]) -> Dict[str, Any]:
    out = {"elemento": element["testo"], "id": element["id"], "esito": DA_PROVARE, "cita_fatti": [],
           "parte_che_afferma": None, "origine": "modello"}
    esito = _canon_esito(item.get("esito"))
    if esito is None:
        counters["analysis_esito_invalid"] += 1
        return out
    ids = _ids(item.get("cita_fatti"))
    bad = [i for i in ids if i not in m["sents"]]
    if bad:                         # a verdict citing a fact that does not exist is void
        counters["analysis_bad_sid"] += len(bad)
        out["scartato"] = f"cita_fatti inesistenti: {', '.join(bad[:6])}"
        return out
    parte = _flat(item.get("parte_che_afferma")) if isinstance(item.get("parte_che_afferma"), str) else ""
    out["cita_fatti"] = ids
    if parte:
        out["parte_che_afferma"] = parte
    out["fatti"] = [str(m["sents"][i].get("fact")) for i in ids if m["sents"][i].get("fact")]   # v79.16.0 (A4)

    def not_pertinent() -> Dict[str, Any]:  # v79.16.1 (item 4): a «provato» / «escluso» with no cited fact sharing a
        counters["analysis_provato_no_pertinence"] += 1      # content lemma with the element: counted and logged;
        out["non_pertinente"] = {"esito_modello": esito, "fatti": [_flat(m["sents"][i].get("text"))[:160] for i in ids]}
        out["esito"] = esito                                  # v79.16.2 (item 1): the esito stays as the model gave it
        return out
    if esito == PROVATO:
        if not ids:
            counters["analysis_provato_unsupported"] += 1
            return out
        who, by_record = record_provenance(ids, m, inp, element)     # v79.16.0 (A2): the record first
        if by_record:
            counters["analysis_provenance_from_record"] += 1
            if who is not None:
                counters["analysis_provato_downgraded"] += 1
                out["esito"], out["parte_che_afferma"] = CONTESTATO, (who or None)
                out["declassato"] = "provato su un'allegazione"
                return out
            if not pertinent(element, [m["sents"][i] for i in ids], inp):
                return not_pertinent()
            out["esito"] = PROVATO
            return out
        counters["analysis_provenance_fallback"] += 1
        by = [reported_by(m["sents"][i], inp, parte) for i in ids]
        if all(by):
            counters["analysis_provato_downgraded"] += 1
            out["esito"], out["parte_che_afferma"] = CONTESTATO, by[0]
            out["declassato"] = "provato su un'allegazione"
            return out
        if not pertinent(element, [m["sents"][i] for i in ids], inp):
            return not_pertinent()
        out["esito"] = PROVATO
        return out
    if esito in (ESCLUSO, CONTESTATO) and not ids:
        counters["analysis_escluso_unsupported" if esito == ESCLUSO else "analysis_contestato_unsupported"] += 1
        return out
    if esito == ESCLUSO and not pertinent(element, [m["sents"][i] for i in ids], inp):
        return not_pertinent()
    if esito == CONTESTATO and not parte:
        who = [w for w in (str(m["sents"][i].get("said_by") or "") or reported_by(m["sents"][i], inp) for i in ids)
               if w]                            # v79.16.0 (A2): the record's speaker first
        out["parte_che_afferma"] = who[0] if who else None
    out["esito"] = esito
    return out


def _base_element(e: Dict[str, Any], origine: str) -> Dict[str, Any]:
    return {"elemento": e["testo"], "id": e["id"], "eid": e.get("eid"), "esito": DA_PROVARE, "cita_fatti": [],
            "fatti": [], "parte_che_afferma": None, "origine": origine, "eccezione": bool(e.get("eccezione")),
            "eccezione_fonte": e.get("eccezione_fonte") or ""}


def _elements_for(label: str, inp: Dict[str, Any], m: Dict[str, Any], items: List[Tuple[Any, Dict[str, Any]]],
                  counters: Dict[str, Any], all_open: bool = False, origine: str = "quadro",
                  bare_ok: bool = True) -> List[Dict[str, Any]]:
    """Every schema element of the article, in the schema's order, with its verified verdict (the reply's first item
    whose E-id names it -- v79.15.1: by id, its article the id's token, else the item's articolo, else (a bare «E<n>»
    naming no article) this one, only when the lines read stand on this one article (bare_ok; on two articles the bare
    id is ambiguous: not read, counted analysis_bad_eid by _bad_eids); none: «da_provare», analysis_element_missing;
    all_open: every one «da_provare»
    (origine: «quadro» for the frame's charge, «rubrica» for the rubrica fallback), the reply's verdicts not read).
    The element's text is the schema's, never the reply's."""
    norm = m["norms"].get(label) or {}
    got: Dict[str, Dict[str, Any]] = {}
    if not all_open:
        for art, it in items:
            target, element = _item_target(art, it, m)
            if target is None and not art and bare_ok and parse_eid(it.get("elemento")):
                target, element = label, element_of(norm, it.get("elemento"), m["labels"])
            if target != label or element is None or element["eid"] in got:
                continue
            got[element["eid"]] = {**_verify_element(it, element, inp, m, counters), "eid": element["eid"],
                                   "eccezione": bool(element.get("eccezione")),
                                   "eccezione_fonte": element.get("eccezione_fonte") or ""}
    out = []
    for e in norm.get("elements") or []:
        if e["eid"] in got:
            out.append(got[e["eid"]])
            continue
        if not all_open:
            counters["analysis_element_missing"] += 1
        out.append(_base_element(e, origine if all_open else "mancante"))
    return out


def _bad_eids(items: List[Tuple[Any, Dict[str, Any]]], m: Dict[str, Any], line_labels: Sequence[str]) -> int:
    """v79.15.1 (item 1): the reply's element verdicts for a line's article (or naming no article) whose «elemento» is
    no E-id of an element of it -- dropped, never read (a text, an unknown id, a number past the last element; a bare
    «E<n>» naming no article while the lines read stand on two articles: ambiguous)."""
    n = 0
    labels = list(dict.fromkeys(line_labels))
    for art, it in items:
        target, element = _item_target(art, it, m)
        if target is not None and target not in labels:
            continue                # another fitted article's verdict: not read, not counted
        if target is None and art:
            continue                # an article that is no fitted norm: not read
        if target is None:          # a bare id: read only on the one line article
            if len(labels) != 1 or element_of(m["norms"][labels[0]], it.get("elemento"), m["labels"]) is None:
                n += 1
        elif element is None:
            n += 1
    return n


def _law_statements(reply: Dict[str, Any], m: Dict[str, Any], counters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The verified affermazioni_di_diritto. One is kept only when (in this order, the first failure dropping it,
    law_statements_dropped, logged with the reason):
      - it has a text and at least one A-id, and every A-id exists among the fitted norms' periods (analysis_bad_aid);
      - every article its text names (citations.article_refs) is a fitted norm and has at least one of its own periods
        among the cited ones -- a statement about one article never verified by another's text
        (law_statements_off_article);
      - its tokens (tokens(): the Q&A filter's own) are grounded in the cited periods' text: at least the data's
        «law_statements.overlap» (0.6) of the statement's tokens among the periods' tokens, the denominator the
        statement's tokens (law_statements_ungrounded) -- a statement adding what the cited text does not say is not
        verified by it."""
    from .citations import article_refs
    need = float((config().get("law_statements") or {}).get("overlap") or 0.6)
    out = []
    raw = reply.get("affermazioni_di_diritto")
    for it in raw if isinstance(raw, list) else []:
        testo = _flat(it.get("testo")) if isinstance(it, dict) else ""
        ids = _ids(it.get("cita_norma")) if isinstance(it, dict) else []
        bad = [i for i in ids if i not in m["all_periods"]]
        counters["analysis_bad_aid"] += len(bad)
        why = "vuota" if not testo else "nessun id" if not ids else "id inesistente" if bad else ""
        if not why:
            for num, code in article_refs(testo):
                label = resolve(f"{num} {code}".strip(), m["labels"])
                if label is None or not any(m["all_periods"][i] == label for i in ids):
                    why = f"art. {num} {code}".strip() + (" non tra le norme applicabili" if label is None else
                                                          ": nessun suo periodo citato")
                    counters["law_statements_off_article"] += 1
                    break
        if not why:
            cited = " ".join(str((m["periods"][m["all_periods"][i]].get(i) or {}).get("text") or "") for i in ids)
            if overlap(testo, cited) < need:
                why = f"non fondata sui periodi citati ({round(overlap(testo, cited), 2)})"
                counters["law_statements_ungrounded"] += 1
        if why:
            counters["law_statements_dropped"] += 1
            logger.info("analysis_step: law statement dropped (%s): %s", why, testo[:120])
            continue
        counters["law_statements_kept"] += 1
        out.append({"testo": testo, "cita_norma": ids,
                    "articoli": list(dict.fromkeys(m["all_periods"][i] for i in ids))})
    return out


def empty_checked(inp: Dict[str, Any]) -> Dict[str, Any]:
    """v79.16.2 (item 3): the verification of no reply -- the first call failed, timed out or answered no JSON object:
    no line, nothing rejected, the frame's charge missing when the frame names one -- so finish applies the same
    fallbacks as after a reply (the frame's charge, the rubrica fallback, «da determinare», in that order) and logs
    analysis_fallback_skipped_reason before the rendering prints anything."""
    return {"lines": {k: None for k in LINE_KINDS}, "rejected": [], "reasons": [],
            "charge_missing": bool(inp.get("charge")), "reply": {}, "counters": empty_counters(), "raised": []}


def rubrica_fallback(inp: Dict[str, Any], allowed: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """v79.15.1 (item 2): the leading line from the frame's named charges / claims (inp «issues»: named_issues) -- the
    first name whose rubrica_key equals a fitted norm's rubrica's (no fuzzy match), the remedy set left out, its first
    such norm in the input's order: stato FALLBACK_LINE, no element in play, no id cited, tesi FALLBACK_TESI on the
    norm's rubrica as the input has it (penal: penal_frame); None when no name matches. v79.17.2 (item 2): allowed --
    the governs-pretesa set when the binding is enforced: a norm outside it is never the fallback's line either."""
    for name in inp.get("issues") or []:
        key = rubrica_key(name)
        if not key:
            continue
        for n in inp.get("norms") or []:
            if n.get("remedy") or not n.get("rubrica") or rubrica_key(n["rubrica"]) != key:
                continue
            if allowed is not None and n["article"] not in allowed:
                continue
            tesi = FALLBACK_TESI["penale" if penal_frame(inp.get("frame"), n["article"]) else "civile"]
            return {"articolo": n["article"], "elemento_in_gioco": None, "elemento_id": None, "eid": None,
                    "tesi": tesi.format(rubrica=_stop(n["rubrica"])), "cita_fatti": [], "cita_norma": [],
                    "stato": FALLBACK_LINE, "nome": name}
    return None


def fallback_miss(inp: Dict[str, Any]) -> str:
    """v79.15.2 (item 2): why rubrica_fallback found no line -- SKIP_NO_NAMED_CHARGE (the frame names no charge or
    claim: no verified descrizione, no istituto, no older-frame name grounded in the case) or SKIP_NO_RUBRICA_MATCH (a
    name, but none equal, word for word after fold, to the rubrica of a fitted norm outside the remedy set)."""
    return SKIP_NO_RUBRICA_MATCH if any(rubrica_key(x) for x in inp.get("issues") or []) else SKIP_NO_NAMED_CHARGE


def objections_of(lines: Sequence[Optional[Dict[str, Any]]], elements: Dict[str, List[Dict[str, Any]]],
                  raised: Sequence[Dict[str, Any]], inp: Dict[str, Any], m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.15.1 (item 3): the objections the verified analysis carries, in order --
      - one per allegation: a cited sentence of a «contestato» element of a line article that a party other than the
        client asserts (reported_by; none such, the element's parte_che_afferma when it is not the client: every cited
        sentence) -- {"fonte": "allegazione", "sid", "parte", "elementi": [{articolo, eid, testo, esito,
        parte_che_afferma}], "fatti_cliente": the element's other cited sentences (and, for the element in play, the
        line's) that are no other party's assertion};
      - a claimant's exception elements the model raised (put as elemento_in_gioco on any attempt -- `raised` -- or a
        line article's exception element with a verdict other than «da_provare»), each once, not already in an
        allegation's objection -- {"fonte": "eccezione", "articolo", "eid", "testo", "esito", "parte_che_afferma",
        "cita_fatti", "fatti_cliente"}."""
    sents = m["sents"]
    out: List[Dict[str, Any]] = []
    by_sid: Dict[str, Dict[str, Any]] = {}
    in_play = {(x["articolo"], x.get("eid")): x for x in lines if x and x.get("articolo") and x.get("eid")}

    def client_facts(ids: Sequence[str]) -> List[str]:
        return [i for i in dict.fromkeys(ids) if i in sents and not reported_by(sents[i], inp)]     # (build B) once

    def whole(who: str) -> str:     # (build B) a party's name cut at its abbreviation's dot («S.r.l») made whole
        for name, _client in _parties(inp):
            if name.endswith(".") and who.endswith(name[:-1]):
                return who + "."
        return who
    for label, els in elements.items():
        for e in els:
            if e.get("esito") != CONTESTATO:
                continue
            cited = [i for i in e.get("cita_fatti") or [] if i in sents]
            alleg = [(i, reported_by(sents[i], inp)) for i in cited]
            alleg = [(i, whole(w)) for i, w in alleg if w]
            parte = _flat(e.get("parte_che_afferma"))
            if not alleg and parte and not _is_client(parte, inp):
                alleg = [(i, parte) for i in cited]
            line = in_play.get((label, e.get("eid")))
            own = client_facts([i for i in cited if i not in dict(alleg)] + list((line or {}).get("cita_fatti") or []))
            item = {"articolo": label, "eid": e.get("eid"), "testo": e["elemento"], "esito": e["esito"],
                    "parte_che_afferma": e.get("parte_che_afferma")}
            for sid, who in alleg:
                if sid not in by_sid:
                    by_sid[sid] = {"fonte": "allegazione", "sid": sid, "parte": who, "elementi": [],
                                   "fatti_cliente": [],     # v79.15.3: the sentence names its own speaker
                                   "nominata": bool(asserting_party(str(sents[sid].get("text") or ""), inp))}
                    out.append(by_sid[sid])
                o = by_sid[sid]
                if all(x["eid"] != item["eid"] or x["articolo"] != label for x in o["elementi"]):
                    o["elementi"].append(item)
                o["fatti_cliente"] += [i for i in own if i not in o["fatti_cliente"] and i != sid]
    if not inp.get("claimant"):
        return out
    done = {(x["articolo"], x["eid"]) for o in out for x in o["elementi"]}
    candidates = [(r["articolo"], r["eid"]) for r in raised or []]
    candidates += [(label, e.get("eid")) for label, els in elements.items() for e in els
                   if e.get("eccezione") and e.get("esito") != DA_PROVARE]
    for label, eid in candidates:
        if (label, eid) in done or not label:
            continue
        done.add((label, eid))
        norm = m["norms"].get(label) or {}
        base = element_of(norm, eid, m["labels"]) or {}
        e = next((x for x in elements.get(label) or [] if x.get("eid") == eid), None) or \
            (_base_element(base, "eccezione") if base else None)
        if e is None:
            continue
        out.append({"fonte": "eccezione", "articolo": label, "eid": eid, "testo": e["elemento"], "esito": e["esito"],
                    "parte_che_afferma": e.get("parte_che_afferma"), "cita_fatti": list(e.get("cita_fatti") or []),
                    "fatti_cliente": client_facts(e.get("cita_fatti") or [])})
    return out


def rendering() -> Dict[str, Any]:
    """v79.15.3: data «rendering» -- the texts the code prints from the verified analysis (the tesi, §7's entries)."""
    return config().get("rendering") or {}


def _quoted(ids: Sequence[str], texts: Dict[str, str]) -> str:
    """v79.15.3: the cited sentences as printed in a code-built text -- each «as written» (data «rendering.tesi.fatto»,
    its final . ; : out), in the order cited, once, joined by «fatti_sep»; "" when none exists."""
    cfg = rendering().get("tesi") or {}
    out = [str(cfg.get("fatto") or "«{testo}»").format(testo=_stop(texts[i])) for i in dict.fromkeys(ids or [])
           if _stop(texts.get(i))]
    return str(cfg.get("fatti_sep") or "; ").join(out)


def code_tesi(line: Optional[Dict[str, Any]], elements: Dict[str, List[Dict[str, Any]]], inp: Dict[str, Any],
              m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """v79.15.3 (item 1): a verified line (stato VERIFIED, an element in play) with its printed tesi built by the code
    -- data «rendering.tesi»[side][esito], side the client's (inp «side»: attore | difesa | neutro), esito the verified
    verdict of the line's elemento_in_gioco (its E-id among the article's elements; none: «da_provare»); {elemento} the
    schema's text, {art} the line's article, {fatti} the verdict's cited sentences no other party asserts (_quoted;
    reported_by, for every esito: the Procura's allegation is never quoted as the fact that excludes or proves the
    element; none: the line's own such sentences; still none: «nessun_fatto»). The
    model's tesi kept as «tesi_modello» (logged, never printed). Any other line (the frame's charge, the rubrica
    fallback, «da determinare», None) unchanged."""
    if not line or line.get("stato") != VERIFIED or not line.get("eid") or "tesi_modello" in line:
        return line
    cfg = rendering().get("tesi") or {}
    side = inp.get("side") or SIDE_NEUTRAL
    e = next((x for x in elements.get(line["articolo"]) or [] if x.get("eid") == line["eid"]), None) or {}
    esito = e.get("esito") if e.get("esito") in ESITI_ALL else DA_PROVARE
    texts = {i: str(s.get("text") or "") for i, s in m["sents"].items()}
    own = lambda xs: [i for i in xs or [] if i in texts and not reported_by(m["sents"][i], inp)]  # noqa: E731
    fatti = _quoted(own(e.get("cita_fatti")), texts) or _quoted(own(line.get("cita_fatti")), texts) or \
        str(cfg.get("nessun_fatto") or LABEL_NO_FACT)
    template = str(((cfg.get(side) or cfg.get(SIDE_NEUTRAL) or {}).get(esito)) or "«{elemento}»")
    return {**line, "tesi_modello": line.get("tesi") or "", "tesi_codice": True, "esito_in_gioco": esito,
            "tesi": template.format(elemento=line.get("elemento_in_gioco") or "", art=line["articolo"], fatti=fatti)}


def finish(checked: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """The verified analysis from a checked reply, the fallbacks applied in this order: (1) the frame's charge missing
    -> its article the leading line (FRAME_LINE, every element «da_provare»), the model's verified leading line (else
    subordinate) the subordinate one; (2) else no verified leading line -> the rubrica fallback (rubrica_fallback,
    FALLBACK_LINE, every element «da_provare», analysis_line_fallback_rubrica); (3) else «da determinare»
    (UNDETERMINED). The elements of the lines' articles and the law statements verified; the objections (objections_of,
    objections_from_allegations, objections_from_exceptions). Its counters added to checked["counters"]."""
    counters = checked["counters"]
    m = _maps(inp)
    lines = dict(checked["lines"])
    principale, subordinata = lines.get("linea_principale"), lines.get("linea_subordinata")
    charge = inp.get("charge") or ""
    frame_label = ""
    gs_f = governs_sets(inp)
    in_set = (lambda label: not gs_f.get("enforced") or (resolve(label, m["labels"]) or label) in gs_f["pretesa"])
    if checked.get("charge_missing") and charge and in_set(charge):     # v79.17.2 (item 2): the frame's charge leads
        counters["analysis_frame_charge_missing"] = 1                   # only when it is in the enforced set
        counters[SKIPPED_REASON] = SKIP_FRAME_CHARGE        # v79.15.2: the frame's article leads, no rubrica read
        frame_label = resolve(charge, m["labels"]) or charge
        subordinata = principale or subordinata
        principale = {"articolo": frame_label, "elemento_in_gioco": None, "elemento_id": None, "eid": None,
                      "tesi": "", "cita_fatti": [], "cita_norma": [], "stato": FRAME_LINE}
    elif principale is None and governs_sets(inp).get("enforced") and not governs_sets(inp)["pretesa"]:
        counters["analysis_line_undetermined"] = 1          # v79.17.2 (item 1): the set is empty -- the honesty line,
        counters["analysis_binding_empty_honesty"] = 1      # never a fallback pick (the frame's charge, a rubrica)
        counters[SKIPPED_REASON] = SKIP_BINDING_EMPTY
        principale = undetermined_line()
    elif principale is None:
        principale = rubrica_fallback(inp, gs_f["pretesa"] if gs_f.get("enforced") else None)   # v79.17.2 (item 2)
        if principale is not None:
            counters["analysis_line_fallback_rubrica"] = 1
            counters[SKIPPED_REASON] = ""
        else:
            counters["analysis_line_undetermined"] = 1
            counters[SKIPPED_REASON] = fallback_miss(inp)  # v79.15.2: no named charge / no exact rubrica
            principale = undetermined_line()
    else:
        counters[SKIPPED_REASON] = SKIP_LINE_VERIFIED       # v79.15.2: a verified leading line, nothing to fall back on
    if subordinata is not None and principale.get("articolo") and same_article(subordinata["articolo"],
                                                                               principale["articolo"]) \
            and subordinata.get("elemento_in_gioco") == principale.get("elemento_in_gioco"):
        subordinata = None          # the same line twice: once
    items = checked["items"] if "items" in checked else _reply_elements(checked.get("reply") or {})
    elements: Dict[str, List[Dict[str, Any]]] = {}
    open_kinds = {FRAME_LINE: "quadro", FALLBACK_LINE: "rubrica"}
    read = list(dict.fromkeys(x["articolo"] for x in (principale, subordinata) if x and x.get("articolo")
                              and x.get("stato") not in open_kinds))
    for x in (principale, subordinata):
        if not x or not x.get("articolo") or x["articolo"] in elements:
            continue
        elements[x["articolo"]] = _elements_for(x["articolo"], inp, m, items, counters,
                                                all_open=(x.get("stato") in open_kinds),
                                                origine=open_kinds.get(x.get("stato"), "quadro"),
                                                bare_ok=len(read) == 1)
    counters["analysis_bad_eid"] += _bad_eids(_reply_elements(checked.get("reply") or {}), m, read) if read else 0
    principale, subordinata = (code_tesi(x, elements, inp, m) for x in (principale, subordinata))   # v79.15.3 (1)
    objections = objections_of((principale, subordinata), elements, checked.get("raised") or [], inp, m)
    counters["line_table_weaker"] = line_table_weaker(principale, subordinata, elements)    # v79.16.0 (measurement)
    if line_tier(inp)[0] != TIER_ALL:       # v79.16.1 (item 3): the fit tier each final verified line stands on
        for x in (principale, subordinata):  # (v79.16.2: an unjudged norm counts as unverified; nothing with the fit off)
            if x and x.get("stato") == VERIFIED and x.get("articolo") in m["norms"]:
                counters[line_fit_counter(m["norms"][x["articolo"]])] += 1
    counters["objections_from_allegations"] = sum(1 for o in objections if o["fonte"] == "allegazione")
    counters["objections_from_exceptions"] = sum(1 for o in objections if o["fonte"] == "eccezione")
    return {"linea_principale": principale, "linea_subordinata": subordinata, "elementi": elements,
            "affermazioni_di_diritto": _law_statements(checked.get("reply") or {}, m, counters),
            "rimedi": [n["article"] for n in inp.get("norms") or [] if n.get("remedy")],
            "frame_charge": charge, "frame_charge_missing": bool(frame_label), "contractual": inp.get("contractual"),
            "claimant": bool(inp.get("claimant")), "side": inp.get("side") or SIDE_NEUTRAL, "obiezioni": objections,
            "eccezioni": [{"articolo": label, "eid": e.get("eid"), "fonte": e.get("eccezione_fonte")}
                          for label, els in elements.items() for e in els if e.get("eccezione")],
            "rejected": checked.get("rejected") or [], "fallback_skipped_reason": counters.get(SKIPPED_REASON) or "",
            "honesty": HONESTY_LINE if counters.get(SKIPPED_REASON) == SKIP_BINDING_EMPTY else ""}   # v79.17.2 (item 1)


def line_table_weaker(principale: Optional[Dict[str, Any]], subordinata: Optional[Dict[str, Any]],
                      elements: Dict[str, List[Dict[str, Any]]]) -> int:
    """v79.16.0 (measurement, not a rule): 1 when the subordinate line's article holds more «provato» elements than
    the leading line's -- the grid decides whether it should lead."""
    if not principale or not subordinata or not principale.get("articolo") or not subordinata.get("articolo"):
        return 0
    if same_article(principale["articolo"], subordinata["articolo"]):
        return 0
    count = lambda label: sum(1 for e in elements.get(label) or [] if e.get("esito") == PROVATO)   # noqa: E731
    return int(count(subordinata["articolo"]) > count(principale["articolo"]))


def merge_replies(first: Dict[str, Any], again: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """v79.15.1 (review): the re-ask's verification merged onto the first reply's, never thrown away -- each line kind
    the second reply's verified line, else the first reply's verified line (analysis_line_kept_first: the re-ask was
    asked for a defect elsewhere -- a bad subordinate line, the frame's charge -- and its own line of that kind is
    rejected); the element verdicts (finish reads «items») the second reply's first, then the first reply's, each by
    id (the first that names an element wins); charge_missing recomputed on the merged lines; the law statements and
    analysis_bad_eid's verdict count the second reply's (its «reply» kept). `again`'s counters (the caller summed the
    first's into it) carried."""
    lines: Dict[str, Optional[Dict[str, Any]]] = {}
    for kind in LINE_KINDS:
        mine, before = (again.get("lines") or {}).get(kind), (first.get("lines") or {}).get(kind)
        if mine is None and before is not None:
            again["counters"]["analysis_line_kept_first"] += 1
            lines[kind] = before
        else:
            lines[kind] = mine
    charge = inp.get("charge") or ""
    missing = bool(charge) and not any(x and same_article(x["articolo"], charge) for x in lines.values())
    items = _reply_elements(again.get("reply") or {}) + _reply_elements(first.get("reply") or {})
    return {**again, "lines": lines, "charge_missing": missing, "items": items}


# --- the step ---------------------------------------------------------------------------------------------------------

def _line_log(analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for kind in LINE_KINDS:
        x = (analysis or {}).get(kind)
        if x:
            out.append({"kind": kind, "articolo": x.get("articolo"), "eid": x.get("eid"),
                        "elemento_in_gioco": x.get("elemento_in_gioco"), "stato": x.get("stato"),
                        "cita_fatti": x.get("cita_fatti"), "cita_norma": x.get("cita_norma"),
                        "esito_in_gioco": x.get("esito_in_gioco")})
    return out


def tesi_log(analysis: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """v79.15.3 (item 1): the model's tesi per line kind (the «analysis_step» line's tesi_modello=) -- logged only,
    never printed: the printed tesi is the code's (code_tesi)."""
    return {kind: _flat((analysis or {}).get(kind, {}).get("tesi_modello"))
            for kind in LINE_KINDS if (analysis or {}).get(kind) and "tesi_modello" in (analysis or {})[kind]}


def analyse(inp: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
            started: Optional[float] = None) -> Dict[str, Any]:
    """The step on a ready input: at most 2 calls within «deadline_s» from started (default: now); the result
    {"status", "error", "analysis" (v79.16.2: the fallbacks' lines on a failed step too; None with no input),
    "counters", "prompt_hash", "seconds", "calls", "reasons", "rejected"}; logged «analysis_step»."""
    t0 = started if started is not None else time.time()
    deadline = float(config().get("deadline_s") or 40)
    counters = empty_counters()
    counters["candidates_no_elements_excluded"] = len(no_element_articles(inp))     # v79.17.1 (item 7)
    calls: List[Dict[str, Any]] = []
    status, error, analysis, reasons, rejected = NO_INPUT, "", None, [], []
    if inp.get("norms") and inp.get("sentences"):
        messages = build_messages(inp)
        first = call_once(messages, call, deadline - (time.time() - t0))
        calls.append({"attempt": 1, "status": first["status"], "seconds": first["seconds"], "error": first["error"]})
        status, error = first["status"], first["error"]
        if first["status"] == OK:
            checked = verify(first["data"], inp)
            reasons = list(checked["reasons"])
            rejected = [dict(r, attempt=1) for r in checked["rejected"]]
            futile = checked["reasons"] == [r for r in checked["reasons"] if r.startswith("l'addebito")] and \
                resolve(inp.get("charge"), [n["article"] for n in inp["norms"]]) is None
            if checked["reasons"] and not futile:
                second = call_once(reask_messages(messages, first["reply"], checked["reasons"], inp), call,
                                   deadline - (time.time() - t0))
                calls.append({"attempt": 2, "status": second["status"], "seconds": second["seconds"],
                              "error": second["error"]})
                counters["analysis_reasked"] = 1
                if second["status"] == OK:
                    again = verify(second["data"], inp)
                    for k in ("analysis_line_rejected", "analysis_line_exception", "analysis_bad_eid",
                              "analysis_line_procedural_rejected",           # v79.16.1 (item 3): the first reply's too
                              "analysis_line_context_rejected", "analysis_line_quesito_only",     # v79.17.0 (item 2)
                              "analysis_line_outside_rito_rejected",                               # v79.17.1 (item 9)
                              "analysis_line_outside_set_rejected"):                                # v79.17.2 (item 2)
                        again["counters"][k] += checked["counters"][k]
                    again["raised"] = list(checked.get("raised") or []) + [
                        r for r in again.get("raised") or [] if r not in (checked.get("raised") or [])]
                    rejected += [dict(r, attempt=2) for r in again["rejected"]]
                    reasons += [r for r in again["reasons"] if r not in reasons]
                    checked = merge_replies(checked, again, inp)
                else:
                    logger.warning("analysis_step: re-ask %s: %s (the first reply's verification stands)",
                                   second["status"], second["error"])
            analysis = finish(checked, inp)
            analysis["rejected"] = rejected
            for k, v in checked["counters"].items():
                counters[k] = v if k in STRING_COUNTERS else counters.get(k, 0) + v     # v79.15.2: a string kept
            counters["candidates_no_elements_excluded"] = len(no_element_articles(inp))    # counted once, from the input
        else:
            counters["analysis_failed"] = 1
            counters["analysis_timeouts"] = int(first["status"] == TIMEOUT)
            logger.warning("analysis_step: %s: %s (the fallbacks on no reply)", first["status"], first["error"])
            checked = empty_checked(inp)        # v79.16.2 (item 3): the frame's charge / the rubrica fallback / «da
            analysis = finish(checked, inp)     # determinare» run on a failed step too, the reason logged
            analysis["rejected"] = []
            for k, v in checked["counters"].items():
                counters[k] = v if k in STRING_COUNTERS else counters.get(k, 0) + v
            counters["candidates_no_elements_excluded"] = len(no_element_articles(inp))
    if status == NO_INPUT:
        counters[SKIPPED_REASON] = SKIP_NO_INPUT
    counters["analysis_calls"] = len(calls)
    seconds = round(time.time() - t0, 3)
    counters["analysis_seconds"] = seconds
    result = {"status": status, "error": error, "analysis": analysis, "counters": counters,     # v79.16.2: the
              # fallback analysis of a failed step too (None: no input)
              "prompt_hash": prompt_hash(), "seconds": seconds, "calls": calls, "reasons": reasons,
              "rejected": rejected}
    _log("analysis_step", {"prompt_hash": result["prompt_hash"], "status": status,
                           "fallback_skipped_reason": counters.get(SKIPPED_REASON) or "",     # v79.15.2 (item 2)
                           "error": error,
                           "seconds": seconds, "calls": calls, "counters": counters,
                           "lines": _line_log(result["analysis"]), "rejected": rejected,
                           "claimant": bool(inp.get("claimant")),                        # v79.15.1 (item 3)
                           "side": inp.get("side") or SIDE_NEUTRAL,                      # v79.15.3 (item 2)
                           "side_why": inp.get("side_why") or "",
                           "tesi_modello": tesi_log(result["analysis"]),                 # v79.15.3 (item 1)
                           "facts_cited": facts_cited_log(result["analysis"]),           # v79.16.0 (A4): the F-ids
                           "no_pertinence": no_pertinence_log(result["analysis"]),       # v79.16.1 (item 4)
                           "line_tier": line_tier(inp)[0] if inp.get("norms") else "",   # v79.16.1 (item 3)
                           "context_only_no_elements": no_element_articles(inp),        # v79.17.1 (item 7)
                           "outside_rito": outside_rito_articles(inp),                   # v79.17.1 (item 9)
                           "binding": governs_sets(inp) if inp.get("norms") else {},     # v79.17.2 (items 1-2)
                           "honesty": (result["analysis"] or {}).get("honesty") or "",
                           "rito": frame_rito(inp.get("frame")),
                           "exceptions": (result["analysis"] or {}).get("eccezioni") or [],
                           "objections": [{"fonte": o.get("fonte"), "sid": o.get("sid"), "parte": o.get("parte"),
                                           "eids": [e.get("eid") for e in o.get("elementi") or []] or [o.get("eid")]}
                                          for o in (result["analysis"] or {}).get("obiezioni") or []]})
    return result


def run(state: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
        inp: Optional[Dict[str, Any]] = None, started: Optional[float] = None) -> Dict[str, Any]:
    """The step on the flow's state (after applicability_fit): ANALYSIS_STEP off -> {"status": "off"} (nothing read,
    nothing called); else analyse(inputs(state)). Never raises: an input that cannot be read is status failed."""
    if not enabled():
        return {"status": OFF, "error": "", "analysis": None, "counters": {}, "prompt_hash": prompt_hash(),
                "seconds": 0.0, "calls": [], "reasons": [], "rejected": []}
    t0 = started if started is not None else time.time()
    if inp is None:
        try:
            inp = inputs(state)
        except Exception as exc:    # noqa: BLE001 -- the step never fails the answer
            logger.warning("analysis_step: input not read: %s", exc)
            counters = empty_counters()
            counters["analysis_failed"] = 1
            counters[SKIPPED_REASON] = SKIP_STEP_FAILED
            result = {"status": FAILED, "error": f"input: {type(exc).__name__}: {exc}"[:300], "analysis": None,
                      "counters": counters, "prompt_hash": prompt_hash(), "seconds": round(time.time() - t0, 3),
                      "calls": [], "reasons": [], "rejected": []}
            _log("analysis_step", {k: result[k] for k in ("prompt_hash", "status")}
                 | {"fallback_skipped_reason": SKIP_STEP_FAILED}
                 | {k: result[k] for k in ("error", "seconds", "calls", "counters")} | {"lines": [], "rejected": []})
            return result
    return analyse(inp, call=call, started=t0)


# --- build B: rendering with the flag on (brief item 2) ----------------------------------------------------------------
# With ANALYSIS_STEP on, the method's §1, §4's element tables (the Esito column), §6, §9, every «x su y» count and the
# «Risposte ai quesiti» strategy line come from the verified analysis only (view): the code prints the verified lines'
# article, elemento_in_gioco and tesi, the verified verdicts and the cited sentences of the case as they are -- nothing
# else but the labels below. The step not run, failed or answering nothing: the leading line «da determinare» (§1:
# LINE_UNVERIFIED_LABEL), no table, no count -- never the old engine's line in its place.

# my drafts (v79.15.0 build B): the labels -- never a text
LABEL_MAIN = "Linea principale"
LABEL_SUB = "Linea subordinata"
LABEL_ELEMENT = "elemento in gioco"
LABEL_FRAME_LINE = "addebito del quadro, linea non verificata"
LABEL_REMEDIES = "Rimedi e parte generale collegati alla linea"
LABEL_ESITI = "Esiti dell'art. {art}"
LABEL_TO_PROVE = "Da provare"
LABEL_CONTESTED = "Contestati"
LABEL_NO_TABLE = "Nessuna linea verificata: nessuna tabella degli elementi."
LABEL_NO_FACT = "nessun fatto citato"
LABEL_QA_MAIN = "La linea principale è: {line}."
LAW_STATEMENTS_TITLE = "**Affermazioni di diritto verificate**"
ESITO_LABELS = {PROVATO: "provato", CONTESTATO: "contestato", DA_PROVARE: "da provare", ESCLUSO: "escluso"}
CONTESTED_BY = "contestato (allegazione {di_parte})"
TABLE_HEAD = ("| Elemento | Fatti citati | Esito |", "|---|---|---|")
SOLUTIONS_HEAD = ("| Linea | Articolo | Elemento in gioco | Tesi | Fatti citati |", "|---|---|---|---|---|")
# v79.15.1 (item 4): the prose review's counters (ProseReview, prose_counters) -- the rendering's too
PROSE_COUNTERS = ("qa_sentences_replaced", "qa_sentences_dropped_dependent", "prose_excluded_article_dropped",
                  "prose_unfitted_article_dropped", "objections_sentences_dropped")
RENDER_COUNTERS = ("qa_sentences_dropped", "norms_sentences_dropped", "legacy_disagrees_line",
                   "legacy_disagrees_esito") + PROSE_COUNTERS


def no_pertinence_log(analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.16.1 (item 4): the verdicts set «da_provare» for want of a pertinent fact -- the element and the facts."""
    out = []
    for label, els in ((analysis or {}).get("elementi") or {}).items():
        for e in els:
            if e.get("non_pertinente"):
                out.append({"articolo": label, "eid": e.get("eid"), "elemento": _flat(e.get("elemento"))[:120],
                            **e["non_pertinente"]})
    return out


def facts_cited_log(analysis: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """v79.16.0 (A4): {article: {eid: [the record's F-ids the verdict cites]}} -- the log keeps the ids, the page
    prints the facts' text only."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for label, els in ((analysis or {}).get("elementi") or {}).items():
        mine = {str(e.get("eid") or e.get("id")): list(e.get("fatti") or []) for e in els if e.get("fatti")}
        if mine:
            out[label] = mine
    return out


def id_maps(inp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """v79.17.0 (item 4): the step's input as the writer's material and the id renderer read it -- {"periods": {id:
    {article, text, comma}}, "sentences": {id: {text, certainty, fact, said_by}}, "norms": [{article, rubrica, fit,
    fit_verified, governs, remedy}]}; {} with no input."""
    if not isinstance(inp, dict):
        return {}
    periods = {p["id"]: {"article": n["article"], "text": p.get("text"), "comma": p.get("comma")}
               for n in inp.get("norms") or [] for p in n.get("periods") or [] if p.get("id")}
    sentences = {s["id"]: {"text": s.get("text"), "certainty": s.get("certainty"), "fact": s.get("fact"),
                           "said_by": s.get("said_by")} for s in inp.get("sentences") or [] if s.get("id")}
    norms = [{k: n.get(k) for k in ("article", "rubrica", "fit", "fit_verified", "governs", "remedy")}
             for n in inp.get("norms") or []]
    return {"periods": periods, "sentences": sentences, "norms": norms}


def attach_object(v: Dict[str, Any], pretesa: Optional[Dict[str, Any]] = None, pretesa_line: str = "",
                  client: str = "", frame_unverified: Optional[Dict[str, Any]] = None,
                  governs: Optional[Dict[str, Any]] = None, record_facts: Sequence[Dict[str, Any]] = (),
                  record_uncertain: Sequence[Dict[str, str]] = ()) -> Dict[str, Any]:
    """v79.17.0 (item 5): the one verified object every code-rendered section reads from, onto the view --
    v["object"] = {pretesa, client, side, lines, esiti, articles_by_governs, facts, open_items}; and on the view itself
    v["pretesa"], v["pretesa_line"] (§1), v["articles_by_governs"] (§5), v["frame_unverified"] (§10), v["client"]
    (§9's risk line). Returns v."""
    a = v.get("analysis") if isinstance(v.get("analysis"), dict) else {}
    v["pretesa"] = dict(pretesa or {})
    v["pretesa_line"] = str(pretesa_line or "")
    v["client"] = client or ""
    v["frame_unverified"] = {k: list(x or []) for k, x in (frame_unverified or {}).items()}
    g = governs or {}
    v["articles_by_governs"] = {"pretesa": list(g.get("principale") or g.get("pretesa") or []),
                                "quesiti": {k: list(x) for k, x in (g.get("quesiti") or {}).items()}
                                if isinstance(g.get("quesiti"), dict) else {},
                                "context": list(g.get("context_only") or g.get("context") or []),
                                "applied": bool(g.get("applied"))}
    open_items = [{"kind": "elemento", "text": e["question"], "article": e["article"]} for e in open_elements(v)]
    open_items += [{"kind": "quadro", "text": f"{k}: {'; '.join(x)}" if x else k, "field": k}
                   for k, x in v["frame_unverified"].items()]
    open_items += [{"kind": str(i.get("kind") or "record"), "text": _flat(i.get("text"))} for i in record_uncertain or []]
    v["object"] = {"pretesa": v["pretesa"], "client": v["client"], "side": a.get("side") or SIDE_NEUTRAL,
                   "lines": [{"kind": k, **{x: (a.get(k) or {}).get(x) for x in ("articolo", "eid", "elemento_in_gioco",
                                                                               "tesi", "stato")}}
                             for k in LINE_KINDS if a.get(k)],
                   "esiti": {art: [{k: e.get(k) for k in ("eid", "elemento", "esito", "parte_che_afferma", "fatti",
                                                          "cita_fatti")} for e in els]
                             for art, els in (a.get("elementi") or {}).items()},
                   "articles_by_governs": v["articles_by_governs"],
                   "facts": [{k: f.get(k) for k in ("id", "text", "stato", "chi")} for f in record_facts or []
                             if isinstance(f, dict)],
                   "open_items": open_items}
    return v


def undetermined_line() -> Dict[str, Any]:
    return {"articolo": None, "elemento_in_gioco": None, "elemento_id": None, "eid": None, "tesi": "",
            "cita_fatti": [], "cita_norma": [], "stato": UNDETERMINED, "label": LINE_UNDETERMINED,
            "label_sintesi": LINE_UNVERIFIED_LABEL}


def view(result: Optional[Dict[str, Any]], inp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the rendering reads (flag on): {"status", "analysis" (the verified one, or a failed step's fallback line;
    none: the leading line «da determinare», nothing else), "sentences" {S-id: text}, "rubriche" {label: rubrica}, "remedies" [labels of the
    fitted norms in the remedy / general-part set], "statements" (the verified affermazioni_di_diritto)}."""
    result = result or {}
    inp = inp or {}
    analysis = result.get("analysis")       # v79.16.2 (item 3): a failed step's fallback line is read too
    if not isinstance(analysis, dict):
        analysis = {"linea_principale": undetermined_line(), "linea_subordinata": None, "elementi": {},
                    "affermazioni_di_diritto": [], "rimedi": [n["article"] for n in inp.get("norms") or []
                                                              if n.get("remedy")]}
    return {"status": result.get("status") or NO_INPUT, "analysis": analysis,
            "sentences": {s["id"]: s["text"] for s in inp.get("sentences") or []},
            "rubriche": {n["article"]: n.get("rubrica") or "" for n in inp.get("norms") or []},
            "remedies": list(analysis.get("rimedi") or []),
            "statements": list(analysis.get("affermazioni_di_diritto") or [])}


def _lines(v: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    a = v.get("analysis") or {}
    out = [(LABEL_MAIN, a.get("linea_principale") or undetermined_line())]
    if a.get("linea_subordinata"):
        out.append((LABEL_SUB, a["linea_subordinata"]))
    return out


def _stop(text: Any) -> str:
    return _flat(text).rstrip(" .;:")


def line_text(line: Optional[Dict[str, Any]]) -> str:
    """One line as printed: «art. X — <tesi> (elemento in gioco: «…»)»; the frame's article kept by the fallback
    «art. X (addebito del quadro, linea non verificata)»; undetermined «da determinare (linea non verificata)»;
    v79.15.1: the rubrica fallback «art. X — <rubrica>: elementi da provare[ dall'accusa]» (no element in play).
    v79.15.3 (item 1): a verified line's tesi is the code's (code_tesi), which names the element itself -- «art. X —
    <tesi>», the «(elemento in gioco: …)» label no longer repeated after it."""
    x = line or undetermined_line()
    if x.get("stato") == UNDETERMINED or not x.get("articolo"):
        return f"{LINE_UNDETERMINED} ({LINE_UNVERIFIED_LABEL})"
    if x.get("stato") == FRAME_LINE:
        return f"art. {x['articolo']} ({LABEL_FRAME_LINE})"
    out = f"art. {x['articolo']} — {_stop(x.get('tesi'))}"
    if x.get("elemento_in_gioco") and not x.get("tesi_codice"):
        out += f" ({LABEL_ELEMENT}: «{x['elemento_in_gioco']}»)"
    return out


def main_tesi(v: Dict[str, Any]) -> str:
    """The leading line's printed text (the main-line-repeated check reads it)."""
    return line_text((v.get("analysis") or {}).get("linea_principale"))


def summary_text(v: Dict[str, Any]) -> str:
    """§1: the leading line and the subordinate one, labels and the verified text only. v79.17.0 (item 5): the pretesa
    first (v["pretesa_line"]: case_frame.pretesa_line -- a derived one with its banner), then the lines."""
    head = [str(v.get("pretesa_line") or "")] if v.get("pretesa_line") else []
    a = v.get("analysis") if isinstance(v.get("analysis"), dict) else {}
    if a.get("honesty"):                    # v79.17.2 (item 1): the set is empty -- said before the lines
        head.append(str(a["honesty"]))
    return "\n\n".join(head + [f"{label}: {line_text(x)}." for label, x in _lines(v)])


def esito_text(e: Dict[str, Any]) -> str:
    if e.get("esito") == CONTESTATO and e.get("parte_che_afferma"):
        from . import issue_records
        return CONTESTED_BY.format(di_parte=issue_records.of_party(e["parte_che_afferma"]))
    return ESITO_LABELS.get(e.get("esito"), ESITO_LABELS[DA_PROVARE])


def _cell(text: Any) -> str:
    return _flat(text).replace("|", "/")


def _facts_cell(ids: Sequence[str], v: Dict[str, Any]) -> str:
    texts = [v["sentences"][i] for i in ids or [] if i in v.get("sentences", {})]
    return " / ".join(texts) if texts else LABEL_NO_FACT


def line_articles(v: Dict[str, Any]) -> List[str]:
    """The lines' articles, once each, the leading one first (none for an undetermined line)."""
    out: List[str] = []
    for _, x in _lines(v):
        if x.get("articolo") and x["articolo"] not in out:
            out.append(x["articolo"])
    return out


def counts(v: Dict[str, Any], label: str) -> Dict[str, int]:
    """{esito: n, "totale": n} of the article's verified verdicts -- the only «x su y» the answer prints."""
    els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
    out = {k: sum(1 for e in els if e.get("esito") == k) for k in ESITO_LABELS}
    out["totale"] = len(els)
    return out


def counts_text(v: Dict[str, Any], label: str) -> str:
    c = counts(v, label)
    if not c["totale"]:
        return ""
    return LABEL_ESITI.format(art=label) + ": " + "; ".join(f"{ESITO_LABELS[k]} {c[k]} su {c['totale']}"
                                                            for k in ESITO_LABELS) + "."


def norms_tables(v: Dict[str, Any], extra_remedies: Sequence[str] = ()) -> str:
    """§4's «Analisi per elementi»: one table per line article (Elemento | Fatti citati | Esito, the verified verdicts),
    its count; then the remedy / general-part articles attached to the line -- named, no table, no count."""
    parts: List[str] = []
    for label in line_articles(v):
        els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
        if not els:
            continue
        rubrica = _flat((v.get("rubriche") or {}).get(label))
        rows = [f"**Art. {label}**" + (f" — {rubrica}" if rubrica else ""), "", *TABLE_HEAD]
        rows += [f"| {_cell(e['elemento'])} | {_cell(_facts_cell(e.get('cita_fatti'), v))} | {_cell(esito_text(e))} |"
                 for e in els]
        parts.append("\n".join(rows))
        parts.append(counts_text(v, label))
    if not parts:
        parts.append(LABEL_NO_TABLE)
    remedies = list(dict.fromkeys(list(v.get("remedies") or []) + [x for x in extra_remedies or [] if is_remedy(x)]))
    remedies = [x for x in remedies if not any(same_article(x, y) for y in line_articles(v))]
    if remedies:
        named = [f"art. {x}" + (f" ({_flat(v['rubriche'][x])})" if _flat((v.get("rubriche") or {}).get(x)) else "")
                 for x in remedies]
        parts.append(f"**{LABEL_REMEDIES}**: " + "; ".join(named) + ".")
    return "\n\n".join(p for p in parts if p)


def solutions_text(v: Dict[str, Any]) -> str:
    """§6: the verified lines only (an undetermined leading line said as such) -- never a containment line or any
    other the analysis did not return."""
    rows = list(SOLUTIONS_HEAD)
    for label, x in _lines(v):
        if x.get("stato") == FALLBACK_LINE:          # v79.15.1 (item 2): the rubrica fallback -- no element, no fact
            rows.append(f"| {label} | art. {_cell(x['articolo'])} | — | {_cell(_stop(x.get('tesi')))} | — |")
        elif x.get("stato") == VERIFIED:
            rows.append(f"| {label} | art. {_cell(x['articolo'])} | {_cell(x.get('elemento_in_gioco'))} | "
                        f"{_cell(_stop(x.get('tesi')))} | {_cell(_facts_cell(x.get('cita_fatti'), v))} |")
        else:
            rows.append(f"| {label} | {_cell(line_text(x))} | — | — | — |")
    return "\n".join(rows)


def conclusions_text(v: Dict[str, Any]) -> str:
    """§9: the §4 counts first (v79.16.0, D2: the leading article's «Esiti dell'art. X» opens the section), the lines,
    the leading article's open and contested elements -- verified, labels only."""
    out: List[str] = []
    for label in line_articles(v)[:1]:
        c = counts_text(v, label)
        if c:
            head, _, rest = c.partition(": ")
            out.append(f"**{head}**: {rest}")
    out += [f"**{label}**: {line_text(x)}." for label, x in _lines(v)]
    risk = risk_word(v)             # v79.17.0 (item 5): the esiti -> the risk in words, computed, never written
    if risk:
        out.append(RISK_LINE.format(client=v.get("client") or "la parte assistita", parola=RISK_WORDS[risk]))
    for label in line_articles(v)[:1]:
        els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
        open_ = [f"«{e['elemento']}»" for e in els if e.get("esito") == DA_PROVARE]
        contested = [f"«{e['elemento']}» ({esito_text(e)})" for e in els if e.get("esito") == CONTESTATO]
        if open_:
            out.append(f"**{LABEL_TO_PROVE}**: " + ", ".join(open_) + ".")
        if contested:
            out.append(f"**{LABEL_CONTESTED}**: " + ", ".join(contested) + ".")
    return "\n".join(out)


def risk_word(v: Dict[str, Any]) -> str:
    """v79.17.0 (item 5): favorevole | incerto | sfavorevole for the client, from the leading line's elements -- for
    the pretesa's holder: every element provato -> fondata (favorevole), an element escluso or none provato with one
    contestato or da provare -> infondata (sfavorevole), else incerto; the client resisting the pretesa (side difesa)
    reads it the other way round. "" with no verified leading article or no element."""
    a = v.get("analysis") or {}
    labels = line_articles(v)
    if not labels or (a.get("linea_principale") or {}).get("stato") != VERIFIED:
        return ""
    els = (a.get("elementi") or {}).get(labels[0]) or []
    if not els:
        return ""
    n = {k: sum(1 for e in els if e.get("esito") == k) for k in (PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO)}
    if n[ESCLUSO]:
        holder = "sfavorevole"
    elif n[PROVATO] == len(els):
        holder = "favorevole"
    elif n[PROVATO] == 0:
        holder = "sfavorevole"
    else:
        holder = "incerto"
    if (a.get("side") or "") == SIDE_DEFENCE:
        holder = {"favorevole": "sfavorevole", "sfavorevole": "favorevole"}.get(holder, holder)
    return holder


# --- v79.16.0: §2 and §10 from the verified elements (A3), the bridges (D2), the context block (D1) ---------------------
# my drafts: labels and one-sentence bridges built from verified data only -- never a model text
OPEN_QUESTION = "se «{elemento}» (art. {art}) risulti provato"
CONTESTED_QUESTION = "se «{elemento}» (art. {art}) sia provato oltre l'allegazione{di_parte}"
QUESTION_BRIDGE = "La questione è se «{elemento}» dell'art. {art} sia {esito}, posto che {fatto}."
# v79.17.0 (item 5): §3 with the pretesa -- «La questione è se <pretesa> sia fondata sull'art. X, posto che …»
PRETESA_BRIDGE = "La questione è se {tipo} «{contenuto}» sia fondata sull'art. {art}, posto che {fatto}."
TIPO_ARTICLES = {"accusa": "l'accusa", "domanda": "la domanda", "ricorso": "il ricorso", "impugnazione": "l'impugnazione",
                 "opposizione": "l'opposizione", "istanza": "l'istanza"}
RISK_LINE = "**Rischio per {client}**: {parola}."
RISK_WORDS = {"favorevole": "favorevole", "incerto": "incerto", "sfavorevole": "sfavorevole"}
FRAME_UNVERIFIED_POINT = "Campo del quadro non verificato: {field} ({why})."
RULINGS_BRIDGE = "Pronunce sulle norme della linea: {articles}."
SOLUTIONS_BRIDGE = "Le linee seguono gli esiti del punto 4."
CONTEXT_TITLE = "CONTESTO VERIFICATO (calcolato dal codice: non contraddirlo, non riscriverlo)"
UNCERTAIN_LABELS = {"ipotesi": "Fatto esposto come ipotesi", "non_letto": "Passaggio non letto dal modello, ripreso dalla regola"}


def open_elements(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The elements of the verified lines' articles held «da_provare» or «contestato», in table order -- each with
    the question it opens: [{"elemento", "article", "esito", "parte", "question"}]. §2's «Fatti mancanti» and §10's
    points read them: the list is empty only when no such element exists."""
    out: List[Dict[str, Any]] = []
    for label in line_articles(v):
        for e in ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []:
            if e.get("esito") == DA_PROVARE:
                q = OPEN_QUESTION.format(elemento=_flat(e.get("elemento")), art=label)
            elif e.get("esito") == CONTESTATO:
                from . import issue_records
                parte = e.get("parte_che_afferma")
                q = CONTESTED_QUESTION.format(elemento=_flat(e.get("elemento")), art=label,
                                              di_parte=f" {issue_records.of_party(parte)}" if parte else "")
            else:
                continue
            out.append({"elemento": _flat(e.get("elemento")), "article": label, "esito": e["esito"],
                        "parte": e.get("parte_che_afferma"), "question": q, "eid": e.get("eid")})
    return out


def record_facts_split(facts: Sequence[Dict[str, Any]], v: Dict[str, Any], narrative=None,
                       limits: Optional[Dict[str, int]] = None) -> Dict[str, List[Any]]:
    """§2 from the record (A3): certi = the facts stated accertato; ipotizzabili = those stated ipotizzato («ipotesi»)
    or allegato («allegazione di <chi>», «allegazione» with no speaker); mancanti = the verified lines' open and
    contested elements (open_elements), each as the question it opens -- never «nessun elemento resta da provare»
    while the tables hold one. narrative(text) -> bool: a caller's filter (lawyer_method.narrative), none: all."""
    from . import issue_records
    certi: List[str] = []
    ipotesi: List[Dict[str, str]] = []
    for f in facts or []:
        text = _flat(f.get("text"))
        if not text or (narrative is not None and not narrative(text)):
            continue
        stato = str(f.get("stato") or "")
        if stato == ACCERTATO_:
            certi.append(text)
        elif stato == ALLEGATO_:
            who = str(f.get("chi") or "")
            ipotesi.append({"text": text, "how": f"allegazione {issue_records.of_party(who)}" if who else "allegazione"})
        else:
            ipotesi.append({"text": text, "how": "ipotesi"})
    lim = limits or {}
    mancanti = [{"fatto": f"«{e['elemento']}» (art. {e['article']})", "apre": e["question"],
                 "esito": ESITO_LABELS[e["esito"]] if e["esito"] in ESITO_LABELS else e["esito"],
                 "elemento": e.get("eid"), "article": e["article"]} for e in open_elements(v)]
    return {"certi": certi[:lim.get("certi") or len(certi)], "ipotizzabili": ipotesi[:lim.get("ipotizzabili") or len(ipotesi)],
            "mancanti": mancanti}


def verify_points(v: Dict[str, Any], uncertain: Sequence[Dict[str, str]] = ()) -> List[str]:
    """§10 (A3): the open elements' questions (open_elements), then the record's uncertain items (fact_record.
    uncertain_items: a fact stated as a hypothesis, a stretch the model left out) -- code-written, no prose restated."""
    out = [f"Da verificare {e['question']}." for e in open_elements(v)]
    for field, why in (v.get("frame_unverified") or {}).items():      # v79.17.0 (item 5): the frame's dropped fields
        out.append(FRAME_UNVERIFIED_POINT.format(field=field, why="; ".join(str(x) for x in why or []) or "non verificato"))
    for item in uncertain or []:
        label = UNCERTAIN_LABELS.get(str(item.get("kind") or ""), "Da verificare")
        out.append(f"{label}: «{_flat(item.get('text'))}».")
    return out


def _esito_of_line(v: Dict[str, Any], line: Dict[str, Any]) -> str:
    els = ((v.get("analysis") or {}).get("elementi") or {}).get(line.get("articolo") or "") or []
    for e in els:
        if (line.get("eid") and e.get("eid") == line.get("eid")) or \
                (line.get("elemento_id") and e.get("id") == line.get("elemento_id")) or \
                verbatim_key(e.get("elemento")) == verbatim_key(line.get("elemento_in_gioco")):
            return esito_text(e)
    return ESITO_LABELS[DA_PROVARE]


def question_bridge(v: Dict[str, Any]) -> str:
    """§3's opening sentence (D2): the leading line's element in play, its article, its verified esito and the first
    fact it cites; "" when the line is not verified."""
    line = (v.get("analysis") or {}).get("linea_principale") or {}
    if line.get("stato") != VERIFIED or not line.get("articolo") or not line.get("elemento_in_gioco"):
        return ""
    facts = [v.get("sentences", {}).get(i) for i in line.get("cita_fatti") or [] if v.get("sentences", {}).get(i)]
    fatto = _stop(facts[0]) if facts else "i fatti narrati"
    p = v.get("pretesa") or {}
    what = _flat(p.get("contenuto")) or _flat(p.get("frase"))
    if what:                        # v79.17.0 (item 5): the pretesa's own question
        from .verified_material import unquoted
        return PRETESA_BRIDGE.format(tipo=TIPO_ARTICLES.get(p.get("tipo") or "domanda", f"la {p.get('tipo')}"),
                                     contenuto=unquoted(what), art=line["articolo"], fatto=fatto)
    return QUESTION_BRIDGE.format(elemento=_flat(line["elemento_in_gioco"]), art=line["articolo"],
                                  esito=_esito_of_line(v, line), fatto=fatto)


def rulings_bridge(v: Dict[str, Any]) -> str:
    """§5's opening sentence (D2): the lines' articles with their rubriche; "" with no verified line article."""
    labels = line_articles(v)
    if not labels:
        return ""
    named = [f"art. {x}" + (f" ({_flat(v['rubriche'][x])})" if _flat((v.get("rubriche") or {}).get(x)) else "")
             for x in labels]
    return RULINGS_BRIDGE.format(articles=", ".join(named))


def allowed_ruling_articles(v: Dict[str, Any], quesiti: Sequence[str] = ()) -> List[str]:
    """D3: the articles §5 may cite rulings on -- the verified lines', the quesiti's, the remedies attached. v79.17.0
    (item 5): with the fit's governs (v["articles_by_governs"]) the quesiti's are the articles whose verdict governs the
    pretesa or a quesito -- never a context-only one."""
    g = v.get("articles_by_governs") or {}
    if g.get("applied") or g.get("pretesa") or g.get("quesiti"):
        tagged = list(g.get("pretesa") or []) + [x for labels in (g.get("quesiti") or {}).values() for x in labels]
        return list(dict.fromkeys(list(line_articles(v)) + tagged + list(v.get("remedies") or [])))
    return list(dict.fromkeys(list(line_articles(v)) + [str(x) for x in quesiti or []] + list(v.get("remedies") or [])))


def context_block(v: Optional[Dict[str, Any]], client: str = "", quesiti: Optional[Dict[str, Sequence[str]]] = None,
                  facts: Sequence[Dict[str, Any]] = ()) -> str:
    """D1: the one context block every writer call receives -- the client and its side, the lines with their
    elements and esiti, the quesiti's articles, the record's facts (id, stato, chi, text). "" with no verified view."""
    if not v or not isinstance(v.get("analysis"), dict):
        return ""
    a = v["analysis"]
    side = {SIDE_CLAIMANT: "parte che agisce", SIDE_DEFENCE: "parte che resiste"}.get(a.get("side") or "", "")
    lines = [CONTEXT_TITLE, f"Parte assistita: {client or 'non indicata'}" + (f" ({side})" if side else "") + "."]
    for label, x in _lines(v):
        lines.append(f"{label}: {line_text(x)}.")
        els = (a.get("elementi") or {}).get(x.get("articolo") or "") or []
        if els:
            lines.append("  Elementi: " + "; ".join(f"«{_flat(e.get('elemento'))}» {esito_text(e)}" for e in els) + ".")
    for q, labels in (quesiti or {}).items():
        if labels:
            lines.append(f"Articoli del quesito «{_flat(q)[:120]}»: " + ", ".join(f"art. {x}" for x in labels) + ".")
    if facts:
        lines.append("Fatti del caso (record):")
        for f in facts:
            tag = str(f.get("stato") or "")
            if f.get("chi"):
                tag += f", {f['chi']}"
            lines.append(f"  {f.get('id') or ''} [{tag}] {_flat(f.get('text'))}")
    return "\n".join(lines)


# --- v79.16.0 (D3, D4): one coherent opinion -- no sentence twice, no prose on a remedy article ----------------------------
_DEDUP_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü«\"(])")
# v79.16.2: no sentence ends on a code or article abbreviation («art. 646 c.p. (addebito del quadro, …)» is one sentence:
# 79.16.0 split it there and dropped §1's label as the Q&A line's duplicate)
_DEDUP_ABBREV = re.compile(r"(?:\b(?:artt?|n|c|p|cc|cp|cpc|cpp|lgs|d|l|comma|co)\.|\b[a-z]\.)$", re.IGNORECASE)
DEDUP_MIN_CHARS = 40


def _dedup_sentences(body: str) -> List[str]:
    """The line's sentences for dedup_sections: split at a sentence end followed by a capital, a quote or a bracket,
    never after an abbreviation (_DEDUP_ABBREV)."""
    out, start = [], 0
    for m in _DEDUP_SPLIT.finditer(body):
        if _DEDUP_ABBREV.search(body[start:m.start()].rstrip()):
            continue
        out.append(body[start:m.start()])
        start = m.end()
    out.append(body[start:])
    return out


def _dedup_key(sentence: str) -> str:
    text = re.sub(r"\*\*|[«»\"“”']", "", str(sentence or "")).lower()
    return " ".join(re.findall(r"[a-zà-ü0-9]+", text))


def dedup_sections(parts: Sequence[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], int]:
    """D3: the sections in print order, each prose sentence printed once -- one already printed in an earlier section
    is dropped from the later one (its line with it when nothing of it is left). Table rows, headings and bold labels
    («**Linea principale**: …») are not prose: never touched. (the sections, how many sentences dropped)."""
    seen: set = set()
    out: List[Tuple[str, str]] = []
    dropped = 0
    for title, text in parts:
        lines_out: List[str] = []
        for line in str(text or "").split("\n"):
            raw = line.strip()
            if not raw or raw.startswith("|") or raw.startswith("**") or raw.startswith("#"):
                lines_out.append(line)
                continue
            lead = re.match(r"^(\s*(?:[-•*]|\d{1,2}\.)\s+)", line)
            prefix = lead.group(1) if lead else ""
            body = line[len(prefix):]
            kept: List[str] = []
            for sentence in _dedup_sentences(body):
                key = _dedup_key(sentence)
                if len(key) >= DEDUP_MIN_CHARS and key in seen:
                    dropped += 1
                    continue
                if len(key) >= DEDUP_MIN_CHARS:
                    seen.add(key)
                kept.append(sentence)
            if kept:
                lines_out.append(prefix + " ".join(x.strip() for x in kept))
        out.append((title, "\n".join(lines_out)))
    return out, dropped


def remedy_prose_out(text: str, v: Dict[str, Any], log: Optional[List[str]] = None) -> str:
    """D4: §4's written paragraphs whose every cited article is a remedy / general-part one (none a line article) are
    dropped -- such articles appear in §4 only as the «Rimedi e parte generale collegati» line."""
    remedies = list(v.get("remedies") or [])
    lines = line_articles(v)
    kept: List[str] = []
    for par in str(text or "").split("\n\n"):
        cited = _cited(par)
        only_remedy = bool(cited) and all((_among(c, remedies) or is_remedy(c, bool((v.get("analysis") or {}).get("contractual"))))
                                          and not _among(c, lines) for c in cited)
        if only_remedy and not par.strip().startswith(("|", "**")):     # a table or a label line is not prose
            if log is not None:
                log.append(_flat(par)[:160])
            continue
        kept.append(par)
    return "\n\n".join(kept)


def answer_strategy(v: Dict[str, Any]) -> List[str]:
    """The «Risposte ai quesiti» strategy sentence: the verified leading line (labels only)."""
    return [LABEL_QA_MAIN.format(line=line_text((v.get("analysis") or {}).get("linea_principale")))]


def statements_block(v: Dict[str, Any]) -> str:
    """The verified affermazioni_di_diritto, each with its articles -- printed in «Risposte ai quesiti»."""
    items = [f"- {_stop(s['testo'])} (art. {', '.join(s.get('articoli') or [])})." for s in v.get("statements") or []
             if _flat(s.get("testo"))]
    return f"{LAW_STATEMENTS_TITLE}\n\n" + "\n".join(items) if items else ""


# --- v79.15.1 (item 3): §7 from the verified analysis; v79.15.3 (item 3): one plain form for every entry ---------------
# Each entry: «Obiezione N.» the allegation's sentence as written (or the exception element), «Risposta.» each element's
# verified status in words, then the client's own cited sentences. The texts are data «rendering.obiezioni» (the
# 79.15.1 labels «Allegazione …: «…» — elementi contestati: …» / «Esito verificato: … Fatti del cliente: …» are gone).


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def status_words(e: Dict[str, Any], side: str) -> str:
    """v79.15.3 (item 3): one element's verified status in words -- data «rendering.obiezioni.stato»[esito] (for a
    claimant «stato_attore» first: «è da provare» for an open or contested element), {elemento} its schema text,
    {art} its article."""
    cfg = rendering().get("obiezioni") or {}
    esito = e.get("esito") if e.get("esito") in ESITI_ALL else DA_PROVARE
    words = ((cfg.get("stato_attore") or {}).get(esito) if side == SIDE_CLAIMANT else None) or \
        (cfg.get("stato") or {}).get(esito) or "«{elemento}»"
    return str(words).format(elemento=_flat(e.get("testo") or e.get("elemento")), art=e.get("articolo") or "")


def objection_items(v: Dict[str, Any]) -> List[Dict[str, str]]:
    """§7's objections from the verified analysis (its «obiezioni»: finish / objections_of), each {"obiezione",
    "risposta", "fonte"}. v79.15.3 (item 3), one form for every entry: the objection -- an allegation's sentence as
    written («rendering.obiezioni.allegazione»; «allegazione_attribuita», «Allegazione <di parte>: «…»», only when the
    sentence names no speaker itself), an exception element «Eccezione dell'art. X: «…»»; the answer -- each element's
    status in words (status_words, joined by «stati_sep»), then «: » and the client's own cited sentences as written
    («risposta_con_fatti»; none: «risposta_senza_fatti»), a capital and a full stop. None when the step is not ok."""
    from . import issue_records
    cfg = rendering().get("obiezioni") or {}
    an = v.get("analysis") or {}
    side = an.get("side") or (SIDE_CLAIMANT if an.get("claimant") else SIDE_NEUTRAL)
    texts = {k: str(x or "") for k, x in (v.get("sentences") or {}).items()}
    out = []
    for o in an.get("obiezioni") or []:
        if o.get("fonte") == "allegazione":
            frase = texts.get(o.get("sid"))
            if not frase:
                continue
            key = "allegazione" if o.get("nominata", True) else "allegazione_attribuita"
            said = str(cfg.get(key) or "{frase}").format(
                frase=_stop(frase), di_parte=issue_records.of_party(o.get("parte") or ""))
            els = o.get("elementi") or []
        else:
            said = str(cfg.get("eccezione") or "«{elemento}»").format(art=o.get("articolo"), elemento=o.get("testo"))
            els = [o]
        stati = str(cfg.get("stati_sep") or "; ").join(status_words(e, side) for e in els)
        facts = _quoted(o.get("fatti_cliente") or [], texts)
        answer = str(cfg.get("risposta_con_fatti") or "{stati}: {fatti}").format(stati=stati, fatti=facts) if facts \
            else str(cfg.get("risposta_senza_fatti") or "{stati}").format(stati=stati)
        out.append({"obiezione": _stop(said), "risposta": _cap(_stop(answer)) + ".", "fonte": o.get("fonte") or ""})
    return out


# --- the Q&A prose: a statement of what a norm provides with no verified counterpart, removed ---------------------------

@lru_cache(maxsize=1)
def _qa_verb_re() -> "re.Pattern[str]":
    verbs = "|".join(f"(?:{x})" for x in (config().get("qa_statements") or {}).get("verbs") or [])
    return re.compile(rf"(?<![\w'’])(?:{verbs})(?!\w)", re.IGNORECASE)


def tokens(text: Any) -> set:
    """A sentence's tokens for the Q&A match: its words (letters only) of «token_min_len» letters or more, lowercased,
    the data's «stopwords» out."""
    cfg = config().get("qa_statements") or {}
    n = int(cfg.get("token_min_len") or 3)
    stop = {str(w).lower() for w in cfg.get("stopwords") or []}
    return {w for w in re.findall(r"[^\W\d_]+", str(text or "").lower()) if len(w) >= n and w not in stop}


def overlap(sentence: Any, statement: Any) -> float:
    """|tokens(sentence) ∩ tokens(statement)| / |tokens(sentence)| -- the denominator the Q&A sentence's tokens (0.0 for
    a sentence with none)."""
    s = tokens(sentence)
    return len(s & tokens(statement)) / len(s) if s else 0.0


def _numbers(refs: Sequence[Tuple[str, str]]) -> set:
    """v79.15.2: the cited articles as art_parts reads them, (number, code) -- («62bis», «cp»); a citation with no
    code («l'art. 1439») -> (number, "")."""
    return {art_parts(f"{n} {c}".strip()) for n, c in refs or [] if n} - {("", "")}


def _cites_same(mine: set, theirs: set) -> bool:
    """v79.15.2 (review): one article cited by both -- the same number and the same code (art_key: «640 c.p.p.» and
    «640 c.c.» are never «640 c.p.»); a citation by number alone («art. 640», no code) matches that number in any
    code."""
    return any(n == tn and (not c or not tc or c == tc) for n, c in mine for tn, tc in theirs)


@lru_cache(maxsize=1)
def _qa_referent_re() -> "re.Pattern[str]":
    words = "|".join(f"(?:{x})" for x in (config().get("qa_statements") or {}).get("referents") or []) or r"(?!x)x"
    return re.compile(rf"(?<![\w'’])(?:{words})(?!\w)", re.IGNORECASE)


def norm_statement(sentence: str) -> Optional[List[Tuple[str, str]]]:
    """A sentence stating what a norm provides: it holds one of the data's «verbs» (the verbs and phrasings of a rule:
    «prevede», «punisce», «commina», «deve», «la pena», «reclusione», …) and either cites an article -> the articles it
    cites, or names a norm without a number (data «referents»: «la norma», «la disposizione», «l'articolo», …) -> []
    (a statement on whichever article); None when it is not one."""
    from .citations import article_refs
    if not _qa_verb_re().search(sentence or ""):
        return None
    refs = article_refs(sentence)
    if refs:
        return refs
    return [] if _qa_referent_re().search(sentence or "") else None


def counterpart(sentence: str, v: Dict[str, Any], labels: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """The verified affermazione_di_diritto the sentence matches: overlap() at least the data's «overlap» (0.6) and,
    when the sentence cites articles, one of the statement's articles' numbers among them (a sentence naming «la
    norma» without a number: any verified statement); None. v79.15.1 (review): `labels` the articles the sentence
    cites by number alone (bare_cited), read as its articles when citations.article_refs finds none. v79.15.2
    (review): the articles compared by number AND code (_cites_same) -- a sentence on «640 c.p.p.» never matches a
    statement on «640 c.p.»; only a citation with no code matches by number."""
    refs = norm_statement(sentence)
    mine = _numbers(refs or []) or {art_parts(x) for x in labels or []} - {("", "")}
    need = float((config().get("qa_statements") or {}).get("overlap") or 0.6)
    for s in v.get("statements") or []:
        theirs = {art_parts(a) for a in s.get("articoli") or []} - {("", "")}
        if (not mine or _cites_same(mine, theirs)) and overlap(sentence, s.get("testo")) >= need:
            return s
    return None


def qa_filter(sentences: Sequence[str], v: Dict[str, Any], log: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """The Q&A sentences kept: a statement of what a norm provides (norm_statement) with no verified counterpart is
    removed (into log: the sentence, its articles, the best overlap)."""
    out = []
    for x in sentences or []:
        refs = norm_statement(x)
        if refs is not None and counterpart(x, v) is None:
            if log is not None:
                best = max([overlap(x, s.get("testo")) for s in v.get("statements") or []] or [0.0])
                log.append({"testo": _flat(x)[:200], "articoli": [f"{n} {c}".strip() for n, c in refs],
                            "overlap": round(best, 2)})
            continue
        out.append(x)
    return out


# --- v79.15.1 (item 4): the prose reviewed -- a rule replaced by its verified statement, never a fragment left ------
# The Q&A («risposte»), §4's written prose («norme») and §7's model-written objections and answers («obiezioni»), with
# the flag on, sentence by sentence and in reading order (ProseReview, one per text: an answer, a §4 section, an
# objection, its answer). Each sentence, in this order:
#   1. it cites (posture.sentence_articles -- the parser 79.14.3's cites_excluded reads; v79.15.1 review: plus an article
#      cited by number alone, bare_cited, resolved by number in the scope) an article the binding excluded
#      or the fit removed (lawyer_method.excluded_labels) -> dropped (ARTICLE_EXCLUDED, prose_excluded_article_dropped);
#   2. it cites an article outside the fitted set -- the analysis input's norms (the records left after the fit, the
#      binding's excluded out by 1.) and the remedies §4 names (the fitted remedy-set norms, the circumstances in the
#      remedy set) -- that the user did not name (the case's sentences, the questions; the frame's verified charge) ->
#      dropped (ARTICLE_UNFITTED, prose_unfitted_article_dropped). A procedural code's article (data
#      «prose_review.exempt_codes») is never outside: the fit judges no procedural norm. No input (the step not run on a
#      state): no fitted set, the check not made;
#   3. the sentence before it was dropped (by any rule here) and it opens with an anaphoric connective (data
#      «prose_review.anaphoric_openers») -> dropped (DEPENDENT, qa_sentences_dropped_dependent) -- and the next one
#      likewise, while each opens so;
#   4. a statement of what a norm provides (norm_statement) with no verified counterpart (counterpart) -> replaced by
#      the first verified affermazione_di_diritto on the same article (the articles it cites; one citing none, in §4,
#      the articles of the block's heading), its text verbatim, «(art. X)» added when the text names none of its
#      articles (REPLACED, qa_sentences_replaced; a statement printed once per text: a second sentence it replaces
#      leaves nothing, and its anaphoric follower is dropped as after a drop); none -- or §7's objection itself
#      (replace=False: a statement of law never stands for the counterparty's objection) -> dropped (DROPPED: per site,
#      qa_sentences_dropped / norms_sentences_dropped / objections_sentences_dropped);
#   5. else kept.
# In §4, a bold heading whose cited articles are all excluded or outside drops its block to the next heading (each
# sentence counted under that heading's reason). Each sentence counted once, under the first rule that holds.

SITE_QA, SITE_NORMS, SITE_OBJECTIONS = "risposte", "norme", "obiezioni"
REPLACED, DROPPED, DEPENDENT = "sostituita", "eliminata", "dipendente"
ARTICLE_EXCLUDED, ARTICLE_UNFITTED = "articolo_escluso", "articolo_non_applicato"
_SITE_DROPPED = {SITE_QA: "qa_sentences_dropped", SITE_NORMS: "norms_sentences_dropped",
                 SITE_OBJECTIONS: "objections_sentences_dropped"}
_KIND_COUNTER = {REPLACED: "qa_sentences_replaced", DEPENDENT: "qa_sentences_dropped_dependent",
                 ARTICLE_EXCLUDED: "prose_excluded_article_dropped", ARTICLE_UNFITTED: "prose_unfitted_article_dropped"}
_LEAD_RE = re.compile(r"^[\s\-*•«»\"“”'(\[]+")


@lru_cache(maxsize=1)
def _bare_cfg() -> Dict[str, Any]:
    cfg = (config().get("prose_review") or {}).get("bare_article") or {}
    suffix = "|".join(cfg.get("suffixes") or ["bis", "ter"])
    num = rf"\d{{1,4}}(?:\s*-?\s*(?:{suffix}))?(?![\w/])"
    ordinals = "|".join(cfg.get("comma_ordinals") or ["primo"])
    clause = rf"\s*,?\s*(?:(?:{ordinals})\s+comma|comma\s+{num}|co\.\s*\d+)\s*,?"
    source = "|".join(cfg.get("source_words") or [r"del\b"])
    return {"suffix": suffix, "num": re.compile(num, re.IGNORECASE),
            "cite": re.compile(rf"(?<![\w])art(?:t?\.|icol[oi])\s*({num})((?:\s*(?:,|\be\b|\bed\b)\s*{num})*)",
                               re.IGNORECASE),
            "clause": re.compile(rf"^{clause}", re.IGNORECASE),
            "inner": re.compile(rf"(\d(?:\s*-?\s*(?:{suffix}))?){clause}\s*(?=c\.|cod\.|codice)", re.IGNORECASE),
            "source": re.compile(rf"^\s*(?:{source})", re.IGNORECASE)}


def _cited(text: Any) -> List[str]:
    """The articles a sentence cites (posture.sentence_articles); v79.15.1 (review): a comma clause between the number
    and its code («art. 1385 secondo comma c.c.») read as the article («1385 c.c.»)."""
    from . import posture
    raw = _bare_cfg()["inner"].sub(r"\1 ", str(text or ""))
    try:
        return posture.sentence_articles(raw)
    except Exception:           # noqa: BLE001 -- an unreadable sentence cites nothing
        return []


def _bare_number(raw: str) -> str:
    return re.sub(rf"\s*-?\s*({_bare_cfg()['suffix']})$", r"-\1", raw.strip().lower(), flags=re.IGNORECASE)


def bare_cited(text: Any, cited: Sequence[str], scope: Dict[str, Any]) -> List[str]:
    """v79.15.1 (review): the articles a sentence cites by number alone («l'art. 1439», «gli artt. 1439 e 1440») --
    numbers posture.sentence_articles did not read, and no source after them (a comma clause skipped: data
    «prose_review.bare_article.source_words» -- «del», «della», «c.», «cost», «legge», …: the number belongs to a named
    source, that parser's matter). Each resolved by its number in the review's scope: the fitted or named article
    with that number, else the excluded one, else «art. <n>» (an article outside the fitted set)."""
    cfg = _bare_cfg()
    text = str(text or "")
    have = {art_parts(c)[0] for c in cited or []}
    known = [str(x) for x in list(scope.get("fitted") or []) + list(scope.get("named") or [])
             + list(scope.get("excluded") or [])]
    out: List[str] = []
    for m in cfg["cite"].finditer(text):
        tail = cfg["clause"].sub("", text[m.end():], count=1)
        if cfg["source"].match(tail):
            continue
        for raw in [m.group(1)] + [x.group(0) for x in cfg["num"].finditer(m.group(2) or "")]:
            n = _bare_number(raw)
            if not n or art_parts(n)[0] in have:
                continue
            hits = [x for x in known if art_parts(x)[0] == art_parts(n)[0]]
            mine = [x for x in hits if _among(x, scope.get("fitted") or []) or _among(x, scope.get("named") or [])]
            out.append((mine or hits or [f"art. {n}"])[0])
    return list(dict.fromkeys(out))


def _among(label: str, labels: Sequence[str]) -> bool:
    return any(same_article(label, x) for x in labels or [])


def prose_scope(v: Optional[Dict[str, Any]], excluded: Sequence[str] = (), named: Sequence[str] = (),
                attached: Sequence[str] = ()) -> Dict[str, Any]:
    """The articles the prose review reads a sentence against: {"excluded": the binding's and the fit's (the caller's
    lawyer_method.excluded_labels), "fitted": the analysis input's norms (view «rubriche»: the fitted norms) and the
    remedies §4 names (view «remedies»; `attached` in the remedy set -- norms_tables' extra_remedies), "named": the
    articles the user named (cited in `named`: the case, his questions, the case's sentences; the frame's verified
    charge), "exempt_codes"}. With no fitted norm (no input) "fitted" is None: the outside check not made."""
    v = v or {}
    fitted = list(dict.fromkeys(list((v.get("rubriche") or {}).keys()) + list(v.get("remedies") or [])
                                + [str(x) for x in attached or [] if is_remedy(x)]))
    said: List[str] = []
    for text in list(named or []) + list((v.get("sentences") or {}).values()):
        said += _cited(text)
    charge = (v.get("analysis") or {}).get("frame_charge")
    if charge:
        said.append(str(charge))
    cfg = config().get("prose_review") or {}
    return {"excluded": list(dict.fromkeys(str(x) for x in excluded or [] if str(x or "").strip())),
            "fitted": fitted if (v.get("rubriche") or {}) else None, "named": list(dict.fromkeys(said)),
            "exempt_codes": [str(x).lower() for x in cfg.get("exempt_codes") or []]}


def outside(cited: Sequence[str], scope: Dict[str, Any]) -> Tuple[str, List[str]]:
    """(the reason, its articles) a sentence citing `cited` is dropped for: ARTICLE_EXCLUDED (any of them excluded),
    else ARTICLE_UNFITTED (any outside the fitted set, not named by the user, not of an exempt code); ("", [])."""
    gone = [c for c in cited if _among(c, scope.get("excluded") or [])]
    if gone:
        return ARTICLE_EXCLUDED, gone
    fitted = scope.get("fitted")
    if fitted is None:
        return "", []
    loose = [c for c in cited if not _among(c, fitted) and not _among(c, scope.get("named") or [])
             and art_parts(c)[1] not in (scope.get("exempt_codes") or [])]
    return (ARTICLE_UNFITTED, loose) if loose else ("", [])


@lru_cache(maxsize=1)
def _anaphoric_re() -> "re.Pattern[str]":
    words = "|".join(re.escape(fold(x)) for x in (config().get("prose_review") or {}).get("anaphoric_openers") or [])
    return re.compile(rf"^(?:{words or '(?!x)x'})(?![a-z0-9])")


def anaphoric(sentence: Any) -> bool:
    """The sentence opens with an anaphoric connective (data «prose_review.anaphoric_openers»; accents and case folded,
    leading quotes, dashes, bullets and bold marks out)."""
    return bool(_anaphoric_re().match(fold(_LEAD_RE.sub("", str(sentence or "")))))


def statement_for(labels: Sequence[str], v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The first verified affermazione_di_diritto (view «statements», in order) on one of the articles; None."""
    for s in v.get("statements") or []:
        if _flat(s.get("testo")) and any(_among(a, labels) for a in s.get("articoli") or []):
            return s
    return None


def statement_text(s: Dict[str, Any]) -> str:
    """A verified statement as it replaces a sentence: its text verbatim, «(art. X)» added when the text names none of
    its articles, a full stop."""
    text = _stop(s.get("testo"))
    arts = [str(a) for a in s.get("articoli") or [] if a]
    if arts and not any(_among(a, _cited(text)) for a in arts):
        text += f" (art. {', '.join(arts)})"
    return text + "."


# v79.15.3 (item 3): a §7 answer sentence asserting an element as integrato / provato that the verified table holds
# «da_provare» or «contestato» -- dropped (objections_sentences_dropped, its event's «motivo» UNPROVEN_ASSERTED).
UNPROVEN_ASSERTED = "elemento_non_provato"
STATEMENT_OFF_ANSWER = "affermazione_su_articolo_non_citato"      # v79.16.0 (C3)


@lru_cache(maxsize=1)
def _answer_check() -> Dict[str, Any]:
    cfg = config().get("objection_answer_check") or {}
    alt = lambda xs: "|".join(f"(?:{x})" for x in xs or []) or r"(?!x)x"      # noqa: E731
    word = lambda key: re.compile(rf"(?<![\w'’])(?:{alt(cfg.get(key))})(?!\w)", re.IGNORECASE)  # noqa: E731
    return {"verb": word("verbs"), "neg": word("negations"), "said": word("attributions"),
            "plain": word("affirmations"), "period": word("period_breaks"),
            "overlap": float(cfg.get("overlap") or 0.6)}


def _named(clause: str, v: Dict[str, Any], overlap: float) -> Tuple[List[Dict[str, Any]], float]:
    """The verified elements of a line article a clause names best (E-id or schema text 3, «E<n>» / «elemento <n>» 2,
    else the share of the element's tokens when at least `overlap`) and that score; ([], 0.0) none."""
    folded, mine = fold(clause), tokens(clause)
    best: List[Dict[str, Any]] = []
    score = 0.0
    for _label, els in ((v.get("analysis") or {}).get("elementi") or {}).items():
        for e in els or []:
            eid, body = str(e.get("eid") or ""), fold(e.get("elemento"))
            n = (_eid_split(eid) or ("", 0))[1]
            if (eid and fold(eid) in folded) or (body and body in folded):
                got = 3.0
            elif n and re.search(rf"(?<![\w:])(?:e|elemento\s+){n}(?!\w)", folded):
                got = 2.0
            else:
                want = tokens(e.get("elemento"))
                got = len(want & mine) / len(want) if want else 0.0
                got = got if got >= overlap else 0.0
            if got > score:
                best, score = [e], got
            elif got and got == score:
                best.append(e)
    return best, score


def asserted_unproven(sentence: Any, v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """v79.15.3 (item 3): the verified element a sentence asserts as integrato / provato while the verified table
    holds it «da_provare» or «contestato»; None. Read clause by clause (data «objection_answer_check»):
      - «affirmations» (senza alcun dubbio, senz'altro, non c'è dubbio che, …) are taken out first;
      - the sentence is cut into periods at «;», at a «:» before a space (an E-id's colon is no cut) and at «period_breaks» (ma, mentre, però, tuttavia, invece, …), a
        period into clauses at «,»;
      - a period holding one of «attributions» (afferma, sostiene, secondo, per l'accusa, …) reports another's claim:
        it asserts nothing («La Procura afferma che E1 è integrato, ma X lo nega»);
      - a clause asserts when it holds one of «verbs» (integrato, provato, sussiste, dimostrato, accertato, …; whole
        words from their start) with none of «negations» BEFORE that verb in the same clause (non, né, mai, nessuno,
        manca, senza, contesta, esclude, and the modal or conditional markers va, deve, resta, se, qualora, …; explicit
        word forms, so «restituzione», «contesto» or «difetto dell'opera» negate nothing);
      - an element's own schema text in the sentence is read as its E-id first (no period break inside it: «Inadempiente
        è invece la parte che l'ha ricevuta» stays whole);
      - the element is the one that clause names best (_named: an id or the text first, then the highest overlap), else
        the one the last preceding verbless clause names, across «:» («Quanto agli artifizi, sono integrati»; «<the
        element's text>: l'elemento è integrato»); a clause with a verb or a reported period ends that topic.
    The first asserting clause whose element is held «da_provare» / «contestato» decides; a tie with a «provato» or
    «escluso» element keeps that clause."""
    text = _flat(sentence)
    cfg = _answer_check()
    if not text or not cfg["verb"].search(text):
        return None
    text = cfg["plain"].sub(" ", text)
    for _label, els in ((v.get("analysis") or {}).get("elementi") or {}).items():
        for e in els or []:             # an element's own text («… è invece la parte …») is cut by nothing: its E-id
            words = re.findall(r"[^\W_]+", str(e.get("elemento") or ""))
            if e.get("eid") and words:
                text = re.sub(r"(?<!\w)" + r"[\W_]+".join(map(re.escape, words)) + r"(?!\w)", f" {e['eid']} ", text,
                              flags=re.IGNORECASE)
    topic: List[Dict[str, Any]] = []
    for period in re.split(r";|:(?=\s|$)", cfg["period"].sub(";", text)):
        if not period.strip():
            continue
        if cfg["said"].search(period):
            topic = []
            continue
        for clause in period.split(","):
            named, _score = _named(clause, v, cfg["overlap"])
            verbs = list(cfg["verb"].finditer(clause))
            if not verbs:
                topic = named or topic
                continue
            negs = [m.start() for m in cfg["neg"].finditer(clause)]
            if all(any(n < m.start() for n in negs) for m in verbs):
                topic = []
                continue
            named = named or topic
            if named and all(e.get("esito") in (DA_PROVARE, CONTESTATO) for e in named):
                return named[0]
            topic = []
    return None


class ProseReview:
    """One text's review (the rules above): call it on each sentence in reading order -> the sentence, its replacement
    or "" (dropped); every change into `events` as {"sede", "esito", "testo", "articoli", "con"?, "overlap"?}."""

    def __init__(self, v: Dict[str, Any], scope: Dict[str, Any], site: str, events: List[Dict[str, Any]],
                 replace: bool = True, answer: Optional[bool] = None, cited_scope: Optional[Sequence[str]] = None):
        self.v, self.scope, self.site, self.events = v or {}, scope or {}, site, events
        self.replace = replace          # False: §7's objection itself -- a rule with no twin dropped, never replaced
        # v79.16.0 (C3): the Q&A -- a replacement statement must cite an article the answer itself cited (cited_scope:
        # the answer's articles); otherwise the sentence is dropped, not replaced
        self.cited_scope = list(cited_scope) if cited_scope is not None else None
        # v79.15.3 (item 3): a §7 answer (site «obiezioni», not the objection itself) -- asserted_unproven checked
        self.answer = (site == SITE_OBJECTIONS and replace) if answer is None else bool(answer)
        self.pending = False            # the last sentence was dropped: an anaphoric next one goes with it
        self.context: List[str] = []    # §4: the block heading's articles
        self.block = ""                 # §4: the reason the current block is dropped
        self.used: set = set()

    def _note(self, kind: str, sentence: str, labels: Sequence[str] = (), **extra: Any) -> None:
        self.events.append({"sede": self.site, "esito": kind, "testo": _flat(sentence)[:200],
                            "articoli": list(dict.fromkeys(labels or [])), **extra})

    def keep(self, sentence: str) -> str:
        """A sentence the caller keeps unreviewed (a code line): it ends a dropped run."""
        self.pending = False
        return sentence

    def __call__(self, sentence: str) -> str:
        text = _flat(sentence)
        if not text:
            return sentence
        cited = _cited(text)
        bare = bare_cited(text, cited, self.scope)      # v79.15.1 (review): «l'art. 1439» cites 1439 too
        cited = list(dict.fromkeys(cited + bare))
        if text.startswith("**"):                   # a heading: a new block
            self.pending, self.block = False, ""
            kind, labels = outside(cited, self.scope)
            if kind and len(labels) == len(cited):
                self.block = kind
                self._note(kind, text, labels)
                return ""
            self.context = cited
            return sentence
        if self.block:
            self._note(self.block, text, cited)
            return ""
        kind, labels = outside(cited, self.scope)
        if kind:
            self._note(kind, text, labels)
            self.pending = True
            return ""
        if self.pending and anaphoric(text):
            self._note(DEPENDENT, text, cited)
            return ""
        gone = asserted_unproven(text, self.v) if self.answer else None
        if gone is not None:                        # v79.15.3 (item 3): «integrato» against the verified table
            self._note(DROPPED, text, cited, motivo=UNPROVEN_ASSERTED, elemento=gone.get("eid"),
                       esito_verificato=gone.get("esito"))
            self.pending = True
            return ""
        refs = norm_statement(text)
        if refs is None and bare and _qa_verb_re().search(text):
            refs = []                               # v79.15.1 (review): a rule on an article cited by number alone
        if refs is None or counterpart(text, self.v, bare) is not None:
            self.pending = False
            return sentence
        from .citations import label as _label
        labels = [_label(r) for r in refs] or list(bare) or list(self.context)
        st = statement_for(labels, self.v) if self.replace else None
        if st is not None and self.cited_scope is not None and \
                not any(_among(a, self.cited_scope) for a in st.get("articoli") or []):
            self._note(DROPPED, text, labels, motivo=STATEMENT_OFF_ANSWER, con=_flat(st.get("testo"))[:200])
            self.pending = True
            return ""
        if st is not None:
            new = statement_text(st)
            if new in self.used:                    # its statement already printed: nothing printed for it -- a drop,
                self._note(REPLACED, text, labels, con=_flat(st.get("testo"))[:200], duplicato=True)
                self.pending = True                 # its anaphoric follower goes with it (v79.15.1 review)
                return ""
            self._note(REPLACED, text, labels, con=_flat(st.get("testo"))[:200])
            self.pending = False
            self.used.add(new)
            return new
        best = max([overlap(text, s.get("testo")) for s in self.v.get("statements") or []] or [0.0])
        self._note(DROPPED, text, labels, overlap=round(best, 2))
        self.pending = True
        return ""


def orphans(before: Sequence[str], after: Sequence[str]) -> set:
    """The indices of `after`'s sentences (the answer-wide checks' output of `before`, sentence lists) that open with an
    anaphoric connective right after sentences of `before` the checks removed (difflib's «delete»: gone, not
    rewritten) -- and the run of such openers after them."""
    from difflib import SequenceMatcher
    a, b = [_flat(x) for x in before or []], [_flat(x) for x in after or []]
    out: set = set()
    for tag, _i1, _i2, j1, _j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "delete":
            continue
        j = j1
        while j < len(b) and anaphoric(b[j]):
            out.add(j)
            j += 1
    return out


def dependent_event(site: str, sentence: str) -> Dict[str, Any]:
    """The event of an anaphoric sentence dropped because the answer-wide checks removed the one before it."""
    return {"sede": site, "esito": DEPENDENT, "testo": _flat(sentence)[:200], "articoli": _cited(sentence),
            "dopo": "controlli"}


def prose_counters(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """The review's counters from its events: qa_sentences_dropped / norms_sentences_dropped /
    objections_sentences_dropped -- a rule sentence with no verified statement for its article, dropped, per site (the
    Q&A, §4, §7); qa_sentences_replaced -- a rule sentence replaced by its article's verified statement;
    qa_sentences_dropped_dependent -- an anaphoric sentence dropped with the one before it;
    prose_excluded_article_dropped / prose_unfitted_article_dropped -- a sentence citing an excluded / an unfitted
    article (the last four over the three sites). Each sentence in one counter only."""
    out = {k: 0 for k in list(_SITE_DROPPED.values()) + list(_KIND_COUNTER.values())}
    out["prose_unknown_id_dropped"] = 0         # v79.17.0 (item 4): a sentence citing an id that does not exist
    for e in events or []:
        if e.get("esito") == DROPPED:
            key = _SITE_DROPPED.get(e.get("sede"))
            if e.get("motivo") == "id_inesistente":
                out["prose_unknown_id_dropped"] += 1
        else:
            key = _KIND_COUNTER.get(e.get("esito"))
        if key:
            out[key] += 1
    return out


# --- the old engine, logged and compared (never rendered with the flag on) ----------------------------------------------

LEGACY_ESITO = {"provato": PROVATO, "da provare": DA_PROVARE, "non risulta": ESCLUSO}


def legacy_esito(r: Dict[str, Any]) -> str:
    from . import issue_records
    if issue_records.contested(r):
        return CONTESTATO
    return LEGACY_ESITO.get(r.get("status") or "da provare", DA_PROVARE)


def compare(v: Dict[str, Any], legacy: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The old engine against the verified analysis: legacy_disagrees_line (1 when the old main line's article is not the
    analysis's leading article -- an undetermined one included), legacy_disagrees_esito (the line articles' elements
    whose old verdict differs: matched verbatim on the record's requisites; an element the record lacks is not
    compared); the old verdicts per element."""
    lead = ((v.get("analysis") or {}).get("linea_principale") or {}).get("articolo") or ""
    old = (legacy.get("linea_principale") or {}).get("articolo") or ""
    disagrees_line = int(not (lead and old and same_article(lead, old)) and bool(lead or old))
    diffs, olds = 0, {}
    for label in line_articles(v):
        rec = next((r for k, r in (records or {}).items() if same_article(k, label)), None) or {}
        by_text = {verbatim_key(r.get("text")): r for r in rec.get("requisites") or [] if isinstance(r, dict)}
        rows = []
        for e in ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []:
            r = by_text.get(verbatim_key(e["elemento"]))
            if r is None:
                continue
            was = legacy_esito(r)
            rows.append({"elemento": e["elemento"], "legacy": was, "verificato": e.get("esito")})
            diffs += int(was != e.get("esito"))
        olds[label] = rows
    return {"legacy_disagrees_line": disagrees_line, "legacy_disagrees_esito": diffs, "elementi": olds}


def open_facts(v: Dict[str, Any], split: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """§2's «Fatti mancanti» with the verified analysis: only the entries on an element the analysis holds «da_provare»
    (the record's element matched verbatim to the line article's verdict) -- the old engine's other open elements
    never listed."""
    want = {(label, verbatim_key(e["elemento"])) for label in line_articles(v)
            for e in ((v.get("analysis") or {}).get("elementi") or {}).get(label) or [] if e.get("esito") == DA_PROVARE}
    kept = []
    for m in (split or {}).get("mancanti") or []:
        label = next((x for x, _ in want if same_article(x, m.get("article"))), None)
        rec = next((r for k, r in (records or {}).items() if same_article(k, m.get("article"))), None) or {}
        text = next((r.get("text") for r in rec.get("requisites") or [] if isinstance(r, dict)
                     and str(r.get("id")) == str(m.get("elemento"))), None)
        if label and text is not None and (label, verbatim_key(text)) in want:
            kept.append(m)
    return {**(split or {}), "mancanti": kept}
