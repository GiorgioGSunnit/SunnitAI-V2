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
max_tokens), on a worker thread waited at most the step's remaining budget -- «deadline_s» (40 s; v79.19.1: plus
«deadline_per_article_s» per fitted norm, at most «deadline_max_s» = 60 s, deadline_for) from the step's
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
            # v79.19.0 (item 6): no verified line, no rubrica match -- the best-tier fitted article leads
            "analysis_line_fallback_fit",
            # v79.19.4 (rule 2): live records graded 2+ that are no line and got no listed reason (must be 0)
            "analysis_silent_records")

# v79.19.4 (rule 2): a live record graded 2 or more is a line or is in «rejected» with a reason from this list --
# bookkeeping after the step, the step's own line choice and order untouched
REJECT_NOT_IN_BLOCK = "not_in_block"        # the record never reached the block sent to the model (no norm built for it)
REJECT_BLOCK_FULL = "block_full"            # reserved: the block never drops a norm today (block_sizes trims text only)
REJECT_BELOW_CUT = "below_cut"              # in the block, not among the step's two lines
REJECT_DUPLICATE = "duplicate_of"           # «duplicate_of:<label>»: an alias of a line's article (charge_aliases)
REJECT_SILENT = "step_silent"               # the catch-all: the step gave no lines (failed, timed out, no input)
RECORD_REJECT_REASONS = (REJECT_NOT_IN_BLOCK, REJECT_BLOCK_FULL, REJECT_BELOW_CUT, REJECT_DUPLICATE, REJECT_SILENT)
RECORD_KIND = "record"                      # the «kind» of such an entry (a line's entries: linea_principale / _subordinata)
PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO = "provato", "contestato", "da_provare", "escluso"
ESITI_ALL = (PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO)
# v79.19.1: (item 1) the frame's charge the step rejected, or missing while a verified line exists, never leads;
# (item 2) the article block sent with its periods omitted (over «max_block_chars»)
COUNTERS = COUNTERS + ("analysis_frame_charge_overridden", "analysis_block_reduced")
OK, FAILED, MALFORMED, TIMEOUT, NO_INPUT, OFF = "ok", "failed", "malformed", "timeout", "no_input", "off"
LINE_KINDS = ("linea_principale", "linea_subordinata")
VERIFIED, FRAME_LINE, UNDETERMINED = "verificata", "addebito_del_quadro", "da_determinare"     # a line's «stato»
FALLBACK_LINE = "rubrica_del_quadro"        # v79.15.1 (item 2): the line found by the frame's named charge's rubrica
FIT_LINE = "norma_applicabile"              # v79.19.0 (item 6): the best-tier fitted article, its elements da provare
FIT_TIERS = {"si": 0, "dipende": 1, "": 2}  # the tiers of a fitted norm (the judgment's verdict; "" unjudged)
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
SKIP_STEP_FAILED = "step_failed"                # the first call failed, timed out or answered no JSON; the input unread
SKIP_NO_INPUT = "no_input"                      # no fitted norm or no sentence: nothing called
SKIP_FRAME_CHARGE_REJECTED = "frame_charge_rejected"    # v79.19.1 (item 1): the frame's charge the step rejected as a
STRING_COUNTERS = (SKIPPED_REASON,)                     # line never leads as the unverified frame line


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


def client_side(frame: Optional[Dict[str, Any]], client: str = "", client_roles: Sequence[str] = ()
                ) -> Tuple[str, str]:
    """v79.15.3 (item 2): (the client's side, why) from the VERIFIED frame only -- SIDE_CLAIMANT, SIDE_DEFENCE or
    SIDE_NEUTRAL (data «claimant», whole words, accent- and case-folded):
      1. the verified ruolo (valore and intake: «imputato/indagato» read as both words): a word of «claimant.roles»
         -> claimant, of «claimant.defendant_roles» -> defence (the procedural role decides first: a committente
         «convenuto» is the defence);
      2. else the client's party in frame.parti (lato «cliente», or the client's name) -- its verified ruolo_nel_fatto
         (case_frame voids an ungrounded one): only claimant words -> claimant, only defence words -> defence;
         «claimant.worker» words (lavoratore) are a claimant's only when the claim is the worker's: the verified rito
         one of «worker.riti» (lavoro) and the frame verifying an addebito_o_domanda (articolo or descrizione);
      3. v79.19.5 (item 1, row 91): else the posture's role words (client_roles: interactive_flow.client_roles -- the
         intake's and the posture's role, the parties' words for the client), the same two lists: only defence words
         -> defence, only claimant words -> claimant («why»: «client_roles: <word>»);
      4. else (no verified role, no side word anywhere, or both sides' words) -> neutral («why» empty).
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
    said = " ".join(str(r or "") for r in client_roles or [])      # v79.19.5 (item 1): the posture's role words
    ups, downs = _role_hits(said, pro), _role_hits(said, contra)
    if downs and not ups:
        return SIDE_DEFENCE, f"client_roles: {downs[0]}"
    if ups and not downs:
        return SIDE_CLAIMANT, f"client_roles: {ups[0]}"
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
               older_issues: Sequence[str] = ()) -> Dict[str, Any]:
    """The step's input. frame: the verified frame's fields (only verified ones; {} none); norms: [{article, text,
    rubrica, elements: [{id, testo, schema_kind?, special?}]}] -- the fitted norms; sentences: [{id, text, certainty,
    source, alleged_by?}] numbered as the flow numbers them (a «sistema» one left out, the numbers kept); questions: the
    user's; client: the client's name (the frame's, else the flow's); client_roles: his procedural role words;
    older_issues: the older frame's central_issue_candidates' names the caller checked (v79.15.1, the rubrica
    fallback). v79.15.1: each element gets its E-id («eid», element_id), «eccezione» and «eccezione_fonte»
    (exception_of); the input carries «claimant» and «issues» (named_issues). v79.15.3: «side» / «side_why»
    (client_side: attore | difesa | neutro, from the verified frame only); «claimant» is side == attore."""
    f = dict(frame or {})
    contractual = is_contractual(f)
    addebito = f.get("addebito_o_domanda") or {}
    # v79.19.1 (item 1): «charge» is the article the TEXT names -- the frame-charge rule (the leading line when the
    # model leaves it out) applies to it alone; an article read from the facts («da_fatti»: a hypothesis the fit
    # confirmed) is «charge_hypothesis», a fitted norm like any other, never forced to lead
    named_label = _cf().charge_label({"frame": f}) if addebito.get("articolo") else ""
    charge = named_label if not addebito.get("da_fatti") else ""
    charge_hyp = named_label if addebito.get("da_fatti") else ""
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
                          "elements": elements, "periods": periods, "fit": str(n.get("fit") or ""),  # v79.19.0 (6)
                          "remedy": is_remedy(label, contractual) and not any(
                              c and same_article(label, c) for c in (charge, charge_hyp))})
    sents = [{"id": str(s.get("id")), "text": _flat(s.get("text")), "certainty": s.get("certainty"),
              "source": s.get("source"), "alleged_by": str(s.get("alleged_by") or "")}
             for s in sentences or [] if isinstance(s, dict) and s.get("id") and _flat(s.get("text"))
             and s.get("source") != "sistema"]
    client = _flat(client or (f.get("cliente") or {}).get("nome"))
    roles = [str(r) for r in client_roles or [] if str(r or "").strip()]
    side, side_why = client_side(f, client, roles)  # v79.15.3 (item 2): the verified side; v79.19.5 (item 1): else the
    #                                                 posture's role words; «claimant» its claimant
    return {"frame": f, "norms": out_norms, "sentences": sents, "questions": [_flat(q) for q in questions or [] if
                                                                              _flat(q)],
            "client": client, "client_roles": roles,
            "charge": charge, "charge_hypothesis": charge_hyp, "contractual": contractual,
            "claimant": side == SIDE_CLAIMANT,
            "side": side, "side_why": side_why, "issues": named_issues(f, older_issues)}


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
    norms = []
    verdicts = fit_verdicts(state)          # v79.19.0 (item 6): each norm's verified judgment (si / dipende)
    for label, rec in (state.get("records") or {}).items():
        if not isinstance(rec, dict) or rec.get("dropped"):
            continue
        key = flow._key(flow._parse_label(label))
        if key in removed:
            continue
        n = ap.norm_input(label, have.get(key), rec, rows)
        n["fit"] = verdicts.get(key, "")
        kinds = {str(r.get("id")): r for r in rec.get("requisites") or [] if isinstance(r, dict) and r.get("id")}
        n["elements"] = [dict(e, **{k: kinds[str(e.get("id"))][k] for k in ("schema_kind", "special")
                                    if (kinds.get(str(e.get("id"))) or {}).get(k)})
                         for e in n.get("elements") or []]          # v79.15.1: the schema's own element kind
        norms.append(n)
    sents = state.get("sentences") or flow.case_sentences(state)
    questions = list(frame.get("quesiti") or []) or flow.asked_questions(state)
    client = (frame.get("cliente") or {}).get("nome") or (state.get("client") or {}).get("name") or ""
    try:
        roles = flow.client_roles(state)
    except Exception:           # noqa: BLE001 -- no role read: the client's name and side words only
        roles = []
    out = make_input(frame, norms, sents, questions, client, roles, older_issues(state, sents))
    # v79.19.4 (rule 2): the live records (not dropped) with their grade, for the accounting after the step; the aliases
    # v79.19.5 (item 5): with the rubrica and the live elements (issue_records.element_status) for the general-part section
    from . import issue_records
    out["graded"] = [{"article": str(label), "grade": rec.get("grade"), "rubrica": _flat(rec.get("rubrica")),
                      "elements": [{"id": str(r.get("id") or ""), "text": _flat(r.get("text")),
                                    "status": str(r.get("status") or ""), "contested": bool(r.get("contested_by"))}
                                   for r in rec.get("requisites") or [] if isinstance(r, dict)
                                   and issue_records.element_status(rec, r) != issue_records.NOT_RELEVANT]}
                     for label, rec in (state.get("records") or {}).items()
                     if isinstance(rec, dict) and not rec.get("dropped")]
    out["aliases"] = {str(k): str(v) for k, v in (state.get("charge_aliases") or {}).items()}
    return out


def account_records(result: Dict[str, Any], inp: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """v79.19.4 (rule 2): (the «rejected» entries for the live records graded 2 or more that are no line, the number
    of them with no listed reason). A record that is a line (same_article) or already rejected by the verifier: no
    entry. Else its reason: the step not ok (no lines at all) -> step_silent; its label an alias of a line's
    (inp["aliases"]) -> «duplicate_of:<line>»; not among the block's norms -> not_in_block; else below_cut (the step's
    two lines are others). Each entry: {kind: record, articolo, why, grade, rank: its 1-based place in the block, or
    0}. Bookkeeping only."""
    analysis = result.get("analysis") if result.get("status") == OK and isinstance(result.get("analysis"), dict) \
        else {}
    lines = [x.get("articolo") for x in (analysis.get(k) for k in LINE_KINDS) if x and x.get("articolo")]
    rejected = [r.get("articolo") for r in result.get("rejected") or [] if r.get("articolo")]
    block = [n.get("article") for n in inp.get("norms") or [] if n.get("article")]
    aliases = inp.get("aliases") or {}
    out: List[Dict[str, Any]] = []
    silent = 0
    for rec in inp.get("graded") or []:
        label, grade = str(rec.get("article") or ""), rec.get("grade")
        if not label or not isinstance(grade, int) or grade < 2:
            continue
        if any(same_article(label, x) for x in lines + rejected):
            continue
        twin = aliases.get(label) or next((k for k, v in aliases.items() if v == label), "")
        rank = next((i for i, x in enumerate(block, 1) if same_article(label, x)), 0)
        if result.get("status") != OK or not lines:
            why = REJECT_SILENT
        elif twin and any(same_article(twin, x) for x in lines):
            why = f"{REJECT_DUPLICATE}:{twin}"
        elif not rank:
            why = REJECT_NOT_IN_BLOCK
        else:
            why = REJECT_BELOW_CUT
        silent += int(why == REJECT_SILENT)
        out.append({"kind": RECORD_KIND, "articolo": label, "why": why, "grade": grade, "rank": rank})
    return out, silent


def has_verified_line(view_: Optional[Dict[str, Any]]) -> bool:
    """v79.19.4 (rule 3): the rendering's view holds a verified line (stato VERIFIED) -- principale or subordinata."""
    a = (view_ or {}).get("analysis") or {}
    return any(isinstance(a.get(k), dict) and a[k].get("stato") == VERIFIED for k in LINE_KINDS)


def fit_verdicts(state: Dict[str, Any]) -> Dict[Any, str]:
    """v79.19.0 (item 6): {norm key: verdict} of the applicability judgment's verified judgments
    (state["applicability"]["judgments"]: si / dipende / no); {} with none."""
    from . import interactive_flow as flow
    out: Dict[Any, str] = {}
    fit = state.get("applicability") if isinstance(state.get("applicability"), dict) else {}
    for j in (fit or {}).get("judgments") or []:
        if isinstance(j, dict) and j.get("verified") and j.get("article") and j.get("fit"):
            try:
                out[flow._key(flow._parse_label(str(j["article"])))] = str(j["fit"])
            except Exception:       # noqa: BLE001 -- an unparseable label: unjudged
                continue
    return out


def fitted_fallback(inp: Dict[str, Any], skip: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """v79.19.0 (item 6): «da determinare» never prints when a fitted article exists -- the best-tier fitted norm (a
    verified «si» first, then «dipende», then the unjudged; the input's order within a tier; the remedy set left out)
    leads with its elements da provare: stato FIT_LINE, no element in play, no id cited, tesi FALLBACK_TESI on the
    norm's rubrica (its article when it has none); None with no such norm. v79.19.1 (item 1): skip -- the articles
    the step rejected as a line never lead this way either."""
    best: Optional[Tuple[int, int, Dict[str, Any]]] = None
    for i, n in enumerate(inp.get("norms") or []):
        if n.get("remedy") or not n.get("article"):
            continue
        if any(same_article(n["article"], x) for x in skip or ()):
            continue
        tier = FIT_TIERS.get(str(n.get("fit") or ""), FIT_TIERS[""])
        if str(n.get("fit") or "") == "no":
            continue
        if best is None or (tier, i) < best[:2]:
            best = (tier, i, n)
    if best is None and skip:
        # the rejected article the only fitted norm: still a fitted norm applicable to the facts -- «da determinare»
        # stays reserved for no fitted article at all (v79.19.0 item 6); it leads as such, never as the frame's charge
        return fitted_fallback(inp)
    if best is None:
        return None
    n = best[2]
    tesi = FALLBACK_TESI["penale" if penal_frame(inp.get("frame"), n["article"]) else "civile"]
    return {"articolo": n["article"], "elemento_in_gioco": None, "elemento_id": None, "eid": None,
            "tesi": tesi.format(rubrica=_stop(n.get("rubrica") or f"art. {n['article']}")), "cita_fatti": [],
            "cita_norma": [], "stato": FIT_LINE, "fit": str(n.get("fit") or "")}


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


def _norm_block(n: Dict[str, Any], elements_only: bool = False) -> str:
    labels = config()["labels"]
    head = [f"ARTICOLO: {n['article']}" + (f" ({config()['remedy_mark']})" if n.get("remedy") else ""),
            f"Rubrica: {n.get('rubrica') or labels['none']}", f"{labels['elements']}:"]
    head += [f"- [{e['eid']}] {e['testo']}" for e in n.get("elements") or []] or [f"- {labels['none']}"]
    if elements_only:           # v79.19.1 (item 2): the schede's text dropped, the elements kept (block_sizes)
        head.append(f"{labels['periods']}: {labels.get('periods_omitted') or '(omesso)'}")
        return "\n".join(head)
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
    sizes = block_sizes(inp)
    parts.append(f"{labels['norms']}:\n\n" + "\n\n".join(_norm_block(n, elements_only=sizes["reduced"])
                                                          for n in inp.get("norms") or []))
    qs = inp.get("questions") or []
    parts.append(f"{labels['questions']}:\n" + ("\n".join(f"- {q}" for q in qs) if qs else labels["none"]))
    return [SystemMessage(content=prompt_system()), HumanMessage(content="\n\n".join(parts))]


def block_sizes(inp: Dict[str, Any]) -> Dict[str, Any]:
    """v79.19.1 (item 2): the article block's size -- {"chars_full": the block with every norm's periods,
    "chars_sent": what the prompt carries, "reduced": the periods omitted (elements only), "norms": n, "cap"}. The
    block is reduced when its full size exceeds data «max_block_chars» (the 79.15.3 bound: the per-article cap times
    the fitted norms a 79.15.3 input carried -- this branch adds the alternative lines' schede, so a block can grow
    past what the step's deadline allows); 0 or missing: never reduced."""
    norms = inp.get("norms") or []
    full = sum(len(_norm_block(n)) for n in norms) + 2 * max(0, len(norms) - 1)
    cap = int(config().get("max_block_chars") or 0)
    reduced = bool(cap) and full > cap
    sent = sum(len(_norm_block(n, elements_only=True)) for n in norms) + 2 * max(0, len(norms) - 1) if reduced else full
    return {"chars_full": full, "chars_sent": sent, "reduced": reduced, "norms": len(norms), "cap": cap}


def deadline_for(inp: Dict[str, Any]) -> float:
    """v79.19.1 (item 2): the step's deadline scales with the article block -- data «deadline_s» (40) plus
    «deadline_per_article_s» (1) per fitted norm, at most «deadline_max_s» (60)."""
    cfg = config()
    base = float(cfg.get("deadline_s") or 40)
    per = float(cfg.get("deadline_per_article_s") or 0)
    cap = float(cfg.get("deadline_max_s") or base)
    return min(cap, base + per * len(inp.get("norms") or []))


def line_elements(norm: Dict[str, Any], inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.15.1: the elements a line on this norm may be played on -- every one; for a claimant, the constitutive
    ones only (exception_of false)."""
    els = list(norm.get("elements") or [])
    return [e for e in els if not e.get("eccezione")] if inp.get("claimant") else els


def allowed_text(inp: Dict[str, Any]) -> str:
    """v79.15.1 (item 2): the re-ask's list of the articles a line may stand on -- the fitted norms not in the remedy
    set (the frame's own charge kept: make_input never marks it), each with at least one usable element -- one per
    line: «- A<token> (art. <label>): <E-ids>» (a claimant's: the constitutive E-ids only)."""
    rows = []
    for n in inp.get("norms") or []:
        ids = [e["eid"] for e in line_elements(n, inp)]
        if n.get("remedy") or not ids:
            continue
        rows.append(f"- A{_ap().article_token(n['article'])} (art. {n['article']}): {', '.join(ids)}")
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


# v79.19.1 (item 2): the reply's text as it streams in, read by call_once when the call is not back in time
_PARTIAL: "contextvars.ContextVar[Optional[List[str]]]" = contextvars.ContextVar("analysis_step_partial", default=None)


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model (ai_chat.chat_model) bound to the kwargs; the reply without its reasoning, the call on the
    request's stage meter -- as applicability._default_call. v79.19.1 (item 2): streamed when call_once set a buffer
    (_PARTIAL), each chunk's text appended to it -- a call abandoned at the deadline leaves its partial reply there;
    the chunks summed are the reply (its finish_reason the last chunk's)."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = _invoke(ai_chat.chat_model.bind(**kwargs), messages, _PARTIAL.get())
    ai_chat._stage_call("analysis_step", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def _invoke(model: Any, messages: List[Any], buf: Optional[List[str]]) -> Any:
    """v79.19.1 (item 2): the model's reply -- streamed into buf (each chunk's text appended, the chunks summed into
    the reply) when a buffer is given and the model streams; else one invoke."""
    if buf is None or not hasattr(model, "stream"):
        return model.invoke(messages)
    response = None
    for chunk in model.stream(messages):
        response = chunk if response is None else response + chunk
        text = chunk.content if isinstance(getattr(chunk, "content", None), str) else ""
        if text:
            buf.append(text)
    if response is None:
        raise RuntimeError("the stream returned no chunk")
    return response


def parse_reply(raw: Any) -> Dict[str, Any]:
    """The reply's object; ValueError when it holds no JSON object."""
    from .llm_reply import parse_json_reply
    data = parse_json_reply(raw if isinstance(raw, str) else str(raw or ""))
    if not isinstance(data, dict):
        raise ValueError("no JSON object in the reply")
    return data


def _prompt_log(messages: Sequence[Any]) -> List[Dict[str, str]]:
    """v79.19.1 (item 2): the call's messages for the calls log -- role and content."""
    out = []
    for m in messages or []:
        kind = type(m).__name__.lower()
        role = "system" if "system" in kind else "assistant" if kind.startswith("ai") else "user"
        out.append({"role": role, "content": str(getattr(m, "content", m))})
    return out


def call_once(messages: List[Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]],
              budget: float, attempt: int = 0) -> Dict[str, Any]:
    """One call on a worker thread, waited at most budget seconds: {"status": ok | failed | timeout | malformed,
    "reply" (the text), "data" (the parsed object), "seconds", "error"}. A call not back in time is abandoned (its pool
    shut down without waiting); never raises. v79.19.1 (item 2): a timed-out call's entry carries «prompt» (the
    messages) and «partial_reply» (what the stream had returned: _PARTIAL); its late reply, if it ever lands, is
    logged «analysis_step_late_reply» (nothing reads it)."""
    started = time.time()
    if budget <= 0:
        return {"status": TIMEOUT, "reply": "", "data": None, "seconds": 0.0, "error": "nessun tempo residuo",
                "prompt": _prompt_log(messages), "partial_reply": ""}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis_step")
    partial: List[str] = []
    ctx = contextvars.copy_context()
    ctx.run(_PARTIAL.set, partial)
    try:
        fut = pool.submit(ctx.run, call or _default_call, messages, request_kwargs())
        done, _ = _wait([fut], timeout=budget)
        if fut not in done:
            def late(f, t0=started, n=attempt):
                try:
                    reply = f.result()
                except Exception as exc:        # noqa: BLE001 -- logged, nothing else
                    reply = f"({type(exc).__name__}: {exc})"
                _log("analysis_step_late_reply", {"attempt": n, "seconds": round(time.time() - t0, 3),
                                                  "chars": len(str(reply or "")), "late_reply": str(reply or "")[:4000]})
            fut.add_done_callback(late)
            return {"status": TIMEOUT, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"oltre {round(budget, 1)} s", "prompt": _prompt_log(messages),
                    "partial_reply": "".join(partial)}
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


def _verify_line(raw: Any, inp: Dict[str, Any], m: Dict[str, Any], counters: Dict[str, Any]
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    """(the verified line, "") or (None, why it is rejected)."""
    if not isinstance(raw, dict):
        return None, "non è un oggetto"
    label = resolve(raw.get("articolo"), m["labels"])
    if label is None:
        return None, f"articolo non tra le norme applicabili: {_flat(raw.get('articolo'))[:60] or '(vuoto)'}"
    norm = m["norms"][label]
    if norm.get("remedy"):
        return None, f"art. {label}: rimedio o parte generale, mai una linea"
    element = element_of(norm, raw.get("elemento_in_gioco"), m["labels"])
    if element is None:
        counters["analysis_bad_eid"] += 1
        said = _flat(raw.get("elemento_in_gioco"))[:40] or "(vuoto)"
        return None, (f"art. {label}: elemento_in_gioco «{said}» non è l'id di un elemento dello schema di questo "
                      f"articolo (A{_ap().article_token(label)}:E1…E{len(norm.get('elements') or [])})")
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
            "eid": element["eid"], "tesi": tesi, "cita_fatti": sf, "cita_norma": sn, "stato": VERIFIED}, ""


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
        line, why = _verify_line(raw, inp, m, counters)
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
    if esito == PROVATO:
        if not ids:
            counters["analysis_provato_unsupported"] += 1
            return out
        by = [reported_by(m["sents"][i], inp, parte) for i in ids]
        if all(by):
            counters["analysis_provato_downgraded"] += 1
            out["esito"], out["parte_che_afferma"] = CONTESTATO, by[0]
            out["declassato"] = "provato su un'allegazione"
            return out
        out["esito"] = PROVATO
        return out
    if esito in (ESCLUSO, CONTESTATO) and not ids:
        counters["analysis_escluso_unsupported" if esito == ESCLUSO else "analysis_contestato_unsupported"] += 1
        return out
    if esito == CONTESTATO and not parte:
        who = [w for w in (reported_by(m["sents"][i], inp) for i in ids) if w]
        out["parte_che_afferma"] = who[0] if who else None
    out["esito"] = esito
    return out


def _base_element(e: Dict[str, Any], origine: str) -> Dict[str, Any]:
    return {"elemento": e["testo"], "id": e["id"], "eid": e.get("eid"), "esito": DA_PROVARE, "cita_fatti": [],
            "parte_che_afferma": None, "origine": origine, "eccezione": bool(e.get("eccezione")),
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


def rubrica_fallback(inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """v79.15.1 (item 2): the leading line from the frame's named charges / claims (inp «issues»: named_issues) -- the
    first name whose rubrica_key equals a fitted norm's rubrica's (no fuzzy match), the remedy set left out, its first
    such norm in the input's order: stato FALLBACK_LINE, no element in play, no id cited, tesi FALLBACK_TESI on the
    norm's rubrica as the input has it (penal: penal_frame); None when no name matches."""
    for name in inp.get("issues") or []:
        key = rubrica_key(name)
        if not key:
            continue
        for n in inp.get("norms") or []:
            if n.get("remedy") or not n.get("rubrica") or rubrica_key(n["rubrica"]) != key:
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
    key = side
    if side == SIDE_DEFENCE and not penal_frame(inp.get("frame"), line["articolo"]) and cfg.get("difesa_civile"):
        key = "difesa_civile"       # v79.19.5 (item 1, row 91): a civil or labour defence -- the party's name, never «L'accusa»
    template = str(((cfg.get(key) or cfg.get(side) or cfg.get(SIDE_NEUTRAL) or {}).get(esito)) or "«{elemento}»")
    return {**line, "tesi_modello": line.get("tesi") or "", "tesi_codice": True, "esito_in_gioco": esito,
            "tesi_fatti": fatti,        # v79.19.5 (item 4): the facts as printed, for the «Tesi dell'accusa» rendering
            "tesi": template.format(elemento=line.get("elemento_in_gioco") or "", art=line["articolo"], fatti=fatti,
                                    controparte=counterparty_name(inp))}


def counterparty_name(inp: Dict[str, Any]) -> str:
    """v79.19.5 (item 1): the first verified party the frame lists as controparte (frame.parti, lato «controparte»);
    none: data «rendering.tesi.controparte_default» («la controparte»)."""
    f = inp.get("frame") if isinstance(inp.get("frame"), dict) else {}
    for p in f.get("parti") or []:
        if isinstance(p, dict) and p.get("lato") == "controparte" and _flat(p.get("nome")):
            return _flat(p["nome"])
    return str((rendering().get("tesi") or {}).get("controparte_default") or "la controparte")


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
    # v79.19.1 (item 1): the frame's charge (the text's article) in the step's rejected list -- the model put it as a
    # line and the verification refused it -- or missing while a verified line exists: it never leads; the step's
    # first verified line leads (analysis_frame_charge_overridden); the fallbacks never put a rejected article first.
    # SKIP_FRAME_CHARGE (the frame's article as the unverified leading line) only for a charge the text names, not
    # rejected, with no verified line at all.
    rejected_labels = [r.get("articolo") for r in checked.get("rejected") or [] if r.get("articolo")]
    charge_rejected = bool(charge) and any(same_article(x, charge) for x in rejected_labels)
    overridden = bool(checked.get("charge_missing") and charge and (charge_rejected or principale is not None))
    if overridden:
        counters["analysis_frame_charge_missing"] = 1
        counters["analysis_frame_charge_overridden"] = 1
        _log("analysis_frame_charge_overridden", {"charge": charge, "rejected": charge_rejected,
                                                  "leading": (principale or {}).get("articolo")})
        if principale is not None and not charge_rejected and subordinata is None:
            # the text's charge, not refused, beside the verified leading line: the subordinate, its elements da provare
            frame_sub = resolve(charge, m["labels"]) or charge
            subordinata = {"articolo": frame_sub, "elemento_in_gioco": None, "elemento_id": None, "eid": None,
                           "tesi": "", "cita_fatti": [], "cita_norma": [], "stato": FRAME_LINE,
                           "posizione": "subordinata"}
    if checked.get("charge_missing") and charge and not overridden:
        counters["analysis_frame_charge_missing"] = 1
        counters[SKIPPED_REASON] = SKIP_FRAME_CHARGE        # v79.15.2: the frame's article leads, no rubrica read
        frame_label = resolve(charge, m["labels"]) or charge
        subordinata = principale or subordinata
        principale = {"articolo": frame_label, "elemento_in_gioco": None, "elemento_id": None, "eid": None,
                      "tesi": "", "cita_fatti": [], "cita_norma": [], "stato": FRAME_LINE}
    elif principale is None:
        principale = rubrica_fallback(inp)
        if principale is not None and charge_rejected and same_article(principale["articolo"], charge):
            principale = None                               # v79.19.1 (item 1): the rejected charge never leads
        if principale is not None:
            counters["analysis_line_fallback_rubrica"] = 1
            counters[SKIPPED_REASON] = ""
        else:
            counters[SKIPPED_REASON] = SKIP_FRAME_CHARGE_REJECTED if charge_rejected else fallback_miss(inp)
            principale = fitted_fallback(inp, skip=rejected_labels if charge_rejected else ())   # v79.19.0 (item 6)
            if principale is not None:
                counters["analysis_line_fallback_fit"] = 1
            else:
                counters["analysis_line_undetermined"] = 1
                principale = undetermined_line()
    else:
        counters[SKIPPED_REASON] = SKIP_LINE_VERIFIED       # v79.15.2: a verified leading line, nothing to fall back on
    if subordinata is not None and principale.get("articolo") and same_article(subordinata["articolo"],
                                                                               principale["articolo"]) \
            and subordinata.get("elemento_in_gioco") == principale.get("elemento_in_gioco"):
        subordinata = None          # the same line twice: once
    items = checked["items"] if "items" in checked else _reply_elements(checked.get("reply") or {})
    elements: Dict[str, List[Dict[str, Any]]] = {}
    open_kinds = {FRAME_LINE: "quadro", FALLBACK_LINE: "rubrica", FIT_LINE: "applicabilità"}     # v79.19.0 (item 6)
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
    counters["objections_from_allegations"] = sum(1 for o in objections if o["fonte"] == "allegazione")
    counters["objections_from_exceptions"] = sum(1 for o in objections if o["fonte"] == "eccezione")
    return {"linea_principale": principale, "linea_subordinata": subordinata, "elementi": elements,
            "affermazioni_di_diritto": _law_statements(checked.get("reply") or {}, m, counters),
            "rimedi": [n["article"] for n in inp.get("norms") or [] if n.get("remedy")],
            "frame_charge": charge, "frame_charge_missing": bool(frame_label), "contractual": inp.get("contractual"),
            "frame_charge_overridden": overridden, "charge_hypothesis": inp.get("charge_hypothesis") or "",
            "claimant": bool(inp.get("claimant")), "side": inp.get("side") or SIDE_NEUTRAL, "obiezioni": objections,
            "eccezioni": [{"articolo": label, "eid": e.get("eid"), "fonte": e.get("eccezione_fonte")}
                          for label, els in elements.items() for e in els if e.get("eccezione")],
            "rejected": checked.get("rejected") or [], "fallback_skipped_reason": counters.get(SKIPPED_REASON) or ""}


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
    {"status", "error", "analysis" (None unless ok), "counters", "prompt_hash", "seconds", "calls", "reasons",
    "rejected"}; logged «analysis_step»."""
    t0 = started if started is not None else time.time()
    deadline = deadline_for(inp)            # v79.19.1 (item 2): 40 s + 1 s per fitted norm, at most 60 s
    counters = empty_counters()
    calls: List[Dict[str, Any]] = []
    status, error, analysis, reasons, rejected = NO_INPUT, "", None, [], []
    block = block_sizes(inp) if inp.get("norms") else {"chars_full": 0, "chars_sent": 0, "reduced": False, "norms": 0}

    def logged(attempt: int, got: Dict[str, Any]) -> Dict[str, Any]:     # v79.19.1 (item 2): the calls log entry
        entry = {"attempt": attempt, "status": got["status"], "seconds": got["seconds"], "error": got["error"]}
        if got["status"] == TIMEOUT:            # the timed-out call's prompt and partial reply, kept in full
            entry["prompt"] = got.get("prompt") or []
            entry["partial_reply"] = got.get("partial_reply") or ""
            _log("analysis_step_timeout", {"attempt": attempt, "budget_s": round(deadline, 1), "seconds": got["seconds"],
                                           "partial_chars": len(entry["partial_reply"]), "block": block,
                                           "prompt": entry["prompt"], "partial_reply": entry["partial_reply"]})
        return entry
    if inp.get("norms") and inp.get("sentences"):
        counters["analysis_block_reduced"] = int(bool(block["reduced"]))
        messages = build_messages(inp)
        first = call_once(messages, call, deadline - (time.time() - t0), attempt=1)
        calls.append(logged(1, first))
        status, error = first["status"], first["error"]
        if first["status"] == OK:
            checked = verify(first["data"], inp)
            reasons = list(checked["reasons"])
            rejected = [dict(r, attempt=1) for r in checked["rejected"]]
            futile = checked["reasons"] == [r for r in checked["reasons"] if r.startswith("l'addebito")] and \
                resolve(inp.get("charge"), [n["article"] for n in inp["norms"]]) is None
            if checked["reasons"] and not futile:
                second = call_once(reask_messages(messages, first["reply"], checked["reasons"], inp), call,
                                   deadline - (time.time() - t0), attempt=2)
                calls.append(logged(2, second))
                counters["analysis_reasked"] = 1
                if second["status"] == OK:
                    again = verify(second["data"], inp)
                    for k in ("analysis_line_rejected", "analysis_line_exception", "analysis_bad_eid"):
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
        else:
            counters["analysis_failed"] = 1
            counters["analysis_timeouts"] = int(first["status"] == TIMEOUT)
            counters[SKIPPED_REASON] = SKIP_STEP_FAILED
            logger.warning("analysis_step: %s: %s (nothing from the step)", first["status"], first["error"])
    if status == NO_INPUT:
        counters[SKIPPED_REASON] = SKIP_NO_INPUT
    counters["analysis_calls"] = len(calls)
    seconds = round(time.time() - t0, 3)
    counters["analysis_seconds"] = seconds
    result = {"status": status, "error": error, "analysis": analysis if status == OK else None, "counters": counters,
              "prompt_hash": prompt_hash(), "seconds": seconds, "calls": calls, "reasons": reasons,
              "rejected": rejected, "deadline_s": round(deadline, 1), "block": block}       # v79.19.1 (item 2)
    # v79.19.4 (rule 2): every live record graded 2+ that is no line is in «rejected» with a listed reason
    accounted, counters["analysis_silent_records"] = account_records(result, inp)
    rejected += accounted
    _log("analysis_step", {"prompt_hash": result["prompt_hash"], "status": status,
                           "fallback_skipped_reason": counters.get(SKIPPED_REASON) or "",     # v79.15.2 (item 2)
                           "error": error,
                           "deadline_s": result["deadline_s"], "block": block,                # v79.19.1 (item 2)
                           "silent_records": counters["analysis_silent_records"],                 # v79.19.4 (rule 2)
                           "records_accounted": len(accounted),
                           "seconds": seconds, "calls": calls, "counters": counters,
                           "lines": _line_log(result["analysis"]), "rejected": rejected,
                           "claimant": bool(inp.get("claimant")),                        # v79.15.1 (item 3)
                           "side": inp.get("side") or SIDE_NEUTRAL,                      # v79.15.3 (item 2)
                           "side_why": inp.get("side_why") or "",
                           "tesi_modello": tesi_log(result["analysis"]),                 # v79.15.3 (item 1)
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
LABEL_FRAME_SUB = "addebito del quadro, elementi da provare"      # v79.19.1 (item 1): the frame's charge as the
# subordinate line beside a verified leading one -- «linea non verificata» prints only when no verified line exists
LABEL_FIT_LINE = "norma applicabile ai fatti, linea non verificata"       # v79.19.0 (item 6)
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
                   "legacy_disagrees_esito") + PROSE_COUNTERS + \
    ("objections_fallback_legacy", "records_unprinted", "rulings_printed")     # v79.19.1 (items 3, 6b, 6c)


def undetermined_line() -> Dict[str, Any]:
    return {"articolo": None, "elemento_in_gioco": None, "elemento_id": None, "eid": None, "tesi": "",
            "cita_fatti": [], "cita_norma": [], "stato": UNDETERMINED, "label": LINE_UNDETERMINED,
            "label_sintesi": LINE_UNVERIFIED_LABEL}


def view(result: Optional[Dict[str, Any]], inp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the rendering reads (flag on): {"status", "analysis" (the verified one; the step not ok: the leading line
    «da determinare», nothing else), "sentences" {S-id: text}, "rubriche" {label: rubrica}, "remedies" [labels of the
    fitted norms in the remedy / general-part set], "statements" (the verified affermazioni_di_diritto)}."""
    result = result or {}
    inp = inp or {}
    analysis = result.get("analysis") if result.get("status") == OK else None
    if not isinstance(analysis, dict):
        # v79.19.0 (item 6): the step not run or failed -- still never «da determinare» while a fitted article exists
        analysis = {"linea_principale": fitted_fallback(inp) or undetermined_line(), "linea_subordinata": None,
                    "elementi": {},
                    "affermazioni_di_diritto": [], "rimedi": [n["article"] for n in inp.get("norms") or []
                                                              if n.get("remedy")],
                    "side": inp.get("side") or SIDE_NEUTRAL, "claimant": bool(inp.get("claimant"))}   # v79.19.5 (5)
    v = {"status": result.get("status") or NO_INPUT, "analysis": analysis,
         "sentences": {s["id"]: s["text"] for s in inp.get("sentences") or []},
         "rubriche": {n["article"]: n.get("rubrica") or "" for n in inp.get("norms") or []},
         "remedies": list(analysis.get("rimedi") or []),
         "statements": list(analysis.get("affermazioni_di_diritto") or [])}
    v["relabel"] = defence_relabel(v)                       # v79.19.5 (item 4, row 90): an accused's labels
    if v["relabel"].get("applied"):
        r = v["relabel"]
        _log("analysis_defence_relabel", {"line": (r.get("accusa") or {}).get("articolo"),
                                          "element": (r.get("accusa") or {}).get("elemento_in_gioco"),
                                          "esito": (r.get("accusa") or {}).get("esito_in_gioco"),
                                          "difesa": (r.get("difesa") or {}).get("articolo"),
                                          "nessun_elemento": bool(r.get("nessun_elemento"))})
    v["general_part"] = general_part_section(v, inp)        # v79.19.5 (item 5, row 94a): the records, never a line
    if v["general_part"].get("records"):
        _log("analysis_general_part_section", {"count": len(v["general_part"]["records"]),
                                               "articles": [x["article"] for x in v["general_part"]["records"]],
                                               "title": v["general_part"].get("title")})
    return v


# --- v79.19.5 (item 4, row 90): an accused's proven element is not «Linea principale» ------------------------------------
FIXED_LINE = "frase_fissa"          # the pseudo-line's «stato»: the one fixed sentence when every element is proven
OPEN_ESITI = (CONTESTATO, DA_PROVARE, ESCLUSO)


def defence_cfg() -> Dict[str, Any]:
    return rendering().get("difesa") or {}


def defence_relabel(v: Dict[str, Any]) -> Dict[str, Any]:
    """v79.19.5 (item 4, row 90): the labels of an accused's lines -- {"applied", "accusa", "difesa", "nessun_elemento",
    "subordinata"}. Applied when the analysis's side is «difesa» and its first line is verified with its element in
    play «provato» (esito_in_gioco): that line is the prosecution's case («Tesi dell'accusa», its tesi re-rendered with
    data «rendering.difesa.accusa_tesi» from the line's element and tesi_fatti); «Linea difensiva» is the first verified
    line whose element in play is contestato / da_provare / escluso (the subordinate, today); none: «nessun_elemento»
    (the one fixed sentence), and the subordinate line follows under «subordinata». Which lines exist and their order
    are unchanged; {"applied": False} otherwise."""
    a = v.get("analysis") or {}
    first, second = a.get("linea_principale"), a.get("linea_subordinata")
    if a.get("side") != SIDE_DEFENCE or not first or first.get("stato") != VERIFIED or \
            first.get("esito_in_gioco") != PROVATO:
        return {"applied": False}
    cfg = defence_cfg()
    template = str(cfg.get("accusa_tesi") or "l'elemento «{elemento}» risulta provato da {fatti}")
    accusa = {**first, "tesi": template.format(elemento=first.get("elemento_in_gioco") or "",
                                               fatti=first.get("tesi_fatti") or LABEL_NO_FACT)}
    difesa = second if second and second.get("stato") == VERIFIED and second.get("esito_in_gioco") in OPEN_ESITI \
        else None
    out = {"applied": True, "accusa": accusa, "difesa": difesa, "nessun_elemento": difesa is None, "subordinata": None}
    if difesa is None:
        out["nessun_elemento_line"] = {"articolo": None, "elemento_in_gioco": None, "elemento_id": None, "eid": None,
                                       "tesi": str(cfg.get("nessun_elemento") or ""), "cita_fatti": [],
                                       "cita_norma": [], "stato": FIXED_LINE}
        out["subordinata"] = second
    return out


def objection_lines(v: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """v79.19.5 (items 2, 4): (the line §7 argues, the other) -- the verified leading and subordinate lines; with the
    defence relabel applied, the «Linea difensiva» when one exists, else the subordinate (the qualification / pena
    line) with the accusa line as the other; the accusa line alone when nothing else exists."""
    a = v.get("analysis") or {}
    r = v.get("relabel") or defence_relabel(v)
    if r.get("applied"):
        if r.get("difesa"):
            return r["difesa"], r["accusa"]
        if r.get("subordinata"):
            return r["subordinata"], r["accusa"]
        return r["accusa"], None
    return a.get("linea_principale"), a.get("linea_subordinata")


# --- v79.19.5 (item 5, row 94a): the general-part defences the facts raise, printed as records, never a line ---------------

def general_part_cfg() -> Dict[str, Any]:
    return rendering().get("parte_generale") or {}


def _record_esito(e: Dict[str, Any]) -> str:
    """A record element's esito in the step's words: «provato», «escluso» (non risulta), «contestato», «da provare»."""
    if e.get("contested"):
        return ESITO_LABELS[CONTESTATO]
    status = str(e.get("status") or "")
    if status == "provato":
        return ESITO_LABELS[PROVATO]
    if status == "non risulta":
        return ESITO_LABELS[ESCLUSO]
    return ESITO_LABELS[DA_PROVARE]


def general_part_section(v: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """v79.19.5 (item 5, row 94a): {"title", "side", "records": [{article, rubrica, grade, elements: [{text, esito}]}]}
    -- the live records (inp «graded», analysis_step.inputs: not dropped) graded 2 or more that are no line article:
    for the «difesa» side those whose article is a c.p. article of the remedy / general-part set (is_remedy) or one of
    data «parte_generale.difese» (41, 52, 55, 56, 62, 62-bis c.p.), under «titolo_difesa»; for the «attore» side with
    the frame contractual and no verified line on a claim article (every verified line's article a remedy, or none),
    those of «parte_generale.rimedi» (c.c.), under «titolo_attore». Print only: no gate, no retrieval, no grading
    touched. Empty records: nothing printed."""
    a = v.get("analysis") or {}
    cfg = general_part_cfg()
    side = a.get("side") or SIDE_NEUTRAL
    lines = [x for x in (a.get("linea_principale"), a.get("linea_subordinata")) if x and x.get("articolo")]
    contractual = bool(inp.get("contractual"))
    if side == SIDE_DEFENCE:
        wanted = {(_in_num(x), "cp") for x in cfg.get("difese") or []}
        title, code = str(cfg.get("titolo_difesa") or ""), "cp"
    elif side == SIDE_CLAIMANT and contractual and not any(
            x.get("stato") == VERIFIED and not is_remedy(x["articolo"], contractual) for x in lines):
        wanted = {(_in_num(x), "cc") for x in cfg.get("rimedi") or []}
        title, code = str(cfg.get("titolo_attore") or ""), "cc"
    else:
        return {"title": "", "side": side, "records": []}
    out = []
    for rec in inp.get("graded") or []:
        label = str(rec.get("article") or "")
        try:
            grade = int(rec.get("grade") or 0)
        except (TypeError, ValueError):
            grade = 0
        if grade < 2 or not label or any(same_article(label, x["articolo"]) for x in lines):
            continue
        num, c = art_parts(label)
        if c != code:
            continue
        if not ((num, c) in wanted or (code == "cp" and is_remedy(label))):
            continue
        out.append({"article": label, "rubrica": _flat(rec.get("rubrica")), "grade": grade,
                    "elements": [{"text": _flat(e.get("text")), "esito": _record_esito(e)}
                                 for e in rec.get("elements") or [] if _flat(e.get("text"))]})
    return {"title": title, "side": side, "records": out}


def _in_num(x: Any) -> str:
    return art_parts(f"{x} x")[0] or str(x)


def general_part_text(v: Dict[str, Any]) -> str:
    """The section as printed: the title in bold, one line per record («art. X — rubrica: «elemento» (esito); …»);
    "" when there is nothing to print."""
    g = v.get("general_part") or {}
    cfg = general_part_cfg()
    if not g.get("records"):
        return ""
    rows = [f"**{g.get('title') or ''}**"]
    for r in g["records"]:
        els = str(cfg.get("elementi_sep") or "; ").join(
            str(cfg.get("elemento") or "«{testo}» ({esito})").format(testo=e["text"], esito=e["esito"])
            for e in r.get("elements") or []) or str(cfg.get("nessun_elemento") or LABEL_NO_FACT)
        rubrica = str(cfg.get("rubrica") or " — {rubrica}").format(rubrica=_stop(r["rubrica"])) if r.get("rubrica") \
            else ""
        rows.append("- " + str(cfg.get("record") or "art. {art}{rubrica}: {elementi}").format(
            art=r["article"], rubrica=rubrica, elementi=els) + ".")
    return "\n".join(rows)


def _lines(v: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """The labelled lines §1, §6 and §9 print. v79.19.5 (item 4, row 90): with the defence relabel applied (an accused,
    the first line's element proven) -- «Tesi dell'accusa» the first line, «Linea difensiva» the open line, or the
    fixed sentence and then the subordinate line; the same lines, the same order."""
    a = v.get("analysis") or {}
    r = v.get("relabel") or defence_relabel(v)
    if r.get("applied"):
        cfg = defence_cfg()
        out = [(str(cfg.get("accusa") or "Tesi dell'accusa"), r["accusa"])]
        if r.get("difesa"):
            out.append((str(cfg.get("difesa") or "Linea difensiva"), r["difesa"]))
        else:
            out.append((str(cfg.get("difesa") or "Linea difensiva"), r["nessun_elemento_line"]))
            if r.get("subordinata"):
                out.append((str(cfg.get("subordinata") or LABEL_SUB), r["subordinata"]))
        return out
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
    if x.get("stato") == FIXED_LINE:            # v79.19.5 (item 4): the one fixed sentence, no article
        return _stop(x.get("tesi"))
    if x.get("stato") == UNDETERMINED or not x.get("articolo"):
        return f"{LINE_UNDETERMINED} ({LINE_UNVERIFIED_LABEL})"
    if x.get("stato") == FRAME_LINE:
        if x.get("posizione") == "subordinata":     # v79.19.1 (item 1): beside a verified leading line
            return f"art. {x['articolo']} ({LABEL_FRAME_SUB})"
        return f"art. {x['articolo']} ({LABEL_FRAME_LINE})"
    if x.get("stato") == FIT_LINE:          # v79.19.0 (item 6)
        return f"art. {x['articolo']} — {_stop(x.get('tesi'))} ({LABEL_FIT_LINE})"
    out = f"art. {x['articolo']} — {_stop(x.get('tesi'))}"
    if x.get("elemento_in_gioco") and not x.get("tesi_codice"):
        out += f" ({LABEL_ELEMENT}: «{x['elemento_in_gioco']}»)"
    return out


def main_tesi(v: Dict[str, Any]) -> str:
    """The leading line's printed text (the main-line-repeated check reads it)."""
    return line_text((v.get("analysis") or {}).get("linea_principale"))


def summary_text(v: Dict[str, Any]) -> str:
    """§1: the leading line and the subordinate one, labels and the verified text only."""
    return "\n\n".join(f"{label}: {line_text(x)}." for label, x in _lines(v))


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
        if x.get("stato") in (FALLBACK_LINE, FIT_LINE):     # v79.15.1 (item 2), v79.19.0 (item 6): no element, no fact
            rows.append(f"| {label} | art. {_cell(x['articolo'])} | — | {_cell(_stop(x.get('tesi')))} | — |")
        elif x.get("stato") == VERIFIED:
            rows.append(f"| {label} | art. {_cell(x['articolo'])} | {_cell(x.get('elemento_in_gioco'))} | "
                        f"{_cell(_stop(x.get('tesi')))} | {_cell(_facts_cell(x.get('cita_fatti'), v))} |")
        elif x.get("stato") == FIXED_LINE:          # v79.19.5 (item 4): the fixed sentence
            rows.append(f"| {label} | — | — | {_cell(_stop(x.get('tesi')))} | — |")
        else:
            rows.append(f"| {label} | {_cell(line_text(x))} | — | — | — |")
    section = general_part_text(v)          # v79.19.5 (item 5): after the lines, the general-part defences as records
    return "\n".join(rows) + (f"\n\n{section}" if section else "")


def conclusions_text(v: Dict[str, Any]) -> str:
    """§9: the lines, the leading article's counts and its open and contested elements -- verified, labels only."""
    out = [f"**{label}**: {line_text(x)}." for label, x in _lines(v)]
    for label in line_articles(v)[:1]:
        c = counts_text(v, label)
        if c:
            head, _, rest = c.partition(": ")
            out.append(f"**{head}**: {rest}")
        els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
        open_ = [f"«{e['elemento']}»" for e in els if e.get("esito") == DA_PROVARE]
        contested = [f"«{e['elemento']}» ({esito_text(e)})" for e in els if e.get("esito") == CONTESTATO]
        if open_:
            out.append(f"**{LABEL_TO_PROVE}**: " + ", ".join(open_) + ".")
        if contested:
            out.append(f"**{LABEL_CONTESTED}**: " + ", ".join(contested) + ".")
    return "\n".join(out)


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
                 replace: bool = True, answer: Optional[bool] = None):
        self.v, self.scope, self.site, self.events = v or {}, scope or {}, site, events
        self.replace = replace          # False: §7's objection itself -- a rule with no twin dropped, never replaced
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
    for e in events or []:
        if e.get("esito") == DROPPED:
            key = _SITE_DROPPED.get(e.get("sede"))
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
