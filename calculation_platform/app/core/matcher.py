"""Deterministic free-text-to-calculator matching.

Scores a short description (e.g. what a user typed) against each
calculator's declared routing metadata — no LLM, no fuzzy ML, just
normalized phrase and token overlap, so results are fully reproducible
and explainable (each candidate reports which terms matched).

Evidence, strongest to weakest:
  +3 for each distinct multi-word phrase (from keywords, aliases, tags,
     or intent_examples) whose whole normalized form appears in the query;
  +1 for each remaining query token found in the calculator's vocabulary
     (all tokens of its keywords/aliases/tags/intent_examples/name/category);
  -4 once if the query covers any declared negative_example (a request
     that looks related but should NOT route here) — enough to knock out
     weak/incidental matches while genuinely strong matches survive.

Candidates that end at or below zero are dropped entirely. The response's
`status` says whether the top candidate is a clear winner ("matched"),
tied with another ("ambiguous" — a router would ask the user to choose),
or nothing matched at all ("no_match").
"""

import re
import unicodedata
from typing import Iterable, List, Set, Tuple

from ..schemas.calculator_definition import CalculatorDefinition
from ..schemas.match_result import MatchCandidate, MatchResponse

_PHRASE_HIT_SCORE = 3
_TOKEN_HIT_SCORE = 1
_NEGATIVE_PENALTY = 4
_MIN_TOKEN_LENGTH = 3

# Function words that carry no routing signal — without this filter, an
# intent example like "totale di una fattura" would make "una" vocabulary
# and let completely unrelated sentences score token hits.
_STOPWORDS = frozenset({
    "una", "uno", "con", "per", "del", "della", "dei", "delle", "degli",
    "sul", "sulla", "sugli", "alla", "alle", "che", "non", "come", "gli",
    "nel", "nella", "the", "and", "for", "with", "how", "much",
})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> Set[str]:
    return {
        t for t in _normalize(text).split()
        if len(t) >= _MIN_TOKEN_LENGTH and t not in _STOPWORDS
    }


def _input_summary(spec) -> dict:
    return {
        "name": spec.name,
        "type": spec.type,
        "unit": spec.unit,
        "description": spec.description,
    }


def _routing_terms(definition: CalculatorDefinition) -> List[str]:
    return [*definition.keywords, *definition.aliases, *definition.tags, *definition.intent_examples]


def _score_definition(
    definition: CalculatorDefinition, normalized_query: str, query_tokens: Set[str]
) -> Tuple[int, List[str]]:
    score = 0
    matched_terms: List[str] = []
    consumed_tokens: Set[str] = set()

    # Phrase evidence: distinct multi-word terms appearing whole in the
    # query. Single-word terms are vocabulary, not phrases — otherwise a
    # one-word tag would outweigh a genuine phrase match.
    seen_phrases: Set[str] = set()
    for phrase in _routing_terms(definition):
        normalized_phrase = _normalize(phrase)
        if " " not in normalized_phrase or normalized_phrase in seen_phrases:
            continue
        seen_phrases.add(normalized_phrase)
        if normalized_phrase in normalized_query:
            score += _PHRASE_HIT_SCORE
            matched_terms.append(phrase)
            consumed_tokens |= _tokens(phrase)

    vocabulary: Set[str] = _tokens(definition.name) | _tokens(definition.category)
    for phrase in _routing_terms(definition):
        vocabulary |= _tokens(phrase)

    for token in sorted(query_tokens - consumed_tokens):
        if token in vocabulary:
            score += _TOKEN_HIT_SCORE
            matched_terms.append(token)

    # One-time penalty if the query covers a declared negative example —
    # subset check on tokens so word order doesn't matter.
    for negative in definition.negative_examples:
        negative_tokens = _tokens(negative)
        if negative_tokens and negative_tokens <= query_tokens:
            score -= _NEGATIVE_PENALTY
            break

    return score, matched_terms


def match_query(query: str, definitions: Iterable[CalculatorDefinition]) -> MatchResponse:
    normalized_query = _normalize(query)
    query_tokens = _tokens(query)

    candidates: List[MatchCandidate] = []
    for definition in definitions:
        score, matched_terms = _score_definition(definition, normalized_query, query_tokens)
        if score <= 0:
            continue
        candidates.append(MatchCandidate(
            calculator_id=definition.id,
            name=definition.name,
            description=definition.description or "",
            score=score,
            matched_terms=matched_terms,
            ambiguity_notes=definition.ambiguity_notes,
            required_inputs=[_input_summary(s) for s in definition.inputs if s.required],
            optional_inputs=[_input_summary(s) for s in definition.inputs if not s.required],
            requires_period=definition.strategy == "date_split_interest",
            supports_tax_year=bool(definition.regime_selector),
        ))

    candidates.sort(key=lambda c: (-c.score, c.calculator_id))

    if not candidates:
        status = "no_match"
    elif len(candidates) == 1 or candidates[0].score > candidates[1].score:
        status = "matched"
    else:
        status = "ambiguous"

    return MatchResponse(query=query, status=status, candidates=candidates)
