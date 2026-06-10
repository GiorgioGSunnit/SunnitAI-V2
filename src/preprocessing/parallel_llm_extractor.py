"""Parallel specialized LLM extraction for the preprocessing pipeline.

Dispatches 6 concurrent LLM calls per document chunk, each targeting a focused
subset of entity types, to avoid token truncation on large structured-output schemas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity models — 25 types, matching _write_parallel_results_to_jsonl contract
# ---------------------------------------------------------------------------

class LegalAct(BaseModel):
    act_type: str = Field(..., description="Type of legal act, e.g. Decree, Law, Decision")
    act_number: int = Field(..., description="Act number")
    act_year: int = Field(..., description="Year the act was issued")
    issuing_institution: Optional[str] = None
    signing_person: Optional[str] = None
    appointed_persons: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    amends_act: Optional[str] = Field(None, description="Format: type-number-year")
    repeals_act: Optional[str] = Field(None, description="Format: type-number-year")
    references_acts: Optional[List[str]] = Field(None, description="Each entry: type-number-year")


class Article(BaseModel):
    parent_act_key: str = Field(..., description="Format: act_type-act_number-act_year")
    index: int = Field(..., description="Article number within the act")
    heading: Optional[str] = None
    text_en: Optional[str] = Field(None, description="2-3 sentence summary only")
    text_ar: Optional[str] = Field(None, description="2-3 sentence summary only")
    version: Optional[str] = None
    previous_version_key: Optional[str] = Field(None, description="Format: parent_act_key-index")


class Clause(BaseModel):
    parent_article_key: str = Field(..., description="Format: parent_act_key-article_index")
    index: int = Field(..., description="Clause number within the article")
    text_en: Optional[str] = Field(None, description="2-3 sentence summary only")
    text_ar: Optional[str] = Field(None, description="2-3 sentence summary only")


class Resolution(BaseModel):
    resolution_id: str
    title: Optional[str] = None
    text: Optional[str] = Field(None, description="2-3 sentence summary only")
    date: Optional[str] = None


class Institution(BaseModel):
    normalized_name: str = Field(..., description="Lowercase kebab-case identifier")
    name: str
    type: Optional[str] = None
    appointed_persons: Optional[List[Dict[str, str]]] = Field(
        None, description="List of {name, role} dicts"
    )


class Company(BaseModel):
    normalized_name: str = Field(..., description="Lowercase kebab-case identifier")
    name: str
    legal_form: Optional[str] = None
    legal_form_topic: Optional[str] = None


class Person(BaseModel):
    normalized_name: str = Field(..., description="Lowercase kebab-case identifier")
    name: str
    role: Optional[str] = None


class Court(BaseModel):
    normalized_name: str = Field(..., description="Lowercase kebab-case identifier")
    name: str


class LegalParty(BaseModel):
    normalized_name: str = Field(..., description="Lowercase kebab-case identifier")
    display_name: str


class CourtCase(BaseModel):
    document_id: str
    chunk_id: str
    case_number: Optional[str] = None
    title: Optional[str] = None
    court_name: Optional[str] = None
    involved_parties: Optional[List[Dict[str, str]]] = Field(
        None, description="List of {name, role} dicts"
    )


class Penalty(BaseModel):
    company_name: str
    type: str
    reason: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    imposing_authority: Optional[str] = None


class Complaint(BaseModel):
    complaint_id: str
    type: Optional[str] = None
    date: Optional[str] = None
    subject: Optional[str] = None
    filer: Optional[str] = None
    against: Optional[str] = None


class Tender(BaseModel):
    tender_id: str
    title: Optional[str] = None
    ref_no: Optional[str] = None
    subject: Optional[str] = None
    issuer: Optional[str] = None
    deadline: Optional[str] = None
    removed_companies: Optional[List[str]] = None


class Award(BaseModel):
    award_id: str
    title: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    recommendation_text: Optional[str] = Field(None, description="2-3 sentence summary only")
    winner: Optional[str] = None
    tender_id: Optional[str] = None
    contract_id: Optional[str] = None


class Contract(BaseModel):
    contract_id: str
    amount: Optional[float] = None
    currency: Optional[str] = None


class ChangeOrder(BaseModel):
    change_order_id: str
    date: Optional[str] = None
    reason: Optional[str] = None
    amount_delta: Optional[float] = None
    currency: Optional[str] = None
    contract_id: Optional[str] = None
    issuing_institution: Optional[str] = None


class Auction(BaseModel):
    auction_id: str
    title: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    terms: Optional[str] = Field(None, description="2-3 sentence summary only")
    conditions: Optional[str] = Field(None, description="2-3 sentence summary only")
    organizer: Optional[str] = None
    asset_ids: Optional[List[str]] = None
    organized_for_company: Optional[str] = None


class Asset(BaseModel):
    asset_id: str
    type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


class Document(BaseModel):
    document_id: str
    document_title: Optional[str] = None
    document_date: Optional[str] = None
    issue_number: Optional[str] = None
    volume_number: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None


class Section(BaseModel):
    document_id: str
    chunk_id: str
    title: Optional[str] = None
    text_en: Optional[str] = Field(None, description="2-3 sentence summary only")
    text_ar: Optional[str] = Field(None, description="2-3 sentence summary only")
    mentioned_institutions: Optional[List[str]] = None


class Topic(BaseModel):
    label: str


class Correction(BaseModel):
    correction_id: str
    text: Optional[str] = Field(None, description="2-3 sentence summary only")
    date: Optional[str] = None
    document_id: Optional[str] = None


class Addendum(BaseModel):
    addendum_id: str
    text: Optional[str] = Field(None, description="2-3 sentence summary only")
    date: Optional[str] = None
    document_id: Optional[str] = None


class Meeting(BaseModel):
    meeting_id: str
    type: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    company_name: Optional[str] = None
    resolutions: Optional[List[str]] = Field(None, description="List of resolution_id strings")
    votes: Optional[List[str]] = Field(None, description="List of vote_id strings")


class Vote(BaseModel):
    vote_id: str
    motion: Optional[str] = None
    for_count: Optional[int] = None
    against_count: Optional[int] = None
    abstain_count: Optional[int] = None
    result: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregate container (all 25 entity lists)
# ---------------------------------------------------------------------------

class DocumentEntities(BaseModel):
    legal_acts: List[LegalAct] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)
    clauses: List[Clause] = Field(default_factory=list)
    resolutions: List[Resolution] = Field(default_factory=list)
    institutions: List[Institution] = Field(default_factory=list)
    companies: List[Company] = Field(default_factory=list)
    persons: List[Person] = Field(default_factory=list)
    courts: List[Court] = Field(default_factory=list)
    legal_parties: List[LegalParty] = Field(default_factory=list)
    court_cases: List[CourtCase] = Field(default_factory=list)
    penalties: List[Penalty] = Field(default_factory=list)
    complaints: List[Complaint] = Field(default_factory=list)
    tenders: List[Tender] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)
    contracts: List[Contract] = Field(default_factory=list)
    change_orders: List[ChangeOrder] = Field(default_factory=list)
    auctions: List[Auction] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    corrections: List[Correction] = Field(default_factory=list)
    addendums: List[Addendum] = Field(default_factory=list)
    meetings: List[Meeting] = Field(default_factory=list)
    votes: List[Vote] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Group container models (one schema per LLM call group)
# ---------------------------------------------------------------------------

class Group1Entities(BaseModel):
    """Legal framework: LegalAct, Article, Clause, Resolution."""
    legal_acts: List[LegalAct] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)
    clauses: List[Clause] = Field(default_factory=list)
    resolutions: List[Resolution] = Field(default_factory=list)


class Group2Entities(BaseModel):
    """Parties: Institution, Company, Person, Court, LegalParty."""
    institutions: List[Institution] = Field(default_factory=list)
    companies: List[Company] = Field(default_factory=list)
    persons: List[Person] = Field(default_factory=list)
    courts: List[Court] = Field(default_factory=list)
    legal_parties: List[LegalParty] = Field(default_factory=list)


class Group3Entities(BaseModel):
    """Proceedings: CourtCase, Penalty, Complaint."""
    court_cases: List[CourtCase] = Field(default_factory=list)
    penalties: List[Penalty] = Field(default_factory=list)
    complaints: List[Complaint] = Field(default_factory=list)


class Group4Entities(BaseModel):
    """Commercial: Tender, Award, Contract, ChangeOrder, Auction, Asset."""
    tenders: List[Tender] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)
    contracts: List[Contract] = Field(default_factory=list)
    change_orders: List[ChangeOrder] = Field(default_factory=list)
    auctions: List[Auction] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)


class Group5Entities(BaseModel):
    """Documents: Document, Section, Topic, Correction, Addendum."""
    documents: List[Document] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    corrections: List[Correction] = Field(default_factory=list)
    addendums: List[Addendum] = Field(default_factory=list)


class Group6Entities(BaseModel):
    """Events: Meeting, Vote."""
    meetings: List[Meeting] = Field(default_factory=list)
    votes: List[Vote] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level pipeline types
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    chunk_id: str
    entities: DocumentEntities = Field(default_factory=DocumentEntities)
    failed_groups: List[str] = Field(default_factory=list)


class LLMExtractorConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    min_confidence: float = 0.7
    use_few_shot: bool = True
    use_chain_of_thought: bool = True
    # Limits concurrent chunks, not total LLM calls (each chunk spawns 6 calls)
    max_concurrent: int = 3


# ---------------------------------------------------------------------------
# Group system prompts
# ---------------------------------------------------------------------------

_SHARED_HEADER = (
    "You extract structured entities from a legal document chunk provided as JSON. "
    "The chunk has fields: chunk_id, chunk_type, title, act, text_en, page_number, "
    "and optionally indices, clauses. "
    "Rules:\n"
    "- For ALL text fields (text_en, text_ar, recommendation_text, terms, conditions, "
    "text, description, subject, reason): write a 2-3 sentence summary. "
    "Do NOT echo or reproduce the source text verbatim.\n"
    "- normalized_name fields: use lowercase kebab-case "
    "(e.g. 'Ministry of Health' → 'ministry-of-health').\n"
    "- Return only entities explicitly present in the chunk. "
    "Return empty lists for entity types not found in this chunk.\n"
    "- Output valid JSON matching the provided schema exactly."
)

_GROUP1_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: LegalAct, Article, Clause, Resolution.\n"
    "LegalAct: act_type (e.g. Decree/Law/Decision/Resolution), act_number (int), act_year (int) "
    "are required.\n"
    "Article.parent_act_key format: 'act_type-act_number-act_year' (e.g. 'Decree-46-2025').\n"
    "Clause.parent_article_key format: 'parent_act_key-article_index' "
    "(e.g. 'Decree-46-2025-1').\n"
    "amends_act, repeals_act, references_acts: use 'type-number-year' format.\n"
    "Article.previous_version_key: 'parent_act_key-article_index' of the superseded version."
)

_GROUP2_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: Institution, Company, Person, Court, LegalParty.\n"
    "Institution.appointed_persons: list of {\"name\": str, \"role\": str} dicts for persons "
    "appointed by or within this institution in the chunk.\n"
    "LegalParty.display_name: the party's full legal name as it appears in the document.\n"
    "Do not extract a Person if only a role (not a name) is mentioned."
)

_GROUP3_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: CourtCase, Penalty, Complaint.\n"
    "CourtCase.document_id: the gazette/publication document identifier. "
    "CourtCase.chunk_id: the chunk's chunk_id field value.\n"
    "CourtCase.involved_parties: list of {\"name\": str, \"role\": str} dicts.\n"
    "Penalty: company_name, type, and reason are all required — skip if any are absent.\n"
    "Complaint.complaint_id: derive a short unique id from subject + date "
    "(e.g. 'complaint-tax-fraud-2025-03-01')."
)

_GROUP4_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: Tender, Award, Contract, ChangeOrder, Auction, Asset.\n"
    "Award.recommendation_text: 2-3 sentence summary — do NOT reproduce the full text.\n"
    "Auction.terms and Auction.conditions: 2-3 sentence summaries each.\n"
    "Contract.contract_id: use an explicit reference number; skip the Contract entirely "
    "if no identifier is present in the chunk.\n"
    "Asset.asset_id: derive from type + location if no explicit id is stated "
    "(e.g. 'vehicle-sabah-al-ahmad-2025')."
)

_GROUP5_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: Document, Section, Topic, Correction, Addendum.\n"
    "Document.document_id: the gazette/publication identifier "
    "(issue number + year, e.g. 'gazette-1234-2025').\n"
    "Section.document_id: same document identifier. Section.chunk_id: the chunk's chunk_id value.\n"
    "Section.text_en and Section.text_ar: 2-3 sentence summaries — do NOT echo source text.\n"
    "Topic.label: a concise legal topic keyword "
    "(e.g. 'Civil Liability', 'Data Protection', 'Banking Regulation').\n"
    "Correction.correction_id and Addendum.addendum_id: derive from date + document reference."
)

_GROUP6_SYSTEM = (
    _SHARED_HEADER + "\n\n"
    "Extract ONLY: Meeting, Vote.\n"
    "Meeting.resolutions: list of resolution_id strings for resolutions passed at this meeting.\n"
    "Meeting.votes: list of vote_id strings for votes cast at this meeting.\n"
    "Vote.vote_id: derive from meeting_id + a sequential index or motion keyword "
    "(e.g. 'meeting-agm-2025-01-vote-1').\n"
    "Vote.for_count, against_count, abstain_count: integer counts only if explicitly stated; "
    "leave null otherwise."
)


# ---------------------------------------------------------------------------
# Group configuration table
# (group_name, schema_class, system_prompt, max_tokens, entity_fields)
# ---------------------------------------------------------------------------

_GROUP_CONFIGS: List[tuple] = [
    (
        "group1", Group1Entities, _GROUP1_SYSTEM, 500,
        ["legal_acts", "articles", "clauses", "resolutions"],
    ),
    (
        "group2", Group2Entities, _GROUP2_SYSTEM, 500,
        ["institutions", "companies", "persons", "courts", "legal_parties"],
    ),
    (
        "group3", Group3Entities, _GROUP3_SYSTEM, 500,
        ["court_cases", "penalties", "complaints"],
    ),
    (
        "group4", Group4Entities, _GROUP4_SYSTEM, 700,
        ["tenders", "awards", "contracts", "change_orders", "auctions", "assets"],
    ),
    (
        "group5", Group5Entities, _GROUP5_SYSTEM, 500,
        ["documents", "sections", "topics", "corrections", "addendums"],
    ),
    (
        "group6", Group6Entities, _GROUP6_SYSTEM, 500,
        ["meetings", "votes"],
    ),
]


def _build_chains(config: LLMExtractorConfig) -> List[tuple]:
    """Build one structured-output chain per group from config.

    Chains are built at call time (not module level) so config.model and
    LLM_BASE_URL are resolved fresh on each pipeline run.

    Returns list of (group_name, chain, system_prompt, entity_fields) tuples.
    """
    base_kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
    }
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        base_kwargs["base_url"] = base_url

    chains = []
    for group_name, schema, system_prompt, max_tokens, entity_fields in _GROUP_CONFIGS:
        llm = ChatOpenAI(**{**base_kwargs, "max_tokens": max_tokens})
        chain = llm.with_structured_output(schema, method="json_mode")
        chains.append((group_name, chain, system_prompt, entity_fields))
    return chains


def _invoke_group(chain: Any, system_prompt: str, chunk_json: str) -> Any:
    """Synchronous single-group extraction call (run in thread via asyncio.to_thread)."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Document chunk:\n{chunk_json}"),
    ]
    return chain.invoke(messages)


async def _extract_chunk(
    chunk: Dict[str, Any],
    chains: List[tuple],
    semaphore: asyncio.Semaphore,
) -> ExtractionResult:
    """Run all 6 group extractions for a single chunk concurrently.

    Each group call runs in a thread pool via asyncio.to_thread for reliability
    on custom base URLs that may not support async. Group failures are isolated —
    a failed group yields empty lists for its entity types; other groups still merge.
    """
    async with semaphore:
        chunk_id = chunk.get("chunk_id", "unknown")
        chunk_json = json.dumps(chunk, ensure_ascii=False)

        tasks = [
            asyncio.to_thread(_invoke_group, chain, system_prompt, chunk_json)
            for _, chain, system_prompt, _ in chains
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        entities = DocumentEntities()
        failed_groups: List[str] = []

        for (group_name, _, _, entity_fields), result in zip(chains, group_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Group %s failed for chunk %s: %s", group_name, chunk_id, result
                )
                failed_groups.append(group_name)
                continue
            for field_name in entity_fields:
                setattr(entities, field_name, getattr(result, field_name, []))

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            failed_groups=failed_groups,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_from_document_parallel(
    payload: Dict[str, Any],
    config: LLMExtractorConfig,
    max_concurrent: int = 3,
) -> List[ExtractionResult]:
    """Extract entities from all chunks in a normalized document payload.

    Dispatches 6 concurrent LLM calls per chunk (one per entity group).
    max_concurrent limits chunks processed simultaneously — total concurrent
    LLM calls = max_concurrent × 6. Default of 3 keeps total calls at 18,
    suitable for rate-limited or custom-endpoint deployments.

    Args:
        payload:        Normalized document dict with a 'chunks' list.
        config:         Extraction config (model, temperature, etc.).
        max_concurrent: Max concurrent chunks. Overrides config.max_concurrent
                        when provided explicitly.

    Returns:
        List of ExtractionResult, one per successfully processed chunk.
        Chunks that raise an unhandled exception are logged and omitted.
    """
    chunks = payload.get("chunks", [])
    if not chunks:
        return []

    chains = _build_chains(config)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _extract_chunk(chunk, chains, semaphore)
        for chunk in chunks
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[ExtractionResult] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            chunk_id = chunks[i].get("chunk_id", f"index-{i}")
            logger.error("Chunk extraction failed for %s: %s", chunk_id, result)
        else:
            results.append(result)

    return results
