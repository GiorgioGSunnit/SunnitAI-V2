#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Drives the preprocessing pipeline from markdown parsing to Neo4j ingestion.
# Coordinates parsing, normalization, extraction, enrichment, and database writes.

import argparse
import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# from .embedding_enrichment import enrich_embeddings  # non-parallel pipeline only
# from .extractor import Node, Rel, extract_from_document  # non-parallel pipeline only
from .parser import parse_file as parse_markdown_file
from .schema import indexing as schema_indexing
from .validate_and_normalize import process_file as validate_and_normalize_process_file
from .write_kg import write_kg_from_extracted
# from .parallel_llm_extractor import extract_from_document_parallel, LLMExtractorConfig  # Commented out - imported conditionally where needed

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def find_markdown_files(data_root: Path) -> List[Path]:
    """Collect markdown files under the data directory in deterministic order."""
    files_dir = data_root / "files"
    search_dir = files_dir if files_dir.exists() else data_root
    return sorted([p for p in search_dir.glob("*.md") if p.is_file()])


def ensure_output_dir(app_dir: Path) -> Path:
    """Create (if needed) and return the directory holding parsed JSON dumps."""
    out_dir = app_dir / "parsed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_normalized_dir(app_dir: Path) -> Path:
    """Create (if needed) and return the directory storing normalized payloads."""
    out_dir = app_dir / "normalized_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_extracted_dir(app_dir: Path) -> Path:
    """Create (if needed) and return the directory for extracted graph facts."""
    out_dir = app_dir / "extracted_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_and_write(source_path_str: str, output_dir_str: str) -> str:
    """Parse a markdown file and write its structured JSON representation."""
    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str)
    result = parse_markdown_file(str(source_path))
    out_path = output_dir / f"{source_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out_path)


def parse_in_parallel(paths: List[Path], output_dir: Path) -> List[Path]:
    """Parse multiple markdown files concurrently and return generated JSON paths."""
    results: List[Path] = []
    with ProcessPoolExecutor() as executor:
        future_to_path = {
            executor.submit(_parse_and_write, str(p), str(output_dir)): p for p in paths
        }
        for future in as_completed(future_to_path):
            src_path = future_to_path[future]
            try:
                out_path = future.result()
                print(f"Parsed: {src_path.name} -> {Path(out_path).name}")
                results.append(Path(out_path))
            except Exception as exc:
                print(f"Failed: {src_path} ({exc})")
    return results


def _write_extracted_jsonl(out_path: Path, nodes: List, rels: List) -> None:                                                                         
    """Persist extracted nodes and relationships into a JSONL file."""
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes:
            rec = {"label": n.label, "key": n.key, "properties": n.properties}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for r in rels:
            rec = {
                "type": r.type,
                "from": {"label": r.from_.label, "key": r.from_.key},
                "to": {"label": r.to.label, "key": r.to.key},
                "properties": r.properties,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_parallel_results_to_jsonl(out_path: Path, results, normalized_data: Dict[str, Any]) -> None:
    """Convert parallel extraction results to JSONL format compatible with write_kg."""
    from .parallel_llm_extractor import ExtractionResult
    
    def write_relation(f, rel_type: str, from_label: str, from_key: Dict, to_label: str, to_key: Dict, properties: Dict = None):
        """Helper to write a relationship."""
        rel_rec = {
            "type": rel_type,
            "from": {"label": from_label, "key": from_key},
            "to": {"label": to_label, "key": to_key},
            "properties": properties or {}
        }
        f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
    
    with out_path.open("w", encoding="utf-8") as f:
        document_id = normalized_data.get("document_id", "unknown")
        
        for result in results:
            # Write Company nodes
            for company in result.entities.companies:
                node_rec = {
                    "label": "Company",
                    "key": {"normalized_name": company.normalized_name},
                    "properties": {
                        "name": company.name,
                        "legal_form": company.legal_form,
                    }
                }
                f.write(json.dumps(node_rec, ensure_ascii=False) + "\n")
                
                # HAS_LEGAL_FORM relationship (Company → Topic)
                if company.legal_form_topic or company.legal_form:
                    topic_label = company.legal_form_topic or company.legal_form
                    write_relation(f, "HAS_LEGAL_FORM", "Company", {"normalized_name": company.normalized_name},
                                 "Topic", {"label": topic_label})
                
                # HAS_CHUNK relationship (Document → Company) - if company is a chunk
                # This is written when company appears as a main chunk type
                chunk = next((c for c in normalized_data.get("chunks", []) if c.get("chunk_id") == result.chunk_id), None)
                if chunk and chunk.get("chunk_type") == "Company":
                    write_relation(f, "HAS_CHUNK", "Document", {"document_id": document_id},
                                 "Company", {"normalized_name": company.normalized_name},
                                 {"source_chunk_type": "Company"})
            
            # Write Institution nodes
            for inst in result.entities.institutions:
                node_rec = {
                    "label": "Institution",
                    "key": {"normalized_name": inst.normalized_name},
                    "properties": {
                        "name": inst.name,
                    }
                }
                if hasattr(inst, 'type') and inst.type:
                    node_rec["properties"]["type"] = inst.type
                f.write(json.dumps(node_rec, ensure_ascii=False) + "\n")
                
                # APPOINTS relationships (Institution → Person)
                if hasattr(inst, 'appointed_persons') and inst.appointed_persons:
                    for appointment in inst.appointed_persons:
                        person_name = appointment.get("name")
                        role = appointment.get("role")
                        if person_name:
                            person_normalized = person_name.lower().replace(" ", "-")
                            write_relation(f, "APPOINTS", "Institution", {"normalized_name": inst.normalized_name},
                                         "Person", {"normalized_name": person_normalized},
                                         {"role": role} if role else {})
            
            # Write Person nodes
            for person in result.entities.persons:
                node_rec = {
                    "label": "Person",
                    "key": {"normalized_name": person.normalized_name},
                    "properties": {
                        "name": person.name,
                    }
                }
                if hasattr(person, 'role') and person.role:
                    node_rec["properties"]["role"] = person.role
                f.write(json.dumps(node_rec, ensure_ascii=False) + "\n")
            
            # Write Penalty nodes and relationships
            for penalty in result.entities.penalties:
                # Create penalty ID from company + reason + type
                penalty_id = f"{penalty.company_name}_{penalty.type}_{penalty.reason[:20]}"
                penalty_id = penalty_id.replace(" ", "_").lower()
                
                penalty_rec = {
                    "label": "Penalty",
                    "key": {"penalty_id": penalty_id},
                    "properties": {
                        "type": penalty.type,
                        "reason": penalty.reason,
                    }
                }
                if penalty.amount:
                    penalty_rec["properties"]["amount"] = penalty.amount
                if penalty.currency:
                    penalty_rec["properties"]["currency"] = penalty.currency
                if penalty.date:
                    penalty_rec["properties"]["date"] = penalty.date
                
                f.write(json.dumps(penalty_rec, ensure_ascii=False) + "\n")
                
                # IMPOSED_ON relationship (Penalty -> Company)
                company_normalized = penalty.company_name.lower().replace(" ", "-")
                rel_rec = {
                    "type": "IMPOSED_ON",
                    "from": {"label": "Penalty", "key": {"penalty_id": penalty_id}},
                    "to": {"label": "Company", "key": {"normalized_name": company_normalized}},
                    "properties": {}
                }
                f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # IMPOSED_BY relationship (Penalty -> Institution)
                if penalty.imposing_authority:
                    inst_normalized = penalty.imposing_authority.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "IMPOSED_BY",
                        "from": {"label": "Penalty", "key": {"penalty_id": penalty_id}},
                        "to": {"label": "Institution", "key": {"normalized_name": inst_normalized}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → Penalty)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "Penalty", "key": {"penalty_id": penalty_id}},
                    "properties": {"source_chunk_type": "Penalty"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
            
            # Write Contract nodes
            for contract in result.entities.contracts:
                if not contract.contract_id:
                    continue
                    
                contract_rec = {
                    "label": "Contract",
                    "key": {"contract_id": contract.contract_id},
                    "properties": {}
                }
                if contract.amount:
                    contract_rec["properties"]["amount"] = contract.amount
                if contract.currency:
                    contract_rec["properties"]["currency"] = contract.currency
                
                f.write(json.dumps(contract_rec, ensure_ascii=False) + "\n")
            
            # Write LegalAct nodes and relationships
            for legal_act in result.entities.legal_acts:
                act_key = {
                    "act_type": legal_act.act_type,
                    "act_number": legal_act.act_number,
                    "act_year": legal_act.act_year
                }
                act_rec = {
                    "label": "LegalAct",
                    "key": act_key,
                    "properties": {}
                }
                f.write(json.dumps(act_rec, ensure_ascii=False) + "\n")
                
                # PUBLISHED_IN relationship (LegalAct → Document)
                write_relation(f, "PUBLISHED_IN", "LegalAct", act_key, 
                             "Document", {"document_id": document_id})
                
                # ISSUED_BY relationship (LegalAct → Institution)
                if legal_act.issuing_institution:
                    inst_normalized = legal_act.issuing_institution.lower().replace(" ", "-")
                    write_relation(f, "ISSUED_BY", "LegalAct", act_key,
                                 "Institution", {"normalized_name": inst_normalized})
                
                # SIGNED_BY relationship (LegalAct → Person)
                if legal_act.signing_person:
                    person_normalized = legal_act.signing_person.lower().replace(" ", "-")
                    write_relation(f, "SIGNED_BY", "LegalAct", act_key,
                                 "Person", {"normalized_name": person_normalized})
                
                # APPOINTS relationships (LegalAct → Person)
                if legal_act.appointed_persons:
                    for person_name in legal_act.appointed_persons:
                        person_normalized = person_name.lower().replace(" ", "-")
                        write_relation(f, "APPOINTS", "LegalAct", act_key,
                                     "Person", {"normalized_name": person_normalized})
                
                # HAS_TOPIC relationships (LegalAct → Topic)
                if legal_act.topics:
                    for topic_label in legal_act.topics:
                        write_relation(f, "HAS_TOPIC", "LegalAct", act_key,
                                     "Topic", {"label": topic_label})
                
                # AMENDS relationship (LegalAct → LegalAct)
                if legal_act.amends_act:
                    parts = legal_act.amends_act.split("-")
                    if len(parts) >= 3:
                        target_key = {
                            "act_type": parts[0],
                            "act_number": int(parts[1]),
                            "act_year": int(parts[2])
                        }
                        write_relation(f, "AMENDS", "LegalAct", act_key,
                                     "LegalAct", target_key)
                
                # REPEALS relationship (LegalAct → LegalAct)
                if legal_act.repeals_act:
                    parts = legal_act.repeals_act.split("-")
                    if len(parts) >= 3:
                        target_key = {
                            "act_type": parts[0],
                            "act_number": int(parts[1]),
                            "act_year": int(parts[2])
                        }
                        write_relation(f, "REPEALS", "LegalAct", act_key,
                                     "LegalAct", target_key)
                
                # REFERENCES relationships (LegalAct → LegalAct)
                if legal_act.references_acts:
                    for ref_act in legal_act.references_acts:
                        parts = ref_act.split("-")
                        if len(parts) >= 3:
                            target_key = {
                                "act_type": parts[0],
                                "act_number": int(parts[1]),
                                "act_year": int(parts[2])
                            }
                            write_relation(f, "REFERENCES", "LegalAct", act_key,
                                         "LegalAct", target_key)
                
                # HAS_CHUNK relationship (Document → LegalAct)
                write_relation(f, "HAS_CHUNK", "Document", {"document_id": document_id},
                             "LegalAct", act_key,
                             {"source_chunk_type": "LegalAct"})
            
            # Write Tender nodes
            for tender in result.entities.tenders:
                tender_rec = {
                    "label": "Tender",
                    "key": {"tender_id": tender.tender_id},
                    "properties": {
                        "title": tender.title,
                        "ref_no": tender.ref_no,
                        "subject": tender.subject,
                        "issuer": tender.issuer,
                        "deadline": tender.deadline
                    }
                }
                # Remove None values
                tender_rec["properties"] = {k: v for k, v in tender_rec["properties"].items() if v is not None}
                f.write(json.dumps(tender_rec, ensure_ascii=False) + "\n")
                
                # ISSUES_TENDER relationship (Institution → Tender)
                if tender.issuer:
                    issuer_normalized = tender.issuer.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "ISSUES_TENDER",
                        "from": {"label": "Institution", "key": {"normalized_name": issuer_normalized}},
                        "to": {"label": "Tender", "key": {"tender_id": tender.tender_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → Tender)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "Tender", "key": {"tender_id": tender.tender_id}},
                    "properties": {"source_chunk_type": "Tender"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
                
                # REMOVED_FROM_TENDER relationships (Tender → Company)
                if tender.removed_companies:
                    for company_name in tender.removed_companies:
                        company_normalized = company_name.lower().replace(" ", "-")
                        write_relation(f, "REMOVED_FROM_TENDER", "Tender", {"tender_id": tender.tender_id},
                                     "Company", {"normalized_name": company_normalized})
            
            # Write Award nodes
            for award in result.entities.awards:
                award_rec = {
                    "label": "Award",
                    "key": {"award_id": award.award_id},
                    "properties": {
                        "title": award.title,
                        "date": award.date,
                        "amount": award.amount,
                        "currency": award.currency,
                        "recommendation_text": award.recommendation_text
                    }
                }
                award_rec["properties"] = {k: v for k, v in award_rec["properties"].items() if v is not None}
                f.write(json.dumps(award_rec, ensure_ascii=False) + "\n")
                
                # AWARDED_TO relationship (Award → Company)
                if award.winner:
                    winner_normalized = award.winner.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "AWARDED_TO",
                        "from": {"label": "Award", "key": {"award_id": award.award_id}},
                        "to": {"label": "Company", "key": {"normalized_name": winner_normalized}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # AWARD_OF relationship (Award → Tender)
                if award.tender_id:
                    rel_rec = {
                        "type": "AWARD_OF",
                        "from": {"label": "Award", "key": {"award_id": award.award_id}},
                        "to": {"label": "Tender", "key": {"tender_id": award.tender_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → Award)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "Award", "key": {"award_id": award.award_id}},
                    "properties": {"source_chunk_type": "Award"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
                
                # RESULTS_IN_CONTRACT relationship (Award → Contract)
                if award.contract_id:
                    write_relation(f, "RESULTS_IN_CONTRACT", "Award", {"award_id": award.award_id},
                                 "Contract", {"contract_id": award.contract_id})
            
            # Write Article nodes
            for article in result.entities.articles:
                article_rec = {
                    "label": "Article",
                    "key": {
                        "parent_act_key": article.parent_act_key,
                        "index": article.index
                    },
                    "properties": {
                        "heading": article.heading,
                        "text_en": article.text_en,
                        "text_ar": article.text_ar,
                        "version": article.version
                    }
                }
                article_rec["properties"] = {k: v for k, v in article_rec["properties"].items() if v is not None}
                f.write(json.dumps(article_rec, ensure_ascii=False) + "\n")
                
                # HAS_ARTICLE relationship (LegalAct → Article)
                # Extract LegalAct key from parent_act_key (format: "Decree-46-2025")
                parts = article.parent_act_key.split("-")
                if len(parts) >= 3:
                    act_type = parts[0]
                    act_number = int(parts[1]) if parts[1].isdigit() else None
                    act_year = int(parts[2]) if parts[2].isdigit() else None
                    
                    if act_number and act_year:
                        rel_rec = {
                            "type": "HAS_ARTICLE",
                            "from": {"label": "LegalAct", "key": {
                                "act_type": act_type,
                                "act_number": act_number,
                                "act_year": act_year
                            }},
                            "to": {"label": "Article", "key": {
                                "parent_act_key": article.parent_act_key,
                                "index": article.index
                            }},
                            "properties": {}
                        }
                        f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_VERSION relationship (Article → Article)
                if article.previous_version_key:
                    parts = article.previous_version_key.rsplit("-", 1)
                    if len(parts) == 2:
                        prev_parent = parts[0]
                        prev_index = int(parts[1]) if parts[1].isdigit() else None
                        if prev_index:
                            write_relation(f, "HAS_VERSION", "Article", 
                                         {"parent_act_key": article.parent_act_key, "index": article.index},
                                         "Article", {"parent_act_key": prev_parent, "index": prev_index},
                                         {"relation": "supersedes"})
            
            # Write Clause nodes
            for clause in result.entities.clauses:
                clause_rec = {
                    "label": "Clause",
                    "key": {
                        "parent_article_key": clause.parent_article_key,
                        "index": clause.index
                    },
                    "properties": {
                        "text_en": clause.text_en,
                        "text_ar": clause.text_ar
                    }
                }
                clause_rec["properties"] = {k: v for k, v in clause_rec["properties"].items() if v is not None}
                f.write(json.dumps(clause_rec, ensure_ascii=False) + "\n")
                
                # HAS_CLAUSE relationship (Article → Clause)
                # Extract Article key from parent_article_key
                rel_rec = {
                    "type": "HAS_CLAUSE",
                    "from": {"label": "Article", "key": {"parent_act_key": clause.parent_article_key.rsplit("-", 1)[0], "index": int(clause.parent_article_key.rsplit("-", 1)[1]) if "-" in clause.parent_article_key else 0}},
                    "to": {"label": "Clause", "key": {
                        "parent_article_key": clause.parent_article_key,
                        "index": clause.index
                    }},
                    "properties": {}
                }
                f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
            
            # Write Meeting nodes
            for meeting in result.entities.meetings:
                meeting_rec = {
                    "label": "Meeting",
                    "key": {"meeting_id": meeting.meeting_id},
                    "properties": {
                        "type": meeting.type,
                        "date": meeting.date,
                        "location": meeting.location
                    }
                }
                meeting_rec["properties"] = {k: v for k, v in meeting_rec["properties"].items() if v is not None}
                f.write(json.dumps(meeting_rec, ensure_ascii=False) + "\n")
                
                # HOLDS_MEETING relationship (Company → Meeting)
                if meeting.company_name:
                    company_normalized = meeting.company_name.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "HOLDS_MEETING",
                        "from": {"label": "Company", "key": {"normalized_name": company_normalized}},
                        "to": {"label": "Meeting", "key": {"meeting_id": meeting.meeting_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # PASSES_RESOLUTION relationships (Meeting → Resolution)
                if meeting.resolutions:
                    for resolution_id in meeting.resolutions:
                        write_relation(f, "PASSES_RESOLUTION", "Meeting", {"meeting_id": meeting.meeting_id},
                                     "Resolution", {"resolution_id": resolution_id})
                
                # HAS_VOTE relationships (Meeting → Vote)
                if meeting.votes:
                    for vote_id in meeting.votes:
                        write_relation(f, "HAS_VOTE", "Meeting", {"meeting_id": meeting.meeting_id},
                                     "Vote", {"vote_id": vote_id})
            
            # Write Auction nodes
            for auction in result.entities.auctions:
                auction_rec = {
                    "label": "Auction",
                    "key": {"auction_id": auction.auction_id},
                    "properties": {
                        "title": auction.title,
                        "date": auction.date,
                        "location": auction.location,
                        "terms": auction.terms,
                        "conditions": auction.conditions
                    }
                }
                auction_rec["properties"] = {k: v for k, v in auction_rec["properties"].items() if v is not None}
                f.write(json.dumps(auction_rec, ensure_ascii=False) + "\n")
                
                # ANNOUNCES_AUCTION relationship (Institution → Auction)
                if auction.organizer:
                    organizer_normalized = auction.organizer.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "ANNOUNCES_AUCTION",
                        "from": {"label": "Institution", "key": {"normalized_name": organizer_normalized}},
                        "to": {"label": "Auction", "key": {"auction_id": auction.auction_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → Auction)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "Auction", "key": {"auction_id": auction.auction_id}},
                    "properties": {"source_chunk_type": "Auction"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
                
                # AUCTION_OF relationships (Auction → Asset)
                if auction.asset_ids:
                    for asset_id in auction.asset_ids:
                        write_relation(f, "AUCTION_OF", "Auction", {"auction_id": auction.auction_id},
                                     "Asset", {"asset_id": asset_id})
                
                # AUCTION_ORGANIZED_FOR relationship (Auction → Company)
                if auction.organized_for_company:
                    company_normalized = auction.organized_for_company.lower().replace(" ", "-")
                    write_relation(f, "AUCTION_ORGANIZED_FOR", "Auction", {"auction_id": auction.auction_id},
                                 "Company", {"normalized_name": company_normalized})
            
            # Write CourtCase nodes
            for court_case in result.entities.court_cases:
                case_rec = {
                    "label": "CourtCase",
                    "key": {
                        "document_id": court_case.document_id,
                        "chunk_id": court_case.chunk_id
                    },
                    "properties": {
                        "case_number": court_case.case_number,
                        "title": court_case.title
                    }
                }
                case_rec["properties"] = {k: v for k, v in case_rec["properties"].items() if v is not None}
                f.write(json.dumps(case_rec, ensure_ascii=False) + "\n")
                
                # ADJUDICATED_BY relationship (CourtCase → Court)
                if court_case.court_name:
                    court_normalized = court_case.court_name.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "ADJUDICATED_BY",
                        "from": {"label": "CourtCase", "key": {
                            "document_id": court_case.document_id,
                            "chunk_id": court_case.chunk_id
                        }},
                        "to": {"label": "Court", "key": {"normalized_name": court_normalized}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → CourtCase)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "CourtCase", "key": {
                        "document_id": court_case.document_id,
                        "chunk_id": court_case.chunk_id
                    }},
                    "properties": {"source_chunk_type": "CourtCase"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
                
                # INVOLVES relationships (CourtCase → LegalParty)
                if court_case.involved_parties:
                    for party_info in court_case.involved_parties:
                        party_name = party_info.get("name")
                        role = party_info.get("role")
                        if party_name:
                            party_normalized = party_name.lower().replace(" ", "-")
                            write_relation(f, "INVOLVES", "CourtCase", 
                                         {"document_id": court_case.document_id, "chunk_id": court_case.chunk_id},
                                         "LegalParty", {"normalized_name": party_normalized},
                                         {"role": role} if role else {})
            
            # Write Document nodes
            for doc in result.entities.documents:
                doc_rec = {
                    "label": "Document",
                    "key": {"document_id": doc.document_id},
                    "properties": {
                        "document_title": doc.document_title,
                        "document_date": doc.document_date,
                        "issue_number": doc.issue_number,
                        "volume_number": doc.volume_number,
                        "document_type": doc.document_type,
                        "language": doc.language
                    }
                }
                doc_rec["properties"] = {k: v for k, v in doc_rec["properties"].items() if v is not None}
                f.write(json.dumps(doc_rec, ensure_ascii=False) + "\n")
                
                # PUBLISHED_IN relationship (LegalAct → Document)
                # This will be written when we have the LegalAct context
                # For now, we write the Document node and the relationship will be created
                # when LegalAct is processed with its document_id reference
            
            # Write Section nodes
            for section in result.entities.sections:
                section_rec = {
                    "label": "Section",
                    "key": {
                        "document_id": section.document_id,
                        "chunk_id": section.chunk_id
                    },
                    "properties": {
                        "title": section.title,
                        "text_en": section.text_en,
                        "text_ar": section.text_ar
                    }
                }
                section_rec["properties"] = {k: v for k, v in section_rec["properties"].items() if v is not None}
                f.write(json.dumps(section_rec, ensure_ascii=False) + "\n")
                
                # HAS_CHUNK relationship (Document → Section)
                doc_chunk_rel = {
                    "type": "HAS_CHUNK",
                    "from": {"label": "Document", "key": {"document_id": document_id}},
                    "to": {"label": "Section", "key": {
                        "document_id": section.document_id,
                        "chunk_id": section.chunk_id
                    }},
                    "properties": {"source_chunk_type": "Section"}
                }
                f.write(json.dumps(doc_chunk_rel, ensure_ascii=False) + "\n")
                
                # MENTIONS relationships (Section → Institution)
                if section.mentioned_institutions:
                    for inst_name in section.mentioned_institutions:
                        inst_normalized = inst_name.lower().replace(" ", "-")
                        write_relation(f, "MENTIONS", "Section",
                                     {"document_id": section.document_id, "chunk_id": section.chunk_id},
                                     "Institution", {"normalized_name": inst_normalized})
            
            # Write Court nodes
            for court in result.entities.courts:
                court_rec = {
                    "label": "Court",
                    "key": {"normalized_name": court.normalized_name},
                    "properties": {"name": court.name}
                }
                f.write(json.dumps(court_rec, ensure_ascii=False) + "\n")
            
            # Write LegalParty nodes
            for legal_party in result.entities.legal_parties:
                party_rec = {
                    "label": "LegalParty",
                    "key": {"normalized_name": legal_party.normalized_name},
                    "properties": {"display_name": legal_party.display_name}
                }
                f.write(json.dumps(party_rec, ensure_ascii=False) + "\n")
            
            # Write Resolution nodes
            for resolution in result.entities.resolutions:
                res_rec = {
                    "label": "Resolution",
                    "key": {"resolution_id": resolution.resolution_id},
                    "properties": {
                        "title": resolution.title,
                        "text": resolution.text,
                        "date": resolution.date
                    }
                }
                res_rec["properties"] = {k: v for k, v in res_rec["properties"].items() if v is not None}
                f.write(json.dumps(res_rec, ensure_ascii=False) + "\n")
                
                # PASSES_RESOLUTION relationship (Meeting → Resolution)
                # This requires matching meeting, will be written when we have meeting context
            
            # Write Vote nodes
            for vote in result.entities.votes:
                vote_rec = {
                    "label": "Vote",
                    "key": {"vote_id": vote.vote_id},
                    "properties": {
                        "motion": vote.motion,
                        "for_count": vote.for_count,
                        "against_count": vote.against_count,
                        "abstain_count": vote.abstain_count,
                        "result": vote.result
                    }
                }
                vote_rec["properties"] = {k: v for k, v in vote_rec["properties"].items() if v is not None}
                f.write(json.dumps(vote_rec, ensure_ascii=False) + "\n")
            
            # Write ChangeOrder nodes
            for change_order in result.entities.change_orders:
                co_rec = {
                    "label": "ChangeOrder",
                    "key": {"change_order_id": change_order.change_order_id},
                    "properties": {
                        "date": change_order.date,
                        "reason": change_order.reason,
                        "amount_delta": change_order.amount_delta,
                        "currency": change_order.currency
                    }
                }
                co_rec["properties"] = {k: v for k, v in co_rec["properties"].items() if v is not None}
                f.write(json.dumps(co_rec, ensure_ascii=False) + "\n")
                
                # AMENDS_CONTRACT relationship (ChangeOrder → Contract)
                if change_order.contract_id:
                    rel_rec = {
                        "type": "AMENDS_CONTRACT",
                        "from": {"label": "ChangeOrder", "key": {"change_order_id": change_order.change_order_id}},
                        "to": {"label": "Contract", "key": {"contract_id": change_order.contract_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # ISSUED_BY relationship (ChangeOrder → Institution)
                if change_order.issuing_institution:
                    inst_normalized = change_order.issuing_institution.lower().replace(" ", "-")
                    write_relation(f, "ISSUED_BY", "ChangeOrder", {"change_order_id": change_order.change_order_id},
                                 "Institution", {"normalized_name": inst_normalized})
            
            # Write Asset nodes
            for asset in result.entities.assets:
                asset_rec = {
                    "label": "Asset",
                    "key": {"asset_id": asset.asset_id},
                    "properties": {
                        "type": asset.type,
                        "description": asset.description,
                        "location": asset.location
                    }
                }
                asset_rec["properties"] = {k: v for k, v in asset_rec["properties"].items() if v is not None}
                f.write(json.dumps(asset_rec, ensure_ascii=False) + "\n")
            
            # Write Complaint nodes
            for complaint in result.entities.complaints:
                complaint_rec = {
                    "label": "Complaint",
                    "key": {"complaint_id": complaint.complaint_id},
                    "properties": {
                        "type": complaint.type,
                        "date": complaint.date,
                        "subject": complaint.subject
                    }
                }
                complaint_rec["properties"] = {k: v for k, v in complaint_rec["properties"].items() if v is not None}
                f.write(json.dumps(complaint_rec, ensure_ascii=False) + "\n")
                
                # FILED_BY relationship (Complaint → Person)
                if complaint.filer:
                    filer_normalized = complaint.filer.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "FILED_BY",
                        "from": {"label": "Complaint", "key": {"complaint_id": complaint.complaint_id}},
                        "to": {"label": "Person", "key": {"normalized_name": filer_normalized}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
                
                # FILED_AGAINST relationship (Complaint → Institution)
                if complaint.against:
                    against_normalized = complaint.against.lower().replace(" ", "-")
                    rel_rec = {
                        "type": "FILED_AGAINST",
                        "from": {"label": "Complaint", "key": {"complaint_id": complaint.complaint_id}},
                        "to": {"label": "Institution", "key": {"normalized_name": against_normalized}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
            
            # Write Correction nodes
            for correction in result.entities.corrections:
                correction_rec = {
                    "label": "Correction",
                    "key": {"correction_id": correction.correction_id},
                    "properties": {
                        "text": correction.text,
                        "date": correction.date
                    }
                }
                correction_rec["properties"] = {k: v for k, v in correction_rec["properties"].items() if v is not None}
                f.write(json.dumps(correction_rec, ensure_ascii=False) + "\n")
                
                # CORRECTS relationship (Correction → Document)
                if correction.document_id:
                    rel_rec = {
                        "type": "CORRECTS",
                        "from": {"label": "Correction", "key": {"correction_id": correction.correction_id}},
                        "to": {"label": "Document", "key": {"document_id": correction.document_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
            
            # Write Addendum nodes
            for addendum in result.entities.addendums:
                addendum_rec = {
                    "label": "Addendum",
                    "key": {"addendum_id": addendum.addendum_id},
                    "properties": {
                        "text": addendum.text,
                        "date": addendum.date
                    }
                }
                addendum_rec["properties"] = {k: v for k, v in addendum_rec["properties"].items() if v is not None}
                f.write(json.dumps(addendum_rec, ensure_ascii=False) + "\n")
                
                # ADDENDS relationship (Addendum → Document)
                if addendum.document_id:
                    rel_rec = {
                        "type": "ADDENDS",
                        "from": {"label": "Addendum", "key": {"addendum_id": addendum.addendum_id}},
                        "to": {"label": "Document", "key": {"document_id": addendum.document_id}},
                        "properties": {}
                    }
                    f.write(json.dumps(rel_rec, ensure_ascii=False) + "\n")
            
            # Write Topic nodes
            for topic in result.entities.topics:
                topic_rec = {
                    "label": "Topic",
                    "key": {"label": topic.label},
                    "properties": {}
                }
                f.write(json.dumps(topic_rec, ensure_ascii=False) + "\n")


async def run_pipeline_parallel(
    limit: Optional[int] = None,
    skip: int = 0,
    max_concurrent: int = 15,
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Execute the full preprocessing pipeline with parallel LLM extraction."""
    normalization_failures: List[str] = []
    ingestion_failures: List[str] = []
    summary: Dict[str, Any] = {
        "parsed_files": 0,
        "normalized_files": 0,
        "extracted_files": 0,
        "ingested_nodes": 0,
        "ingested_rels": 0,
        "ingestion_skipped": False,
        "extraction_mode": "parallel_llm",
        "normalization_failures": normalization_failures,
        "ingestion_failures": ingestion_failures,
    }

    app_dir = Path(__file__).resolve().parents[2]
    data_root = app_dir / "data"
    if not data_root.exists():
        print(f"Data directory not found: {data_root}")
        return summary

    output_dir = ensure_output_dir(app_dir)
    md_files = find_markdown_files(data_root)
    if not md_files:
        print(f"No markdown files found under {data_root}")
        return summary

    # Apply skip and limit
    md_files = md_files[skip:]
    selected = md_files[:limit] if limit else md_files
    skip_msg = f" (skipped first {skip})" if skip > 0 else ""
    print(f"Parsing {len(selected)} of {len(md_files)} files{skip_msg}...")
    parsed_outputs = parse_in_parallel(selected, output_dir)
    print(f"Done. JSON files saved to: {output_dir}")
    summary["parsed_files"] = len(parsed_outputs)

    # Validation & normalization
    normalized_dir = ensure_normalized_dir(app_dir)
    print(f"Validating and normalizing {len(parsed_outputs)} files...")
    normalized_outputs: List[Path] = []
    for out_json in parsed_outputs:
        try:
            norm_path, _stats = validate_and_normalize_process_file(
                out_json, normalized_dir, drop_invalid=False, pretty=True
            )
            print(f"Normalized: {out_json.name} -> {Path(norm_path).name}")
            normalized_outputs.append(Path(norm_path))
        except Exception as exc:
            print(f"Failed normalize: {out_json} ({exc})")
            normalization_failures.append(f"{out_json}: {exc}")
    print(f"Done. Normalized JSON files saved to: {normalized_dir}")
    summary["normalized_files"] = len(normalized_outputs)

        # Parallel LLM Extraction
    extracted_dir = ensure_extracted_dir(app_dir)
    logger.info(f"🚀 Extracting with PARALLEL SPECIALIZED LLMs from {len(normalized_outputs)} files...")
    logger.info(f"   Model: {llm_model}, Max concurrent: {max_concurrent}")
    
    config = LLMExtractorConfig(
        model=llm_model,
        temperature=0.0,
        min_confidence=0.7,
        use_few_shot=True,
        use_chain_of_thought=True
    )
    
    extracted_outputs: List[Path] = []
    for i, norm_json in enumerate(normalized_outputs, 1):
        try:
            logger.info(f"[{i}/{len(normalized_outputs)}] Processing {norm_json.name}...")
            
            with norm_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            
            num_chunks = len(payload.get("chunks", []))
            logger.info(f"  Document has {num_chunks} chunks")
            
            # Use parallel extraction
            results = await extract_from_document_parallel(
                payload,
                config,
                max_concurrent=max_concurrent
            )
            
            # Convert to JSONL format
            out_path = extracted_dir / f"{norm_json.stem}.extracted.jsonl"
            _write_parallel_results_to_jsonl(out_path, results, payload)
            
            total_entities = sum(
                len(r.entities.companies) + 
                len(r.entities.institutions) + 
                len(r.entities.penalties) +
                len(r.entities.persons) +
                len(r.entities.contracts) +
                len(r.entities.legal_acts) +
                len(r.entities.tenders) +
                len(r.entities.awards) +
                len(r.entities.articles) +
                len(r.entities.clauses) +
                len(r.entities.meetings) +
                len(r.entities.auctions) +
                len(r.entities.court_cases) +
                len(r.entities.documents)
                for r in results
            )
            logger.info(f"✅ Extracted: {norm_json.name} -> {out_path.name}")
            logger.info(f"   Total entities: {total_entities} from {len(results)} chunks")
            extracted_outputs.append(out_path)
        except Exception as exc:
            logger.error(f"❌ Failed extract: {norm_json} - {exc}", exc_info=True)
    
    logger.info(f"Done. Extracted JSONL files saved to: {extracted_dir}")
    summary["extracted_files"] = len(extracted_outputs)

    if not extracted_outputs:
        print("No extracted files to ingest into Neo4j.")
        summary["ingestion_skipped"] = True
        return summary

    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print(
            "Neo4j credentials missing (.env must define NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)."
        )
        print("Skipping ingestion step.")
        summary["ingestion_skipped"] = True
        return summary

    try:
        print("Ensuring Neo4j constraints and indexes...")
        schema_indexing.main()
    except Exception as exc:
        print(f"Failed to prepare Neo4j indexes ({exc})")
        summary["ingestion_skipped"] = True
        return summary

    print("Writing extracted graph facts into Neo4j...")
    total_nodes = 0
    total_rels = 0
    for extracted_file in extracted_outputs:
        try:
            n_written, r_written = write_kg_from_extracted(
                extracted_file, database=neo4j_database
            )
            print(
                f"Ingested: {extracted_file.name} -> nodes={n_written}, rels={r_written}"
            )
            total_nodes += n_written
            total_rels += r_written
        except Exception as exc:
            print(f"Failed ingest: {extracted_file} ({exc})")
            ingestion_failures.append(f"{extracted_file}: {exc}")

    print(
        f"Neo4j ingestion completed. Total nodes written: {total_nodes}, total relationships written: {total_rels}"
    )
    summary["ingested_nodes"] = total_nodes
    summary["ingested_rels"] = total_rels

    try:
        print("Reapplying Neo4j index definitions after ingestion...")
        schema_indexing.main()
    except Exception as exc:
        print(f"Failed to refresh Neo4j indexes post-ingestion ({exc})")

    return summary


def run_pipeline(limit: Optional[int] = None, skip: int = 0) -> Dict[str, Any]:
    """Execute the full preprocessing pipeline and return execution metrics."""
    normalization_failures: List[str] = []
    ingestion_failures: List[str] = []
    summary: Dict[str, Any] = {
        "parsed_files": 0,
        "normalized_files": 0,
        "extracted_files": 0,
        "ingested_nodes": 0,
        "ingested_rels": 0,
        "ingestion_skipped": False,
        "extraction_mode": "heuristic",
        "normalization_failures": normalization_failures,
        "ingestion_failures": ingestion_failures,
    }

    app_dir = Path(__file__).resolve().parents[2]
    data_root = app_dir / "data"
    if not data_root.exists():
        print(f"Data directory not found: {data_root}")
        return summary

    output_dir = ensure_output_dir(app_dir)
    md_files = find_markdown_files(data_root)
    if not md_files:
        print(f"No markdown files found under {data_root}")
        return summary

    # Apply skip and limit
    md_files = md_files[skip:]
    selected = md_files[:limit] if limit else md_files
    skip_msg = f" (skipped first {skip})" if skip > 0 else ""
    print(f"Parsing {len(selected)} of {len(md_files)} files{skip_msg}...")
    parsed_outputs = parse_in_parallel(selected, output_dir)
    print(f"Done. JSON files saved to: {output_dir}")
    summary["parsed_files"] = len(parsed_outputs)

    # Validation & normalization
    normalized_dir = ensure_normalized_dir(app_dir)
    print(f"Validating and normalizing {len(parsed_outputs)} files...")
    normalized_outputs: List[Path] = []
    for out_json in parsed_outputs:
        try:
            norm_path, _stats = validate_and_normalize_process_file(
                out_json, normalized_dir, drop_invalid=False, pretty=True
            )
            print(f"Normalized: {out_json.name} -> {Path(norm_path).name}")
            normalized_outputs.append(Path(norm_path))
        except Exception as exc:
            print(f"Failed normalize: {out_json} ({exc})")
            normalization_failures.append(f"{out_json}: {exc}")
    print(f"Done. Normalized JSON files saved to: {normalized_dir}")
    summary["normalized_files"] = len(normalized_outputs)

    # Extraction
    extracted_dir = ensure_extracted_dir(app_dir)
    print(f"Extracting graph facts from {len(normalized_outputs)} files...")
    extracted_outputs: List[Path] = []
    # Non-parallel extraction disabled — requires embedding_enrichment and extractor.Node/Rel
    # for norm_json in normalized_outputs:
    #     try:
    #         with norm_json.open("r", encoding="utf-8") as f:
    #             payload = json.load(f)
    #         nodes, rels = extract_from_document(payload)
    #         print("🚨 🚨 nodes before enrichment: ", nodes)
    #         enrich_embeddings(nodes)
    #         out_path = extracted_dir / f"{norm_json.stem}.extracted.jsonl"
    #         _write_extracted_jsonl(out_path, nodes, rels)
    #         print(f"Extracted: {norm_json.name} -> {out_path.name}")
    #         extracted_outputs.append(out_path)
    #     except Exception as exc:
    #         print(f"Failed extract: {norm_json} ({exc})")
    print(f"Done. Extracted JSONL files saved to: {extracted_dir}")
    summary["extracted_files"] = len(extracted_outputs)

    if not extracted_outputs:
        print("No extracted files to ingest into Neo4j.")
        summary["ingestion_skipped"] = True
        return summary

    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print(
            "Neo4j credentials missing (.env must define NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)."
        )
        print("Skipping ingestion step.")
        summary["ingestion_skipped"] = True
        return summary

    try:
        print("Ensuring Neo4j constraints and indexes...")
        schema_indexing.main()
    except Exception as exc:
        print(f"Failed to prepare Neo4j indexes ({exc})")
        summary["ingestion_skipped"] = True
        return summary

    print("Writing extracted graph facts into Neo4j...")
    total_nodes = 0
    total_rels = 0
    for extracted_file in extracted_outputs:
        try:
            n_written, r_written = write_kg_from_extracted(
                extracted_file, database=neo4j_database
            )
            print(
                f"Ingested: {extracted_file.name} -> nodes={n_written}, rels={r_written}"
            )
            total_nodes += n_written
            total_rels += r_written
        except Exception as exc:
            print(f"Failed ingest: {extracted_file} ({exc})")
            ingestion_failures.append(f"{extracted_file}: {exc}")

    print(
        f"Neo4j ingestion completed. Total nodes written: {total_nodes}, total relationships written: {total_rels}"
    )
    summary["ingested_nodes"] = total_nodes
    summary["ingested_rels"] = total_rels

    try:
        print("Reapplying Neo4j index definitions after ingestion...")
        schema_indexing.main()
    except Exception as exc:
        print(f"Failed to refresh Neo4j indexes post-ingestion ({exc})")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse markdown documents from app/data into JSON, normalize, extract graph facts, and ingest into Neo4j."                                  
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of documents to process (default: all).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N documents (default: 0). Useful for resuming processing.",
    )
    parser.add_argument(
        "--parallel-llm",
        action="store_true",
        help="Use parallel specialized LLM extractors instead of heuristic extraction",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=15,
        help="Maximum concurrent LLM API calls (for --parallel-llm mode)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for extraction (for --parallel-llm mode)",
    )
    args = parser.parse_args()
    
    if args.parallel_llm:
        from .parallel_llm_extractor import extract_from_document_parallel, LLMExtractorConfig
        result = asyncio.run(run_pipeline_parallel(
            limit=args.limit,
            skip=args.skip,
            max_concurrent=args.max_concurrent,
            llm_model=args.llm_model
        ))
    else:
        result = run_pipeline(limit=args.limit, skip=args.skip)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
