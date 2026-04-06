import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from ..rag.ai_chat import chat_model
from .schema.schema import KnowledgeGraphExtraction
from .write_kg import save_to_neo4j

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A graph node produced by the document extraction pipeline."""
    label: str
    key: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rel:
    """A directed graph relationship produced by the document extraction pipeline."""
    type: str
    from_: Node
    to: Node
    properties: Dict[str, Any] = field(default_factory=dict)


def extract_from_document(payload: Dict[str, Any]) -> tuple:
    """Extract Node and Rel objects from a normalized document payload.

    Returns (nodes: List[Node], rels: List[Rel]).
    Stub — full implementation pending.
    """
    return [], []

# Reuse the shared chat_model from ai_chat instead of duplicating LLM config
structured_llm = chat_model.with_structured_output(
    KnowledgeGraphExtraction,
    method="function_calling"
)

# 2. System Prompt definition
system_prompt = """
You are an expert Data Engineer specializing in Knowledge Graph extraction for Legal and Procurement domains.
Your task is to extract structured data from legal documents.
You are given a dictionary of a chunk of a legal document, like the following:
      "chunk_id": "bbfc34fa-2b2a-40c4-a928-4a09315133d5",
      "chunk_type": "Announcement",
      "title": "Announcement <orig>إعلان</orig>",
      "act":
        {
          "act_type": null,
          "act_number": "71",
          "act_year": "2020"
        },
      "text_en": "this is the text of the chunk",
      "page_number": 5

---
### GUIDELINES:
1. **Nodes**: Extract Organizations, Events, People, Legal Docs, Locations, Legal Concepts, Roles, Dates, Legal Actions, and Document Sections.
   - **ID Normalization**: Always convert to lowercase kebab-case (e.g., "Ministry of Health" -> "ministry-of-health").
   - **Embedding Text**: Create a comprehensive paragraph describing the entity based on the context. This is crucial for vector retrieval.
   - **Page Number**: For EVENT nodes, always include the "page_number" field. For other node types, use -1 if not applicable.

2. **Relationships**: STRICTLY use the allowed types:
   ANNOUNCES, PARTICIPATES_IN, AFFILIATED_WITH, REGULATED_BY, HAS_ROLE, LOCATED_AT, REFERS_TO, OCCURS_ON, CONTAINS, REGULATES, IS_PART_OF.

3. **Accuracy**: In the properties, include any relevant detail (date, references, role type, location type, etc.).

---
### NODE TYPES:
1. **ORGANIZATION**: Government bodies, Companies, Ministries.
2. **EVENT**: Tenders (Supply, Construction), Auctions, Legal Proceedings, Hearings.
3. **PERSON**: Individual names (only if critical signatories, debtors, or participants).
4. **LEGAL_DOC**: Laws, Decrees, Articles, Contracts, or any formal legal document referenced.
5. **LOCATION**: Physical or legal locations (e.g., "City Court", "Ministry Headquarters").
6. **LEGAL_CONCEPT**: Legal principles or topics (e.g., "Civil Liability", "Intellectual Property").
7. **ROLE**: Positions or roles (e.g., "CEO", "Judge", "Liquidator").
8. **DATE**: Specific dates or deadlines (e.g., "2025-11-15").
9. **LEGAL_ACTION**: Legal actions or procedures (e.g., "Appeal", "Notification", "Sanction").
10. **DOCUMENT_SECTION**: Sections, articles, or clauses within a legal document (e.g., "Article 5", "Clause 3.2").

---
### RELATIONSHIP RULES (STRICT ENFORCEMENT):
You must ONLY create relationships that follow these rules. Do not invent others.

1. **ANNOUNCES**:
   - Source: ORGANIZATION
   - Target: EVENT or LEGAL_DOC
   - Example: "Ministry of Health" ANNOUNCES "Tender No. 123"

2. **PARTICIPATES_IN**:
   - Source: ORGANIZATION or PERSON
   - Target: EVENT
   - Example: "Company X" PARTICIPATES_IN "Auction Y" as "Winner"

3. **AFFILIATED_WITH**:
   - Source: PERSON or ORGANIZATION
   - Target: ORGANIZATION
   - Example: "CEO John" AFFILIATED_WITH "Company X" or "Subsidiary A" AFFILIATED_WITH "Parent Corp"

4. **REGULATED_BY**:
   - Source: EVENT or ORGANIZATION or LEGAL_ACTION
   - Target: LEGAL_DOC
   - Example: "Tender X" REGULATED_BY "Law No. 49"

5. **HAS_ROLE**:
   - Source: PERSON
   - Target: ROLE
   - Example: "John Doe" HAS_ROLE "CEO" in "Company X"

6. **LOCATED_AT**:
   - Source: EVENT or ORGANIZATION or LEGAL_ACTION
   - Target: LOCATION
   - Example: "Trial ABC" LOCATED_AT "City Court"

7. **REFERS_TO**:
   - Source: LEGAL_DOC or DOCUMENT_SECTION
   - Target: LEGAL_CONCEPT
   - Example: "Decree 123" REFERS_TO "Civil Liability"

8. **OCCURS_ON**:
   - Source: EVENT or LEGAL_ACTION
   - Target: DATE
   - Example: "Hearing XYZ" OCCURS_ON "2025-11-15"

9. **CONTAINS**:
   - Source: LEGAL_DOC
   - Target: DOCUMENT_SECTION
   - Example: "Contract ABC" CONTAINS "Clause 3.2"

10. **REGULATES**:
    - Source: LEGAL_DOC
    - Target: LEGAL_CONCEPT
    - Example: "Law 200" REGULATES "Data Protection"

11. **IS_PART_OF**:
    - Source: EVENT or DOCUMENT_SECTION
    - Target: EVENT or LEGAL_DOC
    - Example: "Preliminary Hearing" IS_PART_OF "Trial XYZ"

---
### Input JSON follows:
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{json}")]
)

# 3. Chain creation
extractor_chain = prompt | structured_llm

test_json_extracts = [
    """{
      "chunk_id": "90047357-1390-4f2c-b377-88ce7aee96aa",
      "chunk_type": "Section",
      "title": "Regarding the amendment of the name of Manhattan College",
      "act": {
        "act_type": null,
        "act_number": "15",
        "act_year": "1979"
      },
      "text_en": "in the United States of America on the lists of the National Bureau for Academic Accreditation and Quality Assurance of Education * Having reviewed Decree-Law No. 15 of 1979 concerning Civil Service and its amending laws. * And Amiri Decree No. (417) of 2010 issued on October 25, 2010, concerning the establishment of the National Bureau for Academic Accreditation and Quality Assurance of Education . * And Ministerial Decision No. (272) issued on October 3, 2012, concerning allowing Kuwaiti students to enroll for study outside the State of Kuwait only in universities for which a ministerial decision is issued by the National Bureau for Academic Accreditation and Quality Assurance of Education . * And Academic Decision No. (2014/25) issued on June 3, 2014, concerning allowing study in Bachelor's and Postgraduate programs in the United States of America. * And the decision of the Board of Directors of the National Bureau for Academic Accreditation and Quality Assurance of Education No. (7) issued on December 31, 2014, which includes the mechanism for issuing decisions related to identifying higher education institutions outside the State of Kuwait whose graduates' qualifications are allowed to be accredited. * And the decision of the Board of Directors of the National Bureau for Academic Accreditation and Quality Assurance of Education (2014/5) concerning the determination of criteria for selecting higher education institutions outside the State of Kuwait. * And based on the letter from the Cultural Office in Washington dated January 30, 2025. * And based on the recommendation of the Committee for Proposing Standards, Reviewing, and Updating Lists of Higher Education Institutions outside the State of Kuwait in its third meeting held on March 11, 2025. * And based on the proposal of the Director General of the National Bureau for Academic Accreditation and Quality Assurance of Education . * And as required by the public interest. (It has been decided)",
      "page_number": 9
    }
        """,
    """
        {
      "chunk_id": "98469ff0-c7f2-487a-8456-8dbaeafe6364",
      "chunk_type": "Article",
      "title": "Article One",
      "act": {
        "act_type": null,
        "act_number": null,
        "act_year": null
      },
      "text_en": "The name of Manhattan College shall be amended to Manhattan University, as stated in Academic Decision No. (2014/25) issued on June 3, 2014, concerning the list of higher education institutions allowed for enrollment to study Bachelor's and Postgraduate programs in the United States of America.",
      "page_number": 9,
      "indices": {
        "article": 1
      }
    }
        """,
    """
        {
      "chunk_id": "9bc8abe6-0329-4c35-8ff6-e153a3b13593",
      "chunk_type": "Decision",
      "title": "Decision No. 40/1 - R/2025",
      "act": {
        "act_type": "Decision",
        "act_number": "3",
        "act_year": "2025",
        "act_type_label": "Decision"
      },
      "text_en": "Governor of the Central Bank of Kuwait Having reviewed Article (21) of Law No. (32) of 1968 concerning Currency, the Central Bank of Kuwait, and the Regulation of the Banking Profession, and its amendments. And on Articles (7) and (10) of Ministerial Resolution No. 1984 regarding subjecting exchange companies to the supervision of the Central Bank of Kuwait . And based on the request of Al-Mabani Exchange Company dated 11/3/2025 regarding the amendment of its data in the register of exchange companies at the Central Bank of Kuwait . Decided",
      "page_number": 9
    }
            """,
    """
               {
      "chunk_id": "46a59229-b4b3-4fd1-a0eb-6bbf4f3616bf",
      "chunk_type": "Article",
      "title": "Article One",
      "act": {
        "act_type": null,
        "act_number": null,
        "act_year": null
      },
      "text_en": "The following amendment shall be noted in the register of exchange companies at the Central Bank of Kuwait regarding the data of Al-Mabani Exchange Company : * Transfer of the company's branch located in Sabah Al-Ahmad Sea City – Plot (1) – Parcel (1035) – Arena Mall Building / Abraj Al-Fawz Company – Floor (00) – Unit No. (10) to Jleeb Al-Shuyoukh area – Plot (2) – Parcel (112\\02) – Street (145) – Ayed Mudhi Dukhnan Al-Gharbiya Building – Floor (00).",
      "page_number": 10,
      "indices": {
        "article": 1
      }
      """,
]


def process_document_batch(json_data=None, batch_size: int = 5):
    """Process document texts in parallel batches."""
    json_extracts = json_data if json_data else test_json_extracts

    batch_inputs = [{"json": t} for t in json_extracts]

    logger.info("Starting batch processing of %d pages...", len(batch_inputs))

    try:
        results = extractor_chain.batch(batch_inputs, config={"max_concurrency": batch_size})

        total_nodes = 0
        total_rels = 0

        for i, res in enumerate(results):
            logger.info("Page %d: %d nodes, %d relationships", i + 1, len(res.nodes), len(res.relationships))
            save_to_neo4j(res)
            total_nodes += len(res.nodes)
            total_rels += len(res.relationships)

        logger.info("Batch complete. Total nodes: %d, total edges: %d", total_nodes, total_rels)
        return results

    except Exception as e:
        logger.error("Batch processing failed: %s", e, exc_info=True)
        return None


if __name__ == "__main__":
    process_document_batch()

    