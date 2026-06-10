# Parses legal document markdown files into structured chunks for downstream extraction.
# Normalizes whitespace, classifies sections, and emits metadata-rich payloads.

import json
import re
import uuid

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}
ORDINAL_WORDS = r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)"


def normalize_space(s):
    """Collapse whitespace and trim leading or trailing spaces."""
    return re.sub(r"\s+", " ", s or "").strip()


def strip_orig(s):
    """Remove <orig> tag blocks from text."""
    return re.sub(r"<orig>.*?</orig>", "", s or "", flags=re.S)


def to_int_safe(s):
    """Convert number strings or words to integers when possible."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return NUMBER_WORDS.get(s.lower())


def strip_ar_tags(s):
    """Strip Arabic <orig> tags while retaining the surrounding text."""
    return re.sub(r"<orig>.*?</orig>", "", s or "", flags=re.S)


PAGE_METADATA_BLOCK = re.compile(r"<page_metadata>.*?</page_metadata>", re.S)
PAGE_MARKER_BLOCK = re.compile(r"<page_(?:start|end)>.*?</page_(?:start|end)>", re.S)
PAGE_START_TAG = re.compile(r"<page_start>(\d+)</page_start>")
PAGE_HEADER_BLOCK = re.compile(r"<header>.*?</header>", re.S)
PAGE_RULE_LINE = re.compile(r"(?m)^\s*---\s*$")


def strip_page_artifacts(s):
    """Remove pagination metadata and headers from markdown content."""
    if not s:
        return ""
    cleaned = PAGE_METADATA_BLOCK.sub("", s)
    cleaned = PAGE_HEADER_BLOCK.sub("", cleaned)
    cleaned = PAGE_MARKER_BLOCK.sub("", cleaned)
    cleaned = PAGE_RULE_LINE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def parse_document_metadata(md_text):
    """Parse the document metadata block into a dictionary."""
    m = re.search(
        r"<document_metadata>\s*(.*?)\s*</document_metadata>",
        md_text,
        flags=re.S | re.I,
    )
    if not m:
        return {}
    block = m.group(1).strip()
    if block.startswith("{") or block.startswith("["):
        try:
            return json.loads(block)
        except Exception:
            pass
    meta = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


H_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def split_sections(md_text):
    """Split markdown text into sections keyed by heading level and title."""
    sections = []
    matches = list(H_PATTERN.finditer(md_text))
    i = 0
    total = len(matches)
    while i < total:
        m = matches[i]
        level = len(m.group(1))
        titles = [m.group(2).strip()]
        start = m.end()
        section_start_pos = m.start()
        j = i + 1
        end = matches[j].start() if j < total else len(md_text)
        body = md_text[start:end].strip()
        while not body and j < total and len(matches[j].group(1)) == level:
            titles.append(matches[j].group(2).strip())
            start = matches[j].end()
            j += 1
            end = matches[j].start() if j < total else len(md_text)
            body = md_text[start:end].strip()
        sections.append({
            "level": level,
            "title": " ".join(titles),
            "body": body,
            "position": section_start_pos
        })
        i = j
    return sections


ARTICLE_PATTERNS = [
    re.compile(
        r"(?i)^Article\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)\b"
    ),
    re.compile(
        r"(?i)^\(Article\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)\)\s*$"
    ),
    re.compile(r"(?i)^Article\s*\(\s*(\d+)\s*\)\s*[:\.]?\s*$"),
    re.compile(r"(?i)^Article\s+(\d+)\s*[:\.]?\s*$"),
    re.compile(
        r"(?i)^Article\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*:\s*$"
    ),
]

LEGAL_ACT_PAT = re.compile(
    r"(?i)\b(Decree|Decision|Ministerial\s+Resolution|Committee\s+Resolution|Regulation|Decree-Law)\b"
)
ANNOUNCEMENT_PAT = re.compile(r"(?i)\bAnnouncement(s)?\b")
NOTICE_PAT = re.compile(r"(?i)^(Notice|Notification|Note)\s*:?\s*$")
CRIMINAL_J_PAT = re.compile(r"(?i)\bCriminal\s+Judgment\b")
COURT_CASE_PAT = re.compile(r"(?i)\bCase\s+No\.\b")
COMPANY_LIQ_PAT = re.compile(
    r"(?i)\bCompany\s+Under\s+Liquidation\b|Dissolution\s+and\s+Liquidation\s+of\s+the\s+Company|First-Time\s+Company\s+Registration|One-?Person\s+Company"
)
GA_MINUTES_PAT = re.compile(
    r"(?i)\bMinutes\s+of\s+the\s+(Ordinary|Extraordinary)\s+General\s+Assembly\s+Meeting\b"
)
AUCTION_ANN_PAT = re.compile(
    r"(?i)Property\s+Sale\s+by\s+Public\s+Auction|Auction\s+(Conditions|Terms)"
)
CHANGE_ORDER_PAT = re.compile(
    r"(?i)(Extension\s+Order|Change\s+Order|Initial\s+Insurance\s*/\s*Extension|Initial\s+Bid\s+Bond\s*/\s*Extension)"
)
ADMIN_PENALTY_PAT = re.compile(
    r"(?i)Imposition\s+of\s+an\s+Administrative\s+Penalty|Administrative\s+Penalty"
)
GRIEVANCE_PAT = re.compile(r"(?i)Complaints\s*/\s*Grievances|Grievance")
LEGAL_ENTITY_PAT = re.compile(
    r"(?i)Legal\s+Entity\b|Limited\s+Liability\s+Company|Joint\s+Stock|One-?Person\s+Company",
    re.I,
)
AMENDMENT_TEXT_PAT = re.compile(
    r"(?i)(Text\s+of\s+the\s+Article\s+(Before|After)\s+Amendment|After\s+Amendment:|Before\s+Amendment:|Text\s+After\s+Amendment|Article\s+Text\s+Before\s+Amendment)"
)
ANY_AMEND_PAT = re.compile(r"(?i)\b(Before|After)\s+Amendment\b")
PREAMBLE_PAT = re.compile(r"(?i)^Having reviewed:")
SCHEDULE_PAT = re.compile(r"(?i)^Schedule\s*\([A-Z]\)\s*$")
TABLE_PAT = re.compile(
    r"(?i)^(Table\s+No\.\s*\(.*?\)(?:\s*(?:-|–)\s*Continued|.*Cont.*)?|Table\s*\([A-Z]\))\s*$"
)
BUDGET_PAT = re.compile(
    r"(?i)Expenditures\s+by\s+Chapter(s)?|Budget|Revenues\s+by\s+Chapter(s)?|Increase\s+in\s+Expenditures\s+over\s+Revenues|Fiscal\s+Year\s+\d{4}/\d{4}"
)
DECIDED_PAT = re.compile(
    r"(?i)^(Has\s+Decided|It\s+is\s+decided)\b|^%s\s*:" % ORDINAL_WORDS
)
INSTITUTION_HDR_PAT = re.compile(
    r"(?i)^(Ministry|Council|Authority|Kuwait\s+Oil\s+Company|Public\s+Authority|Directorate|General\s+Fire\s+Force|General\s+Presidency|Kuwait\s+Municipality|Secretar(?:y|iat)\s+General|Central\s+Agency\s+for\s+Public\s+Tenders|Kuwait\s+National\s+Petroleum\s+Company|Kuwait\s+Integrated\s+Petroleum\s+Industries\s+Company|Kuwait\s+Aviation\s+Fuelling\s+Company|Amiri\s+Diwan)\b"
)
STATEMENT_PAT = re.compile(r"(?i)^Statement\s*$")
SUBJECT_PAT = re.compile(r"(?i)^(Subject|Regarding)\s*:|^Subject$")
EXPLANATORY_MEMO_PAT = re.compile(r"(?i)Explanatory\s+Memorandum")
RECOMM_AWARD_PAT = re.compile(r"(?i)Recommendations\s*/\s*Award(ing)?")
PUBLICATION_ADDENDUM_PAT = re.compile(r"(?i)Publication\s*/\s*Addendum")
PUBLICATION_MINUTES_PAT = re.compile(
    r"(?i)Publication\s*/\s*Preliminary\s+Meeting\s+Minutes(?:\s+and\s+Addendum)?"
)
VOTING_RESULT_PAT = re.compile(r"(?i)^Voting\s+Result\s*:")
VOTING_BREAKDOWN_PAT = re.compile(r"(?i)^(For|Against|Abstention|Abstentions)\s*:")
PROPERTY_DESC_PAT = re.compile(r"(?i)Property\s+Description")
TENDER_OFFERING_PAT = re.compile(
    r"(?i)^Tender\s+Offering\s*/\s*General|Tender\s+Issuance\s*/\s*Public"
)
INSPECTION_PAT = re.compile(r"(?i)^\*?Inspection\s*:")


def number_from_word(val):
    """Transform numeric words into integers; fallback to None."""
    try:
        return int(val) if str(val).isdigit() else NUMBER_WORDS.get(str(val).lower())
    except Exception:
        return None


def is_article(title_clean):
    """Check whether a title denotes an article and return its numeric index."""
    for pat in ARTICLE_PATTERNS:
        m = pat.search(title_clean)
        if m:
            val = m.group(1)
            return True, number_from_word(val) if val else None
    return False, None


def classify_section(title):
    """Assign a chunk type and attributes based on heuristics over the title."""
    title_clean = normalize_space(strip_orig(title))

    ok, num = is_article(title_clean)
    if ok:
        return "Article", {"article": num}

    if AMENDMENT_TEXT_PAT.search(title_clean) or ANY_AMEND_PAT.search(title_clean):
        ver = "After" if re.search(r"(?i)After\s+Amendment", title_clean) else "Before"
        return "ArticleVersion", {"version": ver}

    m = LEGAL_ACT_PAT.search(title_clean)
    if m:
        act_label = m.group(1)
        ctype = re.sub(r"[\s\-]+", "", act_label)
        return ctype, {"act_type": ctype, "act_type_label": act_label}

    if CRIMINAL_J_PAT.search(title_clean) or COURT_CASE_PAT.search(title_clean):
        return "CriminalJudgment", {}
    if COMPANY_LIQ_PAT.search(title_clean):
        return "CompanyUnderLiquidation", {}
    if GA_MINUTES_PAT.search(title_clean) or PUBLICATION_MINUTES_PAT.search(
        title_clean
    ):
        return "GeneralAssemblyMinutes", {}
    if AUCTION_ANN_PAT.search(title_clean):
        return "Auction", {}
    if CHANGE_ORDER_PAT.search(title_clean):
        return "ChangeOrder", {}
    if ADMIN_PENALTY_PAT.search(title_clean):
        return "AdministrativePenalty", {}
    if GRIEVANCE_PAT.search(title_clean):
        return "Grievance", {}
    if LEGAL_ENTITY_PAT.search(title_clean):
        return "LegalEntitySection", {}
    if PREAMBLE_PAT.search(title_clean):
        return "PreambleReferences", {}
    if (
        SCHEDULE_PAT.search(title_clean)
        or TABLE_PAT.search(title_clean)
        or re.fullmatch(r"\d{1,2}", title_clean)
    ):
        return "Schedule", {}
    if BUDGET_PAT.search(title_clean):
        return "Budget", {}
    if DECIDED_PAT.search(title_clean):
        return "DecisionSection", {}
    if INSTITUTION_HDR_PAT.search(title_clean):
        return "InstitutionSection", {}
    if ANNOUNCEMENT_PAT.search(title_clean) or NOTICE_PAT.search(title_clean):
        return "Announcement", {}
    if STATEMENT_PAT.search(title_clean):
        return "Statement", {}
    if SUBJECT_PAT.search(title_clean):
        return "SubjectSection", {}
    if EXPLANATORY_MEMO_PAT.search(title_clean):
        return "ExplanatoryMemorandum", {}
    if RECOMM_AWARD_PAT.search(title_clean):
        return "Award", {}
    if PUBLICATION_ADDENDUM_PAT.search(title_clean):
        return "Addendum", {}
    if VOTING_RESULT_PAT.search(title_clean):
        return "VotingResult", {}
    if VOTING_BREAKDOWN_PAT.search(title_clean):
        return "VotingBreakdown", {}
    if PROPERTY_DESC_PAT.search(title_clean):
        return "AuctionSection", {}
    if TENDER_OFFERING_PAT.search(title_clean) or re.search(
        r"(?i)Bid\s+Bond", title_clean
    ):
        return "TenderSection", {}
    if INSPECTION_PAT.search(title_clean):
        return "InspectionSection", {}
    if re.fullmatch(r"(?i)Correction", title_clean):
        return "Correction", {}

    return "Section", {}


def extract_act_numbers(text):
    """Extract act number and year hints from a section's text."""
    text_plain = normalize_space(strip_orig(text))
    m = re.search(
        r"(?i)\bNo\.?\s*([A-Za-z0-9\-\/]+)\s*(?:of|\/)\s*(\d{4})\b", text_plain
    )
    if m:
        return {"act_number": m.group(1), "act_year": m.group(2)}
    m = re.search(r"(?i)\b(\d{1,5})\s*\/\s*(\d{4})\b", text_plain)
    if m:
        return {"act_number": m.group(1), "act_year": m.group(2)}
    return {"act_number": None, "act_year": None}


CLAUSE_PATS = [
    re.compile(r"(?m)^\s*[\(]?([0-9]+)[\)]?[\.\-\)]\s+(.*)$"),
    re.compile(
        r"(?m)^(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s*:\s+(.*)$",
        re.I,
    ),
]


def extract_clauses(text_en):
    """Identify clause indices and English text snippets within an article."""
    clauses = []
    for pat in CLAUSE_PATS:
        for m in pat.finditer(text_en):
            idx = m.group(1)
            text = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            try:
                idx_n = int(idx) if str(idx).isdigit() else idx
            except Exception:
                idx_n = idx
            clauses.append({"index": idx_n, "text_en": text.strip()})
    seen = set()
    uniq = []
    for c in clauses:
        key = (str(c["index"]), c["text_en"][:60])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def build_page_map(md_text):
    """Build a map of text positions to page numbers."""
    page_map = []
    for m in PAGE_START_TAG.finditer(md_text):
        page_num = int(m.group(1))
        pos = m.start()
        page_map.append((pos, page_num))
    return page_map


def get_page_number(position, page_map):
    """Return the page number for a given text position."""
    if not page_map:
        return None
    current_page = page_map[0][1]
    for pos, page_num in page_map:
        if position >= pos:
            current_page = page_num
        else:
            break
    return current_page


def parse_markdown(md_text):
    """Convert markdown content into a document metadata dict and chunk list."""
    meta = parse_document_metadata(md_text)
    page_map = build_page_map(md_text)
    sections = split_sections(md_text)
    chunks = []
    for sec in sections:
        ctype, attrs = classify_section(sec["title"])
        body = strip_page_artifacts(sec["body"])
        text_en_raw = strip_ar_tags(body)
        text_en = normalize_space(text_en_raw)
        act_hint = extract_act_numbers(sec["title"] + " " + body)
        act = {
            "act_type": attrs.get("act_type"),
            "act_number": act_hint.get("act_number"),
            "act_year": act_hint.get("act_year"),
        }
        if attrs.get("act_type_label"):
            act["act_type_label"] = attrs["act_type_label"]
        indices = {}
        article_index = attrs.get("article")
        if article_index is not None:
            indices["article"] = article_index
        if "version" in attrs and attrs["version"] is not None:
            indices["version"] = attrs["version"]
        if not text_en:
            continue
        page_num = get_page_number(sec["position"], page_map)
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "chunk_type": ctype,
            "title": sec["title"],
            "act": act,
            "text_en": text_en,
        }
        if page_num is not None:
            chunk["page_number"] = page_num
        if indices:
            chunk["indices"] = indices
        if ctype == "Article":
            clauses = extract_clauses(text_en_raw)
            if clauses:
                chunk["clauses"] = clauses
        chunks.append(chunk)
    return {"document": meta, "chunks": chunks}


def parse_file(path):
    """Read a markdown file from disk and parse it into structured data."""
    with open(path, "r", encoding="utf-8") as f:
        md_text = f.read()
    return parse_markdown(md_text)
