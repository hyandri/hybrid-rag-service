# app/ingest.py
import json
import os
from langchain_core.documents import Document

DOMAIN_KEYWORDS = {
    "cardiology": [
        "heart", "cardiac", "myocardial", "arrhythmia", "angioplasty",
        "atherosclerosis", "cardiomyopathy", "echocardiography",
        "electrocardiogram", "hypertension", "ischemia", "pericarditis",
        "valvular", "ventricular"
    ],
    "oncology": [
        "cancer", "tumor", "neoplasm", "metastasis", "chemotherapy",
        "radiation therapy", "immunotherapy", "carcinoma",
        "sarcoma", "leukemia", "lymphoma", "melanoma"
    ],
    "neuroscience":  [
        "alzheimer", "neuron", "brain", "cognitive", "neuroinflammation", "dementia"
    ],
    "immunology":    ["cytokine", "inflammation", "immune", "TNF", "interleukin", "autoimmune"],
    "reproductive":  ["ovary", "follicle", "estrous", "reproductive", "fertility", "oocyte"],
}

SKIP_SECTIONS = {
    "supplementary information", "supplementary material",
    "acknowledgements", "acknowledgments", "funding",
    "competing interests", "author contributions",
    "conflict of interest", "abbreviations", "references",
    "data availability", "ethics"
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general"

def extract_year(paper: dict) -> str:
    doi = paper.get("doi", "")
    if "-026-" in doi or "/2026" in doi:
        return "2026"
    if "-025-" in doi or "/2025" in doi:
        return "2025"
    if "-024-" in doi or "/2024" in doi:
        return "2024"
    return "unknown"

def load_and_chunk_docs(json_path="pmc_cardiology_oncology.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {json_path}")
    chunks = []

    for paper in papers:
        pmcid    = paper.get("pmcid", "unknown")
        title    = paper.get("title", "")
        journal  = paper.get("journal", "")
        doi      = paper.get("doi", "")
        keywords = ", ".join(paper.get("keywords", []))
        abstract = paper.get("abstract", "")
        sections = paper.get("sections", {})
        
        full_text_sample = abstract + " ".join(list(sections.values())[:2])
        domain = detect_domain(full_text_sample)
        year   = extract_year(paper)

        base_metadata = {
            "source":   pmcid,
            "title":    title,
            "journal":  journal,
            "doi":      doi,
            "keywords": keywords,
            "domain":   domain,
            "year":     year,
        }

        # Abstract chunk
        if abstract.strip():
            chunks.append(Document(
                page_content=f"Title: {title}\n\nAbstract:\n{abstract[:1500]}",
                metadata={**base_metadata, "section": "abstract"}
            ))

        # Section chunks — bigger size, skip junk sections
        SKIP_SECTIONS = {
            "supplementary information", "supplementary material",
            "acknowledgements", "acknowledgments", "funding",
            "competing interests", "author contributions",
            "conflict of interest", "abbreviations", "references"
        }

        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            if any(skip in section_name.lower() for skip in SKIP_SECTIONS):
                continue
            if len(section_text) < 100:  # skip tiny sections
                continue

            sub_chunks = split_text(section_text, chunk_size=2000, overlap=300)

            for j, sub in enumerate(sub_chunks):
                chunk_text = (
                    f"Title: {title}\n"
                    f"Section: {section_name}\n\n"
                    f"{sub}"
                )
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **base_metadata,
                        "section":   section_name,
                        "chunk_idx": j
                    }
                ))

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def split_text(text: str, chunk_size: int = 200, overlap: int = 300) -> list[str]:
    """Simple character-level splitter that respects sentence boundaries roughly."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for last period/newline before end
            break_at = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end)
            )
            if break_at > start + overlap:
                end = break_at + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c.strip()]