# app/ingest.py
import json
import os
from langchain_core.documents import Document

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

        # Base metadata shared across all chunks of this paper
        base_metadata = {
            "source":   pmcid,
            "title":    title,
            "journal":  journal,
            "doi":      doi,
            "keywords": keywords,
        }

        # 1. Abstract as its own chunk
        if abstract.strip():
            chunks.append(Document(
                page_content=f"Title: {title}\n\nAbstract:\n{abstract}",
                metadata={**base_metadata, "section": "abstract"}
            ))

        # 2. Each section as its own chunk
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue

            # Split long sections (>1000 chars) into sub-chunks
            sub_chunks = split_text(section_text, chunk_size=600, overlap=100)

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


def split_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
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