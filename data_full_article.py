import requests, json, time, re
from xml.etree import ElementTree as ET

def search_pmcids(keyword_query, max_ids=600):
    print(f"Searching PMC for: {keyword_query}")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": keyword_query,
        "retmax": max_ids,
        "retmode": "json"
    }
    r = requests.get(url, params=params)
    result = r.json()["esearchresult"]
    ids = result["idlist"]
    print(f"Found {len(ids)} PMCIDs")
    return ids

def fetch_full_text_efetch(pmcid):
    """Fetch full text XML via E-utilities efetch — more reliable than OAI."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "retmode": "xml"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        # efetch returns error message in XML if article not available
        if b"<error>" in r.content.lower() or len(r.content) < 500:
            return None
        return r.text
    except Exception as e:
        print(f"  Error: {e}")
        return None

def parse_jats_xml(xml_string):
    try:
        root = ET.fromstring(xml_string.encode("utf-8"))
    except ET.ParseError as e:
        print(f"  ParseError: {e}")
        return None

    def get_text(elem):
        if elem is None:
            return ""
        return " ".join(
            t.strip() for t in elem.itertext() if t.strip()
        )

    # With namespaces present, tags look like {http://...}tag-name
    # Use wildcard *[local-name()=x] equivalent via this helper
    def find_tag(root, local_name):
        for elem in root.iter():
            if elem.tag.split("}")[-1] == local_name:
                return elem
        return None

    def findall_tag(root, local_name):
        return [e for e in root.iter() if e.tag.split("}")[-1] == local_name]

    title    = get_text(find_tag(root, "article-title"))
    abstract = get_text(find_tag(root, "abstract"))
    keywords = [get_text(k) for k in findall_tag(root, "kwd")]
    journal  = get_text(find_tag(root, "journal-title"))

    doi = ""
    for aid in findall_tag(root, "article-id"):
        if aid.get("pub-id-type") == "doi":
            doi = aid.text or ""

    # Section-aware extraction
    sections = {}
    for sec in findall_tag(root, "sec"):
        sec_title_elem = next(
            (c for c in sec if c.tag.split("}")[-1] == "title"), None
        )
        sec_title = get_text(sec_title_elem).lower() if sec_title_elem is not None else "body"

        # Only direct <p> children, not nested subsections
        paragraphs = [
            get_text(c) for c in sec if c.tag.split("}")[-1] == "p"
        ]
        sec_text = " ".join(paragraphs).strip()

        if sec_text:
            base = sec_title
            counter = 1
            while sec_title in sections:
                sec_title = f"{base}_{counter}"
                counter += 1
            sections[sec_title] = sec_text

    return {
        "title":    title,
        "abstract": abstract,
        "journal":  journal,
        "doi":      doi,
        "sections": sections,
        "keywords": keywords
    }

# ── Main ────────────────────────────────────────────────────────────────────
cardio_ids = search_pmcids(
    "cardiovascular OR cardiac OR heart failure OR coronary artery",
    max_ids=300
)
onco_ids = search_pmcids(
    "cancer OR oncology OR carcinoma OR tumor chemotherapy",
    max_ids=200
)

all_ids = list(set(cardio_ids + onco_ids))
print(f"\nTotal unique PMCIDs: {len(all_ids)}")

papers = []
failed = 0

for i, pmcid in enumerate(all_ids):
    print(f"[{i+1}/{len(all_ids)}] PMC{pmcid} ...", end=" ", flush=True)

    xml = fetch_full_text_efetch(pmcid)
    if not xml:
        print("SKIP (no response)")
        failed += 1
        time.sleep(0.4)
        continue

    parsed = parse_jats_xml(xml)
    if not parsed or not parsed["title"]:
        print("SKIP (parse failed)")
        failed += 1
        time.sleep(0.4)
        continue

    papers.append({
        "pmcid": f"PMC{pmcid}",
        **parsed
    })
    print(f"OK — {len(parsed['sections'])} sections")

    time.sleep(0.34)  # stay under 3 req/sec

    if (i + 1) % 100 == 0:
        with open("pmc_cardiology_oncology.json", "w") as f:
            json.dump(papers, f, indent=2)
        print(f"  ── Checkpoint: {len(papers)} papers saved ──")

with open("pmc_cardiology_oncology.json", "w") as f:
    json.dump(papers, f, indent=2)

print(f"\nDone. {len(papers)} papers saved, {failed} failed.")