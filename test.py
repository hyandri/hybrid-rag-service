# Run this one-off debug script first
import requests

pmcid = "13281426"
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
params = {
    "db": "pmc",
    "id": pmcid,
    "rettype": "xml",
    "retmode": "xml"
}
r = requests.get(url, params=params, timeout=30)
print("Status:", r.status_code)
print("First 2000 chars:")
print(r.text[:2000])