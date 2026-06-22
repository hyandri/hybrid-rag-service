# # Run this one-off debug script first
# import requests

# pmcid = "13281426"
# url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# params = {
#     "db": "pmc",
#     "id": pmcid,
#     "rettype": "xml",
#     "retmode": "xml"
# }
# r = requests.get(url, params=params, timeout=30)
# print("Status:", r.status_code)
# print("First 2000 chars:")
# print(r.text[:2000])

# debug_pinecone.py — run once to inspect
# reset.py
# reset.py — run once
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.delete_index("hybrid-rag-portfolio-hf")
print("Deleted. Restart uvicorn to re-upsert.")