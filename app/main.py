# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.ingest import load_and_chunk_docs
from app.retriever import HybridRAGRetriever
from app.generator import RAGGenerator
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI(title="Hybrid RAG — Cardiology & Oncology Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any local frontend file to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading and chunking documents...")
raw_chunks = load_and_chunk_docs("pmc_cardiology_oncology.json")  # ← updated path

print("Initializing Hybrid Retriever...")
retrieverEngine = HybridRAGRetriever(raw_chunks)

print("Initializing Generator...")
generatorEngine = RAGGenerator()

print("API ready.")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def run_rag(request: QueryRequest):
    relevant_docs = retrieverEngine.get_relevant_documents(request.question)
    answer = generatorEngine.generate_answer(request.question, relevant_docs)
    return {
        "answer": answer,
        "sources_used": list(set([
            doc.metadata.get("source", "Unknown") for doc in relevant_docs
        ]))
    }

@app.get("/health")
async def health():
    return {"status": "ok"}