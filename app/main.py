# main.py
import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from app.ingest import load_and_chunk_docs
from app.retriever import HybridRAGRetriever
from app.generator import RAGGenerator
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI(title="Hybrid RAG — Cardiology & Oncology Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading and chunking documents...")
raw_chunks = load_and_chunk_docs("pmc_cardiology_oncology.json")  

print("Initializing Hybrid Retriever...")
retrieverEngine = HybridRAGRetriever(raw_chunks)

print("Initializing Generator...")
generatorEngine = RAGGenerator()

print("API ready.")

sessions: dict[str,list[dict]] = {}  # session_id -> list of {question, answer, sources}

class RouterDecision(BaseModel):
    category: Literal["CHITCHAT", "GENERAL", "BIOMEDICAL_RAG", "MALICIOUS"] = Field(
        description="The categorization of the user query."
    )
    reasoning:str = Field(
        description="Short sentece why chosen"
    )
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.post("/query")
async def run_rag(request: QueryRequest):

    session_id = request.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    relevant_docs = retrieverEngine.get_relevant_documents(request.question)
    answer = generatorEngine.generate_answer(
        query = request.question,
        context_documents = relevant_docs,
        history = history)
    
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    sessions[session_id] = history[-10:]

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": [
            {
                "pmcid":   doc.metadata.get("source"),
                "title":   doc.metadata.get("title"),
                "journal": doc.metadata.get("journal"),
                "section": doc.metadata.get("section"),
                "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{doc.metadata.get('source')}/"
            }
            for doc in relevant_docs
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(sessions)}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}