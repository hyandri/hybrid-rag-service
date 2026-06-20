import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our modular logic
from app.ingest import load_and_chunk_docs
from app.retriever import HybridRAGRetriever
from app.generator import RAGGenerator

load_dotenv()

app = FastAPI(title="Hybrid Search RAG API")

# 1. Dynamically load real documents from the data/ folder on startup
print("Loading and chunking internal documents...")
raw_chunks = load_and_chunk_docs("data")

# 2. Spin up the advanced dual-indexing engine
print("Initializing Hybrid Search (Pinecone + BM25) and Cohere Reranker...")
retrieverEngine = HybridRAGRetriever(raw_chunks)
generatorEngine = RAGGenerator()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def run_rag(request: QueryRequest):
    # Retrieve relevant contexts using hybrid logic + reranker
    relevant_docs = retrieverEngine.get_relevant_documents(request.question)
    
    # Generate response with strictly verified inline citations
    answer = generatorEngine.generate_answer(request.question, relevant_docs)
    
    return {
        "answer": answer,
        "sources_used": list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))
    }