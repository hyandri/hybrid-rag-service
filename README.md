# Enterprise Hybrid Search RAG Service with Reranking

A production-grade, asynchronous Retrieval-Augmented Generation (RAG) pipeline built with **FastAPI**, **LangChain**, and **Pinecone**. This architecture addresses the classic limitations of standard semantic vector databases by merging dense vector search with sparse keyword matching (BM25), optimized via a deep-learning **Cohere Reranker** and structural inline source citations.

---

## 🚀 Why This Architecture Matters (The Enterprise Problem)
Most basic RAG tutorials rely solely on Vector Search (Dense Retrieval). While excellent at understanding contextual semantics, pure vector search routinely fails in industry environments where users query exact alphanumeric codes, serial numbers, specific product models, or corporate acronyms. 

This repository implements a **Hybrid Search + Rerank** blueprint to maximize retrieval precision:
1. **Dense Retrieval (Pinecone):** Captures conceptual, semantic context and intent.
2. **Sparse Retrieval (BM25):** Anchors onto exact keyword hits, nomenclature, and financial numbers.
3. **Reciprocal Rank Fusion (RRF):** Blends the distinct scoring distributions programmatically.
4. **Contextual Compression (Cohere Rerank):** Passes top candidate chunks to a cross-encoder network to score definitive relevance, slicing off only the top $k$ items to dramatically mitigate LLM token costs and eliminate "needle-in-a-haystack" confusion.

---

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python 3.11+)
* **Orchestration:** LangChain / LangChain-Community
* **Vector Database:** Pinecone (Serverless Starter)
* **Keyword Indexing:** Rank_BM25
* **Reranking Engine:** Cohere Rerank V3
* **LLM Engine:** Google Gemini (via ChatGoogleGenerativeAI) / Groq (Llama-3)

---

## 📂 Project Structure
```text
hybrid-rag-service/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application router & system gateway
│   ├── ingest.py        # Automated PDF/Text extraction and semantic chunking
│   ├── retriever.py     # Hybrid search engine (Pinecone + BM25) & Cohere Reranking
│   └── generator.py     # High-precision prompt synthesis & strict citation logic
│
├── data/                # Directory for local internal documents (.txt or .pdf)
├── .env                 # Local API keys (Strictly Git-ignored!)
├── requirements.txt     # System dependencies
└── README.md            # System documentation