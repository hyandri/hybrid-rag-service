# Advanced Hybrid RAG – Biomedical Research Assistant

A Retrieval-Augmented Generation (RAG) system designed for biomedical literature search and question answering. The project combines hybrid retrieval, query expansion, cross-encoder reranking, and conversational memory to improve answer quality over a traditional vector-search-only pipeline.

## Features

* Hybrid Retrieval (Pinecone + BM25)
* LLM-Powered Query Expansion
* Cohere Cross-Encoder Reranking
* Intelligent Query Router & Guardrails
* Sentence-Aware Document Chunking
* Persistent Chat Sessions with SQLite
* Evaluation Pipeline for RAG Performance Analysis

---

## Architecture

```text
User Query
     │
     ▼
┌────────────────────┐
│  Query Router      │
└─────────┬──────────┘
          │
 ┌────────┴────────┐
 ▼                 ▼
General      Biomedical
Response        Query
                    │
                    ▼
          Query Expansion
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Dense Retrieval         Sparse Retrieval
   (Pinecone)                 (BM25)
        │                       │
        └───────────┬───────────┘
                    ▼
          Hybrid Candidate Pool
                    │
                    ▼
          Cohere Reranking
                    │
                    ▼
            Top Context Chunks
                    │
                    ▼
           Llama 3.3 70B
                    │
                    ▼
              Final Answer
```

---

## Key Design Decisions

### Query Router

Incoming requests are classified into:

* CHITCHAT
* GENERAL
* BIOMEDICAL_RAG
* MALICIOUS

This allows non-medical questions to bypass retrieval entirely while blocking obvious prompt injection attempts.

### Sentence-Aware Chunking

Documents are split using sentence boundaries rather than fixed character counts.

Benefits:

* Preserves scientific context
* Reduces fragmented information
* Improves answer faithfulness

### Hybrid Retrieval

The retrieval layer combines:

**Dense Search**

* Semantic similarity retrieval using Pinecone

**Sparse Search**

* Keyword matching using BM25

This approach improves recall compared to using either retrieval strategy alone.

### Query Expansion

Biomedical queries are expanded before retrieval to improve semantic coverage and increase the likelihood of retrieving relevant research passages.

### Cross-Encoder Reranking

Retrieved candidates are reranked using Cohere's `rerank-english-v3.0` model to identify the most relevant context before generation.

### Persistent Session Storage

Conversation history is stored in SQLite and can be restored across sessions while maintaining a rolling conversational context window.

---

## Evaluation Results

The optimized pipeline was evaluated against a baseline RAG implementation.

| Metric            | Baseline RAG | Advanced Hybrid RAG |
| ----------------- | ------------ | ------------------- |
| Faithfulness      | 0.475        | 0.725               |
| Answer Relevancy  | 0.638        | 0.686               |
| Context Recall    | 0.438        | 0.537               |
| Context Precision | 0.528        | 0.510               |

### Observations

* Sentence-aware chunking significantly improved faithfulness.
* Hybrid retrieval improved context recall compared to a vector-only approach.
* Reranking reduced retrieval noise and improved answer relevance.
* Recall remains constrained by the size of the underlying biomedical corpus.

---

## Technology Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Backend          | FastAPI                        |
| LLM              | Groq (Llama 3.3 70B Versatile) |
| Vector Database  | Pinecone                       |
| Embeddings       | all-MiniLM-L6-v2               |
| Sparse Retrieval | BM25                           |
| Reranking        | Cohere rerank-english-v3.0     |
| Storage          | SQLite                         |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/advanced-hybrid-rag.git
cd advanced-hybrid-rag
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
COHERE_API_KEY=your_cohere_key
```

### Install Dependencies

```bash
conda create -n hybrid_rag python=3.11 -y
conda activate hybrid_rag

pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn app.main:app --reload --port 8000
```

---

## Running Evaluations

```bash
python evaluate_rag.py
```

This script evaluates the system using predefined benchmark questions and reports retrieval and generation metrics.

---

## Future Improvements

* Parent-Child Retrieval
* Metadata Filtering
* Multi-Query Retrieval
* Knowledge Graph Integration
* Asynchronous Ingestion Pipeline
* Automated Evaluation Dashboard

---

## What I Learned

This project reinforced that retrieval quality often has a larger impact on RAG performance than simply switching to a larger language model. Improvements such as sentence-aware chunking, hybrid retrieval, and reranking produced measurable gains in faithfulness and retrieval effectiveness.
