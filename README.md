# Hybrid RAG - Biomedical Research Assistant

A Retrieval-Augmented Generation system for querying biomedical literature. It combines dense + sparse hybrid retrieval, query rewriting, cross-encoder reranking, and conversational memory to answer questions grounded strictly in a corpus of PubMed Central papers - with citations, and with an explicit refusal to answer when the corpus doesn't support an answer.

## Features

- Hybrid retrieval (Pinecone dense + BM25 sparse)
- LLM query rewriting, conversation-aware (resolves follow-ups like "and in men?")
- Cohere cross-encoder reranking with a relevance-score threshold
- Query router with guardrails (chitchat / general / biomedical / malicious)
- Domain- and section-aware document chunking
- Persistent chat sessions (SQLite)
- Corpus-grounded evaluation pipeline: retrieval metrics (hit@k, MRR) + LLM-judged generation metrics (faithfulness, relevancy, recall, precision)

## Architecture

```text
User Query
    │
    ▼
Query Router ──────► General / Chitchat / Malicious → short-circuit, no retrieval
    │
    ▼ (biomedical)
Query Rewriter (conversation-aware)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              
Dense (Pinecone)  Sparse (BM25)
    │              │
    └──────┬───────┘
           ▼
   Hybrid Candidate Pool
           ▼
   Cohere Rerank + Threshold
           ▼
    Top Context Chunks
           ▼
     LLM Generation
   (grounded, cited, abstains if unsupported)
           ▼
      Final Answer
```

## Key Design Decisions

**Query Router.** Incoming requests are classified before any retrieval happens, so non-biomedical chitchat skips the pipeline entirely and obvious prompt-injection attempts are blocked before they reach the retriever.

**Conversation-Aware Query Rewriting.** Follow-up questions ("what about in older adults?") are resolved against the last few turns before retrieval, not just at generation time - otherwise a multi-turn conversation silently loses context on the retrieval side even if the generator remembers it.

**Hybrid Retrieval.** Dense (semantic) and sparse (keyword/BM25) retrieval are pooled and deduplicated before reranking, which recovers relevant passages that either method alone tends to miss - dense retrieval on paraphrased or acronym-heavy medical language, sparse retrieval on exact terminology and numeric values.

**Cross-Encoder Reranking with a Score Floor.** Candidates are reranked with Cohere `rerank-english-v3.0`, then filtered against a minimum relevance score rather than a fixed top-N - a question with one clearly relevant paper returns one chunk instead of padding out to a fixed count with unrelated ones.

**Grounded Generation with Abstention.** The generator is instructed to synthesize findings in prose with inline citations, and to say explicitly when the retrieved context doesn't support an answer rather than guessing. This is measured, not just claimed - the eval pipeline tracks abstention rate separately from faithfulness so a correct "I don't know" is never scored as a wrong answer.

## Evaluation

The test set is generated *from the corpus itself* - questions are derived from sampled chunks and paraphrased away from the source passage's own wording, so retrieval is scored on semantic matching rather than near-verbatim keyword overlap. Each question carries a gold source paper, which enables two deterministic retrieval metrics (no LLM judge needed) alongside four LLM-judged generation metrics.

| Metric | Score |
|---|---|
| Hit Rate | 0.737 |
| MRR | 0.737 |
| Faithfulness | 0.886 |
| Answer Relevancy | 0.717 |
| Context Recall | 0.771 |
| Context Precision | 0.679 |
| Abstention Rate | 0.263 |

**Methodology notes:**
- Hit Rate / MRR are computed directly from retrieved vs. gold source IDs - deterministic and reproducible, not judge-dependent.
- Faithfulness / Answer Relevancy / Context Recall / Context Precision are scored by an LLM judge, consolidated into a single call per question to control cost and quota usage.
- Abstention Rate tracks how often the system correctly declines to answer rather than hallucinate; abstained answers are excluded from Faithfulness/Relevancy (nothing to fact-check against a refusal) but still contribute to Context Recall, since a missed retrieval that should have surfaced an answer is a real failure worth capturing.
- Retrieval misses (hit = 0) are the clearest signal of where the pipeline actually breaks, and are reviewed manually rather than averaged away.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| LLM (generation) | Groq - `openai/gpt-oss-120b` |
| Vector Database | Pinecone |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse Retrieval | BM25 |
| Reranking | Cohere `rerank-english-v3.0` |
| Storage | SQLite |
| Eval Judge | Groq / Gemini (provider-switchable) |

## Installation

```bash
git clone https://github.com/yourusername/hybrid-rag-biomedical.git
cd hybrid-rag-biomedical

conda create -n hybrid_rag python=3.11 -y
conda activate hybrid_rag
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
COHERE_API_KEY=your_cohere_key
GEMINI_API_KEY=your_gemini_key   # optional, for eval judge
```

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

## Running Evaluations

```bash
# generate a corpus-grounded test set (once)
python evaluate_rag.py --generate 20

# run the evaluation
python evaluate_rag.py
```

Results are checkpointed per-question, so an interrupted run resumes without recomputing completed questions.

## Future Improvements

- Parent-child retrieval
- Metadata filtering
- Multi-query retrieval
- Knowledge graph integration
- Asynchronous ingestion pipeline
- Automated evaluation dashboard

## What I Learned

Retrieval quality has a larger effect on answer quality than the choice of generation model. The biggest measurable gains here came from fixing retrieval-side bugs (threshold filtering, conversation-aware query rewriting) and from building an evaluation set that couldn't be gamed by near-verbatim keyword matching - a naive eval will report near-perfect scores that say nothing about whether the system actually works.