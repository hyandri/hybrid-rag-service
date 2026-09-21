# app/retriever.py
import os
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
import time

MIN_RELEVANCE = 0.35

class HybridRAGRetriever:
    def __init__(self, documents):
        self.docs = documents
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        print("Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "hybrid-rag-portfolio-hf-2"
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            print("Creating new Pinecone index...")
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            self._upsert_documents(index_name)
        else:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)

            if total_vectors == 0:
                print("Index exists but is empty — upserting documents...")
                self._upsert_documents(index_name)
            else:
                print(f"Index already has {total_vectors} vectors — skipping upsert.")

        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(index_name),
            embedding=self.embeddings
        )
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.6}
        )

        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 15

        print("Initializing Cohere reranker...")
        self.compressor = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
            top_n=6
        )

    def rewrite_query(self, query: str, history: list[dict] | None = None) -> str:
        history = history or []
        history_text = ""
        if history:
            last_turns = history[-2:]
            history_text = "\n".join(f"{h['role']}: {h['content']}" for h in last_turns)

        prompt = f"""You are a biomedical search expert.
Rewrite the following question into a descriptive, natural language search phrase for cross-reference retrieval. 
Expand abbreviations and use precise medical terminology.
If the question refers back to something in the conversation (e.g. "and in men?", "what about young adults"), 
resolve that reference using the conversation below so the rewritten query is fully self-contained.

CRITICAL: Output ONLY a clean, plain-text string. Do NOT use boolean operators like AND, OR, NOT, quotes, or parentheses.
Do not assume a medical specialty or add clinical detail not present in the original question — if a term is ambiguous, keep it general.
Conversation so far:
{history_text if history_text else "None"}

Original question: {query}

Return only the rewritten query, nothing else."""

        try:
            rewritten = self.llm.invoke(prompt).content.strip()
            rewritten = rewritten.replace('"', '').replace("'", "")
            print(f"  Original query: {query}")
            print(f"  Rewritten query: {rewritten}")
            return rewritten
        except Exception:
            return query

    # def get_relevant_documents(self, query: str, history: list[dict] | None = None):
    #     rewritten = self.rewrite_query(query, history)

    #     vector_results_rewritten = self.vector_retriever.invoke(rewritten)
    #     vector_results_original = self.vector_retriever.invoke(query)
    #     bm25_results = self.bm25_retriever.invoke(rewritten)

    #     all_docs = list(
    #         {doc.page_content: doc for doc in (vector_results_rewritten + vector_results_original + bm25_results)}.values()
    #     )
    #     print(f"  Hybrid pool: {len(all_docs)} docs before rerank")

    #     reranked = self.compressor.compress_documents(all_docs, rewritten)

    #     filtered = []
    #     for d in reranked:
    #         score = d.metadata.get("relevance_score", 0.0)
    #         if score >= MIN_RELEVANCE:
    #             filtered.append(d)

    #     print(f"  After rerank: {len(reranked)} → {len(filtered)} above threshold ({MIN_RELEVANCE})")

    #     return filtered if filtered else reranked[:1]

    def get_relevant_documents(self, query: str, history: list[dict] | None = None):
        rewritten = self.rewrite_query(query, history)

        vector_results_rewritten = self.vector_retriever.invoke(rewritten)
        vector_results_original = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(rewritten)

        all_docs = list(
            {doc.page_content: doc for doc in (vector_results_rewritten + vector_results_original + bm25_results)}.values()
        )
        print(f"  Hybrid pool: {len(all_docs)} docs before rerank")

        # --- Cohere trial key rate limit: 10 calls/min ---
        time.sleep(6)
        for attempt in range(3):
            try:
                reranked = self.compressor.compress_documents(all_docs, rewritten)
                break
            except Exception as e:
                if "429" in str(e) or "TooManyRequests" in str(e):
                    print("    .. Cohere rate limited, waiting 15s")
                    time.sleep(15)
                else:
                    raise
        else:
            reranked = all_docs[:6]
        # --- end rate limit block ---

        filtered = []
        for d in reranked:
            score = d.metadata.get("relevance_score", 0.0)
            if score >= MIN_RELEVANCE:
                filtered.append(d)

        print(f"  After rerank: {len(reranked)} → {len(filtered)} above threshold ({MIN_RELEVANCE})")

        return filtered if filtered else reranked[:1]
    
    def _upsert_documents(self, index_name):
        print(f"Upserting {len(self.docs)} chunks to Pinecone...")
        BATCH_SIZE = 100
        for i in range(0, len(self.docs), BATCH_SIZE):
            batch = self.docs[i: i + BATCH_SIZE]
            PineconeVectorStore.from_documents(
                batch,
                self.embeddings,
                index_name=index_name
            )
            print(f"  Uploaded batch {i // BATCH_SIZE + 1} / {-(-len(self.docs) // BATCH_SIZE)}")
        print("Upsert complete.")