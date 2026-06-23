# app/retriever.py
import os
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq

class HybridRAGRetriever:
    def __init__(self, documents):
        self.docs = documents
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        #  1 Embeddings
        print("Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2 Pinecone setup
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "hybrid-rag-portfolio-hf"
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
            # Check if index is empty or already populated
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
            search_kwargs={"k": 20}
        )

        #  3. BM25 
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 20

        #  4. Cohere Reranker 
        print("Initializing Cohere reranker...")
        self.compressor = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
            top_n=5
        )
# add this method to HybridRAGRetriever
    def rewrite_query(self, query: str) -> str:
        prompt = f"""You are a biomedical search expert.
    Rewrite the following question into an optimal search query for retrieving 
    relevant biomedical research papers. Make it specific, use medical terminology,
    and expand abbreviations.

    Original question: {query}

    Return only the rewritten query, nothing else."""
        
        try:
            rewritten = self.llm.invoke(prompt).content.strip()
            print(f"  Original query: {query}")
            print(f"  Rewritten query: {rewritten}")
            return rewritten
        except Exception:
            return query  # fallback to original if rewrite fails

# update get_relevant_documents
    def get_relevant_documents(self, query: str):
        rewritten = self.rewrite_query(query)

        vector_results = self.vector_retriever.invoke(rewritten)
        bm25_results   = self.bm25_retriever.invoke(rewritten)

        all_docs = list(
            {doc.page_content: doc for doc in (vector_results + bm25_results)}.values()
        )
        print(f"  Hybrid pool: {len(all_docs)} docs before rerank")

        reranked = self.compressor.compress_documents(all_docs, rewritten)
        print(f"  After rerank: {len(reranked)} docs returned")
        return reranked

    def _upsert_documents(self, index_name):
        """Upload documents to Pinecone in batches."""
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
