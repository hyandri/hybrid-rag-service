#retriever.py
import os
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereRerank

class HybridRAGRetriever:
    def __init__(self, documents):
        self.docs = documents
        
        # 1. Initialize Local Hugging Face Embeddings
        print("Loading local Hugging Face embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "hybrid-rag-portfolio-hf"
        
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print("Creating a new Pinecone Index...")
            self.pc.create_index(
                name=index_name,
                dimension=384, # Hugging Face MiniLM vectors are 384 dimensions
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        # Store docs in Pinecone Vector Store
        print(f"Generating embeddings and uploading {len(self.docs)} chunks to Pinecone...")
        self.vector_store = PineconeVectorStore.from_documents(
            self.docs, self.embeddings, index_name=index_name
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 2. Initialize BM25 Retriever (Sparse)
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 10

        # 3. Setup Cohere Reranker with explicit model parameter
        print("Initializing Cohere Reranker model...")
        self.compressor = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"), 
            model="rerank-english-v3.0", 
            top_n=3
        )

    def get_relevant_documents(self, query: str):
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        
        all_docs = list({doc.page_content: doc for doc in (vector_results + bm25_results)}.values())
        
        reranked_docs = self.compressor.compress_documents(all_docs, query)
        return reranked_docs