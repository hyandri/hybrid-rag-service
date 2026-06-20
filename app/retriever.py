import os
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereRerank

class HybridRAGRetriever:
    def __init__(self, documents):
        self.docs = documents

        #1 initialize embeddings and pinecone(dense)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))

        #conect pinecone index
        index_name = "hybrid-rag-portfolio"
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name = index_name,
                dimension = 768,
                metric = 'cosine',
                spec = ServerlessSpec(cloud='aws', region = 'us-east-1')
            )

        self.vector_store = PineconeVectorStore.from_documents(
            self.docs, self.embeddings, index_name=index_name
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        #2 initialize bm25 retriever(sparse)
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 10

        #3 setup cohere reranker
        self.compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"), top_n=3)

def get_relevant_documents(self, query: str):
        # Fetch from both
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Reciprocal Rank Fusion (RRF) - Simple programmatic merger
        # Put all unique docs together for the reranker to evaluate
        all_docs = list({doc.page_content: doc for doc in (vector_results + bm25_results)}.values())
        
        # Apply Cohere Reranking to pick the ultimate top 3
        reranked_docs = self.compressor.compress_documents(all_docs, query)
        return reranked_docs