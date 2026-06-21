# app/generator.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_answer(self, query: str, context_documents):

        # Guard — no docs retrieved
        if not context_documents:
            return "I could not find relevant information in the medical literature to answer this question."

        # Build context with rich source labels
        context_text = ""
        for i, doc in enumerate(context_documents):
            pmcid   = doc.metadata.get("source", f"DOC_{i+1}")
            title   = doc.metadata.get("title", "Unknown Title")
            section = doc.metadata.get("section", "unknown")
            journal = doc.metadata.get("journal", "")

            context_text += f"[{pmcid}] {title} — {journal} (section: {section})\n"
            context_text += f"{doc.page_content}\n"
            context_text += f"---\n\n"

        system_prompt = """
You are a biomedical research assistant with expertise in cardiology and oncology.
You answer questions based strictly on the provided research paper excerpts.

Rules:
1. Use ONLY the information provided in the context below. Do not use outside knowledge.
2. Every factual claim MUST be cited inline using the PMC ID format: [PMC12345678]
3. If multiple sources support a claim, cite all of them: [PMC111] [PMC222]
4. If the answer is not found in the provided context, respond exactly:
"The available literature does not contain sufficient information to answer this question."
5. Do not speculate or infer beyond what is explicitly stated.
6. Use clear medical language appropriate for a clinical or research audience.
7. Use bullet points or tables when comparing multiple findings.
8. Never mention "context", "documents", or "excerpts" — refer naturally to "the literature" or "studies".

Context:
{context}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_text, "query": query})