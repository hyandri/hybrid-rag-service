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

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # truncate long assistant answers in history to save tokens
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def generate_answer(
        self,
        query: str,
        context_documents: list,
        history: list[dict] = []
    ) -> str:
        
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
        
        history_text = self._format_history(history)
        
        system_prompt = """
            You are an expert biomedical research assistant with deep knowledge across medical literature.

            Your job is to SYNTHESIZE and EXPLAIN findings — not restate or reformat them.

            RULES:
            1. Use ONLY the provided research context. No outside knowledge.
            2. Write in flowing, analytical prose. Never use bullet points unless comparing 3+ distinct items.
            3. Explain the "so what" — what do the findings mean, why do they matter.
            4. Every factual claim must have an inline citation: [PMC12345678]
            5. If multiple sources agree, synthesize them into one explanation and cite all: [PMC111][PMC222]
            6. If sources contradict, explicitly note the tension between findings.
            7. If the answer is not in the context respond exactly:
            "The available literature does not contain sufficient information to answer this question."
            8. Never use phrases like "according to the literature" or "the document states" — 
            write as an expert explaining findings naturally.
            9. Do not mention "context", "chunks", "documents", or "sources" — 
            refer to "studies", "research", "findings", or "evidence".
            10. Aim for 3-5 sentences of genuine synthesis minimum for any answerable question.

            BAD response style (do NOT do this):
            "According to the literature, aging disrupts X [PMC111]. This includes:
            - Change A
            - Change B  
            - Change C"

            GOOD response style:
            "Aging fundamentally disrupts the spatiotemporal coordination of the murine ovary through 
            a cascade of interconnected processes. Research shows that folliculogenesis progressively 
            decouples from the estrous cycle [PMC111], a shift that precedes full acyclicity and 
            ultimately drives reproductive senescence. This uncoupling is accompanied by structural 
            deterioration including fibrosis, epithelial thickening, and multinucleated giant cell 
            accumulation [PMC111], suggesting that the breakdown is not merely hormonal but reflects 
            broader tissue-level aging."

            Context:
            {context}
            """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history": history_text, "context": context_text, "query": query})