# app/generator.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MAX_CONTEXT_CHARS = 6000


class RAGGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            reasoning_effort="low",
        )

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_context(self, context_documents: list) -> str:
        """
        Build the context block, dropping lowest-ranked chunks (the tail of
        the list — your retriever already reranks, so later = less relevant)
        if the total would blow the token budget. This is what was missing
        before: nothing capped what actually got sent, which is why a big
        retrieval batch could 413 on Groq's TPM limit.
        """
        parts = []
        total_len = 0
        dropped = 0
        for i, doc in enumerate(context_documents):
            pmcid   = doc.metadata.get("source", f"DOC_{i+1}")
            title   = doc.metadata.get("title", "Unknown Title")
            section = doc.metadata.get("section", "unknown")
            journal = doc.metadata.get("journal", "")

            block = (
                f"[{pmcid}] {title} — {journal} (section: {section})\n"
                f"{doc.page_content}\n---\n\n"
            )
            if total_len + len(block) > MAX_CONTEXT_CHARS and parts:
                dropped = len(context_documents) - i
                break
            parts.append(block)
            total_len += len(block)

        if dropped:
            print(f"  [generator] context budget hit — dropped {dropped} "
                  f"lowest-ranked chunk(s) to stay under {MAX_CONTEXT_CHARS} chars")
        return "".join(parts)

    def generate_answer(
        self,
        query: str,
        context_documents: list,
        history: list[dict] | None = None,
    ) -> str:
        history = history or []  

        if not context_documents:
            return "I could not find relevant information in the medical literature to answer this question."

        context_text = self._build_context(context_documents)
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
            11. If earlier conversation is provided below, use it only to resolve pronouns or
            follow-up references (e.g. "what about in older adults?") — never as a source of
            medical facts. Facts still come ONLY from the context.

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

            Previous conversation:
            {history}

            Context:
            {context}
            """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "history": history_text,
            "context": context_text,
            "query": query,
        })