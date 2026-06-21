#generator.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class RAGGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_answer(self, query: str, context_documents):
            #1 format context text with clear source tags
            context_text = ""
            for i, doc in enumerate(context_documents):
                source_path = doc.metadata.get("source",f"Document_{i+1}")
                source_name = os.path.basename(source_path)

                context_text+=f"START SOURCE:{source_name}\n"
                context_text += f"{doc.page_content}\n"
                context_text += f"END SOURCE: {source_name} \n\n"

            system_prompt = """
            You are an internal knowledge assistant.

            Answer using ONLY the provided information.

            Rules:
            1. Do not use outside knowledge or make assumptions.
            2. Every factual statement must include an inline source citation:
            [source_id]
            3. If the answer is not explicitly available, respond exactly:
            "I cannot find the requested information within the available internal documentation."
            4. Do not mention "context" or "documents". Refer naturally if needed.
            5. Use Markdown tables or bullet points when helpful.

            Information:
            {context}
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system",system_prompt),
                ("human","{query}")
            ])

            chain = prompt | self.llm |StrOutputParser()
            return chain.invoke({"context":context_text, "query":query})
