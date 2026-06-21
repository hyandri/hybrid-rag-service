#ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_docs(data_directory="data"):
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    if len(os.listdir(data_directory)) == 0:
        print(f" Warning: '{data_directory}/' folder is empty! Generating a sample file for testing...")
        # Create a sample file so the retrievers have something to index
        fallback_path = os.path.join(data_directory, "sample_policy.txt")
        with open(fallback_path, "w") as f:
            f.write(
                "Company policy requires all employees to complete security training by Q3.\n"
                "For expense reimbursements, submit all receipts via the portal before the 25th of each month."
            )
    
    loader = DirectoryLoader(data_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100
    )

    chunks = text_splitter.split_documents(documents)
    return chunks