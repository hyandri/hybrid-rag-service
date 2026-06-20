import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_docs(data_directory="data"):
    if not os.path.exists(data_directory):
        print("path/data doesnt exists")
    
    loader = DirectoryLoader(data_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100
    )

    chunks = text_splitter.split_documents(documents)
    return chunks