import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def run_ingestion(pdf_path: str, index_name: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding and upserting vectors into Pinecone index: '{index_name}'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Ingestion completed successfully.")
    return vector_store

if __name__ == "__main__":
    index = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    pdf_file = os.path.join("data", "Ebook-Agentic-AI.pdf")
    run_ingestion(pdf_file, index)