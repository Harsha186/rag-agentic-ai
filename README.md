\# Agentic AI RAG Chatbot



A production-ready Retrieval-Augmented Generation (RAG) system built with \*\*LangGraph\*\*, \*\*Pinecone Vector Database\*\*, and \*\*FastAPI\*\*, grounded strictly on the \*Agentic AI eBook\*.



\---



\## Architecture \& Tech Stack

\- \*\*Orchestration:\*\* LangGraph (StateGraph with `retrieve` and `generate` nodes)

\- \*\*Vector Database:\*\* Pinecone Serverless (384-dimensional cosine index)

\- \*\*Embeddings:\*\* HuggingFace `all-MiniLM-L6-v2` (local inference, zero API cost)

\- \*\*API Framework:\*\* FastAPI with Pydantic request/response validation

\- \*\*Document Ingestion:\*\* PyPDFLoader + RecursiveCharacterTextSplitter (1000 chunk size, 200 overlap)



\---



\## Project Structure

```text

rag-agentic-ai/

├── app.py                     # FastAPI application \& /chat endpoint

├── tests\_sample\_queries.py     # Benchmark test runner

├── .env.example               # Environment template

├── requirements.txt           # Project dependencies

├── data/

│   └── Ebook-Agentic-AI.pdf   # Source document

└── src/

&#x20;   ├── ingestion.py           # Document loading, chunking, and Pinecone upsert

&#x20;   └── graph.py               # LangGraph state machine \& RAG pipeline

