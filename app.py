import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph

load_dotenv()

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)