import os
from dotenv import load_dotenv
from src.graph import build_rag_graph

load_dotenv()

def run_tests():
    print("Initializing LangGraph RAG pipeline for evaluation...\n")
    graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))

    test_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?"  # Out-of-domain grounding test
    ]

    for idx, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"Test Query {idx}: {query}")
        print("-" * 80)
        
        state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0
        }
        result = graph.invoke(state)
        
        print(f"Confidence Score: {result['score']}")
        print(f"Retrieved Chunks: {len(result['context'])}")
        print(f"Final Answer:\n{result['answer']}\n")

if __name__ == "__main__":
    run_tests()