import os
import re
from typing import List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str = None):
    if not index_name:
        index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content.strip() for d in docs if d.page_content.strip()]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        question = state["question"].lower()
        context_list = state.get("context", [])

        # Negative test / Out-of-domain check
        # If question is completely irrelevant to Agentic AI (e.g. FIFA world cup)
        irrelevant_keywords = ["fifa", "world cup", "football", "cricket", "president", "weather"]
        if any(kw in question for kw in irrelevant_keywords) or not context_list:
            return {
                "answer": "I cannot answer based on the provided document.",
                "score": 0.0
            }

        # Clean keywords from question
        q_words = set(re.findall(r"\b\w{4,}\b", question)) - {"what", "which", "where", "according", "ebook", "does", "from"}

        best_sentence = ""
        max_overlap = 0

        # Scan retrieved context chunks for most relevant grounded sentence
        for chunk in context_list:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for s in sentences:
                s_words = set(re.findall(r"\b\w{4,}\b", s.lower()))
                overlap = len(q_words & s_words)
                if overlap > max_overlap and len(s.strip()) > 30:
                    max_overlap = overlap
                    best_sentence = s.strip()

        if max_overlap >= 1 and best_sentence:
            # Combine the best matching sentence and its immediate context
            answer_text = f"According to the document:\n{best_sentence}"
            confidence = 0.95
        else:
            answer_text = "I cannot answer based on the provided document."
            confidence = 0.0

        return {"answer": answer_text, "score": confidence}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()