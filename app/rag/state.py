
from typing import List, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.documents import Document


# 1. Decide whether to use RAG or answer directly
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description=(
            "kb for IT support questions, "
            "direct for greetings or simple chat"
        )
    )


# 2. Evaluate retrieved evidence
class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description=(
            "Whether retrieved evidence is sufficient "
            "to answer the user question"
        )
    )


# 3. Shared state for the LangGraph workflow
class AgentState(TypedDict):
    question: str
    current_query: str

    # Pinecone retrieval
    kb_docs: List[Document]
    kb_grade: str

    # Tavily fallback
    web_results: str
    web_grade: str

    # Final response
    answer: str
    source_used: str
    citations: List[dict]

    # Agent execution
    retry_count: int
    trace: List[str]
