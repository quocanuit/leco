from pydantic import BaseModel, Field
from typing import Literal

from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")
    source_type: Literal["judgment", "law"] = Field(default="judgment", title="Source type: judgment or law")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

def build_rag_chain(llm):
    return Offline_RAG(llm)