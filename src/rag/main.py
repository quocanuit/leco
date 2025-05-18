from pydantic import BaseModel, Field

from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")
    debug: bool = Field(False, title="Flag to enable debug output")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")
    debug_info: dict | None = Field(default=None, title="Debug information when debug mode is enabled")


def build_rag_chain(llm, collection_name="judgment_collection"):
    global offline_rag_instance
    
    retriever = VectorDB(collection_name=collection_name).get_retriever()
    offline_rag_instance = Offline_RAG(llm)
    rag_chain = offline_rag_instance.get_chain(retriever)
    
    return rag_chain

offline_rag_instance = None