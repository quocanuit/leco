from pydantic import BaseModel, Field


from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def build_rag_chain(llm, collection_name="judgment_collection"):
    # Không load dữ liệu, chỉ kết nối đến collection có sẵn
    retriever = VectorDB(collection_name=collection_name).get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    return rag_chain