from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG
from src.base.llm_model import get_gemini_llm

class InputQA(BaseModel):
    question: str = Field(..., title="Câu hỏi đầu vào cho mô hình")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Câu trả lời từ mô hình")

def build_rag_chain(data_dir, data_type):
    llm = get_gemini_llm(model="gemini-2.0-flash")

    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)

    retriever = VectorDB(documents=doc_loaded).get_retriever()

    rag_chain = Offline_RAG(llm).get_chain(retriever)

    return rag_chain
