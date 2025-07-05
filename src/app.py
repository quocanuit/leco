import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.base.llm_model import get_gemini_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.memory.user_memory import UserMemory

llm = get_gemini_llm(model="gemini-2.0-flash")
dynamic_rag = build_rag_chain(llm)
user_memory = UserMemory()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/judgment", response_model=OutputQA)
async def judgment(inputs: InputQA):
    user_id = "user-001"
    chat_history = user_memory.get_summary(user_id)

    answer = dynamic_rag.get_chain()({
        "question": inputs.question,
        "source_type": inputs.source_type,
        "chat_history": chat_history
    })

    user_memory.update(user_id, inputs.question, answer)
    return {"answer": answer}