import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.rag.main import build_rag_chain, InputQA, OutputQA

genai_docs = "./data_source/judgment"
genai_chain = build_rag_chain(data_dir=genai_docs, data_type="json")

# -------- App - FastAPI --------
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

# -------- Routes - FastAPI --------
@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/judgment", response_model=OutputQA)
async def judgment(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# -------- Langserve Playground --------
add_routes(app,
           genai_chain,
           playground_type="default",
           path="/judgment",
           enabled_endpoints=["invoke"])
