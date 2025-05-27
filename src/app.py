# import os
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from langserve import add_routes

# from src.base.llm_model import get_gemini_llm
# from src.rag.main import build_rag_chain, InputQA, OutputQA

# llm = get_gemini_llm(model="gemini-2.0-flash")
# genai_docs = "./data_source/judgment"

# # -------- Chains --------

# genai_chain = build_rag_chain(llm, collection_name="judgment_collection")

# # -------- App - FastAPI --------

# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple api server using Langchain's Runnable interfaces",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )

# # -------- Routes - FastAPI --------

# @app.get("/check")
# async def check():
#     return {"status": "ok"}


# @app.post("/judgment", response_model=OutputQA)
# async def judgment(inputs: InputQA):
#     answer = genai_chain.invoke(inputs.question)
#     return {"answer": answer}


# # -------- Langserve Routes - Playground --------
# add_routes(app,
#            genai_chain,
#            playground_type="default",
#            path="/judgment",
#            enabled_endpoints=["invoke"])


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from src.base.llm_model import get_gemini_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.memory.user_memory import UserMemory

llm = get_gemini_llm(model="gemini-2.0-flash")
genai_chain = build_rag_chain(llm, collection_name="judgment_collection")
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
    user_id = "user-001"  # Sau này có thể lấy từ token hoặc header
    chat_history = user_memory.get_summary(user_id)

    print(f"[DEBUG] Chat history for {user_id}:\n{chat_history}")

    answer = genai_chain.invoke({
        "question": inputs.question,
        "chat_history": chat_history
    })

    user_memory.update(user_id, inputs.question, answer)
    return {"answer": answer}

add_routes(app,
           genai_chain,
           playground_type="default",
           path="/judgment",
           enabled_endpoints=["invoke"])