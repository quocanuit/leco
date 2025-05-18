import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_model import get_gemini_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_gemini_llm(model="gemini-2.0-flash")
genai_docs = "./data_source/judgment"

# -------- Chains --------

genai_chain = build_rag_chain(llm, collection_name="judgment_collection")

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
    answer = genai_chain.invoke({"question": inputs.question, "debug": inputs.debug})
    
    debug_info = None
    if inputs.debug:
        from src.rag.main import offline_rag_instance
        if offline_rag_instance and hasattr(offline_rag_instance, "str_parser"):
            debug_info = offline_rag_instance.str_parser.get_debug_info()
    
    return {"answer": answer, "debug_info": debug_info}


# -------- Langserve Routes - Playground --------
add_routes(app,
           genai_chain,
           playground_type="default",
           path="/judgment",
           enabled_endpoints=["invoke"])