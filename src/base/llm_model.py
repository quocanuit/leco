import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_gemini_llm(model: str = "gemini-2.0-flash", **kwargs):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=kwargs.get("temperature", 0.7),
        top_k=kwargs.get("top_k", 40),
        top_p=kwargs.get("top_p", 0.95),
        max_output_tokens=kwargs.get("max_output_tokens", 2048)
    )