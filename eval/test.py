#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from src.base.llm_model import get_gemini_llm
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG

def debug_rag_process():
    """Debug complete RAG process"""
    
    test_question = "Tôi muốn đơn phương ly hôn chồng"
    source_type = "judgment"
    
    print("=" * 60)
    print("LECO DEBUG TEST")
    print("=" * 60)
    print(f"Question: {test_question}")
    print(f"Source type: {source_type}")
    print()
    
    # 1. Initialize LLM
    print("1. Initializing LLM...")
    try:
        llm = get_gemini_llm(model="gemini-2.0-flash")
        print("OK: LLM initialized successfully")
    except Exception as e:
        print(f"ERROR: LLM initialization failed: {e}")
        return
    
    # 2. Initialize Vector DB
    print("\n2. Initializing Vector Database...")
    try:
        collection_name = "judgment_collection" if source_type == "judgment" else "law_collection"
        vector_db = VectorDB(collection_name=collection_name)
        retriever = vector_db.get_retriever()
        print(f"OK: Vector DB initialized (Collection: {collection_name})")
    except Exception as e:
        print(f"ERROR: Vector DB initialization failed: {e}")
        return
    
    # 3. Document retrieval
    print(f"\n3. Retrieving documents for: '{test_question}'")
    try:
        retrieved_docs = retriever.invoke(test_question)
        print(f"OK: Found {len(retrieved_docs)} documents")
        
        print("\n--- RETRIEVED DOCUMENTS ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(f"Metadata: {doc.metadata}")
            print(f"Content (first 200 chars): {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"ERROR: Document retrieval failed: {e}")
        return
    
    # 4. Format context
    print(f"\n4. Formatting context...")
    try:
        rag = Offline_RAG(llm)
        formatted_context = rag.format_docs(retrieved_docs)
        print(f"OK: Context formatted successfully")
        print(f"\n--- FORMATTED CONTEXT ---")
        print(formatted_context)
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR: Context formatting failed: {e}")
        return
    
    # 5. Load prompt template
    print(f"\n5. Loading prompt template...")
    try:
        prompt_path = os.path.join("..", "src", "rag", "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        print(f"OK: Prompt template loaded")
        
        full_prompt = rag.prompt.format(
            context=formatted_context,
            question=test_question,
            chat_history=""
        )
        print(f"\n--- FULL PROMPT ---")
        print(full_prompt)
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR: Prompt loading failed: {e}")
        return
    
    # 6. Generate response
    print(f"\n6. Generating response...")
    try:
        llm_response = llm.invoke(full_prompt)
        raw_response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        print(f"OK: Response generated")
        
        print(f"\n--- LLM RESPONSE ---")
        print(raw_response)
        print("-" * 40)
        
        # Parse the LLM response
        parsed_response = rag.str_parser.parse(raw_response)
        print(f"\n--- PARSED LLM RESPONSE ---")
        print(parsed_response)
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR: Response generation failed: {e}")
        return
    
    print(f"\nDEBUG TEST COMPLETED")

if __name__ == "__main__":
    debug_rag_process()
