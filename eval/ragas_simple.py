#!/usr/bin/env python3
"""
Script đánh giá RAG với RAGAS
"""

import os
import sys
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def ragas_evaluation():
    """Đánh giá RAG sử dụng RAGAS"""
    
    print("Starting RAGAS Evaluation for LECO")
    print("=" * 40)
    
    # Add src to path
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(parent_dir)
    
    from src.base.llm_model import get_gemini_llm
    from src.rag.main import build_rag_chain
    from src.rag.vectorstore import VectorDB
    
    # Initialize
    print("Initializing...")
    llm = get_gemini_llm(model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    rag_chain = build_rag_chain(llm, "judgment_collection")
    vector_db = VectorDB(collection_name="judgment_collection")
    retriever = vector_db.get_retriever()
    
    # Test data
    test_data = [
        {
            "question": "Thủ tục ly hôn đơn phương cần những giấy tờ gì?",
            "ground_truth": "Cần đơn khởi kiện, giấy tờ tùy thân, giấy chứng nhận kết hôn, tài liệu chứng minh lý do ly hôn."
        },
        {
            "question": "Điều kiện để được kết hôn tại Việt Nam là gì?",
            "ground_truth": "Nam từ 20 tuổi, nữ từ 18 tuổi, tự nguyện, không cùng huyết thống, không mắc bệnh tâm thần."
        }
    ]
    
    # Prepare data for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("Processing questions...")
    
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Question {i}/{len(test_data)}")
        
        # Get answer from RAG
        answer = rag_chain.invoke({"question": question})
        answer_text = str(answer)
        
        # Get contexts
        docs = retriever.invoke(question)
        context_list = [doc.page_content for doc in docs[:3]]
        
        # Add to lists
        questions.append(question)
        answers.append(answer_text)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
    
    # Create dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print(f"\nEvaluating {len(questions)} samples...")
    
    # Evaluate with RAGAS
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    
    # Display results
    print("\n" + "=" * 40)
    print("RAGAS RESULTS")
    print("=" * 40)
    
    # Calculate average scores
    avg_answer_relevancy = sum(result['answer_relevancy']) / len(result['answer_relevancy'])
    avg_faithfulness = sum(result['faithfulness']) / len(result['faithfulness'])
    avg_context_precision = sum(result['context_precision']) / len(result['context_precision'])
    avg_context_recall = sum(result['context_recall']) / len(result['context_recall'])
    
    print(f"answer_relevancy: {avg_answer_relevancy:.3f}")
    print(f"faithfulness: {avg_faithfulness:.3f}")
    print(f"context_precision: {avg_context_precision:.3f}")
    print(f"context_recall: {avg_context_recall:.3f}")
    
    # Save results
    os.makedirs("output", exist_ok=True)
    
    with open("output/ragas_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "answer_relevancy": float(avg_answer_relevancy),
            "faithfulness": float(avg_faithfulness),
            "context_precision": float(avg_context_precision),
            "context_recall": float(avg_context_recall)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: output/ragas_results.json")

if __name__ == "__main__":
    ragas_evaluation()
