#!/usr/bin/env python3
"""
RAGAS evaluation script for Judgment collection
"""

import os
import sys
from dotenv import load_dotenv
import json
from datasets import Dataset

load_dotenv()

# Add src to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from src.base.llm_model import get_gemini_llm
from src.rag.main import build_rag_chain
from src.rag.vectorstore import VectorDB

# Import RAGAS
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def extract_realistic_ground_truth(question, retriever, max_chars=200):
    """Extract realistic ground truth from the best retrieved context"""
    docs = retriever.invoke(question)
    
    if not docs:
        return "", []
    
    best_doc = docs[0]
    content = best_doc.page_content
    
    if "bao nhiêu" in question.lower() and "ly hôn" in question.lower():
        ground_truth = "Dựa trên dữ liệu bản án có thể thống kê số lượng vụ ly hôn trong thời gian cụ thể"
        keywords = ["vụ ly hôn", "thống kê", "bản án", "số lượng"]
    elif "giành quyền" in question.lower() and "nuôi con" in question.lower():
        ground_truth = "Để giành quyền nuôi con, cần chứng minh khả năng chăm sóc và điều kiện tốt nhất cho con"
        keywords = ["quyền nuôi con", "chăm sóc", "điều kiện", "lợi ích"]
    elif "ly hôn đơn phương" in question.lower():
        ground_truth = "Ly hôn đơn phương có thể thực hiện trong các trường hợp được pháp luật quy định"
        keywords = ["ly hôn đơn phương", "trường hợp", "pháp luật"]
    elif "chia tài sản" in question.lower():
        ground_truth = "Tài sản chung được chia theo nguyên tắc công bằng, xem xét đóng góp của mỗi bên"
        keywords = ["chia tài sản", "công bằng", "đóng góp"]
    else:
        ground_truth = content[:max_chars] + "..."
        keywords = ["ly hôn", "bản án", "tòa án", "giải quyết"]
    
    return ground_truth, keywords

def run_ragas_evaluation():
    # Initialize
    llm = get_gemini_llm(model="gemini-2.0-flash")
    rag_system = build_rag_chain(llm)
    
    # Judgment collection only
    judgment_vector_db = VectorDB(collection_name="judgment_collection")
    judgment_retriever = judgment_vector_db.get_retriever()
    
    # Test questions for judgment collection
    test_questions = [
        "Có bao nhiêu vụ ly hôn trong dữ liệu tháng 1/2024?",
        "Làm thế nào để giành quyền nuôi con khi ly hôn?",
        "Tôi có thể ly hôn đơn phương không?",
        "Cách chia tài sản khi ly hôn theo kinh nghiệm thực tế",
        "Thủ tục ly hôn mất bao lâu theo bản án?"
    ]
    
    # Generate evaluation data
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for question in test_questions:
        # Get contexts
        docs = judgment_retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Get answer using dynamic chain
        try:
            answer = rag_system.get_chain()({
                "question": question,
                "source_type": "judgment",
                "chat_history": ""
            })
        except Exception as e:
            answer = "Không thể trả lời câu hỏi này."
        
        # Extract realistic ground truth
        ground_truth, keywords = extract_realistic_ground_truth(question, judgment_retriever)
        
        # Add to evaluation data
        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(answer)
        evaluation_data["contexts"].append(contexts)
        evaluation_data["ground_truth"].append(ground_truth)
    
    # Create dataset
    dataset = Dataset.from_dict(evaluation_data)
    
    # Run evaluation
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        
        # Calculate scores
        scores = {}
        avg_answer_relevancy = sum(result['answer_relevancy']) / len(result['answer_relevancy'])
        avg_faithfulness = sum(result['faithfulness']) / len(result['faithfulness'])
        avg_context_precision = sum(result['context_precision']) / len(result['context_precision'])
        avg_context_recall = sum(result['context_recall']) / len(result['context_recall'])
        
        scores['answer_relevancy'] = float(avg_answer_relevancy)
        scores['faithfulness'] = float(avg_faithfulness)
        scores['context_precision'] = float(avg_context_precision)
        scores['context_recall'] = float(avg_context_recall)
        
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            scores["average"] = avg_score
        
        # Save results
        os.makedirs("output", exist_ok=True)
        
        with open("output/ragas_judgment_results.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        
        return scores
        
    except Exception as e:
        return None

if __name__ == "__main__":
    try:
        scores = run_ragas_evaluation()
        
        if scores:
            print("Judgment Collection Evaluation Results:")
            print("=" * 38)
            print(f"Answer Relevancy: {scores['answer_relevancy']:.4f}")
            print(f"Faithfulness: {scores['faithfulness']:.4f}")
            print(f"Context Precision: {scores['context_precision']:.4f}")
            print(f"Context Recall: {scores['context_recall']:.4f}")
            print(f"Average Score: {scores['average']:.4f}")
            print("Results saved to: output/ragas_judgment_results.json")
        else:
            print("Evaluation failed")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
