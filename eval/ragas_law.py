#!/usr/bin/env python3
"""
RAGAS evaluation script for Law collection
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
    
    if "tuổi" in question.lower() and "kết hôn" in question.lower():
        ground_truth = "Nam phải từ đủ 20 tuổi trở lên, nữ phải từ đủ 18 tuổi trở lên"
        keywords = ["20 tuổi", "18 tuổi", "nam", "nữ", "kết hôn"]
    elif "nguyên tắc" in question.lower() and "tài sản" in question.lower():
        ground_truth = "Việc giải quyết tài sản do các bên thỏa thuận; nếu không thỏa thuận được thì Tòa án giải quyết theo quy định của pháp luật"
        keywords = ["giải quyết tài sản", "thỏa thuận", "tòa án", "pháp luật"]
    elif "điều kiện" in question.lower() and "ly hôn" in question.lower():
        ground_truth = "Vợ hoặc chồng có quyền yêu cầu Tòa án giải quyết ly hôn trong các trường hợp pháp luật quy định"
        keywords = ["điều kiện ly hôn", "tòa án", "trường hợp", "pháp luật"]
    elif "quyền" in question.lower() and "nghĩa vụ" in question.lower():
        ground_truth = "Vợ chồng có quyền và nghĩa vụ tôn trọng, chăm sóc, giúp đỡ lẫn nhau"
        keywords = ["quyền", "nghĩa vụ", "vợ chồng", "tôn trọng"]
    else:
        ground_truth = content[:max_chars] + "..."
        keywords = ["luật", "pháp luật", "quy định"]
    
    return ground_truth, keywords

def run_ragas_evaluation():
    # Initialize
    llm = get_gemini_llm(model="gemini-2.0-flash")
    rag_system = build_rag_chain(llm)
    
    # Law collection only
    law_vector_db = VectorDB(collection_name="law_collection")
    law_retriever = law_vector_db.get_retriever()
    
    # Test questions for law collection
    test_questions = [
        "Tuổi tối thiểu để kết hôn theo pháp luật Việt Nam?",
        "Nguyên tắc giải quyết tài sản của vợ chồng khi ly hôn là gì?", 
        "Điều kiện để ly hôn đơn phương là gì?",
        "Quyền và nghĩa vụ của vợ chồng trong hôn nhân",
        "Thủ tục kết hôn theo pháp luật hiện hành"
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
        docs = law_retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Get answer using dynamic chain
        try:
            answer = rag_system.get_chain()({
                "question": question,
                "source_type": "law",
                "chat_history": ""
            })
        except Exception as e:
            answer = "Không thể trả lời câu hỏi này."
        
        # Extract realistic ground truth
        ground_truth, keywords = extract_realistic_ground_truth(question, law_retriever)
        
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
        
        with open("output/ragas_law_results.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        
        return scores
        
    except Exception as e:
        return None

if __name__ == "__main__":
    try:
        scores = run_ragas_evaluation()
        
        if scores:
            print("Law Collection Evaluation Results:")
            print("=" * 35)
            print(f"Answer Relevancy: {scores['answer_relevancy']:.4f}")
            print(f"Faithfulness: {scores['faithfulness']:.4f}")
            print(f"Context Precision: {scores['context_precision']:.4f}")
            print(f"Context Recall: {scores['context_recall']:.4f}")
            print(f"Average Score: {scores['average']:.4f}")
            print("Results saved to: output/ragas_law_results.json")
        else:
            print("Evaluation failed")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
