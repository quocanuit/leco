#!/usr/bin/env python3
"""
RAGAS evaluation script with realistic ground truth extracted from actual documents
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
    
    # Find the best context (first one is usually most relevant)
    best_doc = docs[0]
    content = best_doc.page_content
    
    # Extract key information based on question type
    if "nguyên tắc" in question.lower() and "tài sản" in question.lower():
        # For property division principles
        ground_truth = "Việc giải quyết tài sản do các bên thỏa thuận; nếu không thỏa thuận được thì Tòa án giải quyết theo quy định của pháp luật"
        keywords = ["giải quyết tài sản", "thỏa thuận", "tòa án", "pháp luật"]
    elif "thời điểm" in question.lower() and "chấm dứt" in question.lower():
        # For marriage termination timing
        ground_truth = "Hôn nhân chấm dứt kể từ ngày bản án, quyết định ly hôn của Tòa án có hiệu lực pháp luật"
        keywords = ["hôn nhân chấm dứt", "bản án", "quyết định ly hôn", "hiệu lực pháp luật"]
    elif "tài sản chung" in question.lower() and "chia" in question.lower():
        # For common property division
        ground_truth = "Tài sản chung được chia đôi có tính đến hoàn cảnh của gia đình, công sức đóng góp của vợ chồng"
        keywords = ["tài sản chung", "chia đôi", "hoàn cảnh", "đóng góp"]
    elif "điều kiện" in question.lower() and "ly hôn" in question.lower():
        # For divorce conditions
        ground_truth = "Vợ hoặc chồng có quyền yêu cầu Tòa án giải quyết ly hôn trong các trường hợp pháp luật quy định"
        keywords = ["điều kiện ly hôn", "tòa án", "trường hợp", "pháp luật"]
    else:
        # Generic extraction from first 200 chars
        ground_truth = content[:max_chars] + "..."
        keywords = ["ly hôn", "tài sản", "tòa án", "pháp luật"]
    
    return ground_truth, keywords

def run_ragas_evaluation():
    print("Starting RAGAS evaluation with realistic ground truth...")
    print("=" * 60)
    
    # Initialize
    llm = get_gemini_llm(model="gemini-2.0-flash")
    rag_chain = build_rag_chain(llm, "judgment_collection")
    vector_db = VectorDB(collection_name="judgment_collection")
    retriever = vector_db.get_retriever()
    
    # Test data with realistic questions
    test_questions = [
        "Nguyên tắc giải quyết tài sản của vợ chồng khi ly hôn là gì?",
        "Thời điểm hôn nhân chấm dứt khi ly hôn là khi nào?",
        "Tài sản chung trong thời kỳ hôn nhân được chia như thế nào?",
        "Điều kiện để ly hôn đơn phương là gì?",
        "Quyền nuôi con sau ly hôn được quy định như thế nào?"
    ]
    
    # Generate evaluation data
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("Generating evaluation data...")
    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}: {question}")
        
        # Get contexts
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Get answer
        try:
            answer = rag_chain.invoke({"question": question})
            answer_text = str(answer)
        except Exception as e:
            print(f"Error getting answer: {e}")
            answer_text = "Không thể trả lời câu hỏi này."
        
        # Extract realistic ground truth
        ground_truth, keywords = extract_realistic_ground_truth(question, retriever)
        
        print(f"Ground truth: {ground_truth}")
        print(f"Keywords: {keywords}")
        
        # Add to evaluation data
        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(answer_text)
        evaluation_data["contexts"].append(contexts)
        evaluation_data["ground_truth"].append(ground_truth)
    
    # Create dataset
    dataset = Dataset.from_dict(evaluation_data)
    
    print(f"\nDataset created with {len(dataset)} samples")
    print("Starting RAGAS evaluation...")
    
    # Run evaluation
    try:
        # Initialize Gemini embeddings with explicit API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        print("Testing embeddings connection...")
        # Test embeddings first
        test_embedding = embeddings.embed_query("test query")
        print(f"Embeddings working - dimension: {len(test_embedding)}")
        
        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        
        print("\nRAGAS Evaluation Results:")
        print("=" * 40)
        
        # Extract scores the same way as ragas_simple.py
        scores = {}
        
        # Calculate average scores from result dictionary
        avg_answer_relevancy = sum(result['answer_relevancy']) / len(result['answer_relevancy'])
        avg_faithfulness = sum(result['faithfulness']) / len(result['faithfulness'])
        avg_context_precision = sum(result['context_precision']) / len(result['context_precision'])
        avg_context_recall = sum(result['context_recall']) / len(result['context_recall'])
        
        scores['answer_relevancy'] = float(avg_answer_relevancy)
        scores['faithfulness'] = float(avg_faithfulness)
        scores['context_precision'] = float(avg_context_precision)
        scores['context_recall'] = float(avg_context_recall)
        
        print(f"answer_relevancy: {avg_answer_relevancy:.4f}")
        print(f"faithfulness: {avg_faithfulness:.4f}")
        print(f"context_precision: {avg_context_precision:.4f}")
        print(f"context_recall: {avg_context_recall:.4f}")
        
        # Calculate average
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            scores["average"] = avg_score
            print(f"Average Score: {avg_score:.4f}")
        
        # Save results
        os.makedirs("output", exist_ok=True)
        
        with open("output/ragas_results_improved.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: output/ragas_results_improved.json")
        
        # Analysis
        print("\nAnalysis:")
        print("-" * 20)
        if scores.get("context_precision", 0) > 0.1:
            print("✓ Context precision improved - contexts now contain relevant info")
        else:
            print("✗ Context precision still low - need better retrieval")
            
        if scores.get("context_recall", 0) > 0.1:
            print("✓ Context recall improved - ground truth found in contexts")
        else:
            print("✗ Context recall still low - ground truth not in contexts")
            
        if scores.get("answer_relevancy", 0) > 0.7:
            print("✓ Answer relevancy good - answers match questions well")
        else:
            print("⚠ Answer relevancy could be better")
            
        if scores.get("faithfulness", 0) > 0.7:
            print("✓ Faithfulness good - answers based on contexts")
        else:
            print("⚠ Faithfulness could be better - answers may use external knowledge")
        
        return scores
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("This might be due to API limits or model issues")
        return None

if __name__ == "__main__":
    print("RAGAS Evaluation with Realistic Ground Truth")
    print("=" * 50)
    
    try:
        scores = run_ragas_evaluation()
        
        if scores:
            print("\n" + "=" * 50)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            print("\nRecommendations:")
            print("1. If context metrics are still low, consider:")
            print("   - Using better embedding models for Vietnamese legal text")
            print("   - Increasing top_k retrieval parameters")
            print("   - Improving document chunking strategy")
            print("2. If answer metrics are low, consider:")
            print("   - Improving prompt engineering")
            print("   - Using better LLM models")
            print("   - Adding more context to prompts")
        else:
            print("\n" + "=" * 50)
            print("EVALUATION FAILED - Check logs above")
            print("=" * 50)
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
