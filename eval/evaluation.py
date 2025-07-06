#!/usr/bin/env python3

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

sys.path.append(os.path.join(os.path.dirname(__file__), 'ragas_lib'))
from ragas_lib.ragas_judgment import run_ragas_evaluation as evaluate_judgment
from ragas_lib.ragas_law import run_ragas_evaluation as evaluate_law

def load_config():
    config_file = "eval.json"
    
    default_config = {
        "judgment_questions": [
            "Có bao nhiêu vụ ly hôn trong dữ liệu tháng 1/2024?",
            "Làm thế nào để giành quyền nuôi con khi ly hôn?",
            "Tôi có thể ly hôn đơn phương không?",
            "Cách chia tài sản khi ly hôn theo kinh nghiệm thực tế",
            "Thủ tục ly hôn mất bao lâu theo bản án?"
        ],
        "law_questions": [
            "Tuổi tối thiểu để kết hôn theo pháp luật Việt Nam?",
            "Nguyên tắc giải quyết tài sản của vợ chồng khi ly hôn là gì?",
            "Thủ tục kết hôn theo pháp luật hiện hành"
        ]
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration...")
    
    return default_config

def main():
    print("Starting RAGAS Evaluation for Legal Collections")
    
    config = load_config()
    results = {}
    
    print("Running Judgment Collection Evaluation...")
    try:
        judgment_scores = evaluate_judgment(config.get("judgment_questions"))
        if judgment_scores:
            if 'average' in judgment_scores:
                del judgment_scores['average']
            results['judgment'] = judgment_scores
            print("Judgment evaluation completed successfully")
        else:
            results['judgment'] = None
            print("Judgment evaluation failed")
    except Exception as e:
        print(f"Error in judgment evaluation: {e}")
        results['judgment'] = None
    
    print("Running Law Collection Evaluation...")
    try:
        law_scores = evaluate_law(config.get("law_questions"))
        if law_scores:
            if 'average' in law_scores:
                del law_scores['average']
            results['law'] = law_scores
            print("Law evaluation completed successfully")
        else:
            results['law'] = None
            print("Law evaluation failed")
    except Exception as e:
        print(f"Error in law evaluation: {e}")
        results['law'] = None
    
    os.makedirs("output", exist_ok=True)
    output_file = "output/evaluation_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
