import os
import sys
import json
from pathlib import Path
import time
# Add project root directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from models.data_models import load_repository_from_json
from qa_generator.qa_generator import AgentQAGeneratorV2
from repo_parser import CodeAnalyzerSimple

import argparse
def main():
    REPO_NAME = "django" # streamlink, reflex, conan
    parser = argparse.ArgumentParser(description="Extract all code nodes from code repository")
    parser.add_argument("--output-dir", "-o", default="./codeqa/dataset/generated_questions", help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Number of questions to write per batch")
    
    args = parser.parse_args()
    output_dir = args.output_dir
    batch_size = args.batch_size
    repo_full_path = f"./repo_analysis/full_code_for_embedding/{REPO_NAME}/{REPO_NAME}_repo_full.json"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path_agent = os.path.join(output_dir, f"generated_questions_{REPO_NAME}.jsonl")

    # Analyze code repository
    start_time = time.perf_counter()
    analyzer = CodeAnalyzerSimple()
    repo = load_repository_from_json(repo_full_path)
   
    qa_agent = AgentQAGeneratorV2()

    qa_pairs_llm = qa_agent.generate_questions(repo.structure, output_path_agent)
    end_time = time.perf_counter()
    print(f"LLM-based question generation completed, saved to {output_path_agent}")
    
if __name__ == "__main__":
    main()