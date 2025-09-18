#!/usr/bin/env python3
"""
SWE-QA Agent Main Program
Supports single query and batch processing modes
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Dict, Any

import rich

from agent import SWEQAAgent
from config import Config
from threading import Lock

PROJECT_ROOT = Path(__file__).parent.parent.parent
def single_query(question: str, repo_path: str) -> Dict[str, Any]:
    """
    Single query mode

    Args:
        question: User question
        repo_path: Code repository path

    Returns:
        Query result
    """
    # Validate path
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")

    if not os.path.isdir(repo_path):
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    # Create agent and query
    agent = SWEQAAgent(repo_path)
    result = agent.query(question, repo_path)

    return result

def batch_process(input_file: str, output_file: str, repo_path: str) -> None:
    """
    Batch processing mode (using thread pool for parallelization)

    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
    """
    import concurrent.futures

    if not os.path.exists(input_file):
        raise ValueError(f"Input file does not exist: {input_file}")

    # Read all tasks
    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse input
                data = json.loads(line)
                query = data.get("question")
                code_base_dir = repo_path

                if not query or not code_base_dir:
                    print(f"Warning: Line {line_num} missing required fields")
                    continue

                # Validate path
                if not os.path.exists(code_base_dir):
                    print(
                        f"Warning: Line {line_num} - repository path does not exist: {code_base_dir}"
                    )
                    continue

                tasks.append((line_num, query, code_base_dir))

            except json.JSONDecodeError:
                print(f"Warning: Line {line_num} is not valid JSON")
                continue

    # Create file lock to ensure thread-safe writing
    file_lock = Lock()
    processed_count = 0

    def process_single_task(task):
        """Process a single task"""
        nonlocal processed_count
        line_num, query, code_base_dir = task
        try:
            print(f"Processing line {line_num}: {query[:50]}...")
            # Create independent agent for each thread
            agent = SWEQAAgent(repo_path=code_base_dir)
            result = agent.query(query, code_base_dir)
            
            # Immediately write results to file
            if result is not None:
                qa_obj = {
                    "question": result.get("query"),
                    "answer": result.get("answer"),
                    "status": result.get("status", "Unknown"),
                    "history_size_used": result.get("history_size_used", Config.HISTORY_WINDOW),
                    "retry_attempts": result.get("retry_attempts", 0),
                    "error": result.get("error", None)
                }
                
                with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa_obj, ensure_ascii=False) + "\n")
                    processed_count += 1
                    
                    # Display different messages based on result status
                    status = result.get("status", "Unknown")
                    retry_attempts = result.get("retry_attempts", 0)
                    history_size = result.get("history_size_used", Config.HISTORY_WINDOW)
                    
                    if status == "✅ Success":
                        if retry_attempts > 0:
                            print(f"✓ Line {line_num} completed with {retry_attempts} retries (history: {history_size}) ({processed_count}/{len(tasks)})")
                        else:
                            print(f"✓ Line {line_num} completed successfully ({processed_count}/{len(tasks)})")
                    elif status == "❌ Failed":
                        print(f"✗ Line {line_num} failed after {retry_attempts} retries: {result.get('error', 'Unknown error')} ({processed_count}/{len(tasks)})")
                    else:
                        print(f"⚠ Line {line_num} completed with status: {status} ({processed_count}/{len(tasks)})")
            
            return line_num, result
        except Exception as e:
            traceback.print_exc()
            print(f"✗ Error processing line {line_num}: {e}")
            return line_num, None

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Clear output file
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    # Use thread pool for parallel processing
    # max_workers = min(len(tasks), os.cpu_count() or 1)  # Limit maximum thread count
    max_workers = os.getenv("PARALLEL_WORKERS", 32)
    rich.print(f"Running tasks with thread pool workers {max_workers}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_task, task): task for task in tasks
        }

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(future_to_task):
            line_num, result = future.result()
            # Results have already been written to file in process_single_task, no additional processing needed here

    print(
        f"Processed {processed_count}/{len(tasks)} queries using {max_workers} threads, results saved to {output_file}"
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SWE-QA Agent: Code Repository Question Answering"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Single query mode (save results)
    single_parser = subparsers.add_parser("single", help="Single query mode with optional output file")
    single_parser.add_argument("question", help="Question to ask")
    single_parser.add_argument("repo_path", help="Path to code repository")


    # Batch processing mode
    batch_parser = subparsers.add_parser("batch", help="Batch processing mode")

    args = parser.parse_args()

    # Validate configuration
    Config.validate()

    if args.mode == "single":
        # Single query (can save results)
        result = single_query(args.question, args.repo_path)

    elif args.mode == "batch":
        # Default mode: process all configured repositories
        model = os.getenv("OPENAI_MODEL")
        REPOS_STR = os.getenv("REPOS")
        REPOS = [repo.strip() for repo in REPOS_STR.split(",")]
        for repo in REPOS:
            print(f"Processing {repo}...")
            repo_path = f"{PROJECT_ROOT}/datasets/repos/{repo}"
            input_file = f"{PROJECT_ROOT}/datasets/questions/{repo}.jsonl"
            output_file = f"{PROJECT_ROOT}/datasets/answers/swe_qa_agent/{model}/{repo}.jsonl"
            batch_process(input_file, output_file, repo_path)

    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()
