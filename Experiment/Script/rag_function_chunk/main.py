"""
Parallel RAG Function Chunk Processing Module

This module is used for parallel processing of question-answer pairs from multiple repositories,
using function chunk RAG method to generate answers. Supports multi-threaded concurrent
processing to improve efficiency.

Main Features:
- Load question-answer pair data from JSONL files
- Process questions in parallel using FuncChunkRAG model
- Statistics on processing results and answer quality metrics
- Support batch processing for multiple repositories
"""
from pathlib import Path
import sys
# Get project root directory (SWE-QA/SWE-QA)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import time
import traceback
from dotenv import load_dotenv

import os
import statistics
from methods import QAPair, ResultPair
from methods.rag_function_chunk import FuncChunkRAG

# Load environment variables from .env file
load_dotenv()

import threading
lock = threading.Lock()  # File write lock to prevent concurrent write disorder

# ==================== Important Parameter Configuration ====================
# Model configuration
MODEL = os.getenv("MODEL")

# Concurrency configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

# Load repositories from environment variable
REPOS_ENV = os.getenv("REPOS")
repos = [repo.strip() for repo in REPOS_ENV.split(",") if repo.strip()]

# Path configuration
# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASE_INPUT_PATH = PROJECT_ROOT / "datasets" / "questions"
BASE_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "answers" / "func_chunk"
EMBEDDINGS_BASE_PATH = PROJECT_ROOT / "datasets" / "faiss" / "func_chunk"
# ================================================================================

def load_data_from_jsonl(path):
    """
    Load question-answer pair data from JSONL file
    
    Args:
        path (str): Path to the JSONL file
        max_lines (int): Maximum number of lines to read, defaults to MAX_LINES
        
    Returns:
        list[QAPair]: List of QAPair objects
        
    Note:
        Skips invalid JSON lines and only returns successfully parsed QAPair objects
    """
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                qa_pair = QAPair.model_validate(data)
                data_list.append(qa_pair)
            except Exception as e:
                print(f"[Skip] Invalid JSON line: {e}")
    return data_list

def append_data_to_jsonl(path, data):
    """
    Thread-safely append data to JSONL file
    
    Args:
        path (str): Output file path
        data (dict): Data dictionary to write
        
    Note:
        Uses thread lock to ensure data integrity during concurrent writes
    """
    with lock:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def process_single_question(message: QAPair, rag: FuncChunkRAG):
    """
    Process a single question using RAG model to get answer and return result
    
    Args:
        message (QAPair): QAPair object containing question and answer
        rag (FuncChunkRAG): RAG model instance
        
    Returns:
        dict: Result dictionary containing the following fields:
            - question (str): Original question
            - answer (str): RAG-generated answer
            - ground_truth (str): Ground truth answer
            - thought (str): Reasoning process
            If processing fails, returns a dictionary containing error field
    """
    try:
        rag_qa_pair = rag.process_qa_pair(message)
        answer = rag_qa_pair.answer
        
        # Check if answer is empty or invalid
        if not answer or answer.strip() == "":
            print(f"Failed to process question: returned answer is empty")
            return {"error": "Empty answer"}
        
        # Clean markdown format, extract pure JSON
        cleaned_answer = answer.strip()
        if cleaned_answer.startswith("```json"):
            # Remove ```json start and ``` end
            cleaned_answer = cleaned_answer[7:]  # Remove ```json
            if cleaned_answer.endswith("```"):
                cleaned_answer = cleaned_answer[:-3]  # Remove ending ```
            cleaned_answer = cleaned_answer.strip()
        elif cleaned_answer.startswith("```"):
            # Handle other code block formats
            cleaned_answer = cleaned_answer[3:]
            if cleaned_answer.endswith("```"):
                cleaned_answer = cleaned_answer[:-3]
            cleaned_answer = cleaned_answer.strip()
            
        result_json = json.loads(cleaned_answer)
        result_pair = ResultPair.model_validate(result_json)
        result = {
            "question": message.question,
            "answer": result_pair.answer,
            "ground_truth": result_pair.ground_truth,
            "thought": result_pair.thought,
        }
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to process question: JSON parsing error - {str(e)}, answer content: {answer[:200] if answer else 'None'}")
        return {"error": f"JSON decode error: {str(e)}"}
    except Exception as e:
        print(f"Failed to process question: {str(e)}")
        return {"error": str(e)}
    
import concurrent.futures

def run_questions_concurrently(input_path: str, output_path: str, embeddings_save_path: str, max_workers=MAX_WORKERS):
    """
    Process question-answer pairs concurrently using RAG model to generate answers and statistics
    
    Args:
        input_path (str): Input JSONL file path
        output_path (str): Output JSONL file path
        embeddings_save_path (str): Embeddings save path
        max_workers (int): Maximum number of concurrent worker threads
        
    Returns:
        dict: Dictionary containing processing statistics:
            - processed_count (int): Number of successfully processed items
            - error_count (int): Number of errors
            - total_count (int): Total number of items
            - min_words (int): Minimum word count in answers
            - max_words (int): Maximum word count in answers
            - avg_words (float): Average word count in answers
            - median_words (float): Median word count in answers
    """
    rag = FuncChunkRAG(save_path=embeddings_save_path)
    data_list = load_data_from_jsonl(input_path)
    print(f"len(data_list): {len(data_list)}")

    # Variables for counting answer word counts
    answer_word_counts = []
    processed_count = 0
    error_count = 0

    def task(data):
        nonlocal answer_word_counts, processed_count, error_count
        try:
            data_json = data.model_dump()  # Convert to dictionary format
            # Call interface with only question to get new answer
            res = process_single_question(data, rag)
            if res.get("answer") is not None and res.get("answer") != "null":
                data_json["answer"] = res.get("answer")  # Write back new answer here
                data_json["thought"] = res.get("thought")
                data_json["ground_truth"] = res.get("ground_truth")
                append_data_to_jsonl(output_path, data_json)  # Write immediately
                # Count answer word count
                answer_word_count = len(res.get("answer").split())
                with lock:
                    answer_word_counts.append(answer_word_count)
                    processed_count += 1
                
                print(f"[Complete] Question: {data}\n Answer: {data_json['answer']} Word count: {answer_word_count}\n")
            else:
                with lock:
                    error_count += 1
        except Exception as e:
            with lock:
                error_count += 1
            print(f"[Error] Failed to process question: {data}, error: {e}")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(task, data_list)  # Execute concurrently, result writing handled by task
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user, cleaning up resources...")
        executor.shutdown(wait=False)
        raise
    
    # Statistical results
    if answer_word_counts:
        min_words = min(answer_word_counts)
        max_words = max(answer_word_counts)
        avg_words = statistics.mean(answer_word_counts)
        median_words = statistics.median(answer_word_counts)
        
        print(f"\n📊 Answer Word Count Statistics:")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Error count: {error_count}")
        print(f"  Total count: {len(data_list)}")
        print(f"  Min word count: {min_words}")
        print(f"  Max word count: {max_words}")
        print(f"  Average word count: {avg_words:.2f}")
        print(f"  Median word count: {median_words}")
        
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "total_count": len(data_list),
            "min_words": min_words,
            "max_words": max_words,
            "avg_words": avg_words,
            "median_words": median_words
        }
    else:
        print(f"\n⚠️ No successfully processed answers")
        return {
            "processed_count": 0,
            "error_count": error_count,
            "total_count": len(data_list),
            "min_words": 0,
            "max_words": 0,
            "avg_words": 0,
            "median_words": 0
        }

if __name__ == "__main__":
    for item in repos:
        try:
            start_time = time.time()
            results = run_questions_concurrently(
                input_path=f"{BASE_INPUT_PATH}/{item}.jsonl",
                output_path=f"{BASE_OUTPUT_PATH}/{MODEL}/{item}.jsonl",  
                embeddings_save_path=f"{EMBEDDINGS_BASE_PATH}/{item}_embeddings.json",
                max_workers=MAX_WORKERS,
            )
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n✨ Repository {item} processing completed, total time: {total_time:.2f} seconds")
            
            # Save statistical results to file
            stats_file = f"{BASE_OUTPUT_PATH}/{MODEL}/{item}_stats.json"
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "repository": item,
                    "processing_time": total_time,
                    "statistics": results
                }, f, ensure_ascii=False, indent=2)
            print(f"📈 Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"⚠️ Error occurred while processing repository {item}, skipping this repository, error: {e}")
            print("Complete traceback information:")
            traceback.print_exc()