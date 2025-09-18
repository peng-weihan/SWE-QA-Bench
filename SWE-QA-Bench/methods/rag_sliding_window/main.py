
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import os
from dotenv import load_dotenv

import threading
lock = threading.Lock() 
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from methods import QAPair, ResultPair
from methods.rag_sliding_window import RAGSlidingWindowsCodeQA
# ==================== Important Parameter Configuration ====================
# Model configuration
MODEL = os.getenv("MODEL", "gpt-4o")

# Concurrency configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

# Repository list
REPOS_STR = os.getenv("REPOS", "requests,flask,sqlfluff,pytest,sphinx,xarray,pylint,matplotlib,scikit-learn,astropy,django,sympy")
REPOS = [repo.strip() for repo in REPOS_STR.split(",")]

# Path configuration
BASE_INPUT_PATH = PROJECT_ROOT / "datasets" / "questions"
BASE_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "answers" / "sliding_window"
EMBEDDINGS_BASE_PATH = PROJECT_ROOT / "datasets" / "faiss" / "sliding_window"

# ====================================================

def load_data_from_jsonl(path):
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
    with lock:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def process_single_question(message: QAPair, rag: RAGSlidingWindowsCodeQA):
    """
    Process a single question using RAG model to get answer
    """
    try:
        rag_qa_pair = rag.process_qa_pair(message)
        
        # Check LLM returned answer
        print(f"LLM original answer: {rag_qa_pair.answer}")
        
        # Try to parse JSON
        try:
            # Handle JSON that might be wrapped in markdown code blocks
            answer_text = rag_qa_pair.answer.strip()
            
            # If answer is wrapped in ```json, extract JSON part
            if answer_text.startswith('```json') and answer_text.endswith('```'):
                # Remove ```json and ``` markers
                json_start = answer_text.find('```json') + 7
                json_end = answer_text.rfind('```')
                answer_text = answer_text[json_start:json_end].strip()
            elif answer_text.startswith('```') and answer_text.endswith('```'):
                # Remove ``` markers
                json_start = answer_text.find('```') + 3
                json_end = answer_text.rfind('```')
                answer_text = answer_text[json_start:json_end].strip()
            
            result_json = json.loads(answer_text)
            result_pair = ResultPair.model_validate(result_json)
            result = {
                "question": message.question,
                "answer": result_pair.answer,
                "ground_truth": result_pair.ground_truth,
                "thought": result_pair.thought,
            }
            return result
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {str(json_err)}")
            print(f"Original answer content: {repr(rag_qa_pair.answer)}")
            print(f"Processed answer content: {repr(answer_text)}")
            # If JSON parsing fails, return original answer directly
            return {
                "question": message.question,
                "answer": rag_qa_pair.answer,
                "ground_truth": "",
                "thought": "",
            }
        except Exception as validation_err:
            print(f"Data validation failed: {str(validation_err)}")
            print(f"Parsed JSON: {result_json}")
            return {"error": f"Data validation failed: {str(validation_err)}"}
            
    except Exception as e:
        print(f"Failed to process question: {str(e)}")
        print(f"Question content: {message.question}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Detailed error information: {traceback.format_exc()}")
        return {"error": str(e)}

import concurrent.futures

def run_questions_concurrently(input_path: str, output_path: str, embeddings_save_path: str, max_workers=MAX_WORKERS):
    rag = RAGSlidingWindowsCodeQA(save_path=embeddings_save_path)
    data_list = load_data_from_jsonl(input_path)
    print(f"len(data_list): {len(data_list)}")
    
    answer_word_counts = []
    processed_count = 0
    error_count = 0
    
    def task(data):
        nonlocal processed_count, error_count
        try:
            data_json = data.model_dump()  # Convert to dictionary format
            # Only use question to call interface, get new answer
            res = process_single_question(data, rag)
            if res.get("answer") is not None and res.get("answer") != "null":
                data_json["answer"] = res.get("answer")  # Write back new answer here
                data_json["thought"] = res.get("thought")
                data_json["ground_truth"] = res.get("ground_truth")
                append_data_to_jsonl(output_path, data_json)  # Write immediately
                answer_word_count = len(res.get("answer").split())
                with lock:
                    answer_word_counts.append(answer_word_count)
                    processed_count += 1
                print(f"[Complete] Question: {data.question}\n Answer: {data_json['answer']} Word count: {answer_word_count}\n")
            else:
                print(f"[Skip] Question: {data.question} - Answer is empty or null")

        except Exception as e:
            with lock:
                error_count += 1
            print(f"[Error] Failed to process question: {data.question if hasattr(data, 'question') else str(data)}")
            print(f"[Error] Error type: {type(e).__name__}")
            print(f"[Error] Error message: {str(e)}")
            import traceback
            print(f"[Error] Detailed stack trace: {traceback.format_exc()}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(task, data_list)  # Execute concurrently, result writing handled by task
    
    print(f"\nProcessing completion statistics:")
    print(f"Successfully processed: {processed_count} questions")
    print(f"Processing failed: {error_count} questions")
    print(f"Average answer length: {sum(answer_word_counts) / len(answer_word_counts) if answer_word_counts else 0:.1f} words")

    return {
            "processed_count": processed_count,
            "error_count": error_count,
            "total_count": len(data_list),
            "answer_word_counts": answer_word_counts,
            "min_words": min(answer_word_counts),
            "max_words": max(answer_word_counts),
            "avg_words": sum(answer_word_counts) / len(answer_word_counts),
            "median_words": statistics.median(answer_word_counts)
    }

if __name__ == "__main__":
    for item in REPOS:
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

            stats_file = f"{BASE_OUTPUT_PATH}/{MODEL}/{item}_stats.json"
            # Ensure statistics file directory exists
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "repository": item,
                    "processing_time": total_time,
                    "statistics": results
                }, f, ensure_ascii=False, indent=2)
            print(f"📈 Statistics saved to: {stats_file}")
        except Exception as e:
            print(f"⚠️ Error occurred while processing repository {item}, skipping this repository, error message: {e}")
            print("Complete traceback information:")
            traceback.print_exc()
