"""
LLM-as-a-Judge: Use LLM to score Q&A answers

This script uses GPT-5 Responses API to score candidate answers based on five dimensions:
- Correctness
- Completeness
- Relevance
- Clarity
- Reasoning
"""

import json
import os
import sys
import concurrent.futures
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path

from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_eval_client() -> AzureOpenAI:
    """Get evaluation client from environment variables"""
    base_url = os.getenv("EVAL_LLM_BASE_URL")
    api_version = os.getenv("EVAL_LLM_API_VERSION")
    api_key = os.getenv("EVAL_LLM_API_KEY")
    
    if not all([base_url, api_version, api_key]):
        raise ValueError("Missing required environment variables: EVAL_LLM_BASE_URL, EVAL_LLM_API_VERSION, EVAL_LLM_API_KEY")
    
    return AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
        default_headers={"X-TT-LOGID": "${your_logid}"},
    )


def get_config_from_env() -> Dict[str, Any]:
    """Get all configuration from environment variables"""
    max_workers_str = os.getenv("EVAL_MAX_WORKERS")
    max_workers = int(max_workers_str) if max_workers_str else None
    
    # Parse repo filter list (comma-separated)
    repo_filter_str = os.getenv("EVAL_REPO_FILTER", "")
    repo_filter = None
    if repo_filter_str:
        repo_filter = {repo.strip() for repo in repo_filter_str.split(",") if repo.strip()}
    
    config = {
        "candidate_path": os.getenv("EVAL_CANDIDATE_PATH"),
        "candidate_paths": os.getenv("EVAL_CANDIDATE_PATHS"),  # For batch evaluation, comma-separated
        "candidate_dir": os.getenv("EVAL_CANDIDATE_DIR"),  # Directory mode, auto-discover all answer files
        "reference_path": os.getenv("EVAL_REFERENCE_PATH"),
        "output_path": os.getenv("EVAL_OUTPUT_PATH"),
        "output_dir": os.getenv("EVAL_OUTPUT_DIR"),
        "model": os.getenv("EVAL_LLM_MODEL_NAME"),
        "max_workers": max_workers,
        "repo_filter": repo_filter,  # Repo filter list, None means no filter
    }
    return config


def score_answer(
    question: str,
    reference: str,
    candidate: str,
    eval_client: AzureOpenAI,
    model: str
) -> Optional[Dict[str, int]]:
    """
    Score candidate answer
    
    Args:
        question: Question
        reference: Reference answer
        candidate: Candidate answer
        eval_client: Azure OpenAI client
        model: Model name
        
    Returns:
        Dictionary containing scores for five dimensions, returns None if scoring fails
    """
    prompt = f"""You are a STRICT and RIGOROUS evaluator. You must rate the candidate answer STRICTLY against the reference answer. Be CONSERVATIVE with high scores - only award high scores (16-20) when the candidate answer is truly excellent and closely matches the reference answer in quality and content.

CRITICAL EVALUATION PRINCIPLES:
1. Compare the candidate answer DIRECTLY with the reference answer point by point
2. Any deviation, omission, or inaccuracy should result in score reduction
3. High scores (16-20) should be RARE - reserve them only for answers that are nearly perfect
4. Be strict about factual accuracy - even minor errors should lower the correctness score
5. Missing key points from the reference answer should significantly reduce completeness score
6. Vague or imprecise language should lower clarity scores
7. When in doubt between two score ranges, choose the LOWER one

Evaluation Criteria and Scoring Guidelines (each scored 1 to 20, total score 100):
        1. Correctness (STRICT - penalize any inaccuracies):
            20 — ONLY if completely correct with ALL core points and details accurate, matching reference answer precisely
            16-19 — Mostly correct but must have only TRIVIAL inaccuracies; any noticeable error reduces to 15 or below
            12-15 — Partially correct; has some errors or omissions that affect understanding; main points may be accurate but details are wrong
            8-11 — Several errors or ambiguities that significantly affect understanding of core information
            4-7 — Many errors; misleading or fails to convey key information correctly
            1-3 — Serious errors; completely wrong or misleading
        2. Completeness (STRICT - penalize missing information):
            20 — ONLY if covers ALL key points from reference answer without ANY omission; must match reference in depth
            16-19 — Covers most key points but missing some non-trivial information; minor omissions are acceptable
            12-15 — Missing several important key points; content is noticeably incomplete compared to reference
            8-11 — Important information largely missing; content is one-sided or superficial
            4-7 — Covers very little relevant information; seriously incomplete
            1-3 — Covers almost no relevant information; completely incomplete
        3. Relevance (STRICT - penalize off-topic content):
            20 — ONLY if content is fully focused on question topic with NO irrelevant information whatsoever
            16-19 — Mostly focused but may have minor peripheral information; any significant off-topic content reduces score
            12-15 — Generally on topic but contains some off-topic content that detracts from answer
            8-11 — Topic not sufficiently focused; contains considerable off-topic or tangential content
            4-7 — Content deviates from topic; includes excessive irrelevant information
            1-3 — Majority of content irrelevant to the question
        4. Clarity (STRICT - penalize unclear expression):
            20 — ONLY if language is exceptionally fluent, clear, and precise; very easy to understand without any ambiguity
            16-19 — Mostly fluent and clear but may have minor unclear points; any significant ambiguity reduces score
            12-15 — Generally clear but some expressions are unclear or not concise; may require effort to understand
            8-11 — Expression somewhat awkward; has ambiguity or lacks fluency that hinders understanding
            4-7 — Language obscure; sentences are not smooth; significantly hinders understanding
            1-3 — Expression confusing; very difficult to understand
        5. Reasoning (STRICT - penalize weak logic):
            20 — ONLY if reasoning is exceptionally clear, logical, and well-structured; argumentation is excellent and matches reference quality
            16-19 — Reasoning is clear and logical with solid argumentation; minor logical gaps may exist
            12-15 — Reasoning generally reasonable but has noticeable logical jumps or organization issues
            8-11 — Reasoning is average; has logical jumps or organization problems that affect understanding
            4-7 — Reasoning unclear; lacks logical order; difficult to follow
            1-3 — No clear reasoning; logic is chaotic

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,20], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-20>,
        "completeness": <1-20>,
        "relevance": <1-20>,
        "clarity": <1-20>,
        "reasoning": <1-20>
        }}

SCORING INSTRUCTIONS:
- Read the reference answer carefully and identify ALL key points, details, and structure
- Compare the candidate answer systematically against the reference answer
- For each criterion, start with a conservative score and only increase if the candidate truly deserves it
- If the candidate answer is significantly shorter, less detailed, or less precise than the reference, reduce scores accordingly
- If the candidate answer contains information not in the reference (unless it's clearly relevant and accurate), consider reducing relevance score
- When scoring, ask yourself: "Does this candidate answer match the quality and completeness of the reference answer?" If not, reduce scores
- Average or mediocre answers should receive scores in the 8-15 range, not higher
- Only truly excellent answers that closely match the reference should receive 16-20 scores

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON. Be strict and conservative with your scores."""

    try:
        # Use GPT-5 Responses API for evaluation
        response = eval_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            extra_body={
                "thinking": {
                    "include_thoughts": False,
                    "budget_tokens": 1024
                }
            },
        )
        
        # Extract response text from Responses API format
        score_str = response.choices[0].message.content.strip()
        print(f"Scoring result: {score_str}")
        
        try:
            # Clean possible code block markers
            if score_str.startswith("```json"):
                score_str = score_str[7:]  # Remove ```json
            if score_str.endswith("```"):
                score_str = score_str[:-3]  # Remove ```
            score_str = score_str.strip()
            
            # Parse JSON format scores
            scores = json.loads(score_str)
            # Validate all dimensions are in range 1-20
            required_keys = ["correctness", "completeness", "clarity", "relevance", "reasoning"]
            for key in required_keys:
                if key not in scores or not (1 <= scores[key] <= 20):
                    print(f"Score validation failed: {key} = {scores.get(key)}")
                    return None
            return scores
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Scoring error: {e}")
        return None


def process_single_record(
    candidate_record: Dict[str, Any],
    reference_dict: Dict[str, str],
    eval_client: AzureOpenAI,
    model: str
) -> Optional[Dict[str, Any]]:
    """
    Process a single record function for parallel execution
    
    Args:
        candidate_record: Candidate answer record
        reference_dict: Reference answer dictionary
        eval_client: Azure OpenAI client
        model: Model name
        
    Returns:
        Record containing scoring results, returns None if processing fails
    """
    try:
        question = candidate_record.get("question", "")
        candidate_answer = candidate_record.get("answer", "")
        
        # Get reference answer for the corresponding question from reference dictionary
        reference = reference_dict.get(question, "")
        
        if not reference:
            print(f"Skipping record: Missing reference answer")
            return None
            
        if not candidate_answer or candidate_answer.strip() == "No answer found":
            print(f"Skipping record: Candidate answer is empty or 'No answer found'")
            return None

        # Score candidate answer
        scores = score_answer(question, reference, candidate_answer, eval_client, model)
        
        if scores is None:
            print(f"Skipping record: Scoring failed")
            return None
        
        # Create new record in format similar to existing scoring files
        result_record = {
            "question": question,
            "score": {
                "correctness": scores["correctness"],
                "completeness": scores["completeness"],
                "clarity": scores["clarity"],
                "relevance": scores["relevance"],
                "reasoning": scores["reasoning"]
            }
        }
        
        print(f"Scored question: {question[:50]}... - Scores: {scores} - Total: {sum(scores.values())}")
        return result_record
        
    except Exception as e:
        print(f"Error processing record: {e}")
        return None


def find_answer_files(directory: str) -> List[str]:
    """
    Recursively find all .jsonl files in directory (including *_answers.jsonl and regular .jsonl files)
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of all found answer file paths
    """
    answer_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return answer_files
    
    if not directory_path.is_dir():
        print(f"Warning: Path is not a directory: {directory}")
        return answer_files
    
    # Recursively find all .jsonl files (exclude _score.jsonl files to avoid duplicate processing)
    for file_path in directory_path.rglob("*.jsonl"):
        if file_path.is_file() and not file_path.name.endswith("_score.jsonl"):
            answer_files.append(str(file_path))
    
    return sorted(answer_files)


def evaluate_jsonl_parallel(
    candidate_jsonl_path: str,
    reference_jsonl_path: str,
    output_jsonl_path: str,
    eval_client: AzureOpenAI,
    model: str,
    max_workers: int = 16
) -> None:
    """
    Process JSONL files in parallel
    
    Args:
        candidate_jsonl_path: Candidate answer JSONL file path
        reference_jsonl_path: Reference answer JSONL file path
        output_jsonl_path: Output JSONL file path
        eval_client: Azure OpenAI client
        model: Model name
        max_workers: Maximum number of parallel worker threads
    """
    # Read reference answers and build dictionary
    reference_dict = {}
    with open(reference_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("answer", "")
                if question and answer:
                    reference_dict[question] = answer
            except Exception as e:
                print(f"[Skipped] Invalid reference answer JSON line: {e}")
                continue
    
    print(f"Read {len(reference_dict)} reference answers")
    
    # Read candidate answer records
    candidate_records = []
    with open(candidate_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                candidate_records.append(record)
            except Exception as e:
                print(f"[Skipped] Invalid candidate answer JSON line: {e}")
                continue
    
    print(f"Total {len(candidate_records)} candidate answer records read, starting parallel processing...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # Create or clear output file (if exists, check processed questions first to avoid duplicate processing)
    processed_questions = set()
    if os.path.exists(output_jsonl_path):
        # Read processed questions
        try:
            with open(output_jsonl_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            question = record.get("question", "")
                            if question:
                                processed_questions.add(question)
                        except json.JSONDecodeError:
                            continue
            print(f"Existing output file detected, {len(processed_questions)} questions already processed, will skip them")
        except Exception as e:
            print(f"Error reading existing output file: {e}, will restart")
            processed_questions = set()
    
    # Filter out already processed questions
    if processed_questions:
        original_count = len(candidate_records)
        candidate_records = [
            record for record in candidate_records 
            if record.get("question", "") not in processed_questions
        ]
        skipped_count = original_count - len(candidate_records)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed questions, {len(candidate_records)} remaining")
    
    if not candidate_records:
        print("All questions have been processed, no need to continue")
        return
    
    # Use file lock to ensure thread-safe writing
    file_lock = threading.Lock()
    processed_count = [0]  # Use list to allow modification in closure
    
    def write_result_safely(result: Dict[str, Any]) -> None:
        """Thread-safe write single result"""
        with file_lock:
            try:
                with open(output_jsonl_path, 'a', encoding='utf-8') as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()  # Immediately flush to disk
                processed_count[0] += 1
            except Exception as e:
                print(f"Error writing result: {e}")
    
    # Use thread pool for parallel processing, write results in real-time
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_record = {
            executor.submit(process_single_record, record, reference_dict, eval_client, model): record
            for record in candidate_records
        }
        
        # Process completed tasks, write in real-time
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    write_result_safely(result)
                    if processed_count[0] % 10 == 0:  # Output progress every 10 records
                        print(f"Processed {processed_count[0]}/{len(candidate_records)} records...")
            except Exception as e:
                print(f"Error processing record: {e}")
    
    print(f"Scoring completed, processed {processed_count[0]} records in total, results saved in real-time to: {output_jsonl_path}")


def main() -> None:
    """
    Main function: Read configuration from environment variables and run evaluation
    
    Automatically determines whether to perform single file evaluation or batch evaluation based on environment variables:
    - If EVAL_CANDIDATE_DIR is set, execute directory mode batch evaluation (auto-discover all answer files)
    - If EVAL_CANDIDATE_PATHS is set, execute batch evaluation
    - If EVAL_CANDIDATE_PATH is set, execute single file evaluation
    
    Required environment variables:
    - EVAL_LLM_BASE_URL: Azure OpenAI endpoint
    - EVAL_LLM_API_VERSION: API version
    - EVAL_LLM_API_KEY: API key
    - EVAL_LLM_MODEL_NAME: Model name
    
    Single file evaluation requires:
    - EVAL_CANDIDATE_PATH: Candidate answer file path
    - EVAL_REFERENCE_PATH: Reference answer file path
    - EVAL_OUTPUT_PATH: Output file path
    - EVAL_MAX_WORKERS: Maximum parallel threads (optional, default 16)
    
    Batch evaluation requires:
    - EVAL_CANDIDATE_PATHS: Candidate answer file paths (comma-separated)
    - EVAL_REFERENCE_PATH: Reference answer file path
    - EVAL_OUTPUT_DIR: Output directory (optional)
    - EVAL_MAX_WORKERS: Maximum parallel threads (optional, default 48)
    
    Directory mode batch evaluation requires:
    - EVAL_CANDIDATE_DIR: Candidate answer directory path (recursively find all .jsonl files)
    - EVAL_REFERENCE_PATH: Reference answer file path (or directory, auto-match)
    - EVAL_OUTPUT_DIR: Output directory (optional, default same directory as candidate answer files)
    - EVAL_MAX_WORKERS: Maximum parallel threads (optional, default 48)
    - EVAL_REPO_FILTER: Repo filter list (optional, comma-separated, e.g., "requests,flask,pytest", only process files from these repos)
    """
    config = get_config_from_env()
    
    # Check API configuration
    try:
        eval_client = get_eval_client()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not config["model"]:
        print("Error: Missing model name (set EVAL_LLM_MODEL_NAME environment variable)", file=sys.stderr)
        sys.exit(1)
    
    # Determine if single file evaluation, batch evaluation, or directory mode
    if config["candidate_dir"]:
        # Directory mode: Auto-discover all answer files in directory
        print("Directory mode batch evaluation configuration detected, starting to auto-discover answer files...")
        candidate_dir = config["candidate_dir"]
        reference_path = config["reference_path"]
        output_dir = config["output_dir"]
        max_workers = config["max_workers"] or 48
        
        if not reference_path:
            print("Error: Directory mode evaluation requires EVAL_REFERENCE_PATH environment variable", file=sys.stderr)
            sys.exit(1)
        
        # Find all answer files
        candidate_list = find_answer_files(candidate_dir)
        
        if not candidate_list:
            print(f"Error: No .jsonl files found in directory {candidate_dir}", file=sys.stderr)
            sys.exit(1)
        
        # If repo filter is set, only keep matching files
        repo_filter = config.get("repo_filter")
        if repo_filter:
            filtered_list = []
            for candidate_path in candidate_list:
                candidate_basename = os.path.basename(candidate_path)
                # Extract repo name (remove _answers.jsonl or .jsonl suffix)
                if candidate_basename.endswith("_answers.jsonl"):
                    repo_name = candidate_basename[:-14]  # Remove "_answers.jsonl" (14 characters)
                elif candidate_basename.endswith(".jsonl"):
                    repo_name = candidate_basename[:-6]  # Remove ".jsonl"
                else:
                    repo_name = candidate_basename
                
                if repo_name in repo_filter:
                    filtered_list.append(candidate_path)
            
            candidate_list = filtered_list
            print(f"Applied repo filter: {sorted(repo_filter)}")
            print(f"Remaining {len(candidate_list)} answer files after filtering")
        
        if not candidate_list:
            print(f"Error: No matching answer files found after filtering", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using model: {config['model']}")
        print(f"Reference answers: {reference_path}")
        print(f"Candidate answer directory: {candidate_dir}")
        print(f"Found {len(candidate_list)} answer files")
        
        # Determine if reference answer is file or directory
        reference_is_dir = os.path.isdir(reference_path) if os.path.exists(reference_path) else False
        
        # Calculate base path to maintain relative directory structure
        candidate_base_path = Path(candidate_dir).resolve()
        
        for candidate_path in candidate_list:
            if not os.path.exists(candidate_path):
                print(f"Skipping: Candidate answer file does not exist: {candidate_path}", file=sys.stderr)
                continue
            
            candidate_path_obj = Path(candidate_path).resolve()
            candidate_basename = candidate_path_obj.name
            
            # Determine corresponding reference answer file
            if reference_is_dir:
                # If reference answer is a directory, try to match by filename
                # For example: astropy_answers.jsonl -> astropy.jsonl
                # Or: requests.jsonl -> requests.jsonl
                possible_ref_names = [
                    candidate_basename.replace("_answers.jsonl", ".jsonl"),  # Remove _answers
                    candidate_basename,  # Direct match
                ]
                matched_ref = None
                for ref_name in possible_ref_names:
                    ref_path = os.path.join(reference_path, ref_name)
                    if os.path.exists(ref_path):
                        matched_ref = ref_path
                        break
                
                if not matched_ref:
                    print(f"Skipping: Corresponding reference answer file not found: {candidate_path} (attempted matches: {possible_ref_names})", file=sys.stderr)
                    continue
                
                actual_reference_path = matched_ref
            else:
                # Reference answer is a single file, shared by all candidate answers
                actual_reference_path = reference_path
            
            # Determine output path, maintain relative directory structure
            if candidate_basename.endswith("_answers.jsonl"):
                name_without_ext = candidate_basename[:-14]  # Remove "_answers.jsonl" (14 characters)
            elif candidate_basename.endswith(".jsonl"):
                name_without_ext = candidate_basename[:-6]
            else:
                name_without_ext = candidate_basename
            
            output_filename = f"{name_without_ext}_score.jsonl"
            
            if output_dir:
                # Maintain relative directory structure
                relative_path = candidate_path_obj.relative_to(candidate_base_path)
                relative_dir = relative_path.parent
                output_path = os.path.join(output_dir, str(relative_dir), output_filename) if str(relative_dir) != "." else os.path.join(output_dir, output_filename)
            else:
                # Output to same directory as candidate answer file
                output_path = os.path.join(candidate_path_obj.parent, output_filename)
            
            print(f"\nProcessing: {candidate_path}")
            print(f"Reference answers: {actual_reference_path}")
            print(f"Output: {output_path}")
            
            try:
                evaluate_jsonl_parallel(
                    candidate_path,
                    actual_reference_path,
                    output_path,
                    eval_client,
                    config["model"],
                    max_workers
                )
                print(f"Completed: {candidate_path}")
            except Exception as e:
                print(f"Processing failed: {candidate_path} - {e}", file=sys.stderr)
                continue
        
    elif config["candidate_paths"]:
        # Batch evaluation
        print("Batch evaluation configuration detected, starting batch evaluation...")
        reference_path = config["reference_path"]
        output_dir = config["output_dir"]
        max_workers = config["max_workers"] or 48
        
        if not reference_path:
            print("Error: Batch evaluation requires EVAL_REFERENCE_PATH environment variable", file=sys.stderr)
            sys.exit(1)
        
        candidate_list = [p.strip() for p in config["candidate_paths"].split(",") if p.strip()]
        
        if not candidate_list:
            print("Error: EVAL_CANDIDATE_PATHS is empty", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.exists(reference_path):
            print(f"Error: Reference answer file does not exist: {reference_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using model: {config['model']}")
        print(f"Reference answers: {reference_path}")
        print(f"Number of candidate answer files: {len(candidate_list)}")
        
        for candidate_path in candidate_list:
            if not os.path.exists(candidate_path):
                print(f"Skipping: Candidate answer file does not exist: {candidate_path}", file=sys.stderr)
                continue
            
            candidate_dir = os.path.dirname(candidate_path)
            candidate_basename = os.path.basename(candidate_path)
            if candidate_basename.endswith(".jsonl"):
                name_without_ext = candidate_basename[:-6]
            else:
                name_without_ext = candidate_basename
            
            output_filename = f"{name_without_ext}_score.jsonl"
            
            if output_dir:
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = os.path.join(candidate_dir, output_filename)
            
            print(f"\nProcessing: {candidate_path}")
            print(f"Output: {output_path}")
            
            try:
                evaluate_jsonl_parallel(
                    candidate_path,
                    reference_path,
                    output_path,
                    eval_client,
                    config["model"],
                    max_workers
                )
                print(f"Completed: {candidate_path}")
            except Exception as e:
                print(f"Processing failed: {candidate_path} - {e}", file=sys.stderr)
                continue
        
    elif config["candidate_path"]:
        # Single file evaluation
        print("Single file evaluation configuration detected, starting evaluation...")
        candidate_path = config["candidate_path"]
        reference_path = config["reference_path"]
        output_path = config["output_path"]
        max_workers = config["max_workers"] or 16
        
        if not reference_path:
            print("Error: Single file evaluation requires EVAL_REFERENCE_PATH environment variable", file=sys.stderr)
            sys.exit(1)
        
        if not output_path:
            print("Error: Single file evaluation requires EVAL_OUTPUT_PATH environment variable", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.exists(candidate_path):
            print(f"Error: Candidate answer file does not exist: {candidate_path}", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.exists(reference_path):
            print(f"Error: Reference answer file does not exist: {reference_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Candidate answers: {candidate_path}")
        print(f"Reference answers: {reference_path}")
        print(f"Output file: {output_path}")
        print(f"Using model: {config['model']}")
        
        evaluate_jsonl_parallel(
            candidate_path,
            reference_path,
            output_path,
            eval_client,
            config["model"],
            max_workers
        )
        
    else:
        print("Error: Need to set one of the following environment variables:", file=sys.stderr)
        print("  - EVAL_CANDIDATE_DIR: Directory mode (auto-discover all answer files)", file=sys.stderr)
        print("  - EVAL_CANDIDATE_PATHS: Batch evaluation (comma-separated file paths)", file=sys.stderr)
        print("  - EVAL_CANDIDATE_PATH: Single file evaluation", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
