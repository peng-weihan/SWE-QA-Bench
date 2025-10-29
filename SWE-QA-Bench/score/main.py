import os
from pathlib import Path
import openai
import json
import concurrent.futures
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

def score_answer(question, reference, candidate):
    # ... existing code ...
    prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
    Evaluation Criteria and Scoring Guidelines (each scored 1 to 10):
        1. Correctness:
            10 — Completely correct; core points and details are accurate with no ambiguity.
            8-9 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
            6-7 — Partially correct; some errors or omissions, but main points are generally accurate.
            4-5 — Several errors or ambiguities that affect understanding of the core information.
            2-3 — Many errors; misleading or fails to convey key information.
            1 — Serious errors; completely wrong or misleading.
        2. Completeness:
            10 — Covers all key points from the reference answer without omission.
            8-9 — Covers most key points; only minor non-critical information missing.
            6-7 — Missing several key points; content is somewhat incomplete.
            4-5 — Important information largely missing; content is one-sided.
            2-3 — Covers very little relevant information; seriously incomplete.
            1 — Covers almost no relevant information; completely incomplete.
        3. Relevance:
            10 — Content fully focused on the question topic; no irrelevant information.
            8-9 — Mostly focused; only minor irrelevant or peripheral information.
            6-7 — Generally on topic; some off-topic content but still relevant overall.
            4-5 — Topic not sufficiently focused; contains considerable off-topic content.
            2-3 — Content deviates from topic; includes excessive irrelevant information.
            1 — Majority of content irrelevant to the question.
        4. Clarity:
            10 — Fluent language; clear and precise expression; very easy to understand.
            8-9 — Mostly fluent; clear expression with minor unclear points.
            6-7 — Generally clear; some expressions slightly unclear or not concise.
            4-5 — Expression somewhat awkward; some ambiguity or lack of fluency.
            2-3 — Language obscure; sentences are not smooth; hinders understanding.
            1 — Expression confusing; very difficult to understand.
        5. Reasoning:
            10 — Reasoning is clear, logical, and well-structured; argumentation is excellent.
            8-9 — Reasoning is clear and logical; well-structured with solid argumentation.
            6-7 — Reasoning generally reasonable; mostly clear logic; minor jumps.
            4-5 — Reasoning is average; some logical jumps or organization issues.
            2-3 — Reasoning unclear; lacks logical order; difficult to follow.
            1 — No clear reasoning; logic is chaotic.

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,10], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-10>,
        "completeness": <1-10>,
        "relevance": <1-10>,
        "clarity": <1-10>,
        "reasoning": <1-10>
        }}

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
            ],
            stream=False
        )
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
            # Validate all dimensions are in range 0-10
            for key in ["correctness", "completeness", "clarity", "relevance", "reasoning"]:
                if key not in scores or not (0 <= scores[key] <= 10):
                    print(f"Score validation failed: {key} = {scores.get(key)}")
                    return None
            return scores
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            return None
    except Exception as e:
        print(f"Scoring error: {e}")
        return None

def process_single_record(candidate_record: Dict[str, Any], reference_dict: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Function to process a single record for parallel execution"""
    try:
        question = candidate_record.get("question", "")
        candidate_answer = candidate_record.get("final_answer", "")
        
        # Get reference answer for the corresponding question from reference dictionary
        reference = reference_dict.get(question, "")
        
        if not reference:
            print(f"Skipping record: Missing reference answer")
            return None
            
        if not candidate_answer or candidate_answer.strip() == "No answer found":
            print(f"Skipping record: Candidate answer is empty or 'No answer found'")
            return None

        # Score the candidate answer
        scores = score_answer(question, reference, candidate_answer)
        
        if scores is None:
            print(f"Skipping record: Scoring failed")
            return None
        
        # Create new record with original information and dimension scores
        result_record = {
            "question": question,
            "candidate_answer": candidate_answer,
            "reference": reference,
            "correctness": scores["correctness"],
            "completeness": scores["completeness"],
            "clarity": scores["clarity"],
            "relevance": scores["relevance"],
            "reasoning": scores["reasoning"],
            "total_score": sum(scores.values())  # Calculate total score
        }
        
        print(f"Scored question: {question[:50]}... - Sub-scores: {scores} - Total: {sum(scores.values())}")
        return result_record
        
    except Exception as e:
        print(f"Error processing record: {e}")
        return None

def evaluate_jsonl_parallel(candidate_jsonl_path, reference_jsonl_path, output_jsonl_path, max_workers=16):
    """Parallel processing of JSONL files"""
    # Read reference answers and build dictionary
    reference_dict = {}
    with open(reference_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("aggregated_answer", "")
                if question and answer:
                    reference_dict[question] = answer
            except Exception as e:
                print(f"[Skip] Invalid reference answer JSON line: {e}")
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
                print(f"[Skip] Invalid candidate answer JSON line: {e}")
                continue
    
    print(f"Total read {len(candidate_records)} candidate answer records, starting parallel processing...")
    
    # Use thread pool for parallel processing
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_record = {executor.submit(process_single_record, record, reference_dict): record for record in candidate_records}
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing record: {e}")
    
    print(f"Scoring completed, processed {len(results)} records, preparing to write results...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # Write results
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results saved to: {output_jsonl_path}")
    
if __name__ == "__main__":
    repos = [
        'astropy',
        'django',
        'flask',
        'matplotlib',
        'pylint',
        'pytest',
        'requests',
        'scikit-learn',
        'sphinx',
        'sqlfluff',
        'sympy',
        'xarray',
    ]
    # Set paths
    candidate_base_path = PROJECT_ROOT / "datasets" / "answers" / os.getenv("MODEL") / os.getenv("METHOD")
    reference_base_path = PROJECT_ROOT / "datasets" / "reference"
    output_base_path = PROJECT_ROOT / "datasets" / "scores" / os.getenv("MODEL") / os.getenv("METHOD")
    
    for repo in repos:
        candidate_path = f"{candidate_base_path}/{repo}.jsonl"
        reference_path = f"{reference_base_path}/{repo}.jsonl"
        output_path = f"{output_base_path}/{repo}.jsonl"
        
        print(f"\nStarting to process {repo}...")
        print(f"Candidate answer path: {candidate_path}")
        print(f"Reference answer path: {reference_path}")
        print(f"Output path: {output_path}")
        
        # Check if files exist
        if not os.path.exists(candidate_path):
            print(f"Skipping {repo}: Candidate answer file does not exist")
            continue
        if not os.path.exists(reference_path):
            print(f"Skipping {repo}: Reference answer file does not exist")
            continue
        # Use parallel processing
        evaluate_jsonl_parallel(candidate_path, reference_path, output_path, max_workers=16)
        print(f"Completed processing {repo}")