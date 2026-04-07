#!/usr/bin/env python3
"""Batch processing script: Read questions from JSONL file and batch process"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from qa_agent import QAAgent, QAAgentConfig
from agent_models import ModelConfig
from agent_tools import ToolConfig
from local_env import LocalEnv
from problem_statement import TextProblemStatement

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# File write lock
file_lock = Lock()


def process_single_question(
    question_data: Dict[str, Any],
    repo_path: Path,
    model_config: ModelConfig,
    tool_config: ToolConfig,
    max_steps: int,
    question_idx: int = 0,
    total_questions: int = 0,
) -> Optional[Dict[str, Any]]:
    """Process a single question"""
    question = question_data.get("question", "")
    if not question:
        logger.warning(f"[Q{question_idx}] Empty question, skipping")
        return None
    
    logger.info(f"[Q{question_idx}/{total_questions}] Processing: {question[:50]}...")
    
    try:
        # Create environment
        env = LocalEnv(repo_path=repo_path)
        
        # Create problem statement
        problem_statement = TextProblemStatement(text=question)
        
        # Create Agent configuration
        agent_config = QAAgentConfig(
            model=model_config,
            tools=tool_config,
            max_steps=max_steps,
        )
        
        # Create and run Agent
        agent = QAAgent(agent_config, env, problem_statement)
        result = agent.run()
        
        # Build output result
        output = {
            "question": question,
            "answer": result["answer"],
            "steps": result["steps"],
            "latency": result.get("latency", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            # Preserve other fields from original data
            **{k: v for k, v in question_data.items() if k not in ["question", "answer"]}
        }
        
        logger.info(f"[Q{question_idx}/{total_questions}] Completed in {result['steps']} steps (latency: {result.get('latency', 0):.2f}s)")
        return output
        
    except Exception as e:
        logger.error(f"[Q{question_idx}/{total_questions}] Error processing question: {e}")
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "steps": 0,
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            **{k: v for k, v in question_data.items() if k not in ["question", "answer"]}
        }


def batch_process(
    input_file: Path,
    output_file: Path,
    repo_path: Path,
    model_name: str,
    api_key: str = None,
    api_base: str = None,
    max_steps: int = 10,
    max_workers: int = 1,
):
    """Batch process questions
    
    Args:
        max_workers: Number of threads for parallel processing, default is 1 (serial processing)
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read all questions
    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append((line_num, data))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} is not valid JSON: {e}")
                continue
    
    logger.info(f"Loaded {len(questions)} questions from {input_file}")
    logger.info(f"Using {max_workers} worker(s) for parallel processing")
    
    # Create model configuration
    model_config = ModelConfig(
        name=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    
    # Create tool configuration
    tool_config = ToolConfig()
    
    # Process each question
    results = []
    completed_count = 0
    
    if max_workers == 1:
        # Serial processing
        for idx, (line_num, question_data) in enumerate(questions, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question {idx}/{len(questions)} (line {line_num})")
            logger.info(f"{'='*80}")
            
            result = process_single_question(
                question_data,
                repo_path,
                model_config,
                tool_config,
                max_steps,
                question_idx=idx,
                total_questions=len(questions),
            )
            
            if result:
                results.append(result)
                # Write results in real-time (append mode)
                with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                logger.info(f"Result saved to {output_file}")
    else:
        # Parallel processing
        def process_with_index(args):
            idx, line_num, question_data = args
            result = process_single_question(
                question_data,
                repo_path,
                model_config,
                tool_config,
                max_steps,
                question_idx=idx,
                total_questions=len(questions),
            )
            return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(process_with_index, (idx, line_num, question_data)): (idx, line_num)
                for idx, (line_num, question_data) in enumerate(questions, 1)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_question):
                idx, line_num = future_to_question[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        completed_count += 1
                        # Write results in real-time (append mode, use lock for thread safety)
                        with file_lock:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        logger.info(f"[Q{idx}] Result saved to {output_file} ({completed_count}/{len(questions)} completed)")
                except Exception as e:
                    logger.error(f"[Q{idx}] Task failed: {e}")
                    completed_count += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Batch processing completed!")
    logger.info(f"Total: {len(questions)} questions")
    logger.info(f"Success: {len(results)} results")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Batch process QA questions")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output JSONL file path")
    parser.add_argument("--repo", "-r", required=True, type=Path, help="Repository path")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum execution steps")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of threads for parallel processing, default is 1 (serial)")
    
    args = parser.parse_args()
    
    # If output file exists, clear it (restart)
    if args.output.exists():
        logger.warning(f"Output file {args.output} already exists. It will be overwritten.")
        args.output.unlink()
    
    batch_process(
        input_file=args.input,
        output_file=args.output,
        repo_path=args.repo,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

