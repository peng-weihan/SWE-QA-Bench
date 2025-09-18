import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# 获取项目根目录（SWE-QA/SWE-QA）
PROJECT_ROOT = Path(__file__).parent.parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "datasets" / "questions"
ANSWERS_DIR = PROJECT_ROOT / "datasets" / "answers" / "direct"
# 确保目录存在
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration from environment variables
REPO_MAX_WORKERS = int(os.getenv("REPO_MAX_WORKERS", "1"))
QUESTION_MAX_WORKERS = int(os.getenv("QUESTION_MAX_WORKERS", "1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)
# Load repositories from environment variable
REPOS_ENV = os.getenv("REPOS")
repos = [repo.strip() for repo in REPOS_ENV.split(",") if repo.strip()]

def load_questions_from_file(file_path):
    """Load questions from JSONL file with existing format"""
    questions_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if 'question' in data:
                        questions_data.append(data)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return questions_data

def get_llm_answer(question: str, repo_name: str):
    """Get direct answer from LLM for a question"""
    try:
        system_messages = "You are a direct answer generator. Provide ONLY the direct answer to the question. Do not include explanations, citations, references, or any additional content. Give the most concise and direct response possible. " \
         "\\nProvide the final answer concisely and directly, without code snippets, extra explanations or commentary."
    
        user_messages = f"Repository: {repo_name}\n Question: {question}"
     
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_messages
                },
                {
                    "role": "user",
                    "content": user_messages
                }
            ],
            temperature=TEMPERATURE,
        )
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in getting answer: {e}")
        return f"Error: {e}"

def process_single_question(question_data, repo_name):
    """Process a single question and return the result"""
    try:
        print(f"Processing question: {question_data['question']}")
        question = question_data['question']
        direct_answer = get_llm_answer(question, repo_name)
        print(f"Direct answer: {direct_answer}")
        # Add direct_answer to existing data structure
        question_data['answer'] = direct_answer
        return question_data
    except Exception as e:
        print(f"Error processing question: {e}")
        question_data['answer'] = f"Error: {e}"
        return question_data

def process_repo_parallel(repo, max_workers=QUESTION_MAX_WORKERS):
    """Process all questions in a repository using parallel execution"""
    input_file = QUESTIONS_DIR / f"{repo}.jsonl"
    output_file = ANSWERS_DIR / MODEL / f"{repo}.jsonl"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    questions_data = load_questions_from_file(input_file)
    if not questions_data:
        print(f"No questions found in {input_file}")
        return
    
    print(f"Processing {repo}: Found {len(questions_data)} questions")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 实时写入：每处理完一个问题就立即保存
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(process_single_question, question_data, repo): i 
            for i, question_data in enumerate(questions_data)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(questions_data), desc=f"Processing {repo}", unit="question") as pbar:
            for future in as_completed(future_to_question):
                question_idx = future_to_question[future]
                try:
                    result = future.result()
                    # 实时写入：每处理完一个问题就立即保存
                    save_single_answer_to_file(result, output_file)
                except Exception as e:
                    print(f"Error processing question {question_idx + 1}: {e}")
                finally:
                    pbar.update(1)
    
    print(f"Completed processing repository: {repo}")

def save_single_answer_to_file(question_data, output_file):
    """Save a single question's answer to file in real-time"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(question_data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Error saving single answer: {e}")

def main():
    # Define input file paths (modify as needed)
  
    
    (ANSWERS_DIR / MODEL).mkdir(parents=True, exist_ok=True)
    
    # Configuration for parallel processing
    repo_max_workers = REPO_MAX_WORKERS  # Number of repositories to process simultaneously
    question_max_workers = QUESTION_MAX_WORKERS  # Number of questions to process simultaneously per repo
    
    print(f"Starting parallel processing with {repo_max_workers} repos and {question_max_workers} questions per repo")
    
    # Process repositories in parallel
    with ThreadPoolExecutor(max_workers=repo_max_workers) as executor:
        # Submit all repository processing tasks
        future_to_repo = {
            executor.submit(process_repo_parallel, repo, question_max_workers): repo 
            for repo in repos
        }
        # Process completed repositories
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                future.result()
                print(f"Completed processing repository: {repo}")
            except Exception as e:
                print(f"Error processing repository {repo}: {e}")
    
    print(f"\n{'='*50}")
    print("All files processed!")

if __name__ == "__main__":
    main()
