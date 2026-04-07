import json
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from litellm import OpenAI

load_dotenv()
# ====== Question Classification Tags ======
TAGS = {
    "What": ["Architecture exploration", "Concept / Definition", "Dependency tracing"],
    "Why": ["Design rationale", "Purpose Exploration", "Performance"],
    "Where": ["Data / Control-flow", "Feature Location", "Identifier Location"],
    "How": ["System Design", "Algorithm Implementation", "API / Framework Support"]
}

# ====== Repository Configuration ======
# Multi-repository configuration - can add more repositories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = [
    "flask",
    "requests",
    "django",
    "sqlfluff",
    "pytest",
    "sphinx",
    "astropy",
    "scikit-learn",
    "matplotlib",
    "sympy",
    "xarray",
    "pylint",
]
REPOSITORIES = []
for repo in REPO:
    REPOSITORIES.append({
        "name": repo,
        "input_json": PROJECT_ROOT / "datasets" / "issues" / f"{repo}.json",
        "output_json": PROJECT_ROOT / "datasets" / "issue_questions" / f"{repo}.json"
    })

# ====== Initialize Client ======
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

# File lock for thread-safe writing
file_lock = threading.Lock()
# Global statistics lock
stats_lock = threading.Lock()
# Global statistics
global_stats = {
    "total_repos": 0,
    "completed_repos": 0,
    "total_issues_processed": 0,
    "total_questions_generated": 0,
    "repo_stats": {}
}

def write_result_to_file(result, output_json):
    """Thread-safe writing of results to file"""
    with file_lock:
        try:
            # Read existing data
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
            
            # Add new result
            existing_data.append(result)
            
            # Write to file
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Failed to write file: {e}")

def clean_markdown_json(text: str) -> str:
    """Clean markdown code blocks and return pure JSON string"""
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def process_single_issue(issue, repository_name):
    """Process a single issue and extract questions"""
    # Skip PRs
    if "pull_request" in issue:
        return None

    issue_number = issue.get("number")
    title = issue.get("title", "")
    body = issue.get("body", "")
    
    # Filter condition: issues with body too short
    if len(body.strip()) < os.getenv("MIN_BODY_LENGTH"):  # Skip issues with body shorter than specified character count
        return None
    
    if not body.strip() and not title.strip():
        return None

    prompt = f"""
You are given a GitHub issue from the {repository_name} repository. Extract or rewrite it into one or more **short, clear, concise questions** about understanding the {repository_name} codebase, APIs, or system design.

Rules:
1. Only include questions answerable by code, documentation, or logic.
2. Ignore bug reports, environment issues, or problems that require fixing code.
3. Each question should ideally be <= 20 words.

IMPORTANT: 
- Use ONLY the exact tag names listed above. Do not use "What", "Why", "Where", "How" or any other variations.
- Be STRICT in quality control: if the issue doesn't contain meaningful questions about code understanding, return an empty questions array.
- It's better to return no questions than to generate low-quality or irrelevant questions.
- Only extract questions that genuinely help understand the {repository_name} codebase, APIs, or system design.

GitHub issue from {repository_name} repository:
Title: {title}

Body:
\"\"\"{body}\"\"\"

Output JSON format:
{{
  "issue_number": {issue_number},
  "questions": [
    {{
      "question": "...",
      "tag": "Architecture exploration"
    }}
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model=os.getenv("API_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            temperature=os.getenv("TEMPERATURE")
        )
        text = response.choices[0].message.content.strip()
        text = clean_markdown_json(text)
        if not text:
            print(f"❌ {repository_name} Issue {issue_number} LLM returned empty response")
            return None
        questions_json = json.loads(text)
        print(f"✅ {repository_name} Issue {issue_number} extraction completed")
        return questions_json
    except json.JSONDecodeError as json_err:
        print(f"❌ {repository_name} Issue {issue_number} JSON parsing failed: {json_err}")
        print(f"Cleaned response: {text}")
        return None
    except Exception as e:
        print(f"❌ {repository_name} Issue {issue_number} processing failed: {e}")
        return None

def process_single_repository(repo_config):
    """Process all issues in a single repository"""
    repo_name = repo_config["name"]
    input_json = repo_config["input_json"]
    output_json = repo_config["output_json"]
    
    with stats_lock:
        print(f"\n🚀 Starting to process repository: {repo_name}")
        print(f"📁 Input file: {input_json}")
        print(f"📁 Output file: {output_json}")
    
    # Check if input file exists
    if not os.path.exists(input_json):
        with stats_lock:
            print(f"❌ Input file does not exist: {input_json}")
        return None
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_json)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read issues
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            issues = json.load(f)
    except Exception as e:
        with stats_lock:
            print(f"❌ Failed to read issues file for {repo_name}: {e}")
        return None
    
    # Initialize output file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    # Filter issues to process
    issues_to_process = issues[:os.getenv("MAX_ISSUES_PER_REPO")]
    
    # Count filtered issues
    filtered_count = 0
    for issue in issues_to_process:
        # Skip PRs
        if "pull_request" in issue:
            continue
        
        body = issue.get("body", "")
        # Filter condition: issues with body too short
        if len(body.strip()) < os.getenv("MIN_BODY_LENGTH"):
            continue
        
        if not body.strip() and not issue.get("title", "").strip():
            continue
            
        filtered_count += 1
    
    with stats_lock:
        print(f"📊 {repo_name} original issues: {len(issues_to_process)}")
        print(f"📊 {repo_name} remaining after filtering: {filtered_count} (filtered out {len(issues_to_process) - filtered_count} issues)")
        print(f"🚀 Starting parallel processing of {filtered_count} issues using {os.getenv("MAX_WORKERS_PER_REPO")} threads")
    
    # Parallel processing
    completed_count = 0
    repo_questions = 0
    
    with ThreadPoolExecutor(max_workers=os.getenv("MAX_WORKERS_PER_REPO")) as executor:
        future_to_issue = {executor.submit(process_single_issue, issue, repo_name): issue for issue in issues_to_process}
        for future in as_completed(future_to_issue):
            issue = future_to_issue[future]
            try:
                result = future.result()
                if result:
                    write_result_to_file(result, output_json)
                    completed_count += 1
                    repo_questions += len(result.get("questions", []))
                    with stats_lock:
                        print(f"📝 {repo_name} Issue {result['issue_number']} written to file (completed: {completed_count}/{filtered_count})")
            except Exception as e:
                issue_number = issue.get("number", "unknown")
                with stats_lock:
                    print(f"❌ {repo_name} Issue {issue_number} processing exception: {e}")
    
    with stats_lock:
        print(f"✅ {repo_name} processing completed! Processed {completed_count} issues, generated {repo_questions} questions")
    
    # Return repository statistics
    return {
        "repo_name": repo_name,
        "total_issues": len(issues_to_process),
        "filtered_issues": filtered_count,
        "processed_issues": completed_count,
        "generated_questions": repo_questions
    }

def generate_repo_statistics(repo_config):
    """Generate detailed statistics for a single repository"""
    repo_name = repo_config["name"]
    output_json = repo_config["output_json"]
    
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            final_results = json.load(f)
        
        total_questions = 0
        tag_counts = {}
        questions_per_issue = []
        
        for result in final_results:
            questions = result.get("questions", [])
            total_questions += len(questions)
            questions_per_issue.append(len(questions))
            
            for question in questions:
                tag = question.get("tag", "Unknown")
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "repo_name": repo_name,
            "total_questions": total_questions,
            "tag_counts": tag_counts,
            "questions_per_issue": questions_per_issue,
            "avg_questions_per_issue": total_questions / len(final_results) if final_results else 0
        }
    except Exception as e:
        print(f"❌ Failed to generate statistics for {repo_name}: {e}")
        return None

def print_global_statistics():
    """Print global statistics"""
    print("\n" + "="*80)
    print("🌍 Global Processing Statistics")
    print("="*80)
    
    print(f"📈 Total repositories: {global_stats['total_repos']}")
    print(f"📈 Completed repositories: {global_stats['completed_repos']}")
    print(f"📈 Total issues processed: {global_stats['total_issues_processed']}")
    print(f"📈 Total questions generated: {global_stats['total_questions_generated']}")
    
    if global_stats['total_issues_processed'] > 0:
        avg_questions = global_stats['total_questions_generated'] / global_stats['total_issues_processed']
        print(f"📈 Average questions per issue: {avg_questions:.2f}")
    
    print(f"\n📊 Detailed statistics by repository:")
    print("-" * 60)
    
    for repo_name, stats in global_stats['repo_stats'].items():
        print(f"\n🏷️  {repo_name}:")
        print(f"  • Issues processed: {stats['processed_issues']}")
        print(f"  • Questions generated: {stats['generated_questions']}")
        if stats['processed_issues'] > 0:
            print(f"  • Average questions/issue: {stats['generated_questions']/stats['processed_issues']:.2f}")
        
        # Show classification statistics
        if 'tag_counts' in stats:
            print(f"  • Question categories:")
            for main_category, subcategories in TAGS.items():
                for subcategory in subcategories:
                    count = stats['tag_counts'].get(subcategory, 0)
                    if count > 0:
                        percentage = (count / stats['generated_questions'] * 100) if stats['generated_questions'] > 0 else 0
                        print(f"    - {subcategory}: {count} ({percentage:.1f}%)")

def process_repository_with_stats(repo_config):
    """Process a single repository and update statistics"""
    repo_name = repo_config["name"]
    start_time = time.time()
    
    with stats_lock:
        print(f"\n{'='*60}")
        print(f"📦 Starting to process repository: {repo_name}")
        print(f"{'='*60}")
    
    # Process single repository
    repo_stats = process_single_repository(repo_config)
    
    if repo_stats:
        # Generate detailed statistics
        detailed_stats = generate_repo_statistics(repo_config)
        
        # Thread-safe update of global statistics
        with stats_lock:
            global_stats["completed_repos"] += 1
            global_stats["total_issues_processed"] += repo_stats["processed_issues"]
            global_stats["total_questions_generated"] += repo_stats["generated_questions"]
            global_stats["repo_stats"][repo_config["name"]] = {
                "processed_issues": repo_stats["processed_issues"],
                "generated_questions": repo_stats["generated_questions"]
            }
            
            if detailed_stats:
                global_stats["repo_stats"][repo_config["name"]].update({
                    "tag_counts": detailed_stats["tag_counts"],
                    "avg_questions_per_issue": detailed_stats["avg_questions_per_issue"]
                })
        
        end_time = time.time()
        duration = end_time - start_time
        with stats_lock:
            print(f"⏱️  {repo_name} processing time: {duration:.2f} seconds")
    else:
        with stats_lock:
            print(f"❌ {repo_name} processing failed")
    
    return repo_stats

def main():
    """Main function - batch process multiple repositories (concurrent version)"""
    print("🚀 Starting batch processing of multiple repository issues (concurrent mode)")
    print(f"📋 Repository list to process: {[repo['name'] for repo in REPOSITORIES]}")
    print(f"⚙️  Configuration:")
    print(f"  - Repository concurrency: {os.getenv("REPO_CONCURRENCY")}")
    print(f"  - Max threads per repository: {os.getenv("MAX_WORKERS_PER_REPO")}")
    print(f"  - Max issues per repository: {os.getenv("MAX_ISSUES_PER_REPO")}")
    print(f"  - Minimum body length: {os.getenv("MIN_BODY_LENGTH")}")
    print(f"  - LLM model: {os.getenv("API_MODEL")}")
    print(f"  - LLM temperature: {os.getenv("TEMPERATURE")}")
    
    global_stats["total_repos"] = len(REPOSITORIES)
    
    # Concurrent processing of multiple repositories
    repo_concurrency = min(len(REPOSITORIES), os.getenv("REPO_CONCURRENCY"))  # Limit repository concurrency to avoid excessive resource consumption
    
    with ThreadPoolExecutor(max_workers=repo_concurrency) as executor:
        # Submit all repository processing tasks
        future_to_repo = {
            executor.submit(process_repository_with_stats, repo_config): repo_config 
            for repo_config in REPOSITORIES
        }
        
        # Wait for all repositories to complete processing
        for future in as_completed(future_to_repo):
            repo_config = future_to_repo[future]
            try:
                repo_stats = future.result()
                if repo_stats:
                    with stats_lock:
                        print(f"✅ {repo_config['name']} processing completed")
                else:
                    with stats_lock:
                        print(f"❌ {repo_config['name']} processing failed")
            except Exception as e:
                with stats_lock:
                    print(f"❌ {repo_config['name']} processing exception: {e}")
    
    # Print global statistics
    print_global_statistics()
    
    print(f"\n🎉 All repositories processing completed!")
    print(f"✅ Successfully processed {global_stats['completed_repos']}/{global_stats['total_repos']} repositories")

if __name__ == "__main__":
    main()
