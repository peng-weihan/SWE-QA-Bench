import requests
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ====== Configuration Section ======
# Repository list, each element is (owner, repo)
repos = [
    ("pallets", "flask"),
    ("psf", "requests"),
    ("django", "django"),
    ("sqlfluff", "sqlfluff"),
    ("pytest-dev", "pytest"),
    ("sphinx-doc", "sphinx"),
    ("astropy", "astropy"),
    ("scikit-learn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("sympy", "sympy"),
    ("pydata","xarray"),
    ("PyCQA", "pylint"),
]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
token = os.getenv("GITHUB_TOKEN")
output_dir = PROJECT_ROOT / "datasets" / "issues"
MAX_WORKERS = 4  # Number of repositories to process in parallel

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Thread lock for thread-safe printing
print_lock = threading.Lock()

# ====== GraphQL API endpoint ======
url = "https://api.github.com/graphql"
headers = {"Authorization": f"Bearer {token}"}

# ====== GraphQL Query Template ======
query = """
query ($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issues(first: 100, after: $cursor, states: [OPEN, CLOSED]) {
      pageInfo {
        endCursor
        hasNextPage
      }
      nodes {
        id
        number
        title
        body
        state
        createdAt
        updatedAt
        closedAt
        url
        author {
          login
        }
        labels(first: 10) {
          nodes {
            name
          }
        }
      }
    }
  }
}
"""

def safe_print(message):
    """Thread-safe print function"""
    with print_lock:
        print(message)

def fetch_repo_issues(owner, repo):
    """Fetch issues from a single repository"""
    safe_print(f"\n📦 Starting to fetch issues from {owner}/{repo}...")
    output_json = os.path.join(output_dir, f"{repo}.json")
    output_csv = os.path.join(output_dir, f"{repo}.csv")

    issues = []
    cursor = None
    page = 1

    while True:
        variables = {"owner": owner, "repo": repo, "cursor": cursor}
        resp = requests.post(url, json={"query": query, "variables": variables}, headers=headers)

        if resp.status_code != 200:
            safe_print(f"❌ {owner}/{repo} request failed: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        repo_data = data.get("data", {}).get("repository", {})
        issues_data = repo_data.get("issues", {})

        nodes = issues_data.get("nodes", [])
        issues.extend(nodes)

        safe_print(f"📄 {owner}/{repo} fetched page {page}, total {len(issues)} issues")

        pageInfo = issues_data.get("pageInfo", {})
        if not pageInfo.get("hasNextPage"):
            break

        cursor = pageInfo.get("endCursor")
        page += 1

    # ====== Save as JSON ======
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)
    safe_print(f"✅ {owner}/{repo} saved to {output_json}")

    # ====== Save as CSV ======
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "number", "title", "body", "state", "created_at", "updated_at", "closed_at", "author", "labels", "url"])
        for issue in issues:
            writer.writerow([
                issue["id"],
                issue["number"],
                issue["title"],
                issue.get("body", ""),
                issue["state"],
                issue["createdAt"],
                issue["updatedAt"],
                issue.get("closedAt"),
                issue["author"]["login"] if issue["author"] else None,
                ",".join([label["name"] for label in issue["labels"]["nodes"]]),
                issue["url"]
            ])
    safe_print(f"✅ {owner}/{repo} saved to {output_csv}")
    
    return owner, repo, len(issues)

# ====== Process All Repositories in Parallel ======
print(f"🚀 Starting to fetch issues from {len(repos)} repositories in parallel, using {MAX_WORKERS} threads")
print("="*60)

completed_repos = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_repo = {executor.submit(fetch_repo_issues, owner, repo): (owner, repo) for owner, repo in repos}
    
    # Process completed tasks
    for future in as_completed(future_to_repo):
        owner, repo = future_to_repo[future]
        try:
            result = future.result()
            if result:
                completed_repos.append(result)
                safe_print(f"🎉 {result[0]}/{result[1]} completed! Fetched {result[2]} issues")
        except Exception as e:
            safe_print(f"❌ {owner}/{repo} processing exception: {e}")

# ====== Final Statistics ======
print("\n" + "="*60)
print("📊 Fetch Completion Statistics")
print("="*60)
total_issues = 0
for owner, repo, count in completed_repos:
    print(f"📦 {owner}/{repo}: {count} issues")
    total_issues += count

print(f"\n🎯 Total: {len(completed_repos)} repositories, {total_issues} issues")
print("="*60)
