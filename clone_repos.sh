#!/bin/bash

# File containing repository URLs and commit hashes, one per line: <url> <commit_hash>
REPO_FILE="./repos.txt"

# Directory to store the repositories
TARGET_DIR="./SWE-QA-Bench/datasets/repos"

mkdir -p "$TARGET_DIR"

# Loop through each line in the file
while read -r repo_url commit_hash; do
    # Extract the repository name from the URL
    repo_name=$(basename "$repo_url" .git)
    repo_path="$TARGET_DIR/$repo_name"

    if [ -d "$repo_path" ]; then
        echo "Repository $repo_name already exists, fetching updates..."
        cd "$repo_path" || exit
        git fetch origin
    else
        echo "Cloning repository $repo_name ..."
        git clone "$repo_url" "$repo_path"
        cd "$repo_path" || exit
    fi

    # Checkout the specified commit
    git checkout "$commit_hash"

    # Return to the previous directory
    cd - > /dev/null
done < "$REPO_FILE"

echo "All repositories have been cloned and switched to the specified commits."
