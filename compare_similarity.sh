#!/bin/bash

# Compare gpu_data_loader.cu.bak to every prior commit of gpu_data_loader.cu
# and find which commit version it's most similar to

set -e

BACKUP_FILE="gpu_data_loader.cu.bak"
TARGET_FILE="gpu_data_loader.cu"
REPO_DIR="."

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: $BACKUP_FILE not found"
    exit 1
fi

echo "Comparing $BACKUP_FILE to every commit of $TARGET_FILE..."
echo ""

# Get all commits that modified the target file
commits=$(git log --oneline -- "$TARGET_FILE" | awk '{print $1}')

if [ -z "$commits" ]; then
    echo "No commits found for $TARGET_FILE"
    exit 1
fi

# Temporary directory for extracted versions
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

cp "$BACKUP_FILE" "$tmpdir/backup"

best_similarity=0
best_commit=""
best_message=""

for commit in $commits; do
    # Extract the file version from this commit
    git show "$commit:$TARGET_FILE" > "$tmpdir/version_$commit" 2>/dev/null || continue
    
    # Calculate similarity using diff --suppress-common-lines and line counts
    # Method: count matching lines using diff
    # A better metric: use wc to count lines and diff to count differences
    
    diff_lines=$(diff "$tmpdir/backup" "$tmpdir/version_$commit" | grep -c "^[<>]" || true)
    total_lines=$(wc -l < "$tmpdir/backup")
    
    # Similarity: 1 - (diff_lines / total_lines)
    # Using integer arithmetic to avoid floating point issues
    similarity=$((100 * (total_lines - diff_lines / 2) / total_lines))
    [ $similarity -lt 0 ] && similarity=0
    [ $similarity -gt 100 ] && similarity=100
    
    commit_msg=$(git log -1 --format="%s" "$commit")
    
    printf "%-8s | Similarity: %3d%% | %s\n" "$commit" "$similarity" "$commit_msg"
    
    if [ $similarity -gt $best_similarity ]; then
        best_similarity=$similarity
        best_commit=$commit
        best_message=$commit_msg
    fi
done

echo ""
echo "=========================================="
echo "MOST SIMILAR COMMIT:"
echo "Hash:        $best_commit"
echo "Message:     $best_message"
echo "Similarity:  $best_similarity%"
echo "=========================================="
