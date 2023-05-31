#!/bin/bash

# Set the batch size
batch_size=20

# Get the list of untracked files
untracked_files=$(git ls-files --others --exclude-standard)

# Iterate over the untracked files in batches
counter=0
for file in $untracked_files; do
    # Check the file size (in bytes)
    file_size=$(stat -c%s "$file")

    # Exclude files larger than 50MB (50 * 1024 * 1024 bytes)
    if [ $file_size -gt 52428800 ]; then
        echo "Skipping $file (size exceeds 50MB)"
        continue
    fi

    git add "$file"
    ((counter++))
    
    # Check if the batch size is reached
    if [ $counter -eq $batch_size ]; then
        # Generate a timestamp for the commit message
        timestamp=$(date +%Y-%m-%d_%H:%M:%S)
        
        # Commit the batch of files with a timestamp message
        git commit -m "Batch commit - $timestamp"
        
        # Push the commits to the current branch
        git push origin HEAD
        
        # Reset the counter for the next batch
        counter=0
    fi
done

# Check if there are remaining uncommitted files
if [ $counter -gt 0 ]; then
    # Generate a timestamp for the commit message
    timestamp=$(date +%Y-%m-%d_%H:%M:%S)
    
    # Commit the remaining files with a timestamp message
    git commit -m "Batch commit - $timestamp"
    
    # Push the commits to the current branch
    git push origin HEAD
fi
