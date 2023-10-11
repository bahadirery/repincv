#!/bin/bash

# Specify the directory to start from, or use the current directory by default
DIR="${1:-.}"

# Find and delete .DS_Store files
find "$DIR" -name ".DS_Store" -type f -delete

echo "Deleted all .DS_Store files from $DIR and its subdirectories."
