#!/bin/bash

# Enable strict error handling and debugging
set -euo pipefail
set -x

# Function for consistent timestamp generation
get_timestamp() {
    date +%Y%m%d_%H%M%S
}

# Function to check required commands
check_dependencies() {
    local deps=("tree" "find" "file" "head" "grep")
    for cmd in "${deps[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: Required command '$cmd' not found. Please install it."
            exit 1
        fi
    done
}

# Initialize variables
TIMESTAMP=$(get_timestamp)
TREE_OUTPUT="repo_structure_${TIMESTAMP}.html"
MARKDOWN_OUTPUT="repo_${TIMESTAMP}.md"
MAX_FILE_SIZE=10485760 # 10MB limit for processing

# Comprehensive exclude patterns
EXCLUDE_PATTERNS=(
    'venv'
    'env'
    '__pycache__'
    '*.egg-info'
    '*.pyc'
    '*.pyo'
    '*.swp'
    '*.swo'
    '.git'
    '.idea'
    '.vscode'
    'dist'
    'build'
    '.DS_Store'
    'node_modules'
    '*.log'
    '*.cache'
    '*.tmp'
    '*.temp'
    '*.o'
    '*.obj'
    '*.class'
)

# Check dependencies before proceeding
check_dependencies

# Create exclude string with proper escaping
EXCLUDE_STRING=$(printf "|%s" "${EXCLUDE_PATTERNS[@]}")
EXCLUDE_STRING=${EXCLUDE_STRING:1}

# Generate HTML tree structure with improved formatting
tree -I "${EXCLUDE_STRING}" \
     --charset=utf-8 \
     --dirsfirst \
     -H . \
     -L 3 \
     -T "Repository Structure" \
     --nolinks > "${TREE_OUTPUT}"

# Generate detailed markdown report
{
    echo "# Repository Structure Analysis"
    echo "Generated on: $(date)"
    echo "Generator Version: 2.0"
    echo
    echo "## Directory Tree"
    echo "\`\`\`"
    tree -I "${EXCLUDE_STRING}" --dirsfirst -L 3
    echo "\`\`\`"
    echo
    echo "## File Contents"
    echo

    # Process files with improved filtering and error handling
    find . -type f \
        -not -path '*/\.*' \
        -not -path '*/venv/*' \
        -not -path '*/env/*' \
        -not -name '*.pyc' \
        -not -name '*.pyo' \
        -not -name '*.swp' \
        -not -name '.DS_Store' \
        -size -${MAX_FILE_SIZE}c \
        | sort \
        | while read -r file; do
        
        # Skip binary and large files
        if file "$file" | grep -q "binary"; then
            continue
        fi

        # Get file type and size
        FILE_TYPE=$(file --mime-type -b "$file")
        FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        
        echo "## $(basename "$file")"
        echo
        echo "**Path:** \`$file\`"
        echo "**Type:** ${FILE_TYPE}"
        echo "**Size:** $(numfmt --to=iec-i --suffix=B ${FILE_SIZE})"
        echo
        echo "**First 5 lines:**"
        echo "\`\`\`${FILE_TYPE#*/}"
        head -n 5 "$file" 2>/dev/null || echo "Unable to read file"
        echo "\`\`\`"
        echo
    done
} > "${MARKDOWN_OUTPUT}"

# Verify outputs and display summary
if [[ -f "${TREE_OUTPUT}" && -f "${MARKDOWN_OUTPUT}" ]]; then
    echo "✅ Repository analysis completed successfully:"
    echo "📁 Tree structure: ${TREE_OUTPUT}"
    echo "📄 Markdown report: ${MARKDOWN_OUTPUT}"
    echo "📊 Total files processed: $(find . -type f | wc -l)"
else
    echo "❌ Error: Output files were not generated properly"
    exit 1
fi