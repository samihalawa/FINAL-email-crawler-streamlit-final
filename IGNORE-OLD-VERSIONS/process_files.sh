#!/bin/bash

# Enable strict error handling
set -euo pipefail

process_file() {
    local file="$1"
    local basename_file
    local file_type
    local file_size
    local formatted_size
    
    # Get file information
    basename_file=$(basename "$file")
    file_type=$(file --mime-type -b "$file")
    file_size=$(stat -f%z "$file")
    formatted_size=$(numfmt --to=iec-i --suffix=B "$file_size")
    
    # Output markdown format
    echo "## ${basename_file}"
    echo
    echo "**Path:** \`${file}\`"
    echo "**Type:** ${file_type}"
    echo "**Size:** ${formatted_size}"
    echo
    echo "**First 5 lines:**"
    
    # Determine appropriate code block syntax
    case "$file_type" in
        "application/json") echo '```json' ;;
        "application/gzip") echo '```gzip' ;;
        *) echo '```' ;;
    esac
    
    # Output file content safely
    if [[ "$file_type" == "application/gzip" ]]; then
        gunzip -c "$file" 2>/dev/null | head -n 5 || echo "[Binary content]"
    else
        head -n 5 "$file" 2>/dev/null || echo "[Unable to read file]"
    fi
    
    echo '```'
    echo
}

# Main loop to process files
while IFS= read -r file; do
    if [[ -f "$file" ]]; then
        process_file "$file"
    fi
done 