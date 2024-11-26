#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Create timestamp for backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Define main directories
MAIN_DIRS=(
    "src"
    "releases"
    "backups"
    "docs"
    "config"
    "tests"
    "scripts"
)

# Create directory structure
create_directory_structure() {
    echo "Creating directory structure..."
    for dir in "${MAIN_DIRS[@]}"; do
        mkdir -p "$dir"
    done
    
    # Create subdirectories
    mkdir -p src/main
    mkdir -p releases/stable
    mkdir -p releases/beta
    mkdir -p backups/auto
    mkdir -p backups/manual
    mkdir -p config/env
}

# Function to get file modification time - make more robust
get_mod_time() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f %m "$1" 2>/dev/null
    else 
        stat -c %Y "$1" 2>/dev/null || echo "0"
    fi
}

# Function to categorize streamlit apps
categorize_streamlit_app() {
    local file="$1"
    local mod_time=$(get_mod_time "$file")
    local filename=$(basename "$file")
    local target_dir
    
    case "$filename" in
        RELEASE_*) target_dir="releases/stable/${filename#RELEASE_}" ;;
        *"WORKS"*) target_dir="src/main/$filename" ;;
        *"debug"*|*"copy"*) target_dir="backups/auto/$TIMESTAMP-$filename" ;;
        *) target_dir="src/main/$filename" ;;
    esac
    
    if cp "$file" "$target_dir"; then
        echo "Moved $filename to $(dirname "$target_dir")"
    else
        echo "Error moving $filename" >&2
    fi
}

# Function to organize configuration files
organize_config_files() {
    echo "Organizing configuration files..."
    
    # Move environment files
    for env_file in .env* *.cfg *.ini *.toml; do
        if [[ -f "$env_file" ]]; then
            cp "$env_file" "config/env/"
        fi
    done
    
    # Move setup files
    for setup_file in setup.* requirements.txt; do
        if [[ -f "$setup_file" ]]; then
            cp "$setup_file" "config/"
        fi
    done
}

# Main organization function
organize_repository() {
    echo "Starting repository organization..."
    
    # Create backup of current structure
    tree > "backups/manual/${TIMESTAMP}_structure.txt"
    
    # Create new directory structure
    create_directory_structure
    
    # Organize Streamlit apps
    for file in streamlit*.py; do
        if [[ -f "$file" ]]; then
            categorize_streamlit_app "$file"
        fi
    done
    
    # Organize configuration files
    organize_config_files
    
    # Move documentation
    for doc in *.md *.html; do
        if [[ -f "$doc" && "$doc" != "README.md" ]]; then
            cp "$doc" "docs/"
        fi
    done
    
    # Keep README.md in root
    if [[ -f "README.md" ]]; then
        cp "README.md" "docs/README.md"
    fi
    
    # Move scripts
    for script in *.sh; do
        if [[ -f "$script" ]]; then
            cp "$script" "scripts/"
        fi
    done
    
    # Move test files
    if [[ -d "tests" ]]; then
        cp -r tests/* "tests/"
    fi
    
    echo "Repository organization complete!"
}

# Create a manifest of changes
create_manifest() {
    echo "Creating organization manifest..."
    local manifest_file="docs/organization_manifest_${TIMESTAMP}.md"
    {
        echo "# Repository Organization Manifest"
        echo "Generated: $(date)"
        echo ""
        echo "## System Information"
        echo "- OS: $(uname -s)"
        echo "- Timestamp: ${TIMESTAMP}"
        echo ""
        echo "## Files Organized"
        echo '```'
        find . -type f -not -path "./backups/*" -exec ls -lh {} \;
        echo '```'
    } > "$manifest_file"
    
    echo "Manifest created at $manifest_file"
}

# Main execution
main() {
    # Create backup first
    echo "Creating backup..."
    mkdir -p "backups/manual/${TIMESTAMP}"
    cp -r . "backups/manual/${TIMESTAMP}/"
    
    # Run organization
    organize_repository
    
    # Create manifest
    create_manifest
    
    echo "Repository organization completed successfully!"
}

# Execute main function
main 