#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Add at beginning of script
set -e  # Exit on error

# Add error handling function
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

# Check if debugai directory exists in current directory
if [ ! -d "debugai" ]; then
    handle_error "debugai directory not found in current directory"
fi

# Add error checks
if ! command -v gh &> /dev/null; then
    handle_error "GitHub CLI (gh) not found. Please install it first."
fi

if ! command -v python3 &> /dev/null; then
    handle_error "Python 3 not found. Please install it first."
fi

echo -e "${GREEN}Setting up DebugAI repository...${NC}"

# Change to debugai directory
cd debugai || handle_error "Could not change to debugai directory"

# Rest of the script remains the same until the PyPI part...

# Check PyPI setup
echo -e "${GREEN}Checking PyPI setup...${NC}"
if ! pip show twine > /dev/null; then
    echo -e "${GREEN}Installing twine...${NC}"
    pip install twine
fi

echo -e "${GREEN}Would you like to publish to PyPI? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    echo -e "${GREEN}Building distribution...${NC}"
    python -m build

    echo -e "${GREEN}Checking if package name is available on PyPI...${NC}"
    if ! python -m twine check dist/*; then
        handle_error "Package check failed. Please fix the issues above."
    fi

    echo -e "${GREEN}Publishing to PyPI...${NC}"
    python -m twine upload dist/* --verbose
else
    echo -e "${GREEN}Skipping PyPI publication${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}Repository is now available at: $(gh repo view --json url -q .url)${NC}"

if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    echo -e "${GREEN}Package will be available at: https://pypi.org/project/debugai/${NC}"
fi 