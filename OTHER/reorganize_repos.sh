#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Error handling
set -e
function handle_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

# Check if we're in the email crawler directory
if [[ ! -d "debugai" ]]; then
    handle_error "Please run this script from the email crawler root directory"
fi

echo -e "${GREEN}Reorganizing repositories...${NC}"

# Create parent directory for both repos if it doesn't exist
PARENT_DIR=$(dirname $(pwd))
mkdir -p "$PARENT_DIR"

# Move debugai out to be a sibling repository
echo -e "${GREEN}Moving debugai to be a separate repository...${NC}"
mv debugai "$PARENT_DIR/" || handle_error "Failed to move debugai directory"

# Navigate to debugai directory
cd "$PARENT_DIR/debugai"

# Initialize git for debugai if not already initialized
if [[ ! -d ".git" ]]; then
    git init
fi

# Create necessary directories if they don't exist
mkdir -p {docs,debugai/{cli,core,gui,integrations,utils,workflows},tests}

# Create __init__.py files in all Python package directories
find . -type d -not -path "./.git*" -not -path "./docs*" -exec touch {}/__init__.py \;

# Update version.py if it exists
echo '__version__ = "0.1.0"' > debugai/version.py

# Create main __init__.py
echo 'from .version import __version__' > debugai/__init__.py

# Set up git configuration
git config core.autocrlf false

# Create .gitignore if it doesn't exist
cat > .gitignore << 'EOL'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.idea/
.vscode/
*.swp
.DS_Store
.debugai_history.json
debugai_crash_*.json
EOL

# Make setup_repo.sh executable
chmod +x setup_repo.sh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install --upgrade pip
pip install -e ".[dev,gui]"

echo -e "${GREEN}Repository structure reorganized successfully!${NC}"
echo -e "${GREEN}debugai is now at: $PARENT_DIR/debugai${NC}"
echo -e "${GREEN}Run setup_repo.sh from within the debugai directory to complete setup${NC}"

# Instructions for next steps
echo -e "\n${GREEN}Next steps:${NC}"
echo "1. cd $PARENT_DIR/debugai"
echo "2. ./setup_repo.sh"
echo "3. git add ."
echo "4. git commit -m 'Initial commit'"
echo "5. gh repo create debugai --public --source=." 