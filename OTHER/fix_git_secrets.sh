#!/bin/bash

# Create a backup of current branch
git branch backup_1$(date +%Y%m%d)

# Remove .env file from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Remove the old refs
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Create new .env.example if it doesn't exist
if [ ! -f .env.example ]; then
  echo "# Environment Variables Example
OPENAI_API_KEY=your_api_key_here
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=your_region_here" > .env.example
fi

# Update .gitignore if needed
if ! grep -q "^.env$" .gitignore; then
  echo "
# Environment Variables
.env
.env.*
!.env.example" >> .gitignore
fi

# Force push changes
echo "Ready to force push changes. Run:"
echo "git push origin --force --all" 