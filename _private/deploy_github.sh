#!/bin/bash
#
# GitHub Deployment Script for Genesis Seismic Log
#

set -e

echo "======================================================"
echo "Genesis Seismic Log - GitHub Deployment"
echo "======================================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "ERROR: Not a git repository. Run 'git init' first."
    exit 1
fi

# Check if we have commits
if ! git rev-parse HEAD > /dev/null 2>&1; then
    echo "ERROR: No commits found. Run 'git commit' first."
    exit 1
fi

# Prompt for GitHub username
echo "Enter your GitHub username:"
read -r GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "ERROR: GitHub username cannot be empty."
    exit 1
fi

# Choose HTTPS or SSH
echo ""
echo "Choose authentication method:"
echo "  1) HTTPS (username/password or token)"
echo "  2) SSH (requires SSH keys configured)"
read -p "Enter choice (1 or 2): " AUTH_CHOICE

if [ "$AUTH_CHOICE" = "1" ]; then
    REMOTE_URL="https://github.com/${GITHUB_USERNAME}/genesis-seismic-log.git"
elif [ "$AUTH_CHOICE" = "2" ]; then
    REMOTE_URL="git@github.com:${GITHUB_USERNAME}/genesis-seismic-log.git"
else
    echo "ERROR: Invalid choice. Please enter 1 or 2."
    exit 1
fi

echo ""
echo "======================================================"
echo "GitHub Repository Setup"
echo "======================================================"
echo ""
echo "IMPORTANT: Before proceeding, create a GitHub repository:"
echo ""
echo "  1. Visit: https://github.com/new"
echo "  2. Repository name: genesis-seismic-log"
echo "  3. Visibility: Public (recommended)"
echo "  4. Do NOT initialize with README/license/gitignore"
echo ""
read -p "Have you created the repository? (y/n): " REPO_CREATED

if [ "$REPO_CREATED" != "y" ] && [ "$REPO_CREATED" != "Y" ]; then
    echo "Please create the repository first, then run this script again."
    exit 0
fi

# Check if remote already exists
if git remote get-url origin > /dev/null 2>&1; then
    echo ""
    echo "Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$REMOTE_URL"
else
    echo ""
    echo "Adding remote 'origin'..."
    git remote add origin "$REMOTE_URL"
fi

echo "Remote URL: $REMOTE_URL"

# Push to GitHub
echo ""
echo "======================================================"
echo "Pushing to GitHub"
echo "======================================================"
echo ""
git push -u origin main

echo ""
echo "======================================================"
echo "✓ Deployment Complete!"
echo "======================================================"
echo ""
echo "Repository URL: https://github.com/${GITHUB_USERNAME}/genesis-seismic-log"
echo "Live API: https://seismic.genesisconductor.io"
echo ""
echo "Next steps:"
echo "  1. Add topics/tags to your repository"
echo "  2. Upload social preview image (1200x630px)"
echo "  3. Share with Extropic/Tesla (see DEPLOY_TO_GITHUB.md)"
echo ""
