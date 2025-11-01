#!/bin/bash
# Install git hooks from the hooks/ directory
# Run this once after cloning the repository

set -e

HOOKS_DIR="hooks"
GIT_HOOKS_DIR=".git/hooks"

echo "📦 Installing git hooks..."

for hook in "$HOOKS_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        echo "  Installing $hook_name"
        cp "$hook" "$GIT_HOOKS_DIR/$hook_name"
        chmod +x "$GIT_HOOKS_DIR/$hook_name"
    fi
done

echo "✅ Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
ls -l "$GIT_HOOKS_DIR" | grep -v sample | grep -v total || true
