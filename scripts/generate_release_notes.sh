#!/bin/bash
# Generate release notes from git commits since the last tag
# Usage: ./scripts/generate_release_notes.sh [version]

set -e

# Get version from argument or fail
if [ -z "$1" ]; then
    echo "❌ Usage: $0 <version>"
    echo "   Example: $0 0.3.5"
    exit 1
fi

VERSION="$1"
TAG="v${VERSION}"

# Get the last tag
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -z "$LAST_TAG" ]; then
    echo "❌ No previous tag found. Creating initial release notes..."
    COMMIT_RANGE="HEAD"
else
    echo "📝 Generating release notes from ${LAST_TAG} to ${TAG}..."
    COMMIT_RANGE="${LAST_TAG}..HEAD"
fi

# Create release notes file
NOTES_FILE="RELEASE_NOTES_${VERSION}.md"

cat > "$NOTES_FILE" << EOF
# Release ${TAG}

**Release Date:** $(date +%Y-%m-%d)

## What's Changed

EOF

# Categorize commits
echo "### 🚀 Features" >> "$NOTES_FILE"
git log $COMMIT_RANGE --oneline --grep="^feat" --grep="^feature" -i >> "$NOTES_FILE" 2>/dev/null || echo "- No new features" >> "$NOTES_FILE"

echo "" >> "$NOTES_FILE"
echo "### 🐛 Bug Fixes" >> "$NOTES_FILE"
git log $COMMIT_RANGE --oneline --grep="^fix" --grep="^bug" -i >> "$NOTES_FILE" 2>/dev/null || echo "- No bug fixes" >> "$NOTES_FILE"

echo "" >> "$NOTES_FILE"
echo "### 📚 Documentation" >> "$NOTES_FILE"
git log $COMMIT_RANGE --oneline --grep="^docs" -i >> "$NOTES_FILE" 2>/dev/null || echo "- No documentation changes" >> "$NOTES_FILE"

echo "" >> "$NOTES_FILE"
echo "### 🔧 Maintenance" >> "$NOTES_FILE"
git log $COMMIT_RANGE --oneline --grep="^chore" --grep="^refactor" --grep="^test" -i >> "$NOTES_FILE" 2>/dev/null || echo "- No maintenance changes" >> "$NOTES_FILE"

echo "" >> "$NOTES_FILE"
echo "### 📦 All Changes" >> "$NOTES_FILE"
git log $COMMIT_RANGE --oneline --pretty=format:"- %s (%h)" >> "$NOTES_FILE"

echo "" >> "$NOTES_FILE"
echo "" >> "$NOTES_FILE"
echo "**Full Changelog:** https://github.com/jarkko/fit-analyzer/compare/${LAST_TAG}...${TAG}" >> "$NOTES_FILE"

echo "✅ Release notes generated: $NOTES_FILE"
echo ""
echo "Preview:"
echo "========================================"
cat "$NOTES_FILE"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review and edit $NOTES_FILE"
echo "  2. Run: git tag -a $TAG -F $NOTES_FILE"
echo "  3. Run: git push origin $TAG"
