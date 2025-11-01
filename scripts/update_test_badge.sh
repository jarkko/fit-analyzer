#!/bin/bash
# Update test count badge in README.md before pushing
# This ensures the badge is always accurate without relying on CI to push back

set -e

# Only run if README.md exists
if [ ! -f "README.md" ]; then
    echo "README.md not found, skipping badge update"
    exit 0
fi

# Run tests and capture count
echo "🧪 Running tests to update badge..."
TEST_OUTPUT=$(make test 2>&1 || true)

# Extract test count from output (macOS-compatible)
TEST_COUNT=$(echo "$TEST_OUTPUT" | grep -o '[0-9]* passed' | head -1 | grep -o '[0-9]*')

if [ -z "$TEST_COUNT" ]; then
    echo "⚠️  Could not determine test count, skipping badge update"
    exit 0
fi

echo "📊 Found $TEST_COUNT tests"

# Update badge in README (macOS-compatible)
CURRENT_BADGE=$(grep -o 'tests-[0-9]*%20passed' README.md | head -1)

if [ -n "$CURRENT_BADGE" ]; then
    NEW_BADGE="tests-${TEST_COUNT}%20passed"
    
    if [ "$CURRENT_BADGE" != "$NEW_BADGE" ]; then
        echo "📝 Updating badge: $CURRENT_BADGE -> $NEW_BADGE"
        sed -i.bak "s/tests-[0-9]*%20passed/tests-${TEST_COUNT}%20passed/g" README.md
        rm -f README.md.bak
        
        # Stage the change
        git add README.md
        echo "✅ Badge updated and staged"
    else
        echo "✅ Badge already up to date"
    fi
else
    echo "⚠️  No badge found in README.md"
fi
