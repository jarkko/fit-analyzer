#!/bin/bash
# Get version from setup.py
VERSION=$(grep "version=" setup.py | head -1 | sed 's/.*version="\([^"]*\)".*/\1/')
echo "fit-analyzer v$VERSION"
