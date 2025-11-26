#!/bin/bash
# Migration script for PTN to Vanta rebrand
# This script migrates api_keys.json from ptn_api to vanta_api

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PTN_API_DIR="$REPO_ROOT/ptn_api"
VANTA_API_DIR="$REPO_ROOT/vanta_api"

echo "======================================"
echo "PTN to Vanta Migration Script"
echo "======================================"
echo ""

# Create vanta_api directory if it doesn't exist
if [ ! -d "$VANTA_API_DIR" ]; then
    echo "Error: vanta_api directory not found at $VANTA_API_DIR"
    exit 1
fi

# Check if ptn_api directory exists
if [ ! -d "$PTN_API_DIR" ]; then
    echo "⚠️  ptn_api directory not found. Nothing to migrate."
    exit 0
fi

MIGRATED_COUNT=0

# Migrate api_keys.json
if [ -f "$PTN_API_DIR/api_keys.json" ]; then
    if [ -f "$VANTA_API_DIR/api_keys.json" ]; then
        echo "⚠️  api_keys.json already exists in vanta_api/"
        read -p "   Overwrite with version from ptn_api? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$PTN_API_DIR/api_keys.json" "$VANTA_API_DIR/api_keys.json"
            echo "✓ api_keys.json migrated (existing file overwritten)"
            MIGRATED_COUNT=$((MIGRATED_COUNT + 1))
        else
            echo "  Skipped api_keys.json"
        fi
    else
        cp "$PTN_API_DIR/api_keys.json" "$VANTA_API_DIR/api_keys.json"
        echo "✓ api_keys.json migrated to vanta_api/"
        MIGRATED_COUNT=$((MIGRATED_COUNT + 1))
    fi
else
    echo "ℹ️  No api_keys.json found in ptn_api/"
fi

echo ""
echo "======================================"
echo "Migration Summary"
echo "======================================"
echo "Files migrated: $MIGRATED_COUNT"
echo ""
echo "Note: The old files in ptn_api/ have been left in place."
echo "After verifying everything works, you can safely delete the ptn_api/ directory."
echo ""
echo "The system now has backwards compatibility built in, so it will:"
echo "1. First look for files in vanta_api/"
echo "2. Fall back to ptn_api/ if files are not found"
echo ""
echo "Migration complete!"
