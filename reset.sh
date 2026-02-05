#!/bin/bash
# Hard reset of Birth simulation - backs up and clears all state
# Usage: ./reset.sh [--no-backup]

set -e
cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups/backup_$TIMESTAMP"
NO_BACKUP=false

# Parse args
if [ "$1" == "--no-backup" ]; then
    NO_BACKUP=true
fi

echo "=== Birth Hard Reset ==="
echo ""

# Create backup unless --no-backup
if [ "$NO_BACKUP" = false ]; then
    echo "Creating backup at $BACKUP_DIR..."
    mkdir -p "$BACKUP_DIR"

    # Backup database
    if [ -f "data/birth.db" ]; then
        cp "data/birth.db" "$BACKUP_DIR/"
        echo "  ✓ Database backed up"
    fi

    # Backup personas
    if [ -d "data/personas" ] && [ "$(ls -A data/personas 2>/dev/null)" ]; then
        cp -r "data/personas" "$BACKUP_DIR/"
        echo "  ✓ Personas backed up ($(ls data/personas/*.json 2>/dev/null | wc -l | tr -d ' ') files)"
    fi

    # Backup outputs
    if [ -d "outputs" ] && [ "$(ls -A outputs 2>/dev/null)" ]; then
        cp -r "outputs" "$BACKUP_DIR/"
        echo "  ✓ Outputs backed up"
    fi

    # Backup drops (processed)
    if [ -d "data/drops/processed" ] && [ "$(ls -A data/drops/processed 2>/dev/null)" ]; then
        mkdir -p "$BACKUP_DIR/drops"
        cp -r "data/drops/processed" "$BACKUP_DIR/drops/"
        echo "  ✓ Processed drops backed up"
    fi

    # Backup emergent gallery content
    if [ -f "data/gallery/texts/emergent.txt" ]; then
        mkdir -p "$BACKUP_DIR/gallery/texts"
        cp "data/gallery/texts/emergent.txt" "$BACKUP_DIR/gallery/texts/"
        echo "  ✓ Emergent gallery texts backed up"
    fi
    if [ -f "data/gallery/concepts/emergent.json" ]; then
        mkdir -p "$BACKUP_DIR/gallery/concepts"
        cp "data/gallery/concepts/emergent.json" "$BACKUP_DIR/gallery/concepts/"
        echo "  ✓ Emergent gallery concepts backed up"
    fi

    echo ""
fi

# Clear state
echo "Clearing state..."

# Remove database
if [ -f "data/birth.db" ]; then
    rm "data/birth.db"
    echo "  ✓ Database removed"
fi

# Clear personas (but keep directory)
if [ -d "data/personas" ]; then
    rm -f data/personas/*.json
    echo "  ✓ Personas cleared"
fi

# Clear outputs (but keep directory structure)
if [ -d "outputs" ]; then
    rm -rf outputs/artworks/text/*
    rm -rf outputs/artworks/images/*
    rm -rf outputs/artworks/prompts/*
    rm -rf outputs/snapshots/*
    rm -rf outputs/logs/*
    echo "  ✓ Outputs cleared"
fi

# Clear processed drops
if [ -d "data/drops/processed" ]; then
    rm -f data/drops/processed/*
    echo "  ✓ Processed drops cleared"
fi

# Clear emergent gallery content (keep seeded content)
rm -f "data/gallery/texts/emergent.txt" 2>/dev/null || true
rm -f "data/gallery/concepts/emergent.json" 2>/dev/null || true
echo "  ✓ Emergent gallery content cleared"

# Ensure directories exist
mkdir -p data/personas
mkdir -p data/drops/processed
mkdir -p outputs/artworks/text
mkdir -p outputs/artworks/images
mkdir -p outputs/artworks/prompts
mkdir -p outputs/snapshots
mkdir -p outputs/logs

echo ""
echo "=== Reset Complete ==="
if [ "$NO_BACKUP" = false ]; then
    echo "Backup saved to: $BACKUP_DIR"
fi
echo "Ready for fresh start: ./run.sh"
