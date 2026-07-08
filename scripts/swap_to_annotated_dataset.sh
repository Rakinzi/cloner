#!/bin/bash
set -euo pipefail

ROOT="/Volumes/LaCie/WaxalNLP"
TARGET_DIR="$ROOT/sna_asr"
STAGE_DIR="$ROOT/sna_asr_annotated_stage"
BACKUP_DIR="$ROOT/sna_asr_google_backup"
EXPECTED_CLIPS=15239

cd "/Users/rakinzisilver/Desktop/cloner"

if [ -e "$STAGE_DIR" ]; then
  echo "Removing stale stage directory $STAGE_DIR"
  rm -rf "$STAGE_DIR"
fi

echo "Exporting annotated dataset into $STAGE_DIR"
HF_HUB_DISABLE_XET=1 uv run python scripts/download_waxal_sna.py --output "$STAGE_DIR"

if [ ! -f "$STAGE_DIR/metadata.csv" ]; then
  echo "metadata.csv missing from staged dataset" >&2
  exit 1
fi

if [ ! -f "$STAGE_DIR/metadata_full.csv" ]; then
  echo "metadata_full.csv missing from staged dataset" >&2
  exit 1
fi

clip_count="$(find "$STAGE_DIR/audio" -type f -name '*.wav' ! -name '._*' | wc -l | tr -d ' ')"
if [ "$clip_count" != "$EXPECTED_CLIPS" ]; then
  echo "Expected $EXPECTED_CLIPS wav files, found $clip_count" >&2
  exit 1
fi

if [ -e "$BACKUP_DIR" ]; then
  echo "Removing stale backup directory $BACKUP_DIR"
  rm -rf "$BACKUP_DIR"
fi

if [ -e "$TARGET_DIR" ]; then
  echo "Moving existing Google dataset to backup"
  mv "$TARGET_DIR" "$BACKUP_DIR"
fi

echo "Promoting staged dataset to $TARGET_DIR"
mv "$STAGE_DIR" "$TARGET_DIR"

if [ -e "$BACKUP_DIR" ]; then
  echo "Deleting old Google dataset backup"
  rm -rf "$BACKUP_DIR"
fi

echo "Annotated WAV dataset is now live at $TARGET_DIR"
