#!/usr/bin/env bash
# auto_commit.sh — commit any staged/unstaged changes every ~15-20 mins
cd /root/vlaTraining
git add -A
if git diff --cached --quiet; then
  exit 0  # nothing to commit
fi
git commit -m "checkpoint: auto-save $(date '+%Y-%m-%d %H:%M')"
