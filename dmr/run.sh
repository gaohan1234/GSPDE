#!/usr/bin/env bash
# One-click: reproduce the final-time DMR summary figure from the saved snapshot.
set -e
cd "$(dirname "$0")"
python make_summary_plot.py .
echo
echo "Figures written to: $(pwd)  (summary_snapshots.png, summary_errors.png)"
