#!/usr/bin/env bash
# One-click: load the trained 4D-Gaussian checkpoint and render an x-y slice
# (truth vs fit vs error) for the channel-flow LES window.
set -e
cd "$(dirname "$0")"
python plot_fit_slice.py --run_dir run --checkpoint model_latest.pt
echo
echo "Figure written to: $(pwd)/run/xy_slice_compare.png"
