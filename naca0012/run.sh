#!/usr/bin/env bash
# One-click: draw the final-time Gaussian-atom distribution in physical space.
set -e
cd "$(dirname "$0")"
python plot_teacher_gaussians.py --run . --mapping mapping_f2.npz
echo
echo "Figure written to: $(pwd)  (gaussian_balls_overview.png)"
