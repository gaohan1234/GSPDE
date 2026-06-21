#!/usr/bin/env bash
# One-click reproduction of the GS Gaussian-kernel distribution figure.
set -e
cd "$(dirname "$0")"
python plot_kernels.py
echo
echo "Figure written to: $(pwd)"
