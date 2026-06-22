# GSPDE

Reproduction code and trained models for "Toward Nonlinear Representations with Gaussian-Splat Manifolds for Physics-Informed Learning".

Core cases:
- `burgers_1d/` — 1D viscous Burgers (`./run.sh`)
- `lid_driven_cavity/` — lid-driven cavity (`python reproduce.py`)
- `inverse_darcy/` — inverse Darcy (`python reproduce.py`)

Extension cases:
- `naca0012/` — transonic NACA0012 airfoil (`./run.sh`)
- `dmr/` — double Mach reflection (`./run.sh`)
- `channel_4dfit/` — 4D space–time channel-flow representation (`./run.sh`)
