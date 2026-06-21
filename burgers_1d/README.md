# GS solver — 1D viscous Burgers (Gaussian-kernel distribution)

Minimal, self-contained release that reproduces the **final-time (t = 1.0)
Gaussian-kernel distribution** of the GS (Gaussian) solver for the 1D viscous
Burgers equation, at four viscosities (ν = 0.01, 0.0067, 0.0034, 0.0001).

The GS solution is a sum of a small number of Gaussian kernels evolved forward
in time from the initial condition (the solver is *checkpoint-free* — there are
no trained network weights). At the final time, each kernel has a spatial
contribution; stacking the 20 contributions recovers the GS solution. The
arrays under `data/` store, per viscosity, those final-time contributions and
the solution.

## Contents

```
burgers_1d/
├── README.md
├── requirements.txt
├── run.sh                 # one-click runner
├── plot_kernels.py        # loads data/ and renders the figure
└── data/
    ├── kernels_nu0.0100.npz
    ├── kernels_nu0.0067.npz
    ├── kernels_nu0.0034.npz
    └── kernels_nu0.0001.npz
```

Each `data/kernels_nu<nu>.npz` holds, for that viscosity: `nu`, `x`,
`kernels_final` (shape `(Nx, n_gauss)`, each Gaussian kernel's spatial
contribution at t=1.0) and `u_final` (the GS solution, equal to the sum of the
kernels).

## Run

```bash
pip install -r requirements.txt
./run.sh           # or:  python plot_kernels.py
```

## Output

- `gs_kernels_final.png` — 1×4 panel: the Gaussian-kernel distribution at the
  final time (t=1.0) for each viscosity (stacked kernel contributions with the
  total GS solution overlaid).
