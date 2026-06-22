# DMR — blended-RBF (Gaussian) snapshot tracking

Minimal reproduction of the **final-time** double-Mach-reflection (DMR) result:
the WENO-driven blended-RBF (Gaussian) manifold representation of the density
field at t = 0.6, compared against the WENO reference.

The method is a VarPro blended-RBF / Gaussian manifold (NOT a PINN). The WENO
field is the high-resolution reference ("truth"). The saved snapshot already
contains the reconstructed field, so plotting needs only numpy + matplotlib.

## Contents

```
dmr/
├── README.md
├── run.sh                      # one-click: render the final-time figure
├── make_summary_plot.py        # eval / plotting entry point
├── snaps/snap_t0.6000.npz      # final-time snapshot (U_basis, U_weno, errors)
│
│   # model / solver code (method definition, for reference)
├── run_iednn_varpro.py         # implicit-EDNN VarPro driver
├── dmr_setup.py                # domain + initial condition
├── weno_rhs.py                 # WENO RHS kernel
└── fit_weno_snap_varpro.py     # blended-RBF VarPro fitter
```

## Run

```bash
pip install numpy matplotlib
./run.sh        # or:  python make_summary_plot.py .
```

## Output

- `summary_snapshots.png` — WENO (truth) vs RBF (Gaussian, 1981 DOF) vs error
  for the density at the final time t = 0.6.
- `summary_errors.png` — per-frame relative error summary.

The snapshot `.npz` also stores the atom parameters (`mu_*`, `sigma_*`,
`theta_*`, `c`) and error metrics (`rL2`, `rLinf`), enough for further analysis
without re-running the solver.
