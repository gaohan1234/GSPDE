# NACA0012 subsonic — teacher-forced Gaussian-atom distribution

Minimal reproduction of the **final-time** (iter 2000) Gaussian-atom
distribution in physical space for the teacher-forced NACA0012 subsonic run.
The double-Gaussian manifold (smooth + sharp atom pools) is overlaid on the
airfoil geometry as 2-sigma ellipses.

The method is a VarPro double-Gaussian manifold tracked along a curvilinear
char-WENO5 + LU-SGS teacher trajectory (NOT a PINN). The saved snapshot stores
the Gaussian-atom parameters directly, so plotting needs only
numpy + scipy + matplotlib.

## Contents

```
naca0012/
├── README.md
├── run.sh                          # one-click: render the final-time figure
├── plot_teacher_gaussians.py       # eval / plotting entry point
├── mapping_f2.npz                  # (xi,eta) -> (x,y) curvilinear mapping
├── model_snaps/model_iter02000.npz # final-time Gaussian-atom parameters
│
│   # model / solver code (method definition, for reference)
├── teacher_forced_naca.py          # teacher-forced VarPro manifold tracker
├── naca_setup.py                   # NACA0012 geometry / setup
├── weno_curv.py, weno_curv_char.py # curvilinear char-WENO5 teacher
├── lusgs_curv.py                   # LU-SGS implicit solver
├── dmr_setup.py                    # shared helpers
└── fit_weno_snap_varpro.py         # blended-RBF VarPro fitter
```

## Run

```bash
pip install numpy scipy matplotlib
./run.sh        # or:  python plot_teacher_gaussians.py --run . --mapping mapping_f2.npz
```

## Output

- `gaussian_balls_overview.png` — smooth (blue) and sharp (red) Gaussian atoms
  as 2-sigma ellipses in physical space at the final iteration, over the airfoil.
- `sharp_only_iter*.png`, `sharp_center_density_*.png` — sharp-atom views for
  the saved checkpoint.

Each `model_snaps/*.npz` stores `mu_S*`, `mu_N*`, `log_sig_*`, `theta_*`, `c`,
`scale` — the full Gaussian manifold state.
