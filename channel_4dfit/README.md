# Channel flow (Re_tau=395 LES) — 4D shared-Gaussian fit

Minimal reproduction package for the 4D (x, y, z, t) shared-Gaussian fit of a
channel-flow LES flow-through window. A trained checkpoint (32768 Gaussian
atoms) is loaded and evaluated to render an x-y slice comparison (truth vs fit
vs error) for the velocity/pressure fields.

This is a Gaussian-mixture field representation (NOT a PINN). Unlike the dmr /
naca packages, the model here carries trained parameters, so this eval needs
**PyTorch** (CPU is fine).

## Contents

```
channel_4dfit/
├── README.md
├── run.sh                              # one-click eval + figure
├── fit_one_flowthrough_4d_gaussian.py  # model definition (SharedGaussian4D) + helpers
├── plot_fit_slice.py                   # eval / plotting entry point
├── run/
│   ├── model_latest.pt                 # trained checkpoint (32768 atoms)
│   └── args.txt                        # run config (data_dir -> ./data)
└── data/                               # eval window only (first 36 frames)
    ├── x.npy y.npy z.npy times.npy     # grid coordinates
    └── U_x.npy U_y.npy U_z.npy p.npy   # LES reference fields (float32)
```

The full LES dataset has ~1000 frames (~1.9 GB); only the 36-frame evaluation
window referenced by `args.txt` (`start_idx=0, n_frames=36`) is shipped here,
stored in single precision (~35 MB total).

## Run

```bash
pip install numpy matplotlib torch
./run.sh
# or: python plot_fit_slice.py --run_dir run --checkpoint model_latest.pt
```

Options: `--frame_idx N` (0..35, default 18) selects which time frame to slice;
`--z_idx`, `--overlay_kernels` are also available.

## Output

- `run/xy_slice_compare.png` — for each field (u, v, w, p): truth, fit, and
  error on an x-y slice, plus Gaussian-atom y-distribution histograms.
