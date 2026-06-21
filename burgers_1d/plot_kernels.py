"""
Reproduce the GS Gaussian-kernel distribution at the final time (t = 1.0)
for the 1D viscous Burgers problem, across the four viscosities.

The GS solution is a sum of a small number of Gaussian kernels. At the final
time, each kernel has a spatial contribution; stacking the 20 contributions
recovers the GS solution (the black line). The saved arrays under ./data/
contain, for each viscosity, the final-time kernel contributions
(`kernels_final`, shape (Nx, n_gauss)) and the final-time GS solution
(`u_final`).

Run:
    python plot_kernels.py

Output (written next to this script):
    gs_kernels_final.png   # 1x4 panel: kernel distribution at t=1.0 per viscosity
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# Viscosities to show, ordered from most to least diffusive.
VISCOSITIES = [0.01, 0.0067, 0.0034, 0.0001]


def load_case(nu):
    path = os.path.join(DATA_DIR, f"kernels_nu{nu:.4f}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"Available: {sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, '*.npz')))}"
        )
    d = np.load(path)
    return {
        "nu": float(d["nu"]),
        "x": d["x"],
        "kernels": d["kernels_final"],   # (Nx, n_gauss)
        "u": d["u_final"],               # (Nx,)
    }


def main():
    cases = [load_case(nu) for nu in VISCOSITIES]
    for c in cases:
        print(f"  loaded nu={c['nu']:.4f}  kernels{c['kernels'].shape}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
    for ax, c in zip(axes, cases):
        x, kernels, u = c["x"], c["kernels"], c["u"]
        n_gauss = kernels.shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, n_gauss))

        # Stack each Gaussian kernel's contribution (signed) to show how the
        # individual "Gaussian blobs" compose the solution.
        cumsum = np.zeros_like(x)
        for k in range(n_gauss):
            ax.fill_between(x, cumsum, cumsum + kernels[:, k], color=colors[k], alpha=0.6, lw=0)
            cumsum = cumsum + kernels[:, k]
        # Total GS solution (= sum of all kernels).
        ax.plot(x, u, "k-", lw=2.5, label="GS solution")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.32, 0.32)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.set_title(rf"$\nu = {c['nu']:.4f}$", fontsize=18)
        ax.set_xlabel("Space (x)", fontsize=15)
    axes[0].set_ylabel("Gaussian-kernel contribution at $t=1.0$", fontsize=15)
    axes[-1].legend(fontsize=13, loc="upper right")

    fig.tight_layout()
    out = os.path.join(HERE, "gs_kernels_final.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  saved {os.path.basename(out)}")
    print("Done.")


if __name__ == "__main__":
    main()
