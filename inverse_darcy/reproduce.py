"""Reproduce the inverse-Darcy result figures from trained checkpoints.

For each problem setting the script loads the trained checkpoints, evaluates the
Gaussian-splatting surrogate, and compares the recovered permeability nu and
pressure U against the analytic ground truth.  Run:  python reproduce.py
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gs_model import load_model, truth_fields

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
OUT_DIR = os.path.join(HERE, "figures")
RES = 128
SETTINGS = {
    "nu1": r"$\nu = 1 + 0.5\,\sin(\pi x)\sin(\pi y)$",
    "nu2": r"$\nu = 1 + 0.5\,\sin(2\pi x)\sin(2\pi y)$",
}


def get_parula_cmap():
    parula = np.array([[53,42,135],[64,45,150],[72,47,160],[83,50,175],[94,53,189],[105,56,204],
        [115,59,218],[124,63,228],[131,70,235],[137,79,241],[142,88,247],[147,97,252],[150,103,254],
        [151,109,255],[149,115,252],[147,121,246],[145,127,238],[143,133,229],[141,138,219],
        [138,144,209],[136,150,199],[134,155,189],[133,161,180],[134,166,173],[135,171,167],
        [138,175,161],[142,180,155],[148,184,149],[154,188,143],[161,192,137],[168,196,131],
        [176,199,125],[184,203,119],[193,206,113],[202,208,107],[211,211,101],[220,214,95],
        [229,216,89],[238,218,83],[247,221,77],[255,223,71],[255,226,63],[255,229,54],[255,232,45],
        [255,235,36],[255,238,27],[255,241,18],[255,244,9],[255,247,0],[255,250,0],[255,253,0],
        [255,255,0]]) / 255.0
    return LinearSegmentedColormap.from_list("parula", parula)


def rel_l2(pred, truth):
    return np.linalg.norm(pred - truth) / np.linalg.norm(truth)


def field_pair(ax_t, ax_p, truth, pred, cmap, label):
    vmin = min(truth.min(), pred.min())
    vmax = max(truth.max(), pred.max())
    ext = [0, 1, 0, 1]
    for ax, Z, ttl in ((ax_t, truth, f"True {label}"), (ax_p, pred, f"GS (mean) {label}")):
        im = ax.imshow(Z.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=(ax_t, ax_p), shrink=0.85)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = get_parula_cmap()
    for tag, desc in SETTINGS.items():
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, f"{tag}_run*.pt")))
        nus, Us = [], []
        a1 = a2 = None
        for c in ckpts:
            model, a1, a2 = load_model(c)
            nu, U = model.fields_on_grid(RES)
            nus.append(nu); Us.append(U)
        nu_mean = np.mean(nus, axis=0)
        U_mean = np.mean(Us, axis=0)
        nu_true, U_true = truth_fields(a1, a2, RES)
        print(f"[{tag}] {len(ckpts)} checkpoints | rel-L2  nu = {rel_l2(nu_mean, nu_true):.3e}"
              f"  U = {rel_l2(U_mean, U_true):.3e}")

        fig, ax = plt.subplots(2, 2, figsize=(11, 10))
        field_pair(ax[0, 0], ax[0, 1], nu_true, nu_mean, cmap, r"$\nu$")
        field_pair(ax[1, 0], ax[1, 1], U_true, U_mean, cmap, r"$U$")
        fig.suptitle(desc, fontsize=15)
        out = os.path.join(OUT_DIR, f"inverse_darcy_{tag}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
