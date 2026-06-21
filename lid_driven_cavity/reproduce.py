"""Reproduce the lid-driven cavity result figures from trained checkpoints.

For each Reynolds number it loads the corresponding checkpoint, evaluates the
Gaussian-splatting surrogate, and saves a panel with the velocity fields and
the centerline / lid-vorticity profiles compared against the Ghia et al. (1982)
benchmark.  Run:  python reproduce.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gs_model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
OUT_DIR = os.path.join(HERE, "figures")
RES = [100, 400, 1000, 3200]
GRID = 256


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


# Ghia et al. (1982) benchmark for the four reported Reynolds numbers.
GHIA_U_Y = np.array([1.0,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
GHIA_U = {
    100:  np.array([1.0,0.84123,0.78871,0.73722,0.68717,0.23151,0.00332,-0.13641,-0.20581,-0.2109,-0.15662,-0.1015,-0.06434,-0.04775,-0.04192,-0.03717,0.0]),
    400:  np.array([1.0,0.75837,0.68439,0.61756,0.55892,0.29093,0.16256,0.02135,-0.11477,-0.17119,-0.32726,-0.24299,-0.14612,-0.10338,-0.09266,-0.08186,0.0]),
    1000: np.array([1.0,0.65928,0.57492,0.51117,0.46604,0.33304,0.18719,0.05702,-0.0608,-0.10648,-0.27805,-0.38289,-0.2973,-0.2222,-0.20196,-0.18109,0.0]),
    3200: np.array([1.0,0.53236,0.48296,0.46547,0.46101,0.34682,0.19791,0.07156,-0.04272,-0.08664,-0.24427,-0.34323,-0.41933,-0.37827,-0.35344,-0.32407,0.0]),
}
GHIA_V_X = np.array([1.0,0.9688,0.9609,0.9531,0.9453,0.9063,0.8594,0.8047,0.5,0.2344,0.2266,0.1563,0.0938,0.0781,0.0703,0.0625,0.0])
GHIA_V = {
    100:  np.array([0.0,-0.05906,-0.07391,-0.08864,-0.10313,-0.16914,-0.22445,-0.24533,0.05454,0.17527,0.17507,0.16077,0.12317,0.1089,0.10091,0.09233,0.0]),
    400:  np.array([0.0,-0.12146,-0.15663,-0.19254,-0.22847,-0.23827,-0.44993,-0.38598,0.05186,0.30174,0.30203,0.28124,0.22965,0.2092,0.19713,0.1836,0.0]),
    1000: np.array([0.0,-0.21388,-0.27669,-0.33714,-0.39188,-0.5155,-0.42665,-0.31966,0.02526,0.32235,0.33075,0.37095,0.32627,0.30353,0.29012,0.27485,0.0]),
    3200: np.array([0.0,-0.39017,-0.47425,-0.52357,-0.54053,-0.44307,-0.37401,-0.31184,0.00999,0.28188,0.2903,0.37119,0.42768,0.41906,0.40917,0.3956,0.0]),
}
GHIA_W_X = np.linspace(0.0625, 0.9375, 15)
GHIA_W = {
    100:  np.array([40.011,22.5378,16.2862,12.7844,10.4199,8.69628,7.43218,6.57451,6.13973,6.18946,6.82674,8.2211,10.7414,15.6591,30.7923]),
    400:  np.array([53.6863,34.6351,26.5825,21.0985,16.89,13.704,11.4537,10.0545,9.38889,9.34599,9.88879,11.2018,13.9068,19.6859,35.0773]),
    1000: np.array([75.598,51.0557,40.5437,32.2953,25.4341,20.2666,16.6396,14.8901,14.0928,14.1374,14.9828,16.4807,18.312,23.8707,42.1124]),
    3200: np.array([126.67,89.3391,59.6374,61.7864,47.1443,35.8795,29.4639,25.3889,24.1457,24.4639,25.8572,27.9514,30.4779,34.2327,49.9664]),
}


def evaluate(model, n):
    x = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    xy = torch.stack([X.flatten(), Y.flatten()], 1)
    with torch.no_grad():
        uvp = model(xy)
        w = model.vorticity(xy).reshape(n, n).numpy()
    u = uvp[:, 0].reshape(n, n).numpy()
    v = uvp[:, 1].reshape(n, n).numpy()
    line = torch.linspace(0, 1, n)
    with torch.no_grad():
        xy_u = torch.stack([0.5 * torch.ones_like(line), line], 1)
        xy_v = torch.stack([line, 0.5 * torch.ones_like(line)], 1)
        u_line = model(xy_u)[:, 0].numpy()
        v_line = model(xy_v)[:, 1].numpy()
    return x.numpy(), u, v, w, u_line, v_line


def panel(ax, X, Y, Z, cmap, title):
    im = ax.contourf(X, Y, Z, levels=60, cmap=cmap)
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04)
    plt.colorbar(im, cax=cax)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = get_parula_cmap()
    for Re in RES:
        model = load_model(os.path.join(CKPT_DIR, f"model_Re_{Re}.pt"))
        x, u, v, w, u_line, v_line = evaluate(model, GRID)
        X, Y = np.meshgrid(x, x, indexing="ij")
        umag = np.sqrt(u * u + v * v)

        fig, ax = plt.subplots(2, 3, figsize=(17, 11))
        panel(ax[0, 0], X, Y, u, cmap, f"u  (Re={Re})")
        panel(ax[0, 1], X, Y, v, cmap, f"v  (Re={Re})")
        panel(ax[0, 2], X, Y, umag, cmap, f"|u|  (Re={Re})")

        a = ax[1, 0]
        a.plot(u_line, x, "k-", lw=2, label="GS")
        a.plot(GHIA_U[Re], GHIA_U_Y, "ro", mfc="none", ms=6, label="Ghia (1982)")
        a.set_xlabel("u"); a.set_ylabel("y"); a.set_title("u at x=0.5")
        a.grid(ls=":", alpha=0.6); a.legend(frameon=False)

        a = ax[1, 1]
        a.plot(x, v_line, "k-", lw=2, label="GS")
        a.plot(GHIA_V_X, GHIA_V[Re], "ro", mfc="none", ms=6, label="Ghia (1982)")
        a.set_xlabel("x"); a.set_ylabel("v"); a.set_title("v at y=0.5")
        a.grid(ls=":", alpha=0.6); a.legend(frameon=False)

        a = ax[1, 2]
        a.plot(x, w[:, -1], "k-", lw=2, label="GS")
        a.plot(GHIA_W_X, GHIA_W[Re], "ro", mfc="none", ms=6, label="Ghia (1982)")
        a.set_xlim(0.02, 0.98)
        a.set_xlabel("x"); a.set_ylabel(r"$\omega$"); a.set_title("lid vorticity")
        a.grid(ls=":", alpha=0.6); a.legend(frameon=False)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"lid_driven_Re{Re}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
