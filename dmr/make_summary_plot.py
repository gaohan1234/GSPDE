"""Build a single summary figure from a run_weno_track_varpro/ directory.

Usage:
    python make_summary_plot.py <run_dir>
"""
import os, sys, glob, re
import numpy as np
import matplotlib.pyplot as plt

LX, LY = 4.0, 1.0


def main():
    if len(sys.argv) < 2:
        print("usage: make_summary_plot.py <run_dir>"); sys.exit(1)
    run_dir = sys.argv[1]
    snaps = sorted(glob.glob(os.path.join(run_dir, 'snaps', 'snap_t*.npz')))
    if not snaps:
        print("no snaps found"); sys.exit(1)

    data = []
    for p in snaps:
        d = np.load(p)
        data.append((float(d['t']), d['U_basis'], d['U_weno'],
                     d['rL2'], d['rLinf']))

    n = len(data)
    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 7.5), sharex=True, sharey=True)
    if n == 1:
        axes = axes[:, None]

    # global rho color scale
    rho_min = min(min(d[1][..., 0].min(), d[2][..., 0].min()) for d in data)
    rho_max = max(max(d[1][..., 0].max(), d[2][..., 0].max()) for d in data)
    err_max_g = max(np.abs(d[1][..., 0] - d[2][..., 0]).max() for d in data)

    for j, (t, Ub, Uw, rL2, rLi) in enumerate(data):
        rho_b = Ub[..., 0]; rho_w = Uw[..., 0]; err = rho_b - rho_w
        im0 = axes[0, j].imshow(rho_w.T, origin='lower',
                                extent=[0, LX, 0, LY], aspect='auto',
                                cmap='turbo', vmin=rho_min, vmax=rho_max)
        axes[0, j].set_title(f"WENO  t={t:.3f}")
        im1 = axes[1, j].imshow(rho_b.T, origin='lower',
                                extent=[0, LX, 0, LY], aspect='auto',
                                cmap='turbo', vmin=rho_min, vmax=rho_max)
        axes[1, j].set_title(f"RBF (1981 DOF)\nrL2_$\\rho$={rL2[0]:.2e}")
        im2 = axes[2, j].imshow(err.T, origin='lower',
                                extent=[0, LX, 0, LY], aspect='auto',
                                cmap='seismic', vmin=-err_max_g, vmax=err_max_g)
        axes[2, j].set_title(f"err $\\rho$  L$\\infty$={rLi.max():.2e}")
        if j == 0:
            axes[0, j].set_ylabel("y")
            axes[1, j].set_ylabel("y")
            axes[2, j].set_ylabel("y  /  x  ->")

    for ax in axes[-1]:
        ax.set_xlabel("x")

    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01,
                 label='rho (WENO)')
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01,
                 label='rho (RBF)')
    fig.colorbar(im2, ax=axes[2, :].tolist(), fraction=0.012, pad=0.01,
                 label='rho err')
    fig.suptitle("WENO-driven blended-RBF snapshot tracking on DMR  "
                 "(200x50,  K_s=144, K_n=200, 1981 DOF  vs  40000 grid DOF)",
                 fontsize=12, y=1.02)
    out = os.path.join(run_dir, 'summary_snapshots.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved {out}")

    # also convergence plot
    fig2, ax = plt.subplots(figsize=(7, 4.2))
    ts = [d[0] for d in data]
    rL2_rho = [d[3][0] for d in data]
    rL2_E   = [d[3][3] for d in data]
    rLi_mx  = [d[4].max() for d in data]
    ax.semilogy(ts, rL2_rho, 'o-', label=r'rel L2 $\rho$')
    ax.semilogy(ts, rL2_E, 's-', label='rel L2 E')
    ax.semilogy(ts, rLi_mx, '^--', label=r'rel L$\infty$ max field')
    ax.set_xlabel('t'); ax.set_ylabel('error'); ax.grid(True, which='both', alpha=0.3)
    ax.set_title('Per-frame fit error (warm-started VarPro,  K_s=144 K_n=200)')
    ax.legend()
    out2 = os.path.join(run_dir, 'summary_errors.png')
    fig2.savefig(out2, dpi=140, bbox_inches='tight')
    print(f"saved {out2}")


if __name__ == '__main__':
    main()
