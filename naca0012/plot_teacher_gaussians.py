from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def iter_from_name(path: Path) -> int:
    stem = path.stem
    return int(stem.split('iter')[-1])


def ellipse_points(mu_x, mu_y, sig_p, sig_q, theta, n_pts=181, n_sigma=2.0):
    ang = np.linspace(0.0, 2.0 * np.pi, n_pts)
    cth = math.cos(theta)
    sth = math.sin(theta)
    xp = n_sigma * sig_p * np.cos(ang)
    yq = n_sigma * sig_q * np.sin(ang)
    xi = mu_x + cth * xp - sth * yq
    eta = mu_y + sth * xp + cth * yq
    return np.column_stack([xi, eta])


def make_interpolators(mapping_path: str):
    mp = np.load(mapping_path)
    xi_1d = mp['xi_1d'].astype(np.float64)
    eta_1d = mp['eta_1d'].astype(np.float64)
    Xphys = mp['X'].astype(np.float64)
    Yphys = mp['Y'].astype(np.float64)
    x_interp = RegularGridInterpolator((xi_1d, eta_1d), Xphys,
                                       bounds_error=False, fill_value=np.nan)
    y_interp = RegularGridInterpolator((xi_1d, eta_1d), Yphys,
                                       bounds_error=False, fill_value=np.nan)
    return xi_1d, eta_1d, Xphys, Yphys, x_interp, y_interp


def map_curve(curve_xieta, x_interp, y_interp):
    x = x_interp(curve_xieta)
    y = y_interp(curve_xieta)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def draw_pool(ax, snap, prefix, color, label, x_interp, y_interp, linewidth, alpha):
    mu_x = snap[f'mu_{prefix}x']
    mu_y = snap[f'mu_{prefix}y']
    sig_p = np.exp(snap[f'log_sig_{prefix}'])
    sig_q = np.exp(snap[f'log_sig_{prefix}_perp'])
    theta = snap[f'theta_{prefix}']
    first = True
    for idx in range(mu_x.shape[0]):
        curve = ellipse_points(mu_x[idx], mu_y[idx], sig_p[idx], sig_q[idx],
                               theta[idx])
        x, y = map_curve(curve, x_interp, y_interp)
        if x.size == 0:
            continue
        ax.plot(x, y, color=color, lw=linewidth, alpha=alpha,
                label=label if first else None)
        first = False
    centers = np.column_stack([mu_x, mu_y])
    cx = x_interp(centers)
    cy = y_interp(centers)
    good = np.isfinite(cx) & np.isfinite(cy)
    ax.scatter(cx[good], cy[good], s=8 if prefix == 'N' else 5,
               c=color, alpha=min(alpha + 0.15, 0.95), edgecolors='none')


def mapped_centers(snap, prefix, x_interp, y_interp):
    centers = np.column_stack([snap[f'mu_{prefix}x'], snap[f'mu_{prefix}y']])
    cx = x_interp(centers)
    cy = y_interp(centers)
    good = np.isfinite(cx) & np.isfinite(cy)
    return centers[good], cx[good], cy[good]


def plot_snapshot(ax, snap_path: Path, x_interp, y_interp, Xphys, Yphys):
    snap = np.load(snap_path)
    draw_pool(ax, snap, 'S', '#1f77b4', 'smooth', x_interp, y_interp,
              linewidth=0.6, alpha=0.18)
    draw_pool(ax, snap, 'N', '#d62728', 'sharp', x_interp, y_interp,
              linewidth=0.8, alpha=0.38)
    ax.plot(Xphys[:, 0], Yphys[:, 0], color='black', lw=1.4)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.6)
    ax.set_title(f'iter {iter_from_name(snap_path)}')
    ax.grid(True, alpha=0.15)


def plot_sharp_only(ax, snap_path: Path, x_interp, y_interp, Xphys, Yphys):
    snap = np.load(snap_path)
    draw_pool(ax, snap, 'N', '#d62728', 'sharp', x_interp, y_interp,
              linewidth=0.9, alpha=0.42)
    ax.plot(Xphys[:, 0], Yphys[:, 0], color='black', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.6)
    ax.set_title(f'sharp only, iter {iter_from_name(snap_path)}')
    ax.grid(True, alpha=0.15)


def save_sharp_triptych(run_dir, snap_paths, x_interp, y_interp, Xphys, Yphys):
    fig, axes = plt.subplots(1, len(snap_paths), figsize=(5.2 * len(snap_paths), 4.8))
    if len(snap_paths) == 1:
        axes = [axes]
    for ax, snap_path in zip(axes, snap_paths):
        plot_sharp_only(ax, snap_path, x_interp, y_interp, Xphys, Yphys)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=1, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_path = run_dir / 'sharp_only_iter200_800_2000.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_sharp_center_density(run_dir, snap_paths, x_interp, y_interp, Xphys, Yphys):
    all_rows = []
    x_all = []
    y_all = []
    for snap_path in snap_paths:
        it = iter_from_name(snap_path)
        snap = np.load(snap_path)
        centers_xieta, cx, cy = mapped_centers(snap, 'N', x_interp, y_interp)
        x_all.append(cx)
        y_all.append(cy)
        for (mu_xi, mu_eta), x_val, y_val in zip(centers_xieta, cx, cy):
            all_rows.append([it, mu_xi, mu_eta, x_val, y_val])

    x_all = np.concatenate(x_all) if x_all else np.array([])
    y_all = np.concatenate(y_all) if y_all else np.array([])

    csv_path = run_dir / 'sharp_centers_iter200_800_2000.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'mu_xi', 'mu_eta', 'x_phys', 'y_phys'])
        writer.writerows(all_rows)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(x_all, y_all, gridsize=40, cmap='inferno', mincnt=1)
    ax.plot(Xphys[:, 0], Yphys[:, 0], color='cyan', lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.6)
    ax.set_title('Sharp-center density, iters 200 / 800 / 2000')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.12)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('sharp-center count')
    fig.tight_layout()
    out_path = run_dir / 'sharp_center_density_iter200_800_2000.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--mapping', default='mapping_f2.npz')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    snap_dir = run_dir / 'model_snaps'
    snap_paths = sorted(snap_dir.glob('model_iter*.npz'), key=iter_from_name)
    if not snap_paths:
        raise SystemExit(f'no model snapshots found in {snap_dir}')

    mapping_path = args.mapping
    if not os.path.isabs(mapping_path):
        mapping_path = str((Path.cwd() / mapping_path).resolve())
    _, _, Xphys, Yphys, x_interp, y_interp = make_interpolators(mapping_path)

    n = len(snap_paths)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, snap_path in zip(flat_axes, snap_paths):
        plot_snapshot(ax, snap_path, x_interp, y_interp, Xphys, Yphys)
    for ax in flat_axes[n:]:
        ax.axis('off')

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.suptitle('Teacher-forced Gaussian atoms in physical space (2-sigma ellipses)', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

    out_path = Path(args.out) if args.out else run_dir / 'gaussian_balls_overview.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(out_path)

    selected_iters = {200, 800, 2000}
    selected_paths = [p for p in snap_paths if iter_from_name(p) in selected_iters]
    if selected_paths:
        sharp_triptych = save_sharp_triptych(run_dir, selected_paths,
                                             x_interp, y_interp, Xphys, Yphys)
        density_png, centers_csv = save_sharp_center_density(run_dir, selected_paths,
                                                             x_interp, y_interp,
                                                             Xphys, Yphys)
        print(sharp_triptych)
        print(density_png)
        print(centers_csv)


if __name__ == '__main__':
    main()
