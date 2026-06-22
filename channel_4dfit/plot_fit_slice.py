#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from fit_one_flowthrough_4d_gaussian import SharedGaussian4D, evaluate_model, load_window


def build_index_maps(data_dir: Path):
    x = np.load(data_dir / 'x.npy')
    y = np.load(data_dir / 'y.npy')
    z = np.load(data_dir / 'z.npy')

    x_unique = np.unique(np.round(x, 8))
    y_unique = np.unique(np.round(y, 8))
    z_unique = np.unique(np.round(z, 8))

    ix = np.searchsorted(x_unique, np.round(x, 8))
    iy = np.searchsorted(y_unique, np.round(y, 8))
    iz = np.searchsorted(z_unique, np.round(z, 8))
    return x_unique.astype(np.float32), y_unique.astype(np.float32), z_unique.astype(np.float32), ix, iy, iz


def flat_to_3d(field_flat: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
               nx: int, ny: int, nz: int):
    arr3 = np.empty((nz, ny, nx), dtype=np.float32)
    arr3[iz, iy, ix] = np.asarray(field_flat, dtype=np.float32)
    return arr3


def axis_extent(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 1:
        delta = 0.5
    else:
        delta = float(np.median(np.diff(values))) * 0.5
    return [float(values[0] - delta), float(values[-1] + delta)]


def periodic_distance(values: np.ndarray, center: float):
    return np.remainder(values - center + 0.5, 1.0) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, default='model_latest.pt')
    ap.add_argument('--frame_idx', type=int, default=18)
    ap.add_argument('--z_idx', type=int, default=None)
    ap.add_argument('--eval_batch_size', type=int, default=4096)
    ap.add_argument('--output', type=str, default='xy_slice_compare.png')
    ap.add_argument('--overlay_kernels', action='store_true')
    ap.add_argument('--kernel_top_k', type=int, default=512)
    ap.add_argument('--kernel_alpha', type=float, default=0.55)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')

    with open(run_dir / 'args.txt', 'r') as f:
        run_args = json.load(f)

    data_dir = Path(run_args['data_dir'])
    coords_np, targets_np, _, t_win = load_window(
        data_dir,
        int(run_args['start_idx']),
        int(run_args['n_frames']),
    )

    x_unique, y_unique, z_unique, ix, iy, iz = build_index_maps(data_dir)
    nx = x_unique.size
    ny = y_unique.size
    nz = z_unique.size
    n_cells = nx * ny * nz

    frame_idx = int(args.frame_idx)
    if frame_idx < 0 or frame_idx >= int(run_args['n_frames']):
        raise ValueError(f'frame_idx={frame_idx} out of range for n_frames={run_args["n_frames"]}')

    z_idx = nz // 2 if args.z_idx is None else int(args.z_idx)
    if z_idx < 0 or z_idx >= nz:
        raise ValueError(f'z_idx={z_idx} out of range for nz={nz}')

    frame_coords = coords_np.reshape(int(run_args['n_frames']), n_cells, 4)[frame_idx]
    frame_truth = targets_np.reshape(int(run_args['n_frames']), n_cells, 4)[frame_idx]

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    means = np.asarray(checkpoint['means'], dtype=np.float32)
    stds = np.asarray(checkpoint['stds'], dtype=np.float32)

    model = SharedGaussian4D(
        n_atoms=int(run_args['n_atoms']),
        sigma_min=float(run_args['sigma_min']),
        sigma_max=float(run_args['sigma_max']),
        coarse_fraction=float(run_args['coarse_fraction']),
        coarse_scale=float(run_args['coarse_scale']),
        fine_scale=float(run_args['fine_scale']),
        shear_max=float(run_args['shear_max']),
        fourier_harmonics=int(run_args['fourier_harmonics']),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        mu_x, mu_y, mu_z, mu_t, _, _, _, _ = model.transformed()
        mu_x = mu_x.detach().cpu().numpy()
        mu_y = mu_y.detach().cpu().numpy()
        mu_z = mu_z.detach().cpu().numpy()
        mu_t = mu_t.detach().cpu().numpy()

    coords_t = torch.as_tensor(frame_coords, device=device)
    pred_std = evaluate_model(model, coords_t, args.eval_batch_size)
    pred = pred_std * stds[None, :] + means[None, :]

    names = ['u', 'v', 'w', 'p']
    truth_3d = [flat_to_3d(frame_truth[:, i], ix, iy, iz, nx, ny, nz) for i in range(4)]
    pred_3d = [flat_to_3d(pred[:, i], ix, iy, iz, nx, ny, nz) for i in range(4)]

    x_extent = axis_extent(x_unique)
    y_extent = axis_extent(y_unique)
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]
    fig = plt.figure(figsize=(20, 11), constrained_layout=True)
    gs = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.75])
    axes = np.empty((3, 4), dtype=object)
    for row in range(3):
        for col in range(4):
            axes[row, col] = fig.add_subplot(gs[row, col])
    density_axes = [fig.add_subplot(gs[row, 4]) for row in range(3)]

    slice_t_norm = float(frame_coords[0, 3])
    slice_z_norm = float(frame_coords.reshape(n_cells, 4)[iz == z_idx][0, 2])
    kernel_weight = np.exp(-0.5 * ((periodic_distance(mu_z, slice_z_norm) / 0.08) ** 2 + ((mu_t - slice_t_norm) / 0.08) ** 2))
    kernel_order = np.argsort(kernel_weight)[::-1]
    kernel_keep = kernel_order[:min(args.kernel_top_k, kernel_order.size)]
    kernel_keep = kernel_keep[kernel_weight[kernel_keep] > 1e-4]

    y_centers = np.asarray(mu_y)
    wall_dist = np.minimum(np.abs(y_centers - 0.0), np.abs(y_centers - 1.0))
    wall_dist_sel = np.minimum(np.abs(y_centers[kernel_keep] - 0.0), np.abs(y_centers[kernel_keep] - 1.0)) if kernel_keep.size else np.array([], dtype=np.float32)

    for i, name in enumerate(names):
        truth_slice = truth_3d[i][z_idx]
        pred_slice = pred_3d[i][z_idx]
        err_slice = pred_slice - truth_slice
        slice_rel = np.linalg.norm(err_slice) / np.linalg.norm(truth_slice)

        vmin = min(float(truth_slice.min()), float(pred_slice.min()))
        vmax = max(float(truth_slice.max()), float(pred_slice.max()))
        err_lim = float(np.max(np.abs(err_slice)))

        im0 = axes[0, i].imshow(
            truth_slice,
            origin='lower',
            extent=extent,
            interpolation='bicubic',
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
        )
        im1 = axes[1, i].imshow(
            pred_slice,
            origin='lower',
            extent=extent,
            interpolation='bicubic',
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
        )
        im2 = axes[2, i].imshow(
            err_slice,
            origin='lower',
            extent=extent,
            interpolation='bicubic',
            cmap='RdBu_r',
            vmin=-err_lim,
            vmax=err_lim,
            aspect='equal',
        )

        axes[0, i].set_title(f'{name} truth')
        axes[1, i].set_title(f'{name} fit')
        axes[2, i].set_title(f'{name} err, rel={slice_rel:.3e}')

        if args.overlay_kernels and kernel_keep.size:
            xk = x_extent[0] + (x_extent[1] - x_extent[0]) * mu_x[kernel_keep]
            yk = y_extent[0] + (y_extent[1] - y_extent[0]) * mu_y[kernel_keep]
            size = 10.0 + 60.0 * (kernel_weight[kernel_keep] / kernel_weight[kernel_keep].max())
            for row in range(3):
                axes[row, i].scatter(
                    xk,
                    yk,
                    s=size,
                    c=kernel_weight[kernel_keep],
                    cmap='Greys',
                    alpha=args.kernel_alpha,
                    linewidths=0.0,
                )

        fig.colorbar(im0, ax=axes[0, i], shrink=0.82)
        fig.colorbar(im1, ax=axes[1, i], shrink=0.82)
        fig.colorbar(im2, ax=axes[2, i], shrink=0.82)

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

    bins = np.linspace(0.0, 1.0, 31)
    density_axes[0].hist(y_centers, bins=bins, color='tab:blue', alpha=0.85)
    density_axes[0].set_title('all kernel y')
    density_axes[1].hist(mu_y[kernel_keep], bins=bins, color='tab:orange', alpha=0.85)
    density_axes[1].set_title('slice-relevant y')
    wd_bins = np.linspace(0.0, 0.5, 31)
    density_axes[2].hist(wall_dist, bins=wd_bins, color='tab:green', alpha=0.45, label='all')
    if wall_dist_sel.size:
        density_axes[2].hist(wall_dist_sel, bins=wd_bins, color='tab:red', alpha=0.75, label='slice-relevant')
    density_axes[2].set_title('distance to wall')
    density_axes[2].legend(fontsize=8)
    for ax in density_axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylabel('count')
    density_axes[0].set_xlabel('y')
    density_axes[1].set_xlabel('y')
    density_axes[2].set_xlabel('min(y, 1-y)')

    fig.suptitle(
        f'x-y slice at fixed z index {z_idx} (z={z_unique[z_idx]:.4f}), '
        f'frame_idx={frame_idx}, t={t_win[frame_idx]:.4f}',
        fontsize=14,
    )
    out_path = run_dir / args.output
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(out_path)


if __name__ == '__main__':
    main()