#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class DomainInfo:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    lx: float
    ly: float
    lz: float
    t_min: float
    t_max: float
    lt: float


def save_source_snapshot(run_dir: Path):
    src_dir = run_dir / 'source_snapshot'
    src_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    for name in ['fit_one_flowthrough_4d_gaussian.py']:
        src = here / name
        if src.is_file():
            shutil.copy2(src, src_dir / name)


def infer_period_length(unique_vals: np.ndarray) -> float:
    diffs = np.diff(unique_vals)
    step = float(np.median(diffs))
    return step * len(unique_vals)


def load_window(data_dir: Path, start_idx: int, n_frames: int):
    x = np.load(data_dir / 'x.npy')
    y = np.load(data_dir / 'y.npy')
    z = np.load(data_dir / 'z.npy')
    times = np.load(data_dir / 'times.npy')

    stop_idx = start_idx + n_frames
    t_win = times[start_idx:stop_idx].astype(np.float32)
    fields = []
    for name in ['U_x.npy', 'U_y.npy', 'U_z.npy', 'p.npy']:
        arr = np.load(data_dir / name, mmap_mode='r')[start_idx:stop_idx]
        fields.append(np.asarray(arr, dtype=np.float32))

    x_unique = np.unique(np.round(x, 8))
    y_unique = np.unique(np.round(y, 8))
    z_unique = np.unique(np.round(z, 8))

    domain = DomainInfo(
        x_min=float(x_unique[0]),
        x_max=float(x_unique[-1]),
        y_min=float(y_unique[0]),
        y_max=float(y_unique[-1]),
        z_min=float(z_unique[0]),
        z_max=float(z_unique[-1]),
        lx=infer_period_length(x_unique),
        ly=float(y_unique[-1] - y_unique[0]),
        lz=infer_period_length(z_unique),
        t_min=float(t_win[0]),
        t_max=float(t_win[-1]),
        lt=max(float(t_win[-1] - t_win[0]), 1.0),
    )

    x_norm = ((x - domain.x_min) / domain.lx).astype(np.float32)
    y_norm = ((y - domain.y_min) / max(domain.ly, 1e-12)).astype(np.float32)
    z_norm = ((z - domain.z_min) / domain.lz).astype(np.float32)
    tau_norm = ((t_win - t_win[0]) / domain.lt).astype(np.float32)

    n_cells = x.shape[0]
    n_total = n_frames * n_cells
    coords = np.empty((n_total, 4), dtype=np.float32)
    coords[:, 0] = np.tile(x_norm, n_frames)
    coords[:, 1] = np.tile(y_norm, n_frames)
    coords[:, 2] = np.tile(z_norm, n_frames)
    coords[:, 3] = np.repeat(tau_norm, n_cells)

    targets = np.stack(fields, axis=-1).reshape(n_total, 4)
    return coords, targets, domain, t_win


class SharedGaussian4D(torch.nn.Module):
    def __init__(self, n_atoms: int, sigma_min: float, sigma_max: float,
                 coarse_fraction: float, coarse_scale: float,
                 fine_scale: float, shear_max: float,
                 fourier_harmonics: int):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.coarse_fraction = float(coarse_fraction)
        self.coarse_scale = float(coarse_scale)
        self.fine_scale = float(fine_scale)
        self.shear_max = float(shear_max)
        self.fourier_harmonics = int(fourier_harmonics)

        centers = self._make_initial_centers(n_atoms)
        self.raw_mu_x = torch.nn.Parameter(centers[:, 0].clone())
        self.raw_mu_y = torch.nn.Parameter(centers[:, 1].clone())
        self.raw_mu_z = torch.nn.Parameter(centers[:, 2].clone())
        self.raw_mu_t = torch.nn.Parameter(centers[:, 3].clone())

        init_sigmas = self._make_initial_sigmas(n_atoms)
        self.raw_sigma_x = torch.nn.Parameter(init_sigmas[:, 0].clone())
        self.raw_sigma_y = torch.nn.Parameter(init_sigmas[:, 1].clone())
        self.raw_sigma_z = torch.nn.Parameter(init_sigmas[:, 2].clone())
        self.raw_sigma_t = torch.nn.Parameter(init_sigmas[:, 3].clone())
        self.raw_l21 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))
        self.raw_l31 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))
        self.raw_l32 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))
        self.raw_l41 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))
        self.raw_l42 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))
        self.raw_l43 = torch.nn.Parameter(torch.zeros(n_atoms, device=device))

        self.coeff = torch.nn.Parameter(torch.zeros(n_atoms, 4, device=device))
        self.coeff_fourier = torch.nn.Parameter(
            torch.zeros(self.fourier_dim(), 4, device=device)
        )
        self.bias = torch.nn.Parameter(torch.zeros(4, device=device))

    def fourier_dim(self) -> int:
        return 5 + 6 * self.fourier_harmonics

    @staticmethod
    def _make_initial_centers(n_atoms: int) -> torch.Tensor:
        n_side = max(2, math.ceil(n_atoms ** 0.25))
        one_d = torch.linspace(0.0, 1.0, n_side, device=device)
        mesh = torch.stack(torch.meshgrid(one_d, one_d, one_d, one_d, indexing='ij'), dim=-1)
        centers = mesh.reshape(-1, 4)
        if centers.shape[0] < n_atoms:
            reps = math.ceil(n_atoms / centers.shape[0])
            centers = centers.repeat(reps, 1)
        return centers[:n_atoms]

    def _sigma_to_raw(self, sigma: float) -> float:
        sigma = max(float(sigma), 1e-6)
        return math.log(math.expm1(sigma))

    def _make_initial_sigmas(self, n_atoms: int) -> torch.Tensor:
        coarse_n = int(round(self.coarse_fraction * n_atoms))
        coarse_n = max(1, min(n_atoms - 1, coarse_n)) if n_atoms > 1 else 1
        fine_n = n_atoms - coarse_n
        base_sigma = 0.10
        coarse_sigma = float(max(base_sigma * self.coarse_scale, 1e-6))
        fine_sigma = float(max(base_sigma * self.fine_scale, 1e-6))
        coarse_raw = self._sigma_to_raw(coarse_sigma)
        fine_raw = self._sigma_to_raw(fine_sigma)
        raw = torch.empty((n_atoms, 4), device=device)
        raw[:coarse_n] = coarse_raw
        raw[coarse_n:] = fine_raw
        if fine_n == 0:
            raw[:] = coarse_raw
        return raw

    def transformed(self):
        mu_x = torch.remainder(self.raw_mu_x, 1.0)
        mu_y = self.raw_mu_y
        mu_z = torch.remainder(self.raw_mu_z, 1.0)
        mu_t = self.raw_mu_t

        sig_x = torch.nn.functional.softplus(self.raw_sigma_x) + 1e-6
        sig_y = torch.nn.functional.softplus(self.raw_sigma_y) + 1e-6
        sig_z = torch.nn.functional.softplus(self.raw_sigma_z) + 1e-6
        sig_t = torch.nn.functional.softplus(self.raw_sigma_t) + 1e-6
        return mu_x, mu_y, mu_z, mu_t, sig_x, sig_y, sig_z, sig_t

    def precision_factors(self):
        _, _, _, _, sig_x, sig_y, sig_z, sig_t = self.transformed()
        inv_x = 1.0 / sig_x
        inv_y = 1.0 / sig_y
        inv_z = 1.0 / sig_z
        inv_t = 1.0 / sig_t
        lower = torch.zeros(self.n_atoms, 4, 4, device=device)
        lower[:, 0, 0] = inv_x
        lower[:, 1, 0] = self.shear_max * torch.tanh(self.raw_l21) * inv_x
        lower[:, 1, 1] = inv_y
        lower[:, 2, 0] = self.shear_max * torch.tanh(self.raw_l31) * inv_x
        lower[:, 2, 1] = self.shear_max * torch.tanh(self.raw_l32) * inv_y
        lower[:, 2, 2] = inv_z
        lower[:, 3, 0] = self.shear_max * torch.tanh(self.raw_l41) * inv_x
        lower[:, 3, 1] = self.shear_max * torch.tanh(self.raw_l42) * inv_y
        lower[:, 3, 2] = self.shear_max * torch.tanh(self.raw_l43) * inv_z
        lower[:, 3, 3] = inv_t
        return lower

    def basis(self, coords: torch.Tensor):
        mu_x, mu_y, mu_z, mu_t, _, _, _, _ = self.transformed()
        lower = self.precision_factors()

        dx = torch.remainder(coords[:, None, 0] - mu_x[None, :] + 0.5, 1.0) - 0.5
        dy = coords[:, None, 1] - mu_y[None, :]
        dz = torch.remainder(coords[:, None, 2] - mu_z[None, :] + 0.5, 1.0) - 0.5
        dt = coords[:, None, 3] - mu_t[None, :]
        tx = dx * lower[None, :, 0, 0]
        ty = dx * lower[None, :, 1, 0] + dy * lower[None, :, 1, 1]
        tz = (
            dx * lower[None, :, 2, 0]
            + dy * lower[None, :, 2, 1]
            + dz * lower[None, :, 2, 2]
        )
        tt = (
            dx * lower[None, :, 3, 0]
            + dy * lower[None, :, 3, 1]
            + dz * lower[None, :, 3, 2]
            + dt * lower[None, :, 3, 3]
        )
        r2 = tx * tx + ty * ty + tz * tz + tt * tt
        return torch.exp(-0.5 * r2)

    def fourier_basis(self, coords: torch.Tensor):
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]
        t = coords[:, 3:4]
        y_c = 2.0 * y - 1.0
        t_c = 2.0 * t - 1.0

        feats = [
            y_c,
            y_c * y_c,
            t_c,
            t_c * t_c,
            y_c * t_c,
        ]
        for k in range(1, self.fourier_harmonics + 1):
            wk = 2.0 * math.pi * float(k)
            feats.append(torch.sin(wk * x))
            feats.append(torch.cos(wk * x))
            feats.append(torch.sin(wk * z))
            feats.append(torch.cos(wk * z))
            feats.append(torch.sin(wk * t))
            feats.append(torch.cos(wk * t))
        return torch.cat(feats, dim=1)

    def forward(self, coords: torch.Tensor):
        phi = self.basis(coords)
        psi = self.fourier_basis(coords)
        return phi @ self.coeff + psi @ self.coeff_fourier + self.bias


def train_model(model: SharedGaussian4D, coords_train: torch.Tensor,
                targets_train: torch.Tensor, adam_iters: int,
                batch_size: int, adam_lr: float,
                log_every: int, run_dir: Path,
                means: np.ndarray, stds: np.ndarray,
                monitor_coords: torch.Tensor | None,
                monitor_targets_raw: np.ndarray | None,
                monitor_batch_size: int,
                checkpoint_every: int):
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    history = {'train': []}
    n_train = coords_train.shape[0]

    for it in range(int(adam_iters)):
        idx = torch.randint(0, n_train, (batch_size,), device=coords_train.device)
        pred = model(coords_train[idx])
        loss = torch.mean((pred - targets_train[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % log_every == 0 or it == adam_iters - 1:
            history['train'].append((it, float(loss.detach().cpu())))
            msg = f'iter={it} train_mse={history["train"][-1][1]:.6e}'
            if monitor_coords is not None and monitor_targets_raw is not None:
                rel = compute_rel_metrics(
                    model,
                    monitor_coords,
                    monitor_targets_raw,
                    means,
                    stds,
                    monitor_batch_size,
                )
                msg += (
                    f' u_rel={rel["u"]:.6e}'
                    f' v_rel={rel["v"]:.6e}'
                    f' w_rel={rel["w"]:.6e}'
                    f' p_rel={rel["p"]:.6e}'
                )
            print(msg)

        if checkpoint_every > 0 and ((it + 1) % checkpoint_every == 0 or it == adam_iters - 1):
            torch.save(
                {
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'history': history,
                    'means': means,
                    'stds': stds,
                },
                run_dir / 'model_latest.pt'
            )
    return history


def evaluate_model(model: SharedGaussian4D, coords: torch.Tensor,
                   batch_size: int):
    cur_batch = int(batch_size)
    while True:
        try:
            out = []
            with torch.no_grad():
                for start in range(0, coords.shape[0], cur_batch):
                    out.append(model(coords[start:start + cur_batch]).cpu())
            return torch.cat(out, dim=0).numpy()
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower() or cur_batch <= 1024:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cur_batch = max(1024, cur_batch // 2)
            print(f'eval_oom_retry batch_size={cur_batch}')


def compute_rel_metrics(model: SharedGaussian4D,
                        coords: torch.Tensor,
                        targets_raw: np.ndarray,
                        means: np.ndarray,
                        stds: np.ndarray,
                        batch_size: int):
    pred_std = evaluate_model(model, coords, batch_size)
    pred = pred_std * stds[None, :] + means[None, :]
    names = ['u', 'v', 'w', 'p']
    metrics = {}
    for idx, name in enumerate(names):
        truth = targets_raw[:, idx]
        approx = pred[:, idx]
        metrics[name] = float(np.linalg.norm(approx - truth) / np.linalg.norm(truth))
    return metrics


def make_plots(run_dir: Path, history, errors_by_field: dict[str, dict[str, float]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    train_it = [x for x, _ in history['train']]
    train_val = [y for _, y in history['train']]
    ax.semilogy(train_it, train_val, label='train_mse')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'loss_history.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    names = list(errors_by_field.keys())
    rel_l2 = [errors_by_field[name]['rel_l2'] for name in names]
    max_abs = [errors_by_field[name]['max_abs'] for name in names]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rel_l2, width=width, label='rel_l2')
    ax.bar(x + width / 2, max_abs, width=width, label='max_abs')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'field_errors.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_train_pool(coords_np: np.ndarray, train_pool_size: int):
    n_total = coords_np.shape[0]
    pool_n = n_total if train_pool_size < 0 else min(train_pool_size, n_total)
    perm = np.random.permutation(n_total)
    train_idx = perm[:pool_n]
    return train_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=36)
    ap.add_argument('--n_atoms', type=int, default=2048)
    ap.add_argument('--train_samples', type=int, default=-1)
    ap.add_argument('--adam_iters', type=int, default=4000)
    ap.add_argument('--batch_size', type=int, default=16384)
    ap.add_argument('--adam_lr', type=float, default=3e-3)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--checkpoint_every', type=int, default=1000)
    ap.add_argument('--monitor_samples', type=int, default=131072)
    ap.add_argument('--monitor_batch_size', type=int, default=8192)
    ap.add_argument('--fourier_harmonics', type=int, default=6)
    ap.add_argument('--sigma_min', type=float, default=0.04)
    ap.add_argument('--sigma_max', type=float, default=0.35)
    ap.add_argument('--coarse_fraction', type=float, default=0.5)
    ap.add_argument('--coarse_scale', type=float, default=1.5)
    ap.add_argument('--fine_scale', type=float, default=0.5)
    ap.add_argument('--shear_max', type=float, default=1.0)
    ap.add_argument('--eval_batch_size', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', type=str, default='output')
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tag = f'_{args.tag}' if args.tag else ''
    run_dir = Path(args.out_dir) / f'fit4d_{stamp}{tag}'
    run_dir.mkdir(parents=True, exist_ok=True)
    save_source_snapshot(run_dir)
    with open(run_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    coords_np, targets_np, domain, t_win = load_window(Path(args.data_dir), args.start_idx, args.n_frames)
    means = targets_np.mean(axis=0)
    stds = targets_np.std(axis=0)
    stds = np.maximum(stds, 1e-6)
    targets_std = (targets_np - means[None, :]) / stds[None, :]

    n_total = coords_np.shape[0]
    train_idx = make_train_pool(coords_np, args.train_samples)

    coords_train = torch.as_tensor(coords_np[train_idx], device=device)
    targets_train = torch.as_tensor(targets_std[train_idx], device=device)
    if args.monitor_samples > 0:
        monitor_n = min(args.monitor_samples, n_total)
        monitor_idx = np.random.permutation(n_total)[:monitor_n]
        monitor_coords = torch.as_tensor(coords_np[monitor_idx], device=device)
        monitor_targets_raw = targets_np[monitor_idx]
    else:
        monitor_coords = None
        monitor_targets_raw = None

    model = SharedGaussian4D(
        n_atoms=args.n_atoms,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        coarse_fraction=args.coarse_fraction,
        coarse_scale=args.coarse_scale,
        fine_scale=args.fine_scale,
        shear_max=args.shear_max,
        fourier_harmonics=args.fourier_harmonics,
    ).to(device)

    history = train_model(
        model,
        coords_train,
        targets_train,
        adam_iters=args.adam_iters,
        batch_size=args.batch_size,
        adam_lr=args.adam_lr,
        log_every=args.log_every,
        run_dir=run_dir,
        means=means,
        stds=stds,
        monitor_coords=monitor_coords,
        monitor_targets_raw=monitor_targets_raw,
        monitor_batch_size=args.monitor_batch_size,
        checkpoint_every=args.checkpoint_every,
    )

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'means': means,
            'stds': stds,
            'args': vars(args),
        },
        run_dir / 'model_final.pt'
    )

    coords = torch.as_tensor(coords_np, device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pred_std = evaluate_model(model, coords, args.eval_batch_size)
    pred = pred_std * stds[None, :] + means[None, :]

    names = ['u', 'v', 'w', 'p']
    errors_by_field = {}
    for idx, name in enumerate(names):
        truth = targets_np[:, idx]
        approx = pred[:, idx]
        rel_l2 = float(np.linalg.norm(approx - truth) / np.linalg.norm(truth))
        rmse = float(np.sqrt(np.mean((approx - truth) ** 2)))
        max_abs = float(np.max(np.abs(approx - truth)))
        errors_by_field[name] = {
            'rel_l2': rel_l2,
            'rmse': rmse,
            'max_abs': max_abs,
        }

    make_plots(run_dir, history, errors_by_field)

    summary = {
        'window_start_time': float(t_win[0]),
        'window_end_time': float(t_win[-1]),
        'window_n_frames': int(args.n_frames),
        'n_total_samples': int(n_total),
        'n_train_samples': int(train_idx.size),
        'n_val_samples': 0,
        'n_atoms': int(args.n_atoms),
        'device': str(device),
        'wall_seconds': time.time() - t0,
        'errors': errors_by_field,
        'domain': domain.__dict__,
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / 'summary.txt', 'w') as f:
        f.write(f'window = [{t_win[0]}, {t_win[-1]}], n_frames = {args.n_frames}\n')
        f.write(f'n_atoms = {args.n_atoms}\n')
        f.write(f'n_total_samples = {n_total}\n')
        f.write(f'n_train_samples = {train_idx.size}\n')
        f.write('n_val_samples = 0\n')
        for name in names:
            vals = errors_by_field[name]
            f.write(f'{name}_rel_l2 = {vals["rel_l2"]:.6e}\n')
            f.write(f'{name}_rmse = {vals["rmse"]:.6e}\n')
            f.write(f'{name}_max_abs = {vals["max_abs"]:.6e}\n')
        f.write(f'wall_seconds = {summary["wall_seconds"]:.2f}\n')

    print(f'run_dir = {run_dir}')
    for name in names:
        vals = errors_by_field[name]
        print(f'{name}_rel_l2 = {vals["rel_l2"]:.6e}')
        print(f'{name}_rmse = {vals["rmse"]:.6e}')
        print(f'{name}_max_abs = {vals["max_abs"]:.6e}')
    print(f'wall_seconds = {summary["wall_seconds"]:.2f}')


if __name__ == '__main__':
    main()
