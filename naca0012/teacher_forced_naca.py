"""Phase-3 step 2 (Path A): teacher-forced VarPro manifold tracking.

The teacher (curvilinear char-WENO5 + LU-SGS, numpy) integrates the NACA0012
steady Euler problem in (xi, eta) from freestream toward steady state.  At a
set of checkpoint iterations we PROJECT the current grid state U onto the
proven 344-atom double Gaussian manifold (BlendedRBF, Kaufman VarPro) and
record how well the manifold represents each state along the WHOLE
convergence trajectory (transient shock formation + motion, not just the
endpoint).

Two modes:
  * open-loop  (default): teacher runs freely; projection is a passive
    measurement.  Answers "can the manifold carry every state on the path?".
  * closed-loop (--closed_loop): after each projection the teacher state is
    OVERWRITTEN by the reconstruction, then stepping continues.  Answers
    "is the projection feedback stable / does the shock drift?".

NOTE: this is NOT EDNN solving the PDE.  The driving force is the teacher;
the manifold only tracks.  The endpoint quality is bounded by the teacher.
True PDE-residual EDNN (metric in autograd RHS) is Path B, a later step.

Basis + fitter reused verbatim from case_2d_dmr/fit_weno_snap_varpro.py.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import csv
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from fit_weno_snap_varpro import (   # noqa: E402
    BlendedRBF, fit_varpro, fit_varpro_adam,
    grad_rho_topk, lattice_centres, metrics_full,
    device, FIELD_NAMES,
)

# --- teacher ----------------------------------------------------------
from weno_curv import CurvilinearMesh           # noqa: E402
from weno_curv_char import CurvilinearMeshChar  # noqa: E402
from lusgs_curv import lusgs_step               # noqa: E402

GAMMA = 1.4
Y_SLICE = 0.14
ZAHR_CSV = os.path.join(os.path.dirname(__file__), 'ref_data',
                        'zahr_naca_M085_slice_y014.csv')

_MODEL_KEYS = (
    'mu_Sx', 'mu_Sy', 'mu_Nx', 'mu_Ny', 'log_sig_S', 'log_sig_S_perp',
    'theta_S', 'log_sig_N', 'log_sig_N_perp', 'theta_N', 'c'
)


def save_source_snapshot(run_dir):
    src_dir = os.path.join(run_dir, 'source_snapshot')
    os.makedirs(src_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in [
            'teacher_forced_naca.py',
            'plot_teacher_gaussians.py',
            'plot_summary.py',
            'plot_ref_slice.py',
            'dmr_setup.py',
            'fit_weno_snap_varpro.py',
            'weno_curv.py',
            'weno_curv_char.py',
            'lusgs_curv.py',
            'naca_setup.py',
            'mapping_f2.npz']:
        src = os.path.join(here, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(src_dir, name))
    ref_dir = os.path.join(here, 'ref_data')
    if os.path.isdir(ref_dir):
        dst_ref = os.path.join(src_dir, 'ref_data')
        os.makedirs(dst_ref, exist_ok=True)
        for name in os.listdir(ref_dir):
            src = os.path.join(ref_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_ref, name))


def write_args_txt(out_dir, args, ckpts, sS_box, sN_box):
    with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
        for key in sorted(vars(args)):
            f.write(f'{key} = {getattr(args, key)}\n')
        f.write(f'checkpoint_list = {",".join(str(v) for v in ckpts)}\n')
        f.write(f'sigma_S_box_parsed = {sS_box}\n')
        f.write(f'sigma_N_box_parsed = {sN_box}\n')
        f.write(f'zahr_csv = {ZAHR_CSV}\n')


def mach_from_U(U):
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    E = U[..., 3]
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-30))
    return np.sqrt(u * u + v * v) / a


def prim_from_U(U):
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = (GAMMA - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
    a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-30))
    M = np.sqrt(u * u + v * v) / a
    return rho, M, p


def slice_field(X, Y, F, xline):
    pts = np.column_stack([X.ravel(), Y.ravel()])
    qpts = np.column_stack([xline, np.full_like(xline, Y_SLICE)])
    return griddata(pts, F.ravel(), qpts, method='linear')


def save_zahr_compare(saved, Xphys, Yphys, out_dir):
    if not saved:
        return None

    last_it = max(saved.keys())
    U_teacher, U_proj = saved[last_it]
    rho_t, M_t, p_t = prim_from_U(U_teacher)
    rho_p, M_p, p_p = prim_from_U(U_proj)

    z = np.loadtxt(ZAHR_CSV, delimiter=',', skiprows=1)
    zx, zr, zM, zp = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    xline = np.linspace(zx.min(), zx.max(), 400)

    # Ours uses rho_inf=1, P_inf=1/gamma, while Zahr uses rho_inf=gamma, P_inf=1.
    rho_tl = slice_field(Xphys, Yphys, rho_t, xline) * GAMMA
    rho_pl = slice_field(Xphys, Yphys, rho_p, xline) * GAMMA
    M_tl = slice_field(Xphys, Yphys, M_t, xline)
    M_pl = slice_field(Xphys, Yphys, M_p, xline)
    p_tl = slice_field(Xphys, Yphys, p_t, xline) * GAMMA
    p_pl = slice_field(Xphys, Yphys, p_p, xline) * GAMMA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    series = [
        (zr, rho_tl, rho_pl, r'density $\rho$'),
        (zM, M_tl, M_pl, 'Mach M'),
        (zp, p_tl, p_pl, 'pressure P'),
    ]
    for ax, (zahr_v, teacher_v, proj_v, title) in zip(axes, series):
        ax.plot(zx, zahr_v, 'k-', lw=1.6, label='Zahr DG-tracking')
        ax.plot(xline, teacher_v, 'b--', lw=1.3, label='teacher')
        ax.plot(xline, proj_v, 'r-.', lw=1.4, label='projected')
        ax.set_xlim(-0.5, 1.5)
        ax.set_xlabel('x')
        ax.set_title(title + rf'  along $y={Y_SLICE:.2f}$')
        ax.grid(True, alpha=0.3)
    axes[1].axhline(1.0, color='gray', lw=0.5, ls='--')
    axes[0].legend(loc='best', fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(out_dir, 'vs_zahr_y014.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_zahr_compare_from_tracks(run_dir):
    data = np.load(os.path.join(run_dir, 'tracks.npz'))
    Xphys = data['Xphys']
    Yphys = data['Yphys']
    saved = {}
    for key in data.files:
        if key.startswith('U_teacher_'):
            it = int(key.split('_')[-1])
            saved.setdefault(it, [None, None])[0] = data[key]
        elif key.startswith('U_proj_'):
            it = int(key.split('_')[-1])
            saved.setdefault(it, [None, None])[1] = data[key]
    saved = {it: (vals[0], vals[1]) for it, vals in saved.items()
             if vals[0] is not None and vals[1] is not None}
    return save_zahr_compare(saved, Xphys, Yphys, run_dir)


def build_model(x, y, U_grid, XI, ETA, args, sS_box, sN_box, Nxi, Neta):
    """Construct a fresh BlendedRBF seeded on the CURRENT field."""
    _, _, K_s_used, muSx, muSy = lattice_centres(args.K_s, 1.0, 1.0)
    muNx, muNy = grad_rho_topk(U_grid, XI, ETA, args.K_n)
    pitch_smooth = 1.0 / max(int(round(math.sqrt(K_s_used))), 1)
    pitch_grid = max(1.0 / (Nxi - 1), 1.0 / (Neta - 1))
    log_sig_S_init = math.log(args.sigma_S_init_factor * pitch_smooth)
    log_sig_N_init = math.log(args.sigma_N_init_factor * pitch_grid)
    model = BlendedRBF(
        x, y, mu_S=(muSx, muSy), mu_N=(muNx, muNy),
        log_sig_S_init=log_sig_S_init, log_sig_N_init=log_sig_N_init,
        sigma_S_box=sS_box, sigma_N_box=sN_box,
        anisotropic=args.sharp_anisotropic,
        smooth_anisotropic=args.smooth_anisotropic,
        mu_extend_frac=args.mu_extend_frac,
    )
    ext = float(args.mu_extend_frac)
    model.mu_box = (-ext, 1.0 + ext, -ext, 1.0 + ext)
    model.project_()
    return model


def save_model_snapshot(model, scale, out_dir, it):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        k: getattr(model, k).detach().cpu().numpy()
        for k in _MODEL_KEYS
    }
    payload['scale'] = np.asarray(scale, dtype=np.float64)
    path = os.path.join(out_dir, f'model_iter{it:05d}.npz')
    np.savez(path, **payload)
    return path


def project(model, U_grid, args, adam_iters, lm_iters, desc):
    """Project grid state U onto the manifold; return (U_pred_grid, rL2,
    rLi, rL1).  model is updated in place (warm-startable)."""
    Nxi, Neta, _ = U_grid.shape
    U_true_np = U_grid.reshape(-1, 4)
    field_rms = np.sqrt((U_true_np ** 2).mean(axis=0))
    scale = field_rms.copy() if args.normalize else np.ones(4)
    U_fit_np = U_true_np / scale[None, :]
    U_fit = torch.as_tensor(U_fit_np, device=device)

    if adam_iters > 0:
        fit_varpro_adam(
            model, U_fit, U_fit_np, iters=adam_iters,
            lr_mu=args.adam_lr_mu, lr_sig=args.adam_lr_sig,
            lr_theta=args.adam_lr_theta,
            rebirth_every=args.rebirth_every,
            ridge=args.ridge, rcond=args.rcond,
            progress=True, progress_desc=desc + '-Adam')
    if lm_iters > 0:
        fit_varpro(
            model, U_fit, U_fit_np, iters=lm_iters,
            ridge=args.ridge, rcond=args.rcond,
            progress=True, progress_desc=desc + '-LM')

    U_pred_fit, _ = model.solve_linear(U_fit, rcond=args.rcond,
                                       ridge=args.ridge)
    U_pred_np = U_pred_fit.detach().cpu().numpy() * scale[None, :]
    rL2, rLi, rL1 = metrics_full(U_pred_np, U_true_np)
    return U_pred_np.reshape(Nxi, Neta, 4), rL2, rLi, rL1, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', type=str, default='mapping_f2.npz')
    ap.add_argument('--char', action='store_true', default=True)
    ap.add_argument('--no_char', dest='char', action='store_false')
    # --- teacher pseudo-time schedule ---
    ap.add_argument('--max_iter', type=int, default=2000)
    ap.add_argument('--cfl', type=float, default=5.0)
    ap.add_argument('--cfl_init', type=float, default=0.5)
    ap.add_argument('--cfl_ramp_iter', type=int, default=100)
    ap.add_argument('--n_sweeps', type=int, default=5)
    ap.add_argument('--checkpoints', type=str,
                    default='5,25,50,100,200,400,800,1400,2000',
                    help='teacher iters at which to project')
    ap.add_argument('--closed_loop', action='store_true', default=True,
                    help='overwrite teacher U with the reconstruction after '
                         'each projection (A2); default open-loop (A1)')
    ap.add_argument('--open_loop', dest='closed_loop', action='store_false',
                    help='disable projection feedback and keep the teacher in '
                         'open-loop mode')
    # --- projector pools / widths (match the validated static fit) ---
    ap.add_argument('--K_s', type=int, default=144)
    ap.add_argument('--K_n', type=int, default=200)
    ap.add_argument('--sigma_S_box', type=str, default='0.02,2.0')
    ap.add_argument('--sigma_N_box', type=str, default='3e-3,0.15')
    ap.add_argument('--sigma_S_init_factor', type=float, default=2.0)
    ap.add_argument('--sigma_N_init_factor', type=float, default=1.5)
    ap.add_argument('--mu_extend_frac', type=float, default=0.1)
    ap.add_argument('--sharp_anisotropic', action='store_true', default=True)
    ap.add_argument('--smooth_anisotropic', action='store_true', default=True)
    ap.add_argument('--no_sharp_anisotropic', dest='sharp_anisotropic',
                    action='store_false')
    ap.add_argument('--no_smooth_anisotropic', dest='smooth_anisotropic',
                    action='store_false')
    # --- projector optimiser: first (cold) vs subsequent (warm) ---
    ap.add_argument('--adam_iters_first', type=int, default=1200)
    ap.add_argument('--lm_iters_first', type=int, default=200)
    ap.add_argument('--adam_iters_warm', type=int, default=300)
    ap.add_argument('--lm_iters_warm', type=int, default=80)
    ap.add_argument('--reseed_sharp', action='store_true', default=True,
                    help='rebuild a fresh model (re-seed sharp centres on the '
                         'current field) at every checkpoint instead of warm '
                         'starting; use if the shock moves far between ckpts')
    ap.add_argument('--no_reseed_sharp', dest='reseed_sharp',
                    action='store_false')
    ap.add_argument('--adam_lr_mu', type=float, default=1e-3)
    ap.add_argument('--adam_lr_sig', type=float, default=5e-3)
    ap.add_argument('--adam_lr_theta', type=float, default=5e-3)
    ap.add_argument('--rebirth_every', type=int, default=20)
    ap.add_argument('--ridge', type=float, default=1e-6)
    ap.add_argument('--rcond', type=float, default=1e-6)
    ap.add_argument('--normalize', action='store_true', default=True)
    ap.add_argument('--no_normalize', dest='normalize', action='store_false')
    ap.add_argument('--out_dir', type=str, default='output_teacher_forced')
    ap.add_argument('--tag', type=str, default='closed_closed_A2_repro')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sS_box = tuple(float(v) for v in args.sigma_S_box.split(','))
    sN_box = tuple(float(v) for v in args.sigma_N_box.split(','))
    ckpts = sorted(int(v) for v in args.checkpoints.split(','))

    stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tag = ('_' + args.tag) if args.tag else ''
    mode = 'closed' if args.closed_loop else 'open'
    out = os.path.join(args.out_dir, f'tf_{stamp}_{mode}{tag}')
    os.makedirs(out, exist_ok=True)
    write_args_txt(out, args, ckpts, sS_box, sN_box)
    save_source_snapshot(out)

    print('=' * 64)
    print(f'Teacher-forced VarPro manifold tracking  ({mode}-loop)')
    print('=' * 64)
    print(f'mapping        : {args.mapping}  (char={args.char})')
    print(f'teacher        : cfl {args.cfl_init}->{args.cfl} over '
          f'{args.cfl_ramp_iter}, n_sweeps={args.n_sweeps}, '
          f'max_iter={args.max_iter}')
    print(f'checkpoints    : {ckpts}')
    print(f'manifold       : K_s={args.K_s} + K_n={args.K_n} '
          f'(={(args.K_s + args.K_n)} atoms)')
    print(f'sigma_S_box    : {sS_box}   sigma_N_box: {sN_box}')
    print(f'init factors   : S={args.sigma_S_init_factor} '
          f'N={args.sigma_N_init_factor}  mu_extend={args.mu_extend_frac}')
    print(f'anisotropic    : sharp={args.sharp_anisotropic} '
          f'smooth={args.smooth_anisotropic}')
    print(f'projector iters: first(adam/lm)={args.adam_iters_first}/'
          f'{args.lm_iters_first}  warm={args.adam_iters_warm}/'
          f'{args.lm_iters_warm}  reseed_sharp={args.reseed_sharp}')
    print(f'normalize/seed : {args.normalize}/{args.seed}')
    print(f'out            : {out}')
    print('=' * 64)

    # ---- teacher + collocation -----------------------------------------
    MeshCls = CurvilinearMeshChar if args.char else CurvilinearMesh
    cur = MeshCls(args.mapping)
    mp = np.load(args.mapping)
    xi_1d = mp['xi_1d'].astype(np.float64)
    eta_1d = mp['eta_1d'].astype(np.float64)
    XI, ETA = np.meshgrid(xi_1d, eta_1d, indexing='ij')
    Nxi, Neta = XI.shape
    x = torch.as_tensor(XI.reshape(-1), device=device)
    y = torch.as_tensor(ETA.reshape(-1), device=device)
    Xphys, Yphys = mp['X'], mp['Y']

    U = cur.freestream()

    # warm up numba JIT
    print('warming up teacher JIT ...')
    t0 = time.time()
    _ = lusgs_step(cur, U, cfl_implicit=1.0, n_sweeps=1)
    print(f'  JIT + first step: {time.time() - t0:.2f} s')

    csv_f = open(os.path.join(out, 'track.csv'), 'w', newline='')
    cw = csv.writer(csv_f)
    cw.writerow(['iter', 'cfl', 'teacher_rms', 'Mmax_teacher', 'Mmax_proj',
                 'rL2_rho', 'rL2_rhou', 'rL2_rhov', 'rL2_E',
                 'rLinf_rho', 'rL1_rho'])

    model = None
    model_snaps_dir = os.path.join(out, 'model_snaps')
    ckpt_set = set(ckpts)
    saved = {}   # iter -> (U_teacher, U_proj)
    t_start = time.time()

    for it in range(1, args.max_iter + 1):
        if it <= args.cfl_ramp_iter:
            frac = it / max(args.cfl_ramp_iter, 1)
            cfl_now = args.cfl_init + (args.cfl - args.cfl_init) * frac
        else:
            cfl_now = args.cfl

        U_new, rms, mx = lusgs_step(cur, U, cfl_now, n_sweeps=args.n_sweeps)
        if not np.isfinite(rms) or U_new[..., 0].min() < 1e-6:
            print(f'!! iter {it}: teacher diverged (rms={rms})')
            break
        U = U_new

        if it in ckpt_set:
            first = (model is None)
            if first or args.reseed_sharp:
                model = build_model(x, y, U, XI, ETA, args,
                                    sS_box, sN_box, Nxi, Neta)
            a_it = args.adam_iters_first if first else args.adam_iters_warm
            l_it = args.lm_iters_first if first else args.lm_iters_warm
            U_proj, rL2, rLi, rL1, scale = project(
                model, U, args, a_it, l_it, desc=f'it{it}')
            Mt = float(mach_from_U(U).max())
            Mp = float(mach_from_U(U_proj).max())
            model_path = save_model_snapshot(model, scale, model_snaps_dir, it)
            print(f'\n[ckpt {it:5d}] teacher_rms={rms:.3e} '
                  f'Mmax T={Mt:.3f}/P={Mp:.3f}  '
                  f'rL2 rho={rL2[0]:.3e} rhou={rL2[1]:.3e} '
                  f'rhov={rL2[2]:.3e} E={rL2[3]:.3e}  '
                f't={time.time()-t_start:.0f}s  '
                f'model={os.path.basename(model_path)}\n')
            cw.writerow([it, cfl_now, rms, Mt, Mp,
                         rL2[0], rL2[1], rL2[2], rL2[3], rLi[0], rL1[0]])
            csv_f.flush()
            saved[it] = (U.copy(), U_proj.copy())
            if args.closed_loop:
                U = U_proj.copy()   # feed reconstruction back to teacher

    csv_f.close()

    # ---- figures -------------------------------------------------------
    arr = np.loadtxt(os.path.join(out, 'track.csv'), delimiter=',',
                     skiprows=1, ndmin=2)
    if arr.size:
        its = arr[:, 0]
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].semilogy(its, arr[:, 5], 'o-', label=r'$\rho$')
        ax[0].semilogy(its, arr[:, 6], 's-', label=r'$\rho u$')
        ax[0].semilogy(its, arr[:, 7], '^-', label=r'$\rho v$')
        ax[0].semilogy(its, arr[:, 8], 'd-', label='E')
        ax[0].set_xlabel('teacher iter'); ax[0].set_ylabel('projection rL2')
        ax[0].set_title('manifold projection error along trajectory')
        ax[0].grid(True, alpha=0.3, which='both'); ax[0].legend()

        ax[1].plot(its, arr[:, 3], 'k-o', label='teacher Mmax')
        ax[1].plot(its, arr[:, 4], 'r--s', label='projected Mmax')
        ax[1].set_xlabel('teacher iter'); ax[1].set_ylabel('Mmax')
        ax[1].set_title('peak Mach: teacher vs manifold')
        ax[1].grid(True, alpha=0.3); ax[1].legend()

        ax[2].semilogy(its, arr[:, 2], 'b-o')
        ax[2].set_xlabel('teacher iter')
        ax[2].set_ylabel('teacher rms_R_rho_rel')
        ax[2].set_title('teacher convergence')
        ax[2].grid(True, alpha=0.3, which='both')
        fig.suptitle(f'Teacher-forced ({mode}-loop)  '
                     f'{args.K_s + args.K_n} atoms')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'track.png'), dpi=120,
                    bbox_inches='tight')
        print('saved', os.path.join(out, 'track.png'))

    # Mach panels at first / mid / last checkpoint
    keys = sorted(saved.keys())
    if keys:
        sel = [keys[0], keys[len(keys) // 2], keys[-1]]
        fig, ax = plt.subplots(len(sel), 3, figsize=(15, 4.0 * len(sel)),
                               squeeze=False)
        for r, k in enumerate(sel):
            Ut, Up = saved[k]
            Mt, Mp = mach_from_U(Ut), mach_from_U(Up)
            vmax = max(Mt.max(), Mp.max())
            for c, (M, ttl) in enumerate(
                    ((Mt, f'teacher it{k}'), (Mp, f'proj it{k}'))):
                pc = ax[r][c].pcolormesh(Xphys, Yphys, M, cmap='turbo',
                                         vmin=0, vmax=vmax, shading='auto')
                ax[r][c].set_title(ttl); ax[r][c].set_aspect('equal')
                ax[r][c].set_xlim(-0.2, 1.2); ax[r][c].set_ylim(0, 1.0)
                fig.colorbar(pc, ax=ax[r][c], fraction=0.046)
            pc = ax[r][2].pcolormesh(Xphys, Yphys, np.abs(Mp - Mt),
                                     cmap='inferno', shading='auto')
            ax[r][2].set_title(f'|dMach| it{k}'); ax[r][2].set_aspect('equal')
            ax[r][2].set_xlim(-0.2, 1.2); ax[r][2].set_ylim(0, 1.0)
            fig.colorbar(pc, ax=ax[r][2], fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'mach_panels.png'), dpi=120,
                    bbox_inches='tight')
        print('saved', os.path.join(out, 'mach_panels.png'))

        zahr_path = save_zahr_compare(saved, Xphys, Yphys, out)
        if zahr_path is not None:
            print('saved', zahr_path)

    np.savez(os.path.join(out, 'tracks.npz'),
             **{f'U_teacher_{k}': v[0] for k, v in saved.items()},
             **{f'U_proj_{k}': v[1] for k, v in saved.items()},
             Xphys=Xphys, Yphys=Yphys)
    print('DONE.')


if __name__ == '__main__':
    main()
