"""
Implicit-EDNN with VarPro blended-RBF basis for DMR.

Each step:
    U_n     = Phi(theta_n) c_n                       # eval basis on grid
    U_tilde = U_n + dt * R_WENO(U_n)   (or SSP-RK3)  # advance ON GRID
    (theta_{n+1}, c_{n+1}) = argmin || Phi(theta) c - U_tilde ||^2  # VarPro warm-start

This is "implicit / projection-style" EDNN -- avoids the closure failure
of explicit (J^T J)^-1 J^T R projection by going through the grid.  Both
geometry (theta) and amplitude (c) evolve, so the shock-translation
direction lives in the tangent space.

Defaults in this copy are pinned to reproduce the successful run
`iednn_2026_05_28_09_24_35_I_also_lovethis` from the original case_2d_dmr
workflow.
"""
from __future__ import annotations
import argparse, math, os, shutil, sys, time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmr_setup import LX, LY, smoothed_ic_field
from weno_rhs import weno_rhs_grid
from fit_weno_snap_varpro import (
    BlendedRBF, lattice_centres, grad_rho_topk, fit_varpro,
    fit_varpro_adam, metrics,
)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_source_snapshot(run_dir):
    src_dir = os.path.join(run_dir, 'source_snapshot')
    os.makedirs(src_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in [
            'run_iednn_varpro.py',
            'dmr_setup.py',
            'weno_rhs.py',
            'fit_weno_snap_varpro.py',
            'make_summary_plot.py']:
        src = os.path.join(here, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(src_dir, name))


# ---------------------------------------------------------------- WENO step
def weno_step(U, dx, dy, t, dt, nu, integrator='ssprk3'):
    if integrator == 'euler':
        return U + dt * weno_rhs_grid(U, dx, dy, t, nu=nu)
    R0 = weno_rhs_grid(U, dx, dy, t, nu=nu)
    U1 = U + dt * R0
    R1 = weno_rhs_grid(U1, dx, dy, t + dt, nu=nu)
    U2 = 0.75 * U + 0.25 * (U1 + dt * R1)
    R2 = weno_rhs_grid(U2, dx, dy, t + 0.5 * dt, nu=nu)
    return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * R2)


def max_wavespeed(U):
    rho = U[..., 0].clamp_min(1e-12)
    u = U[..., 1] / rho; v = U[..., 2] / rho
    p = ((1.4 - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
         ).clamp_min(1e-12)
    a = (1.4 * p / rho).sqrt()
    return float((a + u.abs() + v.abs()).max())


# ---------------------------------------------------------------- plot
def _fmt_scalar_or_med(v):
    """Format a scalar or an array (taking median) for plot titles."""
    try:
        if hasattr(v, '__len__') and len(v) > 1:
            return f"med={float(np.median(v)):.3e}"
        return f"{float(v):.3e}"
    except Exception:
        return str(v)


def plot_snapshot(U_b, U_w_fine, U_w_coarse, t, out_path, rL2_b, rL2_c,
                  model_dump=None):
    """4-row figure (all imshow with bicubic):
        row 0: fine WENO ref
        row 1: i-EDNN RBF
        row 2: same-DOF coarse WENO  (None -> blank)
        row 3: RBF - fine ref       (signed error)
    Bottom: sharp atom (mu_Nx, mu_Ny) scattered on rho, marker size ~ sigma_N
    """
    rho_b = U_b[..., 0]; rho_w = U_w_fine[..., 0]
    err = rho_b - rho_w
    rhos = [rho_b, rho_w]
    if U_w_coarse is not None:
        rhos.append(U_w_coarse[..., 0])
    vmin = float(min(r.min() for r in rhos))
    vmax = float(max(r.max() for r in rhos))

    n_rows = 4 + (1 if model_dump is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.6 * n_rows))
    ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]

    im0 = ax0.imshow(rho_w.T, origin='lower', extent=[0, LX, 0, LY],
                     aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax,
                     interpolation='bicubic')
    ax0.set_title(f"WENO fine ref ({rho_w.shape[0]}x{rho_w.shape[1]})  t={t:.4f}")
    plt.colorbar(im0, ax=ax0, fraction=0.025)

    im1 = ax1.imshow(rho_b.T, origin='lower', extent=[0, LX, 0, LY],
                     aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax,
                     interpolation='bicubic')
    ax1.set_title(f"i-EDNN RBF  rL2_$\\rho$={rL2_b[0]:.2e}  "
                  f"rL2_E={rL2_b[3]:.2e}")
    plt.colorbar(im1, ax=ax1, fraction=0.025)

    if U_w_coarse is not None:
        rho_c = U_w_coarse[..., 0]
        im2 = ax2.imshow(rho_c.T, origin='lower', extent=[0, LX, 0, LY],
                         aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax,
                         interpolation='bicubic')
        ax2.set_title(f"WENO coarse ({rho_c.shape[0]}x{rho_c.shape[1]}, "
                      f"same DOF)  rL2_$\\rho$={rL2_c[0]:.2e}"
                      if rL2_c is not None
                      else f"WENO coarse ({rho_c.shape[0]}x{rho_c.shape[1]})")
        plt.colorbar(im2, ax=ax2, fraction=0.025)
    else:
        ax2.axis('off')

    emax = float(np.abs(err).max()) + 1e-30
    im3 = ax3.imshow(err.T, origin='lower', extent=[0, LX, 0, LY],
                     aspect='auto', cmap='seismic', vmin=-emax, vmax=emax,
                     interpolation='bicubic')
    ax3.set_title(f"err(RBF - fine)  max|.|={emax:.2e}")
    plt.colorbar(im3, ax=ax3, fraction=0.025)

    # atom distribution overlay
    if model_dump is not None:
        ax4 = axes[4]
        # background = rho_b (lighter cmap)
        ax4.imshow(rho_b.T, origin='lower', extent=[0, LX, 0, LY],
                   aspect='auto', cmap='gray', vmin=vmin, vmax=vmax,
                   interpolation='bicubic', alpha=0.6)
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection
        muNx = model_dump['mu_Nx']; muNy = model_dump['mu_Ny']
        sigN = model_dump['sigma_N']
        sigN_perp = model_dump.get('sigma_N_perp', sigN)
        thN  = model_dump.get('theta_N', np.zeros_like(sigN))
        muSx = model_dump['mu_Sx']; muSy = model_dump['mu_Sy']
        sigS = np.atleast_1d(model_dump['sigma_S'])
        sigS_perp = np.atleast_1d(model_dump.get('sigma_S_perp', sigS))
        thS = np.atleast_1d(model_dump.get('theta_S', np.zeros_like(sigS)))
        if sigS.size == 1:  sigS      = np.full(muSx.shape, sigS[0])
        if sigS_perp.size == 1: sigS_perp = np.full(muSx.shape, sigS_perp[0])
        if thS.size == 1:   thS       = np.full(muSx.shape, thS[0])

        # smooth atoms as light cyan ellipses (1-sigma contour)
        sE = [Ellipse((x, y), width=2*sp, height=2*sq, angle=np.degrees(th))
              for x, y, sp, sq, th in zip(muSx, muSy, sigS, sigS_perp, thS)]
        ax4.add_collection(PatchCollection(
            sE, facecolor='none', edgecolor='cyan',
            linewidth=0.6, alpha=0.6))
        # sharp atoms as plasma-colored ellipses, color = sigma_par
        nE = [Ellipse((x, y), width=2*sp, height=2*sq, angle=np.degrees(th))
              for x, y, sp, sq, th in zip(muNx, muNy, sigN, sigN_perp, thN)]
        pc = PatchCollection(nE, cmap='plasma', alpha=0.75,
                             edgecolor='k', linewidth=0.3)
        pc.set_array(np.asarray(sigN))
        ax4.add_collection(pc)
        cb = plt.colorbar(pc, ax=ax4, fraction=0.025)
        cb.set_label(r'$\sigma_{N,\parallel}$')
        # show full extended-domain view so out-of-bound atoms are visible
        pad_x = 0.35 * LX; pad_y = 0.35 * LY
        ax4.set_xlim(-pad_x, LX + pad_x)
        ax4.set_ylim(-pad_y, LY + pad_y)
        ax4.axvline(0, color='w', lw=0.8); ax4.axvline(LX, color='w', lw=0.8)
        ax4.axhline(0, color='w', lw=0.8); ax4.axhline(LY, color='w', lw=0.8)
        n_oob = int(((muNx<0)|(muNx>LX)|(muNy<0)|(muNy>LY)).sum())
        ax4.set_title(f"atom layout (1-$\\sigma$ ellipses, "
                      f"sharp out-of-domain: {n_oob}/{muNx.size})  "
                      f"sigma_S={_fmt_scalar_or_med(model_dump['sigma_S'])}")
        # dummy proxies for legend
        ax4.plot([], [], 'o', mfc='none', mec='cyan', label=f'smooth K_s={muSx.size}')
        ax4.plot([], [], 'o', mfc='none', mec='k',   label=f'sharp K_n={muNx.size}')
        ax4.legend(loc='upper right', framealpha=0.85, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)


def dump_model(model):
    with torch.no_grad():
        # sigma_S is scalar in shared mode, (K_s,) array in per-atom mode
        if model.smooth_per_atom:
            sigma_S_out = model.log_sig_S.exp().cpu().numpy().copy()
        else:
            sigma_S_out = float(model.log_sig_S.exp().cpu().numpy())
        d = dict(
            mu_Sx=model.mu_Sx.cpu().numpy().copy(),
            mu_Sy=model.mu_Sy.cpu().numpy().copy(),
            mu_Nx=model.mu_Nx.cpu().numpy().copy(),
            mu_Ny=model.mu_Ny.cpu().numpy().copy(),
            sigma_N=model.log_sig_N.exp().cpu().numpy().copy(),
            sigma_S=sigma_S_out,
            c=model.c.cpu().numpy().copy(),
            K_s=int(model.K_s), K_n=int(model.K_n),
            anisotropic=bool(getattr(model, 'anisotropic', False)),
            smooth_per_atom=bool(getattr(model, 'smooth_per_atom', False)),
            smooth_anisotropic=bool(getattr(model, 'smooth_anisotropic', False)),
        )
        if d['anisotropic']:
            d['sigma_N_perp'] = model.log_sig_N_perp.exp().cpu().numpy().copy()
            d['theta_N'] = model.theta_N.cpu().numpy().copy()
        if d['smooth_anisotropic']:
            d['sigma_S_perp'] = model.log_sig_S_perp.exp().cpu().numpy().copy()
            d['theta_S'] = model.theta_S.cpu().numpy().copy()
        return d


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    # grid
    ap.add_argument('--Nx', type=int, default=200)
    ap.add_argument('--Ny', type=int, default=50)
    ap.add_argument('--Nx_coarse', type=int, default=44,
                    help='same-DOF coarse WENO grid (44*11*4=1936 ~ 1981)')
    ap.add_argument('--Ny_coarse', type=int, default=11)
    ap.add_argument('--T', type=float, default=0.6)
    ap.add_argument('--dt', type=float, default=1e-3)
    ap.add_argument('--cfl', type=float, default=0.0,
                    help='if>0 dt=cfl*min(dx,dy)/amax overrides --dt')
    ap.add_argument('--delta', type=float, default=0.012)
    ap.add_argument('--nu', type=float, default=0.0)
    ap.add_argument('--integrator', choices=['euler', 'ssprk3'],
                    default='ssprk3')
    # basis
    ap.add_argument('--K_s', type=int, default=140)
    ap.add_argument('--K_n', type=int, default=140)
    ap.add_argument('--sigma_S_init_factor', type=float, default=2.0)
    ap.add_argument('--sigma_N_init_factor', type=float, default=0.8)
    ap.add_argument('--sigma_S_box', type=str, default='0.04,2.0')
    ap.add_argument('--sigma_N_box', type=str, default='4e-3,0.5',
                    help='sigma_N box used during TIME STEPPING (tight = '
                        'fewer footprints)')
    ap.add_argument('--sigma_N_box_ic', type=str, default='4e-3,0.5',
                    help='OPTIONAL looser sigma_N box used only during IC '
                        'cold-start. If empty, falls back to --sigma_N_box. '
                        'After IC fit, the box is squeezed to --sigma_N_box '
                        'and atoms are clamped.')
    # IC fit
    ap.add_argument('--ic_optimizer', choices=['lm', 'adam', 'hybrid'],
                    default='hybrid',
                    help="IC cold-start optimizer.  'lm' = Levenberg-Marquardt "
                         "(default); 'adam' = Adam-on-theta VarPro (better "
                         "basin-hopping for cold-start, weaker polish); "
                         "'hybrid' = --adam_iters_ic Adam warmup then "
                         "--fit_iters_ic LM polish.")
    ap.add_argument('--fit_iters_ic', type=int, default=1500,
                    help='LM iters for IC fit (used by lm and hybrid modes)')
    ap.add_argument('--adam_iters_ic', type=int, default=2000,
                    help='Adam iters for IC fit (used by adam and hybrid modes)')
    ap.add_argument('--adam_lr_mu', type=float, default=1e-3,
                    help='Adam lr for sharp atom positions (mu_Nx, mu_Ny)')
    ap.add_argument('--adam_lr_sig', type=float, default=5e-3,
                    help='Adam lr for log_sigma_S, log_sigma_N (log-space)')
    ap.add_argument('--adam_lr_theta', type=float, default=5e-3,
                    help='Adam lr for theta_N (radians); -1 = use --adam_lr_sig.'
                         ' Only used when --sharp_anisotropic.')
    ap.add_argument('--sharp_anisotropic', action='store_true',
                    default=True,
                    help='Use anisotropic Gaussian sharp atoms: each sharp '
                         'atom gets (sigma_par, sigma_perp, theta) instead '
                         'of a single sigma.  +2 nonlinear DOF per atom. '
                         'Smooth pool stays isotropic.  Off by default '
                         '(bit-identical to old behaviour).')
    ap.add_argument('--smooth_per_atom', action='store_true',
                    help='Each smooth atom gets its own sigma_S (K_s DOF '
                         'instead of 1).  Lets the smooth pool form '
                         'multi-radius stencils for high-order local '
                         'reproduction.  Off by default (back-compat).')
    ap.add_argument('--smooth_anisotropic', action='store_true',
                    default=True,
                    help='Per-atom anisotropic smooth pool: each smooth '
                         'atom gets (sigma_par, sigma_perp, theta).  '
                         'Implies --smooth_per_atom.  +3*K_s DOF total '
                         '(replaces the 1 shared scalar).  Off by default.')
    ap.add_argument('--mu_extend_frac', type=float, default=0.3,
                    help='Allow sharp-atom centres to drift OUTSIDE the '
                         'physical domain by this fraction of LX/LY on '
                         'each side.  0.0 = hard clamp at [0,L] (default, '
                         'back-compat).  0.2 = 20%% slack on each side '
                         '(mu_x in [-0.2*LX, 1.2*LX]).  Useful near '
                         'boundary layers / reflected shocks where the '
                         'ideal Gaussian centre would sit outside.')
    ap.add_argument('--fit_rebirth_every_ic', type=int, default=20)
    ap.add_argument('--fit_iters_squeeze', type=int, default=0,
                    help='LM iters for the squeeze re-equilibration after '
                         'sigma_N box clamp.  0 = SKIP the LM re-fit '
                         'entirely (only the linear solve_linear re-solve '
                         'after the clamp is kept).  >0 = run that many LM '
                         'iters.')
    # per-step fit
    ap.add_argument('--fit_iters_step', type=int, default=600,
                    help='warm-start LM iters PER TIME STEP (small, since '
                         'theta moves O(dt))')
    ap.add_argument('--ridge', type=float, default=1e-6,
                    help='Tikhonov ridge on c.  BIASES the solution toward '
                         'zero in EVERY direction; too aggressive causes '
                         'underfit.  Use ONLY if you know the problem and '
                         'usually prefer --rcond instead.')
    ap.add_argument('--rcond', type=float, default=1e-6,
                    help='SVD truncation threshold for the inner lstsq: '
                         'singular values below rcond * max(sigma) are set '
                         'to zero.  This kills NULL-SPACE directions only '
                         '(huge cancellations in c) without biasing the '
                         'signal.  Try 1e-6 .. 1e-8 if you see |c| > 1e+6.')
    ap.add_argument('--rebirth_every_steps', type=int, default=3,
                    help='reseed weakest sharp atoms onto high |grad rho| '
                         'cells every this many time steps')
    ap.add_argument('--reseed_sharp_frac', type=float, default=0.2)
    # output
    ap.add_argument('--n_snaps', type=int, default=9)
    ap.add_argument('--out_dir', type=str, default='output_iednn')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    run_dir = os.path.join(args.out_dir, f"iednn_{stamp}")
    snaps_dir = os.path.join(run_dir, 'snaps')
    os.makedirs(snaps_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
    save_source_snapshot(run_dir)
    print(f"==> {os.path.abspath(run_dir)}")

    Nx, Ny = args.Nx, args.Ny
    dx, dy = LX / Nx, LY / Ny
    xg = (torch.arange(Nx, device=device) + 0.5) * dx
    yg = (torch.arange(Ny, device=device) + 0.5) * dy
    X, Y = torch.meshgrid(xg, yg, indexing='ij')
    Xn = X.cpu().numpy(); Yn = Y.cpu().numpy()

    # IC
    rho0, ru0, rv0, E0 = smoothed_ic_field(X, Y, delta=args.delta,
                                            backend='torch')
    U_ic = torch.stack([rho0, ru0, rv0, E0], dim=-1).to(device).contiguous()
    U_ref = U_ic.clone()
    print(f"[grid] Nx={Nx} Ny={Ny}  dx={dx:.4g} dy={dy:.4g}")

    # ---- equal-DOF coarse WENO (independent parallel march) ----
    Nxc, Nyc = args.Nx_coarse, args.Ny_coarse
    dxc, dyc = LX / Nxc, LY / Nyc
    xgc = (torch.arange(Nxc, device=device) + 0.5) * dxc
    ygc = (torch.arange(Nyc, device=device) + 0.5) * dyc
    Xc, Yc = torch.meshgrid(xgc, ygc, indexing='ij')
    rcho0, rcu0, rcv0, Ec0 = smoothed_ic_field(Xc, Yc, delta=args.delta,
                                                backend='torch')
    U_coarse = torch.stack([rcho0, rcu0, rcv0, Ec0], dim=-1
                            ).to(device).contiguous()
    print(f"[grid coarse] {Nxc}x{Nyc}  dx={dxc:.4g} dy={dyc:.4g}  "
          f"DOF={4*Nxc*Nyc}")

    # basis
    sigS_box = tuple(float(s) for s in args.sigma_S_box.split(','))
    sigN_box_step = tuple(float(s) for s in args.sigma_N_box.split(','))
    if args.sigma_N_box_ic.strip():
        sigN_box_ic = tuple(float(s) for s in
                            args.sigma_N_box_ic.split(','))
    else:
        sigN_box_ic = sigN_box_step
    Nx_g, Ny_g, K_s_eff, mu_Sx_np, mu_Sy_np = lattice_centres(args.K_s, LX, LY)
    sig_S_init = args.sigma_S_init_factor * (LX / Nx_g)
    sig_N_init = args.sigma_N_init_factor * dx
    U0_np = U_ic.cpu().numpy()
    mu_Nx_np, mu_Ny_np = grad_rho_topk(U0_np, Xn, Yn, args.K_n)
    x_flat = X.reshape(-1); y_flat = Y.reshape(-1)
    model = BlendedRBF(
        x_flat, y_flat,
        mu_S=(mu_Sx_np, mu_Sy_np),
        mu_N=(mu_Nx_np, mu_Ny_np),
        log_sig_S_init=math.log(sig_S_init),
        log_sig_N_init=math.log(sig_N_init),
        sigma_S_box=sigS_box, sigma_N_box=sigN_box_ic,
        anisotropic=args.sharp_anisotropic,
        smooth_per_atom=args.smooth_per_atom,
        smooth_anisotropic=args.smooth_anisotropic,
        mu_extend_frac=args.mu_extend_frac)
    print(f"[sharp atoms] anisotropic={args.sharp_anisotropic}  "
          f"(P_nl/atom = {5 if args.sharp_anisotropic else 3})")
    print(f"[smooth atoms] per_atom={model.smooth_per_atom}  "
          f"anisotropic={model.smooth_anisotropic}  "
          f"(P_nl/atom = {3 if model.smooth_anisotropic else (1 if model.smooth_per_atom else 0)}"
          f"{'  + 1 shared scalar' if not model.smooth_per_atom else ''})")
    print(f"[sigma_N box] IC=[{sigN_box_ic[0]:.3g},{sigN_box_ic[1]:.3g}]  "
          f"STEP=[{sigN_box_step[0]:.3g},{sigN_box_step[1]:.3g}]")
    P_lin = K_s_eff + args.K_n + 1
    n_nl_per_sharp = 5 if args.sharp_anisotropic else 3
    if model.smooth_anisotropic:
        n_nl_smooth_total = 3 * K_s_eff
    elif model.smooth_per_atom:
        n_nl_smooth_total = K_s_eff
    else:
        n_nl_smooth_total = 1
    DOF_tot = 4 * P_lin + n_nl_smooth_total + n_nl_per_sharp * args.K_n
    print(f"[basis] K_s={K_s_eff} K_n={args.K_n}  P_lin/field={P_lin}  "
          f"DOF_total={DOF_tot}  (vs WENO 4*Nx*Ny={4*Nx*Ny})")

    # ---- IC cold-start fit ----
    U_ic_flat = U_ic.reshape(-1, 4).contiguous()
    U_ic_np = U_ic.cpu().numpy().reshape(-1, 4)
    print(f"\n=== IC fit (mode={args.ic_optimizer})  "
          f"ridge={args.ridge:.1e}  rcond={args.rcond:.1e} ===")

    if args.ic_optimizer in ('adam', 'hybrid'):
        print(f"  [Adam] {args.adam_iters_ic} iters  "
              f"lr_mu={args.adam_lr_mu:.1e}  lr_sig={args.adam_lr_sig:.1e}")
        t0 = time.time()
        res = fit_varpro_adam(
            model, U_ic_flat, U_ic_np,
            iters=args.adam_iters_ic,
            lr_mu=args.adam_lr_mu, lr_sig=args.adam_lr_sig,
            lr_theta=(None if args.adam_lr_theta < 0
                      else args.adam_lr_theta),
            cosine_decay=True, lr_min_factor=1e-2,
            rebirth_every=args.fit_rebirth_every_ic,
            rebirth_frac=0.08, rebirth_until_frac=0.85,
            ridge=args.ridge, rcond=args.rcond,
            verbose=False,
            progress=True, progress_desc='IC-Adam',
            progress_metric_every=25)
        print(f"  ic-adam wall={time.time()-t0:.1f}s  "
              f"rL2=({res['relL2'][0]:.2e},{res['relL2'][1]:.2e},"
              f"{res['relL2'][2]:.2e},{res['relL2'][3]:.2e})  "
              f"rLinf={res['relLi'].max():.2e}")

    if args.ic_optimizer in ('lm', 'hybrid'):
        print(f"  [LM]   {args.fit_iters_ic} iters")
        t0 = time.time()
        res = fit_varpro(model, U_ic_flat, U_ic_np,
                         iters=args.fit_iters_ic,
                         ridge=args.ridge, rcond=args.rcond,
                         rebirth_every=args.fit_rebirth_every_ic,
                         rebirth_frac=0.08, rebirth_until_frac=0.85,
                         verbose=False,
                         progress=True, progress_desc='IC-LM',
                         progress_metric_every=25)
        print(f"  ic-lm wall={time.time()-t0:.1f}s  "
              f"rL2=({res['relL2'][0]:.2e},{res['relL2'][1]:.2e},"
              f"{res['relL2'][2]:.2e},{res['relL2'][3]:.2e})  "
              f"rLinf={res['relLi'].max():.2e}")

    # ---- squeeze sigma_N box to the STEP setting (if different) ----
    if sigN_box_ic != sigN_box_step:
        with torch.no_grad():
            model.log_sig_N_box = (math.log(sigN_box_step[0]),
                                    math.log(sigN_box_step[1]))
            # clamp current log_sig_N into the new tighter box so that
            # atoms wider than the new upper bound get pulled in BEFORE
            # the first VarPro warm-start sees them
            model.log_sig_N = model.log_sig_N.clamp(
                model.log_sig_N_box[0], model.log_sig_N_box[1])
            if getattr(model, 'anisotropic', False):
                model.log_sig_N_perp = model.log_sig_N_perp.clamp(
                    model.log_sig_N_box[0], model.log_sig_N_box[1])
            # after clamping atom *widths*, the linear coefficients c
            # learned for the WIDE atoms are stale -- re-solve c so the
            # representation is at least linearly optimal at the new theta
            model.solve_linear(U_ic.reshape(-1, 4).contiguous(),
                                ridge=args.ridge, rcond=args.rcond)
        # quick LM re-equilibration: let theta and c both move under the
        # new tight box.  Cheap (few hundred iters) and avoids handing the
        # time-stepper a broken state.
        n_reeq = args.fit_iters_squeeze
        if n_reeq <= 0:
            print(f"  [squeeze] sigma_N box -> "
                  f"[{sigN_box_step[0]:.3g},{sigN_box_step[1]:.3g}]; "
                  f"clamped + relinearised; LM re-eq SKIPPED "
                  f"(fit_iters_squeeze=0)")
        else:
            print(f"  [squeeze] sigma_N box -> "
                  f"[{sigN_box_step[0]:.3g},{sigN_box_step[1]:.3g}]; "
                  f"clamped + relinearised; re-equilibrating {n_reeq} iters...")
            t0 = time.time()
            res = fit_varpro(model, U_ic.reshape(-1, 4).contiguous(),
                             U_ic.cpu().numpy().reshape(-1, 4),
                             iters=n_reeq,
                             ridge=args.ridge, rcond=args.rcond,
                             rebirth_every=0,  # no rebirth, just settle
                             rebirth_frac=0.0, rebirth_until_frac=1.0,
                             verbose=False,
                             progress=True, progress_desc='IC-squeeze',
                             progress_metric_every=25)
            print(f"  squeeze-fit wall={time.time()-t0:.1f}s  "
                  f"rL2=({res['relL2'][0]:.2e},{res['relL2'][1]:.2e},"
                  f"{res['relL2'][2]:.2e},{res['relL2'][3]:.2e})  "
                  f"rLinf={res['relLi'].max():.2e}")

    # snapshot schedule
    snap_times = np.linspace(0.0, args.T, args.n_snaps).tolist()
    snaps_done = set()
    snap_log = []  # (t, rL2, rLinf)

    def eval_basis():
        with torch.no_grad():
            A, _, _ = model.design()
            return (A @ model.c.t()).view(Nx, Ny, 4)

    def save_snap(t_now, U_b, U_w, U_w_c, tag):
        U_b_np = U_b.cpu().numpy(); U_w_np = U_w.cpu().numpy()
        U_w_c_np = U_w_c.cpu().numpy() if U_w_c is not None else None
        rL2, rLi = metrics(U_b_np.reshape(-1, 4), U_w_np.reshape(-1, 4))
        # coarse-WENO metrics: upsample to fine grid via bicubic before err
        rL2_c = None; rLi_c = None
        if U_w_c_np is not None:
            from scipy.interpolate import RectBivariateSpline
            Nxc_, Nyc_, _ = U_w_c_np.shape
            xc_ = (np.arange(Nxc_) + 0.5) / Nxc_ * LX
            yc_ = (np.arange(Nyc_) + 0.5) / Nyc_ * LY
            xf_ = (np.arange(Nx) + 0.5) / Nx * LX
            yf_ = (np.arange(Ny) + 0.5) / Ny * LY
            U_c_up = np.empty_like(U_w_np)
            for k in range(4):
                sp = RectBivariateSpline(xc_, yc_, U_w_c_np[..., k],
                                          kx=3, ky=3)
                U_c_up[..., k] = sp(xf_, yf_)
            rL2_c, rLi_c = metrics(U_c_up.reshape(-1, 4),
                                    U_w_np.reshape(-1, 4))
        mdump = dump_model(model)
        snap_log.append((t_now, rL2, rLi, rL2_c, rLi_c))
        png = os.path.join(snaps_dir, f"snap_t{t_now:.4f}.png")
        plot_snapshot(U_b_np, U_w_np, U_w_c_np, t_now, png,
                       rL2, rL2_c, model_dump=mdump)
        np.savez(os.path.join(snaps_dir, f"snap_t{t_now:.4f}.npz"),
                 U_basis=U_b_np, U_weno=U_w_np,
                 U_weno_coarse=(U_w_c_np if U_w_c_np is not None
                                 else np.zeros(0)),
                 t=t_now, rL2=rL2, rLinf=rLi,
                 rL2_coarse=(rL2_c if rL2_c is not None else np.zeros(0)),
                 rLinf_coarse=(rLi_c if rLi_c is not None else np.zeros(0)),
                 mu_Sx=mdump['mu_Sx'], mu_Sy=mdump['mu_Sy'],
                 mu_Nx=mdump['mu_Nx'], mu_Ny=mdump['mu_Ny'],
                 sigma_N=mdump['sigma_N'], sigma_S=mdump['sigma_S'],
                 sigma_N_perp=mdump.get('sigma_N_perp', np.zeros(0)),
                 theta_N=mdump.get('theta_N', np.zeros(0)),
                 sigma_S_perp=mdump.get('sigma_S_perp', np.zeros(0)),
                 theta_S=mdump.get('theta_S', np.zeros(0)),
                 anisotropic=mdump['anisotropic'],
                 smooth_per_atom=mdump['smooth_per_atom'],
                 smooth_anisotropic=mdump['smooth_anisotropic'],
                 c=mdump['c'])
        extra = (f"  rL2_co_rho={rL2_c[0]:.2e}" if rL2_c is not None else "")
        print(f"  [snap {tag} t={t_now:.4f}] rL2_rho={rL2[0]:.2e} "
              f"rL2_E={rL2[3]:.2e} rLinf={rLi.max():.2e}{extra}")

    save_snap(0.0, eval_basis(), U_ref, U_coarse, 'IC')
    snaps_done.add(0.0)

    # ---- time loop ----
    print(f"\n=== implicit-EDNN time march  integrator={args.integrator} ===")
    t = 0.0; step = 0
    t_wall = time.time()
    log_every = 5
    n_steps_est = int(math.ceil(args.T / args.dt)) if args.cfl <= 0 else None
    pbar = tqdm(total=n_steps_est, desc='i-EDNN', unit='step',
                dynamic_ncols=True, mininterval=0.5,
                smoothing=0.1) if tqdm is not None else None
    last_rL2 = float('nan'); last_rLi = float('nan')

    while t < args.T - 1e-14:
        # dt selection
        if args.cfl > 0:
            U_eval = eval_basis()
            amax = max_wavespeed(U_eval)
            dt = args.cfl * min(dx, dy) / max(amax, 1e-12)
        else:
            dt = args.dt
        dt = min(dt, args.T - t)

        # Step A: eval basis, advance ONE step on the grid via WENO
        U_n = eval_basis()
        U_tilde = weno_step(U_n, dx, dy, t, dt, args.nu, args.integrator)

        # Step A': also march the independent reference + coarse-WENO baseline
        with torch.no_grad():
            U_ref = weno_step(U_ref, dx, dy, t, dt, args.nu, args.integrator)
            U_coarse = weno_step(U_coarse, dxc, dyc, t, dt, args.nu,
                                  args.integrator)

        # Optional: reseed worst sharp atoms onto current |grad rho| sites
        do_reseed = (args.rebirth_every_steps > 0
                     and step % args.rebirth_every_steps == 0
                     and args.reseed_sharp_frac > 0)
        if do_reseed:
            U_now_np = U_tilde.cpu().numpy()
            mx, my = grad_rho_topk(U_now_np, Xn, Yn,
                                    int(args.reseed_sharp_frac * args.K_n))
            with torch.no_grad():
                c_norm = (model.c[:, model.K_s:model.K_s + model.K_n]
                          .abs().sum(0).cpu().numpy())
                weak = np.argpartition(c_norm, mx.size)[:mx.size]
                model.mu_Nx[weak] = torch.from_numpy(mx).to(model.mu_Nx)
                model.mu_Ny[weak] = torch.from_numpy(my).to(model.mu_Ny)
                if getattr(model, 'anisotropic', False):
                    # reset shape DOFs of moved atoms to box midpoint /
                    # isotropic so they start fresh at their new location
                    log_sig_mid = 0.5 * (model.log_sig_N_box[0]
                                         + model.log_sig_N_box[1])
                    model.log_sig_N[weak] = log_sig_mid
                    model.log_sig_N_perp[weak] = log_sig_mid
                    model.theta_N[weak] = 0.0

        # Step B: warm-started VarPro -> (theta_{n+1}, c_{n+1})
        U_tilde_flat = U_tilde.reshape(-1, 4).contiguous()
        U_tilde_np = U_tilde.cpu().numpy().reshape(-1, 4)
        res_step = fit_varpro(model, U_tilde_flat, U_tilde_np,
                              iters=args.fit_iters_step,
                              ridge=args.ridge, rcond=args.rcond,
                              rebirth_every=0,    # no rebirth inside per-step
                              rebirth_frac=0.0,
                              rebirth_until_frac=0.0,
                              verbose=False)
        # ---- per-step LM convergence diagnostics ----
        hist = res_step['history']
        loss_init = hist[0]; loss_final = hist[-1]
        # length of hist: 1 (init) + accepted_iters; if < 1+iters -> early stall
        n_accepted = len(hist) - 1
        stalled = n_accepted < args.fit_iters_step
        # loss reduction ratio (lower = better progress)
        red = loss_final / max(loss_init, 1e-300)
        # tail-decay over last 10% of accepted iters: did it plateau?
        k_tail = max(1, n_accepted // 10)
        tail_red = (hist[-1] / max(hist[-1 - k_tail], 1e-300)
                    if n_accepted >= k_tail else float('nan'))

        t += dt; step += 1

        if step % log_every == 0 or step == 1:
            U_b = eval_basis()
            rL2, rLi = metrics(U_b.cpu().numpy().reshape(-1, 4),
                               U_ref.cpu().numpy().reshape(-1, 4))
            last_rL2 = rL2[0]; last_rLi = rLi.max()
            msg = (f"  step={step:4d}  t={t:.4f}  dt={dt:.2e}  "
                   f"rL2_rho={rL2[0]:.2e}  rL2_E={rL2[3]:.2e}  "
                   f"rLinf={rLi.max():.2e}  "
                   f"LM[acc={n_accepted}/{args.fit_iters_step}"
                   f"{' STALL' if stalled else ''}  "
                   f"loss {loss_init:.2e}->{loss_final:.2e} "
                   f"red={red:.2e} tail={tail_red:.2f}]  "
                   f"wall={time.time()-t_wall:.1f}s")
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)
        if pbar is not None:
            pbar.set_postfix(t=f"{t:.4f}", rL2_rho=f"{last_rL2:.2e}",
                             rLinf=f"{last_rLi:.2e}")
            pbar.update(1)

        # snapshots
        for ts in snap_times:
            if ts in snaps_done: continue
            if abs(t - ts) < 0.6 * dt:
                snaps_done.add(ts)
                save_snap(ts, eval_basis(), U_ref, U_coarse, f'step{step}')

    if pbar is not None:
        pbar.close()

    # final summary
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(f"# implicit-EDNN VarPro blended-RBF\n")
        f.write(f"# Nx={Nx} Ny={Ny}  T={args.T}  dt={args.dt}  "
                f"integrator={args.integrator}\n")
        f.write(f"# K_s={K_s_eff} K_n={args.K_n}  DOF_total={DOF_tot}  "
                f"(grid DOF=4*Nx*Ny={4*Nx*Ny})\n")
        f.write(f"# coarse WENO {Nxc}x{Nyc} DOF={4*Nxc*Nyc}\n")
        f.write(f"# total steps={step}  wall={time.time()-t_wall:.1f}s\n#\n")
        f.write(f"# t   rL2_rho_RBF  rL2_E_RBF  rLinf_RBF  "
                f"rL2_rho_coarse  rL2_E_coarse  rLinf_coarse\n")
        for entry in snap_log:
            ts, rL2, rLi = entry[0], entry[1], entry[2]
            rL2_c = entry[3] if len(entry) > 3 else None
            rLi_c = entry[4] if len(entry) > 4 else None
            line = f"  {ts:.4f}  {rL2[0]:.3e}  {rL2[3]:.3e}  {rLi.max():.3e}"
            if rL2_c is not None:
                line += f"  {rL2_c[0]:.3e}  {rL2_c[3]:.3e}  {rLi_c.max():.3e}"
            f.write(line + "\n")
    print(f"\n[done] wall={time.time()-t_wall:.1f}s  steps={step}\n[out] {run_dir}")


if __name__ == '__main__':
    main()
