"""
Blended-RBF Variable-Projection fit of a WENO snapshot.

Idea (Harris-Kassab-Divo 2017/2019 + classical VarPro):

    U_f(x, y) ~= sum_k  c_f^S[k] phi_S(x, y; mu_S[k], sigma_S)
              +  sum_l  c_f^N[l] phi_N(x, y; mu_N[l], sigma_N[l])
              +  bias_f

    smooth pool   (S): K_s atoms on a uniform lattice, ONE shared
                       width sigma_S (broad).  Captures slowly varying
                       background away from shocks.
    sharp  pool   (N): K_n atoms with individually-tuned (mu, sigma).
                       Initialized on top-K_n |grad rho| cells; each
                       atom has its own sigma in [sigma_N_min, sigma_N_max].
                       Captures C^0 discontinuities.

  Nonlinear parameters (joint over all 4 fields, the only thing GN sees):
        theta = [ log_sigma_S         (1)
                  mu_N_x, mu_N_y      (K_n each)
                  log_sigma_N         (K_n) ]
        => P_nl = 1 + 3 K_n            <<< this is the only nonlinear DOF
  Linear parameters (solved in closed form by LSTSQ, per field):
        c_f, bias_f, shape (K_s + K_n + 1)   for f = 0..3
        => P_lin = 4 (K_s + K_n + 1)

Variable Projection (Golub-Pereyra / Kaufman approx.):
    given theta, set c_f(theta) = argmin || A(theta) c_f - U_f ||  per field
    let r_f(theta) = A(theta) c_f(theta) - U_f
    Kaufman Jacobian: J_f = (dA/dtheta) c_f
    LM step on theta against stacked residual [r_0; r_1; r_2; r_3]
    with damping (J^T J + lam diag(J^T J)) dtheta = -J^T r .

DOF reported = P_nl + P_lin = effective parameter count.

Usage:
    python fit_weno_snap_varpro.py \
        --npz output_weno/weno_XXXX/final_state.npz \
        --K_s 64 --K_n 48 --iters 60
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from dmr_setup import LX, LY

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIELD_NAMES = ['rho', 'rho*u', 'rho*v', 'E']


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def lattice_centres(K, Lx, Ly):
    aspect = Lx / Ly
    Ny_g = max(int(round(math.sqrt(K / aspect))), 1)
    Nx_g = max(int(round(K / Ny_g)), 1)
    xs = (np.arange(Nx_g) + 0.5) * (Lx / Nx_g)
    ys = (np.arange(Ny_g) + 0.5) * (Ly / Ny_g)
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    return Nx_g, Ny_g, Nx_g * Ny_g, Xg.reshape(-1), Yg.reshape(-1)


def grad_rho_topk(U_np, X_np, Y_np, K):
    rho = U_np[..., 0]
    drx = np.zeros_like(rho); dry = np.zeros_like(rho)
    drx[1:-1, :] = 0.5 * (rho[2:, :] - rho[:-2, :])
    dry[:, 1:-1] = 0.5 * (rho[:, 2:] - rho[:, :-2])
    mag = np.sqrt(drx * drx + dry * dry)
    Nx, Ny = rho.shape
    # non-maximal-suppression by greedy pick with min spacing
    flat_idx = np.argsort(mag.ravel())[::-1]
    dx = X_np[1, 0] - X_np[0, 0] if Nx > 1 else 0.0
    dy = Y_np[0, 1] - Y_np[0, 0] if Ny > 1 else 0.0
    chosen = []
    min_d2 = (1.5 * max(dx, dy)) ** 2
    for f in flat_idx:
        ix, iy = np.unravel_index(f, (Nx, Ny))
        cx, cy = X_np[ix, iy], Y_np[ix, iy]
        ok = True
        for (px, py) in chosen:
            if (cx - px) ** 2 + (cy - py) ** 2 < min_d2:
                ok = False; break
        if ok:
            chosen.append((cx, cy))
            if len(chosen) >= K:
                break
    if len(chosen) < K:
        rng = np.random.default_rng(0)
        for f in flat_idx:
            if len(chosen) >= K: break
            ix, iy = np.unravel_index(f, (Nx, Ny))
            chosen.append((X_np[ix, iy] + 0.3 * dx * (rng.random() - 0.5),
                           Y_np[ix, iy] + 0.3 * dy * (rng.random() - 0.5)))
    mux = np.array([p[0] for p in chosen[:K]])
    muy = np.array([p[1] for p in chosen[:K]])
    return mux, muy


def lstsq_cpu(A, b, rcond=1e-12):
    A_cpu = A.detach().cpu(); b_cpu = b.detach().cpu()
    if b_cpu.ndim == 1:
        b_cpu = b_cpu.unsqueeze(1)
    sol = torch.linalg.lstsq(A_cpu, b_cpu, rcond=rcond, driver='gelsd')
    return sol.solution.to(A.device)


def metrics(U_pred_np, U_true_np):
    e = U_pred_np - U_true_np
    Us = U_true_np - U_true_np.mean(axis=0, keepdims=True)
    relL2 = (np.sqrt((e ** 2).mean(axis=0))
             / np.sqrt((Us ** 2).mean(axis=0)))
    relLi = (np.abs(e).max(axis=0)
             / (np.abs(U_true_np).max(axis=0) + 1e-30))
    return relL2, relLi


def metrics_full(U_pred_np, U_true_np):
    """Like metrics() but also returns relL1.  Kept separate to avoid
    breaking 2-tuple unpacking elsewhere."""
    relL2, relLi = metrics(U_pred_np, U_true_np)
    e = U_pred_np - U_true_np
    Us = U_true_np - U_true_np.mean(axis=0, keepdims=True)
    relL1 = (np.abs(e).mean(axis=0)
             / (np.abs(Us).mean(axis=0) + 1e-30))
    return relL2, relLi, relL1


# ----------------------------------------------------------------------
# Blended-RBF model
# ----------------------------------------------------------------------

class BlendedRBF:
    """All quantities held as plain tensors (no torch.nn.Module).  We
    drive the nonlinear params manually for VarPro/LM.
    """

    def __init__(self, x, y, mu_S, mu_N, log_sig_S_init, log_sig_N_init,
                 sigma_S_box=(0.05, 2.0), sigma_N_box=(5e-3, 0.08),
                 anisotropic=False,
                 smooth_per_atom=False, smooth_anisotropic=False,
                 mu_extend_frac=0.0):
        self.x = x; self.y = y
        self.mu_Sx = torch.as_tensor(mu_S[0], device=device)
        self.mu_Sy = torch.as_tensor(mu_S[1], device=device)
        self.K_s = self.mu_Sx.numel()
        self.mu_Nx = torch.as_tensor(mu_N[0], device=device).clone()
        self.mu_Ny = torch.as_tensor(mu_N[1], device=device).clone()
        self.K_n = self.mu_Nx.numel()
        # ---- smooth pool width DOF ----
        # Three modes (back-compat default = shared scalar):
        #   shared       : log_sig_S is a SCALAR (1 DOF)
        #   per-atom iso : log_sig_S is (K_s,) vector (K_s DOF)
        #   per-atom aniso: + log_sig_S_perp (K_s,) + theta_S (K_s,)
        # When shared, log_sig_S_perp/theta_S are allocated as zeros for
        # serialisation but NEVER read.
        # smooth_anisotropic implies smooth_per_atom.
        self.smooth_anisotropic = bool(smooth_anisotropic)
        self.smooth_per_atom = bool(smooth_per_atom) or self.smooth_anisotropic
        if self.smooth_per_atom:
            self.log_sig_S = torch.full((self.K_s,),
                                        float(log_sig_S_init), device=device)
        else:
            self.log_sig_S = torch.tensor(float(log_sig_S_init), device=device)
        self.log_sig_S_perp = torch.full((self.K_s,),
                                         float(log_sig_S_init), device=device)
        self.theta_S = torch.zeros(self.K_s, device=device)
        # ---- sharp pool width DOF ----
        self.log_sig_N = torch.full((self.K_n,),
                                    float(log_sig_N_init), device=device)
        # ---- anisotropic sharp atoms ----
        # phi_N(x; mu, sig_par, sig_perp, theta) =
        #   exp(-0.5 [(d_par/sig_par)^2 + (d_perp/sig_perp)^2])
        # where d_par = cos(t) dx + sin(t) dy, d_perp = -sin(t) dx + cos(t) dy
        # In isotropic mode (anisotropic=False) the perp/theta DOFs are
        # allocated but ignored (not packed, not in Jacobian, not projected),
        # so all behaviour is bit-for-bit identical to the old code path.
        self.anisotropic = bool(anisotropic)
        self.log_sig_N_perp = torch.full((self.K_n,),
                                         float(log_sig_N_init), device=device)
        self.theta_N = torch.zeros(self.K_n, device=device)
        self.log_sig_S_box = (math.log(sigma_S_box[0]),
                              math.log(sigma_S_box[1]))
        self.log_sig_N_box = (math.log(sigma_N_box[0]),
                              math.log(sigma_N_box[1]))
        # mu_extend_frac : allow sharp-atom centres to drift OUTSIDE the
        # physical domain by this fraction of LX/LY on each side.  0.0 =
        # original behaviour (hard clamp at [0,L]); 0.2 = 20% slack on each
        # side (mu_x in [-0.2*LX, 1.2*LX]).  Helps near boundaries where the
        # ideal Gaussian centre would sit outside the domain.
        _ext = float(mu_extend_frac)
        self.mu_box = (-_ext * LX, (1.0 + _ext) * LX,
                       -_ext * LY, (1.0 + _ext) * LY)
        self.c = None     # (4, K_s + K_n + 1) filled by solve_linear
        # Pointwise column gating: smooth rows multiplied by gate_S,
        # sharp rows multiplied by gate_N.  Both default to 1.  Set via
        # set_gate(beta) where beta in [0,1] is the shock-sensor weight.
        N = self.x.numel()
        self.gate_S = torch.ones(N, device=device)
        self.gate_N = torch.ones(N, device=device)

    @torch.no_grad()
    def set_gate(self, beta):
        """beta : (N,) tensor in [0,1].  Smooth columns weighted by (1-beta);
        sharp columns left alone (already locally supported).  This avoids
        rank-deficient Phi when sharp atoms sit in beta~0 regions."""
        beta = torch.as_tensor(beta, device=device, dtype=self.x.dtype)
        self.gate_S = (1.0 - beta).contiguous()
        # keep sharp gate at 1 -- gating both pools makes Phi singular
        self.gate_N = torch.ones_like(beta)

    # ---- packing/unpacking nonlinear parameter vector ----
    # Layout (in order):
    #   log_sig_S          : 1 (shared) or K_s (per-atom)
    #   log_sig_S_perp     : K_s    if smooth_anisotropic
    #   theta_S            : K_s    if smooth_anisotropic
    #   mu_Nx, mu_Ny       : K_n, K_n
    #   log_sig_N          : K_n
    #   log_sig_N_perp     : K_n    if anisotropic (sharp)
    #   theta_N            : K_n    if anisotropic (sharp)
    def pack(self):
        if self.smooth_per_atom:
            parts = [self.log_sig_S]
        else:
            parts = [self.log_sig_S.reshape(1)]
        if self.smooth_anisotropic:
            parts += [self.log_sig_S_perp, self.theta_S]
        parts += [self.mu_Nx, self.mu_Ny, self.log_sig_N]
        if self.anisotropic:
            parts += [self.log_sig_N_perp, self.theta_N]
        return torch.cat(parts)

    def unpack(self, theta):
        s = 0
        if self.smooth_per_atom:
            self.log_sig_S = theta[s:s + self.K_s]; s += self.K_s
        else:
            self.log_sig_S = theta[s]; s += 1
        if self.smooth_anisotropic:
            self.log_sig_S_perp = theta[s:s + self.K_s]; s += self.K_s
            self.theta_S = theta[s:s + self.K_s]; s += self.K_s
        self.mu_Nx = theta[s:s + self.K_n]; s += self.K_n
        self.mu_Ny = theta[s:s + self.K_n]; s += self.K_n
        self.log_sig_N = theta[s:s + self.K_n]; s += self.K_n
        if self.anisotropic:
            self.log_sig_N_perp = theta[s:s + self.K_n]; s += self.K_n
            self.theta_N = theta[s:s + self.K_n]

    @torch.no_grad()
    def project_(self):
        self.log_sig_S = self.log_sig_S.clamp(*self.log_sig_S_box)
        if self.smooth_anisotropic:
            self.log_sig_S_perp = self.log_sig_S_perp.clamp(
                *self.log_sig_S_box)
            self.theta_S = torch.remainder(self.theta_S + math.pi / 2,
                                           math.pi) - math.pi / 2
        self.mu_Nx = self.mu_Nx.clamp(self.mu_box[0], self.mu_box[1])
        self.mu_Ny = self.mu_Ny.clamp(self.mu_box[2], self.mu_box[3])
        self.log_sig_N = self.log_sig_N.clamp(*self.log_sig_N_box)
        if self.anisotropic:
            self.log_sig_N_perp = self.log_sig_N_perp.clamp(
                *self.log_sig_N_box)
            # wrap theta into (-pi/2, pi/2] -- atom is symmetric under
            # theta -> theta + pi, and a swap (sig_par <-> sig_perp,
            # theta -> theta + pi/2) gives the same atom; we don't fight
            # the latter, just the former.
            self.theta_N = torch.remainder(self.theta_N + math.pi / 2,
                                           math.pi) - math.pi / 2

    # ---- design matrix ----
    def _phi_S(self):
        dx = self.x[:, None] - self.mu_Sx[None, :]
        dy = self.y[:, None] - self.mu_Sy[None, :]
        if self.smooth_anisotropic:
            sp = self.log_sig_S.exp()[None, :]
            sq = self.log_sig_S_perp.exp()[None, :]
            ct = torch.cos(self.theta_S)[None, :]
            st = torch.sin(self.theta_S)[None, :]
            d_p = ct * dx + st * dy
            d_q = -st * dx + ct * dy
            phi = torch.exp(-0.5 * (d_p * d_p / (sp * sp)
                                    + d_q * d_q / (sq * sq)))
        elif self.smooth_per_atom:
            sig = self.log_sig_S.exp()[None, :]      # (1, K_s)
            phi = torch.exp(-0.5 * (dx * dx + dy * dy) / (sig * sig))
        else:
            sig = self.log_sig_S.exp()
            phi = torch.exp(-0.5 * (dx * dx + dy * dy) / (sig * sig))
        return phi * self.gate_S[:, None]

    def _phi_N(self):
        dx = self.x[:, None] - self.mu_Nx[None, :]
        dy = self.y[:, None] - self.mu_Ny[None, :]
        if not self.anisotropic:
            sig = self.log_sig_N.exp()[None, :]
            phi = torch.exp(-0.5 * (dx * dx + dy * dy) / (sig * sig))
        else:
            sp = self.log_sig_N.exp()[None, :]          # par (1,K_n)
            sq = self.log_sig_N_perp.exp()[None, :]      # perp
            ct = torch.cos(self.theta_N)[None, :]
            st = torch.sin(self.theta_N)[None, :]
            d_p = ct * dx + st * dy
            d_q = -st * dx + ct * dy
            phi = torch.exp(-0.5 * (d_p * d_p / (sp * sp)
                                    + d_q * d_q / (sq * sq)))
        return phi * self.gate_N[:, None]

    def design(self):
        Phi_S = self._phi_S()
        Phi_N = self._phi_N()
        one = torch.ones((Phi_S.shape[0], 1), device=device,
                         dtype=Phi_S.dtype)
        A = torch.cat([Phi_S, Phi_N, one], dim=1)
        return A, Phi_S, Phi_N

    # ---- linear LS for c per field ----
    def solve_linear(self, U_true, rcond=1e-12, ridge=0.0):
        """Solve min_c ||A c - U||^2 + ridge * ||c||^2.

        ridge>0 is essential when the design matrix becomes near-singular
        (overlapping atoms, clustered sharp atoms, etc.).  Without it, the
        norm of c can blow up to O(1e+10) and floating-point cancellation
        of these huge values produces far-field ripples that look like
        physical oscillations but are pure numerical noise.
        """
        A, _, _ = self.design()
        if ridge > 0.0:
            # augmented system: stack A on top of sqrt(ridge)*I and U on
            # top of zeros -- equivalent to Tikhonov but avoids forming A^T A
            P = A.shape[1]
            sqrt_r = math.sqrt(ridge)
            A_aug = torch.cat([A, sqrt_r * torch.eye(P, device=A.device,
                                                    dtype=A.dtype)], dim=0)
            b_aug = torch.cat([U_true,
                               torch.zeros((P, U_true.shape[1]),
                                            device=U_true.device,
                                            dtype=U_true.dtype)], dim=0)
            sol = lstsq_cpu(A_aug, b_aug, rcond=rcond)
        else:
            sol = lstsq_cpu(A, U_true, rcond=rcond)         # (P, 4)
        self.c = sol.t().contiguous()                    # (4, P)
        U_pred = A @ sol                                 # (N, 4)
        return U_pred, A

    # ---- VarPro/Kaufman Jacobian wrt nonlinear params, stacked over fields ----
    def jacobian_and_residual(self, U_true, ridge=0.0, rcond=1e-12):
        """Return (J, r) where
            r : (4N,)       stacked residual U_pred - U_true
            J : (4N, P_nl)  Kaufman approx J_kau = (dA/dtheta) c
        """
        U_pred, A = self.solve_linear(U_true, ridge=ridge, rcond=rcond)
        r = (U_pred - U_true).t().reshape(-1)            # (4N,)
        N = self.x.numel()
        Phi_S = A[:, :self.K_s]
        Phi_N = A[:, self.K_s:self.K_s + self.K_n]
        c_S = self.c[:, :self.K_s]                        # (4, K_s)
        c_N = self.c[:, self.K_s:self.K_s + self.K_n]    # (4, K_n)

        # ---- helper to lift a (N, K) atom-deriv block to (4N, K) ----
        def _per_atom_cols(dPhi_block, c_block):
            # dPhi_block: (N, K);  c_block: (4, K)
            # output (4N, K) where row block f is dPhi * c_block[f, :]
            blocks = []
            for f in range(4):
                blocks.append(dPhi_block * c_block[f:f + 1, :])
            return torch.cat(blocks, dim=0)

        # ---- smooth pool Jacobian columns ----
        dxS = self.x[:, None] - self.mu_Sx[None, :]
        dyS = self.y[:, None] - self.mu_Sy[None, :]
        if self.smooth_anisotropic:
            spS = self.log_sig_S.exp()[None, :]
            sqS = self.log_sig_S_perp.exp()[None, :]
            inv_spS2 = 1.0 / (spS * spS)
            inv_sqS2 = 1.0 / (sqS * sqS)
            ctS = torch.cos(self.theta_S)[None, :]
            stS = torch.sin(self.theta_S)[None, :]
            d_pS = ctS * dxS + stS * dyS
            d_qS = -stS * dxS + ctS * dyS
            dPhiS_dlogsp = Phi_S * (d_pS * d_pS * inv_spS2)
            dPhiS_dlogsq = Phi_S * (d_qS * d_qS * inv_sqS2)
            dPhiS_dtheta = -Phi_S * (d_pS * d_qS * (inv_spS2 - inv_sqS2))
            J_smooth = [
                _per_atom_cols(dPhiS_dlogsp, c_S),
                _per_atom_cols(dPhiS_dlogsq, c_S),
                _per_atom_cols(dPhiS_dtheta, c_S),
            ]
        elif self.smooth_per_atom:
            sigS = self.log_sig_S.exp()[None, :]
            inv_sS2 = 1.0 / (sigS * sigS)
            r2_S = dxS * dxS + dyS * dyS
            dPhiS_dlogs = Phi_S * (r2_S * inv_sS2)        # (N, K_s)
            J_smooth = [_per_atom_cols(dPhiS_dlogs, c_S)]
        else:
            # shared scalar log_sig_S  -> single column (4N, 1)
            sigS = self.log_sig_S.exp()
            r2_S = dxS * dxS + dyS * dyS
            dPhiS_dlogs = Phi_S * (r2_S / (sigS * sigS))
            col_logsigS_perf = dPhiS_dlogs @ c_S.t()         # (N, 4)
            J_smooth = [col_logsigS_perf.t().reshape(-1, 1)]  # (4N, 1)

        # ---- dPhi_N[:,l] / d (mu_Nx, mu_Ny, log_sig_N[, log_sig_N_perp, theta_N]) ----
        dxN = self.x[:, None] - self.mu_Nx[None, :]      # (N, K_n)
        dyN = self.y[:, None] - self.mu_Ny[None, :]

        if not self.anisotropic:
            sigN = self.log_sig_N.exp()[None, :]
            inv_s2N = 1.0 / (sigN * sigN)
            dPhi_dmux = Phi_N * dxN * inv_s2N
            dPhi_dmuy = Phi_N * dyN * inv_s2N
            r2_N = dxN * dxN + dyN * dyN
            dPhi_dlogs = Phi_N * (r2_N * inv_s2N)
            J_mux = _per_atom_cols(dPhi_dmux, c_N)
            J_muy = _per_atom_cols(dPhi_dmuy, c_N)
            J_logsigN = _per_atom_cols(dPhi_dlogs, c_N)
            J = torch.cat(J_smooth + [J_mux, J_muy, J_logsigN], dim=1)
        else:
            sp = self.log_sig_N.exp()[None, :]       # par width
            sq = self.log_sig_N_perp.exp()[None, :]   # perp width
            inv_sp2 = 1.0 / (sp * sp)
            inv_sq2 = 1.0 / (sq * sq)
            ct = torch.cos(self.theta_N)[None, :]
            st = torch.sin(self.theta_N)[None, :]
            d_p = ct * dxN + st * dyN                # (N, K_n)
            d_q = -st * dxN + ct * dyN
            # dPhi/dmu_x = (d_p ct inv_sp2 - d_q st inv_sq2) * Phi_N
            dPhi_dmux = Phi_N * (d_p * ct * inv_sp2 - d_q * st * inv_sq2)
            dPhi_dmuy = Phi_N * (d_p * st * inv_sp2 + d_q * ct * inv_sq2)
            # dPhi/dlog_s_p = (d_p^2 inv_sp2) * Phi_N  (same shape for perp)
            dPhi_dlogsp = Phi_N * (d_p * d_p * inv_sp2)
            dPhi_dlogsq = Phi_N * (d_q * d_q * inv_sq2)
            # dPhi/dtheta = -d_p d_q (inv_sp2 - inv_sq2) * Phi_N
            dPhi_dtheta = -Phi_N * (d_p * d_q * (inv_sp2 - inv_sq2))

            J_mux = _per_atom_cols(dPhi_dmux, c_N)
            J_muy = _per_atom_cols(dPhi_dmuy, c_N)
            J_logsigP = _per_atom_cols(dPhi_dlogsp, c_N)
            J_logsigQ = _per_atom_cols(dPhi_dlogsq, c_N)
            J_theta = _per_atom_cols(dPhi_dtheta, c_N)
            J = torch.cat(J_smooth + [J_mux, J_muy,
                                       J_logsigP, J_logsigQ, J_theta],
                          dim=1)
        return J, r, U_pred


# ----------------------------------------------------------------------
# Adam VarPro outer loop (for IC cold-start; basin-hopping friendly)
# ----------------------------------------------------------------------

def fit_varpro_adam(model, U_true, U_true_np, iters,
                    lr_mu=1e-3, lr_sig=5e-3, lr_theta=None,
                    betas=(0.9, 0.999), eps=1e-8,
                    cosine_decay=True, lr_min_factor=1e-2,
                    rebirth_every=0, rebirth_frac=0.15,
                    rebirth_until_frac=0.6,
                    rcond=1e-12, ridge=0.0,
                    verbose=True,
                    progress=False, progress_desc='VP-Adam',
                    progress_metric_every=25):
    """Adam-on-theta VarPro: at each iter, eliminate c via solve_linear,
    take an Adam step on the Kaufman gradient grad_theta = J^T r.

    Designed for IC cold-start where LM is fragile (random atom layout,
    saddle-rich landscape).  After this, follow up with LM for polish.

    Per-parameter learning rates: lr_mu for (mu_Nx, mu_Ny); lr_sig for
    log_sig_S, log_sig_N (log-space, so 5e-3 ~ 0.5% width per step).
    """
    t0 = time.time()
    # build per-element lr vector matching theta packing:
    # iso : [log_sig_S(1), mu_Nx(K_n), mu_Ny(K_n), log_sig_N(K_n)]
    # aniso: [..., log_sig_N(K_n), log_sig_N_perp(K_n), theta_N(K_n)]
    K_n = model.K_n
    K_s = model.K_s
    if lr_theta is None:
        lr_theta = lr_sig
    # smooth pool lr's (match the variable-layout pack order)
    lr_parts = []
    if model.smooth_per_atom:
        lr_parts.append(torch.full((K_s,), lr_sig, device=device))
    else:
        lr_parts.append(torch.full((1,), lr_sig, device=device))
    if model.smooth_anisotropic:
        lr_parts.append(torch.full((K_s,), lr_sig, device=device))      # log_sig_S_perp
        lr_parts.append(torch.full((K_s,), lr_theta, device=device))    # theta_S
    # sharp pool lr's
    lr_parts += [
        torch.full((K_n,), lr_mu, device=device),
        torch.full((K_n,), lr_mu, device=device),
        torch.full((K_n,), lr_sig, device=device),
    ]
    if getattr(model, 'anisotropic', False):
        lr_parts += [
            torch.full((K_n,), lr_sig, device=device),     # log_sig_N_perp
            torch.full((K_n,), lr_theta, device=device),    # theta_N
        ]
    lr_vec = torch.cat(lr_parts)
    P_nl = lr_vec.numel()
    m = torch.zeros(P_nl, device=device)
    v = torch.zeros(P_nl, device=device)
    b1, b2 = betas

    J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
    loss = 0.5 * (r * r).sum().item()
    history = [loss]
    if verbose:
        print(f"  Adam it  -1 loss={loss:.3e} (init)")

    rebirth_stop_it = int(rebirth_until_frac * iters)
    pbar = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=iters, desc=progress_desc, unit='it',
                         dynamic_ncols=True, mininterval=0.5,
                         smoothing=0.1)
        except ImportError:
            pbar = None

    last_metric = [None]
    it_idx = [0]

    def _update_postfix(note=None):
        if pbar is None: return
        d = dict(loss=f"{loss:.3e}")
        if (progress_metric_every > 0
                and (it_idx[0] % progress_metric_every == 0
                     or last_metric[0] is None)):
            U_pred_np = U_pred.detach().cpu().numpy()
            rL2, rLi, rL1 = metrics_full(U_pred_np, U_true_np)
            last_metric[0] = (rL1[0], rL2[0], rL2[3])
        if last_metric[0] is not None:
            l1r, l2r, l2E = last_metric[0]
            d['L1_r'] = f"{l1r:.2e}"
            d['L2_r'] = f"{l2r:.2e}"
            d['L2_E'] = f"{l2E:.2e}"
        if note: d['note'] = note
        pbar.set_postfix(d)

    _update_postfix()
    for it in range(iters):
        it_idx[0] = it
        if (rebirth_every > 0 and it > 0
                and it % rebirth_every == 0
                and it < rebirth_stop_it):
            rebirth_weak_atoms(model, U_true, frac=rebirth_frac,
                               verbose=verbose)
            # reset Adam state for moved atoms is messy across re-packings;
            # simplest robust thing: reset both moments globally.  Adam
            # warms back up in O(1/(1-b1)) ~ 10 iters which is cheap.
            m.zero_(); v.zero_()
            J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
            loss = 0.5 * (r * r).sum().item()
            if pbar is not None:
                _update_postfix(note='rebirth')
                pbar.update(1)
                history.append(loss)
                continue

        # Kaufman gradient: g = J^T r
        g = J.t() @ r                                    # (P_nl,)

        # Adam update
        t_step = it + 1
        m.mul_(b1).add_(g, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
        m_hat = m / (1.0 - b1 ** t_step)
        v_hat = v / (1.0 - b2 ** t_step)
        # per-element lr (optionally cosine-decayed)
        if cosine_decay:
            frac = it / max(1, iters - 1)
            scale = lr_min_factor + 0.5 * (1.0 - lr_min_factor) * (
                1.0 + math.cos(math.pi * frac))
        else:
            scale = 1.0
        dtheta = -scale * lr_vec * m_hat / (v_hat.sqrt() + eps)

        # apply
        theta_old = model.pack().clone()
        theta_new = theta_old + dtheta
        model.unpack(theta_new)
        model.project_()

        # refresh c and J at the new theta
        J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
        loss = 0.5 * (r * r).sum().item()
        history.append(loss)
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"  Adam it {it:4d} loss={loss:.3e} scale={scale:.2e}")
        if pbar is not None:
            _update_postfix()
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    U_pred_np = U_pred.detach().cpu().numpy()
    relL2, relLi = metrics(U_pred_np, U_true_np)
    return dict(
        U_pred=U_pred_np, relL2=relL2, relLi=relLi, loss=loss,
        elapsed=time.time() - t0, history=history)


# ----------------------------------------------------------------------
# Levenberg-Marquardt VarPro outer loop
# ----------------------------------------------------------------------

def rebirth_weak_atoms(model, U_true, frac=0.15, verbose=True):
    """Move the weakest sharp atoms (smallest sum_f |c_N[f,l]|) to the
    cells where the current residual magnitude is largest (across fields,
    weighted by |grad rho|).  Reset their log_sigma to the box midpoint
    so they have a fresh chance.
    """
    U_pred, _ = model.solve_linear(U_true)
    res = (U_pred - U_true)                         # (N, 4)
    res_mag = res.abs().sum(dim=1)                  # (N,)
    # forbid picking cells too close to an existing sharp atom we are KEEPING
    K_n = model.K_n
    c_N = model.c[:, model.K_s:model.K_s + model.K_n]      # (4, K_n)
    strength = c_N.abs().sum(dim=0)                        # (K_n,)
    n_rebirth = max(1, int(frac * K_n))
    weak_idx = torch.argsort(strength)[:n_rebirth]
    # candidate cells = sort by residual descending
    cand_order = torch.argsort(res_mag, descending=True)
    # min spacing in pixels: ~1.5 grid_dx; in real units handled by hard
    # rejection radius using existing kept atoms
    keep_mask = torch.ones(model.K_n, dtype=torch.bool, device=device)
    keep_mask[weak_idx] = False
    kept_mux = model.mu_Nx[keep_mask]; kept_muy = model.mu_Ny[keep_mask]
    grid_dx = (model.x.max() - model.x.min()).item() / math.sqrt(model.x.numel())
    min_d2 = (2.0 * grid_dx) ** 2
    new_mux = []; new_muy = []
    for ci in cand_order.tolist():
        cx = model.x[ci].item(); cy = model.y[ci].item()
        ok = True
        for px, py in zip(kept_mux.tolist(), kept_muy.tolist()):
            if (cx - px) ** 2 + (cy - py) ** 2 < min_d2:
                ok = False; break
        if ok:
            for px, py in zip(new_mux, new_muy):
                if (cx - px) ** 2 + (cy - py) ** 2 < min_d2:
                    ok = False; break
        if ok:
            new_mux.append(cx); new_muy.append(cy)
            if len(new_mux) >= n_rebirth: break
    if len(new_mux) < n_rebirth:
        # pad with extras
        for ci in cand_order.tolist():
            if len(new_mux) >= n_rebirth: break
            new_mux.append(model.x[ci].item())
            new_muy.append(model.y[ci].item())
    new_mux = torch.tensor(new_mux, device=device)
    new_muy = torch.tensor(new_muy, device=device)
    # midpoint init in log space
    log_sig_mid = 0.5 * (model.log_sig_N_box[0] + model.log_sig_N_box[1])
    with torch.no_grad():
        model.mu_Nx[weak_idx] = new_mux
        model.mu_Ny[weak_idx] = new_muy
        model.log_sig_N[weak_idx] = log_sig_mid
        if getattr(model, 'anisotropic', False):
            model.log_sig_N_perp[weak_idx] = log_sig_mid
            model.theta_N[weak_idx] = 0.0
    model.project_()
    if verbose:
        print(f"    [rebirth] moved {n_rebirth}/{K_n} weak sharp atoms "
              f"to high-residual cells")


def fit_varpro(model, U_true, U_true_np, iters,
               lam0=1e-3, lam_min=1e-12, lam_max=1e8, nu=3.0,
               rcond=1e-12, ridge=0.0, verbose=True,
               rebirth_every=0, rebirth_frac=0.15,
               rebirth_until_frac=0.6,
               ckpt_every=0, ckpt_cb=None,
               progress=False, progress_desc='VP',
               progress_metric_every=10):
    lam = lam0
    t0 = time.time()
    J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
    loss = 0.5 * (r * r).sum().item()
    if verbose:
        print(f"  VP it  -1 loss={loss:.3e} (init, c just solved)")
    history = [loss]
    rebirth_stop_it = int(rebirth_until_frac * iters)
    pbar = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=iters, desc=progress_desc, unit='it',
                         dynamic_ncols=True, mininterval=0.5,
                         smoothing=0.1)
        except ImportError:
            pbar = None

    def _update_postfix(note=None):
        if pbar is None: return
        d = dict(loss=f"{loss:.3e}", lam=f"{lam:.1e}")
        if (progress_metric_every > 0
                and (it_idx[0] % progress_metric_every == 0
                     or last_metric[0] is None)):
            U_pred_np = U_pred.detach().cpu().numpy()
            rL2, rLi, rL1 = metrics_full(U_pred_np, U_true_np)
            last_metric[0] = (rL1[0], rL2[0], rL2[3])
        if last_metric[0] is not None:
            l1r, l2r, l2E = last_metric[0]
            d['L1_r'] = f"{l1r:.2e}"
            d['L2_r'] = f"{l2r:.2e}"
            d['L2_E'] = f"{l2E:.2e}"
        if note: d['note'] = note
        pbar.set_postfix(d)
    it_idx = [0]
    last_metric = [None]
    _update_postfix()
    for it in range(iters):
        it_idx[0] = it
        if (rebirth_every > 0 and it > 0
                and it % rebirth_every == 0
                and it < rebirth_stop_it):
            rebirth_weak_atoms(model, U_true, frac=rebirth_frac,
                               verbose=verbose)
            J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
            loss = 0.5 * (r * r).sum().item()
            lam = max(lam, lam0)   # reset damping after geometry jump
        JtJ = J.t() @ J
        Jtr = J.t() @ r
        diag = JtJ.diagonal().clone().clamp_min(1e-30)
        theta_old = model.pack().clone()
        accepted = False
        for _trial in range(10):
            M = JtJ + lam * torch.diag(diag)
            try:
                dtheta = torch.linalg.solve(M, -Jtr)
            except Exception:
                dtheta = lstsq_cpu(M, -Jtr, rcond=rcond).squeeze(1)
            theta_new = theta_old + dtheta
            model.unpack(theta_new)
            model.project_()
            U_pred_new, _ = model.solve_linear(U_true, ridge=ridge, rcond=rcond)
            r_new = (U_pred_new - U_true).t().reshape(-1)
            loss_new = 0.5 * (r_new * r_new).sum().item()
            if loss_new < loss:
                lam = max(lam / nu, lam_min)
                loss = loss_new; r = r_new
                accepted = True
                break
            lam = min(lam * nu, lam_max)
            if lam >= lam_max:
                break
        if not accepted:
            # revert
            model.unpack(theta_old); model.project_()
            model.solve_linear(U_true, ridge=ridge, rcond=rcond)
            if rebirth_every > 0 and it < rebirth_stop_it:
                if verbose:
                    print(f"  VP it {it:3d} stalled (lam={lam:.1e}); rebirth")
                rebirth_weak_atoms(model, U_true, frac=rebirth_frac,
                                   verbose=verbose)
                J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
                loss = 0.5 * (r * r).sum().item()
                lam = lam0
                history.append(loss)
                if pbar is not None:
                    _update_postfix(note='rebirth')
                    pbar.update(1)
                continue
            if verbose:
                print(f"  VP it {it:3d} stalled (lam={lam:.1e}); stop")
            break
        # refresh Jacobian at the new theta
        J, r, U_pred = model.jacobian_and_residual(U_true, ridge=ridge, rcond=rcond)
        loss = 0.5 * (r * r).sum().item()
        history.append(loss)
        if verbose and (it % 5 == 0 or it == iters - 1):
            print(f"  VP it {it:3d} loss={loss:.3e} lam={lam:.1e}")
        if pbar is not None:
            _update_postfix()
            pbar.update(1)
        if ckpt_cb is not None and ckpt_every > 0 and (it + 1) % ckpt_every == 0:
            ckpt_cb(it, U_pred, history)
    if pbar is not None:
        pbar.close()
    U_pred_np = U_pred.detach().cpu().numpy()
    relL2, relLi = metrics(U_pred_np, U_true_np)
    return dict(
        U_pred=U_pred_np, relL2=relL2, relLi=relLi, loss=loss,
        elapsed=time.time() - t0, history=history)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--K_s', type=int, default=64)
    ap.add_argument('--K_n', type=int, default=48)
    ap.add_argument('--iters', type=int, default=80)
    ap.add_argument('--sigma_S_init_factor', type=float, default=2.0,
                    help="sigma_S init = factor * smooth-lattice spacing")
    ap.add_argument('--sigma_N_init_factor', type=float, default=1.5,
                    help="sigma_N init (per atom) = factor * WENO mesh dx")
    ap.add_argument('--sigma_S_box', type=str, default='0.04,2.0')
    ap.add_argument('--sigma_N_box', type=str, default='5e-3,0.15')
    ap.add_argument('--out_dir', type=str, default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rebirth_every', type=int, default=20)
    ap.add_argument('--rebirth_frac', type=float, default=0.15)
    ap.add_argument('--rebirth_until_frac', type=float, default=0.6)
    ap.add_argument('--ckpt_every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = np.load(args.npz)
    U_np = d['U']; Xn = d['X']; Yn = d['Y']
    t_snap = float(d['t'])
    Nx, Ny, _ = U_np.shape
    grid_dx = LX / Nx
    print(f"[load] {args.npz}  Nx={Nx} Ny={Ny} t={t_snap:.4f}  "
          f"grid_dx={grid_dx:.4g}")

    x = torch.as_tensor(Xn.reshape(-1), device=device)
    y = torch.as_tensor(Yn.reshape(-1), device=device)
    U_true = torch.as_tensor(U_np.reshape(-1, 4), device=device)
    U_true_np = U_np.reshape(-1, 4)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.npz)), 'fit_varpro')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}\n")

    sigS_box = tuple(float(s) for s in args.sigma_S_box.split(','))
    sigN_box = tuple(float(s) for s in args.sigma_N_box.split(','))

    # ---- centres ----
    Nx_g, Ny_g, K_s_eff, mu_Sx_np, mu_Sy_np = lattice_centres(
        args.K_s, LX, LY)
    lat_dx = LX / Nx_g
    sig_S_init = args.sigma_S_init_factor * lat_dx
    sig_N_init = args.sigma_N_init_factor * grid_dx
    mu_Nx_np, mu_Ny_np = grad_rho_topk(U_np, Xn, Yn, args.K_n)
    print(f"smooth pool: K_s={K_s_eff} ({Nx_g}x{Ny_g})  "
          f"lat_dx={lat_dx:.3g}  sig_S_init={sig_S_init:.3g}")
    print(f"sharp  pool: K_n={args.K_n}  sig_N_init={sig_N_init:.3g}  "
          f"sigN_box=[{sigN_box[0]:.2g},{sigN_box[1]:.2g}]")

    model = BlendedRBF(
        x, y,
        mu_S=(mu_Sx_np, mu_Sy_np),
        mu_N=(mu_Nx_np, mu_Ny_np),
        log_sig_S_init=math.log(sig_S_init),
        log_sig_N_init=math.log(sig_N_init),
        sigma_S_box=sigS_box, sigma_N_box=sigN_box)

    P_nl = 1 + 3 * args.K_n
    P_lin = 4 * (K_s_eff + args.K_n + 1)
    P_total = P_nl + P_lin
    print(f"DOF: P_nl={P_nl}  P_lin={P_lin}  P_total={P_total}\n")

    print(f"-- VarPro LM, iters={args.iters}")

    def ckpt_cb(it, U_pred_t, history):
        U_pred_np = U_pred_t.detach().cpu().numpy()
        rL2, rLi = metrics(U_pred_np, U_true_np)
        with open(os.path.join(out_dir, 'fit_summary.txt'), 'w') as f:
            f.write(
                f"# VarPro blended-RBF LM CKPT iter={it+1}/{args.iters}\n"
                f"# K_s={K_s_eff} K_n={args.K_n}  "
                f"P_nl={P_nl} P_lin={P_lin} P_total={P_total}\n"
                f"# loss={history[-1]:.3e}\n"
                f"# relL2_rho={rL2[0]:.3e} relL2_ru={rL2[1]:.3e} "
                f"relL2_rv={rL2[2]:.3e} relL2_E={rL2[3]:.3e}\n"
                f"# relLinf_max={rLi.max():.3e}\n")
        np.savez(os.path.join(out_dir, 'ckpt.npz'),
                 U_pred=U_pred_np, history=np.asarray(history),
                 mu_Nx=model.mu_Nx.cpu().numpy(),
                 mu_Ny=model.mu_Ny.cpu().numpy(),
                 log_sig_N=model.log_sig_N.cpu().numpy(),
                 log_sig_S=model.log_sig_S.cpu().numpy(),
                 c=model.c.cpu().numpy(),
                 mu_Sx=model.mu_Sx.cpu().numpy(),
                 mu_Sy=model.mu_Sy.cpu().numpy())

    result = fit_varpro(model, U_true, U_true_np, args.iters,
                        rebirth_every=args.rebirth_every,
                        rebirth_frac=args.rebirth_frac,
                        rebirth_until_frac=args.rebirth_until_frac,
                        ckpt_every=args.ckpt_every, ckpt_cb=ckpt_cb)
    print(f"\n[done] loss={result['loss']:.3e}  "
          f"relL2=({result['relL2'][0]:.2e},{result['relL2'][1]:.2e},"
          f"{result['relL2'][2]:.2e},{result['relL2'][3]:.2e})  "
          f"rL_inf={result['relLi'].max():.2e}  "
          f"t={result['elapsed']:.1f}s")

    # ---- summary text ----
    lines = []
    lines.append(f"# VarPro blended-RBF LM fit of {args.npz}")
    lines.append(f"# t={t_snap:.4f}  Nx={Nx} Ny={Ny}")
    lines.append(f"# K_s={K_s_eff} K_n={args.K_n}  "
                 f"P_nl={P_nl}  P_lin={P_lin}  P_total={P_total}")
    lines.append(f"# iters={args.iters}  "
                 f"sigS_box={sigS_box}  sigN_box={sigN_box}")
    lines.append(f"# final  relL2_rho={result['relL2'][0]:.3e}  "
                 f"relL2_ru={result['relL2'][1]:.3e}  "
                 f"relL2_rv={result['relL2'][2]:.3e}  "
                 f"relL2_E={result['relL2'][3]:.3e}")
    lines.append(f"# final  relLinf_max={result['relLi'].max():.3e}  "
                 f"elapsed={result['elapsed']:.2f}s")
    txt = '\n'.join(lines)
    with open(os.path.join(out_dir, 'fit_summary.txt'), 'w') as f:
        f.write(txt + '\n')

    # ---- per-field plot ----
    U_pred = result['U_pred'].reshape(Nx, Ny, 4)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for f in range(4):
        u_t = U_np[..., f]; u_p = U_pred[..., f]; er = u_p - u_t
        vmin, vmax = u_t.min(), u_t.max()
        im0 = axes[f, 0].imshow(u_t.T, origin='lower',
                                extent=[0, LX, 0, LY],
                                aspect='auto', cmap='turbo',
                                vmin=vmin, vmax=vmax)
        axes[f, 0].set_title(f"{FIELD_NAMES[f]} true (WENO)")
        plt.colorbar(im0, ax=axes[f, 0], fraction=0.025)
        im1 = axes[f, 1].imshow(u_p.T, origin='lower',
                                extent=[0, LX, 0, LY],
                                aspect='auto', cmap='turbo',
                                vmin=vmin, vmax=vmax)
        axes[f, 1].set_title(f"{FIELD_NAMES[f]} VarPro")
        plt.colorbar(im1, ax=axes[f, 1], fraction=0.025)
        emax = np.max(np.abs(er)) + 1e-30
        im2 = axes[f, 2].imshow(er.T, origin='lower',
                                extent=[0, LX, 0, LY],
                                aspect='auto', cmap='seismic',
                                vmin=-emax, vmax=emax)
        axes[f, 2].set_title(
            f"err  rL2={result['relL2'][f]:.2e}  "
            f"rLi={result['relLi'][f]:.2e}")
        plt.colorbar(im2, ax=axes[f, 2], fraction=0.025)
    plt.suptitle(
        f"VarPro blended-RBF  t={t_snap:.4f}  "
        f"K_s={K_s_eff} K_n={args.K_n}  P_nl={P_nl}  P_lin={P_lin}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fit.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- sharp-atom centre overlay on rho ----
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(U_np[..., 0].T, origin='lower',
              extent=[0, LX, 0, LY], aspect='auto', cmap='turbo')
    ax.scatter(model.mu_Nx.cpu().numpy(),
               model.mu_Ny.cpu().numpy(),
               s=15, edgecolors='white', facecolors='none', lw=0.8,
               label='sharp atoms')
    ax.scatter(model.mu_Sx.cpu().numpy(),
               model.mu_Sy.cpu().numpy(),
               s=8, c='black', marker='+', label='smooth atoms')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"atom layout  t={t_snap:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'atoms.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig2)

    # ---- convergence plot ----
    fig3, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(result['history'], '-o', ms=3)
    ax.set_xlabel('LM iteration')
    ax.set_ylabel('0.5 ||r||^2')
    ax.set_title('VarPro LM convergence')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig3)

    print(f"\n[out] -> {out_dir}")
    print(txt)


if __name__ == '__main__':
    main()
