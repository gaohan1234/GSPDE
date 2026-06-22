"""
Implicit Yoon-Jameson LU-SGS pseudo-time solver for the 2D Euler
chain-rule curvilinear RHS already implemented in `weno_curv.py`.

Method
------
We solve, per pseudo-time step,

    [ I/Δτ_ij + |∂R/∂U|_ij ] ΔU_ij  =  R_ij

with the *scalar* matrix-free Yoon–Jameson approximation:

    D_ij = 1/Δτ_ij + λ^ξ_ij / Δξ + λ^η_ij / Δη
    λ^ξ_ij = |u ξ_x + v ξ_y|_ij + a_ij * |∇ξ|_ij
    λ^η_ij = |u η_x + v η_y|_ij + a_ij * |∇η|_ij

    H^ξ(U; ξ_x, ξ_y) := ξ_x F(U) + ξ_y G(U)
    H^η(U; η_x, η_y) := η_x F(U) + η_y G(U)

    A^+_d · ΔU  ≈  (0.5/Δs) [ H_d(U+ΔU) − H_d(U) + λ_d ΔU ]
    A^-_d · ΔU  ≈  (0.5/Δs) [ H_d(U+ΔU) − H_d(U) − λ_d ΔU ]

with the donor cell's metrics, so that:

    Forward (L) sweep, i=0..Nxi−1, j=0..Neta−1:
        rhs = R_ij + A^+_ξ(U_{i-1}, ΔU*_{i-1}) + A^+_η(U_{i,j-1}, ΔU*_{i,j-1})
        ΔU*_ij = rhs / D_ij

    Backward (U) sweep, reverse order:
        rhs = − A^-_ξ(U_{i+1}, ΔU_{i+1}) − A^-_η(U_{i,j+1}, ΔU_{i,j+1})
        ΔU_ij = ΔU*_ij + rhs / D_ij

Off-grid neighbours (i<0, i≥Nxi, j<0, j≥Neta) contribute zero.
This corresponds to a Dirichlet ghost on far-field/wall — exact for free-
stream Dirichlet, mildly inconsistent at the wall but irrelevant for the
LU-SGS *preconditioner*: only the explicit residual R determines the
fixed point, not D.

CFL of 100–1000 typically converges a transonic airfoil in ~100–300
steps (≈ 50–100x reduction vs explicit SSPRK3).
"""
from __future__ import annotations
import numpy as np
import numba as nb

from naca_setup import GAMMA


# =================================================================
#  inline flux / spectral radius (numba)
# =================================================================
@nb.njit(cache=True, inline='always')
def _Hxi(U0, U1, U2, U3, sx, sy):
    rho = U0
    u = U1 / rho
    v = U2 / rho
    E = U3
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    F0 = U1
    F1 = U1 * u + p
    F2 = U1 * v
    F3 = u * (E + p)
    G0 = U2
    G1 = U2 * u
    G2 = U2 * v + p
    G3 = v * (E + p)
    return (sx * F0 + sy * G0,
            sx * F1 + sy * G1,
            sx * F2 + sy * G2,
            sx * F3 + sy * G3)


# =================================================================
#  forward / backward sweeps  (in place on dU)
# =================================================================
@nb.njit(cache=True, parallel=False, fastmath=True)
def _lusgs_sweeps(U, R, dU,
                  xi_x, xi_y, eta_x, eta_y,
                  lam_xi, lam_eta,
                  dxi, deta, dtau):
    """
    Single forward + backward symmetric Gauss–Seidel sweep.
    Modifies dU in place.  Initial dU may be zero.
    """
    Nxi = U.shape[0]
    Neta = U.shape[1]

    inv_dxi = 1.0 / dxi
    inv_deta = 1.0 / deta

    # ---------------- forward sweep ----------------
    for i in range(Nxi):
        for j in range(Neta):
            r0 = R[i, j, 0]
            r1 = R[i, j, 1]
            r2 = R[i, j, 2]
            r3 = R[i, j, 3]

            # contribution from (i-1, j) :  A^+_xi  ΔU*_{i-1,j}
            if i > 0:
                du0 = dU[i - 1, j, 0]; du1 = dU[i - 1, j, 1]
                du2 = dU[i - 1, j, 2]; du3 = dU[i - 1, j, 3]
                u0 = U[i - 1, j, 0]; u1 = U[i - 1, j, 1]
                u2 = U[i - 1, j, 2]; u3 = U[i - 1, j, 3]
                sx = xi_x[i - 1, j]; sy = xi_y[i - 1, j]
                lam = lam_xi[i - 1, j]
                Ha0, Ha1, Ha2, Ha3 = _Hxi(u0 + du0, u1 + du1, u2 + du2, u3 + du3, sx, sy)
                Hb0, Hb1, Hb2, Hb3 = _Hxi(u0, u1, u2, u3, sx, sy)
                r0 += 0.5 * inv_dxi * (Ha0 - Hb0 + lam * du0)
                r1 += 0.5 * inv_dxi * (Ha1 - Hb1 + lam * du1)
                r2 += 0.5 * inv_dxi * (Ha2 - Hb2 + lam * du2)
                r3 += 0.5 * inv_dxi * (Ha3 - Hb3 + lam * du3)

            # contribution from (i, j-1) :  A^+_eta ΔU*_{i,j-1}
            if j > 0:
                du0 = dU[i, j - 1, 0]; du1 = dU[i, j - 1, 1]
                du2 = dU[i, j - 1, 2]; du3 = dU[i, j - 1, 3]
                u0 = U[i, j - 1, 0]; u1 = U[i, j - 1, 1]
                u2 = U[i, j - 1, 2]; u3 = U[i, j - 1, 3]
                sx = eta_x[i, j - 1]; sy = eta_y[i, j - 1]
                lam = lam_eta[i, j - 1]
                Ha0, Ha1, Ha2, Ha3 = _Hxi(u0 + du0, u1 + du1, u2 + du2, u3 + du3, sx, sy)
                Hb0, Hb1, Hb2, Hb3 = _Hxi(u0, u1, u2, u3, sx, sy)
                r0 += 0.5 * inv_deta * (Ha0 - Hb0 + lam * du0)
                r1 += 0.5 * inv_deta * (Ha1 - Hb1 + lam * du1)
                r2 += 0.5 * inv_deta * (Ha2 - Hb2 + lam * du2)
                r3 += 0.5 * inv_deta * (Ha3 - Hb3 + lam * du3)

            D = 1.0 / dtau[i, j] + lam_xi[i, j] * inv_dxi + lam_eta[i, j] * inv_deta
            dU[i, j, 0] = r0 / D
            dU[i, j, 1] = r1 / D
            dU[i, j, 2] = r2 / D
            dU[i, j, 3] = r3 / D

    # ---------------- backward sweep ----------------
    for i in range(Nxi - 1, -1, -1):
        for j in range(Neta - 1, -1, -1):
            r0 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0

            # contribution from (i+1, j) :  − A^-_xi  ΔU_{i+1,j}
            if i < Nxi - 1:
                du0 = dU[i + 1, j, 0]; du1 = dU[i + 1, j, 1]
                du2 = dU[i + 1, j, 2]; du3 = dU[i + 1, j, 3]
                u0 = U[i + 1, j, 0]; u1 = U[i + 1, j, 1]
                u2 = U[i + 1, j, 2]; u3 = U[i + 1, j, 3]
                sx = xi_x[i + 1, j]; sy = xi_y[i + 1, j]
                lam = lam_xi[i + 1, j]
                Ha0, Ha1, Ha2, Ha3 = _Hxi(u0 + du0, u1 + du1, u2 + du2, u3 + du3, sx, sy)
                Hb0, Hb1, Hb2, Hb3 = _Hxi(u0, u1, u2, u3, sx, sy)
                # −A^- ΔU = −0.5(H(U+ΔU) − H(U) − λ ΔU) = −0.5(H+−H−) + 0.5 λ ΔU
                r0 -= 0.5 * inv_dxi * (Ha0 - Hb0 - lam * du0)
                r1 -= 0.5 * inv_dxi * (Ha1 - Hb1 - lam * du1)
                r2 -= 0.5 * inv_dxi * (Ha2 - Hb2 - lam * du2)
                r3 -= 0.5 * inv_dxi * (Ha3 - Hb3 - lam * du3)

            # contribution from (i, j+1) :  − A^-_eta ΔU_{i,j+1}
            if j < Neta - 1:
                du0 = dU[i, j + 1, 0]; du1 = dU[i, j + 1, 1]
                du2 = dU[i, j + 1, 2]; du3 = dU[i, j + 1, 3]
                u0 = U[i, j + 1, 0]; u1 = U[i, j + 1, 1]
                u2 = U[i, j + 1, 2]; u3 = U[i, j + 1, 3]
                sx = eta_x[i, j + 1]; sy = eta_y[i, j + 1]
                lam = lam_eta[i, j + 1]
                Ha0, Ha1, Ha2, Ha3 = _Hxi(u0 + du0, u1 + du1, u2 + du2, u3 + du3, sx, sy)
                Hb0, Hb1, Hb2, Hb3 = _Hxi(u0, u1, u2, u3, sx, sy)
                r0 -= 0.5 * inv_deta * (Ha0 - Hb0 - lam * du0)
                r1 -= 0.5 * inv_deta * (Ha1 - Hb1 - lam * du1)
                r2 -= 0.5 * inv_deta * (Ha2 - Hb2 - lam * du2)
                r3 -= 0.5 * inv_deta * (Ha3 - Hb3 - lam * du3)

            D = 1.0 / dtau[i, j] + lam_xi[i, j] * inv_dxi + lam_eta[i, j] * inv_deta
            dU[i, j, 0] += r0 / D
            dU[i, j, 1] += r1 / D
            dU[i, j, 2] += r2 / D
            dU[i, j, 3] += r3 / D


# =================================================================
#  driver-side helpers
# =================================================================
def spectral_radii(cur, U):
    """Per-node λ^ξ, λ^η.  Uses interior U only."""
    rho  = U[..., 0]
    u    = U[..., 1] / rho
    v    = U[..., 2] / rho
    E    = U[..., 3]
    p    = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    a    = np.sqrt(np.maximum(GAMMA * p / rho, 1e-12))
    u_xi  = cur.xi_x  * u + cur.xi_y  * v
    u_eta = cur.eta_x * u + cur.eta_y * v
    lam_xi  = np.abs(u_xi)  + a * cur.grad_xi_mag
    lam_eta = np.abs(u_eta) + a * cur.grad_eta_mag
    return lam_xi, lam_eta


def lusgs_step(cur, U, cfl_implicit, n_sweeps=1):
    """
    One LU-SGS implicit pseudo-time update.

    Parameters
    ----------
    cur : CurvilinearMesh
    U   : (Nxi, Neta, 4) conservative state
    cfl_implicit : Δτ_ij = cfl_implicit * dt_local_ij
    n_sweeps : number of symmetric L+U sweep pairs (1–3 typical)

    Returns
    -------
    U_new : updated state
    rms_R_rho_rel, max_R_rho : residual norms BEFORE the update
    """
    R, _, dt_local = cur.rhs(U)
    lam_xi, lam_eta = spectral_radii(cur, U)

    dtau = np.maximum(cfl_implicit * dt_local, 1e-30)

    dU = np.zeros_like(U)
    for _ in range(n_sweeps):
        _lusgs_sweeps(U, R, dU,
                      cur.xi_x, cur.xi_y, cur.eta_x, cur.eta_y,
                      lam_xi, lam_eta,
                      cur.d_xi, cur.d_eta, dtau)

    U_new = U + dU

    # ---- positivity + trust-region guard ----
    # Halve dU if any cell would have rho<=0 or p<=0, OR if |dU|/|U| exceeds
    # 30% on density / energy (limits how aggressively LU-SGS can punch into
    # a shock).  Up to 20 halvings; after that, reject the step.
    rho0 = U[..., 0]
    E0   = U[..., 3]
    for _ in range(20):
        rho_n = U_new[..., 0]
        u_n   = U_new[..., 1] / np.maximum(rho_n, 1e-30)
        v_n   = U_new[..., 2] / np.maximum(rho_n, 1e-30)
        E_n   = U_new[..., 3]
        p_n   = (GAMMA - 1.0) * (E_n - 0.5 * rho_n * (u_n * u_n + v_n * v_n))
        bad_pos = (rho_n <= 1e-6) | (p_n <= 1e-6) \
                | (~np.isfinite(rho_n)) | (~np.isfinite(p_n))
        rel_rho = np.abs(rho_n - rho0) / np.maximum(np.abs(rho0), 1e-12)
        rel_E   = np.abs(E_n   - E0  ) / np.maximum(np.abs(E0  ), 1e-12)
        bad_big = (rel_rho > 0.3) | (rel_E > 0.3)
        if not (bad_pos.any() or bad_big.any()):
            break
        dU = 0.5 * dU
        U_new = U + dU
    else:
        # could not find any admissible step; reject (return original U)
        U_new = U.copy()

    # diagnostics on R
    rho = U[..., 0]
    R_rho = R[..., 0]
    rms_R_rho_rel = float(np.sqrt(np.mean(R_rho ** 2)) / max(np.mean(rho), 1e-30))
    max_R_rho = float(np.abs(R_rho).max())
    return U_new, rms_R_rho_rel, max_R_rho


if __name__ == '__main__':
    # smoke test: one implicit step on freestream should be ~zero update
    from weno_curv import CurvilinearMesh
    cur = CurvilinearMesh('mapping_c.npz')
    U = cur.freestream()
    U_new, rms, mx = lusgs_step(cur, U, cfl_implicit=100.0)
    print(f'freestream test: rms_R_rho_rel={rms:.3e}  max_R_rho={mx:.3e}')
    print(f'|ΔU|_max = {np.abs(U_new - U).max():.3e}  (should be tiny)')
