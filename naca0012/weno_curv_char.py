"""
Characteristic-wise WENO5 RHS for 2D Euler in chain-rule curvilinear form.

Same chain rule as `weno_curv.py`:

    U_t = - xi_x F_xi - eta_x F_eta - xi_y G_xi - eta_y G_eta

Each scalar directional derivative (F_xi, G_xi, F_eta, G_eta) is computed
by characteristic-wise WENO5 with Roe-averaged face eigenvectors of the
relevant Cartesian Jacobian:

    F_xi, F_eta  : use eigensystem of A_x = dF/dU
    G_xi, G_eta  : use eigensystem of A_y = dG/dU

Per face procedure (1D in the sweep direction, vectorised across the
orthogonal direction):

    1. Roe-average UL, UR  ->  (u, v, H, a)
    2. Form L_face, R_face (4x4) at the Roe average
    3. Project a 6-cell stencil of flux and conservative state into
       characteristic space:  f_k = L . F,  w_k = L . U
    4. Component-wise local LF split, alpha_k = max |lambda_k| over the
       stencil:  fp_k = 0.5 (f_k + alpha_k w_k), fm_k = 0.5 (f_k - alpha_k w_k)
    5. WENO5 reconstruct fp from the left, fm from the right -> f_face_k
    6. Project back:  F_face = R . f_face

Free-stream preservation: characteristic WENO of constant data is constant,
so the directional derivative of constant F is exactly zero, regardless of
the metrics.  The chain rule then gives R = 0 on a uniform state.

Same boundary treatment as `weno_curv.py` (reuses pad_state from the
existing CurvilinearMesh).
"""
from __future__ import annotations
import numpy as np

from naca_setup import GAMMA
from weno_curv import (CurvilinearMesh, NG, EPS,
                       _flux_x, _flux_y, _prim,
                       _weno5_left, _weno5_right,
                       _weno7_left, _weno7_right)


# Face-flux mode for characteristic WENO:
#   True  -> Roe upwind with Harten entropy fix (low dissipation, may carbuncle)
#   False -> global per-family Lax-Friedrichs   (most robust)
USE_ROE = True

# Reconstruction order:
#   False -> WENO5-JS (5-cell stencil, needs NG>=3)
#   True  -> WENO7-JS (7-cell stencil, needs NG>=4); less dissipative at the
#            shock, recovers more of the pre-shock Mach plateau.  Only wired
#            into the Roe branch.
USE_WENO7 = False

# Harten entropy-fix floors as a fraction of the local spectral radius.
#   EPS_AC -> acoustic families (vn-a, vn+a): big floor kills the normal-shock
#             carbuncle.
#   EPS_LI -> linear families (entropy, shear, eigenvalue vn~0): floor that
#             damps the transverse odd-even (sawtooth) mode in regions where
#             the flow is grid-aligned (vn~0) and the Roe contact dissipation
#             vanishes.  Module-level so it can be swept at runtime.
#             0.25 is the threshold that fully cures the eta-direction sawtooth
#             above the LE supersonic pocket on the clean (mapping_c2) mesh:
#             with cfl 0.5->5 / n_sweeps=5 the solve converges to rms~5.7e-5,
#             Mmax 1.335 (Zahr reference 1.33) and stays put through 2500 iters.
#             0.20 only DELAYS the blowup (converges to rms 3e-3 by it 1200 then
#             diverges ~it 1350); <=0.15 blows up fast.  0.30/0.40 also converge
#             but add more dissipation (Mmax 1.332/1.327).  0.05-0.15 was tuned
#             on the OLD cusped mesh and is invalid.
EPS_AC = 0.50
EPS_LI = 0.25


# =================================================================
#  Roe average and eigenvectors of A_x, A_y
# =================================================================
def _roe_uvHa(UL, UR):
    """Roe-averaged (u, v, H, a) for 2D Euler.  Arrays broadcastable."""
    rhoL = UL[..., 0]
    uL   = UL[..., 1] / rhoL
    vL   = UL[..., 2] / rhoL
    EL   = UL[..., 3]
    pL   = (GAMMA - 1.0) * (EL - 0.5 * rhoL * (uL * uL + vL * vL))
    HL   = (EL + pL) / rhoL

    rhoR = UR[..., 0]
    uR   = UR[..., 1] / rhoR
    vR   = UR[..., 2] / rhoR
    ER   = UR[..., 3]
    pR   = (GAMMA - 1.0) * (ER - 0.5 * rhoR * (uR * uR + vR * vR))
    HR   = (ER + pR) / rhoR

    srL = np.sqrt(rhoL)
    srR = np.sqrt(rhoR)
    den = srL + srR
    u = (srL * uL + srR * uR) / den
    v = (srL * vL + srR * vR) / den
    H = (srL * HL + srR * HR) / den
    a2 = (GAMMA - 1.0) * (H - 0.5 * (u * u + v * v))
    a = np.sqrt(np.maximum(a2, 1e-12))
    return u, v, H, a


def _eig_x(u, v, H, a):
    """
    Left and right eigenvectors of A_x = dF/dU for 2D Euler.
    Eigenvalues: (u-a, u, u, u+a)

    Returns L, R with shape (..., 4, 4) where the FIRST trailing axis
    indexes the row (characteristic) and the LAST is the conservative
    component, i.e. L @ R = I_4 broadcasted.
    """
    shp = u.shape
    g1 = GAMMA - 1.0
    q2 = u * u + v * v
    a2 = a * a

    R = np.zeros(shp + (4, 4), dtype=u.dtype)
    # columns of R
    # r_1 (u-a wave)
    R[..., 0, 0] = 1.0
    R[..., 1, 0] = u - a
    R[..., 2, 0] = v
    R[..., 3, 0] = H - u * a
    # r_2 (u, entropy)
    R[..., 0, 1] = 1.0
    R[..., 1, 1] = u
    R[..., 2, 1] = v
    R[..., 3, 1] = 0.5 * q2
    # r_3 (u, shear-v)
    R[..., 0, 2] = 0.0
    R[..., 1, 2] = 0.0
    R[..., 2, 2] = 1.0
    R[..., 3, 2] = v
    # r_4 (u+a wave)
    R[..., 0, 3] = 1.0
    R[..., 1, 3] = u + a
    R[..., 2, 3] = v
    R[..., 3, 3] = H + u * a

    L = np.zeros(shp + (4, 4), dtype=u.dtype)
    inv2a2 = 0.5 / a2
    inv_a2 = 1.0 / a2
    # row 1 (acoustic -)
    L[..., 0, 0] = (g1 * q2 * 0.5 + u * a) * inv2a2
    L[..., 0, 1] = (-g1 * u - a) * inv2a2
    L[..., 0, 2] = (-g1 * v) * inv2a2
    L[..., 0, 3] = g1 * inv2a2
    # row 2 (entropy)
    L[..., 1, 0] = (a2 - g1 * q2 * 0.5) * inv_a2
    L[..., 1, 1] = (g1 * u) * inv_a2
    L[..., 1, 2] = (g1 * v) * inv_a2
    L[..., 1, 3] = -g1 * inv_a2
    # row 3 (shear-v)
    L[..., 2, 0] = -v
    L[..., 2, 1] = 0.0
    L[..., 2, 2] = 1.0
    L[..., 2, 3] = 0.0
    # row 4 (acoustic +)
    L[..., 3, 0] = (g1 * q2 * 0.5 - u * a) * inv2a2
    L[..., 3, 1] = (-g1 * u + a) * inv2a2
    L[..., 3, 2] = (-g1 * v) * inv2a2
    L[..., 3, 3] = g1 * inv2a2

    lam = np.stack([u - a, u, u, u + a], axis=-1)  # (..., 4)
    return L, R, lam


def _eig_y(u, v, H, a):
    """Eigensystem of A_y = dG/dU.  Eigenvalues (v-a, v, v, v+a)."""
    shp = u.shape
    g1 = GAMMA - 1.0
    q2 = u * u + v * v
    a2 = a * a

    R = np.zeros(shp + (4, 4), dtype=u.dtype)
    # r_1 (v-a)
    R[..., 0, 0] = 1.0
    R[..., 1, 0] = u
    R[..., 2, 0] = v - a
    R[..., 3, 0] = H - v * a
    # r_2 (v, entropy)
    R[..., 0, 1] = 1.0
    R[..., 1, 1] = u
    R[..., 2, 1] = v
    R[..., 3, 1] = 0.5 * q2
    # r_3 (v, shear-u)
    R[..., 0, 2] = 0.0
    R[..., 1, 2] = 1.0
    R[..., 2, 2] = 0.0
    R[..., 3, 2] = u
    # r_4 (v+a)
    R[..., 0, 3] = 1.0
    R[..., 1, 3] = u
    R[..., 2, 3] = v + a
    R[..., 3, 3] = H + v * a

    L = np.zeros(shp + (4, 4), dtype=u.dtype)
    inv2a2 = 0.5 / a2
    inv_a2 = 1.0 / a2
    L[..., 0, 0] = (g1 * q2 * 0.5 + v * a) * inv2a2
    L[..., 0, 1] = (-g1 * u) * inv2a2
    L[..., 0, 2] = (-g1 * v - a) * inv2a2
    L[..., 0, 3] = g1 * inv2a2

    L[..., 1, 0] = (a2 - g1 * q2 * 0.5) * inv_a2
    L[..., 1, 1] = (g1 * u) * inv_a2
    L[..., 1, 2] = (g1 * v) * inv_a2
    L[..., 1, 3] = -g1 * inv_a2

    L[..., 2, 0] = -u
    L[..., 2, 1] = 1.0
    L[..., 2, 2] = 0.0
    L[..., 2, 3] = 0.0

    L[..., 3, 0] = (g1 * q2 * 0.5 - v * a) * inv2a2
    L[..., 3, 1] = (-g1 * u) * inv2a2
    L[..., 3, 2] = (-g1 * v + a) * inv2a2
    L[..., 3, 3] = g1 * inv2a2

    lam = np.stack([v - a, v, v, v + a], axis=-1)
    return L, R, lam


# =================================================================
#  characteristic-wise WENO5 directional derivative
# =================================================================
def _weno_char_dflux(F_pad, U_pad, eig_fn, axis, d_s, N_along, N_orth):
    """
    Compute scalar directional derivative dF/ds at all interior cells
    along axis (s = xi if axis=0, s = eta if axis=1), with N_along cells
    in the s direction and N_orth in the other direction.

    Inputs are FULLY padded (NG ghost on every side).  Output shape:
        axis=0:  (N_along, N_orth, 4)
        axis=1:  (N_along, N_orth, 4)
    (The "interior" trim along the orthogonal axis is done here.)
    """
    if axis == 1:
        # Reduce to axis=0 case by swapping the first two axes.
        F_t = np.swapaxes(F_pad, 0, 1)
        U_t = np.swapaxes(U_pad, 0, 1)
        out = _weno_char_dflux(F_t, U_t, eig_fn,
                               axis=0, d_s=d_s,
                               N_along=N_along, N_orth=N_orth)
        return np.swapaxes(out, 0, 1)

    # axis == 0: sweep along first axis at all orthogonal indices.
    # interior in orthogonal direction:
    j0 = NG
    j1 = NG + N_orth
    Fp = F_pad[:, j0:j1, :]                   # (Nx_full, N_orth, 4)
    Up = U_pad[:, j0:j1, :]                   # (Nx_full, N_orth, 4)

    # Face indices: i+1/2 for i = NG-1 .. NG+N_along-1  (N_along+1 faces)
    L = np.arange(NG - 1, NG + N_along)       # left-cell indices of each face
    # Left/right cell values at each face (no reconstruction yet, for Roe avg)
    UL = Up[L, :, :]                          # (Nfaces, N_orth, 4)
    UR = Up[L + 1, :, :]

    u, v, H, a = _roe_uvHa(UL, UR)            # (Nfaces, N_orth)
    Lmat, Rmat, lam = eig_fn(u, v, H, a)      # (Nfaces, N_orth, 4, 4)

    # Build the per-face flux/state stencil in characteristic space.
    # WENO5 needs cells i-2..i+3 (6 cells); WENO7 needs i-3..i+4 (8 cells).
    if USE_WENO7:
        k_lo, k_hi = -3, 5                    # offsets -3..4
    else:
        k_lo, k_hi = -2, 4                    # offsets -2..3
    stencil_F = [Fp[L + k, :, :] for k in range(k_lo, k_hi)]
    stencil_U = [Up[L + k, :, :] for k in range(k_lo, k_hi)]

    # Project to characteristic space:  fk = Lmat @ F  (per face)
    # Lmat shape (Nf, No, 4, 4), F shape (Nf, No, 4) -> contract last of L with F
    def proj(mat, vec):
        # einsum is the cleanest: out[..., k] = sum_m mat[..., k, m] vec[..., m]
        return np.einsum('...km,...m->...k', mat, vec, optimize=False)

    f_char = [proj(Lmat, sf) for sf in stencil_F]
    w_char = [proj(Lmat, su) for su in stencil_U]

    if USE_ROE:
        # Roe upwind in characteristic space with fixed Harten entropy fix.
        if USE_WENO7:
            # f_char/w_char index k=0..7 -> offsets i-3..i+4
            f_L = _weno7_left (f_char[0], f_char[1], f_char[2], f_char[3],
                               f_char[4], f_char[5], f_char[6])
            f_R = _weno7_right(f_char[1], f_char[2], f_char[3], f_char[4],
                               f_char[5], f_char[6], f_char[7])
            w_L = _weno7_left (w_char[0], w_char[1], w_char[2], w_char[3],
                               w_char[4], w_char[5], w_char[6])
            w_R = _weno7_right(w_char[1], w_char[2], w_char[3], w_char[4],
                               w_char[5], w_char[6], w_char[7])
        else:
            # f_char/w_char index k=0..5 -> offsets i-2..i+3
            f_L = _weno5_left (f_char[0], f_char[1], f_char[2], f_char[3], f_char[4])
            f_R = _weno5_right(f_char[1], f_char[2], f_char[3], f_char[4], f_char[5])
            w_L = _weno5_left (w_char[0], w_char[1], w_char[2], w_char[3], w_char[4])
            w_R = _weno5_right(w_char[1], w_char[2], w_char[3], w_char[4], w_char[5])

        abs_lam = np.abs(lam)
        spd = (np.abs(u) + np.abs(v) + a)[..., None]      # (Nf, No, 1)

        # Fixed per-family Harten entropy fix (NO shock sensor).  Big acoustic
        # eps kills the carbuncle on the strong normal shock; the linear eps
        # floor damps the transverse odd-even (sawtooth) mode where the flow
        # is grid-aligned and the Roe contact dissipation vanishes.
        eps_ac = EPS_AC * spd
        eps_li = EPS_LI * spd
        eps_h = np.concatenate([eps_ac, eps_li, eps_li, eps_ac], axis=-1)
        eps_h = np.maximum(eps_h, 1e-8)
        abs_lam_fix = np.where(abs_lam > eps_h,
                               abs_lam,
                               0.5 * (abs_lam * abs_lam + eps_h * eps_h) / eps_h)

        f_face_char = 0.5 * (f_L + f_R) - 0.5 * abs_lam_fix * (w_R - w_L)
    else:
        # Global per-family Lax-Friedrichs split (Jiang-Shu).  alpha_k = max
        # over the whole sweep of |lam_k|.  Most dissipative, most robust.
        alpha_face = np.abs(lam)
        alpha = alpha_face.max(axis=(0, 1), keepdims=True)
        alpha = np.broadcast_to(alpha, alpha_face.shape)
        fp = [0.5 * (f_char[k] + alpha * w_char[k]) for k in range(6)]
        fm = [0.5 * (f_char[k] - alpha * w_char[k]) for k in range(6)]
        fpL = _weno5_left (fp[0], fp[1], fp[2], fp[3], fp[4])
        fmR = _weno5_right(fm[1], fm[2], fm[3], fm[4], fm[5])
        f_face_char = fpL + fmR

    # Project back to conservative-flux space
    F_face = proj(Rmat, f_face_char)                 # (Nf, No, 4)

    # Difference between consecutive faces -> cell derivative
    dF = (F_face[1:] - F_face[:-1]) / d_s            # (N_along, N_orth, 4)
    return dF


# =================================================================
#  CurvilinearMesh with characteristic-WENO RHS
# =================================================================
class CurvilinearMeshChar(CurvilinearMesh):
    """Same as CurvilinearMesh but rhs() uses characteristic-wise WENO5."""

    def rhs(self, U):
        U_pad = self.pad_state(U)
        Fp = _flux_x(U_pad)
        Gp = _flux_y(U_pad)

        # Spectral radii (for dt only)
        rho, u, v, p = _prim(U_pad)
        a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-12))

        dFdxi  = _weno_char_dflux(Fp, U_pad, _eig_x, axis=0,
                                  d_s=self.d_xi,  N_along=self.Nxi,
                                  N_orth=self.Neta)
        dFdeta = _weno_char_dflux(Fp, U_pad, _eig_x, axis=1,
                                  d_s=self.d_eta, N_along=self.Neta,
                                  N_orth=self.Nxi)
        dGdxi  = _weno_char_dflux(Gp, U_pad, _eig_y, axis=0,
                                  d_s=self.d_xi,  N_along=self.Nxi,
                                  N_orth=self.Neta)
        dGdeta = _weno_char_dflux(Gp, U_pad, _eig_y, axis=1,
                                  d_s=self.d_eta, N_along=self.Neta,
                                  N_orth=self.Nxi)

        xi_x  = self.xi_x [:, :, None]
        xi_y  = self.xi_y [:, :, None]
        eta_x = self.eta_x[:, :, None]
        eta_y = self.eta_y[:, :, None]
        R = -(xi_x * dFdxi + eta_x * dFdeta
              + xi_y * dGdxi + eta_y * dGdeta)

        # dt bounds (curvilinear)
        u_int = u[NG:NG + self.Nxi, NG:NG + self.Neta]
        v_int = v[NG:NG + self.Nxi, NG:NG + self.Neta]
        a_int = a[NG:NG + self.Nxi, NG:NG + self.Neta]
        u_xi  = self.xi_x  * u_int + self.xi_y  * v_int
        u_eta = self.eta_x * u_int + self.eta_y * v_int
        lam_xi  = np.abs(u_xi)  + a_int * self.grad_xi_mag
        lam_eta = np.abs(u_eta) + a_int * self.grad_eta_mag
        dt_global = min(self.d_xi  / lam_xi.max(),
                        self.d_eta / lam_eta.max())
        dt_local = 1.0 / (lam_xi / self.d_xi + lam_eta / self.d_eta + 1e-30)
        return R, dt_global, dt_local


# =================================================================
#  smoke tests
# =================================================================
if __name__ == '__main__':
    cur = CurvilinearMeshChar('mapping_c.npz')
    print(f'mesh {cur.Nxi}x{cur.Neta}')

    U = cur.freestream()
    R, dt, _ = cur.rhs(U)
    print(f'freestream: |R|_max={np.abs(R).max():.3e}  '
          f'rms={np.sqrt(np.mean(R**2)):.3e}  (interior should be 0)')
    R_off = R.copy()
    R_off[:, 0, :] = 0.0   # ignore wall row
    print(f'  off-wall : |R|_max={np.abs(R_off).max():.3e}')

    # timing
    import time
    for _ in range(3): cur.rhs(U)
    t0 = time.time()
    N = 30
    for _ in range(N): cur.rhs(U)
    dt_call = (time.time() - t0) / N * 1000
    print(f'characteristic rhs: {dt_call:.1f} ms/call  '
          f'({dt_call * 1000 / (cur.Nxi*cur.Neta):.2f} us/cell)')
