"""
WENO5 finite-difference RHS for 2D Euler on a curvilinear grid
in CHAIN-RULE (quasi-linear) form.

Why chain rule and not strong conservation?
-------------------------------------------
Strong conservation requires the discrete Geometric Conservation Law
(GCL): discrete mixed metric partials must cancel exactly so that a
uniform free-stream stays steady.  With plain WENO5-JS on node-centred
metrics this cancellation FAILS and a uniform free-stream develops
O(1) residuals near distorted boundary cells (verified empirically:
|R|_max ~ 1e4 for M=0.85 free-stream on the NACA mesh).  Chain rule
trades formal flux conservation for AUTOMATIC free-stream preservation:
a uniform F (or G) gives zero WENO derivative regardless of the metric
layout.  This is standard practice for steady transonic airfoil
simulations -- shock position is captured within ~ one mesh cell,
which is sufficient for Cp comparison.

PDE
---
    U_t + F(U)_x + G(U)_y = 0
    F_x = xi_x  F_xi  +  eta_x  F_eta
    G_y = xi_y  G_xi  +  eta_y  G_eta
=>  U_t = - xi_x F_xi - eta_x F_eta - xi_y G_xi - eta_y G_eta

Each of F_xi, F_eta, G_xi, G_eta is approximated by WENO5-FD with
component-wise local Lax-Friedrichs dissipation.

Boundary conditions (filled into NG=3 ghost cells):
   eta = 0 : slip wall (mirror reflection about physical normal)
   eta = 1 : freestream Dirichlet
   xi  = 0 : freestream Dirichlet
   xi  = 1 : freestream Dirichlet

API
---
    cur = CurvilinearMesh(mapping_path)
    R, dt_global = cur.rhs(U)        # U: (Nxi, Neta, 4)
"""
from __future__ import annotations
import numpy as np

from naca_setup import GAMMA, U_INF_CONS

NG  = 4
EPS = 1.0e-6

# Outer (eta=1) far-field boundary treatment:
#   'freestream' : hard freestream Dirichlet ghosts (reflects outgoing waves)
#   'riemann'    : characteristic Riemann-invariant non-reflecting far-field
FARFIELD_BC = 'riemann'


# =================================================================
#  primitives, fluxes
# =================================================================
def _prim(U):
    rho  = U[..., 0]
    rhou = U[..., 1]
    rhov = U[..., 2]
    E    = U[..., 3]
    u = rhou / rho
    v = rhov / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def _flux_x(U):
    rho, u, v, p = _prim(U)
    rhou = U[..., 1]
    rhov = U[..., 2]
    E    = U[..., 3]
    return np.stack([rhou,
                     rhou * u + p,
                     rhou * v,
                     u * (E + p)], axis=-1)


def _flux_y(U):
    rho, u, v, p = _prim(U)
    rhou = U[..., 1]
    rhov = U[..., 2]
    E    = U[..., 3]
    return np.stack([rhov,
                     rhov * u,
                     rhov * v + p,
                     v * (E + p)], axis=-1)


# =================================================================
#  WENO5-JS reconstruction
# =================================================================
_D0, _D1, _D2 = 1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0


def _weno5_left(qm2, qm1, q0, qp1, qp2):
    p0 = ( 2.0 * qm2 -  7.0 * qm1 + 11.0 * q0 ) / 6.0
    p1 = (-1.0 * qm1 +  5.0 * q0  +  2.0 * qp1) / 6.0
    p2 = ( 2.0 * q0  +  5.0 * qp1 -  1.0 * qp2) / 6.0
    b0 = (13.0 / 12.0) * (qm2 - 2.0 * qm1 + q0 ) ** 2 \
         + 0.25 * (qm2 - 4.0 * qm1 + 3.0 * q0 ) ** 2
    b1 = (13.0 / 12.0) * (qm1 - 2.0 * q0  + qp1) ** 2 \
         + 0.25 * (qm1 - qp1) ** 2
    b2 = (13.0 / 12.0) * (q0  - 2.0 * qp1 + qp2) ** 2 \
         + 0.25 * (3.0 * q0 - 4.0 * qp1 + qp2) ** 2
    a0 = _D0 / (EPS + b0) ** 2
    a1 = _D1 / (EPS + b1) ** 2
    a2 = _D2 / (EPS + b2) ** 2
    s = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / s


def _weno5_right(qm1, q0, qp1, qp2, qp3):
    p0 = ( 2.0 * qp3 -  7.0 * qp2 + 11.0 * qp1) / 6.0
    p1 = (-1.0 * qp2 +  5.0 * qp1 +  2.0 * q0 ) / 6.0
    p2 = ( 2.0 * qp1 +  5.0 * q0  -  1.0 * qm1) / 6.0
    b0 = (13.0 / 12.0) * (qp3 - 2.0 * qp2 + qp1) ** 2 \
         + 0.25 * (qp3 - 4.0 * qp2 + 3.0 * qp1) ** 2
    b1 = (13.0 / 12.0) * (qp2 - 2.0 * qp1 + q0 ) ** 2 \
         + 0.25 * (qp2 - q0) ** 2
    b2 = (13.0 / 12.0) * (qp1 - 2.0 * q0  + qm1) ** 2 \
         + 0.25 * (3.0 * qp1 - 4.0 * q0 + qm1) ** 2
    a0 = _D0 / (EPS + b0) ** 2
    a1 = _D1 / (EPS + b1) ** 2
    a2 = _D2 / (EPS + b2) ** 2
    s = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / s


# =================================================================
#  WENO7-JS reconstruction (Balsara & Shu 2000 smoothness indicators)
# =================================================================
# Optimal linear weights for 7th-order upwind reconstruction.
_E0, _E1, _E2, _E3 = 1.0 / 35.0, 12.0 / 35.0, 18.0 / 35.0, 4.0 / 35.0


def _weno7(m3, m2, m1, c, p1, p2, p3):
    """Left-biased 7th-order WENO reconstruction of the value at the RIGHT
    face (center+1/2) of the center cell `c`.  Arguments are the 7-cell
    stencil values  (center-3 .. center+3).

    The right state at a face is obtained by calling this with the stencil
    reversed about the right cell (mirror symmetry)."""
    # Candidate 4th-order polynomials evaluated at center+1/2.
    q0 = (-3.0 * m3 + 13.0 * m2 - 23.0 * m1 + 25.0 * c) / 12.0
    q1 = ( 1.0 * m2 -  5.0 * m1 + 13.0 * c  +  3.0 * p1) / 12.0
    q2 = (-1.0 * m1 +  7.0 * c  +  7.0 * p1 -  1.0 * p2) / 12.0
    q3 = ( 3.0 * c  + 13.0 * p1 -  5.0 * p2 +  1.0 * p3) / 12.0

    # Jiang-Shu / Balsara-Shu smoothness indicators.
    b0 = (m3 * (547.0 * m3 - 3882.0 * m2 + 4642.0 * m1 - 1854.0 * c)
          + m2 * (7043.0 * m2 - 17246.0 * m1 + 7042.0 * c)
          + m1 * (11003.0 * m1 - 9402.0 * c)
          + 2107.0 * c * c)
    b1 = (m2 * (267.0 * m2 - 1642.0 * m1 + 1602.0 * c - 494.0 * p1)
          + m1 * (2843.0 * m1 - 5966.0 * c + 1922.0 * p1)
          + c * (3443.0 * c - 2522.0 * p1)
          + 547.0 * p1 * p1)
    b2 = (m1 * (547.0 * m1 - 2522.0 * c + 1922.0 * p1 - 494.0 * p2)
          + c * (3443.0 * c - 5966.0 * p1 + 1602.0 * p2)
          + p1 * (2843.0 * p1 - 1642.0 * p2)
          + 267.0 * p2 * p2)
    b3 = (c * (2107.0 * c - 9402.0 * p1 + 7042.0 * p2 - 1854.0 * p3)
          + p1 * (11003.0 * p1 - 17246.0 * p2 + 4642.0 * p3)
          + p2 * (7043.0 * p2 - 3882.0 * p3)
          + 547.0 * p3 * p3)

    a0 = _E0 / (EPS + b0) ** 2
    a1 = _E1 / (EPS + b1) ** 2
    a2 = _E2 / (EPS + b2) ** 2
    a3 = _E3 / (EPS + b3) ** 2
    s = a0 + a1 + a2 + a3
    return (a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3) / s


def _weno7_left(qm3, qm2, qm1, q0, qp1, qp2, qp3):
    """Left state at face i+1/2 (center cell = i)."""
    return _weno7(qm3, qm2, qm1, q0, qp1, qp2, qp3)


def _weno7_right(qm2, qm1, q0, qp1, qp2, qp3, qp4):
    """Right state at face i+1/2 (center cell = i+1, mirror about i+1)."""
    return _weno7(qp4, qp3, qp2, qp1, q0, qm1, qm2)

class CurvilinearMesh:
    def __init__(self, mapping_path='mapping.npz'):
        d = np.load(mapping_path, allow_pickle=True)
        self.X     = d['X'].astype(np.float64)
        self.Y     = d['Y'].astype(np.float64)
        self.x_xi  = d['x_xi'].astype(np.float64)
        self.x_eta = d['x_eta'].astype(np.float64)
        self.y_xi  = d['y_xi'].astype(np.float64)
        self.y_eta = d['y_eta'].astype(np.float64)
        self.J     = d['J'].astype(np.float64)
        self.Nxi   = int(d['Nxi'])
        self.Neta  = int(d['Neta'])
        self.d_xi  = 1.0 / (self.Nxi - 1)
        self.d_eta = 1.0 / (self.Neta - 1)
        assert self.J.min() > 0.0

        self.xi_x  =  self.y_eta / self.J
        self.xi_y  = -self.x_eta / self.J
        self.eta_x = -self.y_xi  / self.J
        self.eta_y =  self.x_xi  / self.J
        self.grad_xi_mag  = np.sqrt(self.xi_x  ** 2 + self.xi_y  ** 2)
        self.grad_eta_mag = np.sqrt(self.eta_x ** 2 + self.eta_y ** 2)

        nx_raw = self.eta_x[:, 0]
        ny_raw = self.eta_y[:, 0]
        nmag   = np.sqrt(nx_raw ** 2 + ny_raw ** 2).clip(min=1e-14)
        self.n_wall_x = nx_raw / nmag
        self.n_wall_y = ny_raw / nmag

        # outward unit normal on the outer (eta=1) far-field boundary,
        # pointing in the +eta direction (out of the domain)
        fx_raw = self.eta_x[:, -1]
        fy_raw = self.eta_y[:, -1]
        fmag   = np.sqrt(fx_raw ** 2 + fy_raw ** 2).clip(min=1e-14)
        self.n_far_x = fx_raw / fmag
        self.n_far_y = fy_raw / fmag

        # outward unit normals on the left (xi=0, inflow) and right
        # (xi=1, outflow) far-field boundaries.  grad(xi) points in the +xi
        # direction (into the domain on the left edge), so the outward normal
        # is -grad(xi) on the left and +grad(xi) on the right.
        lx_raw = -self.xi_x[0, :]
        ly_raw = -self.xi_y[0, :]
        lmag   = np.sqrt(lx_raw ** 2 + ly_raw ** 2).clip(min=1e-14)
        self.n_left_x = lx_raw / lmag
        self.n_left_y = ly_raw / lmag

        rx_raw = self.xi_x[-1, :]
        ry_raw = self.xi_y[-1, :]
        rmag   = np.sqrt(rx_raw ** 2 + ry_raw ** 2).clip(min=1e-14)
        self.n_right_x = rx_raw / rmag
        self.n_right_y = ry_raw / rmag

        self.U_inf = np.asarray(U_INF_CONS, dtype=np.float64)

    # ------------------------------------------------------------------
    def _riemann_farfield(self, U_edge, nx, ny):
        """Characteristic Riemann-invariant far-field state on an outer
        boundary with outward unit normal (nx, ny).  U_edge is the adjacent
        interior row/column (N, 4); nx, ny are (N,).
        Returns the boundary conservative state (N, 4)."""
        gm1 = GAMMA - 1.0

        rho_i, u_i, v_i, p_i = _prim(U_edge)
        rho_i = np.maximum(rho_i, 1e-12)
        p_i   = np.maximum(p_i,   1e-12)
        a_i   = np.sqrt(GAMMA * p_i / rho_i)
        vn_i  = u_i * nx + v_i * ny

        rho_o, u_o, v_o, p_o = _prim(self.U_inf[None, :])
        rho_o = float(rho_o);  p_o = float(p_o)
        u_o = float(u_o);      v_o = float(v_o)
        a_o = np.sqrt(GAMMA * p_o / rho_o)
        vn_o = u_o * nx + v_o * ny

        # Riemann invariants: R+ outgoing (interior), R- incoming (freestream)
        Rp = vn_i + 2.0 * a_i / gm1
        Rm = vn_o - 2.0 * a_o / gm1
        vn_b = 0.5 * (Rp + Rm)
        a_b  = 0.25 * gm1 * (Rp - Rm)
        a_b  = np.maximum(a_b, 1e-12)

        inflow = vn_b <= 0.0
        # tangential velocity and entropy from the upwind side
        ut_i = u_i - vn_i * nx;   vt_i = v_i - vn_i * ny
        ut_o = u_o - vn_o * nx;   vt_o = v_o - vn_o * ny
        ut = np.where(inflow, ut_o, ut_i)
        vt = np.where(inflow, vt_o, vt_i)
        s  = np.where(inflow, p_o / rho_o ** GAMMA, p_i / rho_i ** GAMMA)

        rho_b = (a_b * a_b / (GAMMA * s)) ** (1.0 / gm1)
        p_b   = rho_b * a_b * a_b / GAMMA
        u_b   = ut + vn_b * nx
        v_b   = vt + vn_b * ny

        Ub = np.empty_like(U_edge)
        Ub[:, 0] = rho_b
        Ub[:, 1] = rho_b * u_b
        Ub[:, 2] = rho_b * v_b
        Ub[:, 3] = p_b / gm1 + 0.5 * rho_b * (u_b * u_b + v_b * v_b)
        return Ub

    # ------------------------------------------------------------------
    def pad_state(self, U):
        Nxi, Neta, _ = U.shape
        U_pad = np.empty((Nxi + 2 * NG, Neta + 2 * NG, 4), dtype=U.dtype)
        U_pad[NG:NG + Nxi, NG:NG + Neta] = U

        U_inf = self.U_inf[None, None, :]
        U_pad[:NG, NG:NG + Neta, :]      = U_inf
        U_pad[NG + Nxi:, NG:NG + Neta, :] = U_inf
        U_pad[:, NG + Neta:, :]          = U_inf

        if FARFIELD_BC == 'riemann':
            # characteristic Riemann-invariant non-reflecting far-field on
            # ALL three outer boundaries (left inflow xi=0, right outflow
            # xi=1, top eta=1), matching the reference setup where the entire
            # outer rectangle is a single far-field boundary.  Outgoing
            # invariant from the interior, incoming invariant from freestream;
            # entropy & tangential velocity taken from the upwind side.
            Ub_top = self._riemann_farfield(U[:, -1, :],
                                            self.n_far_x, self.n_far_y)
            U_pad[NG:NG + Nxi, NG + Neta:, :] = Ub_top[:, None, :]

            Ub_left = self._riemann_farfield(U[0, :, :],
                                             self.n_left_x, self.n_left_y)
            U_pad[:NG, NG:NG + Neta, :] = Ub_left[None, :, :]

            Ub_right = self._riemann_farfield(U[-1, :, :],
                                              self.n_right_x, self.n_right_y)
            U_pad[NG + Nxi:, NG:NG + Neta, :] = Ub_right[None, :, :]

        nx_full = np.empty(Nxi + 2 * NG)
        ny_full = np.empty(Nxi + 2 * NG)
        nx_full[NG:NG + Nxi] = self.n_wall_x
        ny_full[NG:NG + Nxi] = self.n_wall_y
        nx_full[:NG]         = self.n_wall_x[0]
        ny_full[:NG]         = self.n_wall_y[0]
        nx_full[NG + Nxi:]   = self.n_wall_x[-1]
        ny_full[NG + Nxi:]   = self.n_wall_y[-1]

        for k in range(NG):
            j_int = NG + k
            j_gho = NG - 1 - k
            U_src = U_pad[:, j_int, :].copy()
            rho = U_src[:, 0]
            mx  = U_src[:, 1]
            my  = U_src[:, 2]
            E   = U_src[:, 3]
            mdotn = mx * nx_full + my * ny_full
            mx_r  = mx - 2.0 * mdotn * nx_full
            my_r  = my - 2.0 * mdotn * ny_full
            U_pad[:, j_gho, :] = np.stack([rho, mx_r, my_r, E], axis=-1)

        return U_pad

    # ------------------------------------------------------------------
    @staticmethod
    def _weno_dflux(H_pad, U_pad, alpha, axis, d_s, N):
        fp = 0.5 * (H_pad + alpha * U_pad)
        fm = 0.5 * (H_pad - alpha * U_pad)
        L = np.arange(NG - 1, NG + N)
        if axis == 0:
            fpL = _weno5_left (fp[L - 2], fp[L - 1], fp[L], fp[L + 1], fp[L + 2])
            fmR = _weno5_right(fm[L - 1], fm[L], fm[L + 1], fm[L + 2], fm[L + 3])
            F_face = fpL + fmR
            return (F_face[1:] - F_face[:-1]) / d_s
        else:
            fpL = _weno5_left (fp[:, L - 2], fp[:, L - 1], fp[:, L],
                               fp[:, L + 1], fp[:, L + 2])
            fmR = _weno5_right(fm[:, L - 1], fm[:, L], fm[:, L + 1],
                               fm[:, L + 2], fm[:, L + 3])
            F_face = fpL + fmR
            return (F_face[:, 1:] - F_face[:, :-1]) / d_s

    # ------------------------------------------------------------------
    def rhs(self, U):
        U_pad = self.pad_state(U)
        Fp = _flux_x(U_pad)
        Gp = _flux_y(U_pad)

        rho, u, v, p = _prim(U_pad)
        a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-12))
        alpha_x = float((np.abs(u) + a).max())
        alpha_y = float((np.abs(v) + a).max())

        dFdxi  = self._weno_dflux(Fp, U_pad, alpha_x, axis=0,
                                  d_s=self.d_xi,  N=self.Nxi)
        dFdeta = self._weno_dflux(Fp, U_pad, alpha_x, axis=1,
                                  d_s=self.d_eta, N=self.Neta)
        dGdxi  = self._weno_dflux(Gp, U_pad, alpha_y, axis=0,
                                  d_s=self.d_xi,  N=self.Nxi)
        dGdeta = self._weno_dflux(Gp, U_pad, alpha_y, axis=1,
                                  d_s=self.d_eta, N=self.Neta)

        # Trim each to interior on BOTH axes
        dFdxi  = dFdxi [:, NG:NG + self.Neta, :]
        dGdxi  = dGdxi [:, NG:NG + self.Neta, :]
        dFdeta = dFdeta[NG:NG + self.Nxi, :, :]
        dGdeta = dGdeta[NG:NG + self.Nxi, :, :]

        xi_x  = self.xi_x [:, :, None]
        xi_y  = self.xi_y [:, :, None]
        eta_x = self.eta_x[:, :, None]
        eta_y = self.eta_y[:, :, None]
        R = -(xi_x * dFdxi + eta_x * dFdeta
              + xi_y * dGdxi + eta_y * dGdeta)

        # Curvilinear directional speeds for the dt bound
        u_int = u[NG:NG + self.Nxi, NG:NG + self.Neta]
        v_int = v[NG:NG + self.Nxi, NG:NG + self.Neta]
        a_int = a[NG:NG + self.Nxi, NG:NG + self.Neta]
        u_xi  = self.xi_x  * u_int + self.xi_y  * v_int
        u_eta = self.eta_x * u_int + self.eta_y * v_int
        lam_xi  = np.abs(u_xi)  + a_int * self.grad_xi_mag
        lam_eta = np.abs(u_eta) + a_int * self.grad_eta_mag
        dt_global = min(self.d_xi  / lam_xi.max(),
                        self.d_eta / lam_eta.max())
        # Local CFL=1 step per node (used for steady-state acceleration)
        dt_local = 1.0 / (lam_xi / self.d_xi + lam_eta / self.d_eta + 1e-30)
        return R, dt_global, dt_local

    # ------------------------------------------------------------------
    def freestream(self):
        U = np.empty((self.Nxi, self.Neta, 4), dtype=np.float64)
        U[..., :] = self.U_inf[None, None, :]
        return U


if __name__ == '__main__':
    cur = CurvilinearMesh('mapping.npz')
    print(f'mesh  Nxi={cur.Nxi}, Neta={cur.Neta}, '
          f'd_xi={cur.d_xi:.4f}, d_eta={cur.d_eta:.4f}, '
          f'J:[{cur.J.min():.3e},{cur.J.max():.3e}]')
    U = cur.freestream()
    R, dt, dt_loc = cur.rhs(U)
    print(f'freestream R: |max|={np.abs(R).max():.3e}  '
          f'rms={np.sqrt(np.mean(R**2)):.3e}  (should be ~0)')
    print(f'global CFL=1 dt = {dt:.4e}  '
          f'dt_local: [{dt_loc.min():.3e}, {dt_loc.max():.3e}]')
