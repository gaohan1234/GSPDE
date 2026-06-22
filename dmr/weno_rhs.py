"""
WENO5-JS finite-volume RHS for the 2D Euler equations on a uniform
tensor-product grid, used as the right-hand-side of the EDNN/Galerkin
projection.

We treat the Ngal_x x Ngal_y cell-centered collocation grid as a
finite-volume mesh.  The state U(N, 4) on that mesh is reshaped to
(Nx, Ny, 4) and extended with NG=3 ghost cells per side that encode
the Woodward-Colella DMR boundary conditions:

    bottom  y=0   :  x < X0     -> post-shock Dirichlet (UL)
                      x >= X0    -> reflective slip wall (mirror, v -> -v)
    top     y=LY  :  x < x_s(t) -> post-shock Dirichlet (UL)
                      else       -> pre-shock  Dirichlet (UR)
    left    x=0   :  post-shock Dirichlet (UL)
    right   x=LX  :  supersonic outflow (zero-order extrapolation)

The numerical flux is component-wise Local Lax-Friedrichs with WENO5-JS
left/right reconstruction in each direction (dimensional splitting).
This gives intrinsic upwind dissipation at shocks without any ad-hoc
artificial viscosity.

Public API
----------
    R = weno_rhs_grid(U_grid, dx, dy, t)
        U_grid : (Nx, Ny, 4) torch tensor (cell centers, conservative vars)
        returns -dF/dx - dG/dy at the same cell centers, shape (Nx, Ny, 4)
"""
from __future__ import annotations

import torch

from dmr_setup import (GAMMA, X0, LX, LY,
                       UL_CONS, UR_CONS, x_shock_top)

NG = 3                  # WENO5 needs 3 ghost cells per side
EPS = 1.0e-6


# ---------------- primitives / fluxes ----------------
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
    return torch.stack([rhou,
                        rhou * u + p,
                        rhou * v,
                        u * (E + p)], dim=-1)


def _flux_y(U):
    rho, u, v, p = _prim(U)
    rhou = U[..., 1]
    rhov = U[..., 2]
    E    = U[..., 3]
    return torch.stack([rhov,
                        rhov * u,
                        rhov * v + p,
                        v * (E + p)], dim=-1)


def _max_wave_speed_xy(U):
    """Return (alpha_x, alpha_y) global LF speeds on this padded array."""
    rho, u, v, p = _prim(U)
    a = torch.sqrt((GAMMA * p / rho).clamp_min(1e-12))
    alpha_x = (u.abs() + a).max()
    alpha_y = (v.abs() + a).max()
    return alpha_x, alpha_y


# ---------------- WENO5-JS reconstruction ----------------
# Linear weights
_D0 = 1.0 / 10.0
_D1 = 6.0 / 10.0
_D2 = 3.0 / 10.0


def _weno5_left(qm2, qm1, q0, qp1, qp2):
    """Reconstruct value at i+1/2 from the LEFT (upwind for f+).

    Stencil = (q_{i-2}, q_{i-1}, q_i, q_{i+1}, q_{i+2}).
    All inputs have the same shape; returns same shape.
    """
    # candidate polynomial values at x_{i+1/2}
    p0 = ( 2.0 * qm2 -  7.0 * qm1 + 11.0 * q0 ) / 6.0
    p1 = (-1.0 * qm1 +  5.0 * q0  +  2.0 * qp1) / 6.0
    p2 = ( 2.0 * q0  +  5.0 * qp1 -  1.0 * qp2) / 6.0
    # smoothness indicators (Jiang-Shu)
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
    """Reconstruct value at i+1/2 from the RIGHT (upwind for f-).

    Stencil for cell i+1 viewed leftwards = (q_{i+3}, q_{i+2}, q_{i+1},
    q_i, q_{i-1}) when applied to f-.  Equivalently: at face i+1/2 the
    'right' state is a left-reconstruction of cell (i+1) using mirrored
    stencil; here we accept the natural stencil
    (q_{i-1}, q_i, q_{i+1}, q_{i+2}, q_{i+3}) and form the polynomial
    candidates for the i+1/2 face from the RIGHT side.
    """
    # By symmetry of stencil reversal:
    # mirror i+1/2 reconstruction by swapping indices.
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


# ---------------- DMR ghost-cell padding ----------------
def _pad_dmr(U_grid, dx, dy, t):
    """U_grid (Nx, Ny, 4) -> U_pad (Nx+2NG, Ny+2NG, 4) with DMR BCs.

    `dx`, `dy` are cell sizes, used to recover x, y coordinates of the
    bottom/top ghost cells so the piecewise BC can be applied per column.
    """
    Nx, Ny, _ = U_grid.shape
    dev = U_grid.device
    dt  = U_grid.dtype
    UL  = torch.tensor(UL_CONS, device=dev, dtype=dt)
    UR  = torch.tensor(UR_CONS, device=dev, dtype=dt)

    U_pad = torch.empty(Nx + 2 * NG, Ny + 2 * NG, 4, device=dev, dtype=dt)
    U_pad[NG:NG + Nx, NG:NG + Ny] = U_grid

    # --- left ghost (x < 0): post-shock UL Dirichlet -----------------
    U_pad[:NG, NG:NG + Ny] = UL.view(1, 1, 4)

    # --- right ghost (x > LX): zero-order extrapolation --------------
    U_pad[NG + Nx:, NG:NG + Ny] = U_grid[-1:, :, :]

    # --- bottom ghost (y < 0): piecewise per x ------------------------
    # x-coords of the columns including left/right ghost columns
    i_all = torch.arange(Nx + 2 * NG, device=dev, dtype=dt)
    x_col = (i_all - NG + 0.5) * dx                   # (Nx+2NG,)
    is_pre_ramp = (x_col < X0)                        # (Nx+2NG,)

    # post-shock Dirichlet for x < X0
    U_pre = UL.view(1, 1, 4).expand(Nx + 2 * NG, NG, 4)
    # reflective for x >= X0: mirror interior cells across y=0, flip rhov
    # We need NG bottom ghost cells per column for the interior x-range.
    # For x in left/right ghost columns we fall back to UL (cheap, never
    # actually used by the interior stencils because we trim afterwards).
    U_refl = U_pad[:, NG:NG + NG, :].clone()          # cells (i, NG..NG+NG-1)
    U_refl = torch.flip(U_refl, dims=[1])             # mirror in y
    U_refl[..., 2] = -U_refl[..., 2]                  # flip rhov
    # combine
    mask = is_pre_ramp.view(-1, 1, 1)                 # (Nx+2NG, 1, 1)
    U_bot = torch.where(mask, U_pre, U_refl)
    U_pad[:, :NG, :] = U_bot

    # --- top ghost (y > LY): piecewise per x ------------------------
    xs_t = x_shock_top(t)
    is_post = (x_col < xs_t)                          # (Nx+2NG,)
    U_top = torch.where(is_post.view(-1, 1, 1),
                        UL.view(1, 1, 4).expand(Nx + 2 * NG, NG, 4),
                        UR.view(1, 1, 4).expand(Nx + 2 * NG, NG, 4))
    U_pad[:, NG + Ny:, :] = U_top

    # Fill bottom corners (left/right ghost x-columns) with the same
    # rule as their x-neighbour for cleanliness (won't affect interior).
    # Already handled by the mask above (left/right cols are x<X0 or
    # x>=X0 and reflective uses left/right neighbour values already in
    # U_pad via U_refl, which read from columns 0..Nx+2NG-1 — fine).
    return U_pad


# ---------------- LF + WENO5 flux divergence ----------------
def _div_x(U_pad, dx):
    """Compute dF/dx at interior cells (Nx, Ny). Uses component-wise
    Local Lax-Friedrichs with global alpha.
    """
    F = _flux_x(U_pad)                                # (Nxp, Nyp, 4)
    alpha_x, _ = _max_wave_speed_xy(U_pad)
    fp = 0.5 * (F + alpha_x * U_pad)                  # f+
    fm = 0.5 * (F - alpha_x * U_pad)                  # f-

    # We want F_{i+1/2} at faces i = NG-1 .. NG+Nx-1  (Nx+1 faces)
    # For each face, build stencils of width 5 in the x-direction.
    Nxp, Nyp, _ = U_pad.shape
    Nx = Nxp - 2 * NG
    # face index range: i goes 0..Nx (so that interior cell j
    # uses faces (j-1)+1/2 and j+1/2 -> indices NG-1..NG+Nx-1 in U_pad).
    # We index by left-cell position L = NG-1..NG+Nx-1 (length Nx+1).
    L = torch.arange(NG - 1, NG + Nx, device=U_pad.device)
    # f+_L from cells (L-2, L-1, L, L+1, L+2)
    fpL = _weno5_left(fp[L - 2], fp[L - 1], fp[L], fp[L + 1], fp[L + 2])
    # f-_R from cells (L-1, L, L+1, L+2, L+3)
    fmR = _weno5_right(fm[L - 1], fm[L], fm[L + 1], fm[L + 2], fm[L + 3])
    F_face = fpL + fmR                                # (Nx+1, Nyp, 4)
    # dF/dx at interior cells
    dFdx = (F_face[1:] - F_face[:-1]) / dx            # (Nx, Nyp, 4)
    return dFdx[:, NG:NG + (Nyp - 2 * NG), :]         # (Nx, Ny, 4)


def _div_y(U_pad, dy):
    """Same as _div_x but in y-direction."""
    G = _flux_y(U_pad)
    _, alpha_y = _max_wave_speed_xy(U_pad)
    gp = 0.5 * (G + alpha_y * U_pad)
    gm = 0.5 * (G - alpha_y * U_pad)
    Nxp, Nyp, _ = U_pad.shape
    Ny = Nyp - 2 * NG
    L = torch.arange(NG - 1, NG + Ny, device=U_pad.device)
    # index along axis 1
    gpL = _weno5_left(gp[:, L - 2], gp[:, L - 1], gp[:, L],
                      gp[:, L + 1], gp[:, L + 2])
    gmR = _weno5_right(gm[:, L - 1], gm[:, L], gm[:, L + 1],
                       gm[:, L + 2], gm[:, L + 3])
    G_face = gpL + gmR                                # (Nxp, Ny+1, 4)
    dGdy = (G_face[:, 1:] - G_face[:, :-1]) / dy
    return dGdy[NG:NG + (Nxp - 2 * NG), :, :]


def weno_rhs_grid(U_grid, dx, dy, t, nu=0.0):
    """R(U) = -dF/dx - dG/dy [+ nu * Laplacian(U)] at cell centers.

    Returns shape (Nx, Ny, 4). The optional Laplacian uses the DMR ghost
    cells for second-order central differences, which gives a small,
    isotropic damping on all four conservative variables to suppress
    Galerkin-aliasing modes (set nu=0 to disable).
    """
    U_pad = _pad_dmr(U_grid, dx, dy, t)
    dFdx = _div_x(U_pad, dx)
    dGdy = _div_y(U_pad, dy)
    R = -(dFdx + dGdy)
    if nu and float(nu) > 0.0:
        Nxp, Nyp, _ = U_pad.shape
        Nx = Nxp - 2 * NG
        Ny = Nyp - 2 * NG
        Uc = U_pad[NG:NG + Nx, NG:NG + Ny]
        Uxm = U_pad[NG - 1:NG - 1 + Nx, NG:NG + Ny]
        Uxp = U_pad[NG + 1:NG + 1 + Nx, NG:NG + Ny]
        Uym = U_pad[NG:NG + Nx, NG - 1:NG - 1 + Ny]
        Uyp = U_pad[NG:NG + Nx, NG + 1:NG + 1 + Ny]
        lap = (Uxm - 2.0 * Uc + Uxp) / (dx * dx) \
              + (Uym - 2.0 * Uc + Uyp) / (dy * dy)
        R = R + float(nu) * lap
    return R
