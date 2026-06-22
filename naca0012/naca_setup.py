"""
Transonic NACA0012 (half-domain, symmetry about y=0) setup.

Geometry
--------
Physical domain  Omega = [XMIN, XMAX] x [YMIN, YMAX]
                       = [-0.5, 1.5] x [ 0.0, 3.0]
Airfoil chord on the symmetry line y=0, x in [0, 1].
Upper surface is the NACA 0012 thickness curve:

    y_s(x) = 0.6 * (0.2969 * sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    - 0.1036 * x**4)            x in [0, 1]

(Standard t/c = 0.12; coefficients give a CLOSED trailing edge.)
The "bottom" boundary (eta = 0 in computational space) is therefore:

    y_bot(x) = 0                       for x in [-0.5, 0]  (sym plane upstream)
             = y_s(x)                  for x in [ 0, 1 ]   (airfoil upper)
             = 0                       for x in [ 1, 1.5]  (sym plane downstream)

The other three sides are straight, all at far-field:
    left   (xi = 0):    x = XMIN, y in [0, YMAX]
    right  (xi = 1):    x = XMAX, y in [0, YMAX]
    top    (eta = 1):   y = YMAX, x in [XMIN, XMAX]

Free-stream / IC (transonic Euler reference benchmark)
------------------------------------------------------
M_inf  = 0.85,  alpha = 0      (half-domain symmetric flow)
Non-dimensionalisation: rho_inf = 1, p_inf = 1/gamma  =>  a_inf = 1
=>  u_inf = M_inf,  v_inf = 0,  p_inf = 1/gamma.
With this scaling the freestream speed of sound is unity and
the freestream pressure is 1/gamma; standard for AGARD-style
NACA0012 transonic tests reported on Cp.

Conservative state:
    rho   = 1
    rho u = M_inf
    rho v = 0
    E     = p/(gamma-1) + 0.5 * rho * (u^2 + v^2)
          = (1/gamma)/(gamma-1) + 0.5 * M_inf^2
"""
from __future__ import annotations
import math
import numpy as np

GAMMA = 1.4

# ------------------------------------------------------------------
# Physical domain
# ------------------------------------------------------------------
XMIN, XMAX = -0.5, 1.5
YMIN, YMAX = 0.0, 3.0
LX = XMAX - XMIN          # = 2.0
LY = YMAX - YMIN          # = 3.0

# Airfoil chord
CHORD_X0, CHORD_X1 = 0.0, 1.0
T_THICK = 0.12            # NACA 00xx thickness

# ------------------------------------------------------------------
# Freestream / IC / BC state (M_inf = 0.85, alpha = 0)
# ------------------------------------------------------------------
M_INF = 0.85
ALPHA = 0.0
RHO_INF = 1.0
P_INF   = 1.0 / GAMMA                                # so a_inf = 1
U_INF   = M_INF * math.cos(ALPHA)
V_INF   = M_INF * math.sin(ALPHA)
E_INF   = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * (U_INF**2 + V_INF**2)
U_INF_CONS = (RHO_INF, RHO_INF * U_INF, RHO_INF * V_INF, E_INF)


# ------------------------------------------------------------------
# NACA 0012 surface
# ------------------------------------------------------------------
_C = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)

def naca0012_y(x):
    """Upper surface y_s(x) for x in [0, 1].  Vectorised."""
    x = np.asarray(x, dtype=float)
    sx = np.sqrt(np.clip(x, 0.0, 1.0))
    return 0.6 * (_C[0] * sx + _C[1] * x + _C[2] * x * x
                  + _C[3] * x ** 3 + _C[4] * x ** 4)


def naca0012_dydx(x):
    """d y_s / d x.  Returns +inf at x=0 (vertical tangent at LE)."""
    x = np.asarray(x, dtype=float)
    eps = 1e-12
    sx = np.sqrt(np.clip(x, eps, 1.0))
    return 0.6 * (0.5 * _C[0] / sx + _C[1] + 2.0 * _C[2] * x
                  + 3.0 * _C[3] * x * x + 4.0 * _C[4] * x ** 3)


def y_bottom(x):
    """Bottom boundary y(x):  0 for x<=0 or x>=1, NACA surface for 0<x<1."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    inside = (x > 0.0) & (x < 1.0)
    y[inside] = naca0012_y(x[inside])
    return y


# ------------------------------------------------------------------
# IC field on a physical (X, Y) array  --  freestream everywhere
# ------------------------------------------------------------------
def freestream_field(X, Y, backend='numpy'):
    """Return (rho, rhou, rhov, E) arrays the same shape as X, Y.
    Uniform freestream is the canonical IC for steady transonic NACA0012.
    """
    if backend == 'numpy':
        ones = np.ones_like(np.asarray(X, dtype=float))
    else:
        import torch
        ones = torch.ones_like(X)
    rho  = RHO_INF * ones
    rhou = (RHO_INF * U_INF) * ones
    rhov = (RHO_INF * V_INF) * ones
    E    = E_INF * ones
    return rho, rhou, rhov, E


# ------------------------------------------------------------------
# Convenience converters
# ------------------------------------------------------------------
def primitive_to_conservative(rho, u, v, p, gamma=GAMMA):
    rhou = rho * u
    rhov = rho * v
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rhou, rhov, E


def conservative_to_primitive(rho, rhou, rhov, E, gamma=GAMMA):
    u = rhou / rho
    v = rhov / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


if __name__ == '__main__':
    # quick self-check
    print(f'Domain  [{XMIN},{XMAX}] x [{YMIN},{YMAX}]  LX={LX}, LY={LY}')
    print(f'Airfoil chord [{CHORD_X0},{CHORD_X1}], t/c={T_THICK}')
    print(f'Freestream  M={M_INF}, alpha={ALPHA}')
    print(f'  rho={RHO_INF}, u={U_INF:.4f}, v={V_INF:.4f}, p={P_INF:.6f}')
    print(f'  conservative={U_INF_CONS}')
    x_te = 1.0
    x_le = 0.0
    print(f'NACA0012 thickness:  y(0)={naca0012_y(x_le):.4f}, '
          f'y(0.3)={naca0012_y(0.3):.4f}, y(1)={naca0012_y(x_te):.4e}')
