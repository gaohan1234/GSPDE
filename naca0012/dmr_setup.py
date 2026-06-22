"""
Woodward-Colella Double Mach Reflection setup.

Domain: [0, LX] x [0, LY] with LX=4, LY=1.
A Mach-10 planar shock in air (gamma=1.4) hits a 30-deg ramp.
We avoid a curved/ramped mesh by tilting the geometry: the shock travels
along the original x-axis frame and the ramp is the bottom wall starting
at x = X0 = 1/6. The shock makes a 60-deg angle with the bottom wall,
i.e. its normal points along (cos 30, -sin 30) = (sqrt(3)/2, -1/2).

Pre-shock (right) state:
    rho_R = 1.4, u_R = 0, v_R = 0, p_R = 1.0

Post-shock (left) state from Rankine-Hugoniot for Mach-10 shock
(values are the canonical DMR values):
    rho_L = 8.0
    p_L   = 116.5
    u_L   =  8.25 * cos(30 deg) =  7.144...
    v_L   = -8.25 * sin(30 deg) = -4.125

Initial shock position (parameterized by y):
    x_s(y, 0) = X0 + y / tan(60 deg) = X0 + y / sqrt(3)

Top boundary (y = LY) moves the shock at speed 10/sin(60) = 20/sqrt(3)
in x:
    x_s(y=LY, t) = X0 + LY / sqrt(3) + 10 * t / sin(60 deg)
                 = X0 + (LY + 10 t) / sqrt(3)        (with sin 60 = sqrt(3)/2)
    -- correction: shock moves at speed 10 along its NORMAL, so the
       x-trace at any y advances at 10 / sin(60) = 20/sqrt(3).
"""
import math

GAMMA = 1.4

# Domain
LX = 4.0
LY = 1.0
X0 = 1.0 / 6.0
TAN60 = math.sqrt(3.0)
SIN60 = math.sqrt(3.0) / 2.0
COS30 = math.sqrt(3.0) / 2.0
SIN30 = 0.5

# Pre-shock state (right of the shock)
RHO_R = 1.4
U_R   = 0.0
V_R   = 0.0
P_R   = 1.0

# Post-shock state (left of the shock) -- Mach 3 normal-shock Rankine-Hugoniot
#   rho2/rho1 = (g+1)M^2 / ((g-1)M^2+2) = 21.6/5.6 = 27/7 = 3.857142857
#   p2/p1     = 1 + 2g/(g+1)(M^2-1) = 1 + (2.8/2.4)*8 = 31/3 = 10.33333
#   u_n behind shock (lab frame) = Ws*(1 - rho1/rho2) = 3 * (1 - 7/27) = 20/9
RHO_L = 5.4                                     # 1.4 * 27/7
P_L   = 31.0 / 3.0                              # 10.33333...
SHOCK_SPEED = 3.0                               # Mach 3 (a_R = 1)
_UN_L = 20.0 / 9.0                              # ~ 2.22222
U_L   =  _UN_L * COS30                          # ~ 1.92450
V_L   = -_UN_L * SIN30                          # ~ -1.11111

# Top-boundary shock x-position at time t (Woodward-Colella formula)
def x_shock_top(t):
    # Shock line at 60 deg from x-axis, moving at SHOCK_SPEED in its
    # normal direction.  The TRACE of the shock along the top wall y=LY
    # moves at SHOCK_SPEED / sin60 along x (NOT SHOCK_SPEED/tan60).
    # Equivalent form: X0 + (LY + 2*S*t)/tan60   (matches the classic
    # Mach-10 formula  1/6 + (1+20t)/sqrt(3) ).
    return X0 + (LY + 2.0 * SHOCK_SPEED * t) / TAN60


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


# Conservative versions of the two states (handy for IC/BC tensors)
UL_CONS = primitive_to_conservative(RHO_L, U_L, V_L, P_L)
UR_CONS = primitive_to_conservative(RHO_R, U_R, V_R, P_R)


def smoothed_ic_field(X, Y, delta=0.02, backend='numpy'):
    """Return (rho, rhou, rhov, E) on the (X, Y) grid, with a tanh-smoothed
    shock of half-thickness `delta` along the initial shock line.

    The signed distance from the shock line  x = X0 + y/sqrt(3)  along the
    shock normal (cos30, -sin30) is
        d(x,y) = (x - X0 - y/sqrt(3)) * sin(60 deg)
    so d > 0 means we are on the PRE-shock (right) side.

    Smoothing:  s = 0.5 * (1 + tanh(d / delta))   in [0,1],
        s=0 -> post-shock (left), s=1 -> pre-shock (right).
    Apply to each primitive variable independently then convert.
    """
    if backend == 'numpy':
        import numpy as np
        tanh = np.tanh
    else:
        import torch
        tanh = torch.tanh

    d = (X - X0 - Y / TAN60) * SIN60
    s = 0.5 * (1.0 + tanh(d / delta))

    rho = RHO_L + (RHO_R - RHO_L) * s
    u   = U_L   + (U_R   - U_L)   * s
    v   = V_L   + (V_R   - V_L)   * s
    p   = P_L   + (P_R   - P_L)   * s

    rhou = rho * u
    rhov = rho * v
    E    = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rhou, rhov, E
