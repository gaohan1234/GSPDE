"""Gaussian-splatting PDE surrogate for the 2D lid-driven cavity (evaluation only).

The model represents the velocity (u, v) and pressure (p) fields as a sum of
anisotropic Gaussian kernels.  This file contains only what is needed to load a
trained checkpoint and evaluate its fields; no training code is included.
"""
import torch


class GSSplatEllip(torch.nn.Module):
    def __init__(self, n, r_min, r_max):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(n, 2))
        self.log_a = torch.nn.Parameter(torch.zeros(n))
        self.b = torch.nn.Parameter(torch.zeros(n))
        self.log_c = torch.nn.Parameter(torch.zeros(n))
        self.c_u = torch.nn.Parameter(torch.zeros(n))
        self.c_v = torch.nn.Parameter(torch.zeros(n))
        self.c_p = torch.nn.Parameter(torch.zeros(n))

    def _kernels(self, xy):
        diff = xy.unsqueeze(1) - self.mu.unsqueeze(0)
        a, b, c = torch.exp(self.log_a)[None, :], self.b[None, :], torch.exp(self.log_c)[None, :]
        ia, ic = 1.0 / a, 1.0 / c
        m = -(b * ia * ic)
        dx, dy = diff[..., 0], diff[..., 1]
        y0, y1 = ia * dx, m * dx + ic * dy
        g = torch.exp(-0.5 * (y0 * y0 + y1 * y1))
        return g, y0, y1, ia, ic, m

    def forward(self, xy):
        """Return (u, v, p) at points xy of shape (N, 2)."""
        g, *_ = self._kernels(xy)
        u = (self.c_u[None, :] * g).sum(1, keepdim=True)
        v = (self.c_v[None, :] * g).sum(1, keepdim=True)
        p = (self.c_p[None, :] * g).sum(1, keepdim=True)
        return torch.cat((u, v, p), dim=1)

    def vorticity(self, xy):
        """Analytic vorticity at points xy of shape (N, 2).

        Sign follows the Ghia et al. (1982) lid-vorticity convention (u_y - v_x).
        """
        g, y0, y1, ia, ic, m = self._kernels(xy)
        v0, v1 = ia * y0 + m * y1, ic * y1
        gx, gy = -g * v0, -g * v1
        u_y = (self.c_u[None, :] * gy).sum(1, keepdim=True)
        v_x = (self.c_v[None, :] * gx).sum(1, keepdim=True)
        return u_y - v_x


def load_model(ckpt_path, device="cpu"):
    """Load a trained checkpoint and return an evaluation-ready model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GSSplatEllip(n=ckpt["NKERNEL"], r_min=ckpt["r_min"], r_max=ckpt["r_max"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
