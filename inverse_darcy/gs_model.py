"""Gaussian-splatting surrogate for the 2D inverse Darcy problem (evaluation only).

Given pressure observations, the model jointly represents the pressure field U
and the unknown permeability field nu as sums of isotropic Gaussian kernels.
This file only loads a trained checkpoint and evaluates the two fields on a grid.
"""
import numpy as np
import torch


class DarcyGS:
    """Evaluation-only Gaussian-splatting fields built from trained parameters."""

    def __init__(self, mu, sigma, coeffs_p, coeffs_s):
        self.mu = np.asarray(mu)              # (n, 2) kernel centers
        self.sigma = np.asarray(sigma)        # (n,)   kernel radii
        self.coeffs_p = np.asarray(coeffs_p)  # (n,)   pressure coefficients
        self.coeffs_s = np.asarray(coeffs_s)  # (n,)   permeability coefficients

    def _kernels(self, xy):
        diff = xy[:, None, :] - self.mu[None, :, :]
        sq = (diff ** 2).sum(-1)
        return np.exp(-0.5 * sq / (self.sigma[None, :] ** 2))

    def permeability(self, xy):
        """Permeability field nu = sum_i coeffs_s_i * G_i."""
        return (self.coeffs_s[None, :] * self._kernels(xy)).sum(1)

    def pressure(self, xy):
        """Pressure field U = x(1-x)y(1-y) * sum_i coeffs_p_i * G_i (zero on the boundary)."""
        x, y = xy[:, 0], xy[:, 1]
        bump = x * (1 - x) * y * (1 - y)
        return bump * (self.coeffs_p[None, :] * self._kernels(xy)).sum(1)

    def fields_on_grid(self, res=128):
        ax = np.linspace(0, 1, res)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        xy = np.stack([X.ravel(), Y.ravel()], 1)
        nu = self.permeability(xy).reshape(res, res)
        U = self.pressure(xy).reshape(res, res)
        return nu, U


def load_model(ckpt_path):
    """Load a trained checkpoint -> (DarcyGS model, a1, a2)."""
    ck = torch.load(ckpt_path, map_location="cpu")
    model = DarcyGS(ck["mu"].numpy(), ck["sigma"].numpy(),
                    ck["coeffs_p"].numpy(), ck["coeffs_s"].numpy())
    return model, ck["a1"], ck["a2"]


def truth_fields(a1, a2, res=128):
    """Analytic ground truth: nu = 1 + 0.5 sin(a1 pi x) sin(a2 pi y), U = sin(pi x) sin(pi y)."""
    ax = np.linspace(0, 1, res)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    nu = 1.0 + 0.5 * np.sin(a1 * np.pi * X) * np.sin(a2 * np.pi * Y)
    U = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return nu, U
