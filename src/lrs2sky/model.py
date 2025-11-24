from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import Ridge

__all__ = [
    "fit_basis",
    "fit_coeff_models",
    "predict_coeffs",
    "reconstruct_spectrum",
]


@dataclass
class Basis:
    mu: np.ndarray
    components: np.ndarray  # shape (n_components, n_wave)
    method: str = "pca"

    def project(self, X: np.ndarray) -> np.ndarray:
        X0 = X - self.mu
        B = self.components
        return X0 @ B.T  # coeffs shape (n_samples, n_components)

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self.mu + coeffs @ self.components


@dataclass
class CoeffModels:
    betas: Dict[int, np.ndarray]
    intercepts: Dict[int, float]
    alphas: Dict[int, float]
    feature_names: Tuple[str, ...]


def fit_basis(X: np.ndarray, method: str = "pca", n_components: int = 8) -> Basis:
    """Fit a low-dimensional basis to spectra X (rows are spectra)."""
    mu = np.nanmean(X, axis=0)
    X0 = np.nan_to_num(X - mu, nan=0.0)
    if method.lower() == "pca":
        pca = PCA(n_components=n_components)
        pca.fit(X0)
        components = pca.components_  # (n_components, n_wave)
    elif method.lower() == "nmf":
        Xpos = X - np.nanmin(X)
        nmf = NMF(n_components=n_components, init="nndsvda", max_iter=500)
        W = nmf.fit_transform(np.nan_to_num(Xpos, nan=0.0))
        H = nmf.components_  # (n_components, n_wave)
        components = H
        mu = np.zeros(X.shape[1])
    else:
        raise ValueError("Unknown method: %s" % method)
    return Basis(mu=mu, components=components, method=method)


def _design_matrix(labels: np.ndarray) -> np.ndarray:
    """Build feature matrix [1, x1, x2, ...] from standardized labels."""
    n = labels.shape[0]
    return np.hstack([np.ones((n, 1)), labels])


def fit_coeff_models(labels: np.ndarray, coeffs: np.ndarray, alpha: float = 1.0, feature_names: Optional[Tuple[str, ...]] = None) -> CoeffModels:
    """Fit ridge regression for each coefficient column as a function of labels.

    labels: (n_samples, n_features) standardized.
    coeffs: (n_samples, n_components)
    """
    X = _design_matrix(labels)
    betas = {}
    intercepts = {}
    alphas = {}
    for m in range(coeffs.shape[1]):
        y = coeffs[:, m]
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        betas[m] = model.coef_.astype(float)
        intercepts[m] = 0.0  # intercept is included in X
        alphas[m] = float(alpha)
    return CoeffModels(betas=betas, intercepts=intercepts, alphas=alphas, feature_names=tuple(feature_names) if feature_names else tuple(f"x{i}" for i in range(labels.shape[1])))


def predict_coeffs(models: CoeffModels, labels: np.ndarray) -> np.ndarray:
    X = _design_matrix(labels)
    M = len(models.betas)
    C = np.empty((labels.shape[0], M), dtype=float)
    for m in range(M):
        beta = models.betas[m]
        C[:, m] = X @ beta
    return C


def reconstruct_spectrum(basis: Basis, coeffs: np.ndarray) -> np.ndarray:
    return basis.reconstruct(coeffs)
