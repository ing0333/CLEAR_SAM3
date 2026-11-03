"""Simple transductive graph correction (in the spirit of RS-TransCLIP, much simplified).

Idea: images that look alike should get the same label. Build a k-NN graph over the test
batch's image features and propagate the zero-shot class probabilities along it:
    Z <- (1 - alpha) * softmax(L / T) + alpha * W Z
This is *not* a re-implementation of RS-TransCLIP (which solves a regularised clustering
objective with class-balance and text terms); it is a minimal graph baseline to compare
against PPR, which uses only the label-prior assumption and no image-image similarity.
"""
from __future__ import annotations

import numpy as np


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def knn_graph(F: np.ndarray, k: int = 10) -> np.ndarray:
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(F) - 1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    W = np.zeros_like(S)
    rows = np.arange(len(F))[:, None]
    W[rows, idx] = np.maximum(S[rows, idx], 0)
    W = (W + W.T) / 2
    return W / (W.sum(1, keepdims=True) + 1e-9)


def label_propagation(L: np.ndarray, F: np.ndarray, alpha: float = 0.5, k: int = 10,
                      T: float = 1.0, iters: int = 30) -> np.ndarray:
    """L: [N, C] logits; F: [N, D] image features. Returns log-probabilities [N, C]."""
    Z0 = _softmax(L / T)
    W = knn_graph(F, k)
    Z = Z0.copy()
    for _ in range(iters):
        Z = (1 - alpha) * Z0 + alpha * (W @ Z)
    return np.log(Z + 1e-12)


def transclip_lite(L: np.ndarray, F: np.ndarray, T: float = 1.0, lam: float = 1.0,
                   tau: float = 0.05, iters: int = 10) -> np.ndarray:
    """Closer (still simplified) stand-in for RS-TransCLIP: EM over the test batch.

    Each class gets a prototype mu_c re-estimated from the image features of the images
    currently assigned to it; assignments combine the zero-shot text prior with the
    image-prototype affinity (a GMM with shared isotropic variance and uniform mixing):
        z_i ∝ softmax(L_i / T) * exp(lam * <F_i, mu_c> / tau)
    Unlike TransLP this *moves* class centres, so it can undo a systematic text bias.
    No Laplacian term and no learned covariance (RS-TransCLIP has both).
    """
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    logZ0 = np.log(_softmax(L / T) + 1e-12)
    Z = np.exp(logZ0)
    for _ in range(iters):
        mu = Z.T @ F
        mu = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
        Z = _softmax(logZ0 + lam * (F @ mu.T) / tau)
    return np.log(Z + 1e-12)
