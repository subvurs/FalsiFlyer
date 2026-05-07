"""Proper-baseline (DGP-matched) noise templates.

The #7j → #7k pivot turned on a single observation: the existing
log-scale ``MAP_Bayesian`` baseline was misspecified for proportional
noise, so its losses against ``raw_Q`` over-credited the kernel.  Adding
``MAP_Proportional`` (literal-DGP Bayesian, Gaussian on F-scale with
point-dependent variance) closed the moat.

This module ships the five canonical noise models so the next experiment
can pre-register a literal-DGP Bayesian baseline without re-deriving its
log-likelihood.

Each ``NoiseModel`` exposes:

* ``log_likelihood(y, mu, params) -> float`` — sum of log-density
* ``simulate(mu, params, rng) -> np.ndarray`` — generative form (for
  diagnostics / sanity sweeps; not used by the harness directly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# NoiseModel base
# ---------------------------------------------------------------------------


class NoiseModel:
    """Abstract noise model. Subclasses implement log_likelihood + simulate."""

    name: str = "abstract"

    def log_likelihood(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        raise NotImplementedError

    def simulate(
        self,
        mu: np.ndarray,
        params: Optional[Dict[str, Any]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 5 proper-baseline noise models
# ---------------------------------------------------------------------------


@dataclass
class Gaussian(NoiseModel):
    """y_i ~ Normal(mu_i, sigma^2). sigma is a free param."""

    name: str = "gaussian"

    def log_likelihood(self, y, mu, params=None):
        params = params or {}
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            return float("-inf")
        r = np.asarray(y) - np.asarray(mu)
        n = r.size
        return float(
            -0.5 * n * np.log(2.0 * np.pi * sigma * sigma)
            - 0.5 * np.sum(r * r) / (sigma * sigma)
        )

    def simulate(self, mu, params, rng):
        sigma = float((params or {}).get("sigma", 1.0))
        return np.asarray(mu) + sigma * rng.normal(size=np.asarray(mu).shape)


@dataclass
class Proportional(NoiseModel):
    """y_i ~ Normal(mu_i, (sigma_prop * mu_i)^2). Heteroskedastic.

    Matches the #7k DGP exactly: ``y = mu + sigma_prop * mu * N(0,1)``.
    """

    name: str = "proportional"
    floor: float = 0.0  # support floor for mu (mu must exceed floor + eps)

    def log_likelihood(self, y, mu, params=None):
        params = params or {}
        sigma_prop = float(params.get("sigma_prop", 0.2))
        if sigma_prop <= 0:
            return float("-inf")
        mu_arr = np.maximum(np.asarray(mu), self.floor + 1e-9)
        sigma2 = (sigma_prop * mu_arr) ** 2
        r = np.asarray(y) - mu_arr
        # sum( -0.5 * log(2 pi sigma^2) - 0.5 * r^2 / sigma^2 )
        return float(
            -0.5 * np.sum(np.log(2.0 * np.pi * sigma2))
            - 0.5 * np.sum(r * r / sigma2)
        )

    def simulate(self, mu, params, rng):
        sigma_prop = float((params or {}).get("sigma_prop", 0.2))
        mu_arr = np.asarray(mu)
        return mu_arr + sigma_prop * mu_arr * rng.normal(size=mu_arr.shape)


@dataclass
class Poisson(NoiseModel):
    """y_i ~ Poisson(mu_i). y must be non-negative integer-valued."""

    name: str = "poisson"

    def log_likelihood(self, y, mu, params=None):
        from math import lgamma

        y_arr = np.asarray(y, dtype=float)
        mu_arr = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        # sum( y log(mu) - mu - lgamma(y+1) )
        ll = np.sum(y_arr * np.log(mu_arr) - mu_arr)
        ll -= float(sum(lgamma(yi + 1.0) for yi in y_arr.flatten()))
        return float(ll)

    def simulate(self, mu, params, rng):
        return rng.poisson(lam=np.maximum(np.asarray(mu), 0.0))


@dataclass
class LogGaussian(NoiseModel):
    """log(y_i - floor) ~ Normal(log(mu_i - floor), sigma2_log).

    The log-scale MAP baseline used in #7j; included for legacy parity
    runs, NOT recommended as the literal-DGP baseline for proportional or
    Poisson noise.
    """

    name: str = "log_gaussian"
    floor: float = 0.0
    eps: float = 1e-4

    def log_likelihood(self, y, mu, params=None):
        params = params or {}
        sigma2_log = float(params.get("sigma2_log", 0.05))
        if sigma2_log <= 0:
            return float("-inf")
        y_arr = np.clip(np.asarray(y) - self.floor, self.eps, None)
        mu_arr = np.clip(np.asarray(mu) - self.floor, self.eps, None)
        r = np.log(y_arr) - np.log(mu_arr)
        n = r.size
        return float(
            -0.5 * n * np.log(2.0 * np.pi * sigma2_log)
            - 0.5 * np.sum(r * r) / sigma2_log
        )

    def simulate(self, mu, params, rng):
        params = params or {}
        sigma2_log = float(params.get("sigma2_log", 0.05))
        mu_arr = np.maximum(np.asarray(mu) - self.floor, self.eps)
        log_y = np.log(mu_arr) + np.sqrt(sigma2_log) * rng.normal(size=mu_arr.shape)
        return np.exp(log_y) + self.floor


@dataclass
class Binomial(NoiseModel):
    """y_i ~ Binomial(N_shot_i, mu_i / total_i).

    Ratio model used by Q-style shot-noise likelihoods. ``params`` MUST
    carry per-point ``N_shot`` and ``total`` arrays (same shape as y),
    where ``mu_i / total_i`` is the success probability.
    """

    name: str = "binomial"

    def log_likelihood(self, y, mu, params=None):
        from math import lgamma

        params = params or {}
        N_shot = np.asarray(params.get("N_shot"), dtype=float)
        total = np.asarray(params.get("total"), dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if N_shot.shape != y_arr.shape or total.shape != y_arr.shape:
            raise ValueError("Binomial: N_shot and total must match y in shape")
        p = np.clip(mu_arr / np.maximum(total, 1e-12), 1e-9, 1.0 - 1e-9)
        # log C(N, k) + k log p + (N-k) log(1-p)
        ll = 0.0
        for ki, ni, pi in zip(y_arr.flatten(), N_shot.flatten(), p.flatten()):
            ll += (
                lgamma(ni + 1.0)
                - lgamma(ki + 1.0)
                - lgamma(ni - ki + 1.0)
                + ki * np.log(pi)
                + (ni - ki) * np.log(1.0 - pi)
            )
        return float(ll)

    def simulate(self, mu, params, rng):
        params = params or {}
        N_shot = np.asarray(params.get("N_shot"), dtype=float)
        total = np.asarray(params.get("total"), dtype=float)
        p = np.clip(np.asarray(mu) / np.maximum(total, 1e-12), 0.0, 1.0)
        return rng.binomial(n=N_shot.astype(int), p=p)


# ---------------------------------------------------------------------------
# BaselineLibrary — registry + factory for proper-baseline estimators
# ---------------------------------------------------------------------------


class BaselineLibrary:
    """Registry of (name → NoiseModel) for use by harness adapters.

    The library does not run estimators directly; it is a discoverable
    catalogue of literal-DGP noise models that an adapter can pair with
    its own MAP/MLE optimizer to build a proper baseline.

    Usage::

        lib = BaselineLibrary.default()
        prop = lib.get("proportional")
        ll = prop.log_likelihood(y, mu, params={"sigma_prop": 0.2})
    """

    def __init__(self) -> None:
        self._models: Dict[str, NoiseModel] = {}

    def register(self, name: str, model: NoiseModel) -> None:
        if name in self._models:
            raise ValueError(f"BaselineLibrary: {name!r} already registered")
        self._models[name] = model

    def get(self, name: str) -> NoiseModel:
        if name not in self._models:
            raise KeyError(
                f"Unknown noise model {name!r}; available: {sorted(self._models)}"
            )
        return self._models[name]

    def names(self) -> List[str]:
        return sorted(self._models)

    @classmethod
    def default(cls, *, floor: float = 0.0) -> "BaselineLibrary":
        """Library pre-populated with the five canonical noise models."""
        lib = cls()
        lib.register("gaussian", Gaussian())
        lib.register("proportional", Proportional(floor=floor))
        lib.register("poisson", Poisson())
        lib.register("log_gaussian", LogGaussian(floor=floor))
        lib.register("binomial", Binomial())
        return lib
