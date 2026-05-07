"""Tests for the proper-baseline NoiseModel suite."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falsiflyer import (
    BaselineLibrary,
    Binomial,
    Gaussian,
    LogGaussian,
    Poisson,
    Proportional,
)


def test_gaussian_log_likelihood_and_simulate():
    rng = np.random.default_rng(0)
    g = Gaussian()
    mu = np.array([1.0, 2.0, 3.0])
    y = g.simulate(mu, {"sigma": 0.5}, rng)
    assert y.shape == mu.shape
    ll_at_mu = g.log_likelihood(mu, mu, {"sigma": 0.5})
    ll_far = g.log_likelihood(mu + 10.0, mu, {"sigma": 0.5})
    assert math.isfinite(ll_at_mu)
    assert ll_at_mu > ll_far  # closer is more likely

    # bad sigma
    assert g.log_likelihood(mu, mu, {"sigma": 0.0}) == float("-inf")


def test_proportional_heteroskedastic():
    rng = np.random.default_rng(1)
    p = Proportional(floor=0.0)
    mu = np.array([1.0, 10.0, 100.0])
    sigma_prop = 0.2

    samples = np.stack([p.simulate(mu, {"sigma_prop": sigma_prop}, rng) for _ in range(2000)])
    # Std at large mu should be approx sigma_prop * mu (within sampling).
    emp_std = samples.std(axis=0)
    expected = sigma_prop * mu
    rel = np.abs(emp_std - expected) / expected
    assert np.all(rel < 0.1)

    ll = p.log_likelihood(mu, mu, {"sigma_prop": sigma_prop})
    assert math.isfinite(ll)


def test_poisson_simulate_and_loglikelihood():
    rng = np.random.default_rng(2)
    P = Poisson()
    mu = np.array([5.0, 10.0])
    samples = np.stack([P.simulate(mu, {}, rng) for _ in range(2000)])
    # Mean should approach mu.
    assert np.allclose(samples.mean(axis=0), mu, atol=0.5)

    ll = P.log_likelihood(np.array([5, 10]), mu, {})
    assert math.isfinite(ll)


def test_log_gaussian_invariant_at_mu():
    L = LogGaussian(floor=0.25, eps=1e-4)
    mu = np.array([0.5, 0.6, 0.7])
    ll = L.log_likelihood(mu, mu, {"sigma2_log": 0.05})
    # At y == mu, the squared-residual term is zero; ll = -0.5 * n * log(2 pi sigma2).
    n = mu.size
    expected = -0.5 * n * math.log(2.0 * math.pi * 0.05)
    assert abs(ll - expected) < 1e-9


def test_binomial_log_likelihood_shape_check():
    B = Binomial()
    y = np.array([3, 7, 1])
    mu = np.array([3.0, 7.0, 1.5])
    N_shot = np.array([10, 10, 10])
    total = np.array([10.0, 10.0, 10.0])
    ll = B.log_likelihood(y, mu, {"N_shot": N_shot, "total": total})
    assert math.isfinite(ll)

    with pytest.raises(ValueError):
        B.log_likelihood(
            y, mu,
            {"N_shot": np.array([10, 10]), "total": total},
        )


def test_baseline_library_default_is_complete():
    lib = BaselineLibrary.default()
    assert set(lib.names()) == {
        "gaussian", "proportional", "poisson", "log_gaussian", "binomial",
    }
    assert isinstance(lib.get("proportional"), Proportional)
    with pytest.raises(KeyError):
        lib.get("not_a_model")


def test_baseline_library_no_double_register():
    lib = BaselineLibrary()
    lib.register("gaussian", Gaussian())
    with pytest.raises(ValueError):
        lib.register("gaussian", Gaussian())
