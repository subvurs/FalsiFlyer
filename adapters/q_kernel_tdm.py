"""Adapter: Q kernel-moat single-rate TDM falsification (#7j / #7k).

Demonstrates how to wire FalsiFlyer primitives to a concrete experiment.
The adapter is self-contained: it does NOT import any subvurs research
modules. The ``raw_Q`` slot is plugged in by the caller (so the adapter
itself is free of nyxnet dependencies).

Public surface:

* ``build_dataset(seed_cohort, regime_id) -> Dataset`` — generates and
  freezes a single-rate TDM dataset matching the #7j/#7k spec.
* ``build_harness(estimators, with_proportional=True) -> Harness`` —
  registers raw_NCA, stderr_Shrunk_NCA, MAP_Bayesian, and (optionally)
  MAP_Proportional, plus the cohort-prior builder.

Caller's responsibility: provide an Estimator named ``raw_Q`` (the
proprietary kernel) and pass it into ``build_harness``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from falsiflyer.harness import Estimator, Harness
from falsiflyer.prereg import DecisionRule, freeze_dataset
from falsiflyer.types import Cell, Dataset, Subject


# ---------------------------------------------------------------------------
# Locked dataset constants (mirrors q_kernel_tdm_dataset_gen[_7k].py)
# ---------------------------------------------------------------------------

FLOOR = 0.25
F_0_DEFAULT = 0.95
SIGMA_K = 0.30
N_SHOT = 100

K_POPS = [0.05, 0.10, 0.20]
N_VALUES = [2, 3, 4]
CV_VALUES = [0.10, 0.20, 0.30]
N_SUBJECTS_PER_CELL = 32

TIME_PATTERNS: Dict[int, List[float]] = {
    2: [1.0, 3.0],
    3: [0.5, 1.5, 3.0],
    4: [0.5, 1.0, 2.0, 3.5],
}
TIME_JITTER_FRAC = 0.05

COHORT_SEEDS_7J = [7, 42, 100, 314, 999]
COHORT_SEEDS_7K = [13, 71, 113, 271, 877]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _draw_k_e(k_pop: float, rng: np.random.Generator) -> float:
    return float(np.exp(np.log(k_pop) + rng.normal(0.0, SIGMA_K)))


def _make_subject(
    k_pop: float,
    n: int,
    cv: float,
    seed: int,
    F_0: float = F_0_DEFAULT,
) -> Subject:
    rng = np.random.default_rng(seed)
    k_e = _draw_k_e(k_pop, rng)
    A = F_0 - FLOOR
    nominal = np.array(TIME_PATTERNS[n], dtype=float) / k_pop
    jitter = rng.uniform(-TIME_JITTER_FRAC, TIME_JITTER_FRAC, size=n)
    t = nominal * (1.0 + jitter)
    t = np.maximum(t, 1e-6)
    F_true = FLOOR + A * np.exp(-k_e * t)
    F_obs = F_true + cv * F_true * rng.normal(size=n)
    F_obs = np.clip(F_obs, FLOOR + 1e-6, 1.0 - 1e-6)
    return Subject(
        subject_id="placeholder",  # filled in by caller
        ground_truth=k_e,
        payload={
            "t":      t.tolist(),
            "F_hat":  F_obs.tolist(),
            "F_true": F_true.tolist(),
            "N_shot": [N_SHOT] * n,
            "F_0":    F_0,
            "seed":   seed,
        },
    )


def _make_cell(
    k_pop: float,
    n: int,
    cv: float,
    cohort_seeds: List[int],
    F_0: float = F_0_DEFAULT,
) -> Cell:
    cell_id = f"k={k_pop:.2f}_n={n}_CV={cv:.2f}"
    subjects: List[Subject] = []
    for i in range(N_SUBJECTS_PER_CELL):
        seed = cohort_seeds[i % len(cohort_seeds)] * 1000 + i
        s = _make_subject(k_pop, n, cv, seed, F_0=F_0)
        s.subject_id = f"{cell_id}_s{i:02d}"
        subjects.append(s)
    is_stress = (n in (2, 3)) and (cv in (0.20, 0.30))
    return Cell(
        cell_id=cell_id,
        params={
            "k_pop":     k_pop,
            "n_samples": n,
            "cv":        cv,
        },
        subjects=subjects,
        is_stress_cell=is_stress,
    )


def build_dataset(
    *,
    cohort_seeds: Optional[List[int]] = None,
    regime_id: str = "7k",
    out_path: Optional[str] = None,
    F_0: float = F_0_DEFAULT,
) -> Tuple[Dataset, Optional[str]]:
    """Build and (optionally) freeze a single-rate TDM dataset.

    ``regime_id`` selects the canonical cohort seed schedule:

    * ``"7j"`` → ``[7, 42, 100, 314, 999]`` (original #7j)
    * ``"7k"`` → ``[13, 71, 113, 271, 877]`` (independent re-test #7k)
    * other   → caller MUST supply ``cohort_seeds`` directly.
    """
    if cohort_seeds is None:
        if regime_id == "7j":
            cohort_seeds = list(COHORT_SEEDS_7J)
        elif regime_id == "7k":
            cohort_seeds = list(COHORT_SEEDS_7K)
        else:
            raise ValueError(
                f"regime_id={regime_id!r} requires explicit cohort_seeds"
            )

    cells: List[Cell] = []
    for k_pop in K_POPS:
        for n in N_VALUES:
            for cv in CV_VALUES:
                cells.append(_make_cell(k_pop, n, cv, cohort_seeds, F_0=F_0))

    rule = DecisionRule(
        test_estimator="raw_Q",
        baselines=["raw_NCA", "stderr_Shrunk_NCA", "MAP_Bayesian", "MAP_Proportional"],
        tightening_threshold=0.15,
        pass_fraction=2.0 / 3.0,
        stress_predicate="n in {2, 3} AND CV in {0.20, 0.30}",
        metric="median_rel_err",
    )

    ds, written = freeze_dataset(
        schema_version=f"q_kernel_tdm_{regime_id}",
        cells=cells,
        decision_rule=rule,
        hash_fields=("t", "F_hat", "N_shot", "F_0", "seed"),
        out_path=out_path,
        cell_param_keys=["k_pop", "n_samples", "cv"],
        regime=f"single-rate TDM, regime_id={regime_id}",
        description="F(t) = FLOOR + A*exp(-k_e*t); proportional measurement noise.",
        constants={
            "FLOOR":               FLOOR,
            "F_0":                 F_0,
            "SIGMA_K":             SIGMA_K,
            "N_SHOT":              N_SHOT,
            "N_SUBJECTS_PER_CELL": N_SUBJECTS_PER_CELL,
            "COHORT_SEEDS":        cohort_seeds,
            "TIME_PATTERNS":       {str(k): v for k, v in TIME_PATTERNS.items()},
            "TIME_JITTER_FRAC":    TIME_JITTER_FRAC,
        },
        grid={
            "K_POPS":    K_POPS,
            "N_VALUES":  N_VALUES,
            "CV_VALUES": CV_VALUES,
        },
    )
    return ds, (str(written) if written is not None else None)


# ---------------------------------------------------------------------------
# Estimators (raw_NCA, stderr_Shrunk_NCA, MAP_Bayesian, MAP_Proportional)
# ---------------------------------------------------------------------------


def _fit_nca(t: np.ndarray, F_hat: np.ndarray) -> Dict[str, Any]:
    n = len(t)
    if n < 2:
        return {"k_nca": float("nan"), "stderr": float("nan"),
                "n_points": int(n), "log_residuals": []}
    y = F_hat - FLOOR
    y = np.clip(y, 1e-4, None)
    log_y = np.log(y)
    X = np.column_stack([np.ones_like(t), t])
    try:
        beta, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    except np.linalg.LinAlgError:
        return {"k_nca": float("nan"), "stderr": float("nan"),
                "n_points": int(n), "log_residuals": []}
    slope = float(beta[1])
    k_nca = -slope
    pred = X @ beta
    resid = log_y - pred
    if n >= 3:
        sigma2 = float(np.sum(resid ** 2)) / (n - 2)
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            stderr = float(math.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            stderr = float("nan")
    else:
        stderr = float("nan")
    return {
        "k_nca":         float(k_nca) if math.isfinite(k_nca) else float("nan"),
        "stderr":        stderr,
        "n_points":      int(n),
        "log_residuals": resid.tolist(),
    }


def _build_cohort(subjects: List[Subject], state: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute cohort priors over a cell. NO ground-truth leakage."""
    nca_records: List[Dict[str, Any]] = []
    for s in subjects:
        t = np.asarray(s.payload["t"], dtype=float)
        F_hat = np.asarray(s.payload["F_hat"], dtype=float)
        nca_records.append(_fit_nca(t, F_hat))

    raw_nca = [r["k_nca"] for r in nca_records]
    finite = [v for v in raw_nca if math.isfinite(v) and v > 0]

    # log-scale prior
    if len(finite) >= 2:
        log_arr = np.log(np.asarray(finite, dtype=float))
        mu_log_pop = float(np.mean(log_arr))
        tau2_log = float(np.var(log_arr, ddof=1))
        if not math.isfinite(tau2_log) or tau2_log <= 0:
            tau2_log = 0.05
    else:
        mu_log_pop = float("nan")
        tau2_log = float("nan")

    # linear-scale prior
    if len(finite) >= 2:
        pop_ke_lin = float(np.exp(np.mean(np.log(finite))))
        arr = np.asarray(finite, dtype=float)
        tau2_lin = float(np.mean((arr - pop_ke_lin) ** 2))
    else:
        pop_ke_lin = float("nan")
        tau2_lin = 0.0

    # pooled log-residual variance
    all_resids: List[float] = []
    for r in nca_records:
        all_resids.extend(r["log_residuals"])
    if len(all_resids) >= 3:
        sigma2_log_obs = max(float(np.var(np.asarray(all_resids), ddof=1)), 1e-4)
    else:
        sigma2_log_obs = 0.10

    # pooled proportional-noise scale
    F_0 = float(subjects[0].payload.get("F_0", F_0_DEFAULT))
    A = F_0 - FLOOR
    prop_resids: List[float] = []
    for s, rec in zip(subjects, nca_records):
        k = rec["k_nca"]
        if not (math.isfinite(k) and k > 0) or A <= 0:
            continue
        t = np.asarray(s.payload["t"], dtype=float)
        F_obs = np.asarray(s.payload["F_hat"], dtype=float)
        F_pred = FLOOR + A * np.exp(-k * t)
        F_pred = np.maximum(F_pred, FLOOR + 1e-6)
        prop_resids.extend(((F_obs - F_pred) / F_pred).tolist())
    if len(prop_resids) >= 3:
        sigma_prop = max(float(np.std(np.asarray(prop_resids), ddof=1)), 0.01)
    else:
        sigma_prop = 0.20

    # Stash per-subject NCA records keyed by id for the estimator callbacks
    nca_by_id = {s.subject_id: rec for s, rec in zip(subjects, nca_records)}

    return {
        "mu_log_pop":      mu_log_pop,
        "tau2_log":        tau2_log,
        "pop_ke_lin":      pop_ke_lin,
        "tau2_lin":        tau2_lin,
        "sigma2_log_obs":  sigma2_log_obs,
        "sigma_prop":      sigma_prop,
        "F_0":             F_0,
        "nca_by_id":       nca_by_id,
    }


def _est_raw_nca(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    return cohort["cohort"]["nca_by_id"][s.subject_id]["k_nca"]


def _est_shrunk_nca(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    c = cohort["cohort"]
    rec = c["nca_by_id"][s.subject_id]
    k_nca = rec["k_nca"]
    if not math.isfinite(k_nca):
        return float("nan")
    stderr = rec["stderr"]
    n_pts = rec["n_points"]
    pop_ke_lin = c["pop_ke_lin"]
    tau2_lin = c["tau2_lin"]
    degenerate = (
        not math.isfinite(stderr) or stderr <= 0.0 or n_pts < 3
    )
    if degenerate or tau2_lin <= 0.0:
        if tau2_lin <= 0.0:
            return float(k_nca)
        w = 0.5
        var_within = tau2_lin
    else:
        var_within = stderr * stderr
        w = var_within / (var_within + tau2_lin)
    return float((1.0 - w) * k_nca + w * pop_ke_lin)


def _est_map_bayesian(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        raise RuntimeError(
            "MAP_Bayesian requires scipy. Install with `pip install scipy` "
            "or `pip install falsiflyer[scipy]`."
        ) from e
    c = cohort["cohort"]
    mu_log_pop = c["mu_log_pop"]
    tau2_log = c["tau2_log"]
    sigma2_log_obs = c["sigma2_log_obs"]
    F_0 = c["F_0"]
    if not math.isfinite(mu_log_pop) or not math.isfinite(tau2_log):
        return float("nan")
    A = F_0 - FLOOR
    if A <= 0:
        return float("nan")
    log_A = math.log(A)
    t = np.asarray(s.payload["t"], dtype=float)
    F_hat = np.asarray(s.payload["F_hat"], dtype=float)
    y = np.log(np.clip(F_hat - FLOOR, 1e-4, None))

    def neg_log_post(log_k_e: float) -> float:
        k_e = math.exp(log_k_e)
        residuals = y - log_A + k_e * t
        ll = -0.5 / sigma2_log_obs * float(np.sum(residuals ** 2))
        lp = -0.5 / tau2_log * (log_k_e - mu_log_pop) ** 2
        return -(ll + lp)

    res = minimize_scalar(
        neg_log_post,
        bounds=(math.log(0.005), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(math.exp(res.x)) if res.success else float("nan")


def _est_map_proportional(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        raise RuntimeError(
            "MAP_Proportional requires scipy. Install with `pip install scipy` "
            "or `pip install falsiflyer[scipy]`."
        ) from e
    c = cohort["cohort"]
    mu_log_pop = c["mu_log_pop"]
    tau2_log = c["tau2_log"]
    sigma_prop = c["sigma_prop"]
    F_0 = c["F_0"]
    if (
        not math.isfinite(mu_log_pop)
        or not math.isfinite(tau2_log)
        or sigma_prop <= 0.0
    ):
        return float("nan")
    A = F_0 - FLOOR
    if A <= 0:
        return float("nan")
    t = np.asarray(s.payload["t"], dtype=float)
    F_hat = np.asarray(s.payload["F_hat"], dtype=float)

    def neg_log_post(log_k_e: float) -> float:
        k_e = math.exp(log_k_e)
        F_pred = FLOOR + A * np.exp(-k_e * t)
        F_pred = np.maximum(F_pred, FLOOR + 1e-6)
        sigma2 = (sigma_prop * F_pred) ** 2
        ll_neg = float(np.sum(0.5 * np.log(sigma2)
                              + 0.5 * (F_hat - F_pred) ** 2 / sigma2))
        lp_neg = 0.5 / tau2_log * (log_k_e - mu_log_pop) ** 2
        return ll_neg + lp_neg

    res = minimize_scalar(
        neg_log_post,
        bounds=(math.log(0.005), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(math.exp(res.x)) if res.success else float("nan")


# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------


def build_harness(
    *,
    raw_q: Estimator,
    with_proportional: bool = True,
) -> Harness:
    """Build a Harness with the four classical baselines + caller-provided raw_Q.

    Parameters
    ----------
    raw_q:
        The proprietary kernel under test. Must be a FalsiFlyer ``Estimator``
        named exactly ``"raw_Q"``.
    with_proportional:
        Include ``MAP_Proportional`` (the literal-DGP Bayesian baseline).
        Defaults to True (#7k spec). Set False to reproduce the #7j run.
    """
    if raw_q.name != "raw_Q":
        raise ValueError(
            f"raw_q.name must be 'raw_Q'; got {raw_q.name!r}"
        )
    estimators = {
        "raw_Q":             raw_q,
        "raw_NCA":           Estimator("raw_NCA", _est_raw_nca),
        "stderr_Shrunk_NCA": Estimator("stderr_Shrunk_NCA", _est_shrunk_nca),
        "MAP_Bayesian":      Estimator("MAP_Bayesian", _est_map_bayesian),
    }
    if with_proportional:
        estimators["MAP_Proportional"] = Estimator(
            "MAP_Proportional", _est_map_proportional,
        )
    return Harness(
        estimators=estimators,
        cohort_builders={"cohort": _build_cohort},
    )
