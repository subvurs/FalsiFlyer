"""Regenerate golden-replay fixtures for the q_kernel_tdm adapter.

Run from the repository root::

    PYTHONPATH=.:commercialization/path_c_nyxnet \\
        python3 tests/golden/_generate_fixtures.py

Writes:

* ``q_kernel_tdm_7j_baselines.json`` — locks the #7j dataset bytes and
  the per-cell median-rel-err of every classical baseline.  Independent
  of the proprietary kernel; replayable from any clean checkout.
* ``q_kernel_tdm_7k_baselines.json`` — same shape, regime_id=7k.
* ``q_kernel_tdm_7k_questimator.json`` — locks the FAIL verdict and the
  per-baseline pass counts under the real questimator kernel. Requires
  ``nyxnet`` on ``PYTHONPATH``.

The corresponding pytest ``test_golden_q_kernel_tdm.py`` replays each
fixture and asserts byte-equality.

This script is **not** a pytest module; running it should be a manual
audit-trail step (e.g. when porting the adapter to a new questimator
release).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict

# Make adapter and falsiflyer importable when invoked directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "falsiflyer"))
sys.path.insert(0, str(ROOT / "commercialization" / "path_c_nyxnet"))

from falsiflyer import Estimator  # noqa: E402
from falsiflyer.audit import hash_decision_rule  # noqa: E402
from adapters.q_kernel_tdm import build_dataset, build_harness  # noqa: E402


ROUND = 6  # decimal places for stable cross-platform diffs


def _round(x):
    if x is None:
        return None
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return round(x, ROUND)
    return x


def _nan_raw_q(s, cell, cohort) -> float:
    """raw_Q stub used for baseline-only fixtures (returns NaN)."""
    return float("nan")


def _build_baseline_fixture(regime_id: str) -> Dict:
    """Run harness with stub raw_Q and capture the classical-baseline
    per-cell median rel-err, per-cell pass/fail (where rule allows), and
    the dataset/decision-rule hashes."""
    work = Path(tempfile.mkdtemp(prefix=f"falsiflyer_golden_{regime_id}_"))
    out_path = work / f"frozen_{regime_id}.json"
    ds, _ = build_dataset(regime_id=regime_id, out_path=str(out_path))
    raw_q = Estimator("raw_Q", _nan_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    result = harness.run(ds, out_dir=None, verbose=False)

    cells_payload = []
    baseline_names = ("raw_NCA", "stderr_Shrunk_NCA", "MAP_Bayesian", "MAP_Proportional")
    for cs in result.cells:
        per_baseline = {}
        for est in baseline_names:
            score = cs.estimators.get(est)
            if score is None:
                continue
            per_baseline[est] = {
                "median_rel_err": _round(score.median_rel_err),
                "mean_rel_err":   _round(score.mean_rel_err),
                "n_finite":       int(score.n_finite),
                "n_total":        int(score.n_total),
            }
        cells_payload.append({
            "cell_id":        cs.cell_id,
            "is_stress_cell": bool(cs.is_stress_cell),
            "n_subjects":     int(cs.n_subjects),
            "baselines":      per_baseline,
        })

    return {
        "regime_id":          regime_id,
        "schema_version":     ds.schema_version,
        "dataset_sha256":     ds.sha256_data_payload,
        "decision_rule_hash": hash_decision_rule(ds.decision_rule),
        "n_cells":            ds.n_cells,
        "n_stress_cells":     ds.n_stress_cells,
        "n_subjects":         ds.n_subjects,
        "cells":              cells_payload,
    }


def _build_questimator_fixture(regime_id: str = "7k") -> Dict:
    """Run harness with the real estimate_gamma(method='questimator')
    and capture the FAIL verdict + per-baseline pass counts."""
    from nyxnet.questimator import estimate_gamma  # noqa: WPS433

    def _real_raw_q(s, cell, cohort) -> float:
        return float(estimate_gamma(
            probe_times=s.payload["t"],
            probe_fidelities=s.payload["F_hat"],
            probe_shots=s.payload["N_shot"],
            F_0=s.payload["F_0"],
            method="questimator",
        ))

    work = Path(tempfile.mkdtemp(prefix=f"falsiflyer_golden_q_{regime_id}_"))
    out_path = work / f"frozen_{regime_id}.json"
    ds, _ = build_dataset(regime_id=regime_id, out_path=str(out_path))
    raw_q = Estimator("raw_Q", _real_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    result = harness.run(ds, out_dir=None, verbose=False)
    v = result.verdict

    return {
        "regime_id":            regime_id,
        "schema_version":       ds.schema_version,
        "dataset_sha256":       ds.sha256_data_payload,
        "decision_rule_hash":   hash_decision_rule(ds.decision_rule),
        "kernel":               "nyxnet.questimator.estimate_gamma(method='questimator')",
        "n_stress_cells":       int(v.n_stress_cells),
        "required_n_passing":   int(v.required_n_passing),
        "n_pass_all_baselines": int(v.n_pass_all_baselines),
        "n_pass_each_baseline": dict(v.n_pass_each_baseline),
        "verdict_pass":         bool(v.verdict_pass),
    }


def _write(payload, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> int:
    print("[golden] regenerating baseline-only fixtures …")
    for regime in ("7j", "7k"):
        payload = _build_baseline_fixture(regime)
        _write(payload, HERE / f"q_kernel_tdm_{regime}_baselines.json")

    try:
        print("[golden] regenerating questimator parity fixture …")
        payload = _build_questimator_fixture("7k")
        _write(payload, HERE / "q_kernel_tdm_7k_questimator.json")
    except ImportError as e:
        print(f"[golden] skipped questimator fixture (no nyxnet): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
