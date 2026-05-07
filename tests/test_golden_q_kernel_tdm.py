"""Golden-replay tests for the q_kernel_tdm adapter.

Two layers of replay:

1. ``test_baselines_replay_*`` — runs the harness with a stub raw_Q and
   asserts that the four classical baselines produce byte-identical
   per-cell median rel-err to the recorded fixture. No proprietary
   kernel needed; this is the pure dataset + adapter-baseline parity.

2. ``test_questimator_replay`` — gated on ``nyxnet.questimator`` being
   importable. Runs with the real ``estimate_gamma(method='questimator')``
   and asserts the FAIL verdict + per-baseline pass counts.

Regenerate fixtures with::

    PYTHONPATH=.:commercialization/path_c_nyxnet \\
        python3 tests/golden/_generate_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

from falsiflyer import Estimator
from falsiflyer.audit import hash_decision_rule
from adapters.q_kernel_tdm import build_dataset, build_harness


GOLDEN_DIR = Path(__file__).parent / "golden"
ROUND = 6


def _round(x):
    if x is None:
        return None
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return round(x, ROUND)
    return x


def _nan_raw_q(s, cell, cohort) -> float:
    return float("nan")


@pytest.mark.parametrize("regime_id", ["7j", "7k"])
def test_baselines_replay(regime_id, tmp_path):
    fixture_path = GOLDEN_DIR / f"q_kernel_tdm_{regime_id}_baselines.json"
    expected = json.loads(fixture_path.read_text())

    out_path = tmp_path / f"frozen_{regime_id}.json"
    ds, _ = build_dataset(regime_id=regime_id, out_path=str(out_path))

    # Structural invariants.
    assert ds.schema_version == expected["schema_version"]
    assert ds.sha256_data_payload == expected["dataset_sha256"], (
        f"dataset SHA-256 drift on regime {regime_id}: "
        f"got {ds.sha256_data_payload}, expected {expected['dataset_sha256']}"
    )
    assert hash_decision_rule(ds.decision_rule) == expected["decision_rule_hash"]
    assert ds.n_cells == expected["n_cells"]
    assert ds.n_stress_cells == expected["n_stress_cells"]
    assert ds.n_subjects == expected["n_subjects"]

    # Run with stub raw_Q; only the 4 classical baselines matter here.
    raw_q = Estimator("raw_Q", _nan_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    result = harness.run(ds, out_dir=None, verbose=False)

    expected_by_cell = {c["cell_id"]: c for c in expected["cells"]}
    assert {c.cell_id for c in result.cells} == set(expected_by_cell.keys())

    for cs in result.cells:
        ec = expected_by_cell[cs.cell_id]
        assert cs.is_stress_cell == ec["is_stress_cell"]
        assert cs.n_subjects == ec["n_subjects"]
        for est_name, expected_score in ec["baselines"].items():
            actual = cs.estimators.get(est_name)
            assert actual is not None, (
                f"missing estimator {est_name!r} on cell {cs.cell_id}"
            )
            assert _round(actual.median_rel_err) == expected_score["median_rel_err"], (
                f"{cs.cell_id} / {est_name} median_rel_err drift: "
                f"got {actual.median_rel_err}, "
                f"expected {expected_score['median_rel_err']}"
            )
            assert _round(actual.mean_rel_err) == expected_score["mean_rel_err"]
            assert int(actual.n_finite) == expected_score["n_finite"]
            assert int(actual.n_total) == expected_score["n_total"]


def _nyxnet_available() -> bool:
    try:
        # The questimator lives outside the FalsiFlyer package; check the
        # PYTHONPATH-extended search before importing.
        import importlib

        if importlib.util.find_spec("nyxnet.questimator") is None:
            return False
        import nyxnet.questimator  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _nyxnet_available(),
    reason=(
        "nyxnet.questimator not importable; set "
        "PYTHONPATH=commercialization/path_c_nyxnet to run this replay."
    ),
)
def test_questimator_replay(tmp_path):
    fixture_path = GOLDEN_DIR / "q_kernel_tdm_7k_questimator.json"
    expected = json.loads(fixture_path.read_text())

    from nyxnet.questimator import estimate_gamma  # type: ignore

    def _real_raw_q(s, cell, cohort) -> float:
        return float(estimate_gamma(
            probe_times=s.payload["t"],
            probe_fidelities=s.payload["F_hat"],
            probe_shots=s.payload["N_shot"],
            F_0=s.payload["F_0"],
            method="questimator",
        ))

    out_path = tmp_path / "frozen_7k.json"
    ds, _ = build_dataset(regime_id="7k", out_path=str(out_path))

    assert ds.sha256_data_payload == expected["dataset_sha256"]
    assert hash_decision_rule(ds.decision_rule) == expected["decision_rule_hash"]

    raw_q = Estimator("raw_Q", _real_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    result = harness.run(ds, out_dir=None, verbose=False)
    v = result.verdict

    assert v.verdict_pass is expected["verdict_pass"]
    assert int(v.n_stress_cells) == expected["n_stress_cells"]
    assert int(v.required_n_passing) == expected["required_n_passing"]
    assert int(v.n_pass_all_baselines) == expected["n_pass_all_baselines"]
    assert dict(v.n_pass_each_baseline) == expected["n_pass_each_baseline"]
