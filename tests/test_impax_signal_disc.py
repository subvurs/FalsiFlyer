"""Tests for adapters/impax_signal_disc.py.

Mirrors the structure of the q_kernel_tdm tests:

* freeze + reload round-trip (proves the canonical hash is stable)
* freeze + mutate + reload (proves drift is detected)
* synthetic-truth oracle and random estimators get the verdict right
* cohort builder is ground-truth-blind (no leakage)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from falsiflyer.errors import HashMismatchError
from falsiflyer.harness import Estimator
from falsiflyer.prereg import load_frozen_dataset

from adapters.impax_signal_disc import (
    N_SUBJECTS_PER_CELL,
    SEED_FAMILIES,
    SNR_VALUES,
    N_BINS_VALUES,
    _build_cohort,
    _est_bandpass_energy,
    _est_matched_filter,
    _est_periodogram,
    build_dataset,
    build_harness,
)


# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------


def test_dataset_shape_default():
    ds, _ = build_dataset(regime_id="default")
    assert len(ds.cells) == len(SNR_VALUES) * len(N_BINS_VALUES) * len(SEED_FAMILIES)
    assert len(ds.cells) == 27
    n_stress = sum(1 for c in ds.cells if c.is_stress_cell)
    assert n_stress == 12  # snr ≤ 1.0 AND n_bins ≥ 16 → 2×2×3 = 12
    n_subjects = sum(len(c.subjects) for c in ds.cells)
    assert n_subjects == 27 * N_SUBJECTS_PER_CELL == 864
    # Decision rule embedded in the dataset
    rule = ds.decision_rule
    assert rule["test_estimator"] == "impax_classical"
    assert set(rule["baselines"]) == {
        "raw_periodogram", "raw_matched_filter", "raw_bandpass_energy",
    }
    assert rule["metric"] == "median_rel_err"
    assert rule["tightening_threshold"] == 0.30


def test_dataset_replay_regime_differs():
    ds_a, _ = build_dataset(regime_id="default")
    ds_b, _ = build_dataset(regime_id="replay")
    assert ds_a.sha256_data_payload != ds_b.sha256_data_payload


def test_dataset_unknown_regime_requires_seeds():
    with pytest.raises(ValueError, match="explicit cohort_seeds"):
        build_dataset(regime_id="something_else")


# ---------------------------------------------------------------------------
# Hash invariants
# ---------------------------------------------------------------------------


def test_freeze_reload_roundtrip(tmp_path: Path):
    out = tmp_path / "impax_default.json"
    ds, written = build_dataset(regime_id="default", out_path=str(out))
    assert written is not None
    assert Path(written).exists()
    reloaded = load_frozen_dataset(str(out))
    assert reloaded.sha256_data_payload == ds.sha256_data_payload
    assert len(reloaded.cells) == len(ds.cells)


def test_mutated_signal_byte_rejected(tmp_path: Path):
    out = tmp_path / "impax_mutate.json"
    ds, _ = build_dataset(regime_id="default", out_path=str(out))
    raw = json.loads(out.read_text())
    # Flip one signal byte in cell 0, subject 0
    sig = raw["cells"][0]["subjects"][0]["payload"]["signal"]
    sig[0] = sig[0] + 1.0e-3
    out.write_text(json.dumps(raw))
    with pytest.raises(HashMismatchError):
        load_frozen_dataset(str(out))


def test_mutated_decision_rule_changes_rule_hash():
    """Mutating the decision rule does NOT change the data SHA (rule is
    not part of the data payload), but the audit-level
    ``hash_decision_rule`` MUST flip — that's how threshold-tuning shows
    up in the audit ledger.
    """
    from falsiflyer.audit import hash_decision_rule
    from falsiflyer.prereg import DecisionRule

    ds, _ = build_dataset(regime_id="default")
    rule_a = DecisionRule.deserialize(ds.decision_rule)
    rule_b = rule_a.model_copy(update={"tightening_threshold": 0.05})
    assert hash_decision_rule(rule_a.serialize()) != hash_decision_rule(rule_b.serialize())


# ---------------------------------------------------------------------------
# Estimator semantics
# ---------------------------------------------------------------------------


def _oracle(s, cell, cohort):
    return float(s.payload["true_freq"])


def _random_candidate(s, cell, cohort):
    rng = np.random.default_rng(int(s.payload["seed"]) + 7)
    cands = s.payload["candidate_freqs"]
    return float(cands[int(rng.integers(0, len(cands)))])


def test_oracle_passes_all_stress_cells():
    ds, _ = build_dataset(regime_id="default")
    h = build_harness(impax_classical=Estimator("impax_classical", _oracle))
    res = h.run(ds)
    assert res.verdict.verdict_pass is True
    assert res.verdict.n_pass_all_baselines == res.verdict.n_stress_cells == 12
    for b in ("raw_periodogram", "raw_matched_filter", "raw_bandpass_energy"):
        assert res.verdict.n_pass_each_baseline[b] == 12


def test_random_fails_decision_rule():
    ds, _ = build_dataset(regime_id="default")
    h = build_harness(impax_classical=Estimator("impax_classical", _random_candidate))
    res = h.run(ds)
    assert res.verdict.verdict_pass is False
    assert res.verdict.n_pass_all_baselines == 0


def test_baselines_have_finite_rel_err():
    """All three classical baselines should produce finite rel-err on every cell."""
    ds, _ = build_dataset(regime_id="default")
    h = build_harness(impax_classical=Estimator("impax_classical", _oracle))
    res = h.run(ds)
    for cs in res.cells:
        for bn in ("raw_periodogram", "raw_matched_filter", "raw_bandpass_energy"):
            es = cs.estimators[bn]
            assert es.n_finite > 0
            assert np.isfinite(es.median_rel_err)
            assert es.median_rel_err >= 0.0


def test_matched_filter_strictly_dominates_random():
    """Sanity: matched filter should be much more accurate than random picks."""
    ds, _ = build_dataset(regime_id="default")
    h = build_harness(impax_classical=Estimator("impax_classical", _random_candidate))
    res = h.run(ds)
    # On every stress cell, matched_filter med_rel_err < impax_classical (random) med_rel_err
    for cs in res.cells:
        if not cs.is_stress_cell:
            continue
        mf = cs.estimators["raw_matched_filter"].median_rel_err
        rnd = cs.estimators["impax_classical"].median_rel_err
        assert mf < rnd, (
            f"matched filter ({mf:.3f}) should beat random ({rnd:.3f}) on {cs.cell_id}"
        )


def test_harness_rejects_wrong_test_name():
    with pytest.raises(ValueError, match="impax_classical"):
        build_harness(impax_classical=Estimator("wrong_name", _oracle))


# ---------------------------------------------------------------------------
# Cohort builder no-leakage
# ---------------------------------------------------------------------------


def test_cohort_builder_does_not_read_ground_truth():
    """The cohort builder must not consult subject.ground_truth.

    We construct subjects whose ground_truth is overwritten with junk;
    the cohort builder should still produce identical output as long as
    payload bytes are unchanged.
    """
    ds, _ = build_dataset(regime_id="default")
    cell = ds.cells[0]
    cohort_a = _build_cohort(cell.subjects, [])
    # Swap ground_truth values in-place
    poisoned = []
    for s in cell.subjects:
        s_copy = s.model_copy(deep=True)
        s_copy.ground_truth = 999.999
        poisoned.append(s_copy)
    cohort_b = _build_cohort(poisoned, [])
    assert cohort_a["noise_floor_mean_abs"] == cohort_b["noise_floor_mean_abs"]
    assert cohort_a["candidate_freqs"] == cohort_b["candidate_freqs"]
    assert cohort_a["sample_rate"] == cohort_b["sample_rate"]
    assert cohort_a["n_samples"] == cohort_b["n_samples"]
