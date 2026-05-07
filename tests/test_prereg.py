"""Tests for hash-commit pre-registration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from falsiflyer import (
    Cell,
    DecisionRule,
    DecisionRuleViolation,
    HashMismatchError,
    Subject,
    canonical_hash,
    freeze_dataset,
    load_frozen_dataset,
)


def _toy_cells():
    s1 = Subject(
        subject_id="s0",
        ground_truth=0.10,
        payload={"t": [1.0, 2.0, 3.0], "F_hat": [0.9, 0.7, 0.5], "seed": 7},
    )
    s2 = Subject(
        subject_id="s1",
        ground_truth=0.15,
        payload={"t": [1.0, 2.0, 3.0], "F_hat": [0.85, 0.6, 0.4], "seed": 8},
    )
    c0 = Cell(
        cell_id="k=0.10_n=3",
        params={"k_pop": 0.10, "n": 3},
        subjects=[s1, s2],
        is_stress_cell=True,
    )
    return [c0]


def test_canonical_hash_deterministic():
    cells = _toy_cells()
    h1 = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    h2 = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    assert h1 == h2
    assert len(h1) == 64  # sha-256 hex


def test_canonical_hash_changes_under_payload_mutation():
    cells = _toy_cells()
    h0 = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    cells[0].subjects[0].payload["F_hat"][0] = 0.91  # tiny mutation
    h1 = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    assert h0 != h1


def test_canonical_hash_field_set_matters():
    cells = _toy_cells()
    h_ts = canonical_hash(cells, hash_fields=("t",))
    h_full = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    assert h_ts != h_full


def test_canonical_hash_invariant_under_sub_ulp_noise():
    """SHA must be invariant under platform-libm-level noise.

    Mutations below the 12-decimal-place hash quantization (1e-12 abs)
    should NOT change the digest. This is what makes the hash portable
    across macOS Accelerate and Linux glibc/SVML transcendentals.
    """
    cells = _toy_cells()
    h0 = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))

    # Apply a 1e-15 perturbation — well below the 1e-12 quantization
    # threshold. Hash MUST be invariant.
    cells[0].subjects[0].payload["F_hat"] = [
        v + 1e-15 for v in cells[0].subjects[0].payload["F_hat"]
    ]
    h_noise = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    assert h_noise == h0, "hash must be invariant under sub-ULP libm noise"

    # Apply a 1e-9 perturbation — three orders of magnitude ABOVE the
    # quantization threshold. Hash MUST change.
    cells[0].subjects[0].payload["F_hat"] = [
        v + 1e-9 for v in cells[0].subjects[0].payload["F_hat"]
    ]
    h_real = canonical_hash(cells, hash_fields=("t", "F_hat", "seed"))
    assert h_real != h0, "hash must detect changes above quantization threshold"


def test_freeze_and_load_roundtrip(tmp_path: Path):
    cells = _toy_cells()
    rule = DecisionRule(
        test_estimator="raw_Q",
        baselines=["raw_NCA", "MAP_Bayesian"],
        tightening_threshold=0.15,
        pass_fraction=2.0 / 3.0,
    )
    out_path = tmp_path / "frozen.json"
    ds, written = freeze_dataset(
        schema_version="toy_v1",
        cells=cells,
        decision_rule=rule,
        hash_fields=("t", "F_hat", "seed"),
        out_path=out_path,
    )
    assert written is not None
    assert written.exists()
    assert ds.sha256_data_payload != ""

    loaded = load_frozen_dataset(out_path)
    assert loaded.sha256_data_payload == ds.sha256_data_payload
    assert loaded.schema_version == "toy_v1"
    assert loaded.n_cells == 1
    assert loaded.n_stress_cells == 1
    assert loaded.decision_rule["test_estimator"] == "raw_Q"


def test_load_detects_post_freeze_mutation(tmp_path: Path):
    cells = _toy_cells()
    rule = DecisionRule(
        test_estimator="raw_Q",
        baselines=["raw_NCA"],
        tightening_threshold=0.15,
    )
    out_path = tmp_path / "frozen.json"
    freeze_dataset(
        schema_version="toy_v1",
        cells=cells,
        decision_rule=rule,
        hash_fields=("t", "F_hat", "seed"),
        out_path=out_path,
    )

    # Mutate one F_hat value in the on-disk JSON.
    with out_path.open() as f:
        payload = json.load(f)
    payload["cells"][0]["subjects"][0]["payload"]["F_hat"][0] = 0.99
    with out_path.open("w") as f:
        json.dump(payload, f)

    with pytest.raises(HashMismatchError):
        load_frozen_dataset(out_path)


def test_decision_rule_validation():
    with pytest.raises(DecisionRuleViolation):
        DecisionRule(
            test_estimator="raw_Q",
            baselines=["raw_NCA"],
            tightening_threshold=1.5,  # out of range
        )

    with pytest.raises(DecisionRuleViolation):
        DecisionRule(
            test_estimator="raw_Q",
            baselines=["raw_NCA"],
            pass_fraction=0.0,
        )

    with pytest.raises(DecisionRuleViolation):
        DecisionRule(
            test_estimator="raw_Q",
            baselines=["raw_Q", "raw_NCA"],  # test in baselines
        )

    with pytest.raises(DecisionRuleViolation):
        DecisionRule(
            test_estimator="raw_Q",
            baselines=["raw_NCA"],
            metric="other_metric",
        )


def test_decision_rule_round_trip():
    r = DecisionRule(
        test_estimator="raw_Q",
        baselines=["a", "b"],
        tightening_threshold=0.15,
        pass_fraction=0.5,
        stress_predicate="n in {2,3}",
    )
    payload = r.serialize()
    r2 = DecisionRule.deserialize(payload)
    assert r2 == r
