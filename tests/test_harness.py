"""Tests for the decision-rule-bound benchmark Harness."""

from __future__ import annotations

import math

import pytest

from falsiflyer import (
    Cell,
    DecisionRule,
    DecisionRuleViolation,
    Estimator,
    Harness,
    Subject,
    evaluate_decision_rule,
    freeze_dataset,
)
from falsiflyer.types import CellScore, EstimatorScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_dataset(n_cells: int = 4, n_subjects: int = 8):
    cells = []
    for ci in range(n_cells):
        truth = 0.10 * (ci + 1)
        subjects = [
            Subject(
                subject_id=f"c{ci}_s{si}",
                ground_truth=truth,
                payload={"truth_hint": truth, "noise_seed": 1000 * ci + si},
            )
            for si in range(n_subjects)
        ]
        cells.append(Cell(
            cell_id=f"cell_{ci}",
            params={"k_pop": truth},
            subjects=subjects,
            is_stress_cell=(ci % 2 == 0),
        ))
    rule = DecisionRule(
        test_estimator="winning_Q",
        baselines=["weak_baseline"],
        tightening_threshold=0.30,
        pass_fraction=0.5,
    )
    ds, _ = freeze_dataset(
        schema_version="toy_harness",
        cells=cells,
        decision_rule=rule,
        hash_fields=("noise_seed",),
    )
    return ds


# ---------------------------------------------------------------------------
# Harness behavior
# ---------------------------------------------------------------------------


def test_harness_runs_and_pass_when_test_dominates():
    """winning_Q matches truth; weak_baseline is far off → PASS."""
    ds = _toy_dataset()

    def winning_q(s, cell, cohort):
        # tiny constant offset; rel-err ~ 1e-6
        return s.ground_truth * 1.000001

    def weak_baseline(s, cell, cohort):
        # 50% off
        return s.ground_truth * 1.5

    h = Harness(
        estimators={
            "winning_Q":     Estimator("winning_Q", winning_q),
            "weak_baseline": Estimator("weak_baseline", weak_baseline),
        },
    )
    result = h.run(ds)
    assert result.verdict.verdict_pass
    assert result.verdict.n_pass_all_baselines == result.verdict.n_stress_cells
    # walltime is recorded
    assert result.wall_s >= 0.0


def test_harness_fails_when_test_underperforms():
    """weak_test is far off; strong_baseline matches → FAIL."""
    ds = _toy_dataset()

    def weak_test(s, cell, cohort):
        return s.ground_truth * 2.0  # 100% off

    def strong_baseline(s, cell, cohort):
        return s.ground_truth * 1.001

    rule = DecisionRule(
        test_estimator="weak_test",
        baselines=["strong_baseline"],
        tightening_threshold=0.15,
        pass_fraction=0.5,
    )
    # rebuild dataset with this rule
    ds2, _ = freeze_dataset(
        schema_version="toy_harness2",
        cells=ds.cells,
        decision_rule=rule,
        hash_fields=("noise_seed",),
    )

    h = Harness(estimators={
        "weak_test":       Estimator("weak_test", weak_test),
        "strong_baseline": Estimator("strong_baseline", strong_baseline),
    })
    result = h.run(ds2)
    assert not result.verdict.verdict_pass


def test_harness_validates_required_estimators():
    ds = _toy_dataset()
    h = Harness(estimators={
        "winning_Q": Estimator("winning_Q", lambda s, c, co: s.ground_truth),
        # weak_baseline missing
    })
    with pytest.raises(DecisionRuleViolation):
        h.run(ds)


def test_harness_cell_aware_estimator():
    ds = _toy_dataset()

    def cellaware_test(cell):
        # batch: return dict[subject_id -> estimate]
        return {s.subject_id: s.ground_truth * 1.000001 for s in cell.subjects}

    def baseline(s, cell, cohort):
        return s.ground_truth * 1.4

    h = Harness(estimators={
        "winning_Q":     Estimator("winning_Q", cellaware_test, cell_aware=True),
        "weak_baseline": Estimator("weak_baseline", baseline),
    })
    result = h.run(ds)
    assert result.verdict.verdict_pass


def test_harness_cell_aware_must_return_dict():
    ds = _toy_dataset()

    def bad(cell):
        return [1, 2, 3]  # wrong type

    h = Harness(estimators={
        "winning_Q":     Estimator("winning_Q", bad, cell_aware=True),
        "weak_baseline": Estimator(
            "weak_baseline", lambda s, c, co: s.ground_truth * 1.4,
        ),
    })
    with pytest.raises(TypeError):
        h.run(ds)


def test_harness_cohort_builder_runs_once_per_cell():
    """Builder is invoked once per cell, before estimators."""
    calls = {"n": 0}

    def builder(subjects, state):
        calls["n"] += 1
        return {"shared": 42}

    def est_test(s, cell, cohort):
        assert cohort["my"]["shared"] == 42
        return s.ground_truth * 1.000001

    def est_base(s, cell, cohort):
        return s.ground_truth * 1.5

    ds = _toy_dataset(n_cells=3)

    h = Harness(
        estimators={
            "winning_Q":     Estimator("winning_Q", est_test),
            "weak_baseline": Estimator("weak_baseline", est_base),
        },
        cohort_builders={"my": builder},
    )
    h.run(ds)
    assert calls["n"] == 3


def test_harness_writes_outputs(tmp_path):
    ds = _toy_dataset(n_cells=2)

    def good(s, c, co):
        return s.ground_truth * 1.000001

    def bad(s, c, co):
        return s.ground_truth * 1.5

    h = Harness(estimators={
        "winning_Q":     Estimator("winning_Q", good),
        "weak_baseline": Estimator("weak_baseline", bad),
    })
    result = h.run(ds, out_dir=tmp_path)
    assert result.json_path is not None
    assert result.md_path is not None


# ---------------------------------------------------------------------------
# evaluate_decision_rule directly
# ---------------------------------------------------------------------------


def _mk_cell_score(cell_id, *, stress, test_med, baseline_meds):
    estimators = {
        "test": EstimatorScore(
            geom_mean=float("nan"),
            median_rel_err=test_med,
            mean_rel_err=test_med,
            n_finite=1,
            n_total=1,
        ),
    }
    for name, m in baseline_meds.items():
        estimators[name] = EstimatorScore(
            geom_mean=float("nan"),
            median_rel_err=m,
            mean_rel_err=m,
            n_finite=1,
            n_total=1,
        )
    return CellScore(
        cell_id=cell_id,
        is_stress_cell=stress,
        params={},
        n_subjects=1,
        estimators=estimators,
        per_subject=[],
    )


def test_evaluate_decision_rule_pass_fail_boundary():
    rule = DecisionRule(
        test_estimator="test",
        baselines=["b1", "b2"],
        tightening_threshold=0.20,  # 20% better required
        pass_fraction=0.5,
    )
    cells = [
        # passes both: test=0.05 vs b1=0.10 (50% tighter), b2=0.10 (50% tighter)
        _mk_cell_score("c0", stress=True,  test_med=0.05, baseline_meds={"b1": 0.10, "b2": 0.10}),
        # fails b2: test=0.09 vs b2=0.10 (only 10%)
        _mk_cell_score("c1", stress=True,  test_med=0.09, baseline_meds={"b1": 0.20, "b2": 0.10}),
        # non-stress, ignored
        _mk_cell_score("c2", stress=False, test_med=99.0, baseline_meds={"b1": 0.10, "b2": 0.10}),
    ]
    v = evaluate_decision_rule(cells, rule)
    assert v.n_stress_cells == 2
    assert v.n_pass_all_baselines == 1
    assert v.required_n_passing == 1  # ceil(0.5 * 2)
    assert v.verdict_pass is True


def test_evaluate_decision_rule_nonfinite_baseline_blocks_pass():
    rule = DecisionRule(
        test_estimator="test",
        baselines=["b1"],
        tightening_threshold=0.20,
    )
    cells = [
        _mk_cell_score(
            "c0", stress=True, test_med=0.05, baseline_meds={"b1": float("nan")},
        ),
    ]
    v = evaluate_decision_rule(cells, rule)
    assert v.n_pass_all_baselines == 0
    assert v.verdict_pass is False
