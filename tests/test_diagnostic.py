"""Tests for the disagreement-split diagnostic."""

from __future__ import annotations

import math

from falsiflyer import DisagreementSplit, run_split_diagnostic
from falsiflyer.types import CellScore, EstimatorScore


def _mk_cellscore_with_split(cell_id, *, n_a=4, n_b=4):
    """Make a synthetic CellScore where the A side is uniformly tighter for 'test'.

    Each per_subject row carries:
        - subject_id
        - ground_truth = 1.0
        - is_a (used by predicate)
        - test, base — point estimates
    """
    rows = []
    for i in range(n_a):
        rows.append({
            "subject_id":   f"{cell_id}_a{i}",
            "ground_truth": 1.0,
            "is_a":         True,
            "test":         1.05,   # 5% rel err
            "base":         1.20,   # 20% rel err
        })
    for i in range(n_b):
        rows.append({
            "subject_id":   f"{cell_id}_b{i}",
            "ground_truth": 1.0,
            "is_a":         False,
            "test":         1.30,   # 30% rel err on B side
            "base":         1.20,   # B baseline holds
        })

    return CellScore(
        cell_id=cell_id,
        is_stress_cell=True,
        params={},
        n_subjects=len(rows),
        estimators={
            "test": EstimatorScore(
                geom_mean=float("nan"),
                median_rel_err=0.0,
                mean_rel_err=0.0,
                n_finite=len(rows),
                n_total=len(rows),
            ),
            "base": EstimatorScore(
                geom_mean=float("nan"),
                median_rel_err=0.0,
                mean_rel_err=0.0,
                n_finite=len(rows),
                n_total=len(rows),
            ),
        },
        per_subject=rows,
    )


def test_disagreement_split_isolates_subpopulation_effect():
    cells = [_mk_cellscore_with_split("c0"), _mk_cellscore_with_split("c1")]
    split = DisagreementSplit(
        name="A_vs_B",
        predicate=lambda r: bool(r.get("is_a")),
        a_label="A",
        b_label="B",
    )
    rep = run_split_diagnostic(cells, ["test", "base"], split)
    # Per-cell: test/A should be ~0.05; test/B ~0.30. base/A ~base/B ~0.20.
    for c in rep.cells:
        assert math.isclose(c.a["test"].median_rel_err, 0.05, rel_tol=1e-9)
        assert math.isclose(c.b["test"].median_rel_err, 0.30, rel_tol=1e-9)
        assert math.isclose(c.a["base"].median_rel_err, 0.20, rel_tol=1e-9)
        assert math.isclose(c.b["base"].median_rel_err, 0.20, rel_tol=1e-9)
        assert c.n_a == 4
        assert c.n_b == 4

    # Aggregate: test gets 25-percentage-point spread; base is flat.
    assert rep.aggregate["test"]["a_med"] < rep.aggregate["test"]["b_med"]
    assert math.isclose(
        rep.aggregate["base"]["a_med"],
        rep.aggregate["base"]["b_med"],
        rel_tol=1e-9,
    )

    # render_text should produce a non-empty string mentioning A and B labels.
    text = rep.render_text()
    assert "A_vs_B" in text
    assert "A" in text and "B" in text


def test_split_only_stress_filter():
    """Non-stress cells are excluded by default."""
    stress = _mk_cellscore_with_split("stress")
    nonstress = _mk_cellscore_with_split("nonstress")
    nonstress.is_stress_cell = False
    rep = run_split_diagnostic(
        [stress, nonstress],
        ["test", "base"],
        DisagreementSplit(
            name="A_vs_B",
            predicate=lambda r: bool(r.get("is_a")),
        ),
        only_stress=True,
    )
    assert len(rep.cells) == 1
    assert rep.cells[0].cell_id == "stress"


def test_split_includes_nonstress_when_flag_off():
    stress = _mk_cellscore_with_split("stress")
    nonstress = _mk_cellscore_with_split("nonstress")
    nonstress.is_stress_cell = False
    rep = run_split_diagnostic(
        [stress, nonstress],
        ["test", "base"],
        DisagreementSplit(
            name="A_vs_B",
            predicate=lambda r: bool(r.get("is_a")),
        ),
        only_stress=False,
    )
    assert len(rep.cells) == 2
