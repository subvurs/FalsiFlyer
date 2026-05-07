"""Disagreement-split diagnostic — generalization of diagnose_7j_n3_catchup.

When the harness verdict is borderline, surprising, or about to flip the
commercial story, the right move is rarely "tune the threshold". It is
to split the cohort by a binary attribute and see whether the headline
effect is structural (uniform across both halves) or driven by one
sub-population.

The #7j → #7k pivot turned on exactly such a split: ``clipped`` vs
``unclipped`` subjects. On unclipped subjects, raw_Q's edge fell from
+25% at n=2 to +1% at n=3; on clipped subjects MAP_Bayesian rel-err
ballooned 4×. That diagnostic motivated the ``MAP_Proportional`` proper
baseline and the threshold drop from 30% → 15%.

This module ships that pattern as a generic primitive. The caller
provides:

* a ``DisagreementSplit`` declaring how to split (predicate over a
  ``per_subject`` row),
* a list of estimator names,
* a list of cells (or a HarnessResult).

Output is a ``SplitReport`` with side-by-side median rel-err on each
sub-population, plus a generic interpretation guide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from falsiflyer.types import CellScore


SubjectPredicate = Callable[[Dict[str, Any]], bool]


@dataclass
class DisagreementSplit:
    """Declarative split: name + binary predicate on per-subject rows.

    The predicate receives the per-subject dict already populated by the
    Harness (``subject_id``, ``ground_truth``, plus one column per
    registered estimator). It returns True for the "A" side, False for
    the "B" side.
    """

    name: str
    predicate: SubjectPredicate
    a_label: str = "A"
    b_label: str = "B"


class _BucketStats(BaseModel):
    n: int
    median_rel_err: float
    mean_rel_err: float


class _CellSplit(BaseModel):
    cell_id: str
    is_stress_cell: bool
    a: Dict[str, _BucketStats] = {}
    b: Dict[str, _BucketStats] = {}
    n_a: int = 0
    n_b: int = 0


class SplitReport(BaseModel):
    """Per-cell + aggregated summary of a disagreement-split run."""

    split_name: str
    a_label: str
    b_label: str
    estimators: List[str]
    cells: List[_CellSplit]
    aggregate: Dict[str, Any] = {}

    def render_text(self) -> str:
        lines: List[str] = []
        lines.append(f"Disagreement split: {self.split_name}  "
                     f"(A={self.a_label!r} vs B={self.b_label!r})")
        lines.append("=" * 80)
        header = f"{'cell':28s} {'n_A':>5} {'n_B':>5}"
        for est in self.estimators:
            header += f"  {est+'/A':>14} {est+'/B':>14}"
        lines.append(header)
        lines.append("-" * 80)
        for c in self.cells:
            row = f"{c.cell_id:28s} {c.n_a:>5d} {c.n_b:>5d}"
            for est in self.estimators:
                a = c.a.get(est)
                b = c.b.get(est)
                row += (
                    f"  {(a.median_rel_err if a else float('nan')):>14.3f}"
                    f" {(b.median_rel_err if b else float('nan')):>14.3f}"
                )
            lines.append(row)
        lines.append("")
        lines.append("Aggregate (mean over stress cells):")
        for est, vals in self.aggregate.items():
            lines.append(
                f"  {est:18s}  A_med={vals['a_med']:.3f}  "
                f"B_med={vals['b_med']:.3f}  "
                f"delta_pct={vals['delta_pct']:+.1f}"
            )
        lines.append("")
        lines.append(
            "Interpretation: if test_estimator's A→B delta is similar to "
            "baselines', the headline effect is uniform; if it diverges, "
            "the effect is driven by sub-population structure (cf. "
            "#7j 'clipped vs unclipped' pivot)."
        )
        return "\n".join(lines)


def _bucket_rel_errs(
    rows: List[Dict[str, Any]],
    estimator: str,
) -> Dict[str, float]:
    rels: List[float] = []
    for r in rows:
        truth = r.get("ground_truth")
        v = r.get(estimator)
        if (
            truth is not None
            and truth > 0
            and v is not None
            and isinstance(v, (int, float))
            and math.isfinite(float(v))
        ):
            rels.append(abs(float(v) - truth) / truth)
    if not rels:
        return {"n": 0, "median": float("nan"), "mean": float("nan")}
    return {
        "n":      len(rels),
        "median": float(np.median(rels)),
        "mean":   float(np.mean(rels)),
    }


def run_split_diagnostic(
    cells: Sequence[CellScore],
    estimators: Sequence[str],
    split: DisagreementSplit,
    *,
    only_stress: bool = True,
) -> SplitReport:
    """Score every estimator twice (A side vs B side) on each cell."""
    target_cells = [
        c for c in cells if (not only_stress) or c.is_stress_cell
    ]

    per_cell: List[_CellSplit] = []
    aggregate_scratch: Dict[str, Dict[str, List[float]]] = {
        e: {"a": [], "b": []} for e in estimators
    }

    for c in target_cells:
        a_rows = [r for r in c.per_subject if split.predicate(r)]
        b_rows = [r for r in c.per_subject if not split.predicate(r)]
        cs = _CellSplit(
            cell_id=c.cell_id,
            is_stress_cell=c.is_stress_cell,
            n_a=len(a_rows),
            n_b=len(b_rows),
        )
        for est in estimators:
            a_stats = _bucket_rel_errs(a_rows, est)
            b_stats = _bucket_rel_errs(b_rows, est)
            cs.a[est] = _BucketStats(
                n=a_stats["n"],
                median_rel_err=a_stats["median"],
                mean_rel_err=a_stats["mean"],
            )
            cs.b[est] = _BucketStats(
                n=b_stats["n"],
                median_rel_err=b_stats["median"],
                mean_rel_err=b_stats["mean"],
            )
            if math.isfinite(a_stats["median"]):
                aggregate_scratch[est]["a"].append(a_stats["median"])
            if math.isfinite(b_stats["median"]):
                aggregate_scratch[est]["b"].append(b_stats["median"])
        per_cell.append(cs)

    aggregate: Dict[str, Any] = {}
    for est, buckets in aggregate_scratch.items():
        a_med = float(np.mean(buckets["a"])) if buckets["a"] else float("nan")
        b_med = float(np.mean(buckets["b"])) if buckets["b"] else float("nan")
        if math.isfinite(a_med) and math.isfinite(b_med) and b_med > 0:
            delta_pct = 100.0 * (a_med - b_med) / b_med
        else:
            delta_pct = float("nan")
        aggregate[est] = {
            "a_med":     a_med,
            "b_med":     b_med,
            "delta_pct": delta_pct,
        }

    return SplitReport(
        split_name=split.name,
        a_label=split.a_label,
        b_label=split.b_label,
        estimators=list(estimators),
        cells=per_cell,
        aggregate=aggregate,
    )
