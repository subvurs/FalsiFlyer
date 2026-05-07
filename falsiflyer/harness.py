"""Decision-rule-bound benchmark harness.

The Harness is the runtime side of the pre-registration: it loads a
frozen Dataset, runs the registered Estimator callables on every cell
and subject, aggregates median rel-err per cell per estimator, then
applies the dataset's pre-registered ``DecisionRule`` verbatim.

The decision-rule logic is intentionally a direct port of
``benchmark_q_kernel_tdm_7k.py:evaluate_decision_rule``. Both #7j
(threshold 0.30, 3 baselines) and #7k (threshold 0.15, 4 baselines)
collapse to the same code path; the only knobs are the dataset's own
serialized rule.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from falsiflyer.errors import DecisionRuleViolation
from falsiflyer.prereg import DecisionRule
from falsiflyer.types import (
    Cell,
    CellScore,
    Dataset,
    EstimatorScore,
    Subject,
    Verdict,
)


# ---------------------------------------------------------------------------
# Estimator protocol
# ---------------------------------------------------------------------------


@dataclass
class Estimator:
    """One named estimator the harness will run.

    Attributes
    ----------
    name:
        Name used by the decision rule (e.g. ``"raw_Q"``, ``"MAP_Proportional"``).
    fn:
        Callable that returns a scalar point estimate. Two calling conventions
        are supported:

        * **per-subject mode** (``cell_aware=False``, default): the harness
          calls ``fn(subject, cell, cohort)`` for every subject. ``cohort``
          is a dict that the harness lazily fills with cohort-derived priors
          via ``cohort_builders``.
        * **cell-aware batch mode** (``cell_aware=True``): the harness calls
          ``fn(cell)`` once and expects ``Dict[subject_id -> float]`` back.
          Use this when the estimator wants full control of the cell loop.
    cell_aware:
        See ``fn``.
    """

    name: str
    fn: Callable
    cell_aware: bool = False


CohortBuilder = Callable[[List[Subject], List[Dict[str, Any]]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# HarnessResult
# ---------------------------------------------------------------------------


class HarnessResult(BaseModel):
    """Combined output of a Harness.run().

    ``cells`` carries every per-cell, per-estimator score; ``verdict`` is
    the PASS/FAIL artifact derived from the dataset's own decision rule.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_sha256: str
    decision_rule: Dict[str, Any]
    estimator_names: List[str]
    cells: List[CellScore]
    verdict: Verdict
    wall_s: float
    json_path: Optional[str] = None
    md_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness:
    """Decision-rule-bound benchmark runner.

    Parameters
    ----------
    estimators:
        Mapping ``name -> Estimator``. Names MUST match the dataset's
        ``decision_rule.test_estimator`` and ``decision_rule.baselines``.
    cohort_builders:
        Optional mapping ``key -> callable(subjects, per_subject_state) -> any``.
        Built once per cell, BEFORE per-subject estimators run; results are
        passed into the estimator function as the ``cohort`` argument.
        Used to compute cohort priors (e.g. ``mu_log_pop``, ``tau2_log``,
        ``sigma_prop``) without leaking ground truth.
    """

    def __init__(
        self,
        estimators: Dict[str, Estimator],
        cohort_builders: Optional[Dict[str, CohortBuilder]] = None,
    ) -> None:
        self.estimators = estimators
        self.cohort_builders: Dict[str, CohortBuilder] = dict(cohort_builders or {})

    # --- runtime ----------------------------------------------------------

    def run(
        self,
        dataset: Dataset,
        out_dir: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> HarnessResult:
        rule = DecisionRule.deserialize(dataset.decision_rule)
        self._validate_estimators(rule)

        t0 = time.perf_counter()
        cell_scores: List[CellScore] = []
        for cell in dataset.cells:
            cs = self._score_cell(cell, rule, verbose=verbose)
            cell_scores.append(cs)
        wall_s = time.perf_counter() - t0

        verdict = evaluate_decision_rule(cell_scores, rule)

        result = HarnessResult(
            dataset_sha256=dataset.sha256_data_payload,
            decision_rule=rule.serialize(),
            estimator_names=list(self.estimators.keys()),
            cells=cell_scores,
            verdict=verdict,
            wall_s=wall_s,
        )

        if out_dir is not None:
            json_path, md_path = self._write_outputs(result, dataset, Path(out_dir))
            result.json_path = str(json_path)
            result.md_path = str(md_path)

        return result

    # --- per-cell ---------------------------------------------------------

    def _score_cell(
        self,
        cell: Cell,
        rule: DecisionRule,
        verbose: bool = False,
    ) -> CellScore:
        # Pass 1: build per-subject placeholder records (estimators may need them).
        per_subject_state: List[Dict[str, Any]] = []
        for s in cell.subjects:
            per_subject_state.append({
                "subject_id":   s.subject_id,
                "ground_truth": s.ground_truth,
            })

        # Pass 2: build cohort dict from registered builders.
        cohort: Dict[str, Any] = {}
        for key, builder in self.cohort_builders.items():
            cohort[key] = builder(cell.subjects, per_subject_state)

        # Pass 3: run estimators.
        estimates: Dict[str, Dict[str, float]] = {
            name: {} for name in self.estimators
        }
        for name, est in self.estimators.items():
            if est.cell_aware:
                out = est.fn(cell)
                if not isinstance(out, dict):
                    raise TypeError(
                        f"Cell-aware estimator {name!r} must return Dict[subject_id, float]; "
                        f"got {type(out).__name__}"
                    )
                for sid, val in out.items():
                    estimates[name][sid] = float(val) if val is not None else float("nan")
            else:
                for s in cell.subjects:
                    val = est.fn(s, cell, cohort)
                    estimates[name][s.subject_id] = (
                        float(val) if val is not None and math.isfinite(float(val))
                        else float("nan")
                    )

        # Pass 4: aggregate per-cell estimator scores.
        estimator_scores: Dict[str, EstimatorScore] = {}
        per_subject_dump: List[Dict[str, Any]] = []
        for s in cell.subjects:
            row: Dict[str, Any] = {
                "subject_id":   s.subject_id,
                "ground_truth": s.ground_truth,
            }
            for name in self.estimators:
                row[name] = estimates[name].get(s.subject_id, float("nan"))
            per_subject_dump.append(row)

        for name in self.estimators:
            rels: List[float] = []
            finite_vals: List[float] = []
            for s in cell.subjects:
                v = estimates[name].get(s.subject_id, float("nan"))
                truth = s.ground_truth
                if math.isfinite(v) and v > 0:
                    finite_vals.append(v)
                if (
                    truth is not None
                    and truth > 0
                    and math.isfinite(v)
                ):
                    rels.append(abs(v - truth) / truth)

            if rels:
                med = float(np.median(rels))
                mean = float(np.mean(rels))
            else:
                med = mean = float("nan")
            geom = (
                float(np.exp(np.mean(np.log(finite_vals))))
                if finite_vals else float("nan")
            )
            estimator_scores[name] = EstimatorScore(
                geom_mean=geom,
                median_rel_err=med,
                mean_rel_err=mean,
                n_finite=len(finite_vals),
                n_total=len(cell.subjects),
            )
            if verbose:
                print(
                    f"  [{cell.cell_id}] {name:18s} med_rel={med:.3f}  "
                    f"n_finite={len(finite_vals)}/{len(cell.subjects)}"
                )

        return CellScore(
            cell_id=cell.cell_id,
            is_stress_cell=cell.is_stress_cell,
            params=cell.params,
            n_subjects=len(cell.subjects),
            estimators=estimator_scores,
            per_subject=per_subject_dump,
        )

    # --- validation -------------------------------------------------------

    def _validate_estimators(self, rule: DecisionRule) -> None:
        required = {rule.test_estimator, *rule.baselines}
        missing = required - set(self.estimators.keys())
        if missing:
            raise DecisionRuleViolation(
                f"Harness is missing required estimators: {sorted(missing)}; "
                f"registered: {sorted(self.estimators.keys())}"
            )

    # --- output writers ---------------------------------------------------

    def _write_outputs(
        self,
        result: HarnessResult,
        dataset: Dataset,
        out_dir: Path,
    ) -> tuple[Path, Path]:
        from falsiflyer.report import render_markdown_report

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"falsiflyer_run_{ts}.json"
        md_path = out_dir / f"falsiflyer_run_{ts}.md"

        with json_path.open("w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
        with md_path.open("w") as f:
            f.write(render_markdown_report(dataset, result))
        return json_path, md_path


# ---------------------------------------------------------------------------
# evaluate_decision_rule — direct port of #7j/#7k logic
# ---------------------------------------------------------------------------


def evaluate_decision_rule(
    cell_scores: List[CellScore],
    rule: DecisionRule,
) -> Verdict:
    """raw_Q-must-beat-each-baseline accounting; verbatim from #7k."""
    stress = [c for c in cell_scores if c.is_stress_cell]
    n_stress = len(stress)
    threshold_ratio = 1.0 - rule.tightening_threshold
    metric = rule.metric

    n_pass_each = {b: 0 for b in rule.baselines}
    n_pass_all = 0
    per_cell: List[Dict[str, Any]] = []

    for c in stress:
        test_est = c.estimators.get(rule.test_estimator)
        if test_est is None:
            raise DecisionRuleViolation(
                f"Cell {c.cell_id} missing test estimator {rule.test_estimator!r}"
            )
        test_med = getattr(test_est, metric)

        cell_rec: Dict[str, Any] = {
            "cell_id":       c.cell_id,
            "test_estimator_med": test_med,
            "vs_baseline":   {},
            "passes_all":    False,
        }
        all_pass = True
        for b in rule.baselines:
            base_est = c.estimators.get(b)
            if base_est is None:
                raise DecisionRuleViolation(
                    f"Cell {c.cell_id} missing baseline {b!r}"
                )
            base_med = getattr(base_est, metric)
            if (
                not math.isfinite(test_med)
                or not math.isfinite(base_med)
                or base_med <= 0.0
            ):
                ratio = float("nan")
                pass_b = False
            else:
                ratio = test_med / base_med
                pass_b = ratio <= threshold_ratio
            cell_rec["vs_baseline"][b] = {
                "base_med":          base_med,
                "ratio_test_over_base": (
                    float(ratio) if math.isfinite(ratio) else None
                ),
                "tightening_pct": (
                    float(100.0 * (1.0 - ratio)) if math.isfinite(ratio) else None
                ),
                "pass": bool(pass_b),
            }
            if pass_b:
                n_pass_each[b] += 1
            else:
                all_pass = False
        cell_rec["passes_all"] = all_pass
        if all_pass:
            n_pass_all += 1
        per_cell.append(cell_rec)

    required = int(math.ceil(rule.pass_fraction * n_stress))
    verdict_pass = (n_pass_all >= required) and (n_stress > 0)

    return Verdict(
        test_estimator=rule.test_estimator,
        baselines=list(rule.baselines),
        tightening_threshold=rule.tightening_threshold,
        pass_fraction=rule.pass_fraction,
        n_stress_cells=n_stress,
        required_n_passing=required,
        n_pass_all_baselines=n_pass_all,
        n_pass_each_baseline=n_pass_each,
        verdict_pass=verdict_pass,
        per_cell=per_cell,
    )
