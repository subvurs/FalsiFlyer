"""Markdown report renderer for HarnessResult.

Mirrors ``benchmark_q_kernel_tdm_7k.py:render_markdown`` but is parameter-
ized over the dataset's pre-registered decision rule (test estimator,
baselines, threshold, pass fraction).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

from falsiflyer.types import Dataset

if TYPE_CHECKING:
    from falsiflyer.harness import HarnessResult


def render_markdown_report(dataset: Dataset, result: "HarnessResult") -> str:
    rule = result.decision_rule
    verdict = result.verdict

    lines: List[str] = []
    lines.append(f"# FalsiFlyer run — {dataset.schema_version}")
    lines.append("")
    lines.append(f"- Schema version:  `{dataset.schema_version}`")
    lines.append(f"- Dataset SHA-256: `{result.dataset_sha256}`")
    lines.append(f"- Cells:           {dataset.n_cells} "
                 f"(stress: {dataset.n_stress_cells})")
    lines.append(f"- Subjects:        {dataset.n_subjects}")
    lines.append(f"- Wall:            {result.wall_s:.1f} s")
    lines.append("")

    lines.append("## Decision rule")
    lines.append("")
    test_est = rule["test_estimator"]
    baselines = rule["baselines"]
    thresh_pct = int(round(100 * rule["tightening_threshold"]))
    lines.append(
        f"`{test_est}` must be ≥{thresh_pct}% tighter "
        f"({rule['metric']}) than EACH of "
        + ", ".join(f"`{b}`" for b in baselines)
        + f" on at least {verdict.required_n_passing} of "
        f"{verdict.n_stress_cells} stress cells "
        f"(>= {rule['pass_fraction']:.3f} pass fraction)."
    )
    lines.append("")
    lines.append(
        f"- `n_pass_all_baselines = {verdict.n_pass_all_baselines}`"
    )
    lines.append(
        f"- per-baseline pass counts: `{verdict.n_pass_each_baseline}`"
    )
    flag = "**PASS**" if verdict.verdict_pass else "**FAIL**"
    lines.append(f"- VERDICT: {flag}")
    lines.append("")

    # Header for the per-cell table
    columns = [test_est] + list(baselines)
    lines.append("## Per-cell median rel-err vs ground truth")
    lines.append("")
    header = "| cell_id | stress | " + " | ".join(columns) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(columns)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in result.cells:
        row = f"| {c.cell_id} | {'Y' if c.is_stress_cell else 'n'} "
        for name in columns:
            est = c.estimators.get(name)
            v = est.median_rel_err if est is not None else float("nan")
            row += f"| {v:.3f} "
        row += "|"
        lines.append(row)
    lines.append("")

    # Stress-cell head-to-head
    lines.append("## Stress-cell head-to-head")
    lines.append("")
    lines.append("| cell_id | best estimator | best med_rel | "
                 f"{test_est} med_rel |")
    lines.append("|---|---|---|---|")
    for c in result.cells:
        if not c.is_stress_cell:
            continue
        scored = []
        for name, est in c.estimators.items():
            scored.append((name, est.median_rel_err))
        finite = [(n, v) for (n, v) in scored if math.isfinite(v)]
        if finite:
            best_name, best_med = min(finite, key=lambda p: p[1])
        else:
            best_name, best_med = "n/a", float("nan")
        test_v = c.estimators[test_est].median_rel_err
        lines.append(
            f"| {c.cell_id} | {best_name} | {best_med:.3f} | {test_v:.3f} |"
        )
    lines.append("")

    return "\n".join(lines)
