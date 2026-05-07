"""Worked example: Impax frequency-identification falsification.

End-to-end demo of the Impax adapter exercising all five FalsiFlyer primitives.
Mirrors ``examples/walkthrough_7j_to_7k.py`` but in a non-PK domain (signal
discrimination), to demonstrate that the same package handles cross-domain
binary verdicts.

The "impax_classical" estimator here is a STUB — a continuous-frequency
estimator built on top of the matched-filter peak with quadratic local
refinement. It is NOT the proprietary Impax kernel; it just provides
something stronger than the candidate-restricted baselines so the
walkthrough produces a meaningful PASS verdict. In a real run you would
plug in ``blackbox.impax.impax.scanner.ImpaxFrequencyScanner``.

Run::

    cd FalsiFlyer
    PYTHONPATH=. python examples/walkthrough_impax.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from falsiflyer import (
    AuditLedger,
    ByteIdenticalAnchor,
    DisagreementSplit,
    Estimator,
    FalsificationRecord,
    KernelSlot,
    load_frozen_dataset,
    run_split_diagnostic,
)
from falsiflyer.audit import hash_decision_rule
from adapters.impax_signal_disc import build_dataset, build_harness


# ---------------------------------------------------------------------------
# 1. Stub impax_classical estimator
# ---------------------------------------------------------------------------


def _stub_impax_classical(s, cell, cohort) -> float:
    """Continuous-frequency refinement around the matched-filter peak.

    Standard quadratic interpolation: locate the candidate with max
    sin/cos correlation, then refine via a parabolic fit over a finer
    grid in [f-Δ, f+Δ]. This produces a *continuous* estimate, so it
    can in principle beat the candidate-restricted baselines.
    """
    sig = np.asarray(s.payload["signal"], dtype=float)
    fs = float(s.payload["sample_rate"])
    candidates = np.asarray(s.payload["candidate_freqs"], dtype=float)
    n = len(sig)
    t = np.arange(n, dtype=float) / fs
    two_pi_t = 2.0 * math.pi * t

    # Coarse pass: pick best candidate
    amps = np.empty(len(candidates), dtype=float)
    for i, f in enumerate(candidates):
        c_sum = float(np.dot(sig, np.cos(two_pi_t * f)))
        s_sum = float(np.dot(sig, np.sin(two_pi_t * f)))
        amps[i] = c_sum * c_sum + s_sum * s_sum
    i_max = int(np.argmax(amps))
    f_coarse = float(candidates[i_max])

    # Fine refinement: quadratic search on a small ±10% band
    f_lo = f_coarse * 0.9
    f_hi = f_coarse * 1.1
    fine_grid = np.linspace(f_lo, f_hi, 41)
    fine_amps = np.empty(len(fine_grid), dtype=float)
    for i, f in enumerate(fine_grid):
        c_sum = float(np.dot(sig, np.cos(two_pi_t * f)))
        s_sum = float(np.dot(sig, np.sin(two_pi_t * f)))
        fine_amps[i] = c_sum * c_sum + s_sum * s_sum
    j_max = int(np.argmax(fine_amps))

    # Parabolic interpolation on the fine grid
    if 0 < j_max < len(fine_grid) - 1:
        y0, y1, y2 = fine_amps[j_max - 1], fine_amps[j_max], fine_amps[j_max + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            df = float(fine_grid[1] - fine_grid[0])
            return float(fine_grid[j_max] + delta * df)
    return float(fine_grid[j_max])


# ---------------------------------------------------------------------------
# 2. Disagreement split: hi-noise vs lo-noise stress regimes
# ---------------------------------------------------------------------------


def _is_high_err(per_subject_row) -> bool:
    """Approximate 'difficult subject' by impax_classical rel-err > 30%."""
    truth = per_subject_row.get("ground_truth")
    test = per_subject_row.get("impax_classical")
    if truth is None or test is None or not math.isfinite(test) or truth <= 0:
        return False
    return abs(test - truth) / truth > 0.30


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="falsiflyer_impax_"))
    print(f"[walkthrough] scratch dir: {work}")

    # 1. Pre-register: build and freeze the dataset.
    frozen_path = work / "frozen_impax.json"
    ds, written = build_dataset(regime_id="default", out_path=str(frozen_path))
    assert written is not None
    print(
        f"[walkthrough] froze dataset {ds.schema_version!r}: "
        f"{ds.n_cells} cells / {ds.n_stress_cells} stress / "
        f"{ds.n_subjects} subjects"
    )
    print(f"[walkthrough] sha256 = {ds.sha256_data_payload}")

    # Round-trip the JSON to verify hash invariance.
    reloaded = load_frozen_dataset(frozen_path)
    assert reloaded.sha256_data_payload == ds.sha256_data_payload
    print("[walkthrough] reload-and-verify OK")

    # 2. Build harness with classical baselines + stub impax_classical.
    impax_est = Estimator("impax_classical", _stub_impax_classical)
    harness = build_harness(impax_classical=impax_est)
    print(f"[walkthrough] estimators: {sorted(harness.estimators)}")

    # 3. Run the decision-rule-bound benchmark.
    out_dir = work / "run_outputs"
    result = harness.run(reloaded, out_dir=out_dir, verbose=False)
    v = result.verdict
    print()
    print("=" * 72)
    print(f"VERDICT  : {'PASS' if v.verdict_pass else 'FAIL'}")
    print(f"  test  = {v.test_estimator}")
    print(f"  thr   = {v.tightening_threshold}, pass_frac = {v.pass_fraction:.3f}")
    print(
        f"  stress cells: {v.n_stress_cells}, "
        f"required passing: {v.required_n_passing}"
    )
    print(f"  impax_classical passes ALL baselines on: {v.n_pass_all_baselines}")
    print(f"  per-baseline pass counts: {v.n_pass_each_baseline}")
    print("=" * 72)
    print(f"[walkthrough] artifacts written to {out_dir}")

    # 4. Disagreement-split diagnostic on stress cells.
    print()
    print("[walkthrough] running disagreement-split diagnostic …")
    split = DisagreementSplit(
        name="hi_err_vs_lo_err",
        predicate=_is_high_err,
        a_label="hi_err",
        b_label="lo_err",
    )
    estimator_names = list(result.estimator_names)
    split_report = run_split_diagnostic(result.cells, estimator_names, split)
    print(split_report.render_text())

    # 5. Audit-trail bundle.
    kernel_path = work / "stub_impax_kernel.py"
    kernel_path.write_text("# stub impax_classical kernel\n")
    slot = KernelSlot(
        name="impax_classical",
        anchor=ByteIdenticalAnchor.of_file(kernel_path, notes="stub"),
        description="walkthrough demo slot",
    )
    record = FalsificationRecord(
        record_id="#walk-impax-default",
        dataset_sha256=ds.sha256_data_payload,
        decision_rule_hash=hash_decision_rule(reloaded.decision_rule),
        kernel_slot=slot,
        verdict_pass=v.verdict_pass,
        notes=(
            f"walkthrough impax run: "
            f"{v.n_pass_all_baselines}/{v.n_stress_cells} stress cells passing"
        ),
    )
    ledger = AuditLedger()
    ledger.add(record)
    ledger_path = work / "audit_ledger.json"
    ledger.save(ledger_path)
    print()
    print(f"[walkthrough] audit ledger written: {ledger_path}")

    # Slot verify — should succeed; mutate the file to demonstrate drift.
    slot.verify()
    kernel_path.write_text("# mutated\n")
    try:
        slot.verify()
    except Exception as e:
        print(f"[walkthrough] anchor drift caught: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
