"""Worked example: replicate the #7j → #7k diagnostic pivot.

This walkthrough exercises ALL five FalsiFlyer primitives:

  1. ``adapters.q_kernel_tdm.build_dataset`` — hash-commit pre-registration
  2. ``adapters.q_kernel_tdm.build_harness`` — proper-baseline construction
  3. ``Harness.run``                          — decision-rule-bound benchmark
  4. ``run_split_diagnostic``                 — disagreement-on-clipped split
  5. ``AuditLedger`` + ``ByteIdenticalAnchor``— audit-trail bundle

The "raw_Q" estimator here is a STUB (a noisy NCA) so the example does
not depend on the proprietary kernel.  In a real run you would plug in
the actual ``estimate_gamma(method='questimator')`` call.

Run::

    cd FalsiFlyer
    python examples/walkthrough_7j_to_7k.py
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
from adapters.q_kernel_tdm import (
    FLOOR,
    F_0_DEFAULT,
    build_dataset,
    build_harness,
)


# ---------------------------------------------------------------------------
# 1. Stub raw_Q estimator (caller-provided, NOT shipped by the adapter)
# ---------------------------------------------------------------------------


def _stub_raw_q(s, cell, cohort) -> float:
    """A weak Q stand-in: NCA estimate plus tiny bias toward cohort prior.

    The real Questimator would replace this with the bidirectional T=0.857
    smoothing + Baiame boost over a log-gamma grid (see
    nyxnet/questimator.py:estimate_gamma).
    """
    c = cohort["cohort"]
    rec = c["nca_by_id"].get(s.subject_id)
    if rec is None:
        return float("nan")
    k_nca = rec["k_nca"]
    if not math.isfinite(k_nca):
        return float("nan")
    pop = c["pop_ke_lin"]
    if not math.isfinite(pop):
        return float(k_nca)
    return float(0.7 * k_nca + 0.3 * pop)


# ---------------------------------------------------------------------------
# 2. Disagreement split: subjects with any clipped F_hat vs unclipped
# ---------------------------------------------------------------------------


def _is_clipped(per_subject_row) -> bool:
    """The harness fills per_subject rows with the original Subject.payload?

    No — it carries (subject_id, ground_truth, est1, est2, ...). For the
    split predicate to see clipping, we would normally augment the
    payload with a precomputed flag at dataset-build time.  Since the
    walkthrough doesn't do that, we approximate clipping by a high test
    rel-err threshold (fast and good enough for demo purposes).
    """
    truth = per_subject_row.get("ground_truth")
    test = per_subject_row.get("raw_Q")
    if truth is None or test is None or not math.isfinite(test) or truth <= 0:
        return False
    return abs(test - truth) / truth > 0.30


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="falsiflyer_walk_"))
    print(f"[walkthrough] scratch dir: {work}")

    # Build and freeze the #7k dataset.
    frozen_path = work / "frozen_7k.json"
    ds, written = build_dataset(regime_id="7k", out_path=str(frozen_path))
    assert written is not None
    print(
        f"[walkthrough] froze dataset {ds.schema_version!r}: "
        f"{ds.n_cells} cells / {ds.n_stress_cells} stress / "
        f"{ds.n_subjects} subjects"
    )
    print(f"[walkthrough] sha256 = {ds.sha256_data_payload}")

    # Round-trip: re-load from disk to prove hash invariance.
    reloaded = load_frozen_dataset(frozen_path)
    assert reloaded.sha256_data_payload == ds.sha256_data_payload
    print("[walkthrough] reload-and-verify OK")

    # Build a harness with the four classical baselines + stub raw_Q.
    raw_q = Estimator("raw_Q", _stub_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    print(f"[walkthrough] estimators: {sorted(harness.estimators)}")

    # Run the harness — applies the dataset's pre-registered decision rule.
    out_dir = work / "run_outputs"
    result = harness.run(reloaded, out_dir=out_dir, verbose=False)
    v = result.verdict
    print()
    print("=" * 72)
    print(f"VERDICT  : {'PASS' if v.verdict_pass else 'FAIL'}")
    print(f"  test  = {v.test_estimator}")
    print(f"  thr   = {v.tightening_threshold}, pass_frac = {v.pass_fraction:.3f}")
    print(f"  stress cells: {v.n_stress_cells}, "
          f"required passing: {v.required_n_passing}")
    print(f"  raw_Q passes ALL baselines on: {v.n_pass_all_baselines}")
    print(f"  per-baseline pass counts: {v.n_pass_each_baseline}")
    print("=" * 72)
    print(f"[walkthrough] artifacts written to {out_dir}")

    # Disagreement-split diagnostic.
    print()
    print("[walkthrough] running disagreement-split diagnostic …")
    split = DisagreementSplit(
        name="hi_err_vs_lo_err",
        predicate=_is_clipped,
        a_label="hi_err",
        b_label="lo_err",
    )
    estimator_names = list(result.estimator_names)
    split_report = run_split_diagnostic(result.cells, estimator_names, split)
    print(split_report.render_text())

    # Audit-trail bundle.
    # In a real run, the kernel slot's anchor would be the byte hash of
    # nyxnet/questimator.py at the commit under test.  Here we use a
    # synthetic file for demo purposes.
    kernel_path = work / "stub_kernel_v1.py"
    kernel_path.write_text("# stub raw_Q kernel\n")
    slot = KernelSlot(
        name="raw_Q",
        anchor=ByteIdenticalAnchor.of_file(kernel_path, notes="stub"),
        description="walkthrough demo slot",
    )
    record = FalsificationRecord(
        record_id="#walk-7k",
        dataset_sha256=ds.sha256_data_payload,
        decision_rule_hash=hash_decision_rule(reloaded.decision_rule),
        kernel_slot=slot,
        verdict_pass=v.verdict_pass,
        notes=(
            f"walkthrough run with stub raw_Q: "
            f"{v.n_pass_all_baselines}/{v.n_stress_cells} stress cells passing"
        ),
    )
    ledger = AuditLedger()
    ledger.add(record)
    ledger_path = work / "audit_ledger.json"
    ledger.save(ledger_path)
    print()
    print(f"[walkthrough] audit ledger written: {ledger_path}")

    # Slot verify — should succeed; mutate to demonstrate drift detection.
    slot.verify()
    kernel_path.write_text("# mutated\n")
    try:
        slot.verify()
    except Exception as e:
        print(f"[walkthrough] anchor drift caught: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
