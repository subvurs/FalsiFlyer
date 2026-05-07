"""Parity-check: real Questimator kernel vs the recorded #7k benchmark.

Runs the full single-rate TDM #7k harness through ``FalsiFlyer`` with the
real ``estimate_gamma(method='questimator')`` plugged into the ``raw_Q``
slot, then asserts that the verdict matches the reference recorded in
``benchmark_results/benchmark_q_kernel_tdm_7k_20260504_043912.json``.

What this proves
----------------

* The FalsiFlyer port of the #7j/#7k pipeline ranks identically to the
  source harness (`benchmark_q_kernel_tdm_7k.py`) when fed byte-identical
  input data and the same proprietary kernel.
* The decision rule, stress-cell selection, and per-baseline pass logic
  carry over without drift.

What this does NOT prove
------------------------

* Byte-identical SHA-256 with the original generator. The original
  generator used its own ``hash_data_payload`` routine; FalsiFlyer's
  ``canonical_hash`` is a re-implementation in the same family but is
  not guaranteed to emit the same bytes. The dataset *content* is the
  same; the hash is reported but not asserted.

Run::

    cd /Users/mvm/Desktop/subvurs
    PYTHONPATH=commercialization/path_c_nyxnet:. \\
        python3 examples/parity_check_7k.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make the adapter and the real questimator importable.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "falsiflyer"))
sys.path.insert(0, str(ROOT / "commercialization" / "path_c_nyxnet"))

from falsiflyer import Estimator, load_frozen_dataset  # noqa: E402
from adapters.q_kernel_tdm import build_dataset, build_harness  # noqa: E402
from nyxnet.questimator import estimate_gamma  # noqa: E402


# ---------------------------------------------------------------------------
# Reference values (recorded benchmark)
# ---------------------------------------------------------------------------

REFERENCE_JSON = (
    ROOT / "benchmark_results" / "benchmark_q_kernel_tdm_7k_20260504_043912.json"
)

EXPECTED_N_CELLS = 27
EXPECTED_N_STRESS = 12
EXPECTED_N_SUBJECTS = 864
EXPECTED_VERDICT_PASS = False
EXPECTED_N_PASS_ALL = 0
EXPECTED_PER_BASELINE = {
    "raw_NCA":           12,
    "stderr_Shrunk_NCA":  9,
    "MAP_Bayesian":       6,
    "MAP_Proportional":   3,
}


# ---------------------------------------------------------------------------
# Real raw_Q: the Questimator kernel
# ---------------------------------------------------------------------------


def _real_raw_q(s, cell, cohort) -> float:
    """Wrap nyxnet.questimator.estimate_gamma as a FalsiFlyer Estimator.

    Pulls (t, F_hat, N_shot, F_0) out of subject.payload and dispatches
    on method='questimator' (bidirectional T=0.857 + Baiame boost over a
    log-gamma grid).
    """
    return float(
        estimate_gamma(
            probe_times=s.payload["t"],
            probe_fidelities=s.payload["F_hat"],
            probe_shots=s.payload["N_shot"],
            F_0=s.payload["F_0"],
            method="questimator",
        )
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="falsiflyer_parity_"))
    print(f"[parity] scratch dir: {work}")

    # 1. Build + freeze the #7k dataset via the FalsiFlyer adapter.
    frozen_path = work / "frozen_7k.json"
    ds, written = build_dataset(regime_id="7k", out_path=str(frozen_path))
    print(f"[parity] dataset: {ds.n_cells} cells, "
          f"{ds.n_stress_cells} stress, {ds.n_subjects} subjects")
    print(f"[parity] sha256 (FalsiFlyer): {ds.sha256_data_payload}")

    # Soft-check: structural counts MUST match the recorded benchmark.
    assert ds.n_cells == EXPECTED_N_CELLS, (
        f"n_cells mismatch: {ds.n_cells} != {EXPECTED_N_CELLS}"
    )
    assert ds.n_stress_cells == EXPECTED_N_STRESS, (
        f"n_stress mismatch: {ds.n_stress_cells} != {EXPECTED_N_STRESS}"
    )
    assert ds.n_subjects == EXPECTED_N_SUBJECTS, (
        f"n_subjects mismatch: {ds.n_subjects} != {EXPECTED_N_SUBJECTS}"
    )

    # Reload-and-verify locks in the canonical hash for the FalsiFlyer port.
    reloaded = load_frozen_dataset(frozen_path)
    assert reloaded.sha256_data_payload == ds.sha256_data_payload
    print("[parity] reload-and-verify OK")

    # Try to read the original recorded SHA for an informational comparison.
    if REFERENCE_JSON.exists():
        ref = json.loads(REFERENCE_JSON.read_text())
        ref_sha = ref["meta"]["dataset_sha256"]
        print(f"[parity] sha256 (recorded): {ref_sha}")
        if ref_sha == ds.sha256_data_payload:
            print("[parity] ✓ SHA matches recorded benchmark byte-for-byte")
        else:
            print("[parity] ⚠ SHA differs (expected: hashing scheme is a "
                  "re-impl). Asserting verdict structure instead.")

    # 2. Build a harness with the real questimator kernel and run.
    raw_q = Estimator("raw_Q", _real_raw_q)
    harness = build_harness(raw_q=raw_q, with_proportional=True)
    out_dir = work / "run_outputs"
    print(f"[parity] running harness with real estimate_gamma "
          f"(method='questimator') …")
    result = harness.run(reloaded, out_dir=out_dir, verbose=False)

    v = result.verdict
    print()
    print("=" * 72)
    print(f"VERDICT          : {'PASS' if v.verdict_pass else 'FAIL'}")
    print(f"  test           = {v.test_estimator}")
    print(f"  threshold      = {v.tightening_threshold}")
    print(f"  pass fraction  = {v.pass_fraction:.6f}")
    print(f"  stress cells   = {v.n_stress_cells}, "
          f"required = {v.required_n_passing}")
    print(f"  pass-all count = {v.n_pass_all_baselines}")
    print(f"  per-baseline   = {v.n_pass_each_baseline}")
    print("=" * 72)

    # 3. Hard parity assertions.
    print()
    print("[parity] checking against recorded reference …")
    failures = []

    if v.verdict_pass != EXPECTED_VERDICT_PASS:
        failures.append(
            f"verdict_pass {v.verdict_pass} != {EXPECTED_VERDICT_PASS}"
        )
    if v.n_pass_all_baselines != EXPECTED_N_PASS_ALL:
        failures.append(
            f"n_pass_all_baselines {v.n_pass_all_baselines} "
            f"!= {EXPECTED_N_PASS_ALL}"
        )
    for k, expected in EXPECTED_PER_BASELINE.items():
        actual = v.n_pass_each_baseline.get(k)
        if actual != expected:
            failures.append(
                f"n_pass_each_baseline[{k!r}] {actual} != {expected}"
            )

    if failures:
        print("[parity] ✗ FAIL")
        for f in failures:
            print(f"   - {f}")
        return 1

    print("[parity] ✓ all parity assertions passed")
    print(f"[parity] artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
