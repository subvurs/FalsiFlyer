"""Parity-check: real ``ImpaxFrequencyScanner`` through the FalsiFlyer harness.

Plugs ``blackbox.impax.impax.scanner.ImpaxFrequencyScanner`` into the
``impax_classical`` slot of the ``impax_signal_disc`` adapter and runs
the full decision-rule-bound benchmark. Unlike ``parity_check_7k.py``,
there is no pre-existing recorded benchmark for this exact dataset
shape, so the script does NOT assert a specific verdict; it runs the
falsifier and prints whatever the harness produces.

Why no asserted verdict
-----------------------

The Impax scanner is *candidate-restricted*: it returns one of the
``frequency_bins`` it was given, never a continuous frequency. The
three classical baselines in ``impax_signal_disc`` are also
candidate-restricted. With true_freq drawn log-uniformly from the
continuous interval [FREQ_LO, FREQ_HI], the scanner and the baselines
all incur the same irreducible discretization error — so the rel-err
gap between them is determined by detection accuracy on the discrete
grid, not by continuous-vs-discrete framing.

What this proves
----------------

* The FalsiFlyer shell ingests a real, stochastic, externally-defined
  proprietary kernel without modification.
* The ``impax_signal_disc`` adapter passes the kernel exactly the
  inputs it expects (signal, sample_rate, frequency_bins, noise_estimate).
* Per-cell baseline calibration of ``ImpaxSensor`` works inside the
  cohort-builder workflow.

What this does NOT prove
------------------------

* That the Impax kernel beats classical baselines on this metric. That
  is the falsifiable claim under test, and the run reports the verdict
  as-is.
* The 43x classical-vs-quantum sensing claim from
  ``Quasmology/sensing_comparison/``. That uses a different metric
  (anomaly-score discrimination) and a different framing.

Run::

    cd /Users/mvm/Desktop/subvurs
    PYTHONPATH=.:blackbox/impax \\
        python3 examples/parity_check_impax.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

# Make the FalsiFlyer adapter and the real Impax kernel importable.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "falsiflyer"))
sys.path.insert(0, str(ROOT / "blackbox" / "impax"))

from falsiflyer import Estimator, load_frozen_dataset  # noqa: E402
from falsiflyer.types import Cell  # noqa: E402
from adapters.impax_signal_disc import (  # noqa: E402
    NOISE_SIGMA,
    N_SAMPLES,
    SAMPLE_RATE,
    build_dataset,
    build_harness,
)

try:
    from impax.scanner import ImpaxFrequencyScanner  # type: ignore
    from impax.sensor import ImpaxSensor  # type: ignore
except ImportError as e:
    print(f"[parity] SKIP: could not import blackbox.impax — {e}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Cell-aware estimator: one calibrated scanner per cell, then scan each
# subject. Seeds numpy per-subject so the run is deterministic.
# ---------------------------------------------------------------------------


def _make_real_impax_estimator(detection_threshold: float = 3.0) -> Estimator:
    """Build a cell-aware Estimator wrapping the real ImpaxFrequencyScanner.

    Cell-aware mode lets us amortize the per-cell baseline calibration
    (10 noise measurements on N_SAMPLES samples) across all 32 subjects
    in the cell.
    """

    def _scan_cell(cell: Cell) -> Dict[str, float]:
        # 1. Construct + calibrate one scanner per cell.
        np.random.seed(int(cell.params.get("seed_family", 0)) + 991)
        sensor = ImpaxSensor()
        scanner = ImpaxFrequencyScanner(sensor, detection_threshold=detection_threshold)
        scanner.calibrate_baseline(noise_level=NOISE_SIGMA, n_samples=N_SAMPLES)

        # 2. Scan each subject; seed RNG per-subject for determinism.
        out: Dict[str, float] = {}
        for s in cell.subjects:
            np.random.seed(int(s.payload["seed"]) + 2027)
            signal = np.asarray(s.payload["signal"], dtype=float)
            candidates = np.asarray(s.payload["candidate_freqs"], dtype=float)
            noise_est = float(np.mean(np.abs(signal)))
            detected, _bins, _t, _scores = scanner.scan_bandwidth(
                signal_data=signal,
                sample_rate=float(s.payload["sample_rate"]),
                frequency_bins=candidates,
                noise_estimate=noise_est,
            )
            # If scanner failed to detect, return NaN so the harness
            # counts it as a not-finite estimate (n_finite -= 1).
            out[s.subject_id] = (
                float(detected) if detected is not None else float("nan")
            )
        return out

    return Estimator(name="impax_classical", fn=_scan_cell, cell_aware=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="falsiflyer_impax_parity_"))
    print(f"[parity] scratch dir: {work}")

    # 1. Build + freeze the impax_signal_disc default dataset.
    frozen_path = work / "frozen_impax.json"
    ds, _written = build_dataset(regime_id="default", out_path=str(frozen_path))
    print(
        f"[parity] dataset: {ds.n_cells} cells, "
        f"{ds.n_stress_cells} stress, {ds.n_subjects} subjects"
    )
    print(f"[parity] sha256: {ds.sha256_data_payload}")

    reloaded = load_frozen_dataset(frozen_path)
    assert reloaded.sha256_data_payload == ds.sha256_data_payload
    print("[parity] reload-and-verify OK")

    # 2. Build harness with the real ImpaxFrequencyScanner.
    impax_est = _make_real_impax_estimator()
    harness = build_harness(impax_classical=impax_est)
    out_dir = work / "run_outputs"
    print("[parity] running harness with real ImpaxFrequencyScanner …")
    print("[parity] (this calibrates once per cell, then scans 32 subjects/cell)")
    result = harness.run(reloaded, out_dir=out_dir, verbose=False)

    v = result.verdict
    print()
    print("=" * 72)
    print(f"VERDICT          : {'PASS' if v.verdict_pass else 'FAIL'}")
    print(f"  test           = {v.test_estimator}")
    print(f"  threshold      = {v.tightening_threshold}")
    print(f"  pass fraction  = {v.pass_fraction:.6f}")
    print(
        f"  stress cells   = {v.n_stress_cells}, "
        f"required = {v.required_n_passing}"
    )
    print(f"  pass-all count = {v.n_pass_all_baselines}")
    print(f"  per-baseline   = {v.n_pass_each_baseline}")
    print("=" * 72)

    # 3. Per-cell breakdown so the verdict is interpretable.
    print()
    print("[parity] per-stress-cell median rel-err")
    print("-" * 88)
    header = (
        f"{'cell_id':<32}{'impax':>10}{'periodogram':>14}"
        f"{'matched_filter':>17}{'bandpass':>12}"
    )
    print(header)
    print("-" * 88)
    for cs in result.cells:
        if not cs.is_stress_cell:
            continue
        row = f"{cs.cell_id:<32}"
        row += f"{cs.estimators['impax_classical'].median_rel_err:>10.4f}"
        row += f"{cs.estimators['raw_periodogram'].median_rel_err:>14.4f}"
        row += f"{cs.estimators['raw_matched_filter'].median_rel_err:>17.4f}"
        row += f"{cs.estimators['raw_bandpass_energy'].median_rel_err:>12.4f}"
        print(row)
    print("-" * 88)

    print()
    print(f"[parity] artifacts written to {out_dir}")
    print(
        f"[parity] verdict reported: this is the falsifiable result, "
        f"not asserted against any prior recorded benchmark."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
