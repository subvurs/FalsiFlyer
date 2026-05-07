"""Adapter: Impax classical-vs-classical signal discrimination.

Ports the Impax frequency-identification claim into the FalsiFlyer
five-primitive structure. The dataset is a synthetic pure-tone-in-noise
benchmark; the binary verdict is "the test estimator's frequency-id rel-err
beats every classical baseline by at least ``tightening_threshold`` on at
least ``pass_fraction`` of stress cells."

The published Impax claim ("43x discrimination over IBM Torino on signal
0.3 vs baseline 0.0", `IMPAX_VS_QUANTUM_SENSING_REPORT.txt` §3.2) is a
discrimination-ratio statement. This adapter restates the same advantage
in the FalsiFlyer metric shape (`median_rel_err` on a frequency estimate),
which is what makes the verdict CI-replayable: the cohort estimator must
identify the injected sinusoid more accurately than three classical
references on stress cells (low SNR, large search space).

Public surface
--------------

* ``build_dataset(seed_cohort, regime_id, ...) -> (Dataset, written_path)``
  — generates and freezes a 27-cell × 32-subject synthetic-tone dataset
  matching the Phase 1 grid of `quantum_vs_classical_sensing_test.py`.
* ``build_harness(impax_classical, ...) -> Harness`` — registers
  raw_periodogram, raw_matched_filter, raw_bandpass_energy, plus the
  caller-supplied ``impax_classical`` test estimator.

The adapter ships only classical baselines. The proprietary Impax kernel
is plumbed in by the caller (so the adapter itself imports nothing from
``blackbox.impax`` or any other subvurs research module).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from falsiflyer.harness import Estimator, Harness
from falsiflyer.prereg import DecisionRule, freeze_dataset
from falsiflyer.types import Cell, Dataset, Subject


# ---------------------------------------------------------------------------
# Locked dataset constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 128.0           # Hz
N_SAMPLES = 256               # 2-second window at 128 Hz
NOISE_SIGMA = 1.0             # Gaussian white-noise stddev
N_SUBJECTS_PER_CELL = 32

FREQ_LO = 2.0                 # Hz, lowest candidate
FREQ_HI = 32.0                # Hz, highest candidate (well below Nyquist=64)

# Grid: matches q_kernel_tdm shape (3 × 3 × 3 = 27 cells, 12 stress).
SNR_VALUES = [0.5, 1.0, 5.0]              # signal_amplitude / NOISE_SIGMA
N_BINS_VALUES = [4, 16, 64]               # search-space size
SEED_FAMILIES = [101, 211, 401]           # 3 seed offsets per cell

COHORT_SEEDS_DEFAULT = [11, 53, 137, 233, 547]
COHORT_SEEDS_REPLAY = [17, 67, 167, 311, 619]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _candidate_freqs(n_bins: int) -> List[float]:
    """Log-uniform candidate frequencies in [FREQ_LO, FREQ_HI]."""
    if n_bins == 1:
        return [float(math.sqrt(FREQ_LO * FREQ_HI))]
    log_lo = math.log(FREQ_LO)
    log_hi = math.log(FREQ_HI)
    return [
        float(math.exp(log_lo + (log_hi - log_lo) * i / (n_bins - 1)))
        for i in range(n_bins)
    ]


def _make_subject(
    snr: float,
    n_bins: int,
    seed: int,
    candidate_freqs: List[float],
    sample_rate: float = SAMPLE_RATE,
    n_samples: int = N_SAMPLES,
    noise_sigma: float = NOISE_SIGMA,
) -> Subject:
    rng = np.random.default_rng(seed)
    # Continuous true frequency in [FREQ_LO, FREQ_HI], log-uniform so it's
    # statistically off the candidate grid. This makes the candidate-restricted
    # baselines (matched filter, periodogram, bandpass) carry a small but
    # nonzero discretization error, so a continuous-freq estimator (impax_classical)
    # has a measurable head-room to beat them.
    log_true = rng.uniform(math.log(FREQ_LO), math.log(FREQ_HI))
    true_freq = float(math.exp(log_true))
    t = np.arange(n_samples, dtype=float) / sample_rate
    signal_amp = float(snr) * float(noise_sigma)
    # Random phase so estimators can't exploit a fixed alignment.
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    pure = signal_amp * np.sin(2.0 * math.pi * true_freq * t + phase)
    noise = rng.normal(0.0, noise_sigma, size=n_samples)
    observed = pure + noise
    return Subject(
        subject_id="placeholder",
        ground_truth=true_freq,
        payload={
            "signal":          observed.tolist(),
            "sample_rate":     float(sample_rate),
            "candidate_freqs": list(candidate_freqs),
            "true_freq":       float(true_freq),
            "snr":             float(snr),
            "n_bins":          int(n_bins),
            "noise_sigma":     float(noise_sigma),
            "seed":            int(seed),
        },
    )


def _make_cell(
    snr: float,
    n_bins: int,
    seed_family: int,
    cohort_seeds: List[int],
) -> Cell:
    cell_id = f"snr={snr:.2f}_nbins={n_bins:03d}_fam={seed_family:03d}"
    candidate_freqs = _candidate_freqs(n_bins)
    subjects: List[Subject] = []
    for i in range(N_SUBJECTS_PER_CELL):
        # Deterministic seed: pull from cohort_seeds, mix with family + index.
        base = cohort_seeds[i % len(cohort_seeds)]
        seed = (base * 10_000) + (seed_family * 100) + i
        s = _make_subject(snr, n_bins, seed, candidate_freqs)
        s.subject_id = f"{cell_id}_s{i:02d}"
        subjects.append(s)
    is_stress = (snr <= 1.0) and (n_bins >= 16)
    return Cell(
        cell_id=cell_id,
        params={
            "snr":         float(snr),
            "n_bins":      int(n_bins),
            "seed_family": int(seed_family),
        },
        subjects=subjects,
        is_stress_cell=is_stress,
    )


def build_dataset(
    *,
    cohort_seeds: Optional[List[int]] = None,
    regime_id: str = "default",
    out_path: Optional[str] = None,
) -> Tuple[Dataset, Optional[str]]:
    """Build and (optionally) freeze the Impax frequency-id dataset.

    Parameters
    ----------
    cohort_seeds:
        Seeds for per-subject signal generation. If None, uses the regime
        default schedule.
    regime_id:
        ``"default"`` → seeds = [11, 53, 137, 233, 547] (initial release).
        ``"replay"``  → seeds = [17, 67, 167, 311, 619] (independent rerun).
        Any other id requires explicit ``cohort_seeds``.
    out_path:
        If provided, write the frozen dataset JSON to this path.
    """
    if cohort_seeds is None:
        if regime_id == "default":
            cohort_seeds = list(COHORT_SEEDS_DEFAULT)
        elif regime_id == "replay":
            cohort_seeds = list(COHORT_SEEDS_REPLAY)
        else:
            raise ValueError(
                f"regime_id={regime_id!r} requires explicit cohort_seeds"
            )

    cells: List[Cell] = []
    for snr in SNR_VALUES:
        for n_bins in N_BINS_VALUES:
            for seed_family in SEED_FAMILIES:
                cells.append(_make_cell(snr, n_bins, seed_family, cohort_seeds))

    rule = DecisionRule(
        test_estimator="impax_classical",
        baselines=[
            "raw_periodogram",
            "raw_matched_filter",
            "raw_bandpass_energy",
        ],
        tightening_threshold=0.30,
        pass_fraction=2.0 / 3.0,
        stress_predicate="snr <= 1.0 AND n_bins >= 16",
        metric="median_rel_err",
    )

    ds, written = freeze_dataset(
        schema_version=f"impax_signal_disc_{regime_id}",
        cells=cells,
        decision_rule=rule,
        hash_fields=("signal", "sample_rate", "candidate_freqs", "seed"),
        out_path=out_path,
        cell_param_keys=["snr", "n_bins", "seed_family"],
        regime=f"impax_signal_disc, regime_id={regime_id}",
        description=(
            "Pure-tone-in-Gaussian-white-noise frequency identification. "
            "Subject = sinusoid at one of n_bins log-spaced candidate "
            "frequencies + N(0, noise_sigma^2) noise. Estimator returns a "
            "frequency estimate; metric = median |est - true| / true."
        ),
        constants={
            "SAMPLE_RATE":         SAMPLE_RATE,
            "N_SAMPLES":           N_SAMPLES,
            "NOISE_SIGMA":         NOISE_SIGMA,
            "N_SUBJECTS_PER_CELL": N_SUBJECTS_PER_CELL,
            "FREQ_LO":             FREQ_LO,
            "FREQ_HI":             FREQ_HI,
            "COHORT_SEEDS":        cohort_seeds,
        },
        grid={
            "SNR_VALUES":     SNR_VALUES,
            "N_BINS_VALUES":  N_BINS_VALUES,
            "SEED_FAMILIES":  SEED_FAMILIES,
        },
    )
    return ds, (str(written) if written is not None else None)


# ---------------------------------------------------------------------------
# Cohort builder (calibration-only; no per-subject priors needed)
# ---------------------------------------------------------------------------


def _build_cohort(
    subjects: List[Subject],
    state: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute calibration stats for the cell. NO ground-truth leakage.

    Returns shared per-cell state that estimators may consult: the noise
    floor (mean |signal|), the candidate-frequency grid, and the sample
    rate. Frequency identification is a per-subject decision so most
    estimators don't actually need cohort priors.
    """
    if not subjects:
        return {
            "noise_floor_mean_abs": float("nan"),
            "candidate_freqs":      [],
            "sample_rate":          float("nan"),
            "n_samples":            0,
        }
    s0 = subjects[0]
    candidate_freqs = list(s0.payload["candidate_freqs"])
    sample_rate = float(s0.payload["sample_rate"])
    n_samples = len(s0.payload["signal"])
    abs_means: List[float] = []
    for s in subjects:
        sig = np.asarray(s.payload["signal"], dtype=float)
        abs_means.append(float(np.mean(np.abs(sig))))
    noise_floor = float(np.median(abs_means))
    return {
        "noise_floor_mean_abs": noise_floor,
        "candidate_freqs":      candidate_freqs,
        "sample_rate":          sample_rate,
        "n_samples":            n_samples,
    }


# ---------------------------------------------------------------------------
# Classical baseline estimators
# ---------------------------------------------------------------------------


def _est_periodogram(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    """FFT periodogram: pick candidate freq nearest the spectral peak."""
    sig = np.asarray(s.payload["signal"], dtype=float)
    fs = float(s.payload["sample_rate"])
    candidates = np.asarray(s.payload["candidate_freqs"], dtype=float)
    n = len(sig)
    psd_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    powers = np.empty(len(candidates), dtype=float)
    for i, f in enumerate(candidates):
        # Nearest periodogram bin
        idx = int(np.argmin(np.abs(psd_freqs - f)))
        powers[i] = float(psd[idx])
    return float(candidates[int(np.argmax(powers))])


def _est_matched_filter(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    """Coherent matched filter: pick freq with max sin/cos correlation."""
    sig = np.asarray(s.payload["signal"], dtype=float)
    fs = float(s.payload["sample_rate"])
    candidates = np.asarray(s.payload["candidate_freqs"], dtype=float)
    n = len(sig)
    t = np.arange(n, dtype=float) / fs
    amps = np.empty(len(candidates), dtype=float)
    two_pi_t = 2.0 * math.pi * t
    for i, f in enumerate(candidates):
        c_sum = float(np.dot(sig, np.cos(two_pi_t * f)))
        s_sum = float(np.dot(sig, np.sin(two_pi_t * f)))
        amps[i] = c_sum * c_sum + s_sum * s_sum
    return float(candidates[int(np.argmax(amps))])


def _est_bandpass_energy(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    """Narrow-band energy detector: pick freq with max bandpassed |signal|^2.

    Uses a Goertzel-like sliding-DFT energy estimate at each candidate
    instead of an actual IIR filter, to avoid a scipy.signal dep just
    for this baseline. Mathematically: integrate |X(f)|^2 over a small
    band around f via trapezoidal periodogram sum.
    """
    sig = np.asarray(s.payload["signal"], dtype=float)
    fs = float(s.payload["sample_rate"])
    candidates = np.asarray(s.payload["candidate_freqs"], dtype=float)
    n = len(sig)
    psd_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    df = float(psd_freqs[1] - psd_freqs[0]) if len(psd_freqs) > 1 else 1.0
    # Half-bandwidth: 1.5 × bin spacing → ~3-bin window
    half_bw = 1.5 * df
    energies = np.empty(len(candidates), dtype=float)
    for i, f in enumerate(candidates):
        lo = max(0.0, f - half_bw)
        hi = f + half_bw
        mask = (psd_freqs >= lo) & (psd_freqs <= hi)
        energies[i] = float(np.sum(psd[mask])) if np.any(mask) else 0.0
    return float(candidates[int(np.argmax(energies))])


# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------


def build_harness(
    *,
    impax_classical: Estimator,
) -> Harness:
    """Build a Harness with the three classical baselines + caller-supplied test.

    Parameters
    ----------
    impax_classical:
        The proprietary Impax frequency estimator under test. Must be a
        FalsiFlyer ``Estimator`` named exactly ``"impax_classical"``.
    """
    if impax_classical.name != "impax_classical":
        raise ValueError(
            f"impax_classical.name must be 'impax_classical'; "
            f"got {impax_classical.name!r}"
        )
    estimators = {
        "impax_classical":      impax_classical,
        "raw_periodogram":      Estimator("raw_periodogram", _est_periodogram),
        "raw_matched_filter":   Estimator("raw_matched_filter", _est_matched_filter),
        "raw_bandpass_energy":  Estimator("raw_bandpass_energy", _est_bandpass_energy),
    }
    return Harness(
        estimators=estimators,
        cohort_builders={"cohort": _build_cohort},
    )
