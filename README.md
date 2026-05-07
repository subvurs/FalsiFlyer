# FalsiFlyer

`FalsiFlyer` is the domain-agnostic methodology package extracted from the
ApotheQuery / Q-kernel-moat falsification audit trail (`#7e`–`#7k`).

It packages five hardening primitives so a new experiment can ship a
defensible binary verdict in roughly the same shape as the eight Q-kernel
falsifications:

1. **Hash-commit pre-registration** — `falsiflyer.prereg`
   SHA-256 over a canonical (caller-declared) data payload, frozen with
   the schema version + decision rule + grid in a single JSON artifact.
   `load_frozen_dataset()` re-verifies the hash on read; mismatch raises.
2. **Proper-baseline templates** — `falsiflyer.baselines`
   `Gaussian`, `Proportional`, `Poisson`, `LogGaussian`, `Binomial` noise
   declarations with companion MAP/likelihood templates. Lets you
   instantiate a "literal-DGP Bayesian" baseline like `MAP_Proportional`
   without re-deriving it.
3. **Decision-rule-bound harness** — `falsiflyer.harness`
   "Test estimator must beat each baseline by ≥X% on at least Y of N
   stress cells." The rule is serialized into the dataset and checked
   verbatim by the runner. No post-hoc threshold tuning.
4. **Diagnostic-on-disagreement template** — `falsiflyer.diagnostic`
   When a decision is borderline or surprising, split the cohort by a
   binary attribute and re-score. Generalizes the `clipped vs unclipped`
   diagnostic that produced the #7j → #7k pivot.
5. **Audit-trail bundle** — `falsiflyer.audit`
   Interchangeable kernel slot + byte-identical anchor pin (file path,
   line number, SHA-256 of the file) + falsification record ledger.

## Status (2026-05-06)

Initial standup is complete. The package implements all five primitives,
ships **two worked-example adapters** in two distinct domains:

* `adapters/q_kernel_tdm.py` — pharmacokinetic decay-rate falsification
  (#7j/#7k Q kernel-moat port), 27 cells × 32 subjects, FAIL verdict
  parity-checked against the recorded benchmark.
* `adapters/impax_signal_disc.py` — non-PK frequency-identification
  falsification, 27 cells × 32 subjects (signal+Gaussian-noise traces),
  PASS verdict on a continuous-frequency stub estimator vs three
  candidate-restricted classical baselines.

Both adapters use the same five-primitive shell, demonstrating the
domain-portability claim. **47 self-contained tests + 1 nyxnet-gated**
parity test, plus 3 golden-replay fixtures pinning the q_kernel_tdm
verdict and per-cell baseline rel-err.

Parity for q_kernel_tdm: running
`nyxnet.questimator.estimate_gamma(method='questimator')` through the
FalsiFlyer harness reproduces the recorded FAIL verdict and the exact
per-baseline pass counts (`raw_NCA: 12, stderr_Shrunk_NCA: 9,
MAP_Bayesian: 6, MAP_Proportional: 3`). The dataset SHA differs from
the original because the original generator's hash routine and FalsiFlyer's
`canonical_hash` are independent re-implementations of the same scheme;
the data bytes are identical.

## Sibling to `gh_eval`

`FalsiFlyer` and `gh_eval` are independent. `gh_eval` is the Goodhart-Hardened
Evaluator (composable scoring primitives for OpenEvolve programs);
`FalsiFlyer` is the pre-registration and decision-rule shell that wraps an
**outside** scientific claim. They share zero code and zero dependencies.

## Quick start

```python
from falsiflyer import (
    DecisionRule, freeze_dataset, load_frozen_dataset,
    Estimator, Harness,
)

# 1. pre-register
ds, path = freeze_dataset(
    schema_version="my_experiment_v1",
    cells=[...],
    decision_rule=DecisionRule(
        test_estimator="my_test",
        baselines=["baseline_a", "baseline_b"],
        tightening_threshold=0.15,
        pass_fraction=2/3,
        stress_predicate="n in {2,3} AND CV in {0.20,0.30}",
    ),
    hash_fields=("t", "F_hat", "N_shot"),
    cell_param_keys=["k_pop", "n_samples", "cv"],
    out_path="frozen.json",
)

# 2. (separately, AFTER freeze) run benchmark
ds = load_frozen_dataset("frozen.json")  # re-verifies SHA-256
harness = Harness(
    estimators={
        "my_test":   my_estimator,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
    },
)
result = harness.run(ds)
print(result.verdict.verdict_pass, result.verdict.n_pass_each_baseline)
```

## Worked example A: q_kernel_tdm (#7j / #7k)

The shipped adapter `adapters/q_kernel_tdm.py` is a self-contained port
of the single-rate TDM falsification flow that produced the #7j → #7k
pivot. It exposes:

| Function          | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `build_dataset`   | Generate + freeze the 27-cell / 12-stress / 864-subj dataset |
| `build_harness`   | Register the four classical baselines + cohort-prior builder |

The proprietary kernel (`raw_Q`) is plugged in by the caller; the
adapter itself imports nothing from `nyxnet`.

Demo runners:

```bash
# All-five-primitives end-to-end demo with a stub raw_Q
PYTHONPATH=. python3 examples/walkthrough_7j_to_7k.py

# Real-kernel parity check vs the recorded #7k benchmark
PYTHONPATH=FalsiFlyer:commercialization/path_c_nyxnet \
    python3 examples/parity_check_7k.py
```

## Worked example B: impax_signal_disc (signal-discrimination)

The second adapter `adapters/impax_signal_disc.py` ports a non-PK claim
through the same five primitives: identifying the dominant pure-tone
frequency in a noisy time-series. Each subject is a 256-sample signal
at 128 Hz with a continuous true frequency drawn log-uniformly from
[2 Hz, 32 Hz] and Gaussian noise scaled by SNR.

Three classical baselines all operate on a discrete candidate grid (so
they incur an irreducible discretization rel-err that a continuous
estimator can in principle beat):

| Baseline               | Method                                                  |
|------------------------|---------------------------------------------------------|
| `raw_periodogram`      | FFT magnitude peak → nearest candidate                  |
| `raw_matched_filter`   | Coherent sin/cos correlation, argmax over candidates    |
| `raw_bandpass_energy`  | Per-candidate Goertzel-style narrowband energy          |

Cell grid: SNR ∈ {0.5, 1.0, 5.0} × n_bins ∈ {4, 16, 64} × seed-fam ∈
{101, 211, 401} = 27 cells × 32 subjects. Stress predicate:
`snr ≤ 1.0 AND n_bins ≥ 16` (12 stress cells).

Decision rule: `tightening_threshold=0.30`, `pass_fraction=2/3`,
`metric=median_rel_err`. Test estimator name: `impax_classical`.

Demo runners:

```bash
# Stub-kernel walkthrough (continuous-frequency parabolic refinement)
PYTHONPATH=. python3 examples/walkthrough_impax.py
# → VERDICT: PASS, 12/12 stress cells, all three baselines beaten

# Real-kernel parity check (blackbox.impax.impax.scanner.ImpaxFrequencyScanner)
PYTHONPATH=FalsiFlyer:blackbox/impax \
    python3 examples/parity_check_impax.py
# → VERDICT: FAIL, 0/12 stress cells passing
```

**Real-kernel result (informational)**: When the actual proprietary
`ImpaxFrequencyScanner` (broadband-detection + binary frequency search,
backed by `ImpaxSensor`'s Nyx-dynamics anomaly score) is plugged in, it
**fails the decision rule on this metric**. Per-cell median rel-err
ranges 0.10–0.64 for impax vs 0.01–0.09 for matched-filter on stress
cells. This is a clean, reproducible falsification of a narrowly scoped
claim ("Impax beats classical on continuous-frequency rel-err with a
discrete candidate grid"). It does **not** contradict the existing 43x
classical-vs-quantum claim in `Quasmology/sensing_comparison/`, which
uses a different metric (anomaly-score discrimination on binary
signal-vs-noise). The two results probe different falsifiable claims.

## Tests

```bash
cd FalsiFlyer
pytest                                                 # 47 pass + 1 skip
PYTHONPATH=.:../commercialization/path_c_nyxnet pytest # 48 pass
```

Test layout:

```
tests/
├── test_prereg.py                  hash determinism, mutation detection, freeze/load round-trip
├── test_baselines.py               5 noise models: log-likelihood + simulate
├── test_harness.py                 PASS/FAIL verdict, validation, cohort builders, output writes
├── test_diagnostic.py              disagreement-split correctness, only_stress filter
├── test_audit.py                   anchor drift, kernel slot, ledger, decision-rule hash
├── test_impax_signal_disc.py       impax adapter: shape, hash, oracle PASS, no-leakage
├── test_golden_q_kernel_tdm.py     golden-replay (7j baselines, 7k baselines, 7k questimator)
└── golden/
    ├── q_kernel_tdm_7j_baselines.json    — pins #7j SHA + per-cell baseline median rel-err
    ├── q_kernel_tdm_7k_baselines.json    — pins #7k SHA + per-cell baseline median rel-err
    ├── q_kernel_tdm_7k_questimator.json  — pins FAIL verdict + per-baseline pass counts
    └── _generate_fixtures.py             — manual regenerator (audit-trail step)
```

The questimator-replay test is `skipif` when `nyxnet` is not on the
path, so the package is fully self-testable in a clean environment.

## File layout

```
FalsiFlyer/
├── pyproject.toml                  setuptools, deps numpy + pydantic
├── README.md                       this file
├── falsiflyer/                     core library (zero subvurs imports)
│   ├── __init__.py                 public re-exports
│   ├── errors.py                   FalsiFlyerError + 3 leaves
│   ├── types.py                    Subject, Cell, Dataset, *Score, Verdict
│   ├── prereg.py                   canonical_hash, freeze, load, DecisionRule
│   ├── baselines.py                5 NoiseModels + BaselineLibrary
│   ├── harness.py                  Estimator, Harness, evaluate_decision_rule
│   ├── diagnostic.py               DisagreementSplit, run_split_diagnostic
│   ├── audit.py                    Anchor, KernelSlot, Record, Ledger
│   └── report.py                   render_markdown_report
├── adapters/
│   ├── q_kernel_tdm.py             #7j/#7k worked example (PK domain)
│   └── impax_signal_disc.py        signal-discrimination worked example (non-PK)
├── examples/
│   ├── walkthrough_7j_to_7k.py     q_kernel_tdm five-primitives demo (stub raw_Q)
│   ├── parity_check_7k.py          real questimator vs recorded benchmark
│   ├── walkthrough_impax.py        impax_signal_disc five-primitives demo (stub kernel)
│   └── parity_check_impax.py       real ImpaxFrequencyScanner through harness
├── tests/                          pytest, 48 pass (47 + 1 gated)
└── docs/
    ├── ARCHITECTURE.md             5-primitive data flow + design invariants
    └── ADAPTER_GUIDE.md            10-step guide to porting a new experiment
```

## Provenance

Extracted from:

- `commercialization/path_c_nyxnet/tools/q_kernel_tdm_dataset_gen.py`
- `commercialization/path_c_nyxnet/tools/q_kernel_tdm_dataset_gen_7k.py`
- `commercialization/path_c_nyxnet/tools/benchmark_q_kernel_tdm.py`
- `commercialization/path_c_nyxnet/tools/benchmark_q_kernel_tdm_7k.py`
- `commercialization/path_c_nyxnet/tools/diagnose_7j_n3_catchup.py`

Methodology source: `ApotheQuery_BRIEF.txt` (#7e–#7k), product spec:
`POST_KERNEL_MOAT_PRODUCT_DIRECTIONS.txt` Section 6.2 (Direction 1,
Falsification-as-a-Service).

## Next steps (post-standup)

1. **Pharma-analytics buyer pilot** (POST_KERNEL_MOAT §6.3) — package
   `FalsiFlyer` + `q_kernel_tdm` adapter as the live demo for the first
   12-week paid pilot.
2. ~~**Second worked-example adapter**~~ — **DONE** (2026-05-06):
   `adapters/impax_signal_disc.py` ports a non-PK signal-discrimination
   claim through the same five primitives. Domain portability validated.
3. ~~**Open-source release**~~ — **DONE** (2026-05-06): Apache-2.0,
   `LICENSE`/`NOTICE`/`CONTRIBUTING.md`/`CHANGELOG.md` in place, GitHub
   Actions CI matrix on Python 3.9–3.12 × {core, audit, scipy} extras.
   Published as v0.2.0.
4. ~~**Audit ledger durability spec**~~ — **DONE** (2026-05-06): hash
   chain + Ed25519 sign/verify in `falsiflyer.audit`, JSONL append-only
   format with cross-process chain check, full threat model in
   `docs/AUDIT_LEDGER_SPEC.md`. 25 audit tests, 65 total green.
5. **`hash_data_payload` byte parity (optional)** — if the open-source
   release wants byte-identical SHA replay against the original
   `q_kernel_tdm_dataset_gen_7k.py`, port that hash routine into a
   `legacy_hash` adapter mode. Not required for verdict parity.
6. ~~**Impax parity demo**~~ — **DONE** (2026-05-06):
   `examples/parity_check_impax.py` plugs the real
   `blackbox.impax.impax.scanner.ImpaxFrequencyScanner` into the
   impax_signal_disc harness. The harness produces a FAIL verdict
   (0/12 stress cells), demonstrating the framework cleanly surfaces
   real-kernel falsification on a different domain than the original
   #7e–#7k audit trail.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

> Copyright 2026 Mark Eatherly
>
> Licensed under the Apache License, Version 2.0. You may obtain a
> copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
>
> Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
> OF ANY KIND, either express or implied.

The Apache 2.0 patent grant explicitly covers the hash-chain + Ed25519
audit-ledger construction in `falsiflyer.audit`. Contributors and
downstream users get the same patent license under inbound=outbound.

For threat-model questions about the audit ledger, see
[`docs/AUDIT_LEDGER_SPEC.md`](docs/AUDIT_LEDGER_SPEC.md).
For contribution scope, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
For change history, see [`CHANGELOG.md`](CHANGELOG.md).
