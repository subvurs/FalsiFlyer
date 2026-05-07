# FalsiFlyer Architecture

`FalsiFlyer` is a domain-agnostic Python package that lifts the methodology
from the eight-falsification audit trail (#7e through #7k) of the Q
kernel-moat hypothesis into reusable primitives. The package's design
goal is one sentence:

> Make the steps that produced a defensible PASS/FAIL verdict on a
> proprietary estimator reproducible on any other proprietary
> estimator, without re-deriving the audit machinery from scratch.

The audit trail is the product. The estimator under test is the input.

## Five primitives

| Primitive                     | Module                     | Source-tool ancestor                        |
|-------------------------------|----------------------------|---------------------------------------------|
| Hash-commit pre-registration  | `falsiflyer.prereg`           | `q_kernel_tdm_dataset_gen[_7k].py`          |
| Proper-baseline templates     | `falsiflyer.baselines`        | MAP_Bayesian / MAP_Proportional in `_7k`    |
| Decision-rule-bound harness   | `falsiflyer.harness`          | `benchmark_q_kernel_tdm[_7k].py`            |
| Disagreement-split diagnostic | `falsiflyer.diagnostic`       | `diagnose_7j_n3_catchup.py`                 |
| Audit-trail bundle            | `falsiflyer.audit`            | `ApotheQuery_BRIEF.txt` §1.1, §3, §4        |

## Data flow

```
                    ┌────────────────────────────────────┐
                    │  caller (experiment / commercial   │
                    │  team) supplies:                   │
                    │   • subjects + cell grid           │
                    │   • DecisionRule (test, baselines, │
                    │     threshold, pass_fraction)      │
                    │   • hash_fields                    │
                    │   • test estimator (raw_Q kernel)  │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │  freeze_dataset(...) → Dataset + sha256_data_payload│
        │  (writes JSON; ANY post-freeze tweak fails reload)  │
        └────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │  Harness(estimators=…, cohort_builders=…)          │
        │    .run(dataset)                                   │
        │      ├─ DecisionRule.deserialize from dataset      │
        │      ├─ validate registered estimators match rule  │
        │      ├─ for each cell:                             │
        │      │    pass-1 build per-subject state           │
        │      │    pass-2 run cohort_builders → cohort dict │
        │      │    pass-3 run estimators (per-subject or    │
        │      │            cell-aware batch)                │
        │      │    pass-4 aggregate EstimatorScore          │
        │      └─ evaluate_decision_rule(cell_scores, rule)  │
        │           → Verdict (PASS / FAIL)                  │
        └────────────────┬───────────────────────────────────┘
                                     │
                  borderline?        │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │  run_split_diagnostic(cells, estimators, split)    │
        │  → SplitReport                                     │
        │  (clipped vs unclipped, n=2 vs n=3, etc.)          │
        └────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │  AuditLedger.add(FalsificationRecord(              │
        │      record_id, dataset_sha256,                    │
        │      decision_rule_hash, kernel_slot, verdict))    │
        │  KernelSlot.verify() → AnchorDriftError on drift   │
        └────────────────────────────────────────────────────┘
```

## Key design invariants

### Caller-declared hash fields (no hidden coupling)

`canonical_hash` does NOT hash the JSON serialization of the Dataset.
Instead it walks named fields out of `Subject.payload` (and named keys
out of `Cell.params`), encoding floats and float-arrays via
`np.tobytes()` and other types via canonical JSON. This matches the
encoding used by the source generators (`q_kernel_tdm_dataset_gen.py`)
and decouples the hash from pydantic version, key-ordering, or float
formatting changes.

The hash field set is part of the audit trail; changing it is a
pre-registration violation, not a refactor.

### DecisionRule lives inside the Dataset

The pre-registered rule (test estimator, baselines, threshold, pass
fraction, metric, stress predicate text) is serialized into
`Dataset.decision_rule`. The Harness re-reads it verbatim. There is no
in-memory route for the caller to supply a different threshold at
runtime than the one committed to disk; the verdict is auditable purely
from the frozen JSON.

### Cohort builders are pre-estimator, post-payload

Estimators that depend on cohort-derived priors (e.g. MAP_Bayesian
needing `mu_log_pop` and `tau2_log`, MAP_Proportional needing
`sigma_prop`) need those priors computed BEFORE they run, but only from
the data that's actually in the Dataset (no ground-truth leakage).

The Harness exposes a `cohort_builders` mapping: each builder is called
once per cell with `(subjects, per_subject_state)` and its return value
is plumbed into estimator invocations as the `cohort` argument
(namespaced by builder key). This is exactly the pattern that the #7k
benchmark uses for `_compute_cohort_priors`.

### Stress cells are part of the dataset, not the harness

`Cell.is_stress_cell` is a frozen attribute of the dataset. The harness
honours it; it does not re-derive stress cells from the params. This
prevents post-hoc stress-cell tuning (a Goodhart attack on
`pass_fraction`).

### Five canonical noise models, no novel statistics

`falsiflyer.baselines` ships Gaussian, Proportional, Poisson, LogGaussian,
and Binomial. Each implements `log_likelihood(y, mu, params)` and
`simulate(mu, params, rng)`. The library is a discoverable catalogue,
not an estimator: adapters pair a noise model with their own MAP / MLE
optimizer (compare `_est_map_bayesian` vs `_est_map_proportional` in
`adapters/q_kernel_tdm.py`).

The #7j → #7k pivot, in this taxonomy, is exactly: replace LogGaussian
with Proportional in the baseline construction.

### Audit ledger is append-only

`AuditLedger.add()` rejects duplicate `record_id`s. The ledger
JSON-round-trips cleanly so the durable artifact across regulatory
cycles is a single file.

`ByteIdenticalAnchor.of_file()` + `verify()` provides the immutability
discipline that made the cumulative #7e–#7k verdict defensible: every
falsification ran against the same `nyxnet/questimator.py` bytes. The
`KernelSlot` wraps an anchor with a stable slot name (`raw_Q`) and a
description; the slot name is what the decision rule references, the
anchor records which implementation was actually executed.

## What's intentionally NOT in FalsiFlyer

* **No estimator implementations.** Adapters supply estimators; the core
  library ships only the noise-model templates (`falsiflyer.baselines`).
  Concrete baselines (e.g. raw_NCA, MAP_Bayesian for q_kernel_tdm; FFT
  periodogram, matched filter, bandpass energy for impax_signal_disc)
  all live in `adapters/`, not `falsiflyer/`.

* **No QIT / Quasmology / Nyx terms.** The package is meant to be
  shippable to a regulator, an auditor, or a competitor's analytics
  team. The Q kernel-moat is a *user* of FalsiFlyer, not part of it.

* **No background workers, no cloud calls, no auto-retry.** The harness
  is a single-process pure function over a frozen dataset. Determinism
  is the priority.

* **No automatic threshold tuning.** Tuning the decision rule between
  freeze and run is a falsification offence; the package does not offer
  helpers that would make that easy.

## Sibling relationship to gh_eval

`gh_eval` (Goodhart-Hardened Evaluator) and `FalsiFlyer` are siblings:

* **gh_eval** hardens *one* OpenEvolve evaluator against Goodhart-style
  exploits. Inputs: a single program. Output: a single combined score.
  Domain knowledge lives in adapter configs.

* **FalsiFlyer** structures *one* binary falsification of a fixed kernel
  against a frozen dataset. Inputs: a Dataset + a kernel + a decision
  rule. Output: PASS/FAIL plus an audit ledger entry.

They share style (pydantic v2 models, fail-closed defaults, adapter
sub-package, golden-replay tests, MD docs) but not code: gh_eval has no
hash-commit, no stress-cell concept, and no per-cell decision rule;
FalsiFlyer has no rubric weights, no static-point gates, and no noise
floor.

## File layout

```
FalsiFlyer/
├── pyproject.toml                  setuptools, deps numpy + pydantic
├── README.md                       package overview + quick start
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
├── adapters/                       experiment-specific bindings
│   ├── __init__.py
│   ├── q_kernel_tdm.py             #7j/#7k port: PK decay-rate falsification
│   └── impax_signal_disc.py        signal-discrimination port: non-PK frequency-id
├── examples/
│   ├── walkthrough_7j_to_7k.py     q_kernel_tdm five-primitives demo (stub raw_Q)
│   ├── parity_check_7k.py          real-kernel parity vs the recorded #7k benchmark
│   ├── walkthrough_impax.py        impax_signal_disc five-primitives demo (stub kernel)
│   └── parity_check_impax.py       real ImpaxFrequencyScanner through harness
├── tests/                          pytest, 48 tests pass (47 self-contained + 1 nyxnet-gated)
│   ├── test_prereg.py
│   ├── test_baselines.py
│   ├── test_harness.py
│   ├── test_diagnostic.py
│   ├── test_audit.py
│   ├── test_impax_signal_disc.py
│   ├── test_golden_q_kernel_tdm.py
│   └── golden/
│       ├── q_kernel_tdm_7j_baselines.json    pins #7j SHA + per-cell baseline median rel-err
│       ├── q_kernel_tdm_7k_baselines.json    pins #7k SHA + per-cell baseline median rel-err
│       ├── q_kernel_tdm_7k_questimator.json  pins FAIL verdict + per-baseline pass counts
│       └── _generate_fixtures.py             manual regenerator (audit-trail step)
└── docs/
    ├── ARCHITECTURE.md             this file
    └── ADAPTER_GUIDE.md            how to write a new adapter
```

## Parity verification

The first proof that the FalsiFlyer port preserves the methodology is a
parity check against the recorded #7k benchmark
(`benchmark_results/benchmark_q_kernel_tdm_7k_20260504_043912.json`).

`examples/parity_check_7k.py` plugs the real
`nyxnet.questimator.estimate_gamma(method='questimator')` into the
`raw_Q` slot and runs the full harness on the FalsiFlyer-built dataset.
Expected (and observed) parity:

| Field                  | Recorded | FalsiFlyer port |
|------------------------|----------|--------------|
| n_cells / stress / subjects | 27 / 12 / 864 | 27 / 12 / 864 |
| verdict_pass           | FAIL     | FAIL         |
| n_pass_all_baselines   | 0        | 0            |
| raw_NCA pass count     | 12       | 12           |
| stderr_Shrunk_NCA      | 9        | 9            |
| MAP_Bayesian           | 6        | 6            |
| MAP_Proportional       | 3        | 3            |

The dataset SHA differs from the recorded one because the original
generator's `hash_data_payload` and FalsiFlyer's `canonical_hash` are
independent re-implementations of the same scheme; the data bytes are
identical (same RNG seeds → same subjects). The pinned SHA inside the
FalsiFlyer golden fixtures is the FalsiFlyer-canonical one, locked across
runs.

The three `tests/golden/` fixtures back this up at the test layer: any
future drift in the adapter's classical baselines, the `canonical_hash`
function, or the Questimator implementation trips a CI failure with a
specific cell + estimator pointing to where the drift occurred.

## Domain portability — the second adapter

`adapters/impax_signal_disc.py` is the second worked example, and is
intentionally in a non-PK domain: pure-tone frequency identification in
Gaussian noise. It uses the same five-primitive shell with no
modification to the core library:

* same `Subject`/`Cell`/`Dataset` containers (the payload dict carries
  raw signal samples instead of concentration-time pairs)
* same `freeze_dataset` / `load_frozen_dataset` round-trip
* same `Harness` and `evaluate_decision_rule` machinery
* same `hash_decision_rule` audit-ledger primitive
* same `DisagreementSplit` / `run_split_diagnostic` template

The adapter ships three classical baselines (`raw_periodogram`,
`raw_matched_filter`, `raw_bandpass_energy`) all operating on a discrete
candidate frequency grid. The test estimator slot (`impax_classical`)
is filled by the caller with any `(subject, cell, cohort) -> float`
callable; the walkthrough demo uses a continuous-frequency stub
(matched-filter peak + ±10% parabolic interpolation) and produces a
PASS verdict on 12/12 stress cells.

This is the FalsiFlyer analogue of "the same regression-tester works on a
different codebase": the methodology is in the package, the science is
in the adapter.

The companion `examples/parity_check_impax.py` plugs the real
`blackbox.impax.impax.scanner.ImpaxFrequencyScanner` into this harness
in cell-aware mode (one calibrated `ImpaxSensor` per cell, then scan
each subject). The result is a FAIL verdict on continuous-frequency
rel-err — a clean, reproducible falsification of a narrowly scoped
claim, surfaced through the same five-primitive shell that produced the
#7e–#7k audit trail. That this run yields FAIL rather than PASS is the
point: the framework reports the verdict that the data and decision
rule produce, regardless of which way it goes.
