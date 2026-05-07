# Writing a FalsiFlyer adapter

An *adapter* binds the FalsiFlyer primitives to a specific experiment.
This guide walks through every step using the shipped
`adapters/q_kernel_tdm.py` (port of the #7j/#7k single-rate TDM
falsification flow) as the worked example. After this you should be
able to stand up a new adapter for any (estimator, dataset, decision
rule) triple.

## When you need an adapter

You need an adapter when you have:

* a candidate estimator under test (e.g. a proprietary kernel `raw_Q`)
* a generative process you can simulate (or a fixed empirical dataset
  + ground truth)
* one or more classical baselines you want raw_Q to beat
* a binary verdict you want to commit to in advance

If any of those are missing, fix that first; FalsiFlyer will not paper
over a missing decision rule.

## Adapter contract

A complete adapter exposes two callables:

| Function           | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `build_dataset()`  | Generate cells, freeze with hash + decision rule          |
| `build_harness()`  | Register estimators (incl. caller-supplied test) + cohort |

It MUST NOT import the proprietary kernel directly. The kernel is
plumbed in at `build_harness` time as a pre-built `Estimator` named
exactly the test estimator name in the decision rule.

## Step 1 — Decide the dataset shape

A `Dataset` is `cells: List[Cell]`. Each `Cell` has:

* `cell_id` — human-readable, e.g. `"k=0.10_n=3_CV=0.20"`
* `params` — dict of structural parameters (these get hashed)
* `subjects: List[Subject]` — the rows
* `is_stress_cell: bool` — pre-registered "this cell counts toward PASS"

A `Subject` has:

* `subject_id` — globally unique
* `payload` — dict of per-subject observation data
* `ground_truth` — Optional[float], scalar truth for rel-err scoring

For the q_kernel_tdm adapter:

```python
def _make_subject(k_pop, n, cv, seed, F_0=0.95):
    ...
    return Subject(
        subject_id="placeholder",      # filled by _make_cell
        ground_truth=k_e,              # truth for rel-err
        payload={
            "t":      t.tolist(),
            "F_hat":  F_obs.tolist(),
            "F_true": F_true.tolist(),
            "N_shot": [N_SHOT] * n,
            "F_0":    F_0,
            "seed":   seed,
        },
    )
```

The `payload` dict is intentionally a free-form dict so multi-modal
experiments (different sensor types, varying observation lengths) can
all use the same `Cell`/`Subject` containers.

## Step 2 — Pick hash fields

`canonical_hash` walks caller-declared fields out of every subject
payload. Pick the minimum field set that defines the data the
estimators will consume. Adding informational fields to the payload
later is fine; they will not be hashed unless they are also added to
`hash_fields`.

For #7k:

```python
hash_fields=("t", "F_hat", "N_shot", "F_0", "seed")
```

Note `F_true` is NOT in `hash_fields` — it's stashed for downstream
analysis but the hash only commits the bytes the estimator actually
sees plus the seed (so Bayesian re-runs that re-simulate get a free
rebound check).

## Step 3 — Define the DecisionRule

```python
rule = DecisionRule(
    test_estimator="raw_Q",
    baselines=["raw_NCA", "stderr_Shrunk_NCA", "MAP_Bayesian", "MAP_Proportional"],
    tightening_threshold=0.15,
    pass_fraction=2.0 / 3.0,
    stress_predicate="n in {2, 3} AND CV in {0.20, 0.30}",
    metric="median_rel_err",
)
```

* `tightening_threshold=0.15` ⇒ raw_Q's median rel-err must be ≤ 85%
  of each baseline's, on each stress cell.
* `pass_fraction=2/3` ⇒ raw_Q must pass ALL baselines simultaneously
  on at least ⌈2/3 · n_stress⌉ stress cells.
* `stress_predicate` is human-readable text only; the harness honours
  `Cell.is_stress_cell`, not this string.

The rule is committed inside the dataset at freeze time.

## Step 4 — Freeze the dataset

```python
ds, written = freeze_dataset(
    schema_version="q_kernel_tdm_7k",
    cells=cells,
    decision_rule=rule,
    hash_fields=hash_fields,
    out_path=out_path,            # optional; writes JSON
    cell_param_keys=["k_pop", "n_samples", "cv"],
    regime="single-rate TDM, regime_id=7k",
    description="F(t) = FLOOR + A*exp(-k_e*t); proportional measurement noise.",
    constants={...},
    grid={...},
)
```

`out_path` is optional. If supplied, freeze_dataset writes the JSON;
otherwise the in-memory Dataset is returned. Loading it back with
`load_frozen_dataset` will recompute the canonical hash and raise
`HashMismatchError` on any post-freeze byte change in a hashed field.

`constants` and `grid` are pure-record fields (the FLOOR, F_0, time
patterns, k_pop schedule) — they help reviewers reproduce the
generation; the harness ignores them.

## Step 5 — Implement the cohort builder (if needed)

If your baselines depend on cohort-derived priors, write a function:

```python
def _build_cohort(subjects: List[Subject],
                  state: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute priors over a cell with no ground-truth leakage."""
    nca_records = [_fit_nca(s.payload["t"], s.payload["F_hat"]) for s in subjects]
    finite = [r["k_nca"] for r in nca_records
              if math.isfinite(r["k_nca"]) and r["k_nca"] > 0]
    mu_log_pop = float(np.mean(np.log(finite))) if len(finite) >= 2 else float("nan")
    ...
    return {
        "mu_log_pop":     mu_log_pop,
        "tau2_log":       tau2_log,
        "sigma_prop":     sigma_prop,
        "F_0":            F_0,
        "nca_by_id":      {s.subject_id: rec for s, rec in zip(subjects, nca_records)},
    }
```

Critical: ONLY consume `subject.payload` and (your own derived state).
Reading `subject.ground_truth` here is data leakage and will silently
make your test estimator look great in benchmarks.

## Step 6 — Implement estimators

Per-subject mode (default):

```python
def _est_raw_nca(s: Subject, cell: Cell, cohort: Dict[str, Any]) -> float:
    return cohort["cohort"]["nca_by_id"][s.subject_id]["k_nca"]
```

The `cohort` argument is a dict keyed by *builder name*. So if the
harness was constructed with
`cohort_builders={"cohort": _build_cohort}` then estimators see
`cohort["cohort"][...]`. (Multiple builders allowed; e.g.
`{"priors": _build_priors, "filters": _build_filters}`.)

Cell-aware batch mode (set `cell_aware=True` on the Estimator):

```python
def _est_batch(cell: Cell) -> Dict[str, float]:
    return {s.subject_id: <my-batch-output> for s in cell.subjects}
```

The harness expects a dict `{subject_id -> float}` back. Use this when
the estimator wants full control of the cell loop (e.g. it caches a
shared per-cell intermediate).

## Step 7 — Build the harness

```python
def build_harness(*, raw_q: Estimator, with_proportional: bool = True) -> Harness:
    if raw_q.name != "raw_Q":
        raise ValueError(f"raw_q.name must be 'raw_Q'; got {raw_q.name!r}")
    estimators = {
        "raw_Q":             raw_q,
        "raw_NCA":           Estimator("raw_NCA", _est_raw_nca),
        "stderr_Shrunk_NCA": Estimator("stderr_Shrunk_NCA", _est_shrunk_nca),
        "MAP_Bayesian":      Estimator("MAP_Bayesian", _est_map_bayesian),
    }
    if with_proportional:
        estimators["MAP_Proportional"] = Estimator(
            "MAP_Proportional", _est_map_proportional,
        )
    return Harness(
        estimators=estimators,
        cohort_builders={"cohort": _build_cohort},
    )
```

The caller passes in the proprietary kernel as `raw_q`. The adapter
ships only the classical baselines.

## Step 8 — Tests

At minimum your adapter should have:

* a "freeze + reload" test (proves the hash is stable)
* a "freeze + mutate-bytes + reload" test (proves drift is caught)
* a per-cell synthetic-truth test that demonstrates the harness gets
  the verdict right when raw_Q matches truth and baselines don't, and
  vice versa

The shipped `tests/` show the patterns; your adapter tests live next
to your adapter, not under `tests/`.

### Golden-replay fixtures (recommended)

Once your adapter is producing the verdict you expect, freeze it. The
q_kernel_tdm worked example uses two layers:

1. **Baseline-only fixture** — runs the harness with a stub raw_Q (NaN)
   and pins per-cell median rel-err for every classical baseline, plus
   the dataset SHA-256 and decision-rule hash. Self-contained: no
   proprietary kernel needed for replay. See
   `tests/golden/q_kernel_tdm_{7j,7k}_baselines.json`.
2. **Kernel-parity fixture** — runs with the real proprietary kernel
   plugged in and pins the FAIL/PASS verdict + per-baseline pass counts.
   The pytest case is `skipif` when the proprietary kernel isn't on
   the path, so the package is fully testable in a clean env. See
   `tests/golden/q_kernel_tdm_7k_questimator.json` and the
   `test_questimator_replay` case.

Manual regenerator: `tests/golden/_generate_fixtures.py`.
Re-run it whenever a deliberate methodology change lands; the diff in
the JSON fixtures IS the audit-trail entry for that change.

The combination — pinned SHA + pinned per-cell baseline scores + pinned
verdict — catches:

* a drift in `canonical_hash` field encoding (SHA changes)
* a drift in any classical baseline implementation (per-cell median changes)
* a regression in the proprietary kernel (per-baseline pass counts shift)

Each of those produces a specific pytest failure that points directly
at the affected cell + estimator.

## Step 9 — Audit-trail bundle

For each falsification run:

```python
slot = KernelSlot(
    name="raw_Q",
    anchor=ByteIdenticalAnchor.of_file(
        "nyxnet/questimator.py", line_number=274,
        notes="estimate_gamma fn body",
    ),
    description="Questimator at commit ${SHA}",
)
slot.verify()  # raise AnchorDriftError on byte change

record = FalsificationRecord(
    record_id="#7l",
    dataset_sha256=ds.sha256_data_payload,
    decision_rule_hash=hash_decision_rule(ds.decision_rule),
    kernel_slot=slot,
    verdict_pass=result.verdict.verdict_pass,
    notes="optional headline summary",
)

ledger = AuditLedger.load("audit_ledger.json") if Path("audit_ledger.json").exists() else AuditLedger()
ledger.add(record)
ledger.save("audit_ledger.json")
```

The ledger is the durable artifact across regulatory cycles. Append
one record per falsification; never amend, never delete.

## Anti-patterns to avoid

* **Pre-registering the threshold AFTER seeing benchmark output.** This
  is the canonical Goodhart attack on a frozen-dataset audit. The
  package makes this hard but not impossible (you can always re-freeze
  the dataset). The remedy is the audit ledger: every falsification
  produces a record with the rule's hash; threshold-tuning shows up as
  a different `decision_rule_hash` on the same `dataset_sha256`.

* **Mutating cell-param fields without re-hashing.** The `cell_param_keys`
  argument to `freeze_dataset` controls which cell-param keys are
  hashed. If you add a new structural parameter to `Cell.params` after
  freezing, the hash drifts on next reload — which is the whole point.

* **Using ground_truth in the cohort builder.** This is data leakage.
  Cohort builders MUST be ground-truth-blind.

* **Skipping the disagreement-split diagnostic.** When verdicts are
  close to the threshold, the right move is *not* "tune the threshold"
  — it's run `run_split_diagnostic` to see whether the headline effect
  is structural or driven by a sub-population. The #7j → #7k pivot
  came from exactly that diagnostic.

* **Making the adapter import the proprietary kernel.** Then the
  adapter is no longer Nyx-free, the package can't be shipped to a
  third-party auditor, and the moat itself becomes part of the
  audit-tooling repo.
