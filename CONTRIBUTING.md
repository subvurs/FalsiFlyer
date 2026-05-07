# Contributing to FalsiFlyer

Thanks for considering a contribution. `FalsiFlyer` is research-grade
infrastructure for pre-registered, decision-rule-bound benchmarks; the
contribution scope reflects that.

## Quick start

```bash
git clone <repo>
cd FalsiFlyer
pip install -e ".[test,audit,scipy]"
pytest -q
```

The full test suite should run in under three seconds and finish
**65 passed, 1 skipped** on a clean checkout. The skipped test is a
SciPy-dependent baseline parity check; install `[scipy]` to enable it.

## What we accept

**In scope:**

- Bug fixes against any of the five primitives (`prereg`, `baselines`,
  `harness`, `diagnostic`, `audit`) or any shipped adapter.
- New adapters for new domains, following `docs/ADAPTER_GUIDE.md`.
  An adapter PR should ship: the adapter module, an `examples/walkthrough_*.py`
  driver, and a `tests/test_*.py` module covering at least
  build-dataset / round-trip / build-harness / decision-rule.
- New noise-model templates in `falsiflyer.baselines` if the DGP is
  general (Beta, NegBinomial, Weibull, etc.).
- Documentation clarifications, especially in `docs/AUDIT_LEDGER_SPEC.md`
  if a threat model edge case is missing.
- Performance improvements that preserve byte-identical output.

**Out of scope (please don't):**

- Changing the canonical-bytes encoding in `falsiflyer.audit`. Doing so
  breaks every existing chained ledger. If you have a serious reason
  (e.g. an actual JCS bug on float-heavy payloads), open an issue first.
- Adding "convenience" wrappers that hide the SHA-256 commit step.
  The whole package exists to make that step explicit.
- Removing the fail-closed defaults (e.g. defaulting `cell_aware=True`
  on `Estimator`, or auto-`compute_chain` in `add_signed`). If a default
  could silently weaken a verdict, leave it strict.
- Adding a third on-disk audit format. The two we ship (single-file
  JSON, append-only JSONL) cover the deployment shapes; another
  format would mean a third migration path.

## Code style

- `from __future__ import annotations` at the top of every module.
- Pydantic v2 `BaseModel`; `model_dump(mode="json")` when serializing.
- Public types re-exported from `falsiflyer/__init__.py`.
- Google-style docstrings on every public class and function.
- No silent fallbacks. Raise `FalsiFlyerError` (or a subclass) on any
  invariant violation.
- Keep `falsiflyer/` core dependencies tight: `numpy` and `pydantic`.
  Anything else is an optional extra (`scipy`, `audit`, etc.).

## Adding an adapter

Follow `docs/ADAPTER_GUIDE.md`. The minimum surface is:

- `build_dataset(regime_id="default", out_path=None) -> (Dataset, Path|None)`
- `build_harness(**estimator_overrides) -> Harness`
- A walkthrough script in `examples/walkthrough_<name>.py`
- A test module asserting build → freeze → reload → verdict-finite

If your kernel is proprietary, ship a stub-estimator adapter (like
`q_kernel_tdm`) and a parity-check example that exercises the real
kernel via runtime import — do not vendor proprietary code into this
repo.

## License + sign-off

All contributions are released under the same Apache 2.0 license as
the rest of the project (see `LICENSE`). By submitting a PR, you assert
you have the right to license the code under those terms.

There is no separate CLA. We follow the Apache 2.0 inbound=outbound
convention: contributions are accepted under the terms of the project
license itself (Section 5 of the Apache 2.0 license).

## Reporting security issues

The audit ledger has a documented threat model in
`docs/AUDIT_LEDGER_SPEC.md`. If you find a way to bypass any of the
detection guarantees there (§4.1 through §4.4), please report it
privately first — open a GitHub Security Advisory rather than a
public issue. Bugs in the §4.5–§4.7 NOT-detected list are documented
limitations, not vulnerabilities.

## Questions

Open a GitHub Discussion. For research-context questions about why
the package looks the way it does, the design notes are in
`docs/ARCHITECTURE.md` and the threat model in
`docs/AUDIT_LEDGER_SPEC.md`.
