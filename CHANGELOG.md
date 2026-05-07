# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-06

### Added

- **Audit ledger durability extension** (`falsiflyer.audit`):
  - Hash-chained records via `prev_record_hash` + `record_hash` fields.
    Genesis sentinel `GENESIS_PREV_HASH = "0" * 64`.
  - Ed25519 signing wrappers `SigningKey` and `VerifyKey` with stable
    32-byte raw-public-key SHA-256 fingerprints. Signature is over the
    `record_hash`, binding signer + body + chain position.
  - `AuditLedger.add_signed(record, signing_key)` chains and signs in
    one call.
  - `AuditLedger.compute_chain()` migrates a v0.1 unsigned ledger into
    chained mode without re-signing.
  - `AuditLedger.verify_chain()` detects content tamper, mid-chain
    inserts, unchained tail.
  - `AuditLedger.verify_signatures(registry)` verifies under a
    `Mapping[fingerprint, VerifyKey]` registry; skips unsigned records.
  - `AuditLedger.append_to_file(record, path, signing_key=None)` and
    `AuditLedger.load_jsonl(path)` for cross-process append-only JSONL.
  - `LedgerError` raised on chain breaks and signature mismatches.
- New optional dependency: `pip install falsiflyer[audit]` pulls in
  `cryptography>=41.0`. Core install remains `numpy + pydantic` only;
  signing imports are lazy.
- `docs/AUDIT_LEDGER_SPEC.md` documenting threat model, on-disk
  formats, detection guarantees, API contract, key-management
  recommendations.
- 18 new tests in `tests/test_audit.py` covering chain construction,
  tamper detection, signature round-trip, multi-signer, JSONL append
  semantics, unchained-tail refusal.
- Apache 2.0 LICENSE + NOTICE.
- CONTRIBUTING.md, CHANGELOG.md, GitHub Actions CI matrix
  (Python 3.9–3.12 × {core, audit, scipy}).
- Project metadata: SPDX `Apache-2.0` license declaration, classifiers,
  repository URLs, keywords.

### Changed

- `FalsificationRecord` gains four optional fields:
  `prev_record_hash`, `record_hash`, `signature`, `signer_fingerprint`.
  All default to `None` for backward compatibility with v0.1 ledgers.
- Public API additions in `falsiflyer.__init__`:
  `SigningKey`, `VerifyKey`, `LedgerError`, `GENESIS_PREV_HASH`.

### Backward compatibility

- v0.1 unsigned ledgers (`{"records": [...]}` with no chain fields) load
  and save under v0.2 unchanged. `add()` and `save()`/`load()` keep
  their v0.1 semantics. To migrate to chained mode call
  `compute_chain()` once after load.
- The single existing on-disk format (single-file JSON via
  `AuditLedger.save`) gains the new optional fields but does not
  break old readers.

## [0.1.0] — 2026-05-05

### Added

- Initial release. Five primitives:
  - `falsiflyer.prereg` — hash-commit + frozen-dataset loader + decision rule
  - `falsiflyer.baselines` — proper-baseline (DGP-matched) noise templates
    (`Gaussian`, `Proportional`, `Poisson`, `LogGaussian`, `Binomial`)
  - `falsiflyer.harness` — decision-rule-bound benchmark runner
  - `falsiflyer.diagnostic` — disagreement-split diagnostic template
  - `falsiflyer.audit` — kernel slot, byte-identical anchor, ledger
    (unsigned, single-file JSON)
- Adapters:
  - `adapters/q_kernel_tdm` — wrapping the `benchmark_q_kernel_tdm_7k`
    recorded baseline; `examples/parity_check_7k.py` reproduces the
    PASS verdict end-to-end.
  - `adapters/impax_signal_disc` — wrapping the real
    `ImpaxFrequencyScanner` proprietary kernel;
    `examples/walkthrough_impax.py` (stub) +
    `examples/parity_check_impax.py` (real kernel; FAIL verdict
    on continuous-frequency rel-err, honestly reported).
- Documentation: `README.md`, `docs/ARCHITECTURE.md`,
  `docs/ADAPTER_GUIDE.md`.
- 47 tests covering all five primitives + both adapters.

[0.2.0]: #
[0.1.0]: #
