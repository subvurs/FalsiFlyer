# Audit Ledger Durability Specification

**Module**: `falsiflyer.audit`
**Version**: 0.2 (chain + sign extension; v0.1 unsigned ledgers remain valid)
**Status**: implemented and tested (`tests/test_audit.py` — 25 tests)

This document specifies the threat model, detection guarantees, and operational
contract of the FalsiFlyer audit ledger. Read it before reasoning about whether a
given ledger row "counts" as durable evidence.

---

## 1. Purpose

The Q-kernel-moat falsifications produced one durable invariant: the
`nyxnet/questimator.py` source was byte-identical across all eight
falsifications. That immutability is what made the cumulative verdict
defensible — the same kernel was tested against eight different adversarial
baselines, and any future reviewer can recompute the bytes and verify.

The audit ledger generalizes that pattern. A `FalsificationRecord` ties together:

1. **What was tested** — `kernel_slot.anchor.sha256` (byte-identical pin).
2. **What it was tested on** — `dataset_sha256` (the frozen-dataset hash).
3. **What rule was used** — `decision_rule_hash` (SHA-256 of the canonical
   serialized `DecisionRule` object).
4. **What happened** — `verdict_pass: bool` plus free-text `notes`.
5. **Who is on the hook for it** — `signer_fingerprint` + Ed25519 `signature`
   over the record's `record_hash`.
6. **Where it sits in time** — `prev_record_hash` + `record_hash`, hash-linked
   back to a genesis sentinel.

A reviewer can take any single record from the ledger, find the kernel file at
the recorded SHA, find the frozen dataset at the recorded SHA, recompute the
decision rule from those, and check that the verdict matches. Tampering with
any of those bindings is detected by `verify_chain()` and/or
`verify_signatures()`.

---

## 2. On-Disk Formats

Two persistent formats are supported. Both are valid; pick by deployment shape.

### 2.1 Single-file JSON (`AuditLedger.save` / `AuditLedger.load`)

```json
{
  "records": [
    {
      "record_id": "#1",
      "timestamp": "2026-05-06T12:00:00+00:00",
      "dataset_sha256": "…",
      "decision_rule_hash": "…",
      "kernel_slot": { "name": "raw_Q", "anchor": { … } },
      "verdict_pass": false,
      "notes": "…",
      "prev_record_hash": "0000…0000",
      "record_hash": "…",
      "signature": "…",
      "signer_fingerprint": "…"
    }
  ]
}
```

Suited for atomic-replace workflows (small ledger, single writer, full
re-serialize on every change). Compatible with v0.1 ledgers that have
`prev_record_hash`/`record_hash`/`signature`/`signer_fingerprint` absent or
`null`.

### 2.2 JSONL append-only (`AuditLedger.append_to_file` / `load_jsonl`)

One record per line; one `json.dumps` call per record; no whole-file rewrite.

```
{"record_id":"#1", … ,"prev_record_hash":"0000…","record_hash":"…","signature":"…"}
{"record_id":"#2", … ,"prev_record_hash":"…","record_hash":"…","signature":"…"}
```

Suited for cross-process / cross-session writes (CI runs that each append a
record). The append helper reads only the **last non-empty line** to determine
the chain anchor, so the hot-path cost is O(1) per write in the common
"append-and-go" case. (The current implementation reads the whole file because
audit ledgers are small; the contract allows a tail-seek implementation.)

---

## 3. Hash-Chain Construction

### 3.1 Genesis

```python
GENESIS_PREV_HASH = "0" * 64   # 64 hex zeroes
```

The first record on a chain has `prev_record_hash == GENESIS_PREV_HASH`.

### 3.2 Canonical record bytes

Chain fields are excluded from the hash input (otherwise the hash would depend
on itself). The excluded set is:

```python
_CHAIN_FIELDS = {
    "prev_record_hash", "record_hash",
    "signature",        "signer_fingerprint",
}
```

Every other field on `FalsificationRecord` is dumped via Pydantic's
`model_dump(mode="json")`, then re-serialized via:

```python
json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
```

This is **JCS-equivalent for JSON-native values** (strings, ints, floats,
bools, None, lists, dicts). Float canonicalization beyond Python's
`repr()` is out of scope; if your records contain floats, normalize them
to bounded-precision strings or fixed-precision decimals before adding to
the ledger.

### 3.3 Record hash

```python
record_hash = sha256(canonical_record_bytes(payload) ++ prev_record_hash.ascii)
```

The `prev_record_hash` is concatenated **after** the canonical body, in ASCII.
This binds each record's hash to its predecessor; the predecessor's hash
depends on its own predecessor, and so on back to genesis.

### 3.4 Signature

```python
signature        = Ed25519.sign(signing_key, record_hash.ascii_bytes)
signer_fingerprint = sha256(public_key_raw).hexdigest()
```

The signature is over the **record_hash**, not the body. This means: a valid
signature plus a valid chain implies that the signer attested to the body
**that produced this hash, at this position in the chain**. It does **not**
mean the signer attested to the body in isolation — re-using a record at a
different chain position would invalidate the signature.

---

## 4. Threat Model and Detection Guarantees

The ledger is designed to detect **post-hoc tampering** of an existing ledger.
It is **not** a write-once medium; durability against insider compromise of the
write path requires WORM storage (object-lock S3, append-only SQLite,
git-with-pre-receive-hooks, etc.) layered underneath.

### 4.1 Detected: content tamper

Editing a content field (`notes`, `verdict_pass`, `dataset_sha256`, …) on a
chained record changes its canonical bytes, which changes the recomputed
`record_hash`, which mismatches the stored `record_hash`.

→ `verify_chain()` raises `LedgerError` with `"Content tamper at record #N"`.

Tested in `test_verify_chain_detects_content_tamper`.

### 4.2 Detected: mid-chain insert

Inserting a forged record between two real ones forces either:

- the forged record's `prev_record_hash` to mismatch its actual neighbour
  (chain break), or
- the next real record's `prev_record_hash` to mismatch the forged record's
  `record_hash` (chain break).

→ `verify_chain()` raises `LedgerError` with `"Chain break at record #N"`.

Tested in `test_verify_chain_detects_mid_record_insert`.

### 4.3 Detected: tail truncation

If you drop record #N from a chain of length N+K, the remaining ledger still
verifies its own chain (truncation isn't a chain break per se). Detection
relies on **external pinning**: the consumer should know the expected
`record_hash` of the most-recent record (e.g. published in a tamper-evident
public log, posted to a CI status badge, mirrored to a second store).

→ Not detected by `verify_chain()` alone. Mitigation: publish the latest
`record_hash` out-of-band.

### 4.4 Detected: external forgery (with signing)

An attacker without the signing key cannot produce a valid Ed25519 signature
over a forged `record_hash`. Even if they reconstruct a plausible chain
(picking a `prev_record_hash`, computing `record_hash` over a forged body),
their signature won't verify under the registered `VerifyKey`.

→ `verify_signatures(registry)` raises `LedgerError` with
`"signature invalid"`.

Tested in `test_verify_signatures_rejects_mutated_record`,
`test_verify_signatures_rejects_wrong_key`.

### 4.5 NOT detected: signing-key compromise

If the signing key is exfiltrated, the attacker can sign arbitrary records.
Mitigations are operational: rotate keys (each record records the
`signer_fingerprint`, so a registry-wide check can flag old fingerprints),
require multi-party signatures by publishing two ledgers signed by different
keys and cross-checking.

### 4.6 NOT detected: dataset / kernel substitution

The ledger only records the SHAs. If an attacker substitutes both the kernel
file at the recorded path AND the dataset file at the recorded path AND
recomputes both SHAs to match, the ledger will validate against the
substituted artifacts. Mitigations: (a) anchor the kernel file's path and SHA
in `KernelSlot.anchor` and verify with `slot.verify()` at run time;
(b) verify the dataset SHA against `DatasetCommit.sha256_data_payload` at load
time (this is what `load_frozen_dataset` does already).

### 4.7 NOT detected: mixed-mode swap

The ledger format allows mixing chained records with v0.1 unsigned records
on the same ledger. A v0.1 unsigned record can be replaced with a different
v0.1 unsigned record without detection — there is no chain field to mismatch.
Mitigation: once a ledger has any chained record, treat the addition of any
unchained record as a policy violation (enforced by `append_to_file`, which
refuses to extend an unchained tail).

---

## 5. API Contract

### 5.1 Backward compatibility

A v0.1 unsigned ledger (records with `prev_record_hash=None`,
`record_hash=None`, `signature=None`, `signer_fingerprint=None`) loads and
saves cleanly under v0.2. `add()` continues to work without chain or signing.

To migrate a v0.1 ledger to chained mode:

```python
ledger = AuditLedger.load("legacy.json")
ledger.compute_chain()       # fills prev_record_hash + record_hash for all records
ledger.save("legacy_v2.json")
ledger.verify_chain()        # no raise
```

`compute_chain()` does **not** sign; signatures are computed at add-time
by the keyholder. Migrating to a signed ledger requires re-running the
falsifications under a signer.

### 5.2 Method-by-method contract

| Method | Mutates | Reads | Raises |
|---|---|---|---|
| `add(record)` | ledger.records | record_id | `FalsiFlyerError` on duplicate id |
| `add_signed(record, sk)` | ledger.records, record.{prev_record_hash, record_hash, signature, signer_fingerprint} | last record_hash | `FalsiFlyerError` on dup id; `LedgerError` if tail unchained |
| `compute_chain()` | every record's prev_record_hash + record_hash | every record's content | — |
| `verify_chain()` | — | every record's chain + content | `LedgerError` on any mismatch |
| `verify_signatures(reg)` | — | every record's signature + record_hash + signer_fingerprint | `LedgerError` on missing key, missing fingerprint, missing record_hash, or invalid signature |
| `save(path)` | filesystem | self | — |
| `load(path)` | — | filesystem | `pydantic.ValidationError` on schema break |
| `append_to_file(rec, path, sk=None)` | filesystem, rec | last line of path | `LedgerError` on unchained tail |
| `load_jsonl(path)` | — | filesystem | `pydantic.ValidationError` on schema break |

`verify_chain()` and `verify_signatures()` are **decoupled by design**: a
ledger may be chain-valid but signature-invalid (key was compromised and the
attacker re-signed a forged body but used a different key) and vice versa
(legitimate signatures over a chain that a non-signer corrupted in transit).
Always run both in production.

### 5.3 Cryptography is optional

`SigningKey` and `VerifyKey` lazy-import `cryptography.hazmat.primitives.
asymmetric.ed25519`. Importing `FalsiFlyer` itself does not require
`cryptography`. Install with:

```
pip install falsiflyer[audit]
```

Calling `SigningKey.generate()` without `cryptography` installed raises
`ImportError` with a one-line install hint.

---

## 6. Operational Recommendations

### 6.1 Key management

- Generate signing keys per-environment (CI, staging, prod), not per-developer.
- Store seed bytes in a KMS / hardware module / OS keychain. The
  `SigningKey.from_seed_bytes` helper is the boundary; never commit a
  raw seed to source control.
- Publish the public-key fingerprint(s) somewhere reviewers can find them
  (whitepaper, README, project landing page).

### 6.2 Append cadence

One `append_to_file` per falsification run is the natural unit. The
`record_id` should be deterministic and human-readable (e.g. `"#7e"`,
`"#7k"`, `"impax-2026-05-06"`).

### 6.3 External pinning

Publish the latest `record_hash` somewhere out-of-band:

- Signed CI status badge.
- Dispatch as a webhook to an immutable log.
- Mirror to a second store (different cloud, different account).

A reviewer who knows the expected latest `record_hash` detects truncation
attacks in §4.3.

### 6.4 What the ledger does NOT replace

- It is **not** a notarization service. Records are signed by a key the
  project controls, not by a third party.
- It is **not** a witness chain. The signer attests "I ran this falsification
  and got this verdict"; reviewers must independently verify the kernel and
  dataset SHAs.
- It is **not** an immutable log. The OS can still `rm -f` the file; use
  WORM storage for durability against insider compromise.

What it IS: a tamper-evident, signature-backed binding between a kernel hash,
a dataset hash, a decision rule hash, and a verdict, hash-chained to all
prior records. Tampering with any of those inputs is detectable post-hoc by
any reviewer with read access to the ledger and the registered verify keys.

---

## 7. Test Coverage

`tests/test_audit.py` exercises:

- Anchor SHA computation, drift, missing-file
- KernelSlot delegation
- v0.1 unsigned add, dup-id rejection, save/load round-trip
- Decision-rule canonical hash
- `compute_chain` + `verify_chain` happy path
- `verify_chain` on content tamper
- `verify_chain` on mid-chain insert
- `verify_chain` on unchained tail
- Save/load round-trip preserves chain
- `SigningKey.generate` + `verify_key.verify` round-trip
- Fingerprint stability across `from_seed_bytes` calls
- `VerifyKey.from_hex` / `to_hex` round-trip
- `add_signed` chains and signs
- Verify rejects mutated record (chain breaks first)
- Verify rejects wrong key (signature invalid)
- Verify rejects missing key
- Verify skips unsigned records
- JSONL append chain across multiple calls
- JSONL append unsigned chain (no crypto dep)
- JSONL append detects in-place file edit
- JSONL load on missing file returns empty ledger
- JSONL append refuses to extend an unchained tail

25 tests; full suite 65 pass + 1 skip after extension.
