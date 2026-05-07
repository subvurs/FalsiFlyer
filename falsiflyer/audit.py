"""Audit-trail bundle: kernel slot, byte-identical anchor, ledger.

The Q-kernel-moat falsifications produced one durable invariant: the
``nyxnet/questimator.py`` source was byte-identical across all eight
falsifications. That immutability is what made the cumulative verdict
defensible — we tested the same kernel against eight different
adversarial baselines.

This module ships four primitives that codify that pattern:

1. ``ByteIdenticalAnchor`` — pin (file_path, line_number, sha256). Verify
   on every run; raise ``AnchorDriftError`` on drift.
2. ``KernelSlot`` — interchangeable test-estimator slot with a stable
   slot name and a current implementation hash. The slot name is what
   the decision rule references; the hash records which implementation
   was tested.
3. ``FalsificationRecord`` + ``AuditLedger`` — the durable history. Each
   record carries (id, dataset SHA-256, decision-rule hash, verdict),
   plus an optional hash-chained ``prev_record_hash`` / ``record_hash``
   pair and an optional Ed25519 signature.
4. ``SigningKey`` / ``VerifyKey`` — Ed25519 wrappers that bind one
   signer to many records. Verify-time tampering detection covers
   single-record edits, mid-file inserts, and chain truncation.

Threat model and detection guarantees are spelled out in
``docs/AUDIT_LEDGER_SPEC.md``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field, model_validator

from falsiflyer.errors import AnchorDriftError, FalsiFlyerError


# ---------------------------------------------------------------------------
# Byte-identical anchor
# ---------------------------------------------------------------------------


class ByteIdenticalAnchor(BaseModel):
    """Pin to a specific file's exact bytes.

    Fields
    ------
    file_path:
        Absolute or repo-relative path to the anchored file.
    line_number:
        Optional line of interest within the anchored file (informational).
        Hash is over the **whole file**, not just the line.
    sha256:
        SHA-256 of the file's bytes at commit time.
    notes:
        Free text (e.g. "Questimator estimate_gamma fn body").
    """

    file_path: str
    line_number: Optional[int] = None
    sha256: str
    notes: str = ""

    @classmethod
    def of_file(
        cls,
        file_path: Union[str, Path],
        line_number: Optional[int] = None,
        notes: str = "",
    ) -> "ByteIdenticalAnchor":
        p = Path(file_path)
        digest = _file_sha256(p)
        return cls(
            file_path=str(p),
            line_number=line_number,
            sha256=digest,
            notes=notes,
        )

    def verify(self) -> None:
        """Recompute SHA-256 of ``file_path``; raise on drift."""
        p = Path(self.file_path)
        if not p.exists():
            raise AnchorDriftError(
                f"Anchor file missing: {self.file_path}"
            )
        actual = _file_sha256(p)
        if actual != self.sha256:
            raise AnchorDriftError(
                f"Anchor drift on {self.file_path}:\n"
                f"  recorded: {self.sha256}\n"
                f"  current:  {actual}"
            )


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# KernelSlot
# ---------------------------------------------------------------------------


class KernelSlot(BaseModel):
    """An interchangeable test-estimator slot.

    The slot's ``name`` is the stable identifier the decision rule
    references (e.g. ``"raw_Q"``).  The slot's ``anchor`` records which
    implementation was actually run.  Swapping the kernel = constructing
    a new ``KernelSlot`` with the same name but a fresh anchor.
    """

    name: str
    anchor: ByteIdenticalAnchor
    description: str = ""

    def verify(self) -> None:
        self.anchor.verify()


# ---------------------------------------------------------------------------
# Hash-chain primitives
# ---------------------------------------------------------------------------


# Sentinel prev_record_hash for the first record on a fresh chain.
GENESIS_PREV_HASH = "0" * 64

# Fields excluded from the canonical-record payload — these are computed
# AFTER the record's content is fixed and would otherwise create a
# circular hash-of-hash dependency.
_CHAIN_FIELDS = frozenset({
    "prev_record_hash",
    "record_hash",
    "signature",
    "signer_fingerprint",
})


def _canonical_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode a record's content fields as a stable byte string.

    Drops chain fields, then serializes via ``json.dumps`` with
    ``sort_keys=True`` and tight separators. This is JCS-equivalent
    enough for hash stability across pydantic versions and platforms,
    so long as values are JSON-native (str/int/float/bool/None/list/dict).
    """
    content = {k: v for k, v in payload.items() if k not in _CHAIN_FIELDS}
    return json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _compute_record_hash(payload: Mapping[str, Any], prev_record_hash: str) -> str:
    """SHA-256 over (canonical record bytes ++ prev_record_hash)."""
    h = hashlib.sha256()
    h.update(_canonical_record_bytes(payload))
    h.update(prev_record_hash.encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Ed25519 signing wrappers (lazy-imported)
# ---------------------------------------------------------------------------


def _ed25519_module():
    """Import cryptography's ed25519 module on demand.

    Signing/verifying is opt-in; the package's core does not require
    ``cryptography`` to be installed. Raises ImportError with a
    user-friendly message if signing methods are called without it.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: WPS433
        from cryptography.hazmat.primitives import serialization  # noqa: WPS433
    except ImportError as e:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "falsiflyer.audit signing requires the 'cryptography' package. "
            "Install with: pip install cryptography"
        ) from e
    return ed25519, serialization


class VerifyKey:
    """Ed25519 verify key wrapper.

    Holds the public key and a stable fingerprint (SHA-256 of the raw
    32-byte public key, hex). The fingerprint is what records carry so
    a multi-signer ledger can be checked against a key registry without
    embedding the keys themselves in every row.
    """

    def __init__(self, raw_public_bytes: bytes):
        if len(raw_public_bytes) != 32:
            raise ValueError(
                f"Ed25519 public key must be 32 bytes; got {len(raw_public_bytes)}"
            )
        self._raw = bytes(raw_public_bytes)
        ed25519, _ = _ed25519_module()
        self._impl = ed25519.Ed25519PublicKey.from_public_bytes(self._raw)

    @property
    def raw_bytes(self) -> bytes:
        return self._raw

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._raw).hexdigest()

    def verify(self, signature: bytes, message: bytes) -> None:
        """Raise if the signature does not validate. No return value."""
        from cryptography.exceptions import InvalidSignature  # local import

        try:
            self._impl.verify(signature, message)
        except InvalidSignature as e:
            raise FalsiFlyerError(f"Signature invalid for fingerprint {self.fingerprint}") from e

    @classmethod
    def from_hex(cls, hex_pubkey: str) -> "VerifyKey":
        return cls(bytes.fromhex(hex_pubkey))

    def to_hex(self) -> str:
        return self._raw.hex()


class SigningKey:
    """Ed25519 signing key wrapper.

    Hold this in memory only when signing; do not serialize it into the
    ledger. ``SigningKey.generate()`` is fine for tests and demos; for
    deployment, load from a hardware module / KMS / OS keychain via
    ``SigningKey.from_seed_bytes``.
    """

    def __init__(self, raw_seed_bytes: bytes):
        if len(raw_seed_bytes) != 32:
            raise ValueError(
                f"Ed25519 seed must be 32 bytes; got {len(raw_seed_bytes)}"
            )
        self._seed = bytes(raw_seed_bytes)
        ed25519, _ = _ed25519_module()
        self._impl = ed25519.Ed25519PrivateKey.from_private_bytes(self._seed)
        self._public_raw = self._impl.public_key().public_bytes_raw()

    @classmethod
    def generate(cls) -> "SigningKey":
        ed25519, _ = _ed25519_module()
        priv = ed25519.Ed25519PrivateKey.generate()
        return cls(priv.private_bytes_raw())

    @classmethod
    def from_seed_bytes(cls, seed_bytes: bytes) -> "SigningKey":
        return cls(seed_bytes)

    @property
    def verify_key(self) -> VerifyKey:
        return VerifyKey(self._public_raw)

    @property
    def fingerprint(self) -> str:
        return self.verify_key.fingerprint

    def sign(self, message: bytes) -> bytes:
        return self._impl.sign(message)


# ---------------------------------------------------------------------------
# FalsificationRecord
# ---------------------------------------------------------------------------


class FalsificationRecord(BaseModel):
    """One row in the audit ledger.

    Content fields (committed at construction):
        record_id, timestamp, dataset_sha256, decision_rule_hash,
        kernel_slot, verdict_pass, notes

    Chain fields (filled by AuditLedger when added):
        prev_record_hash, record_hash

    Signature fields (filled by sign-then-add):
        signature (hex), signer_fingerprint
    """

    record_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_sha256: str
    decision_rule_hash: str
    kernel_slot: KernelSlot
    verdict_pass: bool
    notes: str = ""

    prev_record_hash: Optional[str] = None
    record_hash: Optional[str] = None
    signature: Optional[str] = None
    signer_fingerprint: Optional[str] = None

    def canonical_payload(self) -> Dict[str, Any]:
        """Return the content-field dict (used for hash + signature)."""
        full = self.model_dump(mode="json")
        return {k: v for k, v in full.items() if k not in _CHAIN_FIELDS}


# ---------------------------------------------------------------------------
# AuditLedger
# ---------------------------------------------------------------------------


class LedgerError(FalsiFlyerError):
    """Audit-ledger durability violation (chain break, signature mismatch)."""


class AuditLedger(BaseModel):
    """Append-only ledger of FalsificationRecords.

    JSON-round-trippable; the ledger file is the durable artifact that
    makes "Falsification-as-a-Service" auditable across regulatory cycles.

    Two operating modes (mix freely on the same ledger):

    * **Unsigned**: ``add(record)`` — record_id uniqueness only.
      Backward-compatible with the v0.1 ledger format.
    * **Signed + chained**: ``add_signed(record, signing_key)`` — fills
      ``prev_record_hash``, ``record_hash``, ``signature``, and
      ``signer_fingerprint``, then appends. ``verify_chain()`` and
      ``verify_signatures()`` walk the chain end-to-end.

    For cross-process append-only semantics, prefer the JSONL flavor:
    ``AuditLedger.append_to_file(record, path, signing_key=...)`` /
    ``AuditLedger.load_jsonl(path)``.
    """

    records: List[FalsificationRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "AuditLedger":
        ids = [r.record_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise FalsiFlyerError(
                f"AuditLedger has duplicate record_ids: {ids}"
            )
        return self

    # -- unsigned API (v0.1 compatible) -------------------------------------

    def add(self, record: FalsificationRecord) -> None:
        if any(r.record_id == record.record_id for r in self.records):
            raise FalsiFlyerError(
                f"AuditLedger.add: record_id {record.record_id!r} already present"
            )
        self.records.append(record)

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AuditLedger":
        p = Path(path)
        with p.open() as f:
            payload = json.load(f)
        return cls.model_validate(payload)

    # -- signed + chained API -----------------------------------------------

    def _last_record_hash(self) -> str:
        if not self.records:
            return GENESIS_PREV_HASH
        last = self.records[-1]
        if last.record_hash is None:
            raise LedgerError(
                "Ledger has unchained tail records; cannot extend chain. "
                "Use compute_chain() first or start a fresh ledger."
            )
        return last.record_hash

    def add_signed(
        self,
        record: FalsificationRecord,
        signing_key: SigningKey,
    ) -> FalsificationRecord:
        """Chain-link, sign, and append the record.

        Mutates the input record (fills chain + signature fields) and
        returns it. Raises ``FalsiFlyerError`` if the record_id is already
        present, or ``LedgerError`` if the chain cannot be extended.
        """
        if any(r.record_id == record.record_id for r in self.records):
            raise FalsiFlyerError(
                f"AuditLedger.add_signed: record_id {record.record_id!r} already present"
            )
        prev_hash = self._last_record_hash()
        record.prev_record_hash = prev_hash
        record.record_hash = _compute_record_hash(record.canonical_payload(), prev_hash)
        record.signature = signing_key.sign(record.record_hash.encode("ascii")).hex()
        record.signer_fingerprint = signing_key.fingerprint
        self.records.append(record)
        return record

    def compute_chain(self) -> None:
        """Recompute prev_record_hash + record_hash for every record.

        Use this when migrating a v0.1 unsigned ledger to chained mode,
        or when records were appended out of order. Does NOT touch
        signatures (signatures are computed at add-time by the signer
        who held the key); use ``add_signed`` for new signed entries.
        """
        prev = GENESIS_PREV_HASH
        for r in self.records:
            r.prev_record_hash = prev
            r.record_hash = _compute_record_hash(r.canonical_payload(), prev)
            prev = r.record_hash

    def verify_chain(self) -> None:
        """Walk the records; raise ``LedgerError`` on any chain break.

        Detects: edited content of a chained record, mid-file inserts,
        chain truncation, swapped-order entries.
        """
        prev = GENESIS_PREV_HASH
        for i, r in enumerate(self.records):
            if r.record_hash is None or r.prev_record_hash is None:
                raise LedgerError(
                    f"Record #{i} ({r.record_id!r}) is unchained (record_hash/prev_record_hash missing)"
                )
            if r.prev_record_hash != prev:
                raise LedgerError(
                    f"Chain break at record #{i} ({r.record_id!r}): "
                    f"prev_record_hash={r.prev_record_hash!r}, expected {prev!r}"
                )
            recomputed = _compute_record_hash(r.canonical_payload(), prev)
            if r.record_hash != recomputed:
                raise LedgerError(
                    f"Content tamper at record #{i} ({r.record_id!r}): "
                    f"stored record_hash={r.record_hash!r}, recomputed={recomputed!r}"
                )
            prev = r.record_hash

    def verify_signatures(
        self,
        verify_keys: Mapping[str, VerifyKey],
    ) -> None:
        """Verify each record's signature against a key registry.

        ``verify_keys`` maps signer_fingerprint -> VerifyKey. Records
        without a signature are skipped (allowed; a partially-signed
        ledger is a documentation artifact, not a tamper signal). Raises
        ``LedgerError`` on the first invalid signature or missing key.
        """
        for i, r in enumerate(self.records):
            if r.signature is None:
                continue
            if r.signer_fingerprint is None:
                raise LedgerError(
                    f"Record #{i} ({r.record_id!r}) has signature but no signer_fingerprint"
                )
            if r.record_hash is None:
                raise LedgerError(
                    f"Record #{i} ({r.record_id!r}) signed but unchained"
                )
            vk = verify_keys.get(r.signer_fingerprint)
            if vk is None:
                raise LedgerError(
                    f"Record #{i} ({r.record_id!r}): no verify key for "
                    f"fingerprint {r.signer_fingerprint!r}"
                )
            try:
                vk.verify(bytes.fromhex(r.signature), r.record_hash.encode("ascii"))
            except FalsiFlyerError as e:
                raise LedgerError(
                    f"Record #{i} ({r.record_id!r}) signature invalid: {e}"
                ) from e

    # -- JSONL append-only file API ----------------------------------------

    @staticmethod
    def append_to_file(
        record: FalsificationRecord,
        path: Union[str, Path],
        signing_key: Optional[SigningKey] = None,
    ) -> FalsificationRecord:
        """Append a record to a JSONL ledger file with chain check.

        File format: one JSON object per line, no trailing newline
        required on the final line. Opens with mode ``"a"``; reads only
        the last existing line to determine the chain anchor. Suitable
        for cross-process / cross-session appending without rewriting
        the whole file.

        If ``signing_key`` is provided, the record is signed before
        write. If omitted, the record is chained but unsigned (still
        catches mid-file edits at verify time, just not external forgery).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        prev_hash = _last_jsonl_record_hash(p)

        record.prev_record_hash = prev_hash
        record.record_hash = _compute_record_hash(record.canonical_payload(), prev_hash)
        if signing_key is not None:
            record.signature = signing_key.sign(record.record_hash.encode("ascii")).hex()
            record.signer_fingerprint = signing_key.fingerprint

        line = json.dumps(record.model_dump(mode="json"), default=str)
        with p.open("a") as f:
            f.write(line + "\n")
        return record

    @classmethod
    def load_jsonl(cls, path: Union[str, Path]) -> "AuditLedger":
        """Load a JSONL ledger; chain validity is NOT checked here.

        Call ``verify_chain()`` and/or ``verify_signatures()`` after
        loading to assert durability invariants.
        """
        p = Path(path)
        records: List[FalsificationRecord] = []
        if not p.exists():
            return cls(records=[])
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(FalsificationRecord.model_validate(payload))
        return cls(records=records)


def _last_jsonl_record_hash(path: Path) -> str:
    """Return the last record's record_hash, or GENESIS_PREV_HASH on empty/missing.

    Reads the whole file; for very large ledgers a tail-seek
    implementation could be substituted, but JSONL append correctness
    only requires the last non-empty line, and audit ledgers are
    small (one row per falsification run).
    """
    if not path.exists():
        return GENESIS_PREV_HASH
    last_line: Optional[str] = None
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return GENESIS_PREV_HASH
    last = json.loads(last_line)
    rh = last.get("record_hash")
    if rh is None:
        raise LedgerError(
            f"JSONL ledger at {path} has unchained tail line; cannot extend "
            "chain via append_to_file. Migrate via compute_chain()+save() first."
        )
    return rh


# ---------------------------------------------------------------------------
# Decision-rule hashing helper
# ---------------------------------------------------------------------------


def hash_decision_rule(rule_payload: Dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of a serialized DecisionRule.

    Used by ``FalsificationRecord.decision_rule_hash`` to detect any
    post-hoc threshold tuning between the freeze and run steps.
    """
    canonical = json.dumps(rule_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
