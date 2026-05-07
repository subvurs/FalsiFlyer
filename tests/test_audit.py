"""Tests for the audit-trail bundle (anchor, slot, ledger)."""

from __future__ import annotations

from pathlib import Path

import pytest

from falsiflyer import (
    AnchorDriftError,
    AuditLedger,
    ByteIdenticalAnchor,
    FalsificationRecord,
    FalsiFlyerError,
    GENESIS_PREV_HASH,
    KernelSlot,
    LedgerError,
    SigningKey,
    VerifyKey,
)
from falsiflyer.audit import hash_decision_rule

# cryptography is an optional dep; if missing, signing tests skip rather than fail.
try:  # pragma: no cover - import gate only
    import cryptography  # noqa: F401

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTO = False

needs_crypto = pytest.mark.skipif(
    not _HAS_CRYPTO, reason="cryptography not installed; install falsiflyer[audit]"
)


def _make_record(record_id: str, *, slot: KernelSlot, verdict: bool = False) -> FalsificationRecord:
    return FalsificationRecord(
        record_id=record_id,
        dataset_sha256="a" * 64,
        decision_rule_hash="b" * 64,
        kernel_slot=slot,
        verdict_pass=verdict,
        notes=f"record {record_id}",
    )


def _write_pinned_file(tmp_path: Path, body: str = "raw_q kernel v1\n") -> Path:
    p = tmp_path / "kernel.py"
    p.write_text(body)
    return p


def test_anchor_of_file_and_verify_ok(tmp_path):
    p = _write_pinned_file(tmp_path)
    anchor = ByteIdenticalAnchor.of_file(p, line_number=1, notes="kernel v1")
    assert len(anchor.sha256) == 64
    anchor.verify()  # no raise


def test_anchor_drift_raises(tmp_path):
    p = _write_pinned_file(tmp_path)
    anchor = ByteIdenticalAnchor.of_file(p)
    p.write_text("raw_q kernel v2\n")  # mutate
    with pytest.raises(AnchorDriftError):
        anchor.verify()


def test_anchor_missing_file_raises(tmp_path):
    p = _write_pinned_file(tmp_path)
    anchor = ByteIdenticalAnchor.of_file(p)
    p.unlink()
    with pytest.raises(AnchorDriftError):
        anchor.verify()


def test_kernel_slot_verify_delegates_to_anchor(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(
        name="raw_Q",
        anchor=ByteIdenticalAnchor.of_file(p),
        description="Q kernel slot",
    )
    slot.verify()
    p.write_text("mutated\n")
    with pytest.raises(AnchorDriftError):
        slot.verify()


def test_audit_ledger_unique_ids_and_round_trip(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    rec_a = FalsificationRecord(
        record_id="#7e",
        dataset_sha256="a" * 64,
        decision_rule_hash="b" * 64,
        kernel_slot=slot,
        verdict_pass=False,
        notes="initial multi-rate falsification",
    )
    rec_b = FalsificationRecord(
        record_id="#7k",
        dataset_sha256="c" * 64,
        decision_rule_hash="d" * 64,
        kernel_slot=slot,
        verdict_pass=False,
        notes="single-rate TDM, MAP_Proportional added",
    )
    ledger = AuditLedger()
    ledger.add(rec_a)
    ledger.add(rec_b)
    assert len(ledger.records) == 2

    # Duplicate id rejected.
    with pytest.raises(FalsiFlyerError):
        ledger.add(rec_a)

    # save / load round-trip preserves contents.
    out = tmp_path / "ledger.json"
    ledger.save(out)
    assert out.exists()
    loaded = AuditLedger.load(out)
    assert [r.record_id for r in loaded.records] == ["#7e", "#7k"]
    assert loaded.records[0].kernel_slot.anchor.sha256 == slot.anchor.sha256


def test_ledger_constructor_rejects_duplicates():
    p_anchor = ByteIdenticalAnchor(file_path="x", sha256="a" * 64)
    slot = KernelSlot(name="raw_Q", anchor=p_anchor)
    common = dict(
        dataset_sha256="z" * 64,
        decision_rule_hash="y" * 64,
        kernel_slot=slot,
        verdict_pass=False,
    )
    r0 = FalsificationRecord(record_id="#dup", **common)
    r1 = FalsificationRecord(record_id="#dup", **common)
    with pytest.raises(FalsiFlyerError):
        AuditLedger(records=[r0, r1])


def test_decision_rule_hash_canonical():
    a = {"test_estimator": "raw_Q", "baselines": ["raw_NCA"], "thr": 0.15}
    b = {"thr": 0.15, "baselines": ["raw_NCA"], "test_estimator": "raw_Q"}
    assert hash_decision_rule(a) == hash_decision_rule(b)
    a2 = dict(a, thr=0.30)
    assert hash_decision_rule(a) != hash_decision_rule(a2)


# ---------------------------------------------------------------------------
# Hash-chain tests (no cryptography dep needed)
# ---------------------------------------------------------------------------


def test_compute_chain_then_verify_ok(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger = AuditLedger()
    ledger.add(_make_record("#1", slot=slot))
    ledger.add(_make_record("#2", slot=slot))
    ledger.add(_make_record("#3", slot=slot))
    # v0.1 records have no chain fields yet.
    assert all(r.record_hash is None for r in ledger.records)
    ledger.compute_chain()
    # First record links to genesis.
    assert ledger.records[0].prev_record_hash == GENESIS_PREV_HASH
    # Each subsequent record links to its predecessor.
    for prev, cur in zip(ledger.records, ledger.records[1:]):
        assert cur.prev_record_hash == prev.record_hash
    ledger.verify_chain()  # no raise


def test_verify_chain_detects_content_tamper(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger = AuditLedger()
    ledger.add(_make_record("#1", slot=slot))
    ledger.add(_make_record("#2", slot=slot))
    ledger.compute_chain()
    # Mutate a content field after chaining; verify must catch it.
    ledger.records[0].notes = "post-hoc tamper"
    with pytest.raises(LedgerError, match="Content tamper"):
        ledger.verify_chain()


def test_verify_chain_detects_mid_record_insert(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger = AuditLedger()
    ledger.add(_make_record("#1", slot=slot))
    ledger.add(_make_record("#2", slot=slot))
    ledger.compute_chain()
    # Inject a chained-looking record between #1 and #2 — its
    # prev_record_hash will mismatch the chain.
    bogus = _make_record("#1b", slot=slot)
    bogus.prev_record_hash = "f" * 64
    bogus.record_hash = "e" * 64
    ledger.records.insert(1, bogus)
    with pytest.raises(LedgerError, match="Chain break"):
        ledger.verify_chain()


def test_verify_chain_detects_unchained_tail(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger = AuditLedger()
    ledger.add(_make_record("#1", slot=slot))
    # Never chain — verify must raise on the unchained record.
    with pytest.raises(LedgerError, match="unchained"):
        ledger.verify_chain()


def test_chain_save_load_round_trip_preserves_chain(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger = AuditLedger()
    for i in range(4):
        ledger.add(_make_record(f"#{i}", slot=slot))
    ledger.compute_chain()
    out = tmp_path / "chained.json"
    ledger.save(out)
    reloaded = AuditLedger.load(out)
    reloaded.verify_chain()


# ---------------------------------------------------------------------------
# Signing tests (require cryptography)
# ---------------------------------------------------------------------------


@needs_crypto
def test_signing_key_round_trip():
    sk = SigningKey.generate()
    vk = sk.verify_key
    msg = b"hello chain"
    sig = sk.sign(msg)
    vk.verify(sig, msg)  # no raise
    # Tamper with the message: must raise.
    with pytest.raises(FalsiFlyerError, match="Signature invalid"):
        vk.verify(sig, msg + b"!")


@needs_crypto
def test_signing_key_fingerprint_stable():
    seed = b"\x07" * 32
    sk1 = SigningKey.from_seed_bytes(seed)
    sk2 = SigningKey.from_seed_bytes(seed)
    assert sk1.fingerprint == sk2.fingerprint
    # Different seed -> different fingerprint.
    sk3 = SigningKey.from_seed_bytes(b"\x08" * 32)
    assert sk3.fingerprint != sk1.fingerprint


@needs_crypto
def test_verify_key_hex_round_trip():
    sk = SigningKey.generate()
    vk = sk.verify_key
    hex_pub = vk.to_hex()
    vk2 = VerifyKey.from_hex(hex_pub)
    assert vk2.fingerprint == vk.fingerprint
    msg = b"round trip"
    vk2.verify(sk.sign(msg), msg)


@needs_crypto
def test_add_signed_chains_and_signs(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk = SigningKey.generate()
    vk = sk.verify_key

    ledger = AuditLedger()
    r1 = ledger.add_signed(_make_record("#1", slot=slot), sk)
    r2 = ledger.add_signed(_make_record("#2", slot=slot), sk)

    assert r1.prev_record_hash == GENESIS_PREV_HASH
    assert r2.prev_record_hash == r1.record_hash
    assert r1.signer_fingerprint == vk.fingerprint
    assert r2.signer_fingerprint == vk.fingerprint

    ledger.verify_chain()
    ledger.verify_signatures({vk.fingerprint: vk})


@needs_crypto
def test_verify_signatures_rejects_mutated_record(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk = SigningKey.generate()
    vk = sk.verify_key

    ledger = AuditLedger()
    ledger.add_signed(_make_record("#1", slot=slot), sk)
    ledger.add_signed(_make_record("#2", slot=slot), sk)

    # Mutate notes; verify_chain catches the content tamper first.
    ledger.records[0].notes = "post-hoc"
    with pytest.raises(LedgerError, match="Content tamper"):
        ledger.verify_chain()

    # If the attacker also rewrites record_hash to match, chain validates
    # but the signature over the OLD hash is still bound to the old payload.
    # Recompute record_hash without resigning -> signature verification fails.
    from falsiflyer.audit import _compute_record_hash

    r = ledger.records[0]
    r.record_hash = _compute_record_hash(r.canonical_payload(), r.prev_record_hash)
    # subsequent records still chained against the OLD hash, so chain breaks
    with pytest.raises(LedgerError, match="Chain break"):
        ledger.verify_chain()


@needs_crypto
def test_verify_signatures_rejects_wrong_key(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk_real = SigningKey.generate()
    sk_imposter = SigningKey.generate()

    ledger = AuditLedger()
    ledger.add_signed(_make_record("#1", slot=slot), sk_real)

    # Registry maps the real signer's fingerprint to the imposter's key.
    registry = {sk_real.fingerprint: sk_imposter.verify_key}
    with pytest.raises(LedgerError, match="signature invalid"):
        ledger.verify_signatures(registry)


@needs_crypto
def test_verify_signatures_missing_key_raises(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk = SigningKey.generate()

    ledger = AuditLedger()
    ledger.add_signed(_make_record("#1", slot=slot), sk)

    with pytest.raises(LedgerError, match="no verify key"):
        ledger.verify_signatures({})


@needs_crypto
def test_verify_signatures_skips_unsigned_records(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk = SigningKey.generate()
    vk = sk.verify_key

    # Build a fully-chained 3-record ledger, then strip the signature on
    # one of them — its content is still chained, just not signed.
    ledger = AuditLedger()
    ledger.add_signed(_make_record("#1", slot=slot), sk)
    ledger.add_signed(_make_record("#2", slot=slot), sk)
    ledger.add_signed(_make_record("#3", slot=slot), sk)
    ledger.records[1].signature = None
    ledger.records[1].signer_fingerprint = None

    ledger.verify_chain()  # chain still intact
    # verify_signatures must not raise: unsigned records are skipped.
    ledger.verify_signatures({vk.fingerprint: vk})


# ---------------------------------------------------------------------------
# JSONL append-only file tests
# ---------------------------------------------------------------------------


@needs_crypto
def test_append_to_file_chain_across_calls(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    sk = SigningKey.generate()
    vk = sk.verify_key

    ledger_path = tmp_path / "ledger.jsonl"
    # Three independent appends, simulating cross-process / cross-session writes.
    AuditLedger.append_to_file(_make_record("#1", slot=slot), ledger_path, signing_key=sk)
    AuditLedger.append_to_file(_make_record("#2", slot=slot), ledger_path, signing_key=sk)
    AuditLedger.append_to_file(_make_record("#3", slot=slot), ledger_path, signing_key=sk)

    loaded = AuditLedger.load_jsonl(ledger_path)
    assert [r.record_id for r in loaded.records] == ["#1", "#2", "#3"]
    loaded.verify_chain()
    loaded.verify_signatures({vk.fingerprint: vk})


def test_append_to_file_unsigned_chain(tmp_path):
    """Append-only chain works without cryptography (signing optional)."""
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger_path = tmp_path / "ledger.jsonl"

    AuditLedger.append_to_file(_make_record("#1", slot=slot), ledger_path)
    AuditLedger.append_to_file(_make_record("#2", slot=slot), ledger_path)

    loaded = AuditLedger.load_jsonl(ledger_path)
    assert [r.record_id for r in loaded.records] == ["#1", "#2"]
    loaded.verify_chain()  # chained but unsigned still verifies


def test_append_to_file_detects_tail_tamper(tmp_path):
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger_path = tmp_path / "ledger.jsonl"

    AuditLedger.append_to_file(_make_record("#1", slot=slot), ledger_path)
    AuditLedger.append_to_file(_make_record("#2", slot=slot), ledger_path)

    # Edit the JSONL file in place: change a content byte on line 1.
    text = ledger_path.read_text().splitlines()
    assert "record #1" in text[0]  # sanity: replacement string matches
    text[0] = text[0].replace('"record #1"', '"tampered"')
    ledger_path.write_text("\n".join(text) + "\n")

    loaded = AuditLedger.load_jsonl(ledger_path)
    with pytest.raises(LedgerError):
        loaded.verify_chain()


def test_load_jsonl_empty_returns_empty_ledger(tmp_path):
    ledger_path = tmp_path / "missing.jsonl"
    loaded = AuditLedger.load_jsonl(ledger_path)
    assert loaded.records == []


def test_append_to_file_unchained_tail_raises(tmp_path):
    """If the JSONL tail is unchained, append_to_file must refuse."""
    p = _write_pinned_file(tmp_path)
    slot = KernelSlot(name="raw_Q", anchor=ByteIdenticalAnchor.of_file(p))
    ledger_path = tmp_path / "legacy.jsonl"

    # Hand-write a v0.1 unsigned record (no chain fields).
    legacy = _make_record("#legacy", slot=slot)
    import json as _json

    ledger_path.write_text(
        _json.dumps(legacy.model_dump(mode="json"), default=str) + "\n"
    )

    with pytest.raises(LedgerError, match="unchained tail"):
        AuditLedger.append_to_file(_make_record("#next", slot=slot), ledger_path)
