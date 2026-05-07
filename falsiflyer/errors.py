"""FalsiFlyer exception hierarchy. All public errors descend from FalsiFlyerError."""

from __future__ import annotations


class FalsiFlyerError(Exception):
    """Root of the FalsiFlyer exception tree."""


class HashMismatchError(FalsiFlyerError):
    """Frozen dataset SHA-256 does not match the recorded value.

    Raised when ``load_frozen_dataset`` recomputes the canonical hash and it
    differs from the ``sha256_data_payload`` field stored at freeze time.
    Implies the dataset has been mutated post-freeze; the run MUST abort.
    """


class DecisionRuleViolation(FalsiFlyerError):
    """Pre-registered decision rule is malformed or self-inconsistent."""


class AnchorDriftError(FalsiFlyerError):
    """A byte-identical anchor reference no longer matches its recorded hash.

    Raised by ``ByteIdenticalAnchor.verify()`` when the file at the pinned
    path has been modified since the anchor was committed.  Used to detect
    silent drift in the test estimator across falsification runs.
    """
