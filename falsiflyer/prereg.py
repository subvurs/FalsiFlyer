"""Pre-registration primitives: hash-commit, freeze, load, decision rule.

The hash function is the byte-identical anchor of the audit trail.  Two
independent runs MUST produce the same SHA-256 if they hashed the same
canonical fields, and ``load_frozen_dataset`` MUST raise ``HashMismatchError``
on any post-freeze mutation that touches the hashed fields.

Design choice: the hash is **not** computed over the JSON serialization of
the Dataset (which would couple it to pydantic's encoder, key ordering,
float formatting, etc.).  Instead, we emit a deterministic byte stream
from caller-declared fields read out of each subject's payload.  This is
the same scheme used in the #7j/#7k generators — see
``q_kernel_tdm_dataset_gen.py:hash_data_payload``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

from falsiflyer.errors import DecisionRuleViolation, HashMismatchError
from falsiflyer.types import Cell, Dataset, Subject


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


class DecisionRule(BaseModel):
    """Pre-registered binary decision rule.

    The rule's serialization is committed inside the frozen dataset, then
    re-read verbatim by the harness.  ``tightening_threshold`` and
    ``pass_fraction`` MUST NOT be changed between freeze and run.

    Attributes
    ----------
    test_estimator:
        Name of the estimator under falsification (the one whose claim is
        being tested). e.g. ``"raw_Q"``.
    baselines:
        Estimators that ``test_estimator`` must beat. The harness checks
        ``test_estimator`` against EACH baseline independently, AND-ed.
    tightening_threshold:
        Minimum relative tightening (0.15 = "15% better median rel-err").
    pass_fraction:
        Fraction of stress cells on which ``test_estimator`` must beat
        ALL baselines simultaneously. e.g. ``2/3`` ⇒ "8 of 12".
    stress_predicate:
        Human-readable description of which cells are stress cells. The
        harness honors ``Cell.is_stress_cell``; the predicate text is
        stored for audit-readability.
    metric:
        Aggregation metric. ``"median_rel_err"`` (default) or
        ``"mean_rel_err"``.
    """

    test_estimator: str
    baselines: List[str] = Field(default_factory=list, min_length=1)
    tightening_threshold: float = 0.15
    pass_fraction: float = 2.0 / 3.0
    stress_predicate: str = ""
    metric: str = "median_rel_err"

    @model_validator(mode="after")
    def _check_ranges(self) -> "DecisionRule":
        if not (0.0 < self.tightening_threshold < 1.0):
            raise DecisionRuleViolation(
                f"tightening_threshold must be in (0, 1); "
                f"got {self.tightening_threshold}"
            )
        if not (0.0 < self.pass_fraction <= 1.0):
            raise DecisionRuleViolation(
                f"pass_fraction must be in (0, 1]; got {self.pass_fraction}"
            )
        if self.test_estimator in self.baselines:
            raise DecisionRuleViolation(
                f"test_estimator {self.test_estimator!r} also listed as baseline"
            )
        if self.metric not in ("median_rel_err", "mean_rel_err"):
            raise DecisionRuleViolation(
                f"metric must be median_rel_err or mean_rel_err; got {self.metric!r}"
            )
        return self

    def serialize(self) -> dict:
        return self.model_dump()

    @classmethod
    def deserialize(cls, payload: dict) -> "DecisionRule":
        return cls(**payload)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


# Type alias: a hash-field spec is just an iterable of field names that
# will be looked up inside Cell.params and Subject.payload (in that order).
HashFieldSpec = Iterable[str]


def _hash_value(h: "hashlib._Hash", label: str, value: Any) -> None:
    """Append ``label=...`` for one value, choosing a canonical encoding.

    * arrays / lists of floats → ``np.asarray(...).tobytes()`` (locks the
      bit pattern; identical to the #7j generator).
    * lists of ints            → ``np.asarray(..., dtype=np.int64).tobytes()``.
    * str/int/float/bool/None  → JSON-encoded then UTF-8.
    """
    h.update(f"|{label}=".encode("utf-8"))
    if value is None:
        h.update(b"null")
        return
    if isinstance(value, (str, int, bool)):
        h.update(json.dumps(value, sort_keys=True).encode("utf-8"))
        return
    if isinstance(value, float):
        h.update(np.asarray([value], dtype=np.float64).tobytes())
        return
    if isinstance(value, np.ndarray):
        h.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            h.update(b"[]")
            return
        if all(isinstance(v, (bool, int)) and not isinstance(v, bool) for v in value):
            h.update(np.asarray(value, dtype=np.int64).tobytes())
            return
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            h.update(np.asarray(value, dtype=np.float64).tobytes())
            return
        # mixed/nested — fall through to JSON
        h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        return
    if isinstance(value, dict):
        h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        return
    # Last resort: stringified
    h.update(repr(value).encode("utf-8"))


def canonical_hash(
    cells: List[Cell],
    hash_fields: HashFieldSpec,
    cell_param_keys: Optional[Iterable[str]] = None,
) -> str:
    """Deterministic SHA-256 over caller-declared fields.

    Walk order: cell loop preserves ``cells`` ordering; subject loop
    preserves ``cell.subjects`` ordering.  For each cell, hash the named
    cell-param keys (default = all keys, sorted); for each subject, hash
    ``subject_id`` + ``ground_truth`` + each named ``hash_fields`` key
    looked up in ``subject.payload``.

    Identical-byte output across runs is required for the audit trail.
    """
    h = hashlib.sha256()
    fields = list(hash_fields)
    for cell in cells:
        keys = (
            sorted(cell.params.keys())
            if cell_param_keys is None
            else list(cell_param_keys)
        )
        h.update(f"cell={cell.cell_id}".encode("utf-8"))
        for k in keys:
            _hash_value(h, f"cp.{k}", cell.params.get(k))
        h.update(f"|stress={int(cell.is_stress_cell)}".encode("utf-8"))
        for subj in cell.subjects:
            h.update(b"||subj=")
            h.update(subj.subject_id.encode("utf-8"))
            _hash_value(h, "truth", subj.ground_truth)
            for f in fields:
                _hash_value(h, f, subj.payload.get(f))
            h.update(b";")
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DatasetCommit — small carrier that re-emits a Dataset with a fresh hash.
# ---------------------------------------------------------------------------


class DatasetCommit(BaseModel):
    """Bundle of (cells, schema_version, decision_rule, hash_fields).

    Use ``DatasetCommit.freeze()`` to compute the hash and produce a
    ``Dataset`` ready for JSON serialization.
    """

    schema_version: str
    cells: List[Cell]
    decision_rule: DecisionRule
    hash_fields: List[str]
    cell_param_keys: Optional[List[str]] = None
    regime: str = ""
    description: str = ""
    constants: dict = Field(default_factory=dict)
    grid: dict = Field(default_factory=dict)

    def freeze(self) -> Dataset:
        ds = Dataset(
            schema_version=self.schema_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            regime=self.regime,
            description=self.description,
            constants=self.constants,
            grid=self.grid,
            hash_fields=list(self.hash_fields),
            cell_param_keys=(
                list(self.cell_param_keys)
                if self.cell_param_keys is not None
                else None
            ),
            decision_rule=self.decision_rule.serialize(),
            cells=list(self.cells),
            sha256_data_payload="",
        )
        digest = canonical_hash(
            ds.cells, ds.hash_fields, ds.cell_param_keys
        )
        ds.sha256_data_payload = digest
        return ds


# ---------------------------------------------------------------------------
# Public freeze / load helpers
# ---------------------------------------------------------------------------


def freeze_dataset(
    schema_version: str,
    cells: List[Cell],
    decision_rule: DecisionRule,
    hash_fields: HashFieldSpec,
    out_path: Optional[Union[str, Path]] = None,
    cell_param_keys: Optional[Iterable[str]] = None,
    regime: str = "",
    description: str = "",
    constants: Optional[dict] = None,
    grid: Optional[dict] = None,
) -> Tuple[Dataset, Optional[Path]]:
    """Hash, optionally write to disk, return (Dataset, path|None)."""
    commit = DatasetCommit(
        schema_version=schema_version,
        cells=list(cells),
        decision_rule=decision_rule,
        hash_fields=list(hash_fields),
        cell_param_keys=list(cell_param_keys) if cell_param_keys is not None else None,
        regime=regime,
        description=description,
        constants=constants or {},
        grid=grid or {},
    )
    ds = commit.freeze()
    written: Optional[Path] = None
    if out_path is not None:
        written = Path(out_path)
        written.parent.mkdir(parents=True, exist_ok=True)
        with written.open("w") as f:
            json.dump(ds.model_dump(mode="json"), f, indent=2, default=str)
    return ds, written


def load_frozen_dataset(path: Union[str, Path]) -> Dataset:
    """Load a frozen dataset; re-verify its SHA-256.

    Raises
    ------
    HashMismatchError
        If the recomputed canonical hash does not match the stored value.
    """
    p = Path(path)
    with p.open() as f:
        payload = json.load(f)
    ds = Dataset.model_validate(payload)
    expected = ds.sha256_data_payload
    actual = canonical_hash(ds.cells, ds.hash_fields, ds.cell_param_keys)
    if actual != expected:
        raise HashMismatchError(
            f"Dataset SHA-256 mismatch:\n"
            f"  in-file:  {expected}\n"
            f"  computed: {actual}\n"
            f"  path:     {p}"
        )
    return ds
