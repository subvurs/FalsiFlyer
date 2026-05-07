"""Domain-agnostic data types for falsiflyer.

A ``Dataset`` is a list of ``Cell``s, each holding a list of ``Subject``s.
Both ``Cell.params`` and ``Subject.payload`` are free-form dicts; the
hash function (in ``falsiflyer.prereg``) walks user-declared fields out of
those dicts deterministically.

Estimator outputs are scored per-subject and aggregated per-cell.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Subject(BaseModel):
    """One unit of data the estimators are evaluated on.

    Attributes
    ----------
    subject_id:
        Stable, unique identifier across the cohort.
    payload:
        Arbitrary observation data. The hash function reads named fields
        out of this dict; see ``HashFieldSpec``.
    ground_truth:
        Optional ground-truth scalar used to compute relative error. May be
        ``None`` for prediction tasks where the truth is unknown at freeze
        time and stamped in later.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: Optional[float] = None


class Cell(BaseModel):
    """A grid cell of subjects sharing structural parameters.

    Attributes
    ----------
    cell_id:
        Human-readable identifier (typically ``f"k={k_pop}_n={n}_CV={cv}"``).
    params:
        Free-form structural parameters of the cell (k_pop, n, CV, ...).
    subjects:
        Members of the cell.
    is_stress_cell:
        True if this cell is in the pre-registered stress region. Decision
        rules only count stress cells in PASS/FAIL accounting.
    """

    cell_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    subjects: List[Subject] = Field(default_factory=list)
    is_stress_cell: bool = False


class Dataset(BaseModel):
    """A frozen, hash-committed pre-registration artifact.

    The dataset's ``sha256_data_payload`` is computed by ``canonical_hash``
    over the fields named in ``hash_fields``.  Any post-freeze mutation
    that touches those fields will invalidate the hash on reload.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str
    generated_at: str
    regime: str = ""
    description: str = ""
    constants: Dict[str, Any] = Field(default_factory=dict)
    grid: Dict[str, Any] = Field(default_factory=dict)
    hash_fields: List[str] = Field(default_factory=list)
    cell_param_keys: Optional[List[str]] = None
    decision_rule: Dict[str, Any] = Field(default_factory=dict)
    cells: List[Cell] = Field(default_factory=list)
    sha256_data_payload: str = ""

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_subjects(self) -> int:
        return sum(len(c.subjects) for c in self.cells)

    @property
    def n_stress_cells(self) -> int:
        return sum(1 for c in self.cells if c.is_stress_cell)


# ---------------------------------------------------------------------------
# Scoring containers (populated by the Harness)
# ---------------------------------------------------------------------------


class EstimatorScore(BaseModel):
    """Per-cell aggregate score for a single estimator."""

    geom_mean: float
    median_rel_err: float
    mean_rel_err: float
    n_finite: int
    n_total: int


class CellScore(BaseModel):
    """All estimator scores plus per-subject point estimates for one cell."""

    cell_id: str
    is_stress_cell: bool
    params: Dict[str, Any] = Field(default_factory=dict)
    n_subjects: int
    estimators: Dict[str, EstimatorScore] = Field(default_factory=dict)
    per_subject: List[Dict[str, Any]] = Field(default_factory=list)


class Verdict(BaseModel):
    """The structured PASS/FAIL output of a decision-rule-bound run."""

    test_estimator: str
    baselines: List[str]
    tightening_threshold: float
    pass_fraction: float
    n_stress_cells: int
    required_n_passing: int
    n_pass_all_baselines: int
    n_pass_each_baseline: Dict[str, int]
    verdict_pass: bool
    per_cell: List[Dict[str, Any]] = Field(default_factory=list)
