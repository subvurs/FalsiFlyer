"""falsiflyer — pre-registration + decision-rule-bound benchmark primitives.

Five public modules:

* ``falsiflyer.prereg``   — hash-commit + frozen-dataset loader + decision rule
* ``falsiflyer.baselines``— proper-baseline (DGP-matched) noise templates
* ``falsiflyer.harness``  — decision-rule-bound benchmark runner
* ``falsiflyer.diagnostic``— disagreement-split diagnostic template
* ``falsiflyer.audit``    — kernel slot, byte-identical anchor, ledger
"""

from falsiflyer.errors import (
    FalsiFlyerError,
    HashMismatchError,
    DecisionRuleViolation,
    AnchorDriftError,
)
from falsiflyer.types import (
    Subject,
    Cell,
    Dataset,
    EstimatorScore,
    CellScore,
    Verdict,
)
from falsiflyer.prereg import (
    DatasetCommit,
    DecisionRule,
    HashFieldSpec,
    canonical_hash,
    freeze_dataset,
    load_frozen_dataset,
)
from falsiflyer.baselines import (
    NoiseModel,
    Gaussian,
    Proportional,
    Poisson,
    LogGaussian,
    Binomial,
    BaselineLibrary,
)
from falsiflyer.harness import (
    Estimator,
    Harness,
    HarnessResult,
    evaluate_decision_rule,
)
from falsiflyer.diagnostic import (
    DisagreementSplit,
    SplitReport,
    run_split_diagnostic,
)
from falsiflyer.audit import (
    KernelSlot,
    ByteIdenticalAnchor,
    FalsificationRecord,
    AuditLedger,
    LedgerError,
    SigningKey,
    VerifyKey,
    GENESIS_PREV_HASH,
)
from falsiflyer.report import render_markdown_report

__version__ = "0.2.1"

__all__ = [
    # errors
    "FalsiFlyerError",
    "HashMismatchError",
    "DecisionRuleViolation",
    "AnchorDriftError",
    # types
    "Subject",
    "Cell",
    "Dataset",
    "EstimatorScore",
    "CellScore",
    "Verdict",
    # prereg
    "DatasetCommit",
    "DecisionRule",
    "HashFieldSpec",
    "canonical_hash",
    "freeze_dataset",
    "load_frozen_dataset",
    # baselines
    "NoiseModel",
    "Gaussian",
    "Proportional",
    "Poisson",
    "LogGaussian",
    "Binomial",
    "BaselineLibrary",
    # harness
    "Estimator",
    "Harness",
    "HarnessResult",
    "evaluate_decision_rule",
    # diagnostic
    "DisagreementSplit",
    "SplitReport",
    "run_split_diagnostic",
    # audit
    "KernelSlot",
    "ByteIdenticalAnchor",
    "FalsificationRecord",
    "AuditLedger",
    "LedgerError",
    "SigningKey",
    "VerifyKey",
    "GENESIS_PREV_HASH",
    # report
    "render_markdown_report",
]
