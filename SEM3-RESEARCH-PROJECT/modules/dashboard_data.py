"""
modules/dashboard_data.py
=========================
Module 9 — Dashboard Data Loading and Validation (no Streamlit dependency).

Purpose
-------
Provide pure-Python functions for loading, validating, and processing the
evaluation artifacts produced by train_models.py.  Keeping these functions
separate from the Streamlit page makes them fully unit-testable.

What this module does
---------------------
- Load and validate models/metrics.json with schema checks.
- Load per-sample evaluation arrays (models/eval_arrays.npz) for ROC/PR curves
  when available.
- Compute derived metrics (specificity, additional differences) from saved data.
- Produce a safe CSV export string from recorded metrics.
- Provide a clear status report when artifacts are missing or incomplete.

What this module does NOT do
-----------------------------
- Never make network requests.
- Never retrain models or overwrite training artifacts.
- Never fabricate or infer metrics from single predictions.
- Never expose stack traces or sensitive file paths to callers.
- Never silently replace missing values with zero.

Evaluation arrays (optional, saved by train_models.py)
-------------------------------------------------------
When train_models.py is run with --save-eval-arrays, it saves:
    models/eval_arrays.npz
containing arrays:
    m1_y_true   — int8 array of held-out true labels for Model 1
    m1_y_prob   — float32 array of phishing probabilities from Model 1
    m2_y_true   — int8 array of held-out true labels for Model 2 (if trained)
    m2_y_prob   — float32 array of phishing probabilities from Model 2

If this file does not exist, ROC and PR curves are reported as unavailable.

Public API
----------
    from modules.dashboard_data import (
        load_dashboard_data,
        load_eval_arrays,
        compute_derived_metrics,
        metric_differences,
        build_export_csv,
        DashboardData,
        EvalArrays,
    )
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths  (relative to project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR         = _PROJECT_ROOT / "models"
METRICS_JSON_PATH  = MODELS_DIR / "metrics.json"
EVAL_ARRAYS_PATH   = MODELS_DIR / "eval_arrays.npz"

# ---------------------------------------------------------------------------
# Schema — the keys we expect in a trained model section
# ---------------------------------------------------------------------------

_REQUIRED_METRIC_KEYS = {
    "accuracy", "precision", "recall", "f1_score",
    "fpr", "fnr", "roc_auc",
    "confusion_matrix", "test_size", "feature_count",
}

_CM_KEYS = {"tn", "fp", "fn", "tp"}

# Metric metadata: (display_label, key, higher_is_better, description)
METRIC_DEFINITIONS: list[tuple[str, str, bool, str]] = [
    ("Accuracy",     "accuracy",  True,
     "Overall fraction of correctly classified URLs. "
     "Misleading when classes are imbalanced."),
    ("Precision",    "precision", True,
     "Of all URLs predicted as phishing, what fraction are genuinely phishing? "
     "High precision = fewer false alarms."),
    ("Recall",       "recall",    True,
     "Of all genuinely phishing URLs, what fraction did the model catch? "
     "High recall = fewer missed phishing sites (false negatives)."),
    ("F1-Score",     "f1_score",  True,
     "Harmonic mean of precision and recall. "
     "Balances the trade-off between false alarms and missed threats."),
    ("Specificity",  "specificity", True,
     "Of all legitimate URLs, what fraction were correctly identified as legitimate? "
     "Equal to 1 − FPR."),
    ("FPR",          "fpr",       False,
     "False Positive Rate: fraction of legitimate URLs incorrectly flagged as phishing. "
     "Lower is better."),
    ("FNR",          "fnr",       False,
     "False Negative Rate: fraction of phishing URLs missed by the model. "
     "Lower is better. Equal to 1 − Recall."),
    ("ROC-AUC",      "roc_auc",   True,
     "Area under the ROC curve. Measures discrimination across all thresholds. "
     "0.5 = random, 1.0 = perfect. Requires probability scores."),
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricSet:
    """
    Validated metrics for one model.

    All values are float or None (never substituted with zero).
    missing_keys lists which expected keys were absent in the raw data.
    """
    accuracy:   float | None = None
    precision:  float | None = None
    recall:     float | None = None
    f1_score:   float | None = None
    fpr:        float | None = None
    fnr:        float | None = None
    roc_auc:    float | None = None
    specificity: float | None = None    # derived: 1 - fpr
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    tp: int | None = None
    test_size:   int | None = None
    feature_count: int | None = None
    train_size:  int | None = None
    training_duration_seconds: float | None = None
    test_phishing_count: int | None = None
    test_legit_count:    int | None = None
    missing_keys: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class DatasetMeta:
    """Metadata about the dataset and training run."""
    dataset:          str = ""
    total_rows:       int | None = None
    phishing_count:   int | None = None
    legitimate_count: int | None = None
    label_column:     str = ""
    test_fraction:    float | None = None
    random_seed:      int | None = None
    trained_at_utc:   str = ""
    phishing_label:   int | None = None   # integer encoding for phishing class
    legitimate_label: int | None = None


@dataclass
class MetricDiff:
    """
    Absolute difference for one metric (Model 2 − Model 1).

    direction: "higher_better" | "lower_better"
    """
    key:            str = ""
    label:          str = ""
    m1_value:       float | None = None
    m2_value:       float | None = None
    diff:           float | None = None   # m2 - m1; None if either is missing
    higher_better:  bool = True

    @property
    def m2_is_better(self) -> bool | None:
        """True when Model 2 has a better value; None when indeterminate."""
        if self.diff is None:
            return None
        if abs(self.diff) < 1e-9:
            return None          # effectively equal
        return (self.diff > 0) == self.higher_better


@dataclass
class DashboardData:
    """
    Fully validated dashboard payload, ready for rendering.

    valid: True when at least model1 metrics were successfully loaded.
    error: Human-readable description when valid is False.
    """
    valid:      bool = False
    error:      str  = ""
    metadata:   DatasetMeta = field(default_factory=DatasetMeta)
    model1:     MetricSet   = field(default_factory=MetricSet)
    model2:     MetricSet   = field(default_factory=MetricSet)
    diffs:      list[MetricDiff] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    # Whether both models were evaluated on the same held-out split
    same_test_split: bool = False
    same_test_split_note: str = ""


@dataclass
class EvalArrays:
    """
    Per-sample evaluation arrays for ROC and PR curves.

    available: False when eval_arrays.npz does not exist or is unreadable.
    """
    available:  bool = False
    error:      str  = ""
    m1_y_true:  Any  = None   # numpy int8 array or None
    m1_y_prob:  Any  = None   # numpy float32 array or None
    m2_y_true:  Any  = None   # numpy int8 array or None (None if M2 not trained)
    m2_y_prob:  Any  = None   # numpy float32 array or None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, key: str, warnings: list[str]) -> float | None:
    """Convert val to float in [0, 1]; add a warning and return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        warnings.append(f"Metric '{key}' cannot be converted to float: {val!r}")
        return None
    if not math.isfinite(f):
        warnings.append(f"Metric '{key}' is not finite: {f}")
        return None
    # Allow small floating-point overshoot for metrics that should be in [0,1]
    if key not in ("sec_tls_days_remain", "training_duration_seconds"):
        if f < -0.001 or f > 1.001:
            warnings.append(
                f"Metric '{key}' has value {f:.6f} which is outside [0, 1]. "
                "This may indicate a data issue."
            )
    return f


def _safe_int(val: Any, key: str, warnings: list[str]) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        warnings.append(f"Field '{key}' cannot be converted to int: {val!r}")
        return None


def _parse_model_section(
    raw: dict,
    model_label: str,
    warnings: list[str],
) -> MetricSet:
    """
    Parse and validate a model section from metrics.json.

    Returns a MetricSet.  Never raises.
    """
    ms = MetricSet()

    if not raw:
        ms.skipped = True
        ms.skip_reason = "No data in metrics file."
        return ms

    if raw.get("skipped"):
        ms.skipped = True
        ms.skip_reason = str(raw.get("reason", "Model was not trained."))
        return ms

    w_local: list[str] = []

    ms.accuracy  = _safe_float(raw.get("accuracy"),  "accuracy",  w_local)
    ms.precision = _safe_float(raw.get("precision"), "precision", w_local)
    ms.recall    = _safe_float(raw.get("recall"),    "recall",    w_local)
    ms.f1_score  = _safe_float(raw.get("f1_score"),  "f1_score",  w_local)
    ms.fpr       = _safe_float(raw.get("fpr"),       "fpr",       w_local)
    ms.fnr       = _safe_float(raw.get("fnr"),       "fnr",       w_local)
    ms.roc_auc   = _safe_float(raw.get("roc_auc"),   "roc_auc",   w_local)

    # Derived: specificity = 1 - FPR (only when FPR is available)
    if ms.fpr is not None:
        ms.specificity = round(1.0 - ms.fpr, 6)

    # Confusion matrix
    cm = raw.get("confusion_matrix")
    if isinstance(cm, dict):
        ms.tn = _safe_int(cm.get("tn"), "cm.tn", w_local)
        ms.fp = _safe_int(cm.get("fp"), "cm.fp", w_local)
        ms.fn = _safe_int(cm.get("fn"), "cm.fn", w_local)
        ms.tp = _safe_int(cm.get("tp"), "cm.tp", w_local)
        # Cross-validate confusion matrix against scalar metrics where possible
        total = sum(x or 0 for x in [ms.tn, ms.fp, ms.fn, ms.tp])
        if ms.test_size is None:
            ms.test_size = _safe_int(raw.get("test_size"), "test_size", w_local)
        if ms.test_size is not None and total != ms.test_size:
            warnings.append(
                f"{model_label}: Confusion matrix total ({total}) does not match "
                f"test_size ({ms.test_size}). Metrics may be inconsistent."
            )
    else:
        w_local.append(f"{model_label}: confusion_matrix is absent or invalid.")

    ms.test_size   = _safe_int(raw.get("test_size"),   "test_size",   w_local)
    ms.feature_count = _safe_int(raw.get("feature_count"), "feature_count", w_local)
    ms.train_size  = _safe_int(raw.get("train_size"),  "train_size",  w_local)
    ms.test_phishing_count = _safe_int(raw.get("test_phishing_count"), "test_phishing_count", w_local)
    ms.test_legit_count    = _safe_int(raw.get("test_legit_count"),    "test_legit_count",    w_local)

    dur = raw.get("training_duration_seconds")
    if dur is not None:
        try:
            ms.training_duration_seconds = float(dur)
        except (TypeError, ValueError):
            pass

    # Track which required keys were missing
    ms.missing_keys = [
        k for k in _REQUIRED_METRIC_KEYS
        if raw.get(k) is None
    ]

    for w in w_local:
        warnings.append(w)

    return ms


def _parse_metadata(raw: dict, warnings: list[str]) -> DatasetMeta:
    dm = DatasetMeta()
    if not raw:
        warnings.append("'metadata' section is absent from metrics.json.")
        return dm
    dm.dataset          = str(raw.get("dataset", ""))
    dm.total_rows       = _safe_int(raw.get("total_rows"),   "total_rows",  warnings)
    dm.phishing_count   = _safe_int(raw.get("phishing_count"), "phishing_count", warnings)
    dm.legitimate_count = _safe_int(raw.get("legitimate_count"), "legitimate_count", warnings)
    dm.label_column     = str(raw.get("label_column", ""))
    dm.trained_at_utc   = str(raw.get("trained_at_utc", ""))
    tf = raw.get("test_fraction")
    if tf is not None:
        try:
            dm.test_fraction = float(tf)
        except (TypeError, ValueError):
            warnings.append(f"'test_fraction' is not a valid float: {tf!r}")
    rs = raw.get("random_seed")
    if rs is not None:
        try:
            dm.random_seed = int(rs)
        except (TypeError, ValueError):
            pass
    enc = raw.get("label_encoding", {})
    if isinstance(enc, dict):
        try:
            dm.phishing_label   = int(enc.get("phishing", 1))
            dm.legitimate_label = int(enc.get("legitimate", 0))
        except (TypeError, ValueError):
            pass
    return dm


def _compute_diffs(m1: MetricSet, m2: MetricSet) -> list[MetricDiff]:
    """Compute Model 2 − Model 1 differences for all defined metrics."""
    diffs: list[MetricDiff] = []
    for label, key, higher_better, _ in METRIC_DEFINITIONS:
        v1 = getattr(m1, key, None)
        v2 = getattr(m2, key, None)
        diff = None
        if v1 is not None and v2 is not None:
            try:
                diff = round(float(v2) - float(v1), 8)
            except (TypeError, ValueError):
                pass
        diffs.append(MetricDiff(
            key=key, label=label,
            m1_value=v1, m2_value=v2,
            diff=diff,
            higher_better=higher_better,
        ))
    return diffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dashboard_data(
    metrics_path: Path | None = None,
) -> DashboardData:
    """
    Load and validate the evaluation metrics from metrics.json.

    Parameters
    ----------
    metrics_path : Path | None
        Override the default path for testing. Uses METRICS_JSON_PATH when None.

    Returns
    -------
    DashboardData
        Always returns a DashboardData — never raises an unhandled exception.
        Check .valid to see whether loading succeeded.
    """
    path = metrics_path or METRICS_JSON_PATH
    dd = DashboardData()

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        dd.error = (
            f"Evaluation artifact not found: {path.name}. "
            "Run train_models.py to generate evaluation metrics."
        )
        return dd
    except json.JSONDecodeError as exc:
        dd.error = f"metrics.json is not valid JSON: {exc}"
        return dd
    except OSError as exc:
        dd.error = f"Cannot read {path.name}: {exc}"
        return dd

    if not isinstance(raw, dict):
        dd.error = "metrics.json does not contain a JSON object at the top level."
        return dd

    warnings: list[str] = []

    dd.metadata = _parse_metadata(raw.get("metadata", {}), warnings)
    dd.model1   = _parse_model_section(raw.get("model1", {}), "Model 1", warnings)
    dd.model2   = _parse_model_section(raw.get("model2", {}), "Model 2", warnings)

    if dd.model1.skipped:
        dd.error = (
            "Model 1 metrics are not available (skipped or empty). "
            "Re-run train_models.py."
        )
        return dd

    # Compute derived metric diffs
    dd.diffs = _compute_diffs(dd.model1, dd.model2)

    # Determine whether both models were evaluated on the same held-out split
    if not dd.model2.skipped:
        if (
            dd.model1.test_size is not None
            and dd.model2.test_size is not None
            and dd.model1.test_size == dd.model2.test_size
            and dd.metadata.random_seed is not None
        ):
            dd.same_test_split = True
            dd.same_test_split_note = (
                f"Both models were evaluated on the same {dd.model1.test_size}-sample "
                f"held-out test set (random_seed={dd.metadata.random_seed}, "
                f"test_fraction={dd.metadata.test_fraction})."
            )
        else:
            dd.same_test_split = False
            dd.same_test_split_note = (
                "The models may not have been evaluated on identical held-out rows "
                "(test set sizes differ or random seed is missing). "
                "See train_models.py for split details."
            )

    dd.warnings = warnings
    dd.valid = True
    return dd


def load_eval_arrays(
    arrays_path: Path | None = None,
) -> EvalArrays:
    """
    Load per-sample evaluation arrays for ROC and PR curves.

    Returns EvalArrays with available=False if the file does not exist
    or cannot be loaded.  Never raises.
    """
    path = arrays_path or EVAL_ARRAYS_PATH
    ea = EvalArrays()

    if not path.exists():
        ea.error = (
            f"Per-sample evaluation arrays not found ({path.name}). "
            "Re-run train_models.py --save-eval-arrays to generate them."
        )
        return ea

    try:
        import numpy as np
        data = np.load(str(path), allow_pickle=False)
    except Exception as exc:
        ea.error = f"Cannot load {path.name}: {exc}"
        return ea

    try:
        ea.m1_y_true = data["m1_y_true"]
        ea.m1_y_prob = data["m1_y_prob"]
        # Model 2 arrays are optional (may not exist if M2 was skipped)
        if "m2_y_true" in data and "m2_y_prob" in data:
            ea.m2_y_true = data["m2_y_true"]
            ea.m2_y_prob = data["m2_y_prob"]
        ea.available = True
    except KeyError as exc:
        ea.error = f"eval_arrays.npz is missing required key: {exc}"

    return ea


def compute_derived_metrics(ms: MetricSet) -> MetricSet:
    """
    Fill derived fields in a MetricSet that may not be directly stored.

    Currently derives:
    - specificity = 1 − fpr  (when fpr is available)

    Returns the same MetricSet (mutated in-place) for convenience.
    """
    if ms.fpr is not None and ms.specificity is None:
        ms.specificity = round(1.0 - ms.fpr, 6)
    return ms


def metric_differences(m1: MetricSet, m2: MetricSet) -> list[MetricDiff]:
    """
    Return a list of MetricDiff objects (M2 − M1) for all defined metrics.

    Missing values are represented as diff=None.
    """
    return _compute_diffs(m1, m2)


def build_export_csv(dd: DashboardData) -> str:
    """
    Build a CSV string from DashboardData for download.

    Returns an empty string when no valid data is available.
    Never raises.

    The CSV contains:
    - One metadata row
    - One row per model per metric
    - Absolute differences (M2 − M1)

    Does NOT include per-sample predictions, raw URLs, user data, or
    file paths beyond the dataset filename.
    """
    if not dd.valid:
        return ""

    buf = io.StringIO()
    writer = csv.writer(buf)

    # ── Metadata section ─────────────────────────────────────────────────────
    writer.writerow(["# WebTrust AI — Evaluation Metrics Export"])
    writer.writerow(["# Generated from models/metrics.json"])
    writer.writerow(["# Positive class (phishing) label",
                     dd.metadata.phishing_label])
    writer.writerow(["# Dataset", dd.metadata.dataset or "—"])
    writer.writerow(["# Total rows", dd.metadata.total_rows or "—"])
    writer.writerow(["# Phishing samples", dd.metadata.phishing_count or "—"])
    writer.writerow(["# Legitimate samples", dd.metadata.legitimate_count or "—"])
    writer.writerow(["# Test fraction", dd.metadata.test_fraction or "—"])
    writer.writerow(["# Random seed", dd.metadata.random_seed or "—"])
    writer.writerow(["# Trained at (UTC)", dd.metadata.trained_at_utc or "—"])
    writer.writerow([])

    # ── Metric table ─────────────────────────────────────────────────────────
    writer.writerow([
        "metric", "display_label", "higher_is_better",
        "model1_value", "model2_value",
        "diff_m2_minus_m1", "m2_better",
        "description",
    ])

    diff_map = {d.key: d for d in dd.diffs}

    for _, key, higher_better, description in METRIC_DEFINITIONS:
        d = diff_map.get(key)
        if d is None:
            v1, v2, diff, better = None, None, None, None
        else:
            v1, v2, diff, better = d.m1_value, d.m2_value, d.diff, d.m2_is_better

        writer.writerow([
            key,
            d.label if d else key,
            "yes" if higher_better else "no",
            f"{v1:.6f}" if v1 is not None else "N/A",
            f"{v2:.6f}" if v2 is not None else ("skipped" if dd.model2.skipped else "N/A"),
            f"{diff:+.6f}" if diff is not None else "N/A",
            ("M2" if better is True else "M1" if better is False else "equal") if better is not None else "N/A",
            description,
        ])

    # ── Confusion matrix section ──────────────────────────────────────────────
    writer.writerow([])
    writer.writerow(["# Confusion matrices (positive class = phishing)"])
    writer.writerow(["model", "tn", "fp", "fn", "tp", "test_size",
                     "test_phishing_count", "test_legit_count",
                     "feature_count", "train_size"])

    for label, ms in [("Model 1 (Baseline)", dd.model1),
                      ("Model 2 (Extended)", dd.model2)]:
        if ms.skipped:
            writer.writerow([label, "skipped"])
        else:
            writer.writerow([
                label,
                ms.tn if ms.tn is not None else "N/A",
                ms.fp if ms.fp is not None else "N/A",
                ms.fn if ms.fn is not None else "N/A",
                ms.tp if ms.tp is not None else "N/A",
                ms.test_size or "N/A",
                ms.test_phishing_count or "N/A",
                ms.test_legit_count    or "N/A",
                ms.feature_count or "N/A",
                ms.train_size    or "N/A",
            ])

    return buf.getvalue()


def describe_validity_issues(dd: DashboardData) -> list[str]:
    """
    Return a list of plain-language notes about experimental validity.

    These are structural observations about the evaluation setup, not
    verdicts about whether the models are correct.
    """
    notes: list[str] = []

    if not dd.valid:
        notes.append("Evaluation artifacts are missing or unreadable.")
        return notes

    meta = dd.metadata

    # Dataset size
    if meta.total_rows is not None and meta.total_rows < 1000:
        notes.append(
            f"The dataset contains only {meta.total_rows} rows. "
            "Results on small datasets may not generalise to real phishing traffic."
        )

    # Synthetic data flag
    dataset_lower = meta.dataset.lower()
    if "synthetic" in dataset_lower or "test" in dataset_lower or "demo" in dataset_lower:
        notes.append(
            f"The dataset appears to be synthetic or a demo set "
            f"('{meta.dataset}'). Metrics from a synthetic dataset "
            "do NOT reflect real-world phishing-detection performance "
            "and must not be used as research findings."
        )

    # Perfect metrics
    m1 = dd.model1
    if (
        m1.accuracy == 1.0 and m1.precision == 1.0
        and m1.recall == 1.0 and m1.f1_score == 1.0
    ):
        notes.append(
            "Model 1 reports perfect scores (accuracy = recall = F1 = 1.0). "
            "This almost always indicates trivially separable synthetic data "
            "or label leakage. These results have no research validity."
        )

    m2 = dd.model2
    if not m2.skipped and (
        m2.accuracy == 1.0 and m2.precision == 1.0
        and m2.recall == 1.0 and m2.f1_score == 1.0
    ):
        notes.append(
            "Model 2 also reports perfect scores. "
            "This confirms the dataset is not suitable for real research."
        )

    # Same split
    if not dd.model2.skipped:
        if not dd.same_test_split:
            notes.append(
                "The two models may not have been evaluated on identical held-out rows. "
                "Direct metric comparisons between the models require the same test set."
            )
        else:
            notes.append(dd.same_test_split_note)

    # Test set size cross-check
    if m1.test_size is not None and meta.total_rows is not None:
        expected_test = round(meta.total_rows * (meta.test_fraction or 0.2))
        if abs(m1.test_size - expected_test) > 2:
            notes.append(
                f"Model 1 test set size ({m1.test_size}) differs from the expected "
                f"value ({expected_test}) given test_fraction={meta.test_fraction}. "
                "This may indicate a row-count discrepancy in the saved artifacts."
            )

    # Missing metrics
    if m1.missing_keys:
        notes.append(
            f"Model 1 is missing recorded values for: {', '.join(m1.missing_keys)}."
        )
    if not m2.skipped and m2.missing_keys:
        notes.append(
            f"Model 2 is missing recorded values for: {', '.join(m2.missing_keys)}."
        )

    # ROC/PR curves
    notes.append(
        "ROC and precision-recall curves require per-sample evaluation arrays "
        "(models/eval_arrays.npz). "
        "Re-run train_models.py --save-eval-arrays to generate them."
    )

    return notes
