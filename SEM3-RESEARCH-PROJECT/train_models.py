"""
train_models.py
===============
Module 7 — Training pipeline for Model 1 and Model 2.

Usage
-----
    python train_models.py --dataset data/dataset.csv

    # With custom label column and paths:
    python train_models.py \\
        --dataset data/dataset.csv \\
        --label-col label \\
        --phishing-value 1 \\
        --legit-value 0 \\
        --test-size 0.2 \\
        --random-seed 42

    # See all options:
    python train_models.py --help

Dataset format
--------------
The dataset must be a CSV file with at least:
  - One column containing the raw URL string.
  - One column containing the class label (phishing vs legitimate).

Optional columns for Model 2 (security features):
  - Security feature columns described in modules/ml_predictor.py.
  - If these columns are absent, Model 2 training is skipped.

Supported label formats (auto-detected or configurable):
  phishing = 1, legitimate = 0   (integer)
  phishing = "phishing", legitimate = "legitimate"   (string, case-insensitive)
  phishing = "1", legitimate = "0"   (string-encoded integer)

Outputs (saved to models/)
--------------------------
  model1_baseline.joblib   — sklearn Pipeline (scaler + Random Forest)
  model2_extended.joblib   — sklearn Pipeline (scaler + Random Forest)
  model1_features.joblib   — ordered list of Model 1 feature names
  model2_features.joblib   — ordered list of Model 2 feature names
  metrics.json             — evaluation metrics for both models
  metrics.csv              — same metrics in CSV form
  eval_arrays.npz          — per-sample labels and probabilities (optional,
                             generated only when --save-eval-arrays is used)

Label leakage prevention
------------------------
- Preprocessing (scaler, imputer) is fitted ONLY on training data.
- The same held-out test split is used to evaluate both models.
- Feature names are saved separately from the model pipeline for verification.
- No post-detection information (e.g., whether a domain has been blacklisted
  after the fact) should be included in the dataset features.

Security features for Model 2 (must be pre-computed in dataset)
----------------------------------------------------------------
Because fetching live security data for every training URL would be slow,
insecure, and inconsistent, the Model 2 security features (sec_has_csp,
sec_has_hsts, etc.) must be pre-computed and included as columns in the
dataset CSV.  See modules/ml_predictor.MODEL2_SECURITY_FEATURES for the
exact expected column names.

If the dataset does not contain these columns, Model 2 training is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_models")

# ---------------------------------------------------------------------------
# Imports — fail with a clear message if dependencies are missing
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        classification_report,
    )
except ImportError as exc:
    print(
        f"\nERROR: Required package missing: {exc}\n"
        "Install with:\n"
        "    .venv\\Scripts\\pip install scikit-learn pandas numpy joblib\n"
    )
    sys.exit(1)

from modules.url_validator import validate_url
from modules.feature_extractor import extract_features, numeric_feature_names
from modules.ml_predictor import (
    MODEL1_FEATURES,
    MODEL2_FEATURES,
    MODEL2_SECURITY_FEATURES,
    MODELS_DIR,
    MODEL1_PATH,
    MODEL2_PATH,
    MODEL1_FEAT_PATH,
    MODEL2_FEAT_PATH,
    METRICS_JSON_PATH,
    METRICS_CSV_PATH,
    LABEL_PHISHING,
    LABEL_LEGITIMATE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE   = 0.20          # 20 % held-out test set
N_ESTIMATORS = 200          # Random Forest trees
MIN_SAMPLES_PER_CLASS = 50  # abort training if a class has fewer samples

# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------


def _detect_label_column(df: pd.DataFrame, hint: str | None) -> tuple[str, str]:
    """
    Detect the label column and URL column from the DataFrame.

    Returns (label_col, url_col).  Raises ValueError when detection fails.
    """
    url_candidates  = ["url", "URL", "Url", "address", "Address", "link", "Link"]
    label_candidates = [
        "label", "Label", "LABEL",
        "status", "Status", "STATUS",
        "class", "Class", "CLASS",
        "type", "Type", "phishing",
    ]

    if hint:
        if hint not in df.columns:
            raise ValueError(
                f"Label column '{hint}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        label_col = hint
    else:
        label_col = next((c for c in label_candidates if c in df.columns), None)
        if label_col is None:
            raise ValueError(
                f"Cannot detect label column. Tried: {label_candidates}. "
                f"Dataset columns: {list(df.columns)}. "
                "Use --label-col to specify it."
            )

    url_col = next((c for c in url_candidates if c in df.columns), None)
    return label_col, url_col


def _normalise_labels(
    series: pd.Series,
    phishing_value: str | int | None,
    legit_value: str | int | None,
) -> pd.Series:
    """
    Convert the raw label column to integer labels {0: legitimate, 1: phishing}.

    Tries several common label formats automatically when phishing_value/legit_value
    are not supplied.
    """
    raw = series.astype(str).str.strip().str.lower()
    unique_vals = raw.unique().tolist()

    # Explicit mapping from CLI
    if phishing_value is not None and legit_value is not None:
        p = str(phishing_value).strip().lower()
        l = str(legit_value).strip().lower()
        def _map_explicit(v: str) -> int:
            if v == p:
                return LABEL_PHISHING
            if v == l:
                return LABEL_LEGITIMATE
            return -1
        mapped = raw.map(_map_explicit)
        bad = (mapped == -1).sum()
        if bad > 0:
            raise ValueError(
                f"{bad} rows have labels other than '{phishing_value}' or "
                f"'{legit_value}'. Check --phishing-value and --legit-value."
            )
        return mapped

    # Auto-detect common formats
    auto_maps = [
        {"1": LABEL_PHISHING, "0": LABEL_LEGITIMATE},
        {"-1": LABEL_PHISHING, "1": LABEL_LEGITIMATE},
        {"phishing": LABEL_PHISHING, "legitimate": LABEL_LEGITIMATE},
        {"phishing": LABEL_PHISHING, "benign": LABEL_LEGITIMATE},
        {"phishing": LABEL_PHISHING, "safe": LABEL_LEGITIMATE},
        {"malicious": LABEL_PHISHING, "benign": LABEL_LEGITIMATE},
        {"1": LABEL_PHISHING, "-1": LABEL_LEGITIMATE},
        {"yes": LABEL_PHISHING, "no": LABEL_LEGITIMATE},
    ]
    for mapping in auto_maps:
        if all(v in mapping for v in unique_vals):
            return raw.map(mapping).astype(int)

    # If 2 unique values — map largest/alphabetically last to phishing
    if len(unique_vals) == 2:
        vals_sorted = sorted(unique_vals)
        mapping = {vals_sorted[0]: LABEL_LEGITIMATE, vals_sorted[1]: LABEL_PHISHING}
        log.warning(
            "Label format not recognised. Mapping '%s'→phishing, '%s'→legitimate. "
            "Verify this is correct. Use --phishing-value and --legit-value to override.",
            vals_sorted[1], vals_sorted[0],
        )
        return raw.map(mapping).astype(int)

    raise ValueError(
        f"Cannot auto-detect label encoding. Unique values: {unique_vals}. "
        "Use --phishing-value and --legit-value."
    )


def _extract_row_features(
    url: str,
    row: pd.Series,
    feature_cols: list[str],
) -> dict[str, float] | None:
    """
    Extract MODEL1_FEATURES for a single row.

    First tries to read columns directly from the row (fast — assumes dataset
    already contains pre-computed features).  Falls back to live feature
    extraction when URL column is available but feature columns are absent.

    Returns None on failure.
    """
    # Check if traditional features exist as columns in the row
    existing = {f: row[f] for f in feature_cols if f in row.index}
    if len(existing) == len(feature_cols):
        # All features pre-computed in dataset
        return {f: float(existing[f]) for f in feature_cols}

    # Live extraction from URL string
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        vr = validate_url(url)
        if not vr.is_valid:
            return None
        fr = extract_features(vr)
        return {f: float(int(fr.numeric[f])) if isinstance(fr.numeric[f], bool)
                else float(fr.numeric[f])
                for f in feature_cols}
    except Exception:
        return None


def load_dataset(
    path: str,
    label_col: str | None,
    url_col_hint: str | None,
    phishing_value,
    legit_value,
) -> tuple[pd.DataFrame, str, str]:
    """
    Load and validate the dataset CSV.

    Returns (dataframe_with_features, label_col, url_col).
    Raises ValueError for incompatible datasets.
    """
    log.info("Loading dataset: %s", path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Cannot read dataset: {exc}") from exc

    log.info("Raw dataset shape: %s rows × %s columns", *df.shape)

    # Detect columns
    detected_label_col, detected_url_col = _detect_label_column(df, label_col)
    url_col = url_col_hint or detected_url_col
    log.info("Label column: '%s'", detected_label_col)
    log.info("URL column: '%s'", url_col or "(not detected — using pre-computed features)")

    # Normalise labels
    df["_label"] = _normalise_labels(df[detected_label_col], phishing_value, legit_value)

    # Drop rows with invalid labels
    before = len(df)
    df = df[df["_label"].isin([LABEL_PHISHING, LABEL_LEGITIMATE])].copy()
    after = len(df)
    if before != after:
        log.warning("Dropped %d rows with unrecognised labels.", before - after)

    # Remove duplicates
    if url_col and url_col in df.columns:
        dup_count = df.duplicated(subset=[url_col]).sum()
        if dup_count > 0:
            log.info("Removing %d duplicate URL rows.", dup_count)
            df = df.drop_duplicates(subset=[url_col])

    # Class balance check
    counts = df["_label"].value_counts()
    log.info("Class distribution: %s", counts.to_dict())
    for cls in [LABEL_LEGITIMATE, LABEL_PHISHING]:
        n = counts.get(cls, 0)
        if n < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Class {cls} has only {n} samples (minimum {MIN_SAMPLES_PER_CLASS}). "
                "Provide a larger or more balanced dataset."
            )

    return df, detected_label_col, url_col or ""


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


def _build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    url_col: str,
    source_label: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a feature matrix X and label vector y from the DataFrame.

    If feature columns exist in df, uses them directly.
    Otherwise, performs live URL feature extraction (slow for large datasets).
    """
    label_vec = df["_label"].values

    # Check if ALL feature columns are already present
    present = [c for c in feature_cols if c in df.columns]
    if len(present) == len(feature_cols):
        log.info("%s: Using pre-computed feature columns from dataset.", source_label)
        X = df[feature_cols].copy()
        # Coerce to numeric; replace non-numeric with NaN for imputation later
        X = X.apply(pd.to_numeric, errors="coerce")
        return X, label_vec

    log.info(
        "%s: Feature columns not found in dataset — extracting from URL column. "
        "This may take several minutes for large datasets.",
        source_label,
    )

    if not url_col or url_col not in df.columns:
        raise ValueError(
            f"Feature columns {feature_cols[:3]}... not in dataset and no URL column "
            "found. Cannot extract features."
        )

    rows: list[dict[str, float]] = []
    kept_indices: list[int] = []
    skipped = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        url = str(row.get(url_col, ""))
        extracted = _extract_row_features(url, row, feature_cols)
        if extracted is None:
            skipped += 1
            continue
        rows.append(extracted)
        kept_indices.append(i)

        if (i + 1) % 500 == 0:
            log.info("  Extracted %d / %d rows (skipped %d)", i + 1, len(df), skipped)

    if skipped > 0:
        log.warning("Skipped %d rows due to invalid URLs or extraction failures.", skipped)

    if len(rows) < MIN_SAMPLES_PER_CLASS * 2:
        raise ValueError(
            f"Only {len(rows)} rows had extractable features. "
            "Cannot train with so few rows."
        )

    X = pd.DataFrame(rows, columns=feature_cols)
    y = label_vec[kept_indices]
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _build_pipeline() -> Pipeline:
    """Build a sklearn Pipeline: imputer → scaler → Random Forest."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def _evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_label: str,
) -> tuple[dict, np.ndarray | None]:
    """
    Evaluate a fitted pipeline on the held-out test set.

    Returns (metrics_dict, phishing_proba_array).
    phishing_proba_array is None when probability output is unavailable.
    """
    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    roc_auc: float | None = None
    phishing_proba: np.ndarray | None = None
    try:
        if len(np.unique(y_test)) == 2:
            proba = pipeline.predict_proba(X_test)
            clf_classes = list(pipeline.classes_)
            if LABEL_PHISHING in clf_classes:
                phish_col = clf_classes.index(LABEL_PHISHING)
                phishing_proba = proba[:, phish_col].astype(np.float32)
                roc_auc = float(roc_auc_score(y_test, phishing_proba))
    except Exception as exc:
        log.warning("%s: ROC-AUC could not be computed: %s", model_label, exc)

    log.info("%s Results:", model_label)
    log.info("  Accuracy:  %.4f", acc)
    log.info("  Precision: %.4f", prec)
    log.info("  Recall:    %.4f", rec)
    log.info("  F1-score:  %.4f", f1)
    log.info("  FPR:       %.4f", fpr)
    log.info("  FNR:       %.4f", fnr)
    if roc_auc is not None:
        log.info("  ROC-AUC:   %.4f", roc_auc)
    log.info("  Confusion matrix (TN=%d FP=%d FN=%d TP=%d)", tn, fp, fn, tp)

    report = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
    )

    metrics_dict = {
        "accuracy":    round(float(acc), 6),
        "precision":   round(float(prec), 6),
        "recall":      round(float(rec), 6),
        "f1_score":    round(float(f1), 6),
        "fpr":         round(float(fpr), 6),
        "fnr":         round(float(fnr), 6),
        "roc_auc":     round(float(roc_auc), 6) if roc_auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
        "test_size":   int(len(y_test)),
        "test_phishing_count":   int((y_test == LABEL_PHISHING).sum()),
        "test_legit_count":      int((y_test == LABEL_LEGITIMATE).sum()),
    }
    return metrics_dict, phishing_proba


def _train_one(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    model_label: str,
    save_path: Path,
    feat_path: Path,
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """
    Train one model and return (metrics_dict, y_test, phishing_proba).

    phishing_proba is None when probability output is unavailable.
    """
    log.info("Training %s (%d features, %d train rows)…",
             model_label, len(feature_names), len(y_train))

    t0 = time.time()
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    log.info("Training done in %.1fs. Evaluating on held-out test set…", elapsed)

    metrics, phishing_proba = _evaluate(pipeline, X_test, y_test, model_label)
    metrics["training_duration_seconds"] = round(elapsed, 2)
    metrics["feature_count"] = len(feature_names)
    metrics["feature_names"] = feature_names
    metrics["train_size"]    = int(len(y_train))
    metrics["train_phishing_count"] = int((y_train == LABEL_PHISHING).sum())
    metrics["train_legit_count"]    = int((y_train == LABEL_LEGITIMATE).sum())

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(save_path))
    joblib.dump(feature_names, str(feat_path))
    log.info("Saved %s → %s", model_label, save_path)

    return metrics, y_test, phishing_proba


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model 1 and Model 2 for WebTrust AI phishing detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the labelled CSV dataset.",
    )
    parser.add_argument(
        "--label-col", default=None,
        help="Name of the label column (auto-detected if omitted).",
    )
    parser.add_argument(
        "--url-col", default=None,
        help="Name of the URL column (auto-detected if omitted).",
    )
    parser.add_argument(
        "--phishing-value", default=None,
        help="Value in the label column representing phishing (e.g. 1, 'phishing').",
    )
    parser.add_argument(
        "--legit-value", default=None,
        help="Value in the label column representing legitimate (e.g. 0, 'legitimate').",
    )
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE,
        help=f"Fraction of data held out for testing (default {TEST_SIZE}).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--skip-model2", action="store_true",
        help="Train Model 1 only (skip Model 2 even if security features exist).",
    )
    parser.add_argument(
        "--save-eval-arrays", action="store_true",
        help=(
            "Save per-sample evaluation arrays (models/eval_arrays.npz) "
            "containing held-out true labels and predicted probabilities. "
            "These arrays are required for ROC and precision-recall curves "
            "on the Module 9 research dashboard."
        ),
    )
    args = parser.parse_args()

    # ── Load and validate dataset ──────────────────────────────────────────
    try:
        df, label_col, url_col = load_dataset(
            args.dataset,
            args.label_col,
            args.url_col,
            args.phishing_value,
            args.legit_value,
        )
    except ValueError as exc:
        log.error("Dataset error: %s", exc)
        sys.exit(1)

    total_rows   = len(df)
    n_phishing   = int((df["_label"] == LABEL_PHISHING).sum())
    n_legitimate = int((df["_label"] == LABEL_LEGITIMATE).sum())
    log.info("Dataset: %d total rows (%d phishing, %d legitimate)",
             total_rows, n_phishing, n_legitimate)

    # ── Build Model 1 feature matrix ──────────────────────────────────────
    log.info("Building Model 1 feature matrix…")
    try:
        X1, y1 = _build_feature_matrix(df, MODEL1_FEATURES, url_col, "Model 1")
    except ValueError as exc:
        log.error("Model 1 feature error: %s", exc)
        sys.exit(1)

    # Stratified train/test split — identical split used for both models
    X1_train, X1_test, y1_train, y1_test, idx_train, idx_test = (
        *train_test_split(
            X1, y1, np.arange(len(X1)),
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=y1,
        ),
    )

    # ── Train Model 1 ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("MODEL 1 — Baseline (Traditional URL Features)")
    log.info("=" * 60)
    m1_metrics, m1_y_test, m1_y_prob = _train_one(
        X1_train, y1_train,
        X1_test,  y1_test,
        MODEL1_FEATURES,
        "Model 1",
        MODEL1_PATH,
        MODEL1_FEAT_PATH,
    )

    # ── Model 2 — check whether security features are available ───────────
    m2_metrics: dict = {}
    m2_y_test:  np.ndarray | None = None
    m2_y_prob:  np.ndarray | None = None
    sec_cols_available = all(c in df.columns for c in MODEL2_SECURITY_FEATURES)

    if args.skip_model2:
        log.info("Skipping Model 2 (--skip-model2 flag set).")
    elif not sec_cols_available:
        missing_sec = [c for c in MODEL2_SECURITY_FEATURES if c not in df.columns]
        log.warning(
            "Model 2 SKIPPED — security feature columns are absent from the dataset.\n"
            "  Missing columns: %s\n"
            "  To train Model 2, add these columns to your dataset CSV by pre-computing\n"
            "  security features using modules.security_analyzer for each URL.\n"
            "  See modules/ml_predictor.py MODEL2_SECURITY_FEATURES for the full list.",
            missing_sec,
        )
        m2_metrics["skipped"] = True
        m2_metrics["reason"] = (
            f"Security feature columns missing from dataset: {missing_sec}"
        )
    else:
        log.info("=" * 60)
        log.info("MODEL 2 — Extended (URL + Security Features)")
        log.info("=" * 60)
        try:
            X2, y2 = _build_feature_matrix(df, MODEL2_FEATURES, url_col, "Model 2")
        except ValueError as exc:
            log.error("Model 2 feature error: %s", exc)
            m2_metrics["skipped"] = True
            m2_metrics["reason"] = str(exc)
        else:
            # Re-index using same train/test split as Model 1 where possible
            # (y2 may differ in row count if some URLs failed extraction)
            if len(y2) == len(y1):
                X2_train = X2.iloc[idx_train]
                X2_test  = X2.iloc[idx_test]
                y2_train = y2[idx_train]
                y2_test  = y2[idx_test]
            else:
                log.warning(
                    "Model 2 row count (%d) differs from Model 1 (%d). "
                    "Using independent split for Model 2.", len(y2), len(y1),
                )
                X2_train, X2_test, y2_train, y2_test = train_test_split(
                    X2, y2,
                    test_size=args.test_size,
                    random_state=args.random_seed,
                    stratify=y2,
                )

            m2_metrics, m2_y_test, m2_y_prob = _train_one(
                X2_train, y2_train,
                X2_test,  y2_test,
                MODEL2_FEATURES,
                "Model 2",
                MODEL2_PATH,
                MODEL2_FEAT_PATH,
            )

    # ── Optionally save per-sample evaluation arrays ──────────────────────
    if args.save_eval_arrays:
        arrays_path = MODELS_DIR / "eval_arrays.npz"
        save_kwargs: dict = {
            "m1_y_true": m1_y_test.astype(np.int8),
        }
        if m1_y_prob is not None:
            save_kwargs["m1_y_prob"] = m1_y_prob
        if m2_y_test is not None:
            save_kwargs["m2_y_true"] = m2_y_test.astype(np.int8)
        if m2_y_prob is not None:
            save_kwargs["m2_y_prob"] = m2_y_prob
        np.savez_compressed(str(arrays_path), **save_kwargs)
        log.info("Saved evaluation arrays → %s", arrays_path)

    # ── Save metrics ──────────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).isoformat()
    metrics_doc = {
        "metadata": {
            "dataset": str(Path(args.dataset).name),
            "total_rows": total_rows,
            "phishing_count": n_phishing,
            "legitimate_count": n_legitimate,
            "label_column": label_col,
            "label_encoding": {
                "phishing": LABEL_PHISHING,
                "legitimate": LABEL_LEGITIMATE,
            },
            "test_fraction": args.test_size,
            "random_seed": args.random_seed,
            "trained_at_utc": now_utc,
        },
        "model1": m1_metrics,
        "model2": m2_metrics,
        "eval_arrays_saved": args.save_eval_arrays,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_doc, f, indent=2, default=str)
    log.info("Saved metrics → %s", METRICS_JSON_PATH)

    # CSV summary
    csv_rows = []
    for model_key, mlabel in [("model1", "Model 1 (Baseline)"),
                               ("model2", "Model 2 (Extended)")]:
        m = metrics_doc[model_key]
        if not m or m.get("skipped"):
            csv_rows.append({
                "model": mlabel,
                "status": "skipped",
                "reason": m.get("reason", "Not trained"),
            })
        else:
            cm = m.get("confusion_matrix", {})
            csv_rows.append({
                "model":                mlabel,
                "status":               "trained",
                "accuracy":             m.get("accuracy"),
                "precision":            m.get("precision"),
                "recall":               m.get("recall"),
                "f1_score":             m.get("f1_score"),
                "fpr":                  m.get("fpr"),
                "fnr":                  m.get("fnr"),
                "roc_auc":              m.get("roc_auc"),
                "test_size":            m.get("test_size"),
                "feature_count":        m.get("feature_count"),
                "train_size":           m.get("train_size"),
                "training_duration_s":  m.get("training_duration_seconds"),
                "cm_tn":                cm.get("tn"),
                "cm_fp":                cm.get("fp"),
                "cm_fn":                cm.get("fn"),
                "cm_tp":                cm.get("tp"),
            })

    if csv_rows:
        keys = [k for k in csv_rows[0].keys()]
        with open(METRICS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        log.info("Saved metrics CSV → %s", METRICS_CSV_PATH)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("Model 1 F1-score: %.4f", m1_metrics.get("f1_score", 0))
    if args.save_eval_arrays:
        log.info("Evaluation arrays saved to models/eval_arrays.npz")
    else:
        log.info("Tip: re-run with --save-eval-arrays to enable ROC/PR curve display.")
    if m2_metrics and not m2_metrics.get("skipped"):
        log.info("Model 2 F1-score: %.4f", m2_metrics.get("f1_score", 0))
    else:
        log.info("Model 2: Not trained.")
    log.info("Artifacts saved in: %s", MODELS_DIR)
    log.info("Run the app with: py -m streamlit run app.py")


if __name__ == "__main__":
    main()
