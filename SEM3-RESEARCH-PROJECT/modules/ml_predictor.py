"""
modules/ml_predictor.py
=======================
Module 7 — Machine Learning Phishing Detection

Purpose
-------
Load trained phishing detection models (Model 1 and Model 2) and run inference
on a single URL using the existing feature extractors.

This module is responsible for:
  - Defining the canonical feature schemas for Model 1 and Model 2.
  - Loading saved model artifacts (joblib pipelines) safely.
  - Building the correct feature vector from already-extracted results.
  - Returning structured predictions with confidence scores.
  - Deriving security features from SecurityAnalysisResult for Model 2.

What this module does NOT do
----------------------------
- Never retrain models at application startup.
- Never make network requests (feature extraction is handled by callers).
- Never invent or impute a security feature when the security analysis failed.
- Never classify a URL as phishing without a valid model artifact.
- Never display fabricated probabilities.

Feature schemas
---------------
Model 1 uses the 25-feature numeric vector produced by
modules.feature_extractor.numeric_feature_names().

Model 2 uses all 25 Model 1 features PLUS 14 security features derived from
the SecurityAnalysisResult produced by modules.security_analyzer.analyze_security().

Public API
----------
    from modules.ml_predictor import load_models, predict_model1, predict_model2
    from modules.ml_predictor import PredictionResult, ModelLoadResult

Security features for Model 2
------------------------------
These 14 binary/integer features are derived from SecurityAnalysisResult:

  sec_has_csp          — Content-Security-Policy header present (bool→int)
  sec_has_hsts         — Strict-Transport-Security present AND applicable (bool→int)
  sec_has_xcto         — X-Content-Type-Options: nosniff present (bool→int)
  sec_has_xfo          — X-Frame-Options present (bool→int)
  sec_has_referrer     — Referrer-Policy present (bool→int)
  sec_has_permissions  — Permissions-Policy present (bool→int)
  sec_cookie_any       — At least one cookie observed (bool→int)
  sec_cookie_secure    — ≥1 cookie has Secure flag (bool→int)
  sec_cookie_httponly  — ≥1 cookie has HttpOnly flag (bool→int)
  sec_cookie_samesite  — ≥1 cookie has explicit SameSite attribute (bool→int)
  sec_tls_applicable   — URL uses HTTPS (tls.applicable) (bool→int)
  sec_tls_verified     — TLS certificate verified successfully (bool→int)
  sec_tls_days_remain  — Days until cert expires; -999 when not applicable/unknown
  sec_finding_count    — Total number of consolidated security findings (int)

These features are only computable when security analysis succeeded.  When
security analysis has not been run or failed, call predict_model2() with
security_result=None.  The function will fail gracefully with an appropriate
error in the PredictionResult.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = _PROJECT_ROOT / "models"

MODEL1_PATH       = MODELS_DIR / "model1_baseline.joblib"
MODEL2_PATH       = MODELS_DIR / "model2_extended.joblib"
MODEL1_FEAT_PATH  = MODELS_DIR / "model1_features.joblib"
MODEL2_FEAT_PATH  = MODELS_DIR / "model2_features.joblib"
METRICS_JSON_PATH = MODELS_DIR / "metrics.json"
METRICS_CSV_PATH  = MODELS_DIR / "metrics.csv"

# ---------------------------------------------------------------------------
# Feature schema constants
# ---------------------------------------------------------------------------

# Model 1: 25 traditional URL features (must match feature_extractor output exactly)
MODEL1_FEATURES: list[str] = [
    "url_length",
    "url_hostname_length",
    "url_path_length",
    "url_query_length",
    "url_fragment_length",
    "url_num_query_params",
    "char_count_dot",
    "char_count_hyphen",
    "char_count_underscore",
    "char_count_slash",
    "char_count_percent",
    "char_count_at",
    "host_num_labels",
    "host_subdomain_count",
    "domain_registered_length",
    "kw_num_matched",
    "flag_uses_https",
    "flag_is_ip_address",
    "flag_has_non_default_port",
    "flag_has_at_symbol",
    "flag_has_query",
    "flag_has_fragment",
    "flag_has_punycode",
    "flag_has_keyword",
    "flag_is_url_shortener",
]

# Model 2: Model 1 features + 14 security features
MODEL2_SECURITY_FEATURES: list[str] = [
    "sec_has_csp",
    "sec_has_hsts",
    "sec_has_xcto",
    "sec_has_xfo",
    "sec_has_referrer",
    "sec_has_permissions",
    "sec_cookie_any",
    "sec_cookie_secure",
    "sec_cookie_httponly",
    "sec_cookie_samesite",
    "sec_tls_applicable",
    "sec_tls_verified",
    "sec_tls_days_remain",
    "sec_finding_count",
]

MODEL2_FEATURES: list[str] = MODEL1_FEATURES + MODEL2_SECURITY_FEATURES

# Sentinel value for unavailable numeric security features
_SEC_DAYS_UNAVAILABLE: int = -999

# Class labels
LABEL_LEGITIMATE = 0
LABEL_PHISHING   = 1
CLASS_NAMES      = {LABEL_LEGITIMATE: "Legitimate", LABEL_PHISHING: "Potentially Phishing"}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """
    Result of a single model prediction.

    Fields
    ------
    model_name : str
        Human-readable model identifier.
    success : bool
        True when a valid prediction was produced.
    error : str
        Non-empty when success is False.
    predicted_class : int | None
        0 = Legitimate, 1 = Potentially Phishing, None on failure.
    predicted_label : str
        "Legitimate", "Potentially Phishing", or "Unavailable".
    phishing_probability : float | None
        Probability for the phishing class (0.0–1.0), or None when the model
        does not support probability output.
    legitimate_probability : float | None
        Probability for the legitimate class (0.0–1.0), or None.
    feature_count : int
        Number of features used for this prediction.
    features_used : list[str]
        Ordered list of feature names used.
    """
    model_name: str = ""
    success: bool = False
    error: str = ""
    predicted_class: int | None = None
    predicted_label: str = "Unavailable"
    phishing_probability: float | None = None
    legitimate_probability: float | None = None
    feature_count: int = 0
    features_used: list[str] = field(default_factory=list)


@dataclass
class ModelLoadResult:
    """
    Status of loading a single model artifact.

    Fields
    ------
    model_name : str
    loaded : bool
    error : str
    feature_count : int
    features : list[str]
    """
    model_name: str = ""
    loaded: bool = False
    error: str = ""
    feature_count: int = 0
    features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy model cache — loaded once per process
# ---------------------------------------------------------------------------

_model1_pipeline: Any | None = None   # sklearn Pipeline
_model2_pipeline: Any | None = None

_model1_load_error: str = ""
_model2_load_error: str = ""

_models_loaded: bool = False


def load_models(force_reload: bool = False) -> tuple[ModelLoadResult, ModelLoadResult]:
    """
    Load Model 1 and Model 2 pipelines from disk.

    Caches both pipelines in module-level variables so subsequent calls do
    not re-read from disk unless force_reload=True.

    Returns
    -------
    (model1_load_result, model2_load_result)
        Each is a ModelLoadResult describing whether loading succeeded.
    """
    global _model1_pipeline, _model2_pipeline
    global _model1_load_error, _model2_load_error
    global _models_loaded

    if _models_loaded and not force_reload:
        r1 = ModelLoadResult(
            model_name="Model 1 (Baseline)",
            loaded=_model1_pipeline is not None,
            error=_model1_load_error,
            feature_count=len(MODEL1_FEATURES),
            features=MODEL1_FEATURES,
        )
        r2 = ModelLoadResult(
            model_name="Model 2 (Extended)",
            loaded=_model2_pipeline is not None,
            error=_model2_load_error,
            feature_count=len(MODEL2_FEATURES),
            features=MODEL2_FEATURES,
        )
        return r1, r2

    import joblib  # imported here so the module can be imported without joblib at import time

    def _load(path: Path, name: str) -> tuple[Any | None, str]:
        if not path.exists():
            return None, f"Model artifact not found: {path}. Run train_models.py first."
        try:
            pipeline = joblib.load(str(path))
            return pipeline, ""
        except Exception as exc:
            return None, f"Failed to load {name} from {path}: {exc}"

    _model1_pipeline, _model1_load_error = _load(MODEL1_PATH, "Model 1")
    _model2_pipeline, _model2_load_error = _load(MODEL2_PATH, "Model 2")
    _models_loaded = True

    r1 = ModelLoadResult(
        model_name="Model 1 (Baseline)",
        loaded=_model1_pipeline is not None,
        error=_model1_load_error,
        feature_count=len(MODEL1_FEATURES),
        features=MODEL1_FEATURES,
    )
    r2 = ModelLoadResult(
        model_name="Model 2 (Extended)",
        loaded=_model2_pipeline is not None,
        error=_model2_load_error,
        feature_count=len(MODEL2_FEATURES),
        features=MODEL2_FEATURES,
    )
    return r1, r2


# ---------------------------------------------------------------------------
# Feature vector builders
# ---------------------------------------------------------------------------


def build_model1_vector(feature_result) -> tuple[list[float], list[str], str]:
    """
    Build the Model 1 feature vector from a FeatureResult.

    Parameters
    ----------
    feature_result : FeatureResult
        Result from modules.feature_extractor.extract_features().

    Returns
    -------
    (values, names, error)
        values : list of float, in MODEL1_FEATURES order.
        names  : MODEL1_FEATURES column names.
        error  : non-empty string when a required feature is missing.
    """
    numeric = feature_result.numeric
    values: list[float] = []
    missing: list[str] = []

    for fname in MODEL1_FEATURES:
        val = numeric.get(fname)
        if val is None:
            missing.append(fname)
            values.append(0.0)
        else:
            # Booleans map to 0/1
            values.append(float(int(val)) if isinstance(val, bool) else float(val))

    if missing:
        return values, MODEL1_FEATURES, (
            f"Missing features in FeatureResult.numeric: {missing}. "
            "Ensure feature_extractor is up to date."
        )
    return values, MODEL1_FEATURES, ""


def build_model2_vector(
    feature_result,
    security_result,
) -> tuple[list[float], list[str], str]:
    """
    Build the Model 2 feature vector from a FeatureResult and a SecurityAnalysisResult.

    Parameters
    ----------
    feature_result : FeatureResult
        Result from modules.feature_extractor.extract_features().
    security_result : SecurityAnalysisResult | None
        Result from modules.security_analyzer.analyze_security(), or None
        when security analysis has not been run or failed.

    Returns
    -------
    (values, names, error)
        error is non-empty when security_result is None/failed or traditional
        features are missing.

    Notes
    -----
    When security_result is None or security_result.success is False, the
    security feature values cannot be computed from observed data.  The
    function returns an error rather than imputing misleading values.
    """
    # First get Model 1 part
    m1_values, _, m1_err = build_model1_vector(feature_result)

    # Build security feature part
    if security_result is None:
        return (
            m1_values + [0.0] * len(MODEL2_SECURITY_FEATURES),
            MODEL2_FEATURES,
            "Security analysis has not been run for this URL. "
            "Model 2 requires a successful security analysis. "
            "Run 'Analyze Website Security' in Tab 3 first.",
        )

    if not security_result.success:
        return (
            m1_values + [0.0] * len(MODEL2_SECURITY_FEATURES),
            MODEL2_FEATURES,
            f"Security analysis failed: {security_result.error}. "
            "Model 2 security features cannot be computed without a successful analysis.",
        )

    sec_values = _extract_security_features(security_result)

    error = m1_err  # propagate any traditional feature warning
    return m1_values + sec_values, MODEL2_FEATURES, error


def _extract_security_features(security_result) -> list[float]:
    """
    Derive the 14 binary/integer security features from a SecurityAnalysisResult.

    Features are ordered to match MODEL2_SECURITY_FEATURES exactly.
    """
    # Index header checks by name
    hc_by_name: dict[str, Any] = {
        hc.name: hc for hc in security_result.header_checks
    }

    def _header_present(name: str) -> int:
        hc = hc_by_name.get(name)
        if hc is None:
            return 0
        return 1 if (hc.applicable and hc.present) else 0

    def _header_clean(name: str) -> int:
        """Present AND no improvement suggestion (well-configured)."""
        hc = hc_by_name.get(name)
        if hc is None:
            return 0
        return 1 if (hc.applicable and hc.present and not hc.suggestion) else 0

    sec_has_csp         = _header_present("Content-Security-Policy")
    sec_has_hsts        = _header_present("Strict-Transport-Security")
    sec_has_xcto        = _header_clean("X-Content-Type-Options")   # only if nosniff
    sec_has_xfo         = _header_present("X-Frame-Options")
    sec_has_referrer    = _header_present("Referrer-Policy")
    sec_has_permissions = _header_present("Permissions-Policy")

    # Cookie features
    cookies = security_result.cookie_checks
    sec_cookie_any      = 1 if security_result.cookies_present else 0
    sec_cookie_secure   = 1 if any(c.has_secure   for c in cookies) else 0
    sec_cookie_httponly = 1 if any(c.has_httponly  for c in cookies) else 0
    sec_cookie_samesite = 1 if any(c.samesite      for c in cookies) else 0

    # TLS features
    tls = security_result.tls
    sec_tls_applicable = 1 if tls.applicable else 0
    sec_tls_verified   = 1 if (tls.applicable and tls.verified) else 0

    if tls.applicable and tls.days_remaining is not None:
        sec_tls_days_remain = float(tls.days_remaining)
    else:
        sec_tls_days_remain = float(_SEC_DAYS_UNAVAILABLE)

    sec_finding_count = float(len(security_result.findings))

    return [
        float(sec_has_csp),
        float(sec_has_hsts),
        float(sec_has_xcto),
        float(sec_has_xfo),
        float(sec_has_referrer),
        float(sec_has_permissions),
        float(sec_cookie_any),
        float(sec_cookie_secure),
        float(sec_cookie_httponly),
        float(sec_cookie_samesite),
        float(sec_tls_applicable),
        float(sec_tls_verified),
        sec_tls_days_remain,
        sec_finding_count,
    ]


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


def _run_prediction(
    pipeline: Any,
    feature_vector: list[float],
    feature_names: list[str],
    model_name: str,
) -> PredictionResult:
    """
    Run a loaded sklearn Pipeline on a single-row feature vector.

    Handles both predict() and predict_proba() safely.
    """
    try:
        import pandas as pd
        row = pd.DataFrame([feature_vector], columns=feature_names)
        predicted = pipeline.predict(row)
        pred_class = int(predicted[0])
    except Exception as exc:
        return PredictionResult(
            model_name=model_name,
            success=False,
            error=f"Prediction failed: {exc}",
            feature_count=len(feature_names),
            features_used=feature_names,
        )

    phish_prob: float | None = None
    legit_prob: float | None = None

    try:
        proba = pipeline.predict_proba(row)
        # proba shape: (1, n_classes); classes are ordered by pipeline.classes_
        classes = list(pipeline.classes_)
        if LABEL_PHISHING in classes and LABEL_LEGITIMATE in classes:
            phish_prob = float(proba[0][classes.index(LABEL_PHISHING)])
            legit_prob  = float(proba[0][classes.index(LABEL_LEGITIMATE)])
        elif len(classes) == 2:
            # Fallback: assume [0,1] ordering
            legit_prob  = float(proba[0][0])
            phish_prob  = float(proba[0][1])
    except Exception:
        pass  # Model may not support predict_proba — not an error

    return PredictionResult(
        model_name=model_name,
        success=True,
        error="",
        predicted_class=pred_class,
        predicted_label=CLASS_NAMES.get(pred_class, f"Class {pred_class}"),
        phishing_probability=phish_prob,
        legitimate_probability=legit_prob,
        feature_count=len(feature_names),
        features_used=feature_names,
    )


def predict_model1(feature_result) -> PredictionResult:
    """
    Run Model 1 inference on a FeatureResult.

    Loads the model if not already cached.  Returns a PredictionResult with
    success=False and a descriptive error when the model is unavailable or
    when the feature vector cannot be built.

    Parameters
    ----------
    feature_result : FeatureResult
        Output of modules.feature_extractor.extract_features().
    """
    load_models()

    if _model1_pipeline is None:
        return PredictionResult(
            model_name="Model 1 (Baseline)",
            success=False,
            error=_model1_load_error,
            features_used=MODEL1_FEATURES,
            feature_count=len(MODEL1_FEATURES),
        )

    vector, names, build_err = build_model1_vector(feature_result)
    if build_err and not vector:
        return PredictionResult(
            model_name="Model 1 (Baseline)",
            success=False,
            error=build_err,
            features_used=MODEL1_FEATURES,
            feature_count=len(MODEL1_FEATURES),
        )

    result = _run_prediction(_model1_pipeline, vector, names, "Model 1 (Baseline)")
    if build_err and result.success:
        # Feature build had warnings but prediction still ran
        result.error = f"Warning: {build_err}"
    return result


def predict_model2(feature_result, security_result) -> PredictionResult:
    """
    Run Model 2 inference on a FeatureResult + SecurityAnalysisResult.

    Loads the model if not already cached.  When security_result is None or
    failed, returns a PredictionResult explaining that security analysis is required.

    Parameters
    ----------
    feature_result : FeatureResult
        Output of modules.feature_extractor.extract_features().
    security_result : SecurityAnalysisResult | None
        Output of modules.security_analyzer.analyze_security(), or None.
    """
    load_models()

    if _model2_pipeline is None:
        return PredictionResult(
            model_name="Model 2 (Extended)",
            success=False,
            error=_model2_load_error,
            features_used=MODEL2_FEATURES,
            feature_count=len(MODEL2_FEATURES),
        )

    vector, names, build_err = build_model2_vector(feature_result, security_result)
    if build_err and security_result is None:
        return PredictionResult(
            model_name="Model 2 (Extended)",
            success=False,
            error=build_err,
            features_used=MODEL2_FEATURES,
            feature_count=len(MODEL2_FEATURES),
        )
    if build_err and (security_result is not None and not security_result.success):
        return PredictionResult(
            model_name="Model 2 (Extended)",
            success=False,
            error=build_err,
            features_used=MODEL2_FEATURES,
            feature_count=len(MODEL2_FEATURES),
        )

    result = _run_prediction(_model2_pipeline, vector, names, "Model 2 (Extended)")
    if build_err and result.success:
        result.error = f"Warning: {build_err}"
    return result


# ---------------------------------------------------------------------------
# Metrics loader
# ---------------------------------------------------------------------------


def load_metrics() -> tuple[dict, str]:
    """
    Load the saved evaluation metrics from models/metrics.json.

    Returns
    -------
    (metrics_dict, error_message)
        metrics_dict is empty when metrics are unavailable.
        error_message is non-empty when loading failed.
    """
    if not METRICS_JSON_PATH.exists():
        return {}, (
            f"Metrics file not found: {METRICS_JSON_PATH}. "
            "Run train_models.py to train the models and generate metrics."
        )
    try:
        import json
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, ""
    except Exception as exc:
        return {}, f"Failed to load metrics: {exc}"
