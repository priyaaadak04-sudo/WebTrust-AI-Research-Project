"""
modules/risk_assessor.py
========================
Module 8 — Risk Assessment, Prediction Explanation, and Result Processing

Purpose
-------
Accept validated prediction results from Module 7 and the URL/feature data
collected by the earlier modules, then produce a structured, explainable risk
assessment for the analyst.

This module converts raw model outputs into:
  - A risk category (Low / Medium / High / Critical) with configurable thresholds.
  - A plain-language rationale grounded in actually observed evidence.
  - An evidence list based on features collected by existing extractors.
  - A disagreement flag when Model 1 and Model 2 predict different classes.
  - Actionable analyst recommendations matched to the risk category.
  - Explicit representation of missing or unavailable information.

What this module does NOT do
----------------------------
- Never fabricates features, probabilities, or evidence.
- Never silently substitutes default probabilities for missing model outputs.
- Never claims a URL is definitively safe or malicious.
- Never makes network requests (all evidence must already be collected).
- Never exposes stack traces or internal details in user-facing output.
- Never claims SHAP or feature-importance support — explanations are based on
  observed feature values and documented model input, not on post-hoc attribution.

Risk thresholds
---------------
Thresholds are PROVISIONAL OPERATIONAL DEFAULTS, not scientifically validated
cutoffs. They must be calibrated against held-out validation data and the
organization's specific security use case before deployment.

    Low:      phishing_probability < 0.20
    Medium:   0.20 <= phishing_probability < 0.50
    High:     0.50 <= phishing_probability < 0.80
    Critical: 0.80 <= phishing_probability <= 1.00

The PRIMARY probability for risk categorisation is chosen as follows:
  - If Model 2 produced a valid prediction, use its phishing probability.
  - If only Model 1 produced a valid prediction, use its probability.
  - If neither model produced a valid prediction, risk category is "Unknown".

This selection does not imply Model 2 is more accurate. It reflects that
Model 2 incorporates more observed evidence when available. If your validation
data shows otherwise, change PRIMARY_MODEL_PREFERENCE accordingly.

Public API
----------
    from modules.risk_assessor import assess_risk, RiskAssessment, RiskCategory

    assessment = assess_risk(
        url=result.normalized_url,
        feature_result=feature_result,
        prediction1=ml1_result,
        prediction2=ml2_result,          # None if not run
        security_result=security_result, # None if not run
        validation_result=vr,
    )
"""

from __future__ import annotations

import math
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modules.ml_predictor import PredictionResult, LABEL_PHISHING, LABEL_LEGITIMATE

# ---------------------------------------------------------------------------
# Risk category definition
# ---------------------------------------------------------------------------


class RiskCategory(str, Enum):
    """
    Operational risk category derived from the model's phishing probability.

    These categories are provisional defaults. They are NOT scientific verdicts.
    """
    LOW      = "Low"
    MEDIUM   = "Medium"
    HIGH     = "High"
    CRITICAL = "Critical"
    UNKNOWN  = "Unknown"     # No valid model prediction was available


# ---------------------------------------------------------------------------
# Configurable risk thresholds
# ---------------------------------------------------------------------------

# Boundaries are INCLUSIVE on the left and EXCLUSIVE on the right,
# except the Critical upper boundary which is inclusive (probability = 1.0).
#
# Adjust these thresholds after validation against held-out data.
# Document any change and the validation methodology.

THRESHOLD_LOW_MAX      = 0.20   # [0.00, 0.20) → Low
THRESHOLD_MEDIUM_MAX   = 0.50   # [0.20, 0.50) → Medium
THRESHOLD_HIGH_MAX     = 0.80   # [0.50, 0.80) → High
                                # [0.80, 1.00] → Critical

# Which model's probability is used as the primary risk score when both models
# succeeded. "model2" uses Model 2 when available; "model1" always uses Model 1.
# Change this only after validating that the chosen model is more reliable on
# your dataset and use case.
PRIMARY_MODEL_PREFERENCE: str = "model2"   # "model1" or "model2"

# Maximum absolute probability difference that triggers a disagreement warning.
# Two models are flagged as disagreeing when they predict DIFFERENT classes
# (regardless of probability magnitude).
# Additionally, a "notable difference" warning is raised when both predict the
# same class but their probabilities differ by more than this value.
NOTABLE_PROBABILITY_DIFF = 0.25


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """
    A single piece of observed or missing evidence.

    kind : "observed" | "missing" | "model_input"
        "observed"    — a feature or security attribute directly measured.
        "missing"     — information that would be relevant but was not collected.
        "model_input" — a model input feature whose value is notable.
    label : str
        Short human-readable label.
    detail : str
        Plain-language description of what was observed or why it is relevant.
    severity : "info" | "warn" | "flag"
        Rough indication of how notable this item is for phishing detection.
        "flag" does not mean the URL is phishing — it is a structural observation.
    """
    kind: str      # "observed" | "missing" | "model_input"
    label: str
    detail: str
    severity: str  # "info" | "warn" | "flag"


@dataclass
class RiskAssessment:
    """
    Structured risk assessment produced by assess_risk().

    Fields
    ------
    url : str
        The URL that was assessed.
    risk_category : RiskCategory
        Operational risk category derived from the primary model probability.
    primary_probability : float | None
        The phishing probability used to derive the risk category.
        None when no valid model prediction was available.
    primary_model : str
        Which model's probability was used ("Model 1", "Model 2", or "None").
    risk_rationale : str
        Plain-language explanation of why this risk category was assigned.
    model1_result : PredictionResult | None
        Raw Model 1 prediction (passed through from Module 7).
    model2_result : PredictionResult | None
        Raw Model 2 prediction, or None when not available.
    models_disagree : bool
        True when both models returned valid predictions but predicted
        different classes.
    disagreement_note : str
        Human-readable description of the disagreement, empty when there is none.
    notable_prob_diff : bool
        True when both models agree on class but their probabilities differ
        by more than NOTABLE_PROBABILITY_DIFF.
    evidence : list[EvidenceItem]
        Observed, missing, and model-input evidence items.
    recommendations : list[str]
        Actionable analyst recommendations matching the risk category.
    security_analysis_available : bool
        True when a successful security analysis result was provided.
    assessed_at : str
        ISO-8601 UTC timestamp of the assessment.
    threshold_config : dict[str, float]
        Snapshot of the thresholds used (for auditability).
    """
    url: str = ""
    risk_category: RiskCategory = RiskCategory.UNKNOWN
    primary_probability: float | None = None
    primary_model: str = "None"
    risk_rationale: str = ""
    model1_result: PredictionResult | None = None
    model2_result: PredictionResult | None = None
    models_disagree: bool = False
    disagreement_note: str = ""
    notable_prob_diff: bool = False
    evidence: list[EvidenceItem] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    security_analysis_available: bool = False
    assessed_at: str = ""
    threshold_config: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_probability(prob: Any, name: str) -> tuple[float | None, str]:
    """
    Validate that *prob* is a finite float in [0.0, 1.0].

    Returns (float, "") on success.
    Returns (None, error_message) when the value is invalid.
    """
    if prob is None:
        return None, f"{name} is None"
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return None, f"{name} cannot be converted to float: {prob!r}"
    if not math.isfinite(p):
        return None, f"{name} is not finite: {p}"
    if p < 0.0 or p > 1.0:
        return None, f"{name} is out of range [0, 1]: {p}"
    return p, ""


def _categorise(prob: float) -> RiskCategory:
    """
    Assign a RiskCategory from a validated phishing probability.

    Assumes prob is a finite float in [0.0, 1.0] (validated by caller).
    """
    if prob < THRESHOLD_LOW_MAX:
        return RiskCategory.LOW
    if prob < THRESHOLD_MEDIUM_MAX:
        return RiskCategory.MEDIUM
    if prob < THRESHOLD_HIGH_MAX:
        return RiskCategory.HIGH
    return RiskCategory.CRITICAL


def _select_primary(
    pred1: PredictionResult | None,
    pred2: PredictionResult | None,
) -> tuple[float | None, str, str]:
    """
    Choose the primary phishing probability and its source model.

    Returns (probability, model_label, selection_note).
    probability is None when neither model produced a valid probability.
    """
    p1, _ = _validate_probability(
        pred1.phishing_probability if pred1 and pred1.success else None,
        "Model 1 phishing_probability"
    )
    p2, _ = _validate_probability(
        pred2.phishing_probability if pred2 and pred2.success else None,
        "Model 2 phishing_probability"
    )

    if PRIMARY_MODEL_PREFERENCE == "model2":
        if p2 is not None:
            return p2, "Model 2", ""
        if p1 is not None:
            return p1, "Model 1", (
                "Model 2 result was unavailable; Model 1 probability is used instead."
            )
    else:
        if p1 is not None:
            return p1, "Model 1", ""
        if p2 is not None:
            return p2, "Model 2", (
                "Model 1 result was unavailable; Model 2 probability is used instead."
            )

    return None, "None", "Neither model produced a valid phishing probability."


def _check_disagreement(
    pred1: PredictionResult | None,
    pred2: PredictionResult | None,
) -> tuple[bool, bool, str]:
    """
    Check for class-level disagreement and notable probability differences.

    Returns (models_disagree, notable_prob_diff, note).
    """
    if not (pred1 and pred1.success and pred2 and pred2.success):
        return False, False, ""

    c1 = pred1.predicted_class
    c2 = pred2.predicted_class

    if c1 is None or c2 is None:
        return False, False, ""

    if c1 != c2:
        l1 = pred1.predicted_label
        l2 = pred2.predicted_label
        p1 = pred1.phishing_probability
        p2 = pred2.phishing_probability
        p1_str = f"{p1*100:.1f}%" if p1 is not None else "N/A"
        p2_str = f"{p2*100:.1f}%" if p2 is not None else "N/A"
        note = (
            f"Model 1 predicts {l1} (phishing prob: {p1_str}), "
            f"while Model 2 predicts {l2} (phishing prob: {p2_str}). "
            "Further investigation is recommended before drawing conclusions."
        )
        return True, False, note

    # Same class — check probability magnitude difference
    p1, _ = _validate_probability(pred1.phishing_probability, "M1")
    p2, _ = _validate_probability(pred2.phishing_probability, "M2")
    if p1 is not None and p2 is not None:
        diff = abs(p2 - p1)
        if diff > NOTABLE_PROBABILITY_DIFF:
            note = (
                f"Both models predict {pred1.predicted_label}, but their phishing "
                f"probabilities differ notably: Model 1 = {p1*100:.1f}%, "
                f"Model 2 = {p2*100:.1f}% (difference: {diff*100:.1f}%). "
                "This may reflect differences in the security features observed."
            )
            return False, True, note

    return False, False, ""


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _url_evidence(
    validation_result,
    feature_result,
) -> list[EvidenceItem]:
    """
    Extract notable URL structural observations from ValidationResult and FeatureResult.

    Only emits observations that are actually present in the collected data.
    """
    items: list[EvidenceItem] = []

    if validation_result is None:
        items.append(EvidenceItem(
            kind="missing", label="URL validation result",
            detail="URL validation data was not supplied to the risk assessor.",
            severity="warn",
        ))
        return items

    vr = validation_result

    # Scheme
    if not vr.uses_https:
        items.append(EvidenceItem(
            kind="observed", label="Plain HTTP",
            detail=(
                "The URL uses plain HTTP (unencrypted). "
                "Legitimate financial and sensitive services almost always use HTTPS."
            ),
            severity="flag",
        ))

    # IP address as hostname
    if vr.is_ip_address:
        items.append(EvidenceItem(
            kind="observed", label="IP address as hostname",
            detail=(
                f"The URL's hostname is a direct IP address ({vr.hostname}). "
                "Legitimate services almost always use a registered domain name."
            ),
            severity="flag",
        ))

    # Non-default port
    if vr.port is not None:
        default_ports = {"http": 80, "https": 443}
        if vr.port != default_ports.get(vr.scheme, -1):
            items.append(EvidenceItem(
                kind="observed", label="Non-standard port",
                detail=f"The URL uses port {vr.port}, which is not the default for {vr.scheme.upper()}.",
                severity="warn",
            ))

    if feature_result is None:
        items.append(EvidenceItem(
            kind="missing", label="URL feature extraction",
            detail="Feature extraction data was not supplied to the risk assessor.",
            severity="warn",
        ))
        return items

    num = feature_result.numeric

    # URL length
    url_len = num.get("url_length", 0)
    if url_len > 100:
        items.append(EvidenceItem(
            kind="model_input", label="Long URL",
            detail=f"URL length is {url_len} characters. Unusually long URLs are a common structural indicator.",
            severity="warn" if url_len > 150 else "info",
        ))

    # Subdomain depth
    sub_count = num.get("host_subdomain_count", 0)
    if sub_count >= 2:
        items.append(EvidenceItem(
            kind="model_input", label="Multiple subdomain labels",
            detail=(
                f"The hostname has {sub_count} subdomain label(s) before the registered domain. "
                "Phishing URLs sometimes use deep subdomains to embed brand names."
            ),
            severity="warn" if sub_count >= 3 else "info",
        ))

    # Punycode / IDN
    if num.get("flag_has_punycode"):
        items.append(EvidenceItem(
            kind="observed", label="Punycode / IDN hostname",
            detail=(
                "The hostname contains an Internationalized Domain Name (IDN) or "
                "Punycode-encoded label. This can be used in homograph attacks."
            ),
            severity="flag",
        ))

    # URL shortener
    if num.get("flag_is_url_shortener"):
        items.append(EvidenceItem(
            kind="observed", label="URL shortener detected",
            detail=(
                "The hostname matches a known URL shortening service. "
                "The true destination is hidden — examine the resolved URL before proceeding."
            ),
            severity="flag",
        ))

    # Keyword matches
    kw_matched = feature_result.keywords_matched
    if kw_matched:
        kw_str = ", ".join(kw_matched[:8])
        items.append(EvidenceItem(
            kind="model_input", label="Sensitive keywords in URL",
            detail=(
                f"The URL contains keyword(s) commonly found in phishing URLs: {kw_str}. "
                "Legitimate sites also use these terms — this is a structural observation only."
            ),
            severity="info",
        ))

    # At-sign in URL (outside authority)
    if num.get("flag_has_at_symbol"):
        items.append(EvidenceItem(
            kind="observed", label="@ symbol in URL",
            detail=(
                "The URL contains an @ symbol (outside the authority section). "
                "This technique can be used to disguise the actual destination."
            ),
            severity="flag",
        ))

    # Percent-encoding
    pct_count = num.get("char_count_percent", 0)
    if pct_count > 5:
        items.append(EvidenceItem(
            kind="model_input", label="High percent-encoding",
            detail=f"The URL contains {pct_count} percent-encoded characters, which is above normal.",
            severity="warn",
        ))

    return items


def _security_evidence(security_result) -> list[EvidenceItem]:
    """
    Extract security-configuration observations from SecurityAnalysisResult.
    Returns an empty list when security analysis was not run.
    """
    items: list[EvidenceItem] = []

    if security_result is None:
        items.append(EvidenceItem(
            kind="missing", label="Website security analysis",
            detail=(
                "Website security analysis (Tab 3) has not been run for this URL. "
                "HTTP headers, cookie attributes, and TLS certificate data are unavailable."
            ),
            severity="warn",
        ))
        return items

    if not security_result.success:
        items.append(EvidenceItem(
            kind="missing", label="Website security analysis failed",
            detail=(
                f"Security analysis could not be completed: {security_result.error} "
                "HTTP header, cookie, and TLS evidence is unavailable."
            ),
            severity="warn",
        ))
        return items

    # TLS
    tls = security_result.tls
    if tls.applicable:
        if tls.verified:
            days = tls.days_remaining
            if days is not None and days < 7:
                items.append(EvidenceItem(
                    kind="observed", label="TLS certificate expiring very soon",
                    detail=f"The TLS certificate expires in {days} day(s).",
                    severity="flag",
                ))
            elif days is not None and days < 30:
                items.append(EvidenceItem(
                    kind="observed", label="TLS certificate expiring soon",
                    detail=f"The TLS certificate expires in {days} day(s).",
                    severity="warn",
                ))
            else:
                items.append(EvidenceItem(
                    kind="observed", label="TLS certificate verified",
                    detail=(
                        f"TLS certificate is valid. Issuer: {tls.issuer or '—'}. "
                        f"Valid until: {tls.valid_until or '—'}."
                    ),
                    severity="info",
                ))
        else:
            items.append(EvidenceItem(
                kind="observed", label="TLS certificate verification failed",
                detail=(
                    f"{tls.error or tls.observation} "
                    "A valid certificate does not prove legitimacy, but an invalid one "
                    "is an additional concern."
                ),
                severity="flag",
            ))
    else:
        items.append(EvidenceItem(
            kind="observed", label="No TLS (plain HTTP)",
            detail=(
                "The URL uses plain HTTP; TLS certificate analysis is not applicable. "
                "Sensitive financial services should use HTTPS."
            ),
            severity="flag",
        ))

    # Security headers — count missing
    hc_list = security_result.header_checks
    missing_hdrs = [
        hc.name for hc in hc_list if hc.applicable and not hc.present
    ]
    present_hdrs = [
        hc.name for hc in hc_list if hc.applicable and hc.present and not hc.suggestion
    ]
    if missing_hdrs:
        items.append(EvidenceItem(
            kind="observed", label="Missing security headers",
            detail=(
                f"{len(missing_hdrs)} of the checked security header(s) are absent: "
                + ", ".join(missing_hdrs[:6])
                + (". Many legitimate sites also omit some headers.")
            ),
            severity="warn" if len(missing_hdrs) >= 4 else "info",
        ))
    if present_hdrs:
        items.append(EvidenceItem(
            kind="observed", label="Security headers present",
            detail=(
                f"{len(present_hdrs)} security header(s) are correctly configured: "
                + ", ".join(present_hdrs[:4])
                + ("." if len(present_hdrs) <= 4 else " (and others).")
            ),
            severity="info",
        ))

    # Cookies
    cookies = security_result.cookie_checks
    if security_result.cookies_present:
        insecure = [c for c in cookies if not c.has_secure or not c.has_httponly]
        if insecure:
            items.append(EvidenceItem(
                kind="observed", label="Cookies with weak security attributes",
                detail=(
                    f"{len(insecure)} observed cookie(s) are missing Secure or HttpOnly. "
                    "Cookie values are never displayed."
                ),
                severity="warn",
            ))
        else:
            items.append(EvidenceItem(
                kind="observed", label="Cookies use secure attributes",
                detail="All observed cookies have Secure and HttpOnly attributes.",
                severity="info",
            ))
    else:
        items.append(EvidenceItem(
            kind="observed", label="No cookies in inspected response",
            detail=(
                "No Set-Cookie headers were observed in the single HEAD response. "
                "The site may set cookies on subsequent interactions."
            ),
            severity="info",
        ))

    return items


def _build_rationale(
    category: RiskCategory,
    primary_prob: float | None,
    primary_model: str,
    selection_note: str,
    models_disagree: bool,
    notable_prob_diff: bool,
    disagreement_note: str,
) -> str:
    """Build a plain-language rationale string for the risk category."""
    if category == RiskCategory.UNKNOWN:
        return (
            "Risk category could not be determined because no valid phishing "
            "probability was available from either model. This may be because "
            "the models have not been run, the model artifacts are missing, or "
            "the URL could not be processed."
        )

    pct = f"{primary_prob * 100:.1f}%" if primary_prob is not None else "N/A"
    base = (
        f"The risk category '{category.value}' is assigned based on a phishing "
        f"probability of {pct} from {primary_model}."
    )

    if selection_note:
        base += f" Note: {selection_note}"

    if models_disagree:
        base += f" Model disagreement detected: {disagreement_note}"
    elif notable_prob_diff:
        base += f" Notable probability difference: {disagreement_note}"

    base += (
        " This is an operational estimate, not a verified ground-truth classification. "
        "Always apply analyst judgement and additional verification."
    )
    return base


def _build_recommendations(
    category: RiskCategory,
    models_disagree: bool,
    security_available: bool,
) -> list[str]:
    """Return actionable recommendations matched to the risk category."""

    recs: list[str] = []

    if category == RiskCategory.UNKNOWN:
        recs.append(
            "Run ML Detection (Tab 4) to obtain model predictions before drawing conclusions."
        )
        if not security_available:
            recs.append(
                "Run Website Security analysis (Tab 3) to gather additional evidence."
            )
        return recs

    if category == RiskCategory.LOW:
        recs.append(
            "No strong phishing indicators were detected by the model. "
            "Continue standard security verification procedures."
        )
        if not security_available:
            recs.append(
                "Consider running Website Security analysis (Tab 3) for additional evidence."
            )

    elif category == RiskCategory.MEDIUM:
        recs.append(
            "Moderate phishing risk detected. Inspect suspicious URL/domain signals "
            "and gather additional evidence before clearing this URL."
        )
        recs.append(
            "Verify the destination domain through an approved threat-intelligence source."
        )
        if not security_available:
            recs.append(
                "Run Website Security analysis (Tab 3) — the extended model (Model 2) "
                "requires security features for a more complete assessment."
            )

    elif category == RiskCategory.HIGH:
        recs.append(
            "High phishing risk. Prioritize analyst review of this URL before "
            "any interaction with the destination."
        )
        recs.append(
            "Verify the domain registration, certificate issuer, and brand ownership "
            "using approved security tools."
        )
        recs.append(
            "Do not submit credentials or interact with this site until it has been cleared."
        )

    elif category == RiskCategory.CRITICAL:
        recs.append(
            "Critical phishing risk. Escalate according to your organization's "
            "incident-response policy."
        )
        recs.append(
            "Do not visit or interact with this URL until it has been reviewed and cleared."
        )
        recs.append(
            "Consider flagging the domain for monitoring and blocking at the network level "
            "if supported by your organization's tools."
        )

    if models_disagree:
        recs.append(
            "Model disagreement detected: the two models predict different classes. "
            "Do not rely on either prediction alone — gather additional evidence before deciding."
        )

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_risk(
    url: str,
    feature_result,         # FeatureResult | None
    prediction1: PredictionResult | None,
    prediction2: PredictionResult | None,
    security_result=None,    # SecurityAnalysisResult | None
    validation_result=None,  # ValidationResult | None
) -> RiskAssessment:
    """
    Produce a structured risk assessment from Module 7 prediction outputs and
    the evidence collected by earlier modules.

    Parameters
    ----------
    url : str
        The URL being assessed (from ValidationResult.normalized_url or the
        raw input).
    feature_result : FeatureResult | None
        Output of extract_features(), or None when unavailable.
    prediction1 : PredictionResult | None
        Model 1 prediction from predict_model1(), or None when not run.
    prediction2 : PredictionResult | None
        Model 2 prediction from predict_model2(), or None when not run.
    security_result : SecurityAnalysisResult | None
        Output of analyze_security(), or None when not run.
    validation_result : ValidationResult | None
        Output of validate_url(), or None when unavailable.

    Returns
    -------
    RiskAssessment
        Always returns a RiskAssessment — never raises an unhandled exception
        on bad or missing inputs.
    """
    # ── Timestamp ────────────────────────────────────────────────────────────
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Select primary probability ───────────────────────────────────────────
    primary_prob, primary_model, selection_note = _select_primary(prediction1, prediction2)

    # ── Risk category ────────────────────────────────────────────────────────
    if primary_prob is None:
        category = RiskCategory.UNKNOWN
    else:
        category = _categorise(primary_prob)

    # ── Disagreement check ──────────────────────────────────────────────────
    models_disagree, notable_prob_diff, disagreement_note = _check_disagreement(
        prediction1, prediction2
    )

    # ── Rationale ────────────────────────────────────────────────────────────
    rationale = _build_rationale(
        category, primary_prob, primary_model, selection_note,
        models_disagree, notable_prob_diff, disagreement_note,
    )

    # ── Evidence ─────────────────────────────────────────────────────────────
    security_available = bool(security_result and security_result.success)
    evidence: list[EvidenceItem] = []
    evidence.extend(_url_evidence(validation_result, feature_result))
    evidence.extend(_security_evidence(security_result))

    # ── Recommendations ──────────────────────────────────────────────────────
    recommendations = _build_recommendations(category, models_disagree, security_available)

    # ── Threshold snapshot ───────────────────────────────────────────────────
    threshold_config = {
        "low_max":    THRESHOLD_LOW_MAX,
        "medium_max": THRESHOLD_MEDIUM_MAX,
        "high_max":   THRESHOLD_HIGH_MAX,
    }

    return RiskAssessment(
        url=url,
        risk_category=category,
        primary_probability=primary_prob,
        primary_model=primary_model,
        risk_rationale=rationale,
        model1_result=prediction1,
        model2_result=prediction2,
        models_disagree=models_disagree,
        disagreement_note=disagreement_note,
        notable_prob_diff=notable_prob_diff,
        evidence=evidence,
        recommendations=recommendations,
        security_analysis_available=security_available,
        assessed_at=now_utc,
        threshold_config=threshold_config,
    )
