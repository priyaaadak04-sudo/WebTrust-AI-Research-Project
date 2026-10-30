"""
pages/model_comparison.py
--------------------------
Module 9 — Administrator/Researcher Dashboard for Model Comparison
              and Evaluation Metrics.

Access control
--------------
Requires the admin role.  Analysts and unauthenticated users cannot access
this page.  Authorization is enforced server-side in this file before any
content is rendered, not merely by hiding a navigation link.

Research question
-----------------
Do website security configuration features improve phishing-detection
performance compared with using traditional URL/domain features alone?

What this page shows
--------------------
A. Dataset and Experiment Information
B. Evaluation Metrics Comparison (with absolute differences M2 − M1)
C. Metric Comparison Bar Chart
D. Confusion Matrices for both models
E. ROC Curves (when per-sample evaluation arrays are available)
F. Precision-Recall Curves (when per-sample evaluation arrays are available)
G. Research Findings Summary
H. Experimental Validity Assessment
I. Feature Lists
J. Export (CSV download)

What this page does NOT do
---------------------------
- Never displays fabricated, interpolated, or example metric values.
- Never declares one model superior based on a single metric alone.
- Does not expose per-analyst URL submission data.
- Does not allow triggering, overwriting, or modifying training from the UI.
- Does not claim statistical significance without a supporting test.
- Does not present synthetic results as real-world research evidence.

Session state keys used
-----------------------
    wt_dashboard_csv — cached CSV string for the export download button
"""

import streamlit as st

from modules.styles import inject_css
from modules.access_control import require_admin
from modules.authentication import get_current_display_name, logout
from modules.ml_predictor import MODEL1_FEATURES, MODEL2_FEATURES
from modules.dashboard_data import (
    load_dashboard_data,
    load_eval_arrays,
    build_export_csv,
    describe_validity_issues,
    METRIC_DEFINITIONS,
    DashboardData,
    MetricSet,
)

# ---------------------------------------------------------------------------
# Access control — admin only; MUST be the first executable statement
# ---------------------------------------------------------------------------

require_admin()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

inject_css()

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------

_display_name = get_current_display_name()

nav_brand, nav_right = st.columns([4, 2])

with nav_brand:
    st.markdown(
        """
        <div class="wt-topbar" style="padding-bottom:0.8rem;">
            <div class="wt-brand">
                <span style="font-size:1.4rem;">🛡️</span>
                <div>
                    <div class="wt-brand-name">WebTrust AI</div>
                    <div class="wt-brand-sub">Financial Website Risk Intelligence</div>
                </div>
            </div>
            <span class="badge" style="background:#F3E8FF;color:#7C3AED;font-size:0.7rem;">
                Research Dashboard
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    st.markdown("<div style='padding-top:1.15rem;'>", unsafe_allow_html=True)
    nb1, nb2, nb3 = st.columns(3)
    with nb1:
        if st.button("← Home", type="secondary", use_container_width=True):
            st.switch_page("pages/home.py")
    with nb2:
        if st.button("🔍 Analyze URL", type="secondary", use_container_width=True):
            st.switch_page("pages/analyst_detection.py")
    with nb3:
        if st.button("Sign out", type="secondary", use_container_width=True):
            logout()
            st.switch_page("pages/home.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="text-align:right; font-size:0.8rem; color:#64748B;
                margin-top:-0.6rem; margin-bottom:0.6rem;">
        Signed in as <strong style="color:#475569;">{_display_name}</strong>
        &nbsp;&middot;&nbsp; Researcher / Admin
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="margin-bottom:1.2rem;">
        <h2 style="font-size:1.5rem;font-weight:800;color:#0F172A;
                   letter-spacing:-0.02em;margin:0 0 0.3rem 0;">
            Model Comparison Dashboard
        </h2>
        <p style="font-size:0.88rem;color:#64748B;margin:0 0 0.3rem 0;line-height:1.5;">
            <strong>Research question:</strong>
            Do website security configuration features improve phishing-detection
            performance compared with using traditional URL/domain features alone?
        </p>
        <p style="font-size:0.85rem;color:#64748B;margin:0;line-height:1.5;">
            Evaluation results are loaded from
            <code>models/metrics.json</code>. All values are computed on a
            held-out test set; training-set metrics are never shown.
            This page is restricted to authorised researchers only.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

dd: DashboardData = load_dashboard_data()
eval_arrays       = load_eval_arrays()

# ---------------------------------------------------------------------------
# Handle load failure
# ---------------------------------------------------------------------------

if not dd.valid:
    st.markdown(
        f"""
        <div class="wt-alert wt-alert-warn" style="margin-bottom:1rem;">
            <span class="wt-alert-label">Metrics unavailable:</span>
            {dd.error}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="wt-notice">
            <strong>How to generate metrics:</strong><br>
            1. Prepare a labelled CSV dataset (URL column + label column).<br>
            2. Activate the virtual environment:
               <code>.venv\\Scripts\\Activate.ps1</code><br>
            3. Run:
               <code>python train_models.py --dataset data/your_dataset.csv
               --save-eval-arrays</code><br>
            4. Reload this page once training completes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Show non-fatal warnings
for warn in dd.warnings:
    st.markdown(
        f'<div class="wt-alert wt-alert-warn">'
        f'<span class="wt-alert-label">⚠ Data warning:</span>{warn}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="wt-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fmt(val, pct: bool = True, digits: int = 4) -> str:
    """Format a numeric metric for display; show N/A for None."""
    if val is None:
        return '<span style="color:#94A3B8;font-style:italic;">N/A</span>'
    try:
        f = float(val)
        if pct:
            return f"{f * 100:.2f}%"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _diff_html(diff_val: float | None, higher_better: bool) -> str:
    """Return an HTML span for the M2 − M1 difference."""
    if diff_val is None:
        return '<span style="color:#94A3B8;font-style:italic;">N/A</span>'
    diff_pct = diff_val * 100
    if abs(diff_pct) < 0.01:
        return '<span style="color:#64748B;font-size:0.75rem;">≈ equal</span>'
    # M2 better when diff>0 and higher_better, or diff<0 and lower_better
    m2_better = (diff_pct > 0) == higher_better
    color  = "#15803D" if m2_better else "#B91C1C"
    prefix = "M2 " + ("+" if diff_pct > 0 else "") + f"{diff_pct:.2f}%"
    return f'<span style="color:{color};font-size:0.75rem;">{prefix}</span>'


def _metric_row_html(
    label: str, key: str,
    ms1: MetricSet, ms2: MetricSet,
    higher_better: bool,
    m2_skipped: bool,
) -> str:
    v1 = getattr(ms1, key, None)
    v2 = getattr(ms2, key, None) if not m2_skipped else None
    diff = round(float(v2) - float(v1), 6) if (v1 is not None and v2 is not None) else None
    diff_cell = (
        '<td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;">'
        + ('—' if m2_skipped else _diff_html(diff, higher_better))
        + '</td>'
    )
    _na = "<em style='color:#94A3B8;'>Not trained</em>"
    return (
        f'<tr style="border-bottom:1px solid #F1F5F9;">'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;font-weight:600;">{label}</td>'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;">{_fmt(v1)}</td>'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;">'
        f'{_na if m2_skipped else _fmt(v2)}</td>'
        + diff_cell +
        f'</tr>'
    )


# ---------------------------------------------------------------------------
# A. Dataset and Experiment Information
# ---------------------------------------------------------------------------

meta = dd.metadata
m1   = dd.model1
m2   = dd.model2
m2_skipped = m2.skipped

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:0 0 0.5rem 0;">
        A. Dataset and Experiment Information
    </h3>
    """,
    unsafe_allow_html=True,
)

# Synthetic data warning
if "synthetic" in meta.dataset.lower() or "test" in meta.dataset.lower():
    st.markdown(
        f"""
        <div class="wt-alert wt-alert-warn" style="margin-bottom:0.8rem;">
            <span class="wt-alert-label">⚠ Synthetic dataset:</span>
            The current evaluation was run on <strong>{meta.dataset}</strong>, which
            appears to be synthetic or demo data. Metrics from a synthetic dataset
            do <strong>not</strong> reflect real-world phishing-detection performance
            and must not be presented as research findings.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="wt-card">
        <p class="wt-card-title">Dataset and Training Metadata</p>
        <div class="wt-grid">
            <div class="wt-field">
                <div class="wt-field-label">Dataset</div>
                <div class="wt-field-value">{meta.dataset or "—"}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Total rows</div>
                <div class="wt-field-value">{meta.total_rows if meta.total_rows is not None else "—"}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Phishing samples (train + test)</div>
                <div class="wt-field-value">{meta.phishing_count if meta.phishing_count is not None else "—"}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Legitimate samples (train + test)</div>
                <div class="wt-field-value">{meta.legitimate_count if meta.legitimate_count is not None else "—"}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Positive class (phishing) label</div>
                <div class="wt-field-value">
                    {meta.phishing_label if meta.phishing_label is not None else "—"}
                    &nbsp;(Phishing = {meta.phishing_label}, Legitimate = {meta.legitimate_label})
                </div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Test fraction</div>
                <div class="wt-field-value">{meta.test_fraction if meta.test_fraction is not None else "—"}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Random seed</div>
                <div class="wt-field-value">{meta.random_seed if meta.random_seed is not None else "—"}</div>
            </div>
            <div class="wt-field" style="grid-column:span 2;">
                <div class="wt-field-label">Trained at (UTC)</div>
                <div class="wt-field-value">{meta.trained_at_utc or "—"}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Per-model evaluation details
_m1_eval_info = (
    f"Test set: {m1.test_size} samples "
    f"({m1.test_phishing_count} phishing, {m1.test_legit_count} legitimate), "
    f"{m1.feature_count} features, "
    f"train size: {m1.train_size}"
) if (m1.test_size is not None) else "Evaluation details not recorded."

_m2_eval_info = (
    f"Test set: {m2.test_size} samples "
    f"({m2.test_phishing_count} phishing, {m2.test_legit_count} legitimate), "
    f"{m2.feature_count} features, "
    f"train size: {m2.train_size}"
) if (not m2_skipped and m2.test_size is not None) else (
    m2.skip_reason if m2_skipped else "Evaluation details not recorded."
)

st.markdown(
    f"""
    <div class="wt-card">
        <p class="wt-card-title">Evaluation Configuration</p>
        <div class="wt-grid">
            <div class="wt-field">
                <div class="wt-field-label">Model 1 (Baseline) — evaluation</div>
                <div class="wt-field-value" style="font-size:0.82rem;">{_m1_eval_info}</div>
            </div>
            <div class="wt-field">
                <div class="wt-field-label">Model 2 (Extended) — evaluation</div>
                <div class="wt-field-value" style="font-size:0.82rem;">{_m2_eval_info}</div>
            </div>
            <div class="wt-field" style="grid-column:span 2;">
                <div class="wt-field-label">Same held-out test split?</div>
                <div class="wt-field-value" style="font-size:0.82rem;">
                    {"✅ Yes — " + dd.same_test_split_note if dd.same_test_split
                     else ("⚠ " + dd.same_test_split_note if not m2_skipped else "N/A (Model 2 not trained)")}
                </div>
            </div>
            <div class="wt-field" style="grid-column:span 2;">
                <div class="wt-field-label">Per-sample evaluation arrays</div>
                <div class="wt-field-value" style="font-size:0.82rem;">
                    {"✅ Available (ROC and PR curves can be generated)"
                     if eval_arrays.available
                     else "⚠ Not available — " + eval_arrays.error}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# B. Evaluation Metrics Comparison
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.5rem 0;">
        B. Evaluation Metrics Comparison
    </h3>
    <p style="font-size:0.83rem;color:#64748B;margin:0 0 0.6rem 0;">
        All metrics are computed on the held-out test set. The <strong>Δ</strong>
        column shows <em>Model 2 − Model 1</em> (positive = M2 higher; negative = M1 higher).
        A green value means the direction favours Model 2; red means Model 1 is better.
        Higher is better for Accuracy, Precision, Recall, F1, Specificity, and ROC-AUC.
        Lower is better for FPR and FNR.
    </p>
    """,
    unsafe_allow_html=True,
)

# Build metric rows from METRIC_DEFINITIONS + computed specificity
_trows = ""
for lbl, key, higher_better, description in METRIC_DEFINITIONS:
    _trows += _metric_row_html(lbl, key, m1, m2, higher_better, m2_skipped)

# Additional context rows (non-metric counts)
_extra_rows = ""
for lbl, k in [
    ("Test set size",        "test_size"),
    ("Features used",        "feature_count"),
    ("Train set size",       "train_size"),
    ("Training time (s)",    "training_duration_seconds"),
]:
    v1_ = getattr(m1, k, None)
    v2_ = getattr(m2, k, None) if not m2_skipped else None
    _extra_rows += (
        f'<tr style="border-bottom:1px solid #F1F5F9;background:#FAFAFA;">'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;font-weight:600;'
        f'color:#64748B;">{lbl}</td>'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;'
        f'color:#64748B;">{v1_ if v1_ is not None else "—"}</td>'
        f'<td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;'
        f'color:#64748B;">{"—" if m2_skipped else (v2_ if v2_ is not None else "—")}</td>'
        f'<td></td>'
        f'</tr>'
    )

st.markdown(
    f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
        <thead>
            <tr style="border-bottom:2px solid #E2E8F0;background:#F8FAFC;">
                <th style="text-align:left;padding:0.5rem 0.7rem;font-size:0.7rem;
                           font-weight:700;letter-spacing:0.06em;
                           text-transform:uppercase;color:#64748B;">
                    Metric
                </th>
                <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                           font-weight:700;letter-spacing:0.06em;
                           text-transform:uppercase;color:#64748B;">
                    Model 1<br><span style="font-weight:400;">(Baseline, {len(MODEL1_FEATURES)} features)</span>
                </th>
                <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                           font-weight:700;letter-spacing:0.06em;
                           text-transform:uppercase;color:#64748B;">
                    Model 2<br><span style="font-weight:400;">(Extended, {len(MODEL2_FEATURES)} features)</span>
                </th>
                <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                           font-weight:700;letter-spacing:0.06em;
                           text-transform:uppercase;color:#64748B;">
                    Δ (M2 − M1)
                </th>
            </tr>
        </thead>
        <tbody>
            {_trows}
            {_extra_rows}
        </tbody>
    </table>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric legend
with st.expander("Metric definitions", expanded=False):
    for lbl, _, higher_better, desc in METRIC_DEFINITIONS:
        st.markdown(
            f"**{lbl}** ({'↑ higher is better' if higher_better else '↓ lower is better'})  \n"
            f"{desc}"
        )

# ---------------------------------------------------------------------------
# C. Metrics Bar Chart
# ---------------------------------------------------------------------------

if not m2_skipped:
    st.markdown(
        """
        <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
                   margin:1.4rem 0 0.4rem 0;">
            C. Metrics Bar Chart
        </h3>
        <p style="font-size:0.83rem;color:#64748B;margin:0 0 0.6rem 0;">
            Bar chart comparing primary metrics across both models.
            Only metrics where both models have recorded values are shown.
        </p>
        """,
        unsafe_allow_html=True,
    )

    chart_labels = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC-AUC"]
    chart_keys   = ["accuracy", "precision", "recall", "f1_score", "specificity", "roc_auc"]
    m1_vals, m2_vals, valid_labels = [], [], []

    for lbl, key in zip(chart_labels, chart_keys):
        v1_ = getattr(m1, key, None)
        v2_ = getattr(m2, key, None)
        if v1_ is not None and v2_ is not None:
            valid_labels.append(lbl)
            m1_vals.append(round(float(v1_), 4))
            m2_vals.append(round(float(v2_), 4))

    if valid_labels:
        import pandas as pd
        chart_df = pd.DataFrame(
            {"Model 1 (Baseline)": m1_vals, "Model 2 (Extended)": m2_vals},
            index=valid_labels,
        )
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.markdown(
            '<p style="font-size:0.85rem;color:#94A3B8;">No overlapping metrics to chart.</p>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="wt-notice" style="margin-top:1rem;">
            <strong>C. Metrics Bar Chart:</strong> Not available — Model 2 was not trained.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# D. Confusion Matrices
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        D. Confusion Matrices (Test Set)
    </h3>
    <p style="font-size:0.83rem;color:#64748B;margin:0 0 0.6rem 0;">
        Rows = actual class, Columns = predicted class.
        Positive class = Phishing (label 1).
        TN = legitimate correctly identified.
        TP = phishing correctly identified.
        FP = legitimate misclassified as phishing (false alarm).
        FN = phishing missed by the model (false negative).
    </p>
    """,
    unsafe_allow_html=True,
)


def _render_cm(ms: MetricSet, model_label: str) -> str:
    if ms.skipped:
        return (
            f'<div class="wt-card" style="text-align:center;">'
            f'<p class="wt-card-title">{model_label}</p>'
            f'<p style="font-size:0.83rem;color:#94A3B8;margin:0;">Not available</p>'
            f'<p style="font-size:0.8rem;color:#64748B;">{ms.skip_reason}</p>'
            f'</div>'
        )
    tn = ms.tn if ms.tn is not None else "—"
    fp = ms.fp if ms.fp is not None else "—"
    fn = ms.fn if ms.fn is not None else "—"
    tp = ms.tp if ms.tp is not None else "—"
    total = ms.test_size
    return f"""
    <div class="wt-card">
        <p class="wt-card-title">{model_label}</p>
        <table style="border-collapse:collapse;width:100%;text-align:center;font-size:0.85rem;">
            <thead>
                <tr>
                    <th style="padding:0.3rem;"></th>
                    <th style="padding:0.4rem 0.7rem;font-size:0.75rem;color:#64748B;">
                        Predicted: Legit
                    </th>
                    <th style="padding:0.4rem 0.7rem;font-size:0.75rem;color:#64748B;">
                        Predicted: Phishing
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:0.4rem 0.7rem;font-size:0.75rem;color:#64748B;
                               font-weight:600;">Actual: Legit</td>
                    <td style="padding:0.7rem;background:#DCFCE7;border-radius:5px;
                               font-weight:700;color:#15803D;">{tn}<br>
                        <span style="font-size:0.7rem;font-weight:400;">TN</span></td>
                    <td style="padding:0.7rem;background:#FEE2E2;border-radius:5px;
                               font-weight:700;color:#B91C1C;">{fp}<br>
                        <span style="font-size:0.7rem;font-weight:400;">FP</span></td>
                </tr>
                <tr>
                    <td style="padding:0.4rem 0.7rem;font-size:0.75rem;color:#64748B;
                               font-weight:600;">Actual: Phishing</td>
                    <td style="padding:0.7rem;background:#FEE2E2;border-radius:5px;
                               font-weight:700;color:#B91C1C;">{fn}<br>
                        <span style="font-size:0.7rem;font-weight:400;">FN</span></td>
                    <td style="padding:0.7rem;background:#DCFCE7;border-radius:5px;
                               font-weight:700;color:#15803D;">{tp}<br>
                        <span style="font-size:0.7rem;font-weight:400;">TP</span></td>
                </tr>
            </tbody>
        </table>
        <p style="font-size:0.75rem;color:#64748B;margin:0.5rem 0 0 0;">
            Test set: {total} samples total.
            FP = legitimate misclassified as phishing (false alarm).<br>
            FN = phishing URL missed (false negative — the more dangerous error).
        </p>
    </div>
    """


cm_col1, cm_col2 = st.columns(2)
with cm_col1:
    st.markdown(_render_cm(m1, "Model 1 — Baseline"), unsafe_allow_html=True)
with cm_col2:
    st.markdown(_render_cm(m2, "Model 2 — Extended"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# E. ROC Curves
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        E. ROC Curves
    </h3>
    """,
    unsafe_allow_html=True,
)

if eval_arrays.available:
    try:
        from sklearn.metrics import roc_curve, auc as sklearn_auc
        import numpy as np

        fig_data: list[dict] = []

        m1_fpr, m1_tpr, _ = roc_curve(eval_arrays.m1_y_true, eval_arrays.m1_y_prob,
                                        pos_label=1)
        m1_roc_auc = sklearn_auc(m1_fpr, m1_tpr)
        fig_data.append({
            "fpr": m1_fpr.tolist(), "tpr": m1_tpr.tolist(),
            "auc": m1_roc_auc, "label": "Model 1 (Baseline)",
        })

        if eval_arrays.m2_y_true is not None and eval_arrays.m2_y_prob is not None:
            m2_fpr, m2_tpr, _ = roc_curve(eval_arrays.m2_y_true, eval_arrays.m2_y_prob,
                                            pos_label=1)
            m2_roc_auc = sklearn_auc(m2_fpr, m2_tpr)
            fig_data.append({
                "fpr": m2_fpr.tolist(), "tpr": m2_tpr.tolist(),
                "auc": m2_roc_auc, "label": "Model 2 (Extended)",
            })

        # Build a Pandas DataFrame for Streamlit line chart
        import pandas as pd

        # Sample the curve points for display efficiency (max 200 points each)
        roc_dfs = []
        for fd in fig_data:
            n = len(fd["fpr"])
            step = max(1, n // 200)
            idx_ = list(range(0, n, step))
            if idx_[-1] != n - 1:
                idx_.append(n - 1)
            roc_dfs.append(
                pd.DataFrame(
                    {fd["label"] + f" (AUC={fd['auc']:.3f})": [fd["tpr"][i] for i in idx_]},
                    index=[fd["fpr"][i] for i in idx_],
                )
            )

        roc_df = roc_dfs[0]
        for extra in roc_dfs[1:]:
            roc_df = roc_df.join(extra, how="outer").sort_index()

        st.markdown(
            '<p style="font-size:0.83rem;color:#64748B;margin:0 0 0.4rem 0;">'
            'X-axis: False Positive Rate. Y-axis: True Positive Rate (Recall). '
            'A perfect classifier reaches the top-left corner. '
            'A random classifier follows the diagonal (AUC = 0.5).'
            '</p>',
            unsafe_allow_html=True,
        )
        st.line_chart(roc_df, use_container_width=True)

    except Exception as _roc_exc:
        st.markdown(
            f'<div class="wt-alert wt-alert-warn">'
            f'<span class="wt-alert-label">ROC curve error:</span>'
            f'Could not generate ROC curves: {_roc_exc}</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"""
        <div class="wt-notice">
            ROC curves are not available: {eval_arrays.error}
            <br><br>
            Re-run training with:
            <code>python train_models.py --dataset data/your_dataset.csv
            --save-eval-arrays</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# F. Precision-Recall Curves
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        F. Precision-Recall Curves
    </h3>
    """,
    unsafe_allow_html=True,
)

if eval_arrays.available:
    try:
        from sklearn.metrics import precision_recall_curve
        import numpy as np
        import pandas as pd

        pr_dfs = []
        for y_true_, y_prob_, lbl_ in [
            (eval_arrays.m1_y_true, eval_arrays.m1_y_prob, "Model 1 (Baseline)"),
            (eval_arrays.m2_y_true, eval_arrays.m2_y_prob, "Model 2 (Extended)"),
        ]:
            if y_true_ is None or y_prob_ is None:
                continue
            pr_, rc_, _ = precision_recall_curve(y_true_, y_prob_, pos_label=1)
            n = len(pr_)
            step = max(1, n // 200)
            idx_ = list(range(0, n, step))
            if idx_[-1] != n - 1:
                idx_.append(n - 1)
            pr_dfs.append(
                pd.DataFrame(
                    {lbl_: [pr_[i] for i in idx_]},
                    index=[rc_[i] for i in idx_],
                )
            )

        if pr_dfs:
            pr_df = pr_dfs[0]
            for extra in pr_dfs[1:]:
                pr_df = pr_df.join(extra, how="outer").sort_index()

            st.markdown(
                '<p style="font-size:0.83rem;color:#64748B;margin:0 0 0.4rem 0;">'
                'X-axis: Recall (True Positive Rate). Y-axis: Precision. '
                'Higher curves indicate better performance. '
                'Particularly relevant for imbalanced datasets.'
                '</p>',
                unsafe_allow_html=True,
            )
            st.line_chart(pr_df, use_container_width=True)
        else:
            st.markdown(
                '<div class="wt-notice">No PR curve data available.</div>',
                unsafe_allow_html=True,
            )
    except Exception as _pr_exc:
        st.markdown(
            f'<div class="wt-alert wt-alert-warn">'
            f'<span class="wt-alert-label">PR curve error:</span>'
            f'Could not generate precision-recall curves: {_pr_exc}</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"""
        <div class="wt-notice">
            Precision-recall curves are not available: {eval_arrays.error}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# G. Research Findings Summary
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        G. Research Findings Summary
    </h3>
    """,
    unsafe_allow_html=True,
)

if m2_skipped:
    st.markdown(
        """
        <div class="wt-notice">
            Model 2 was not trained, so a model comparison is not possible.
            To answer the research question, train Model 2 on a dataset that
            includes the 14 website security feature columns
            (sec_has_csp, sec_has_hsts, etc.).
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Summarise diffs — determine overall direction
    better_for_m2 = sum(1 for d in dd.diffs if d.m2_is_better is True)
    better_for_m1 = sum(1 for d in dd.diffs if d.m2_is_better is False)
    equal_or_na   = sum(1 for d in dd.diffs if d.m2_is_better is None)

    # Build per-metric finding rows
    _finding_rows = ""
    for d in dd.diffs:
        if d.m1_value is None and d.m2_value is None:
            _status = "Not available"
            _color  = "#94A3B8"
            _icon   = "—"
        elif d.m2_is_better is True:
            _status = f"M2 better by {abs(d.diff)*100:.2f}%"
            _color  = "#15803D"
            _icon   = "✓"
        elif d.m2_is_better is False:
            _status = f"M1 better by {abs(d.diff)*100:.2f}%"
            _color  = "#B91C1C"
            _icon   = "↓"
        else:
            _status = "Approximately equal"
            _color  = "#64748B"
            _icon   = "≈"

        _finding_rows += f"""
        <tr style="border-bottom:1px solid #F1F5F9;">
            <td style="padding:0.5rem 0.7rem;font-size:0.83rem;font-weight:600;">
                {d.label}
            </td>
            <td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;">
                {_fmt(d.m1_value)}
            </td>
            <td style="padding:0.5rem 0.7rem;font-size:0.83rem;text-align:center;">
                {_fmt(d.m2_value)}
            </td>
            <td style="padding:0.5rem 0.7rem;font-size:0.83rem;
                        text-align:center;color:{_color};font-weight:600;">
                {_icon} {_status}
            </td>
        </tr>
        """

    # Overall descriptive summary
    if better_for_m2 > better_for_m1:
        _overall = (
            f"Based on {better_for_m2} of the evaluated metrics, "
            "Model 2 shows higher values (in the favourable direction) compared with Model 1. "
            f"Model 1 shows higher values on {better_for_m1} metric(s)."
        )
    elif better_for_m1 > better_for_m2:
        _overall = (
            f"Based on {better_for_m1} of the evaluated metrics, "
            "Model 1 shows higher values (in the favourable direction) compared with Model 2. "
            f"Model 2 shows higher values on {better_for_m2} metric(s)."
        )
    else:
        _overall = (
            "Model 1 and Model 2 show equal values on the available metrics. "
            "No directional advantage can be identified from the current evaluation."
        )

    st.markdown(
        f"""
        <div class="wt-card">
            <p class="wt-card-title">Per-Metric Findings (positive class = Phishing)</p>
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
                <thead>
                    <tr style="border-bottom:2px solid #E2E8F0;background:#F8FAFC;">
                        <th style="text-align:left;padding:0.5rem 0.7rem;font-size:0.7rem;
                                   font-weight:700;letter-spacing:0.06em;
                                   text-transform:uppercase;color:#64748B;">Metric</th>
                        <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                                   font-weight:700;letter-spacing:0.06em;
                                   text-transform:uppercase;color:#64748B;">Model 1</th>
                        <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                                   font-weight:700;letter-spacing:0.06em;
                                   text-transform:uppercase;color:#64748B;">Model 2</th>
                        <th style="text-align:center;padding:0.5rem 0.7rem;font-size:0.7rem;
                                   font-weight:700;letter-spacing:0.06em;
                                   text-transform:uppercase;color:#64748B;">Finding</th>
                    </tr>
                </thead>
                <tbody>{_finding_rows}</tbody>
            </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wt-card" style="margin-top:0.5rem;">
            <p class="wt-card-title">Overall Descriptive Summary</p>
            <p style="font-size:0.84rem;color:#475569;line-height:1.6;margin:0 0 0.5rem 0;">
                {_overall}
            </p>
            <p style="font-size:0.83rem;color:#475569;line-height:1.6;margin:0;">
                <strong>Caution:</strong>
                This summary describes observed metric differences in the evaluation data.
                It does <strong>not</strong> establish statistical significance.
                No statistical test or confidence interval was computed.
                These findings apply only to the evaluated dataset and must not be
                generalised beyond the scope of this experiment.
                A model with more features does not automatically perform better.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# H. Experimental Validity Assessment
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        H. Experimental Validity Assessment
    </h3>
    """,
    unsafe_allow_html=True,
)

validity_notes = describe_validity_issues(dd)
if validity_notes:
    _v_items = "".join(
        f'<li style="font-size:0.84rem;color:#374151;line-height:1.6;'
        f'margin-bottom:0.3rem;">{note}</li>'
        for note in validity_notes
    )
    st.markdown(
        f"""
        <div class="wt-card">
            <p class="wt-card-title">Validity Notes and Limitations</p>
            <ul style="margin:0 0 0 1rem;padding:0;">{_v_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Interpretation notes
st.markdown(
    """
    <div class="wt-card" style="margin-top:0.5rem;">
        <p class="wt-card-title">Metric Interpretation Notes</p>
        <p style="font-size:0.84rem;color:#475569;line-height:1.6;margin:0 0 0.6rem 0;">
            <strong>Recall</strong> (sensitivity) measures how many phishing URLs the model
            correctly identifies. High recall means fewer missed phishing sites (false negatives).
            For phishing detection, high recall is generally more critical than high precision,
            because missing a phishing site poses greater risk than a false alarm.
        </p>
        <p style="font-size:0.84rem;color:#475569;line-height:1.6;margin:0 0 0.6rem 0;">
            <strong>False Positive Rate (FPR)</strong> measures how often legitimate URLs are
            incorrectly flagged as phishing. A high FPR degrades analyst trust and creates
            unnecessary investigation burden.
        </p>
        <p style="font-size:0.84rem;color:#475569;line-height:1.6;margin:0 0 0.6rem 0;">
            <strong>ROC-AUC</strong> summarises the model's ability to discriminate between
            classes across all classification thresholds. A value of 0.5 means the model
            performs no better than random chance; 1.0 is perfect discrimination.
        </p>
        <p style="font-size:0.84rem;color:#475569;line-height:1.6;margin:0;">
            <strong>Neither model is declared universally superior.</strong>
            Examine all metrics in context. A model with higher accuracy but lower recall
            may miss more phishing sites — which is the more serious failure mode.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# I. Feature Lists
# ---------------------------------------------------------------------------

with st.expander("I. Feature Lists (Model 1 and Model 2)", expanded=False):
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**Model 1 — {len(MODEL1_FEATURES)} features (traditional URL features)**")
        st.code("\n".join(MODEL1_FEATURES), language=None)
    with fc2:
        st.markdown(
            f"**Model 2 — {len(MODEL2_FEATURES)} features "
            f"(Model 1 features + {len(MODEL2_FEATURES) - len(MODEL1_FEATURES)} security features)**"
        )
        st.code("\n".join(MODEL2_FEATURES), language=None)

# ---------------------------------------------------------------------------
# J. Export
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h3 style="font-size:1.1rem;font-weight:700;color:#0F172A;
               margin:1.4rem 0 0.3rem 0;">
        J. Export Evaluation Metrics (CSV)
    </h3>
    <p style="font-size:0.83rem;color:#64748B;margin:0 0 0.6rem 0;">
        Download the recorded evaluation metrics, confusion matrix counts, and
        metric differences as a CSV file. The export does not include raw training
        data, individual URL predictions, user information, or file paths.
    </p>
    """,
    unsafe_allow_html=True,
)

# Cache the CSV in session state to avoid rebuilding on every widget interaction
_SK_CSV = "wt_dashboard_csv"
if _SK_CSV not in st.session_state:
    st.session_state[_SK_CSV] = build_export_csv(dd)

_csv_content = st.session_state.get(_SK_CSV, "")

if _csv_content:
    st.download_button(
        label="⬇ Download metrics CSV",
        data=_csv_content.encode("utf-8"),
        file_name="webtrust_ai_evaluation_metrics.csv",
        mime="text/csv",
        use_container_width=False,
    )
else:
    st.markdown(
        '<p style="font-size:0.85rem;color:#94A3B8;">No exportable data available.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Research disclaimer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-notice" style="margin-top:1.2rem;">
        <strong>Research disclaimer:</strong>
        This dashboard displays actual evaluation metrics computed during the most
        recent training run. Metrics are calculated on a held-out test set that was not
        used during training. They reflect the performance of the trained models on the
        available dataset. If the dataset is synthetic, these results do not reflect
        real-world phishing-detection performance.
        This page is restricted to authorised researchers and must not be shared with
        analyst users or external parties.
        No automatic verdict about model superiority is generated — all comparisons
        must be interpreted in the context of the full evaluation methodology.
    </div>
    """,
    unsafe_allow_html=True,
)
