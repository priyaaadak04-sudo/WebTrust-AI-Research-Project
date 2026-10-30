"""
pages/analyst_detection.py
--------------------------
Module 4 — Analyst Detection Interface.

This is the primary analyst-facing page where a security analyst submits a
suspicious URL and views the assessment produced by currently integrated modules.

Access control
--------------
Requires authentication (analyst or admin role).
The admin researcher dashboard is NOT shown here.
Research-only model comparison metrics are never displayed on this page.

Current capabilities
---------------------
- URL validation (Module 1: fully implemented).
- Traditional URL feature extraction (Module 5: fully implemented).
- Website security analysis (Module 6: fully implemented).
- ML phishing classification: not yet integrated.

Integration contract for teammates
------------------------------------
When the ML models are ready, call them via the agreed interface. Display only
the analyst-facing result. Do not expose Model 1 vs. Model 2 comparison metrics
on this page.

Session state keys used
------------------------
  wt_analysis_url       — the URL that was most recently analysed
  wt_analysis_result    — ValidationResult for that URL
  wt_analysis_features  — FeatureResult for that URL (None if invalid)
  wt_security_result    — SecurityAnalysisResult for that URL (None if not run)
  wt_security_url       — URL that produced wt_security_result
"""

import streamlit as st

from modules.styles import inject_css
from modules.access_control import require_analyst_or_admin
from modules.authentication import (
    get_current_display_name,
    get_current_role,
    logout,
    ROLE_ADMIN,
)
from modules.url_validator import validate_url, ValidationResult
from modules.feature_extractor import extract_features, FeatureResult
from modules.security_analyzer import analyze_security, SecurityAnalysisResult
from modules.ml_predictor import (
    predict_model1,
    predict_model2,
    load_models,
    load_metrics,
    PredictionResult,
)
from modules.risk_assessor import assess_risk, RiskAssessment, RiskCategory

# ---------------------------------------------------------------------------
# Access control — must be the first executed statement after imports
# ---------------------------------------------------------------------------

require_analyst_or_admin()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

inject_css()

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------

_display_name = get_current_display_name()
_role = get_current_role()
_role_label = "Researcher / Admin" if _role == ROLE_ADMIN else "Security Analyst"

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
            <span class="badge" style="background:#DBEAFE;color:#1D4ED8;font-size:0.7rem;">
                URL Analysis
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    st.markdown("<div style='padding-top:1.15rem;'>", unsafe_allow_html=True)
    if _role == ROLE_ADMIN:
        nb1, nb2, nb3 = st.columns(3)
        with nb1:
            if st.button("← Home", type="secondary", use_container_width=True):
                st.switch_page("pages/home.py")
        with nb2:
            if st.button("📊 Research", type="secondary", use_container_width=True):
                st.switch_page("pages/model_comparison.py")
        with nb3:
            if st.button("Sign out", type="secondary", use_container_width=True):
                logout()
                st.switch_page("pages/home.py")
    else:
        nb1, nb2 = st.columns(2)
        with nb1:
            if st.button("← Home", type="secondary", use_container_width=True):
                st.switch_page("pages/home.py")
        with nb2:
            if st.button("Sign out", type="secondary", use_container_width=True):
                logout()
                st.switch_page("pages/home.py")
    st.markdown("</div>", unsafe_allow_html=True)

# User context line
st.markdown(
    f"""
    <div style="text-align:right; font-size:0.8rem; color:#64748B;
                margin-top:-0.6rem; margin-bottom:0.6rem;">
        Signed in as <strong style="color:#475569;">{_display_name}</strong>
        &nbsp;&middot;&nbsp; {_role_label}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page title and description
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="margin-bottom:1.2rem;">
        <h2 style="font-size:1.5rem;font-weight:800;color:#0F172A;
                   letter-spacing:-0.02em;margin:0 0 0.3rem 0;">
            Analyze a Suspicious URL
        </h2>
        <p style="font-size:0.88rem;color:#64748B;margin:0;line-height:1.5;">
            Submit a suspicious URL for structured analysis. The platform validates URL
            structure and extracts traditional features. No network requests are made.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# URL input area — compact
# ---------------------------------------------------------------------------

input_col, btn_col = st.columns([5, 1])
with input_col:
    url_input = st.text_input(
        label="URL to analyze",
        placeholder="https://suspicious-example.com/login",
        label_visibility="collapsed",
        help="Enter the complete URL. No network requests are made during analysis.",
        key="url_input_field",
    )
with btn_col:
    analyze_clicked = st.button(
        "Analyze", type="primary", use_container_width=True
    )

st.markdown(
    """
    <div class="wt-notice" style="margin-top:0.5rem;margin-bottom:0.8rem;">
        <strong>Notice:</strong>
        URL validation checks structure and destination restrictions only.
        A syntactically valid URL or an HTTPS scheme does not prove the destination
        is trustworthy — phishing sites routinely obtain valid certificates.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run analysis and store in session state
# ---------------------------------------------------------------------------
# Analysis is run when the button is clicked and the result is cached in
# session state so that switching between tabs does not re-run validation
# or wipe the URL input.

_SK_URL        = "wt_analysis_url"
_SK_RESULT     = "wt_analysis_result"
_SK_FEATURES   = "wt_analysis_features"
_SK_SEC_RESULT = "wt_security_result"
_SK_SEC_URL    = "wt_security_url"
_SK_ML1_RESULT = "wt_ml1_result"
_SK_ML2_RESULT = "wt_ml2_result"
_SK_RISK       = "wt_risk_assessment"

if analyze_clicked:
    if not url_input or not url_input.strip():
        st.error(
            "No URL entered. Please type a URL in the field above and click **Analyze**."
        )
        st.stop()

    _vr: ValidationResult = validate_url(url_input)
    _fr: FeatureResult | None = extract_features(_vr) if _vr.is_valid else None

    st.session_state[_SK_URL]      = url_input.strip()
    st.session_state[_SK_RESULT]   = _vr
    st.session_state[_SK_FEATURES] = _fr
    # Clear any previous security/ML/risk analysis so the new URL starts fresh
    st.session_state.pop(_SK_SEC_RESULT, None)
    st.session_state.pop(_SK_SEC_URL, None)
    st.session_state.pop(_SK_ML1_RESULT, None)
    st.session_state.pop(_SK_ML2_RESULT, None)
    st.session_state.pop(_SK_RISK, None)

# Retrieve whatever is stored (may be None on first load)
_stored_url: str | None                        = st.session_state.get(_SK_URL)
result: ValidationResult | None                = st.session_state.get(_SK_RESULT)
features: FeatureResult | None                 = st.session_state.get(_SK_FEATURES)
security_result: SecurityAnalysisResult | None = st.session_state.get(_SK_SEC_RESULT)
ml1_result: PredictionResult | None            = st.session_state.get(_SK_ML1_RESULT)
ml2_result: PredictionResult | None            = st.session_state.get(_SK_ML2_RESULT)
risk_assessment: RiskAssessment | None         = st.session_state.get(_SK_RISK)

# Nothing to show yet
if result is None:
    st.stop()

st.markdown('<hr class="wt-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Summary strip — always visible once analysis has run
# ---------------------------------------------------------------------------

_v_badge  = '<span class="badge badge-valid">Valid</span>'   if result.is_valid \
            else '<span class="badge badge-invalid">Invalid</span>'

# Domain legitimacy: not assessed — URL structural validation never implies
# domain ownership or brand authenticity.
_leg_badge = '<span class="badge badge-notchk">Not Assessed</span>'

# ML classification: show risk category when available, otherwise Not Run.
if risk_assessment is not None:
    _rc_badge_map = {
        "Low":      "badge-valid",
        "Medium":   "badge-unverif",
        "High":     "badge-blocked",
        "Critical": "badge-invalid",
        "Unknown":  "badge-notchk",
    }
    _rc_val = risk_assessment.risk_category.value
    _rc_cls = _rc_badge_map.get(_rc_val, "badge-notchk")
    _ml_badge = f'<span class="badge {_rc_cls}">{_rc_val} Risk</span>'
elif ml1_result is not None or ml2_result is not None:
    _ml_badge = '<span class="badge badge-notchk">Assessed</span>'
else:
    _ml_badge = '<span class="badge badge-notchk">Not Run</span>'

# Destination status: read the actual value returned by the validator.
# The validator sets this to "not_checked" by default and only changes it
# when a DNS check is explicitly requested.  Never override with a
# hardcoded "verified" label.
_dest_strip_meta = {
    "not_checked":            ("badge-notchk",  "Not Checked"),
    "blocked":                ("badge-blocked",  "Blocked"),
    "destination_unverified": ("badge-unverif",  "Unverified"),
    "verified_for_policy":    ("badge-verified", "Verified for Policy"),
}
_dest_strip_cls, _dest_strip_label = _dest_strip_meta.get(
    result.destination_status,
    ("badge-notchk", result.destination_status),
)
_dest_strip_badge = (
    f'<span class="badge {_dest_strip_cls}">{_dest_strip_label}</span>'
)

st.markdown(
    f"""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                padding:0.9rem 1.4rem;margin-bottom:1.2rem;
                display:flex;gap:2rem;flex-wrap:wrap;align-items:center;">
        <div>
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:#64748B;margin-bottom:0.3rem;">
                URL Validation
            </div>
            {_v_badge}
        </div>
        <div>
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:#64748B;margin-bottom:0.3rem;">
                Destination Status
            </div>
            {_dest_strip_badge}
        </div>
        <div>
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:#64748B;margin-bottom:0.3rem;">
                Domain Legitimacy
            </div>
            {_leg_badge}
        </div>
        <div>
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:#64748B;margin-bottom:0.3rem;">
                ML Phishing Classification
            </div>
            {_ml_badge}
        </div>
        <div style="margin-left:auto;font-size:0.8rem;color:#64748B;
                    font-family:'Consolas','Courier New',monospace;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                    max-width:280px;" title="{_stored_url or ''}">
            {_stored_url or ''}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# If validation failed, show errors and stop — do not show tabs
if not result.is_valid:
    if result.errors:
        for err in result.errors:
            st.markdown(
                f'<div class="wt-alert wt-alert-error">'
                f'<span class="wt-alert-label">Validation error:</span>{err}</div>',
                unsafe_allow_html=True,
            )
    if result.warnings:
        for warn in result.warnings:
            st.markdown(
                f'<div class="wt-alert wt-alert-warn">'
                f'<span class="wt-alert-label">Warning:</span>{warn}</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        """
        <div class="wt-notice" style="margin-top:0.6rem;">
            The URL did not pass validation. Analysis cannot proceed for an invalid
            or prohibited URL. Correct the URL and try again.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Validation warnings (non-fatal) shown above the tabs for valid URLs
if result.warnings:
    for warn in result.warnings:
        st.markdown(
            f'<div class="wt-alert wt-alert-warn">'
            f'<span class="wt-alert-label">Warning:</span>{warn}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Four-tab analysis interface
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔗 URL Components", "📊 Traditional Features", "🔒 Website Security", "🤖 ML Detection"]
)

# ============================================================================
# TAB 1 — URL Components
# ============================================================================

with tab1:

    port_display = (
        f'<span class="wt-field-value">{result.port}</span>'
        if result.port is not None
        else '<span class="wt-field-value muted">Not specified</span>'
    )
    path_display = (
        f'<span class="wt-field-value">{result.path}</span>'
        if result.path
        else '<span class="wt-field-value muted">/</span>'
    )
    https_badge = (
        '<span class="badge badge-https">HTTPS</span>'
        if result.uses_https
        else '<span class="badge badge-http">HTTP — unencrypted</span>'
    )
    host_type_badge = (
        '<span class="badge badge-ip">IP address</span>'
        if result.is_ip_address
        else '<span class="badge badge-host">Hostname</span>'
    )

    _status_meta = {
        "not_checked":            ("badge-notchk",  "Not checked",
                                   "No destination verification has been performed."),
        "blocked":                ("badge-blocked",  "Blocked",
                                   "Destination falls within a prohibited address range."),
        "destination_unverified": ("badge-unverif",  "Unverified",
                                   "A DNS check was attempted but could not confirm the destination."),
        "verified_for_policy":    ("badge-verified", "Verified for policy",
                                   "DNS check passed policy — not a safety certificate."),
    }
    dest_cls, dest_label, dest_explain = _status_meta.get(
        result.destination_status,
        ("badge-notchk", result.destination_status, ""),
    )
    dest_badge = f'<span class="badge {dest_cls}">{dest_label}</span>'

    st.markdown(
        f"""
        <div class="wt-card">
            <p class="wt-card-title">URL Components</p>
            <div class="wt-grid">
                <div class="wt-field">
                    <div class="wt-field-label">Scheme</div>
                    <div class="wt-field-value">{result.scheme or '—'}</div>
                </div>
                <div class="wt-field">
                    <div class="wt-field-label">Hostname</div>
                    <div class="wt-field-value">{result.hostname or '—'}</div>
                </div>
                <div class="wt-field">
                    <div class="wt-field-label">Port</div>
                    {port_display}
                </div>
                <div class="wt-field">
                    <div class="wt-field-label">Path</div>
                    {path_display}
                </div>
                <div class="wt-field">
                    <div class="wt-field-label">Encryption</div>
                    {https_badge}
                </div>
                <div class="wt-field">
                    <div class="wt-field-label">Host type</div>
                    {host_type_badge}
                </div>
                <div class="wt-field" style="grid-column: span 2;">
                    <div class="wt-field-label">Destination status</div>
                    <div class="wt-dest-row">
                        {dest_badge}
                        <span class="wt-dest-explain">{dest_explain}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wt-card">
            <div class="wt-status-row">
                <span class="wt-status-label">Domain legitimacy</span>
                <span class="badge badge-notchk">Not Assessed</span>
            </div>
            <span style="font-size:0.85rem;color:#475569;">
                URL structural validation does not verify domain ownership, brand
                authenticity, or registration history. A well-formed hostname is not
                evidence of legitimacy.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# TAB 2 — Traditional Features
# ============================================================================

with tab2:

    if features is None:
        st.info("Feature extraction requires a valid URL.")
    else:
        num  = features.numeric
        desc = features.descriptive

        # ── URL length and structure ────────────────────────────────────────
        st.markdown(
            f"""
            <div class="wt-card">
                <p class="wt-card-title">URL Length and Structure</p>
                <div class="wt-grid">
                    <div class="wt-field">
                        <div class="wt-field-label">Total URL length</div>
                        <div class="wt-field-value">{num["url_length"]} characters</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Path length</div>
                        <div class="wt-field-value">{num["url_path_length"]} characters</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Query string length</div>
                        <div class="wt-field-value">
                            {"Not present" if not num["flag_has_query"] else f'{num["url_query_length"]} characters'}
                        </div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Query parameters</div>
                        <div class="wt-field-value">{num["url_num_query_params"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Fragment</div>
                        <div class="wt-field-value">
                            {"Present" if num["flag_has_fragment"] else "Not present"}
                        </div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Port</div>
                        <div class="wt-field-value">
                            {"Non-default (" + str(result.port) + ")" if num["flag_has_non_default_port"] else
                             (str(result.port) + " (default)" if result.port else "Not specified")}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Character counts ────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="wt-card">
                <p class="wt-card-title">Character Counts</p>
                <div class="wt-grid">
                    <div class="wt-field">
                        <div class="wt-field-label">Dots (.)</div>
                        <div class="wt-field-value">{num["char_count_dot"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Hyphens (-)</div>
                        <div class="wt-field-value">{num["char_count_hyphen"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Underscores (_)</div>
                        <div class="wt-field-value">{num["char_count_underscore"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Slashes (/)</div>
                        <div class="wt-field-value">{num["char_count_slash"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Percent-encoded (%)</div>
                        <div class="wt-field-value">{num["char_count_percent"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">At-sign (@)</div>
                        <div class="wt-field-value">{num["char_count_at"]}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Hostname and domain features ────────────────────────────────────
        _reg_dom_display = (
            desc["registered_domain"]
            if desc["registered_domain"] != "unknown"
            else "Could not be determined"
        )
        _sub_display = (
            str(num["host_subdomain_count"])
            if num["host_subdomain_count"] >= 0
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="wt-card">
                <p class="wt-card-title">Hostname and Domain Features</p>
                <div class="wt-grid">
                    <div class="wt-field">
                        <div class="wt-field-label">Hostname length</div>
                        <div class="wt-field-value">{num["url_hostname_length"]} characters</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Hostname labels</div>
                        <div class="wt-field-value">{num["host_num_labels"]}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Subdomain labels</div>
                        <div class="wt-field-value">{_sub_display}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Registered domain</div>
                        <div class="wt-field-value">{_reg_dom_display}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Direct IP address</div>
                        <div class="wt-field-value">{"Yes" if num["flag_is_ip_address"] else "No"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Punycode / IDN</div>
                        <div class="wt-field-value">{"Detected" if num["flag_has_punycode"] else "Not detected"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">URL shortener</div>
                        <div class="wt-field-value">{"Detected" if num["flag_is_url_shortener"] else "Not detected"}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Keyword indicators ──────────────────────────────────────────────
        _kw_list_display = (
            ", ".join(f"<code>{kw}</code>" for kw in features.keywords_matched)
            if features.keywords_matched
            else "None"
        )
        st.markdown(
            f"""
            <div class="wt-card">
                <p class="wt-card-title">Keyword Indicators</p>
                <p style="font-size:0.83rem;color:#475569;margin:0 0 0.8rem;line-height:1.55;">
                    Potentially relevant keywords found in the URL (case-insensitive).
                    Legitimate websites frequently use these terms — a match is a
                    structural observation, not a phishing verdict.
                </p>
                <div class="wt-field" style="margin-bottom:0.5rem;">
                    <div class="wt-field-label">Keywords matched ({num["kw_num_matched"]})</div>
                    <div class="wt-field-value" style="margin-top:0.25rem;">
                        {_kw_list_display}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature extraction warnings (e.g. registered-domain ambiguity)
        if features.warnings:
            for w in features.warnings:
                st.markdown(
                    f'<div class="wt-alert wt-alert-warn">'
                    f'<span class="wt-alert-label">Note:</span>{w}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="wt-notice">
                <strong>Feature extraction notice:</strong>
                These structural observations are extracted from the URL string and hostname.
                They do not classify this URL as phishing or legitimate. They will supply
                inputs to a machine-learning model when that module is integrated.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB 3 — Website Security  (Module 6)
# ============================================================================

with tab3:

    # ── Trigger button ───────────────────────────────────────────────────────
    sec_col1, sec_col2 = st.columns([5, 2])
    with sec_col1:
        st.markdown(
            """
            <p style="font-size:0.85rem;color:#475569;margin:0.2rem 0 0.6rem 0;
                       line-height:1.55;">
                Click <strong>Analyze Website Security</strong> to inspect HTTP
                security headers, cookies, and the TLS certificate for this URL.
                This makes a live network request to the destination.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with sec_col2:
        run_security = st.button(
            "Analyze Website Security",
            type="primary",
            use_container_width=True,
            key="run_security_btn",
        )

    if run_security:
        with st.spinner("Connecting to destination and inspecting security…"):
            try:
                _sr = analyze_security(result)
                st.session_state[_SK_SEC_RESULT] = _sr
                st.session_state[_SK_SEC_URL]    = _stored_url
                security_result = _sr
            except Exception as _exc:
                st.error(f"Security analysis encountered an unexpected error: {_exc}")
                security_result = None

    # ── Results ─────────────────────────────────────────────────────────────
    if security_result is None:
        st.markdown(
            """
            <div class="wt-notice" style="margin-top:0.5rem;">
                No security analysis has been run for this URL yet.
                Click <strong>Analyze Website Security</strong> above to start.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not security_result.success:
        st.markdown(
            f"""
            <div class="wt-alert wt-alert-error" style="margin-top:0.5rem;">
                <span class="wt-alert-label">Analysis failed:</span>
                {security_result.error}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="wt-notice">
                The security analysis could not be completed. This may be due to a
                network error, a blocked destination, a TLS failure, or the server
                not responding. No conclusions can be drawn from an incomplete analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── A. HTTP Security Headers ─────────────────────────────────────────
        st.markdown(
            """
            <div class="wt-card">
                <p class="wt-card-title">A. HTTP Security Headers</p>
                <p class="wt-card-desc">
                    These response headers instruct browsers how to handle the page
                    securely. A missing header does not prove the site is malicious —
                    many legitimate sites omit some headers.
                </p>
            """,
            unsafe_allow_html=True,
        )

        # Render a compact table of header results
        _hdr_rows = ""
        for hc in security_result.header_checks:
            if not hc.applicable:
                _status_badge = '<span class="badge badge-notchk">Not Applicable</span>'
            elif hc.present and not hc.suggestion:
                _status_badge = '<span class="badge badge-valid">Present</span>'
            elif hc.present and hc.suggestion:
                _status_badge = '<span class="badge badge-unverif">Potential Issue</span>'
            else:
                _status_badge = '<span class="badge badge-notchk">Missing</span>'

            _val_display = (
                f'<code style="font-size:0.78rem;word-break:break-all;">{hc.value}</code>'
                if hc.value
                else '<span style="color:#94A3B8;font-style:italic;">—</span>'
            )
            _obs = hc.not_applicable_reason if not hc.applicable else hc.observation

            _hdr_rows += f"""
            <tr>
                <td style="font-weight:600;font-size:0.82rem;white-space:nowrap;
                            vertical-align:top;padding:0.55rem 0.7rem;">
                    {hc.name}
                </td>
                <td style="padding:0.55rem 0.7rem;vertical-align:top;">
                    {_status_badge}
                </td>
                <td style="padding:0.55rem 0.7rem;vertical-align:top;">
                    {_val_display}
                </td>
                <td style="font-size:0.8rem;color:#475569;padding:0.55rem 0.7rem;
                            vertical-align:top;line-height:1.5;">
                    {_obs}
                </td>
            </tr>
            """

        st.markdown(
            f"""
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
                <thead>
                    <tr style="border-bottom:2px solid #E2E8F0;background:#F8FAFC;">
                        <th style="text-align:left;padding:0.5rem 0.7rem;
                                   font-size:0.7rem;font-weight:700;
                                   letter-spacing:0.06em;text-transform:uppercase;
                                   color:#64748B;">Header</th>
                        <th style="text-align:left;padding:0.5rem 0.7rem;
                                   font-size:0.7rem;font-weight:700;
                                   letter-spacing:0.06em;text-transform:uppercase;
                                   color:#64748B;">Status</th>
                        <th style="text-align:left;padding:0.5rem 0.7rem;
                                   font-size:0.7rem;font-weight:700;
                                   letter-spacing:0.06em;text-transform:uppercase;
                                   color:#64748B;">Value</th>
                        <th style="text-align:left;padding:0.5rem 0.7rem;
                                   font-size:0.7rem;font-weight:700;
                                   letter-spacing:0.06em;text-transform:uppercase;
                                   color:#64748B;">Observation</th>
                    </tr>
                </thead>
                <tbody>
                    {_hdr_rows}
                </tbody>
            </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── B. Cookie Security ───────────────────────────────────────────────
        st.markdown(
            """
            <div class="wt-card">
                <p class="wt-card-title">B. Cookie Security</p>
            """,
            unsafe_allow_html=True,
        )

        if not security_result.cookies_present:
            st.markdown(
                """
                <p style="font-size:0.85rem;color:#475569;margin:0 0 0.5rem 0;">
                    No cookies observed in the inspected response.
                    This does not mean the website never uses cookies —
                    cookies may be set on subsequent requests.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _ck_rows = ""
            for ck in security_result.cookie_checks:
                def _yn_badge(val: bool, yes_cls: str = "badge-valid",
                              no_cls: str = "badge-notchk") -> str:
                    return (
                        f'<span class="badge {yes_cls}">Yes</span>'
                        if val else
                        f'<span class="badge {no_cls}">No</span>'
                    )
                _ss = ck.samesite or '<span style="color:#94A3B8;font-style:italic;">Absent</span>'
                _obs_txt = " ".join(ck.observations) if ck.observations else "No issues observed."
                _ck_rows += f"""
                <tr style="border-bottom:1px solid #F1F5F9;">
                    <td style="font-weight:600;font-size:0.82rem;padding:0.55rem 0.7rem;
                                vertical-align:top;">{ck.name}</td>
                    <td style="padding:0.55rem 0.7rem;vertical-align:top;">
                        {_yn_badge(ck.has_secure)}
                    </td>
                    <td style="padding:0.55rem 0.7rem;vertical-align:top;">
                        {_yn_badge(ck.has_httponly)}
                    </td>
                    <td style="padding:0.55rem 0.7rem;vertical-align:top;font-size:0.82rem;">
                        {_ss}
                    </td>
                    <td style="font-size:0.79rem;color:#475569;padding:0.55rem 0.7rem;
                                vertical-align:top;line-height:1.5;">
                        {_obs_txt}
                    </td>
                </tr>
                """
            st.markdown(
                f"""
                <p style="font-size:0.83rem;color:#475569;margin:0 0 0.8rem 0;">
                    Cookie values, session tokens, and authentication data
                    are <strong>never</strong> displayed here.
                    Only security attribute names are shown.
                </p>
                <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
                    <thead>
                        <tr style="border-bottom:2px solid #E2E8F0;background:#F8FAFC;">
                            <th style="text-align:left;padding:0.5rem 0.7rem;
                                       font-size:0.7rem;font-weight:700;
                                       letter-spacing:0.06em;text-transform:uppercase;
                                       color:#64748B;">Cookie name</th>
                            <th style="text-align:left;padding:0.5rem 0.7rem;
                                       font-size:0.7rem;font-weight:700;
                                       letter-spacing:0.06em;text-transform:uppercase;
                                       color:#64748B;">Secure</th>
                            <th style="text-align:left;padding:0.5rem 0.7rem;
                                       font-size:0.7rem;font-weight:700;
                                       letter-spacing:0.06em;text-transform:uppercase;
                                       color:#64748B;">HttpOnly</th>
                            <th style="text-align:left;padding:0.5rem 0.7rem;
                                       font-size:0.7rem;font-weight:700;
                                       letter-spacing:0.06em;text-transform:uppercase;
                                       color:#64748B;">SameSite</th>
                            <th style="text-align:left;padding:0.5rem 0.7rem;
                                       font-size:0.7rem;font-weight:700;
                                       letter-spacing:0.06em;text-transform:uppercase;
                                       color:#64748B;">Observations</th>
                        </tr>
                    </thead>
                    <tbody>
                        {_ck_rows}
                    </tbody>
                </table>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── C. TLS Certificate ───────────────────────────────────────────────
        tls = security_result.tls
        st.markdown(
            """
            <div class="wt-card">
                <p class="wt-card-title">C. TLS Certificate</p>
            """,
            unsafe_allow_html=True,
        )

        if not tls.applicable:
            st.markdown(
                f"""
                <div class="wt-status-row">
                    <span class="wt-status-label">TLS inspection</span>
                    <span class="badge badge-notchk">Not Applicable</span>
                </div>
                <p style="font-size:0.85rem;color:#64748B;margin:0;">
                    {tls.not_applicable_reason}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif tls.error and not tls.verified:
            st.markdown(
                f"""
                <div class="wt-status-row">
                    <span class="wt-status-label">TLS verification</span>
                    <span class="badge badge-invalid">Failed</span>
                </div>
                <p style="font-size:0.85rem;color:#7F1D1D;margin:0 0 0.5rem 0;">
                    {tls.error}
                </p>
                <p style="font-size:0.82rem;color:#64748B;margin:0;line-height:1.5;">
                    Certificate verification failures are not automatically evidence of
                    phishing — they may indicate a misconfigured server or an expired
                    certificate on a legitimate site.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _days_display = (
                f"{tls.days_remaining} days remaining"
                if tls.days_remaining is not None
                else "—"
            )
            _days_color = (
                "#B91C1C" if (tls.days_remaining is not None and tls.days_remaining < 7)
                else "#92400E" if (tls.days_remaining is not None and tls.days_remaining < 30)
                else "#0F172A"
            )
            _tls_badge = (
                '<span class="badge badge-valid">Verified</span>'
                if tls.verified
                else '<span class="badge badge-invalid">Failed</span>'
            )
            st.markdown(
                f"""
                <div class="wt-grid" style="margin-bottom:0.8rem;">
                    <div class="wt-field">
                        <div class="wt-field-label">Verification result</div>
                        <div style="margin-top:0.25rem;">{_tls_badge}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Subject (CN)</div>
                        <div class="wt-field-value">{tls.subject or "—"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Issuer</div>
                        <div class="wt-field-value">{tls.issuer or "—"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Valid from</div>
                        <div class="wt-field-value">{tls.valid_from or "—"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Valid until</div>
                        <div class="wt-field-value">{tls.valid_until or "—"}</div>
                    </div>
                    <div class="wt-field">
                        <div class="wt-field-label">Days remaining</div>
                        <div class="wt-field-value" style="color:{_days_color};">
                            {_days_display}
                        </div>
                    </div>
                </div>
                <p style="font-size:0.82rem;color:#475569;margin:0;line-height:1.5;">
                    {tls.observation}
                </p>
                <p style="font-size:0.8rem;color:#94A3B8;margin:0.5rem 0 0 0;">
                    A valid TLS certificate confirms encrypted transport and that
                    the certificate authority has verified the domain. It does
                    <strong>not</strong> establish that a website is legitimate
                    or free from phishing.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── D. Security Findings ────────────────────────────────────────────
        st.markdown(
            '<div class="wt-card"><p class="wt-card-title">D. Security Findings</p>',
            unsafe_allow_html=True,
        )

        if not security_result.findings:
            st.markdown(
                """
                <p style="font-size:0.85rem;color:#475569;margin:0;">
                    No security issues were observed in the inspected response.
                    This does not mean the site is safe or legitimate — only that
                    no issues were detected in the areas checked.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _severity_meta = {
                "high":   ("#FEE2E2", "#991B1B", "HIGH"),
                "medium": ("#FEF9C3", "#92400E", "MEDIUM"),
                "low":    ("#EFF6FF", "#1E40AF", "LOW"),
                "info":   ("#F1F5F9", "#475569", "INFO"),
            }
            _findings_html = ""
            for f in security_result.findings:
                bg, tc, label = _severity_meta.get(f.severity, ("#F1F5F9", "#475569", f.severity.upper()))
                _findings_html += f"""
                <div style="background:{bg};border-radius:8px;padding:0.8rem 1rem;
                             margin-bottom:0.75rem;">
                    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                        <span style="background:rgba(0,0,0,0.08);border-radius:4px;
                                     padding:0.1rem 0.45rem;font-size:0.68rem;
                                     font-weight:800;letter-spacing:0.08em;color:{tc};">
                            {label}
                        </span>
                        <strong style="font-size:0.88rem;color:#0F172A;">{f.title}</strong>
                    </div>
                    <div style="font-size:0.82rem;color:#374151;margin-bottom:0.3rem;">
                        <strong>Observed:</strong> {f.observed}
                    </div>
                    <div style="font-size:0.82rem;color:#374151;margin-bottom:0.3rem;">
                        <strong>Why it matters:</strong> {f.why_it_matters}
                    </div>
                    <div style="font-size:0.82rem;color:#374151;margin-bottom:0.3rem;">
                        <strong>Remediation:</strong> {f.remediation}
                    </div>
                    <div style="font-size:0.78rem;color:#6B7280;">
                        <em>Evidence: {f.evidence}</em>
                    </div>
                </div>
                """
            st.markdown(
                _findings_html + "</div>",
                unsafe_allow_html=True,
            )

        # ── E. Recommended Improvements ─────────────────────────────────────
        st.markdown(
            '<div class="wt-card"><p class="wt-card-title">E. Recommended Improvements</p>',
            unsafe_allow_html=True,
        )

        if not security_result.improvements:
            st.markdown(
                """
                <p style="font-size:0.85rem;color:#475569;margin:0;">
                    No improvement recommendations based on the inspected response.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _items = "".join(
                f'<li style="font-size:0.84rem;color:#374151;line-height:1.6;">{imp}</li>'
                for imp in security_result.improvements
            )
            st.markdown(
                f'<ul style="margin:0 0 0 1rem;padding:0;">{_items}</ul></div>',
                unsafe_allow_html=True,
            )

        # ── Analyst caveat ──────────────────────────────────────────────────
        st.markdown(
            """
            <div class="wt-notice" style="margin-top:0.5rem;">
                <strong>Security analysis notice:</strong>
                The findings above are based solely on HTTP response headers, Set-Cookie
                attributes, and TLS certificate data observed during a single request.
                A website may lack some security headers for legitimate technical reasons.
                These observations do not independently establish whether a site is
                phishing or legitimate. Always apply professional judgement.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB 4 — ML Detection  (Module 7)
# ============================================================================

with tab4:

    # ── Load model status once per tab render ────────────────────────────────
    _m1_load, _m2_load = load_models()

    # ── Trigger button ───────────────────────────────────────────────────────
    ml_col1, ml_col2 = st.columns([5, 2])
    with ml_col1:
        st.markdown(
            """
            <p style="font-size:0.85rem;color:#475569;margin:0.2rem 0 0.6rem 0;
                       line-height:1.55;">
                Click <strong>Run ML Detection</strong> to classify this URL with
                both trained models. Model 1 uses traditional URL features only.
                Model 2 additionally uses website security features — it requires
                a completed security analysis from Tab 3.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with ml_col2:
        run_ml = st.button(
            "Run ML Detection",
            type="primary",
            use_container_width=True,
            key="run_ml_btn",
            disabled=features is None,
        )

    if run_ml and features is not None:
        _sr_for_ml = st.session_state.get(_SK_SEC_RESULT)
        _p1 = predict_model1(features)
        _p2 = predict_model2(features, _sr_for_ml)
        st.session_state[_SK_ML1_RESULT] = _p1
        st.session_state[_SK_ML2_RESULT] = _p2
        ml1_result = _p1
        ml2_result = _p2
        # Compute risk assessment immediately after predictions
        _ra = assess_risk(
            url=result.normalized_url or _stored_url or "",
            feature_result=features,
            prediction1=_p1,
            prediction2=_p2,
            security_result=_sr_for_ml,
            validation_result=result,
        )
        st.session_state[_SK_RISK] = _ra
        risk_assessment = _ra

    # ── Helper: build badge and probability bar HTML strings ────────────────
    def _pred_badge_html(pred: PredictionResult | None) -> str:
        """Return an HTML badge string for the prediction status."""
        if pred is None:
            return '<span class="badge badge-notchk">Not Run</span>'
        if not pred.success:
            return '<span class="badge badge-notchk">Unavailable</span>'
        if pred.predicted_class == 1:
            return '<span class="badge badge-blocked">Potentially Phishing</span>'
        return '<span class="badge badge-valid">Legitimate</span>'

    def _prob_bar_html(prob: float | None, label: str) -> str:
        """Return an HTML probability bar string, or empty string when prob is None."""
        if prob is None:
            return ""
        pct = round(prob * 100, 1)
        color = "#EF4444" if label == "phishing" else "#22C55E"
        lbl_text = "Phishing" if label == "phishing" else "Legitimate"
        return (
            f'<div style="margin-top:0.35rem;">'
            f'<div style="font-size:0.72rem;color:#64748B;margin-bottom:0.15rem;">'
            f'{lbl_text} probability: {pct}%</div>'
            f'<div style="background:#F1F5F9;border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;'
            f'"></div></div></div>'
        )

    # ── Results ─────────────────────────────────────────────────────────────
    if ml1_result is None and ml2_result is None:
        # Not yet run — show model availability
        _m1_avail = "✅ Available" if _m1_load.loaded else f"⚠ {_m1_load.error}"
        _m2_avail = "✅ Available" if _m2_load.loaded else f"⚠ {_m2_load.error}"
        st.markdown(
            f"""
            <div class="wt-notice" style="margin-top:0.5rem;">
                <strong>Model 1 (Baseline):</strong> {_m1_avail}<br>
                <strong>Model 2 (Extended):</strong> {_m2_avail}<br><br>
                Click <strong>Run ML Detection</strong> above to classify this URL.
                For Model 2, complete <strong>Website Security analysis</strong> first (Tab 3).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Render comparison cards — inline, no nested function ─────────────
        # st.html() is used instead of st.markdown(unsafe_allow_html=True) to
        # avoid Streamlit's context-sensitive HTML sanitisation, which causes raw
        # HTML tags to appear as plain text when markdown is called inside nested
        # column+tab contexts in Streamlit ≥ 1.40.
        # The feature count is rendered with st.caption() (a native Streamlit
        # component) so it is always displayed as readable text, never as HTML.

        c1, c2 = st.columns(2)

        # ── Model 1 card ─────────────────────────────────────────────────────
        with c1:
            if ml1_result is None:
                st.html(
                    '<div class="wt-card" style="min-height:160px;">'
                    '<p class="wt-card-title">Model 1 — Baseline</p>'
                    '<span class="badge badge-notchk">Not Run</span>'
                    '</div>'
                )
            else:
                _b1   = _pred_badge_html(ml1_result)
                _pb1  = _prob_bar_html(ml1_result.phishing_probability, "phishing")
                _lb1  = _prob_bar_html(ml1_result.legitimate_probability, "legitimate")
                _pn1  = (
                    '<div style="font-size:0.75rem;color:#94A3B8;margin-top:0.5rem;'
                    'line-height:1.5;">Probability reflects model confidence, '
                    'not ground truth.</div>'
                    if (ml1_result.success and ml1_result.phishing_probability is not None)
                    else ""
                )
                _en1  = (
                    f'<div style="font-size:0.78rem;color:#B45309;margin-top:0.4rem;">'
                    f'\u2139 {ml1_result.error}</div>'
                    if ml1_result.error else ""
                )
                st.html(
                    f'<div class="wt-card" style="min-height:160px;">'
                    f'<p class="wt-card-title" style="margin-bottom:0.6rem;">'
                    f'Model 1 \u2014 Baseline</p>'
                    f'<div style="margin-bottom:0.4rem;">{_b1}</div>'
                    f'{_pb1}{_lb1}{_pn1}{_en1}'
                    f'</div>'
                )
                # Feature count as a native Streamlit caption — never raw HTML
                st.caption(f"{ml1_result.feature_count} feature(s) used")

        # ── Model 2 card ─────────────────────────────────────────────────────
        with c2:
            if ml2_result is None:
                st.html(
                    '<div class="wt-card" style="min-height:160px;">'
                    '<p class="wt-card-title">Model 2 — Extended</p>'
                    '<span class="badge badge-notchk">Not Run</span>'
                    '</div>'
                )
            else:
                _b2   = _pred_badge_html(ml2_result)
                _pb2  = _prob_bar_html(ml2_result.phishing_probability, "phishing")
                _lb2  = _prob_bar_html(ml2_result.legitimate_probability, "legitimate")
                _pn2  = (
                    '<div style="font-size:0.75rem;color:#94A3B8;margin-top:0.5rem;'
                    'line-height:1.5;">Probability reflects model confidence, '
                    'not ground truth.</div>'
                    if (ml2_result.success and ml2_result.phishing_probability is not None)
                    else ""
                )
                _en2  = (
                    f'<div style="font-size:0.78rem;color:#B45309;margin-top:0.4rem;">'
                    f'\u2139 {ml2_result.error}</div>'
                    if ml2_result.error else ""
                )
                st.html(
                    f'<div class="wt-card" style="min-height:160px;">'
                    f'<p class="wt-card-title" style="margin-bottom:0.6rem;">'
                    f'Model 2 \u2014 Extended</p>'
                    f'<div style="margin-bottom:0.4rem;">{_b2}</div>'
                    f'{_pb2}{_lb2}{_pn2}{_en2}'
                    f'</div>'
                )
                # Feature count as a native Streamlit caption — never raw HTML
                st.caption(f"{ml2_result.feature_count} feature(s) used")

        # Analyst disclaimer
        st.markdown(
            """
            <div class="wt-notice" style="margin-top:0.6rem;">
                <strong>ML classification notice:</strong>
                These classifications are produced by trained statistical models and
                reflect patterns in the training data. A "Legitimate" prediction does
                not confirm that the website is safe or trustworthy. A
                "Potentially Phishing" prediction does not confirm malicious intent.
                Always apply professional judgement and verify findings independently.
                Do not use model output as the sole basis for blocking, reporting,
                or clearing a URL.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Risk Assessment (Module 8) ───────────────────────────────────────────
    # Shown whenever a risk_assessment is available (even across tab switches).
    if risk_assessment is not None:
        _ra = risk_assessment

        # Risk category colour map
        _risk_color = {
            RiskCategory.LOW:      ("#DCFCE7", "#166534", "🟢"),
            RiskCategory.MEDIUM:   ("#FEF9C3", "#92400E", "🟡"),
            RiskCategory.HIGH:     ("#FEE2E2", "#991B1B", "🔴"),
            RiskCategory.CRITICAL: ("#FEE2E2", "#7F1D1D", "🔴"),
            RiskCategory.UNKNOWN:  ("#F1F5F9", "#475569", "⚪"),
        }
        _rc_bg, _rc_tc, _rc_icon = _risk_color.get(_ra.risk_category, ("#F1F5F9", "#475569", "⚪"))
        _primary_pct = (
            f"{_ra.primary_probability * 100:.1f}%"
            if _ra.primary_probability is not None
            else "N/A"
        )

        # ── Risk category header card ────────────────────────────────────────
        st.markdown(
            f"""
            <div style="background:{_rc_bg};border:1px solid #E2E8F0;border-radius:10px;
                         padding:1rem 1.4rem;margin-top:1rem;margin-bottom:0.5rem;">
                <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.5rem;">
                    <span style="font-size:1.5rem;">{_rc_icon}</span>
                    <div>
                        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                                    text-transform:uppercase;color:{_rc_tc};margin-bottom:0.1rem;">
                            Risk Assessment
                        </div>
                        <div style="font-size:1.15rem;font-weight:800;color:{_rc_tc};">
                            {_ra.risk_category.value} Risk
                        </div>
                    </div>
                    <div style="margin-left:auto;text-align:right;">
                        <div style="font-size:0.68rem;color:{_rc_tc};font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.07em;">
                            Primary probability
                        </div>
                        <div style="font-size:1.2rem;font-weight:800;color:{_rc_tc};">
                            {_primary_pct}
                        </div>
                        <div style="font-size:0.72rem;color:{_rc_tc};opacity:0.8;">
                            from {_ra.primary_model}
                        </div>
                    </div>
                </div>
                <p style="font-size:0.83rem;color:{_rc_tc};margin:0;line-height:1.6;">
                    {_ra.risk_rationale}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Disagreement warning ─────────────────────────────────────────────
        if _ra.models_disagree or _ra.notable_prob_diff:
            _warn_icon = "⚠️" if _ra.models_disagree else "ℹ️"
            st.markdown(
                f"""
                <div class="wt-alert wt-alert-warn" style="margin-top:0.4rem;">
                    <span class="wt-alert-label">{_warn_icon}
                        {"Model disagreement" if _ra.models_disagree else "Notable difference"}:
                    </span>
                    {_ra.disagreement_note}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Evidence card ────────────────────────────────────────────────────
        if _ra.evidence:
            _sev_meta = {
                "flag":  ("#FEE2E2", "#991B1B", "FLAG"),
                "warn":  ("#FEF9C3", "#92400E", "WARN"),
                "info":  ("#EFF6FF", "#1D4ED8", "INFO"),
            }
            _ev_html = ""
            for ev in _ra.evidence:
                _ev_bg, _ev_tc, _ev_lbl = _sev_meta.get(
                    ev.severity, ("#F1F5F9", "#475569", ev.severity.upper())
                )
                _kind_icon = {"observed": "👁", "missing": "—", "model_input": "📐"}.get(
                    ev.kind, ""
                )
                _ev_html += f"""
                <div style="display:flex;gap:0.6rem;align-items:flex-start;
                             border-bottom:1px solid #F1F5F9;padding:0.5rem 0;">
                    <span style="background:{_ev_bg};color:{_ev_tc};border-radius:4px;
                                  padding:0.1rem 0.4rem;font-size:0.65rem;font-weight:700;
                                  letter-spacing:0.07em;white-space:nowrap;margin-top:0.1rem;">
                        {_ev_lbl}
                    </span>
                    <div>
                        <div style="font-size:0.82rem;font-weight:600;color:#0F172A;">
                            {_kind_icon} {ev.label}
                        </div>
                        <div style="font-size:0.79rem;color:#475569;line-height:1.5;">
                            {ev.detail}
                        </div>
                    </div>
                </div>
                """
            st.markdown(
                f"""
                <div class="wt-card" style="margin-top:0.5rem;">
                    <p class="wt-card-title">Evidence</p>
                    {_ev_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Recommendations card ─────────────────────────────────────────────
        if _ra.recommendations:
            _rec_items = "".join(
                f'<li style="font-size:0.84rem;color:#374151;line-height:1.65;'
                f'margin-bottom:0.25rem;">{rec}</li>'
                for rec in _ra.recommendations
            )
            st.markdown(
                f"""
                <div class="wt-card" style="margin-top:0.5rem;">
                    <p class="wt-card-title">Analyst Recommendations</p>
                    <ul style="margin:0 0 0 1rem;padding:0;">{_rec_items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Assessment metadata footer ───────────────────────────────────────
        st.markdown(
            f"""
            <div style="font-size:0.72rem;color:#94A3B8;text-align:right;
                         margin-top:0.3rem;">
                Assessed at {_ra.assessed_at} UTC &nbsp;·&nbsp;
                Thresholds: Low &lt;{int(_ra.threshold_config.get('low_max',0.20)*100)}%,
                Medium &lt;{int(_ra.threshold_config.get('medium_max',0.50)*100)}%,
                High &lt;{int(_ra.threshold_config.get('high_max',0.80)*100)}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="wt-notice" style="margin-top:0.4rem;">
                <strong>Risk assessment notice:</strong>
                Risk categories are provisional operational estimates derived from
                model outputs and observed evidence. Thresholds have not been
                scientifically validated. This assessment does not independently
                determine whether a URL is phishing or legitimate. Apply analyst
                judgement and additional verification before taking action.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Overall analyst reminder — shown below the tabs
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-notice" style="margin-top:1rem;">
        <strong>Analyst reminder:</strong>
        The findings above are structured technical observations from currently
        integrated modules. They support your investigation but do not independently
        determine whether this URL is legitimate or malicious. Always apply
        professional judgement and additional verification steps as appropriate.
    </div>
    """,
    unsafe_allow_html=True,
)
