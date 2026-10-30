"""
pages/url_analysis.py
---------------------
WebTrust AI — URL Analysis page (Module 1: URL Input and Validation).

This page provides a focused view of URL validation results.
It is access-controlled — unauthenticated users are redirected to login.

Security notes
--------------
- This page performs NO network requests during URL analysis.
- DNS checking is a separate, explicitly triggered operation.
- Query strings are not displayed to protect sensitive data.
- A syntactically valid URL is NOT a safety certificate.
"""

import streamlit as st

from modules.url_validator import validate_url, ValidationResult
from modules.styles import inject_css
from modules.access_control import require_analyst_or_admin

# ---------------------------------------------------------------------------
# Access control — must run before any content is rendered
# ---------------------------------------------------------------------------

require_analyst_or_admin()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

inject_css()

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------

nav_col, btn_col = st.columns([5, 1])
with nav_col:
    st.markdown(
        """
        <div class="wt-topbar" style="margin-bottom:1.4rem;">
            <div class="wt-brand">
                <span style="font-size:1.4rem;">🛡️</span>
                <div>
                    <div class="wt-brand-name">WebTrust AI</div>
                    <div class="wt-brand-sub">Financial Website Risk Intelligence</div>
                </div>
            </div>
            <span class="badge" style="background:#DBEAFE;color:#1D4ED8;font-size:0.7rem;">URL Validation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Home navigation button — placed in the right column for a nav-bar feel
with btn_col:
    st.markdown("<div style='padding-top:1.25rem;'>", unsafe_allow_html=True)
    if st.button("← Home", type="secondary"):
        st.switch_page("pages/home.py")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input card
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-card">
        <p class="wt-card-title">Analyze a URL</p>
        <p class="wt-card-desc">
            Check URL structure and identify validation issues.
            No network requests are made during this analysis — the website is not visited.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

url_input = st.text_input(
    label="URL to analyze",
    placeholder="https://example.com/login",
    help="Enter the full URL to inspect. No network requests will be made during syntax analysis.",
)

analyze_clicked = st.button("Analyze URL", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Security notice (always visible)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-notice">
        <strong>Security notice:</strong>
        URL validation checks structure, not trustworthiness.
        A valid URL or HTTPS connection does not prove that a website is legitimate.
        This tool does not classify URLs as phishing or legitimate.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Analysis and results
# ---------------------------------------------------------------------------

if analyze_clicked:

    st.markdown('<hr class="wt-divider">', unsafe_allow_html=True)

    # Guard: empty input
    if not url_input or not url_input.strip():
        st.error("No URL entered. Please type a URL in the field above and click **Analyze URL**.")
        st.stop()

    result: ValidationResult = validate_url(url_input)

    # ── Validation status card ───────────────────────────────────────────────
    if result.is_valid:
        status_badge = '<span class="badge badge-valid">Valid</span>'
        status_msg = "The URL is syntactically well-formed."
    else:
        status_badge = '<span class="badge badge-invalid">Invalid</span>'
        status_msg = "The URL failed one or more structural checks."

    st.markdown(
        f"""
        <div class="wt-card">
            <div class="wt-status-row">
                <span class="wt-status-label">Validation status</span>
                {status_badge}
            </div>
            <span style="font-size:0.85rem;color:#475569;">{status_msg}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Errors ───────────────────────────────────────────────────────────────
    if result.errors:
        errors_html = "".join(
            f'<div class="wt-alert wt-alert-error">'
            f'<span class="wt-alert-label">Error:</span>{err}</div>'
            for err in result.errors
        )
        st.markdown(errors_html, unsafe_allow_html=True)

    # ── Warnings ─────────────────────────────────────────────────────────────
    if result.warnings:
        warnings_html = "".join(
            f'<div class="wt-alert wt-alert-warn">'
            f'<span class="wt-alert-label">Warning:</span>{warn}</div>'
            for warn in result.warnings
        )
        st.markdown(warnings_html, unsafe_allow_html=True)

    # ── Parsed components (show when at least a hostname or scheme was found) ─
    if result.hostname or result.scheme:

        # Prepare display values
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

        # Destination status
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
