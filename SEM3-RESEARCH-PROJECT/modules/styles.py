"""
modules/styles.py
-----------------
Shared CSS injected into every page of the WebTrust AI application.
"""

GLOBAL_CSS = """
<style>
    /* ── Remove Streamlit's default top padding ── */
    [data-testid="stAppViewContainer"] > .main > div:first-child {
        padding-top: 0.5rem !important;
    }

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* ── App background ── */
    [data-testid="stAppViewContainer"] {
        background: #F8FAFC !important;
    }
    [data-testid="stHeader"] {
        background: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0;
    }

    /* ── Hide the default Streamlit sidebar nav label ── */
    [data-testid="stSidebarNav"] li span { font-size: 0.88rem; }

    /* ── Shared page header ── */
    .wt-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 0 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    .wt-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .wt-brand-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .wt-brand-sub {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 400;
    }

    /* ── Generic badge ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-valid   { background:#DCFCE7; color:#15803D; }
    .badge-invalid { background:#FEE2E2; color:#B91C1C; }
    .badge-https   { background:#DCFCE7; color:#15803D; }
    .badge-http    { background:#FEF9C3; color:#92400E; }
    .badge-ip      { background:#FEF9C3; color:#92400E; }
    .badge-host    { background:#F1F5F9; color:#475569; }
    .badge-notchk  { background:#F1F5F9; color:#475569; }
    .badge-blocked { background:#FEE2E2; color:#B91C1C; }
    .badge-unverif { background:#FEF9C3; color:#92400E; }
    .badge-verified{ background:#DBEAFE; color:#1D4ED8; }
    .badge-avail   { background:#DCFCE7; color:#15803D; }
    .badge-planned { background:#F1F5F9; color:#64748B; }

    /* ── Cards ── */
    .wt-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .wt-card-title {
        margin: 0 0 0.2rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #0F172A;
    }
    .wt-card-desc {
        margin: 0 0 1.1rem 0;
        font-size: 0.85rem;
        color: #64748B;
    }

    /* ── Feature card (homepage) ── */
    .feat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem 1.5rem 1.2rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .feat-card-icon {
        font-size: 1.5rem;
        margin-bottom: 0.6rem;
        display: block;
    }
    .feat-card-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.5rem 0;
    }
    .feat-card-desc {
        font-size: 0.83rem;
        color: #64748B;
        line-height: 1.55;
        margin: 0 0 1rem 0;
    }

    /* ── Hero section ── */
    .wt-hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 2.8rem 2.4rem 2.4rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .wt-hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 0.75rem;
    }
    .wt-hero h2 {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.25;
        letter-spacing: -0.02em;
        margin: 0 0 1rem 0;
    }
    .wt-hero p {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.65;
        margin: 0 0 1.8rem 0;
        max-width: 600px;
    }

    /* ── Section headings ── */
    .wt-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 0.4rem;
    }
    .wt-section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.01em;
    }
    .wt-section-sub {
        font-size: 0.88rem;
        color: #64748B;
        margin: 0 0 1.6rem 0;
        line-height: 1.55;
    }

    /* ── How it works steps ── */
    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .how-step-num {
        flex-shrink: 0;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: #EFF6FF;
        border: 2px solid #BFDBFE;
        color: #2563EB;
        font-size: 0.82rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .how-step-num.planned {
        background: #F8FAFC;
        border-color: #E2E8F0;
        color: #94A3B8;
    }
    .how-step-text {
        padding-top: 0.2rem;
    }
    .how-step-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 0.15rem 0;
    }
    .how-step-title.planned { color: #94A3B8; }
    .how-step-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin: 0;
    }

    /* ── About section ── */
    .wt-about {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin-top: 0.5rem;
    }
    .wt-about p {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.65;
        margin: 0 0 0.8rem 0;
    }
    .wt-about p:last-child { margin-bottom: 0; }

    /* ── Validation page ── */
    .wt-status-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.1rem;
    }
    .wt-status-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
    }
    .wt-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem 1.2rem;
        margin-top: 0.5rem;
    }
    @media (max-width: 500px) { .wt-grid { grid-template-columns: 1fr; } }
    .wt-field {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 7px;
        padding: 0.6rem 0.85rem;
    }
    .wt-field-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.2rem;
    }
    .wt-field-value {
        font-size: 0.88rem;
        color: #0F172A;
        font-family: 'Consolas', 'Courier New', monospace;
        word-break: break-all;
    }
    .wt-field-value.muted {
        color: #94A3B8;
        font-style: italic;
        font-family: inherit;
    }
    .wt-alert {
        border-radius: 7px;
        padding: 0.65rem 0.9rem;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid;
    }
    .wt-alert-error { background:#FEF2F2; border-left-color:#EF4444; color:#7F1D1D; }
    .wt-alert-warn  { background:#FFFBEB; border-left-color:#F59E0B; color:#78350F; }
    .wt-alert-label { font-weight:700; margin-right:0.3rem; }
    .wt-notice {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .wt-notice strong { color: #475569; }
    .wt-divider {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.4rem 0;
    }
    .wt-dest-row {
        display: flex;
        align-items: baseline;
        gap: 0.55rem;
        flex-wrap: wrap;
    }
    .wt-dest-explain { font-size:0.82rem; color:#64748B; }

    /* ── Streamlit button overrides ── */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.4rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border-radius: 7px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.4rem !important;
        border-color: #E2E8F0 !important;
        color: #0F172A !important;
    }
</style>
"""


def inject_css() -> None:
    """Inject the shared WebTrust AI stylesheet into the current Streamlit page."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
