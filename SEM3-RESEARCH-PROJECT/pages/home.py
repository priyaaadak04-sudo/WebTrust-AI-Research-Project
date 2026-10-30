"""
pages/home.py
-------------
Module 1 — Homepage.

Displays the product hero, feature overview, workflow steps, and about section.
Navigation respects the current authentication state.
"""

import streamlit as st

from modules.styles import inject_css
from modules.authentication import (
    is_authenticated,
    get_current_display_name,
    get_current_role,
    logout,
    ROLE_ADMIN,
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

inject_css()

# ---------------------------------------------------------------------------
# Scroll-to-features: triggered by the "Explore the Platform" button
# ---------------------------------------------------------------------------
# Streamlit does not expose a native scroll API. When the user clicks
# "Explore the Platform", we set a session flag on that rerun and then
# inject a small JS snippet that scrolls the page to the #features anchor.

if st.session_state.get("_scroll_to_features"):
    st.session_state["_scroll_to_features"] = False
    st.markdown(
        """
        <script>
            window.setTimeout(function () {
                var el = document.getElementById('features-anchor');
                if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            }, 120);
        </script>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------

_authenticated = is_authenticated()
_display_name = get_current_display_name()
_role = get_current_role()

nav_brand_col, nav_links_col = st.columns([3, 2])

with nav_brand_col:
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
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_links_col:
    st.markdown("<div style='padding-top:1.15rem; text-align:right;'>", unsafe_allow_html=True)
    if _authenticated:
        if _role == ROLE_ADMIN:
            # Admin: three buttons — Analyze URL, Research Dashboard, Sign out
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Analyze URL", type="primary", use_container_width=True):
                    st.switch_page("pages/analyst_detection.py")
            with btn_col2:
                if st.button("📊 Research", type="secondary", use_container_width=True):
                    st.switch_page("pages/model_comparison.py")
            with btn_col3:
                if st.button("Sign out", type="secondary", use_container_width=True):
                    logout()
                    st.rerun()
        else:
            # Analyst: two buttons — Analyze URL, Sign out
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Analyze URL", type="primary", use_container_width=True):
                    st.switch_page("pages/analyst_detection.py")
            with btn_col2:
                if st.button("Sign out", type="secondary", use_container_width=True):
                    logout()
                    st.rerun()
    else:
        if st.button("Sign in", type="primary"):
            st.switch_page("pages/login.py")
    st.markdown("</div>", unsafe_allow_html=True)

# Signed-in user greeting
if _authenticated:
    role_label = "Researcher / Admin" if _role == ROLE_ADMIN else "Security Analyst"
    st.markdown(
        f"""
        <div style="text-align:right; font-size:0.8rem; color:#64748B;
                    margin-top:-0.6rem; margin-bottom:0.6rem;">
            Signed in as <strong style="color:#475569;">{_display_name}</strong>
            &nbsp;&middot;&nbsp; {role_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-hero">
        <div class="wt-hero-eyebrow">Financial Website Risk Intelligence</div>
        <h2>Investigate Suspicious<br>Financial Websites</h2>
        <p>
            Help banking and fintech security analysts investigate suspicious URLs,
            examine website-security indicators, and review evidence relevant to
            potential phishing attempts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero CTA buttons
# Use wider columns so neither label is truncated on typical browser widths.
cta_a, cta_b, cta_spacer = st.columns([2.4, 2.4, 2.2])
with cta_a:
    if st.button("Analyze a Suspicious URL", type="primary", use_container_width=True):
        if _authenticated:
            st.switch_page("pages/analyst_detection.py")
        else:
            st.switch_page("pages/login.py")
with cta_b:
    if st.button("Explore the Platform", type="secondary", use_container_width=True):
        # Set a flag so the next rerun scrolls the page to the Features section.
        st.session_state["_scroll_to_features"] = True
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Features section
# ---------------------------------------------------------------------------

# Anchor target for scroll-to-features.  The <span> is zero-height so it
# does not affect layout; the negative margin pulls the scroll destination
# above the section heading so it isn't hidden behind the top bar.
st.markdown(
    '<span id="features-anchor" style="display:block;height:0;margin-top:-1rem;"></span>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wt-section-label">Platform Capabilities</div>
    <h3 class="wt-section-heading">What WebTrust AI Can Do</h3>
    <p class="wt-section-sub">
        WebTrust AI supports banking and fintech security analysts investigating
        suspicious URLs potentially impersonating Indian banking and fintech brands.
    </p>
    """,
    unsafe_allow_html=True,
)

fc1, fc2 = st.columns(2)
fc3, fc4 = st.columns(2)
fc5, fc6 = st.columns(2)

with fc1:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">🔗</span>
            <div class="feat-card-title">URL Structure Analysis</div>
            <p class="feat-card-desc">
                Examine URL structure, hostname characteristics, and potentially
                suspicious patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc2:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">🔒</span>
            <div class="feat-card-title">Website Security Analysis</div>
            <p class="feat-card-desc">
                Examine HTTP security headers, cookie security attributes, and TLS
                certificate properties when these checks are implemented.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc3:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">🤖</span>
            <div class="feat-card-title">ML-Based Phishing Detection</div>
            <p class="feat-card-desc">
                Use machine-learning models to assess whether a website may be
                associated with phishing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc4:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">📋</span>
            <div class="feat-card-title">Risk Assessment and Evidence</div>
            <p class="feat-card-desc">
                Present actual analysis findings and relevant indicators to help
                analysts investigate suspicious links.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc5:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">🔍</span>
            <div class="feat-card-title">Analyst Investigation Dashboard</div>
            <p class="feat-card-desc">
                Support the triage and investigation of suspicious URLs potentially
                impersonating Indian banking and fintech brands.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc6:
    st.markdown(
        """
        <div class="feat-card">
            <span class="feat-card-icon">📊</span>
            <div class="feat-card-title">Research and Model Evaluation</div>
            <p class="feat-card-desc">
                Compare traditional-feature phishing detection against an approach
                that also uses website-security configuration features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Inline CTA
if st.button("Open URL Validation →", type="primary"):
    if _authenticated:
        st.switch_page("pages/analyst_detection.py")
    else:
        st.switch_page("pages/login.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How it works section — built with native Streamlit components so that no
# raw HTML is ever visible to the user regardless of Streamlit's rendering
# behaviour for unsafe_allow_html content.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-section-label">Workflow</div>
    <h3 class="wt-section-heading">How It Works</h3>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='wt-section-sub'>Follow these steps when investigating a suspicious "
    "financial website URL. Findings depend on which analysis modules are currently "
    "available in the platform.</p>",
    unsafe_allow_html=True,
)

# Each step is rendered as a native Streamlit container — no raw HTML reaches
# the browser as visible text.
_steps = [
    (
        "Submit a Suspicious URL",
        "The analyst submits a URL reported for investigation.",
    ),
    (
        "Validate the URL",
        "The system checks the URL structure and applies existing validation "
        "safeguards — scheme, hostname, port, path, and prohibited destinations. "
        "No network requests are made at this stage.",
    ),
    (
        "Extract Website Features",
        "The system extracts available URL, hostname, domain, and webpage-related "
        "features implemented in the application.",
    ),
    (
        "Review Security Findings and ML Results",
        "Website security findings and ML classification are displayed only when "
        "the corresponding modules are implemented and integrated.",
    ),
    (
        "Support Further Investigation",
        "The analyst reviews the available evidence and decides whether further "
        "investigation is required. The final determination is always made by the analyst.",
    ),
]

for _i, (_title, _desc) in enumerate(_steps, start=1):
    _num_col, _text_col = st.columns([1, 11])
    with _num_col:
        st.markdown(
            f'<div style="width:2rem;height:2rem;border-radius:50%;'
            f'background:#EFF6FF;border:2px solid #BFDBFE;color:#2563EB;'
            f'font-size:0.82rem;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;margin-top:0.15rem;">{_i}</div>',
            unsafe_allow_html=True,
        )
    with _text_col:
        st.markdown(
            f"**{_title}**  \n"
            f"<span style='font-size:0.85rem;color:#64748B;line-height:1.55;'>"
            f"{_desc}</span>",
            unsafe_allow_html=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="wt-section-label">About</div>
    <h3 class="wt-section-heading">Built for Security Analysts</h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wt-about">
        <p>
            WebTrust AI is designed for security and fraud analysts in fintech and banking
            organisations who encounter suspicious financial website URLs as part of their
            investigation work. The platform provides structured, repeatable analysis to help
            analysts quickly assess the technical properties of a URL before committing to
            further investigation steps.
        </p>
        <p>
            The platform is built to <strong>support</strong> investigation workflows, not to
            replace human judgement. Automated checks surface structural and technical indicators,
            but every finding must be reviewed in context by a qualified analyst.
        </p>
        <p>
            <strong>Disclaimer:</strong> WebTrust AI supports security investigation. Its results
            do not independently establish that a website is legitimate or malicious.
        </p>
        <p>
            This project is in active development. Capabilities are added progressively as each
            analysis module is completed and validated.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="
        border-top: 1px solid #E2E8F0;
        margin-top: 2.5rem;
        padding-top: 1.6rem;
        padding-bottom: 1.2rem;
        color: #64748B;
        font-size: 0.8rem;
        line-height: 1.65;
    ">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;
                    align-items:flex-start;gap:1.2rem;margin-bottom:1.1rem;">
            <div>
                <div style="font-size:0.95rem;font-weight:700;color:#0F172A;
                            margin-bottom:0.2rem;">🛡️ WebTrust AI</div>
                <div style="font-size:0.78rem;color:#64748B;margin-bottom:0.5rem;">
                    Financial Website Risk Intelligence
                </div>
                <div style="font-size:0.8rem;color:#64748B;max-width:360px;
                            line-height:1.6;">
                    An analyst-oriented platform for examining suspicious financial
                    website URLs and reviewing available security indicators.
                </div>
            </div>
            <div style="font-size:0.82rem;color:#64748B;">
                <div style="font-weight:600;color:#475569;margin-bottom:0.4rem;">
                    Navigate
                </div>
                <div style="margin-bottom:0.25rem;">
                    <a href="/" target="_self"
                       style="color:#2563EB;text-decoration:none;">Home</a>
                </div>
                <div>
                    <a href="/login" target="_self"
                       style="color:#2563EB;text-decoration:none;">Sign In</a>
                </div>
            </div>
        </div>
        <div style="border-top:1px solid #E2E8F0;padding-top:0.9rem;
                    display:flex;flex-wrap:wrap;justify-content:space-between;
                    gap:0.5rem;">
            <div style="font-size:0.76rem;color:#94A3B8;">
                Analysis results support investigation and do not independently
                establish website legitimacy.
            </div>
            <div style="font-size:0.76rem;color:#94A3B8;">
                &copy; 2026 WebTrust AI
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
