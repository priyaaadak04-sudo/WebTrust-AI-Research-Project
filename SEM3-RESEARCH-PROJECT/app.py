"""
app.py
------
WebTrust AI — Financial Website Risk Intelligence Platform.

Entry point for the Streamlit application.

Pages
-----
  Home               — Product overview, features, workflow, and about section.
  Login              — Module 2: authentication form.
  Analyst Detection  — Modules 4/5/6/7/8: analyst-facing URL analysis interface.
  URL Analysis       — Module 1: standalone URL validation view (legacy).
  Model Comparison   — Module 9: admin-only research dashboard (Model 1 vs 2).

Navigation access for the Research Dashboard
---------------------------------------------
  Authenticated admin users see a "📊 Research" button on the Home page
  and on the Analyst Detection page top bar.  Analyst users do not see this
  button.  The page itself also enforces require_admin() server-side.

Navigation is managed via Streamlit's st.navigation (Streamlit ≥ 1.29).
The sidebar nav is hidden; each page renders its own top bar.
Access control is enforced at the top of each protected page — not only in the UI.

Run
---
    .venv\\Scripts\\Activate.ps1
    py -m streamlit run app.py

Security notes
--------------
- URL validation and traditional feature extraction make NO network requests.
- Website security analysis makes live network requests only when explicitly
  triggered by the analyst.
- ML inference uses locally saved model artifacts (no network required).
- Credentials are loaded from environment variables at startup.
- Session state is used for authentication; it is per-session, not persistent.
- The Model Comparison page is admin-only; access is enforced server-side.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call in the app)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WebTrust AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Page definitions
# ---------------------------------------------------------------------------

home_page = st.Page(
    "pages/home.py",
    title="Home",
    icon="🏠",
    default=True,
)

login_page = st.Page(
    "pages/login.py",
    title="Sign In",
    icon="🔑",
)

analyst_detection_page = st.Page(
    "pages/analyst_detection.py",
    title="Analyze URL",
    icon="🔍",
)

url_analysis_page = st.Page(
    "pages/url_analysis.py",
    title="URL Validation",
    icon="🔗",
)

model_comparison_page = st.Page(
    "pages/model_comparison.py",
    title="Model Comparison",
    icon="📊",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
# position="hidden" — each page renders its own branded top bar.
# The Streamlit built-in nav sidebar is suppressed to keep the UI clean.
# model_comparison_page is included in navigation but access control inside
# the page file itself enforces the admin-only restriction server-side.

pg = st.navigation(
    [home_page, login_page, analyst_detection_page, url_analysis_page,
     model_comparison_page],
    position="hidden",
)

pg.run()
