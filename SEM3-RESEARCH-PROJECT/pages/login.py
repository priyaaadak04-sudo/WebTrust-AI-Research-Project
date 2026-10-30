"""
pages/login.py
--------------
Module 2 — Login page.

Displays a professional login form, validates credentials via
modules/authentication.py, and redirects to the appropriate page
on success.
"""

import streamlit as st

from modules.styles import inject_css
from modules.authentication import authenticate, login, is_authenticated

# ---------------------------------------------------------------------------
# If already signed in, skip the login page
# ---------------------------------------------------------------------------

if is_authenticated():
    st.switch_page("pages/analyst_detection.py")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

inject_css()

# ---------------------------------------------------------------------------
# Additional login-page styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Centre the login card on the page */
        .wt-login-wrap {
            max-width: 420px;
            margin: 3rem auto 0 auto;
        }
        .wt-login-brand {
            text-align: center;
            margin-bottom: 2rem;
        }
        .wt-login-brand-name {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        .wt-login-brand-sub {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.3rem;
        }
        .wt-login-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 2rem 2rem 1.6rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .wt-login-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.25rem 0;
        }
        .wt-login-sub {
            font-size: 0.82rem;
            color: #64748B;
            margin: 0 0 1.5rem 0;
        }
        .wt-demo-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-top: 1.2rem;
            font-size: 0.8rem;
            color: #64748B;
        }
        .wt-demo-box strong { color: #475569; }
        .wt-demo-row {
            display: flex;
            justify-content: space-between;
            padding: 0.2rem 0;
        }
        .wt-demo-label { color: #475569; font-weight: 600; }
        .wt-demo-val   { font-family: 'Consolas','Courier New',monospace; color: #0F172A; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Login form
# ---------------------------------------------------------------------------

st.markdown('<div class="wt-login-wrap">', unsafe_allow_html=True)

# Brand
st.markdown(
    """
    <div class="wt-login-brand">
        <div style="font-size:2.2rem; margin-bottom:0.4rem;">🛡️</div>
        <div class="wt-login-brand-name">WebTrust AI</div>
        <div class="wt-login-brand-sub">Financial Website Risk Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Card
st.markdown('<div class="wt-login-card">', unsafe_allow_html=True)
st.markdown(
    """
    <p class="wt-login-heading">Sign in to your account</p>
    <p class="wt-login-sub">Enter your credentials to access the platform.</p>
    """,
    unsafe_allow_html=True,
)

with st.form("login_form", clear_on_submit=False):
    email = st.text_input(
        "Email address",
        placeholder="you@organisation.com",
        autocomplete="email",
    )
    password = st.text_input(
        "Password",
        type="password",
        placeholder="••••••••",
        autocomplete="current-password",
    )
    submitted = st.form_submit_button(
        "Sign in",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not email or not email.strip():
        st.error("Please enter your email address.")
    elif not password:
        st.error("Please enter your password.")
    else:
        account = authenticate(email, password)
        if account is None:
            st.error("Incorrect email address or password. Please try again.")
        else:
            login(account)
            st.success(f"Signed in as {account.display_name}.")
            st.switch_page("pages/analyst_detection.py")

st.markdown("</div>", unsafe_allow_html=True)  # close .wt-login-card

# Back to home link
st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Return to home page", type="secondary", use_container_width=True):
    st.switch_page("pages/home.py")

st.markdown("</div>", unsafe_allow_html=True)  # close .wt-login-wrap
