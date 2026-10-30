"""
modules/access_control.py
=========================
Module 2 — Role-Based Access Control (enforcement layer).

Responsibilities
----------------
- Enforce authentication and role checks BEFORE rendering protected content.
- Redirect unauthenticated users to the login page.
- Redirect authenticated users who lack the required role.

Usage pattern (call at the TOP of every protected page, before any rendering)
------------------------------------------------------------------------------

    from modules.access_control import require_authenticated, require_role
    from modules.authentication import ROLE_ANALYST, ROLE_ADMIN

    # Allow any signed-in user:
    require_authenticated()

    # Allow only analysts:
    require_role(ROLE_ANALYST)

    # Allow only admins:
    require_role(ROLE_ADMIN)

Design note
-----------
Role enforcement is implemented in Python, not solely through UI hiding.
A user cannot gain access to protected functionality by navigating directly
to a page URL — Streamlit re-runs the full page script on every navigation,
so the guard at the top of the file is always executed.
"""

import streamlit as st

from modules.authentication import (
    is_authenticated,
    get_current_role,
    ROLE_ANALYST,
    ROLE_ADMIN,
)

# Page paths used for redirection
_LOGIN_PAGE = "pages/login.py"
_HOME_PAGE = "pages/home.py"

# ---------------------------------------------------------------------------
# Enforcement guards
# ---------------------------------------------------------------------------

def require_authenticated() -> None:
    """
    Redirect to the login page if no user is currently signed in.

    Call this at the top of every page that requires authentication,
    before rendering any content.
    """
    if not is_authenticated():
        st.switch_page(_LOGIN_PAGE)


def require_role(role: str) -> None:
    """
    Enforce that the signed-in user holds *role*.

    - Unauthenticated users are sent to the login page.
    - Authenticated users with a different role are sent to the home page
      and shown an access-denied message.

    Call this at the top of every role-restricted page, before rendering.
    """
    if not is_authenticated():
        st.switch_page(_LOGIN_PAGE)

    current_role = get_current_role()
    if current_role != role:
        st.switch_page(_HOME_PAGE)


def require_admin() -> None:
    """Convenience wrapper: require the admin role."""
    require_role(ROLE_ADMIN)


def require_analyst_or_admin() -> None:
    """Allow either an analyst or an admin to access the page."""
    if not is_authenticated():
        st.switch_page(_LOGIN_PAGE)

    current_role = get_current_role()
    if current_role not in (ROLE_ANALYST, ROLE_ADMIN):
        st.switch_page(_HOME_PAGE)
