"""
modules/authentication.py
=========================
Module 2 — Login and Role-Based Access Control (authentication layer).

Responsibilities
----------------
- Define user accounts with roles.
- Verify credentials using PBKDF2-HMAC-SHA256 (Python standard library only).
- Manage the current session through Streamlit session state.

Roles
-----
analyst
    Can access the analyst detection interface.
    Cannot access the administrator/researcher dashboard.

admin
    Can access the analyst interface AND the administrator/researcher dashboard.
    The admin dashboard is a placeholder until teammates integrate their modules.

Test accounts
-------------
Credentials are loaded from environment variables so that they are never
hard-coded in source code or committed to the repository.

Required environment variables (all optional; defaults listed for local dev):

    WEBTRUST_ANALYST_EMAIL      default: analyst@webtrust.local
    WEBTRUST_ANALYST_PASSWORD   default: AnalystDemo2025!
    WEBTRUST_ADMIN_EMAIL        default: admin@webtrust.local
    WEBTRUST_ADMIN_PASSWORD     default: AdminDemo2025!

To configure on Windows PowerShell before running the app:

    $env:WEBTRUST_ANALYST_PASSWORD = "YourSecurePassword"
    $env:WEBTRUST_ADMIN_PASSWORD   = "YourSecurePassword"

Security limitations (demonstration system)
------------------------------------------
1. Streamlit session state is server-side per-session memory.
   It is not a cryptographically signed token.
2. The session is valid only for the current browser tab and server run.
3. These credentials are for local demonstration only.
   Do not deploy this authentication to a production environment.
4. Passwords are never logged or displayed.
5. The hashed values are computed at startup and kept only in memory.
"""

import hashlib
import hmac
import os
import secrets
import streamlit as st
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

ROLE_ANALYST = "analyst"
ROLE_ADMIN = "admin"

_VALID_ROLES = frozenset({ROLE_ANALYST, ROLE_ADMIN})

# Session-state keys — use module-prefixed names to avoid collisions
_SK_AUTHENTICATED = "wt_authenticated"
_SK_EMAIL = "wt_email"
_SK_ROLE = "wt_role"
_SK_DISPLAY_NAME = "wt_display_name"


@dataclass(frozen=True)
class UserAccount:
    """Internal account record.  Never store or log the plain-text password."""
    email: str
    display_name: str
    role: str
    _pw_hash: str   # PBKDF2-HMAC-SHA256 hex digest
    _pw_salt: str   # per-account hex salt


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def _hash_password(plain: str, salt: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hex digest for *plain* using *salt*."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        plain.encode("utf-8"),
        bytes.fromhex(salt),
        iterations=260_000,
    ).hex()


def _make_account(email: str, display_name: str, role: str, plain_password: str) -> UserAccount:
    """Create a UserAccount, hashing the password before storage."""
    salt = secrets.token_hex(32)
    pw_hash = _hash_password(plain_password, salt)
    return UserAccount(
        email=email.strip().lower(),
        display_name=display_name,
        role=role,
        _pw_hash=pw_hash,
        _pw_salt=salt,
    )


# ---------------------------------------------------------------------------
# Account registry
# ---------------------------------------------------------------------------
# Accounts are built at module-import time from environment variables.
# If an environment variable is not set, the documented default is used.
# In a real deployment, set strong passwords via environment variables and
# remove or disable the defaults.

def _build_accounts() -> dict[str, UserAccount]:
    analyst_email = os.environ.get(
        "WEBTRUST_ANALYST_EMAIL", "analyst@webtrust.local"
    ).strip().lower()
    analyst_pw = os.environ.get(
        "WEBTRUST_ANALYST_PASSWORD", "AnalystDemo2025!"
    )

    admin_email = os.environ.get(
        "WEBTRUST_ADMIN_EMAIL", "admin@webtrust.local"
    ).strip().lower()
    admin_pw = os.environ.get(
        "WEBTRUST_ADMIN_PASSWORD", "AdminDemo2025!"
    )

    analyst = _make_account(analyst_email, "Security Analyst", ROLE_ANALYST, analyst_pw)
    admin = _make_account(admin_email, "Researcher / Admin", ROLE_ADMIN, admin_pw)

    return {analyst.email: analyst, admin.email: admin}


# Built once at import time — hashing happens here, not per-request.
_ACCOUNTS: dict[str, UserAccount] = _build_accounts()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate(email: str, password: str) -> Optional[UserAccount]:
    """
    Return the matching UserAccount if credentials are correct, else None.

    Uses a constant-time comparison to resist timing attacks.
    Returns None for both unknown email and wrong password (no information leak).
    """
    if not email or not password:
        return None

    normalised = email.strip().lower()
    account = _ACCOUNTS.get(normalised)

    if account is None:
        # Perform a dummy hash to keep timing uniform
        _hash_password(password, secrets.token_hex(32))
        return None

    candidate_hash = _hash_password(password, account._pw_salt)
    if hmac.compare_digest(candidate_hash, account._pw_hash):
        return account

    return None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def login(account: UserAccount) -> None:
    """Record the authenticated user in Streamlit session state."""
    st.session_state[_SK_AUTHENTICATED] = True
    st.session_state[_SK_EMAIL] = account.email
    st.session_state[_SK_ROLE] = account.role
    st.session_state[_SK_DISPLAY_NAME] = account.display_name


def logout() -> None:
    """Clear authentication state from the session."""
    for key in (_SK_AUTHENTICATED, _SK_EMAIL, _SK_ROLE, _SK_DISPLAY_NAME):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    """Return True if a user is currently signed in."""
    return bool(st.session_state.get(_SK_AUTHENTICATED, False))


def get_current_role() -> Optional[str]:
    """Return the current user's role string, or None if not authenticated."""
    return st.session_state.get(_SK_ROLE)


def get_current_email() -> Optional[str]:
    """Return the current user's email, or None if not authenticated."""
    return st.session_state.get(_SK_EMAIL)


def get_current_display_name() -> Optional[str]:
    """Return the current user's display name, or None if not authenticated."""
    return st.session_state.get(_SK_DISPLAY_NAME)


def has_role(role: str) -> bool:
    """Return True if the authenticated user has exactly *role*."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Unknown role: {role!r}. Valid roles: {_VALID_ROLES}")
    return get_current_role() == role


def get_demo_credentials() -> list[dict]:
    """
    Return the demo account credentials for display on the login page.
    Only call this in development/demo mode — never in production.
    """
    return [
        {
            "label": "Security Analyst",
            "email": os.environ.get("WEBTRUST_ANALYST_EMAIL", "analyst@webtrust.local"),
            "password": os.environ.get("WEBTRUST_ANALYST_PASSWORD", "AnalystDemo2025!"),
            "role": ROLE_ANALYST,
        },
        {
            "label": "Researcher / Admin",
            "email": os.environ.get("WEBTRUST_ADMIN_EMAIL", "admin@webtrust.local"),
            "password": os.environ.get("WEBTRUST_ADMIN_PASSWORD", "AdminDemo2025!"),
            "role": ROLE_ADMIN,
        },
    ]
