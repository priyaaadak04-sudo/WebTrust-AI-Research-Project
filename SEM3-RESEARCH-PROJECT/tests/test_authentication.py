"""
tests/test_authentication.py
=============================
Automated tests for Module 2 — Authentication layer.

Framework : Python built-in unittest
Run with  : py -m unittest discover -s tests -v
            OR
            py -m unittest tests.test_authentication -v

Test philosophy
---------------
- No real network requests.
- No Streamlit runtime required (session-state functions are not tested here;
  they depend on a running Streamlit app and are tested manually).
- authenticate() and password utilities are pure-Python and fully testable.
- Credentials are built from the same environment-variable logic as the app,
  so tests automatically use the same accounts.
"""

import os
import unittest

from modules.authentication import (
    authenticate,
    _hash_password,
    _make_account,
    get_demo_credentials,
    ROLE_ANALYST,
    ROLE_ADMIN,
)


class TestPasswordHashing(unittest.TestCase):
    """Tests for the password-hashing and verification utilities."""

    def test_hash_is_deterministic(self):
        """Same password + same salt must always produce the same digest."""
        salt = "a" * 64   # 32-byte hex salt
        h1 = _hash_password("TestPassword1!", salt)
        h2 = _hash_password("TestPassword1!", salt)
        self.assertEqual(h1, h2, "Hashing must be deterministic given the same inputs.")

    def test_different_salt_produces_different_hash(self):
        """Different salts must produce different digests for the same password."""
        h1 = _hash_password("TestPassword1!", "a" * 64)
        h2 = _hash_password("TestPassword1!", "b" * 64)
        self.assertNotEqual(h1, h2, "Different salts must produce different hashes.")

    def test_different_password_produces_different_hash(self):
        """Different passwords must produce different digests with the same salt."""
        salt = "c" * 64
        h1 = _hash_password("PasswordOne!", salt)
        h2 = _hash_password("PasswordTwo!", salt)
        self.assertNotEqual(h1, h2, "Different passwords must hash differently.")

    def test_hash_output_is_hex_string(self):
        """The hash output must be a lowercase hex string."""
        result = _hash_password("any", "d" * 64)
        self.assertIsInstance(result, str)
        self.assertTrue(
            all(c in "0123456789abcdef" for c in result),
            "Hash must be a lowercase hex string.",
        )


class TestMakeAccount(unittest.TestCase):
    """Tests for UserAccount construction."""

    def test_account_role_stored(self):
        account = _make_account("test@example.com", "Test User", ROLE_ANALYST, "SomePassword1!")
        self.assertEqual(account.role, ROLE_ANALYST)

    def test_email_lowercased(self):
        account = _make_account("Test@Example.COM", "Test User", ROLE_ANALYST, "SomePassword1!")
        self.assertEqual(account.email, "test@example.com")

    def test_password_not_stored_in_plain(self):
        """The plain password must not appear in the stored account."""
        plain = "VerySecretPassword999!"
        account = _make_account("u@example.com", "U", ROLE_ANALYST, plain)
        # Check all string attributes
        for field_name in ("email", "display_name", "role", "_pw_hash", "_pw_salt"):
            self.assertNotEqual(
                getattr(account, field_name),
                plain,
                f"Plain password must not be stored in field {field_name!r}.",
            )


class TestAuthenticate(unittest.TestCase):
    """Tests for the authenticate() function using real account records."""

    def setUp(self):
        """Retrieve demo credentials to use in tests (matches app configuration)."""
        self.demo = get_demo_credentials()
        self.analyst_creds = next(c for c in self.demo if c["role"] == ROLE_ANALYST)
        self.admin_creds = next(c for c in self.demo if c["role"] == ROLE_ADMIN)

    def test_analyst_correct_credentials(self):
        """Correct analyst credentials must return the analyst account."""
        result = authenticate(self.analyst_creds["email"], self.analyst_creds["password"])
        self.assertIsNotNone(result, "Valid analyst credentials must authenticate.")
        self.assertEqual(result.role, ROLE_ANALYST)

    def test_admin_correct_credentials(self):
        """Correct admin credentials must return the admin account."""
        result = authenticate(self.admin_creds["email"], self.admin_creds["password"])
        self.assertIsNotNone(result, "Valid admin credentials must authenticate.")
        self.assertEqual(result.role, ROLE_ADMIN)

    def test_wrong_password_returns_none(self):
        """A wrong password must return None — not raise an exception."""
        result = authenticate(self.analyst_creds["email"], "WrongPassword999!")
        self.assertIsNone(result, "Wrong password must return None.")

    def test_wrong_email_returns_none(self):
        """An unknown email must return None."""
        result = authenticate("nobody@nonexistent.example", "SomePassword1!")
        self.assertIsNone(result, "Unknown email must return None.")

    def test_empty_email_returns_none(self):
        """Empty email string must return None without raising."""
        result = authenticate("", "SomePassword1!")
        self.assertIsNone(result, "Empty email must return None.")

    def test_empty_password_returns_none(self):
        """Empty password string must return None without raising."""
        result = authenticate(self.analyst_creds["email"], "")
        self.assertIsNone(result, "Empty password must return None.")

    def test_none_inputs_return_none(self):
        """None inputs must return None without raising a TypeError."""
        result = authenticate(None, None)  # type: ignore[arg-type]
        self.assertIsNone(result, "None inputs must return None.")

    def test_email_case_insensitive(self):
        """Email matching must be case-insensitive."""
        email_upper = self.analyst_creds["email"].upper()
        result = authenticate(email_upper, self.analyst_creds["password"])
        self.assertIsNotNone(
            result,
            "Email matching should be case-insensitive (uppercased email must still work).",
        )

    def test_analyst_cannot_get_admin_role(self):
        """An analyst account must never return the admin role."""
        result = authenticate(self.analyst_creds["email"], self.analyst_creds["password"])
        self.assertIsNotNone(result)
        self.assertNotEqual(
            result.role, ROLE_ADMIN,
            "An analyst account must not be returned with the admin role.",
        )

    def test_admin_cannot_get_analyst_role(self):
        """An admin account must not be returned with the analyst role."""
        result = authenticate(self.admin_creds["email"], self.admin_creds["password"])
        self.assertIsNotNone(result)
        self.assertNotEqual(
            result.role, ROLE_ANALYST,
            "An admin account must not be returned with the analyst role.",
        )

    def test_cross_credential_swap_fails(self):
        """Using analyst email with admin password must fail."""
        result = authenticate(self.analyst_creds["email"], self.admin_creds["password"])
        self.assertIsNone(
            result,
            "Swapping credentials across accounts must not authenticate.",
        )

    def test_password_is_not_leaked_in_return_value(self):
        """The returned UserAccount must not contain the plain-text password."""
        plain = self.analyst_creds["password"]
        result = authenticate(self.analyst_creds["email"], plain)
        self.assertIsNotNone(result)
        for field_name in ("email", "display_name", "role", "_pw_hash", "_pw_salt"):
            self.assertNotEqual(
                getattr(result, field_name),
                plain,
                f"Plain password must not appear in returned field {field_name!r}.",
            )


class TestRoleSeparation(unittest.TestCase):
    """Tests that verify role constants are distinct and correctly named."""

    def test_role_constants_are_distinct(self):
        self.assertNotEqual(ROLE_ANALYST, ROLE_ADMIN, "Role constants must be distinct.")

    def test_role_constants_are_strings(self):
        self.assertIsInstance(ROLE_ANALYST, str)
        self.assertIsInstance(ROLE_ADMIN, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
