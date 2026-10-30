"""
tests/test_security_analyzer.py
================================
Automated tests for Module 6 — Website Security Analysis.

Framework : Python built-in unittest
Run with  :
    py -m unittest tests.test_security_analyzer -v
    OR
    py -m unittest discover -s tests -v

Test philosophy
---------------
- No real external network requests are made.  All HTTP responses and TLS
  certificate data are provided through mocks.
- Sensitive cookie values are never expected to appear in results.
- Tests verify SSRF-blocking logic by asserting that prohibited addresses
  cause analysis to fail with an appropriate error.
- Every test assertion includes a short comment explaining its intent.

Test index
----------
 1.  All security headers present — header_checks show Present
 2.  All security headers missing — header_checks show Missing, findings generated
 3.  Mixed headers — present/missing correctly distinguished
 4.  HSTS on HTTP — not applicable, not an error
 5.  X-Content-Type-Options wrong value — Potential Issue observation
 6.  X-Frame-Options unknown value — Potential Issue observation
 7.  No cookies in response — cookies_present False, correct message
 8.  Cookie with all secure attributes — no observations
 9.  Cookie missing Secure, HttpOnly, SameSite — observations generated
10.  SameSite=None without Secure — observation generated
11.  Cookie value never appears in CookieCheckResult
12.  HTTPS destination — TLS check is applicable
13.  HTTP destination — TLS not applicable
14.  TLS certificate verification failure — error captured, verified=False
15.  TLS certificate expiring in 5 days — days_remaining and finding
16.  TLS certificate expired — verified=False, observation notes expiry
17.  DNS resolution failure — analysis fails with DNS error
18.  Invalid URL — ValueError raised
19.  Non-ValidationResult input — TypeError raised
20.  Blocked private address (192.168.x.x) — analysis fails, blocked reason
21.  Blocked loopback address (127.0.0.1) — analysis fails
22.  Redirect to prohibited address — analysis fails at redirect
23.  Too many redirects — analysis fails with redirect limit message
24.  Connection timeout — analysis fails with timeout message
25.  Findings list populated for missing headers
26.  Improvements list contains suggestions for missing headers
27.  Findings for TLS failure
28.  No phishing verdict assigned by analyze_security (no 'phishing' in findings)
29.  HTTP URL produces TLS not-applicable result
30.  status_code captured from response
31.  Regression: validate_url tests still pass (imported and run)
32.  Regression: extract_features tests still pass (imported and run)
33.  Redirect destination validated before following
34.  Cookie name extraction correct, value absent from result
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import ssl
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.url_validator import validate_url, ValidationResult
from modules.security_analyzer import (
    analyze_security,
    SecurityAnalysisResult,
    _analyse_headers,
    _analyse_cookies,
    _parse_set_cookie,
    _build_findings,
    _resolve_and_check,
    TLSCheckResult,
    HeaderCheckResult,
    CookieCheckResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_https_vr(url: str = "https://example.com") -> ValidationResult:
    """Return a valid ValidationResult for an HTTPS URL."""
    vr = validate_url(url)
    assert vr.is_valid, f"Expected valid URL, got: {vr.errors}"
    return vr


def _valid_http_vr(url: str = "http://example.com") -> ValidationResult:
    vr = validate_url(url)
    assert vr.is_valid, f"Expected valid URL, got: {vr.errors}"
    return vr


def _make_mock_response(
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock urllib response object."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    hdr_dict = headers or {}
    # Make resp.headers behave like a dict and be iterable
    resp.headers = MagicMock()
    resp.headers.__iter__ = MagicMock(return_value=iter(hdr_dict.keys()))
    resp.headers.__getitem__ = MagicMock(side_effect=lambda k: hdr_dict[k])
    resp.headers.get = MagicMock(side_effect=lambda k, d=None: hdr_dict.get(k, d))
    return resp


def _all_security_headers() -> dict[str, str]:
    """Return a dict with all expected security headers present."""
    return {
        "content-security-policy": "default-src 'self'",
        "strict-transport-security": "max-age=31536000; includeSubDomains",
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
        "referrer-policy": "strict-origin-when-cross-origin",
        "permissions-policy": "geolocation=()",
    }


# ---------------------------------------------------------------------------
# _analyse_headers tests (unit — no network)
# ---------------------------------------------------------------------------


class TestAnalyseHeaders(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test 1 — All headers present
    # -----------------------------------------------------------------------
    def test_01_all_headers_present(self):
        """When all security headers are present, every check should be 'present'."""
        results = _analyse_headers(_all_security_headers(), uses_https=True)
        for hc in results:
            if hc.applicable:
                self.assertTrue(hc.present, msg=f"{hc.name} should be present")
                # No improvement suggestion when the value is correct
                self.assertEqual(
                    hc.suggestion, "",
                    msg=f"{hc.name} should have no suggestion when correctly set",
                )

    # -----------------------------------------------------------------------
    # Test 2 — All headers missing
    # -----------------------------------------------------------------------
    def test_02_all_headers_missing(self):
        """When all security headers are absent, every check should report Missing."""
        results = _analyse_headers({}, uses_https=True)
        for hc in results:
            if hc.applicable:
                self.assertFalse(hc.present, msg=f"{hc.name} should not be present")
                self.assertNotEqual(hc.suggestion, "", msg=f"{hc.name} should have a suggestion")

    # -----------------------------------------------------------------------
    # Test 3 — Mixed headers
    # -----------------------------------------------------------------------
    def test_03_mixed_headers(self):
        """Some present, some absent — each is checked independently."""
        headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
        }
        results = _analyse_headers(headers, uses_https=True)
        by_name = {hc.name: hc for hc in results}

        # Present headers
        self.assertTrue(by_name["X-Content-Type-Options"].present)
        self.assertTrue(by_name["X-Frame-Options"].present)
        # Absent headers
        self.assertFalse(by_name["Content-Security-Policy"].present)
        self.assertFalse(by_name["Strict-Transport-Security"].present)

    # -----------------------------------------------------------------------
    # Test 4 — HSTS on HTTP is Not Applicable
    # -----------------------------------------------------------------------
    def test_04_hsts_not_applicable_on_http(self):
        """HSTS must be marked not applicable for plain HTTP destinations."""
        results = _analyse_headers({}, uses_https=False)
        by_name = {hc.name: hc for hc in results}
        hsts = by_name["Strict-Transport-Security"]
        self.assertFalse(hsts.applicable, msg="HSTS must not be applicable over HTTP")
        combined = (hsts.not_applicable_reason + hsts.observation).lower()
        self.assertIn("plain http", combined)

    # -----------------------------------------------------------------------
    # Test 5 — X-Content-Type-Options wrong value
    # -----------------------------------------------------------------------
    def test_05_xcto_wrong_value(self):
        """X-Content-Type-Options with a non-'nosniff' value should flag Potential Issue."""
        results = _analyse_headers({"x-content-type-options": "something-else"}, uses_https=True)
        by_name = {hc.name: hc for hc in results}
        xcto = by_name["X-Content-Type-Options"]
        self.assertTrue(xcto.present)
        self.assertNotEqual(xcto.suggestion, "", msg="Should have a suggestion")
        self.assertIn("nosniff", xcto.suggestion.lower())

    # -----------------------------------------------------------------------
    # Test 6 — X-Frame-Options unknown value
    # -----------------------------------------------------------------------
    def test_06_xfo_unknown_value(self):
        """X-Frame-Options with a non-standard value should flag Potential Issue."""
        results = _analyse_headers({"x-frame-options": "ALLOWALL"}, uses_https=True)
        by_name = {hc.name: hc for hc in results}
        xfo = by_name["X-Frame-Options"]
        self.assertTrue(xfo.present)
        self.assertNotEqual(xfo.suggestion, "")


# ---------------------------------------------------------------------------
# _analyse_cookies tests (unit — no network)
# ---------------------------------------------------------------------------


class TestAnalyseCookies(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test 7 — No cookies in response
    # -----------------------------------------------------------------------
    def test_07_no_cookies(self):
        """When no Set-Cookie header is present, cookies_present must be False."""
        results, present = _analyse_cookies({})
        self.assertFalse(present, msg="cookies_present should be False when no Set-Cookie")
        self.assertEqual(results, [])

    # -----------------------------------------------------------------------
    # Test 8 — Cookie with all secure attributes
    # -----------------------------------------------------------------------
    def test_08_cookie_all_secure(self):
        """A cookie with Secure, HttpOnly, SameSite=Strict should have no observations."""
        raw = "session=REDACTED; Secure; HttpOnly; SameSite=Strict; Path=/"
        result = _parse_set_cookie(raw)
        self.assertIsNotNone(result)
        self.assertTrue(result.has_secure)
        self.assertTrue(result.has_httponly)
        self.assertEqual(result.samesite, "Strict")
        # No security observations expected
        self.assertEqual(result.observations, [], msg="Secure cookie should have no observations")

    # -----------------------------------------------------------------------
    # Test 9 — Cookie missing Secure, HttpOnly, SameSite
    # -----------------------------------------------------------------------
    def test_09_cookie_missing_attributes(self):
        """A cookie missing Secure, HttpOnly, and SameSite should generate observations."""
        raw = "prefs=something"
        result = _parse_set_cookie(raw)
        self.assertIsNotNone(result)
        self.assertFalse(result.has_secure)
        self.assertFalse(result.has_httponly)
        self.assertIsNone(result.samesite)
        # Three separate observations expected
        self.assertEqual(len(result.observations), 3,
                         msg=f"Expected 3 observations, got: {result.observations}")

    # -----------------------------------------------------------------------
    # Test 10 — SameSite=None without Secure
    # -----------------------------------------------------------------------
    def test_10_samesite_none_without_secure(self):
        """SameSite=None without Secure must generate an observation."""
        raw = "cross=abc; SameSite=None"
        result = _parse_set_cookie(raw)
        self.assertIsNotNone(result)
        self.assertFalse(result.has_secure)
        # Should mention SameSite=None requires Secure
        samesite_obs = [o for o in result.observations if "samesite=none" in o.lower()]
        self.assertTrue(len(samesite_obs) > 0, msg=f"Expected SameSite=None observation. Got: {result.observations}")

    # -----------------------------------------------------------------------
    # Test 11 — Cookie value never appears in result
    # -----------------------------------------------------------------------
    def test_11_cookie_value_not_in_result(self):
        """The cookie value must never appear in the CookieCheckResult."""
        sensitive_value = "super-secret-session-token-12345"
        raw = f"auth={sensitive_value}; Secure; HttpOnly; SameSite=Lax"
        result = _parse_set_cookie(raw)
        self.assertIsNotNone(result)
        # The name is safe to display; the value must not appear
        self.assertNotIn(sensitive_value, result.name,
                         msg="Sensitive cookie value must not appear in the cookie name field")
        # Verify no field contains the sensitive value
        for obs in result.observations:
            self.assertNotIn(sensitive_value, obs,
                             msg="Sensitive cookie value must not appear in observations")


# ---------------------------------------------------------------------------
# _resolve_and_check tests (unit — mocked DNS)
# ---------------------------------------------------------------------------


class TestResolveAndCheck(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test 20 — Blocked private address
    # -----------------------------------------------------------------------
    def test_20_blocked_private_address(self):
        """A hostname resolving to a private IP must be blocked."""
        with patch("modules.security_analyzer.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.100", 443))
            ]
            ip, err = _resolve_and_check("internal.example.local", 443)
            # Expect a block error
            self.assertEqual(ip, "", msg="Should return empty IP for blocked address")
            self.assertIn("prohibited", err.lower(), msg=f"Expected blocked reason, got: {err}")

    # -----------------------------------------------------------------------
    # Test 21 — Blocked loopback address
    # -----------------------------------------------------------------------
    def test_21_blocked_loopback_address(self):
        """A hostname resolving to loopback must be blocked."""
        with patch("modules.security_analyzer.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 80))
            ]
            ip, err = _resolve_and_check("localhost.attacker.com", 80)
            self.assertEqual(ip, "")
            self.assertIn("prohibited", err.lower())

    # -----------------------------------------------------------------------
    # Test 17 — DNS resolution failure
    # -----------------------------------------------------------------------
    def test_17_dns_failure(self):
        """A DNS resolution failure must return an empty IP and an error message."""
        with patch("modules.security_analyzer.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name or service not known")
            ip, err = _resolve_and_check("nonexistent.example.invalid", 443)
            self.assertEqual(ip, "")
            self.assertIn("DNS resolution failed", err)


# ---------------------------------------------------------------------------
# analyze_security integration tests (mocked network)
# ---------------------------------------------------------------------------


class TestAnalyzeSecurity(unittest.TestCase):

    def _patch_fetch(self, status: int = 200, headers: dict | None = None):
        """Context manager: patch _fetch_headers_safe to return fixed headers."""
        if headers is None:
            headers = _all_security_headers()
        return patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(status, headers, ""),
        )

    def _patch_tls(self, result: TLSCheckResult | None = None):
        """Context manager: patch _inspect_tls_certificate."""
        if result is None:
            result = TLSCheckResult(
                applicable=True,
                verified=True,
                subject="example.com",
                issuer="Test CA",
                valid_from="2024-01-01",
                valid_until="2026-01-01",
                hostname_match=True,
                days_remaining=365,
                observation="Certificate is valid and verified.",
            )
        return patch("modules.security_analyzer._inspect_tls_certificate", return_value=result)

    # -----------------------------------------------------------------------
    # Test 18 — Invalid URL (non-ValidationResult)
    # -----------------------------------------------------------------------
    def test_18_invalid_input_type(self):
        """Passing a non-ValidationResult must raise TypeError."""
        with self.assertRaises(TypeError):
            analyze_security("https://example.com")

    # -----------------------------------------------------------------------
    # Test 19 — ValidationResult with is_valid=False
    # -----------------------------------------------------------------------
    def test_19_invalid_vr(self):
        """Passing a ValidationResult with is_valid=False must raise ValueError."""
        vr = validate_url("not a valid url")
        self.assertFalse(vr.is_valid)
        with self.assertRaises(ValueError):
            analyze_security(vr)

    # -----------------------------------------------------------------------
    # Test 12 — HTTPS destination — TLS check is applicable
    # -----------------------------------------------------------------------
    def test_12_https_tls_applicable(self):
        """For an HTTPS URL, TLS inspection must be performed."""
        vr = _valid_https_vr()
        with self._patch_fetch(), self._patch_tls() as mock_tls:
            result = analyze_security(vr)
        self.assertTrue(result.success)
        self.assertTrue(result.tls.applicable, msg="TLS should be applicable for HTTPS")
        mock_tls.assert_called_once()

    # -----------------------------------------------------------------------
    # Test 13 — HTTP destination — TLS not applicable
    # -----------------------------------------------------------------------
    def test_13_http_tls_not_applicable(self):
        """For a plain HTTP URL, TLS inspection must not be performed."""
        vr = _valid_http_vr()
        with self._patch_fetch(headers={}), \
             patch("modules.security_analyzer._inspect_tls_certificate") as mock_tls:
            result = analyze_security(vr)
        self.assertTrue(result.success)
        self.assertFalse(result.tls.applicable, msg="TLS should not apply for HTTP")
        mock_tls.assert_not_called()

    # -----------------------------------------------------------------------
    # Test 30 — status_code captured
    # -----------------------------------------------------------------------
    def test_30_status_code_captured(self):
        """The HTTP status code must be stored in the result."""
        vr = _valid_https_vr()
        with self._patch_fetch(status=200), self._patch_tls():
            result = analyze_security(vr)
        self.assertEqual(result.status_code, 200)

    # -----------------------------------------------------------------------
    # Test 25 — Findings generated for missing headers
    # -----------------------------------------------------------------------
    def test_25_findings_for_missing_headers(self):
        """Missing security headers must generate findings."""
        vr = _valid_https_vr()
        with self._patch_fetch(headers={}), self._patch_tls() as _:
            result = analyze_security(vr)
        # Should have findings for each missing applicable header
        self.assertTrue(len(result.findings) > 0, msg="Expected findings for missing headers")
        # Each finding should have a non-empty title
        for f in result.findings:
            self.assertNotEqual(f.title, "")

    # -----------------------------------------------------------------------
    # Test 26 — Improvements for missing headers
    # -----------------------------------------------------------------------
    def test_26_improvements_for_missing_headers(self):
        """Missing headers must contribute to the improvements list."""
        vr = _valid_https_vr()
        with self._patch_fetch(headers={}), self._patch_tls() as _:
            result = analyze_security(vr)
        self.assertTrue(len(result.improvements) > 0,
                        msg="Expected at least one improvement suggestion")

    # -----------------------------------------------------------------------
    # Test 28 — No phishing verdict
    # -----------------------------------------------------------------------
    def test_28_no_phishing_verdict(self):
        """analyze_security must never assign a phishing verdict."""
        vr = _valid_https_vr()
        with self._patch_fetch(headers={}), self._patch_tls():
            result = analyze_security(vr)
        for f in result.findings:
            for text in (f.title, f.observed, f.why_it_matters, f.remediation):
                self.assertNotIn(
                    "phishing",
                    text.lower(),
                    msg=f"Finding '{f.title}' contains a phishing verdict: {text}",
                )

    # -----------------------------------------------------------------------
    # Test 14 — TLS certificate verification failure
    # -----------------------------------------------------------------------
    def test_14_tls_verification_failure(self):
        """TLS verification failure must be captured and result in verified=False."""
        vr = _valid_https_vr()
        failed_tls = TLSCheckResult(
            applicable=True,
            verified=False,
            error="TLS certificate verification failed: certificate has expired",
            observation="TLS certificate verification failed.",
        )
        with self._patch_fetch(), \
             patch("modules.security_analyzer._inspect_tls_certificate", return_value=failed_tls):
            result = analyze_security(vr)
        self.assertTrue(result.success)
        self.assertFalse(result.tls.verified)
        self.assertIn("failed", result.tls.error.lower())

    # -----------------------------------------------------------------------
    # Test 27 — Finding generated for TLS failure
    # -----------------------------------------------------------------------
    def test_27_finding_for_tls_failure(self):
        """A TLS verification failure must generate a high-severity finding."""
        vr = _valid_https_vr()
        failed_tls = TLSCheckResult(
            applicable=True,
            verified=False,
            error="TLS certificate verification failed",
            observation="Failed.",
        )
        with self._patch_fetch(), \
             patch("modules.security_analyzer._inspect_tls_certificate", return_value=failed_tls):
            result = analyze_security(vr)
        tls_findings = [f for f in result.findings if "tls" in f.title.lower() or "certificate" in f.title.lower()]
        self.assertTrue(len(tls_findings) > 0, msg="Expected a TLS finding")
        self.assertEqual(tls_findings[0].severity, "high")

    # -----------------------------------------------------------------------
    # Test 15 — TLS expiring in 5 days
    # -----------------------------------------------------------------------
    def test_15_tls_expiring_soon(self):
        """A certificate expiring in 5 days must set days_remaining and generate a finding."""
        vr = _valid_https_vr()
        expiring_tls = TLSCheckResult(
            applicable=True,
            verified=True,
            subject="example.com",
            issuer="Test CA",
            valid_from="2024-01-01",
            valid_until="2025-01-06",
            hostname_match=True,
            days_remaining=5,
            observation="Certificate expires in 5 day(s). Renewal is recommended soon.",
        )
        with self._patch_fetch(), \
             patch("modules.security_analyzer._inspect_tls_certificate", return_value=expiring_tls):
            result = analyze_security(vr)
        self.assertEqual(result.tls.days_remaining, 5)
        expiry_findings = [f for f in result.findings if "expir" in f.title.lower()]
        self.assertTrue(len(expiry_findings) > 0, msg="Expected a certificate expiry finding")

    # -----------------------------------------------------------------------
    # Test 16 — TLS certificate expired
    # -----------------------------------------------------------------------
    def test_16_tls_expired(self):
        """An expired certificate must result in verified=False."""
        vr = _valid_https_vr()
        expired_tls = TLSCheckResult(
            applicable=True,
            verified=False,
            subject="example.com",
            issuer="Test CA",
            valid_from="2020-01-01",
            valid_until="2021-01-01",
            hostname_match=False,
            days_remaining=-100,
            error="TLS certificate verification failed: certificate has expired",
            observation="Certificate has expired.",
        )
        with self._patch_fetch(), \
             patch("modules.security_analyzer._inspect_tls_certificate", return_value=expired_tls):
            result = analyze_security(vr)
        self.assertFalse(result.tls.verified)
        self.assertLess(result.tls.days_remaining, 0)

    # -----------------------------------------------------------------------
    # Test 17 — Analysis fails when DNS fails
    # -----------------------------------------------------------------------
    def test_17_dns_failure_causes_analysis_failure(self):
        """DNS failure for the initial destination must produce success=False."""
        vr = _valid_https_vr()
        with patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(None, {}, "DNS resolution failed for 'example.com': [Errno 11001]"),
        ):
            result = analyze_security(vr)
        self.assertFalse(result.success)
        self.assertIn("DNS", result.error)

    # -----------------------------------------------------------------------
    # Test 22 — Redirect to prohibited address blocked
    # -----------------------------------------------------------------------
    def test_22_redirect_to_prohibited_blocked(self):
        """A redirect leading to a private address must cause analysis to fail."""
        vr = _valid_https_vr()
        with patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(
                None,
                {},
                "Redirect destination blocked: Destination 'redirected.example.com' "
                "resolves to a prohibited address (private address (192.168.0.1)).",
            ),
        ):
            result = analyze_security(vr)
        self.assertFalse(result.success)
        self.assertIn("blocked", result.error.lower())

    # -----------------------------------------------------------------------
    # Test 23 — Too many redirects
    # -----------------------------------------------------------------------
    def test_23_too_many_redirects(self):
        """Exceeding the redirect limit must cause analysis to fail."""
        vr = _valid_https_vr()
        with patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(
                301,
                {},
                "Too many redirects (limit 5). Analysis stopped at redirect boundary.",
            ),
        ):
            result = analyze_security(vr)
        self.assertFalse(result.success)
        self.assertIn("redirect", result.error.lower())

    # -----------------------------------------------------------------------
    # Test 24 — Connection timeout
    # -----------------------------------------------------------------------
    def test_24_connection_timeout(self):
        """A connection timeout must produce success=False with a timeout message."""
        vr = _valid_https_vr()
        with patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(None, {}, "Connection to 'https://example.com' timed out after 18s."),
        ):
            result = analyze_security(vr)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error.lower())

    # -----------------------------------------------------------------------
    # Test 29 — HTTP URL TLS not applicable
    # -----------------------------------------------------------------------
    def test_29_http_url_tls_not_applicable(self):
        """For HTTP URLs the TLS result must have applicable=False."""
        vr = _valid_http_vr()
        with self._patch_fetch(headers={}), \
             patch("modules.security_analyzer._inspect_tls_certificate") as mock_tls:
            result = analyze_security(vr)
        self.assertFalse(result.tls.applicable)
        mock_tls.assert_not_called()

    # -----------------------------------------------------------------------
    # Test 33 — Redirect destination validated via validate_url
    # -----------------------------------------------------------------------
    def test_33_redirect_validates_destination(self):
        """
        Redirect destination is validated through validate_url; an invalid
        destination causes analysis to fail with a validation error.
        """
        vr = _valid_https_vr()
        with patch(
            "modules.security_analyzer._fetch_headers_safe",
            return_value=(
                None,
                {},
                "Redirect destination 'javascript:alert(1)' failed URL validation: "
                "URL scheme 'javascript' is not supported.",
            ),
        ):
            result = analyze_security(vr)
        self.assertFalse(result.success)

    # -----------------------------------------------------------------------
    # Test 34 — Cookie name correct, value absent
    # -----------------------------------------------------------------------
    def test_34_cookie_name_correct_value_absent(self):
        """Cookie parsing returns the correct name; the value is never stored."""
        raw = "mycookie=MY-VERY-SECRET-VALUE; Secure; HttpOnly; SameSite=Lax"
        ck = _parse_set_cookie(raw)
        self.assertIsNotNone(ck)
        self.assertEqual(ck.name, "mycookie")
        # The value should not be accessible anywhere in the result
        import dataclasses
        result_dict = dataclasses.asdict(ck)
        for field_val in result_dict.values():
            if isinstance(field_val, str):
                self.assertNotIn(
                    "MY-VERY-SECRET-VALUE",
                    field_val,
                    msg=f"Cookie value found in field: {field_val}",
                )
            elif isinstance(field_val, list):
                for item in field_val:
                    if isinstance(item, str):
                        self.assertNotIn(
                            "MY-VERY-SECRET-VALUE",
                            item,
                            msg=f"Cookie value found in list field: {item}",
                        )

    # -----------------------------------------------------------------------
    # Test 8 (integration) — Cookie with secure attributes — no findings
    # -----------------------------------------------------------------------
    def test_08_secure_cookie_no_findings(self):
        """A cookie with all recommended attributes should not generate cookie findings."""
        vr = _valid_https_vr()
        headers = dict(_all_security_headers())
        headers["set-cookie"] = "session=REDACTED; Secure; HttpOnly; SameSite=Strict; Path=/"
        with self._patch_fetch(headers=headers), self._patch_tls():
            result = analyze_security(vr)
        self.assertTrue(result.cookies_present)
        cookie_findings = [
            f for f in result.findings if "cookie" in f.title.lower()
        ]
        # A fully-secure cookie should not contribute findings
        self.assertEqual(
            cookie_findings, [],
            msg=f"Unexpected cookie findings: {[f.title for f in cookie_findings]}",
        )

    # -----------------------------------------------------------------------
    # Test 9 (integration) — Cookie missing attributes generates findings
    # -----------------------------------------------------------------------
    def test_09_insecure_cookie_generates_findings(self):
        """A cookie missing Secure and HttpOnly must contribute findings."""
        vr = _valid_https_vr()
        headers = dict(_all_security_headers())
        headers["set-cookie"] = "prefs=something"
        with self._patch_fetch(headers=headers), self._patch_tls():
            result = analyze_security(vr)
        self.assertTrue(result.cookies_present)
        cookie_findings = [
            f for f in result.findings if "cookie" in f.title.lower()
        ]
        self.assertTrue(len(cookie_findings) > 0, msg="Expected cookie attribute findings")


# ---------------------------------------------------------------------------
# Regression: existing module tests still pass
# ---------------------------------------------------------------------------


class TestRegressionUrlValidator(unittest.TestCase):
    """
    Test 31 — Run a representative selection of url_validator cases to verify
    Module 6 changes did not break Module 1.
    """

    def test_31a_valid_https(self):
        r = validate_url("https://example.com")
        self.assertTrue(r.is_valid)

    def test_31b_blocked_loopback(self):
        r = validate_url("https://127.0.0.1/admin")
        self.assertFalse(r.is_valid)
        self.assertEqual(r.destination_status, "blocked")

    def test_31c_blocked_private(self):
        r = validate_url("https://192.168.1.1")
        self.assertFalse(r.is_valid)
        self.assertEqual(r.destination_status, "blocked")

    def test_31d_unsupported_scheme(self):
        r = validate_url("ftp://example.com")
        self.assertFalse(r.is_valid)

    def test_31e_http_warning(self):
        r = validate_url("http://example.com")
        self.assertTrue(r.is_valid)
        self.assertTrue(any("plain HTTP" in w for w in r.warnings))


class TestRegressionFeatureExtractor(unittest.TestCase):
    """
    Test 32 — Run a representative selection of feature extractor cases to
    verify Module 6 changes did not break Module 5.
    """

    def test_32a_basic_https(self):
        from modules.feature_extractor import extract_features
        vr = validate_url("https://example.com/path?q=1")
        self.assertTrue(vr.is_valid)
        fr = extract_features(vr)
        self.assertTrue(fr.numeric["flag_uses_https"])
        self.assertEqual(fr.numeric["url_num_query_params"], 1)

    def test_32b_keyword_detection(self):
        from modules.feature_extractor import extract_features
        vr = validate_url("https://bank.example.com/login")
        fr = extract_features(vr)
        self.assertIn("login", fr.keywords_matched)

    def test_32c_invalid_vr_raises(self):
        from modules.feature_extractor import extract_features
        vr = validate_url("not-a-url-at-all")
        self.assertFalse(vr.is_valid)
        with self.assertRaises(ValueError):
            extract_features(vr)


# ---------------------------------------------------------------------------
# _build_findings unit tests
# ---------------------------------------------------------------------------


class TestBuildFindings(unittest.TestCase):

    def _header_checks_all_missing(self) -> list[HeaderCheckResult]:
        """Return header checks with all applicable headers missing."""
        checks = []
        for hdef in [
            ("Content-Security-Policy", "content-security-policy"),
            ("X-Content-Type-Options", "x-content-type-options"),
            ("X-Frame-Options", "x-frame-options"),
        ]:
            checks.append(HeaderCheckResult(
                name=hdef[0],
                present=False,
                value=None,
                explanation=f"Explanation for {hdef[0]}",
                observation=f"{hdef[0]} is absent.",
                suggestion=f"Add {hdef[0]}.",
                applicable=True,
            ))
        return checks

    def test_findings_deduplicated(self):
        """Improvements list should not contain exact duplicates."""
        hc = self._header_checks_all_missing()
        findings, improvements = _build_findings(
            hc, [], TLSCheckResult(applicable=False)
        )
        # Check for duplicates
        seen = set()
        for item in improvements:
            self.assertNotIn(item, seen, msg=f"Duplicate improvement: {item}")
            seen.add(item)

    def test_no_findings_for_https_not_applicable_tls(self):
        """TLS with applicable=False must not generate findings."""
        findings, _ = _build_findings(
            [], [], TLSCheckResult(applicable=False)
        )
        tls_findings = [f for f in findings if "tls" in f.title.lower()]
        self.assertEqual(tls_findings, [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
