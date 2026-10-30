"""
tests/test_url_validator.py
===========================
Automated tests for Module 1 — URL Input and Validation.

Framework : Python built-in unittest
Run with  : py -m unittest discover -s tests -v
           OR
           py -m unittest tests.test_url_validator -v

Test philosophy
---------------
- No real network requests are made.
- No real DNS lookups are made in the core validate_url() tests.
- check_dns_destination() is tested using unittest.mock.patch to intercept
  socket.getaddrinfo so that no live DNS queries are issued.
- Every test assertion is accompanied by a short comment explaining the
  security or functional intent so that reviewers understand the purpose.

Test index (26 required scenarios + extras)
-------------------------------------------
 1. Valid HTTPS URL
 2. Valid HTTP URL
 3. URL containing a path
 4. URL containing query parameters
 5. URL containing a fragment
 6. URL without a scheme (bare hostname — should infer https://)
 7. Empty input
 8. Whitespace-only input
 9. Malformed URL (causes hostname to be missing)
10. Unsupported URL scheme (ftp://)
11. Unsupported URL scheme (javascript:)
12. Missing hostname
13. Invalid port (non-numeric in netloc)
14. Port outside valid range (0)
15. Port outside valid range (65536)
16. Embedded credentials (username:password@)
17. Direct IPv4 loopback address (127.0.0.1)
18. Direct IPv4 private address (192.168.1.1)
19. Direct IPv6 loopback address (::1)
20. Direct IPv6 link-local address (fe80::1)
21. Public IP address (8.8.8.8) — not blocked, warning issued
22. Unverified hostname — is_valid True, destination_status not_checked
23. Internationalized domain name (IDN)
24. Hostname with a trailing dot
25. Malformed hostname (contains spaces after parse)
26. DNS check — hostname that cannot be resolved (mock)
27. DNS check — hostname that resolves to a prohibited address (mock)
28. DNS check — hostname with mixed public and prohibited results (mock)
29. Blocked scheme: file://
30. Blocked scheme: data:
31. Valid URL with explicit port
32. Valid URL — is_ip_address False for a regular hostname
33. IPv4-mapped IPv6 address treated as private IPv4
34. Port 443 explicit on HTTPS URL
35. URL with no path (root implied)
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that 'modules' is importable
# when tests are run from the project root with:
#   py -m unittest discover -s tests -v
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.url_validator import validate_url, check_dns_destination


# ===========================================================================
# Helper
# ===========================================================================

def _make_addr_info(ip_str: str) -> tuple:
    """Build a minimal getaddrinfo-style tuple for a given IP string."""
    import socket
    if ":" in ip_str:
        family = socket.AF_INET6
        sockaddr = (ip_str, 0, 0, 0)
    else:
        family = socket.AF_INET
        sockaddr = (ip_str, 0)
    return (family, socket.SOCK_STREAM, 0, "", sockaddr)


# ===========================================================================
# Tests — validate_url()
# ===========================================================================


class TestValidateUrl(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test 1 — Valid HTTPS URL
    # -----------------------------------------------------------------------
    def test_01_valid_https_url(self):
        """A standard HTTPS URL must be accepted and marked is_valid=True."""
        result = validate_url("https://example.com")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.scheme, "https")
        self.assertEqual(result.hostname, "example.com")
        self.assertTrue(result.uses_https)
        self.assertFalse(result.is_ip_address)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.destination_status, "not_checked")

    # -----------------------------------------------------------------------
    # Test 2 — Valid HTTP URL
    # -----------------------------------------------------------------------
    def test_02_valid_http_url(self):
        """An HTTP URL must be accepted; a warning about no encryption is expected."""
        result = validate_url("http://example.com")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.scheme, "http")
        self.assertFalse(result.uses_https)
        # Must warn about unencrypted transport.
        self.assertTrue(
            any("plain HTTP" in w for w in result.warnings),
            msg=f"Expected HTTP warning. Warnings: {result.warnings}",
        )

    # -----------------------------------------------------------------------
    # Test 3 — URL with a path
    # -----------------------------------------------------------------------
    def test_03_url_with_path(self):
        """Path component must be extracted correctly."""
        result = validate_url("https://example.com/login/dashboard")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.path, "/login/dashboard")

    # -----------------------------------------------------------------------
    # Test 4 — URL with query parameters
    # -----------------------------------------------------------------------
    def test_04_url_with_query_parameters(self):
        """Query string must be extracted.  Sensitive data — do not log it."""
        result = validate_url("https://example.com/search?q=phishing&page=2")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.query, "q=phishing&page=2")

    # -----------------------------------------------------------------------
    # Test 5 — URL with a fragment
    # -----------------------------------------------------------------------
    def test_05_url_with_fragment(self):
        """Fragment identifier must be extracted."""
        result = validate_url("https://example.com/page#section1")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.fragment, "section1")

    # -----------------------------------------------------------------------
    # Test 6 — URL without a scheme (bare hostname)
    # -----------------------------------------------------------------------
    def test_06_url_without_scheme(self):
        """
        A bare hostname (no scheme) should have https:// prepended and be
        accepted, with a warning indicating inference was applied.
        """
        result = validate_url("example.com")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.scheme, "https")
        self.assertEqual(result.hostname, "example.com")
        # Must warn that the scheme was inferred.
        self.assertTrue(
            any("assumed" in w.lower() or "inferred" in w.lower() or "provided" in w.lower()
                for w in result.warnings),
            msg=f"Expected scheme-inference warning. Warnings: {result.warnings}",
        )

    # -----------------------------------------------------------------------
    # Test 7 — Empty input
    # -----------------------------------------------------------------------
    def test_07_empty_input(self):
        """Empty string must be rejected with is_valid=False."""
        result = validate_url("")
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)

    # -----------------------------------------------------------------------
    # Test 8 — Whitespace-only input
    # -----------------------------------------------------------------------
    def test_08_whitespace_only_input(self):
        """A string of only spaces/tabs must be rejected."""
        result = validate_url("   \t  ")
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)

    # -----------------------------------------------------------------------
    # Test 9 — Malformed URL
    # -----------------------------------------------------------------------
    def test_09_malformed_url(self):
        """
        A string that is not a valid URL must be rejected.
        'https://' alone has no hostname and must fail.
        """
        result = validate_url("https://")
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)

    # -----------------------------------------------------------------------
    # Test 10 — Unsupported scheme (ftp)
    # -----------------------------------------------------------------------
    def test_10_unsupported_scheme_ftp(self):
        """ftp:// must be rejected because it is not HTTP or HTTPS."""
        result = validate_url("ftp://example.com/file.txt")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("ftp" in e.lower() or "scheme" in e.lower() for e in result.errors),
            msg=f"Expected scheme error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 11 — Unsupported scheme (javascript:)
    # -----------------------------------------------------------------------
    def test_11_unsupported_scheme_javascript(self):
        """javascript: must be rejected — it is a common XSS vector."""
        result = validate_url("javascript:alert('xss')")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("javascript" in e.lower() or "scheme" in e.lower() for e in result.errors),
            msg=f"Expected scheme error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 12 — Missing hostname
    # -----------------------------------------------------------------------
    def test_12_missing_hostname(self):
        """A URL with scheme but no hostname must be rejected."""
        result = validate_url("https:///path/only")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("hostname" in e.lower() for e in result.errors),
            msg=f"Expected hostname error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 13 — Invalid port (non-numeric)
    # -----------------------------------------------------------------------
    def test_13_invalid_port_non_numeric(self):
        """A non-numeric port must be rejected."""
        result = validate_url("https://example.com:abc/path")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("port" in e.lower() or "hostname" in e.lower() for e in result.errors),
            msg=f"Expected port/hostname error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 14 — Port 0 (outside valid range)
    # -----------------------------------------------------------------------
    def test_14_port_zero(self):
        """Port 0 is outside the valid range 1–65535."""
        result = validate_url("https://example.com:0/path")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("port" in e.lower() for e in result.errors),
            msg=f"Expected port range error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 15 — Port 65536 (outside valid range)
    # -----------------------------------------------------------------------
    def test_15_port_too_high(self):
        """Port 65536 exceeds the maximum TCP/UDP port number."""
        result = validate_url("https://example.com:65536/path")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("port" in e.lower() for e in result.errors),
            msg=f"Expected port range error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 16 — Embedded credentials
    # -----------------------------------------------------------------------
    def test_16_embedded_credentials(self):
        """URLs with username:password@ must be rejected."""
        result = validate_url("https://user:pass@example.com/page")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("credential" in e.lower() or "password" in e.lower()
                or "username" in e.lower() for e in result.errors),
            msg=f"Expected credential error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 17 — IPv4 loopback (127.0.0.1)
    # -----------------------------------------------------------------------
    def test_17_ipv4_loopback(self):
        """
        127.0.0.1 is a loopback address.  Requests to loopback are an
        SSRF risk and must be blocked (destination_status='blocked').
        """
        result = validate_url("https://127.0.0.1/admin")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")
        self.assertTrue(result.is_ip_address)
        self.assertTrue(
            any("loopback" in e.lower() for e in result.errors),
            msg=f"Expected loopback error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 18 — IPv4 private address (192.168.1.1)
    # -----------------------------------------------------------------------
    def test_18_ipv4_private_address(self):
        """
        192.168.1.1 is a private RFC-1918 address.  Fetching it from an
        application server would be an SSRF attack against the internal network.
        """
        result = validate_url("https://192.168.1.1/router")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")
        self.assertTrue(
            any("private" in e.lower() for e in result.errors),
            msg=f"Expected private-address error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 19 — IPv6 loopback (::1)
    # -----------------------------------------------------------------------
    def test_19_ipv6_loopback(self):
        """
        [::1] is the IPv6 loopback address, equivalent to 127.0.0.1.
        It must be blocked regardless of notation.
        """
        result = validate_url("https://[::1]/admin")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")
        self.assertTrue(
            any("loopback" in e.lower() for e in result.errors),
            msg=f"Expected loopback error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 20 — IPv6 link-local (fe80::1)
    # -----------------------------------------------------------------------
    def test_20_ipv6_link_local(self):
        """
        fe80::/10 addresses are link-local and must not be reachable
        from an application server.
        """
        result = validate_url("https://[fe80::1]/path")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")
        self.assertTrue(
            any("link-local" in e.lower() for e in result.errors),
            msg=f"Expected link-local error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 21 — Public IP address (8.8.8.8)
    # -----------------------------------------------------------------------
    def test_21_public_ip_address(self):
        """
        A publicly routable IP address must NOT be blocked by the module,
        but a warning should be issued.  The destination has not been
        verified as legitimate.
        """
        result = validate_url("https://8.8.8.8/check")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertTrue(result.is_ip_address)
        # destination_status stays 'not_checked' — it is public but unverified.
        self.assertNotEqual(result.destination_status, "blocked")
        # Must warn about direct IP.
        self.assertTrue(
            any("ip address" in w.lower() or "direct ip" in w.lower()
                or "ip" in w.lower() for w in result.warnings),
            msg=f"Expected IP warning. Warnings: {result.warnings}",
        )

    # -----------------------------------------------------------------------
    # Test 22 — Unverified hostname (ordinary domain)
    # -----------------------------------------------------------------------
    def test_22_unverified_hostname(self):
        """
        A standard domain must be syntactically valid but destination_status
        must be 'not_checked' — the module never marks a hostname as safe
        without an explicit DNS check.
        """
        result = validate_url("https://example.com/page")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.destination_status, "not_checked")

    # -----------------------------------------------------------------------
    # Test 23 — Internationalized domain name (IDN)
    # -----------------------------------------------------------------------
    def test_23_idn_hostname(self):
        """
        A Unicode hostname should be accepted and IDNA-encoded.
        '例え.jp' encodes to 'xn--r8jz45g.jp'.
        The normalized hostname must be the ACE form.
        """
        result = validate_url("https://例え.jp/page")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        # The hostname must be the ACE form, not the raw unicode.
        self.assertFalse(
            any(ord(c) > 127 for c in result.hostname),
            msg=f"Expected ACE hostname, got: {result.hostname}",
        )

    # -----------------------------------------------------------------------
    # Test 24 — Hostname with a trailing dot
    # -----------------------------------------------------------------------
    def test_24_hostname_with_trailing_dot(self):
        """
        A trailing dot (FQDN notation) is valid in DNS and must be accepted.
        The hostname field must retain the trailing dot.
        """
        result = validate_url("https://example.com./page")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertTrue(result.hostname.endswith("."),
                        msg=f"Expected trailing dot. Hostname: {result.hostname}")

    # -----------------------------------------------------------------------
    # Test 25 — Malformed hostname (spaces)
    # -----------------------------------------------------------------------
    def test_25_malformed_hostname_spaces(self):
        """Input containing spaces that cannot form a valid hostname must fail."""
        result = validate_url("https://not a hostname/page")
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)

    # -----------------------------------------------------------------------
    # Test 26 — DNS check: hostname that cannot be resolved (mock)
    # -----------------------------------------------------------------------
    def test_26_dns_check_unresolvable(self):
        """
        When DNS resolution raises socket.gaierror, check_dns_destination()
        must return destination_unverified (not safe).
        No real DNS query is made — socket.getaddrinfo is mocked.
        """
        import socket
        with patch("modules.url_validator.socket.getaddrinfo",
                   side_effect=socket.gaierror("Name or service not known")):
            dns_result = check_dns_destination("nonexistent.invalid")

        self.assertEqual(dns_result["destination_status"], "destination_unverified")
        self.assertTrue(len(dns_result["errors"]) > 0)
        self.assertEqual(dns_result["resolved_ips"], [])

    # -----------------------------------------------------------------------
    # Test 27 — DNS check: hostname resolves to a prohibited address (mock)
    # -----------------------------------------------------------------------
    def test_27_dns_check_prohibited_address(self):
        """
        When DNS returns a private address, check_dns_destination() must
        return destination_status='blocked'.
        socket.getaddrinfo is mocked — no real DNS query is made.
        """
        mock_result = [_make_addr_info("10.0.0.1")]
        with patch("modules.url_validator.socket.getaddrinfo", return_value=mock_result):
            dns_result = check_dns_destination("internal.corp")

        self.assertEqual(dns_result["destination_status"], "blocked")
        self.assertIn("10.0.0.1", dns_result["resolved_ips"])
        self.assertTrue(
            any("prohibited" in e.lower() or "private" in e.lower()
                for e in dns_result["errors"]),
            msg=f"Expected prohibited-address error. Errors: {dns_result['errors']}",
        )

    # -----------------------------------------------------------------------
    # Test 28 — DNS check: mixed public and prohibited addresses (mock)
    # -----------------------------------------------------------------------
    def test_28_dns_check_mixed_results(self):
        """
        When DNS returns both a public and a private address, the result
        must be 'blocked'.  ALL returned addresses must pass the policy —
        not just the first one.
        """
        mock_result = [
            _make_addr_info("8.8.8.8"),    # public
            _make_addr_info("192.168.1.1"), # private — must trigger block
        ]
        with patch("modules.url_validator.socket.getaddrinfo", return_value=mock_result):
            dns_result = check_dns_destination("mixed.example.com")

        self.assertEqual(dns_result["destination_status"], "blocked")
        self.assertTrue(
            any("192.168.1.1" in e for e in dns_result["errors"]),
            msg=f"Expected 192.168.1.1 in errors. Errors: {dns_result['errors']}",
        )

    # -----------------------------------------------------------------------
    # Test 29 — Blocked scheme: file://
    # -----------------------------------------------------------------------
    def test_29_blocked_scheme_file(self):
        """file:// must be rejected — it allows reading the local filesystem."""
        result = validate_url("file:///etc/passwd")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("file" in e.lower() or "scheme" in e.lower() for e in result.errors),
            msg=f"Expected scheme error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 30 — Blocked scheme: data:
    # -----------------------------------------------------------------------
    def test_30_blocked_scheme_data(self):
        """data: URIs must be rejected — they can embed arbitrary content."""
        result = validate_url("data:text/html,<h1>Test</h1>")
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("data" in e.lower() or "scheme" in e.lower() for e in result.errors),
            msg=f"Expected scheme error. Errors: {result.errors}",
        )

    # -----------------------------------------------------------------------
    # Test 31 — Valid URL with explicit port
    # -----------------------------------------------------------------------
    def test_31_valid_url_with_explicit_port(self):
        """Port 8443 must be parsed and stored as an integer."""
        result = validate_url("https://example.com:8443/api")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.port, 8443)

    # -----------------------------------------------------------------------
    # Test 32 — is_ip_address is False for regular hostname
    # -----------------------------------------------------------------------
    def test_32_is_ip_address_false_for_hostname(self):
        """is_ip_address must be False for a regular domain name."""
        result = validate_url("https://example.com")
        self.assertFalse(result.is_ip_address)

    # -----------------------------------------------------------------------
    # Test 33 — IPv4-mapped IPv6 treated as private IPv4
    # -----------------------------------------------------------------------
    def test_33_ipv4_mapped_ipv6_private(self):
        """
        ::ffff:192.168.1.1 is an IPv4-mapped IPv6 address.
        Its embedded IPv4 address (192.168.1.1) is private, so the
        destination must be blocked even though the outer form is IPv6.
        This prevents bypassing IP-block rules via IPv6 notation.
        """
        result = validate_url("https://[::ffff:192.168.1.1]/path")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")

    # -----------------------------------------------------------------------
    # Test 34 — Explicit port 443 on HTTPS URL
    # -----------------------------------------------------------------------
    def test_34_explicit_port_443_https(self):
        """Port 443 explicitly stated on an HTTPS URL is valid."""
        result = validate_url("https://example.com:443/page")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.port, 443)

    # -----------------------------------------------------------------------
    # Test 35 — URL with no path (root)
    # -----------------------------------------------------------------------
    def test_35_url_no_path(self):
        """A URL with no path component must be valid; path defaults to ''."""
        result = validate_url("https://example.com")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.path, "")

    # -----------------------------------------------------------------------
    # Test 36 — to_dict() returns a plain dictionary
    # -----------------------------------------------------------------------
    def test_36_to_dict_returns_dict(self):
        """to_dict() must return a plain dict with all expected keys."""
        result = validate_url("https://example.com")
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        expected_keys = {
            "original_input", "normalized_url", "is_valid", "scheme",
            "hostname", "port", "path", "query", "fragment", "uses_https",
            "is_ip_address", "destination_status", "errors", "warnings",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    # -----------------------------------------------------------------------
    # Test 37 — Non-string input
    # -----------------------------------------------------------------------
    def test_37_non_string_input(self):
        """Non-string input must not raise an unhandled exception."""
        result = validate_url(None)  # type: ignore[arg-type]
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)

    # -----------------------------------------------------------------------
    # Test 38 — DNS check: resolves to public address (mock)
    # -----------------------------------------------------------------------
    def test_38_dns_check_public_address(self):
        """
        When DNS returns only public addresses, check_dns_destination()
        must return 'verified_for_policy' with a TOCTOU / DNS-rebinding warning.
        """
        mock_result = [_make_addr_info("93.184.216.34")]  # example.com's IP
        with patch("modules.url_validator.socket.getaddrinfo", return_value=mock_result):
            dns_result = check_dns_destination("example.com")

        self.assertEqual(dns_result["destination_status"], "verified_for_policy")
        self.assertEqual(dns_result["errors"], [])
        # Must warn about TOCTOU / re-enforcement requirement.
        self.assertTrue(
            any("connection" in w.lower() or "redirect" in w.lower()
                or "rebind" in w.lower() or "re-enforce" in w.lower()
                for w in dns_result["warnings"]),
            msg=f"Expected TOCTOU/rebinding warning. Warnings: {dns_result['warnings']}",
        )

    # -----------------------------------------------------------------------
    # Test 39 — 10.x.x.x private address
    # -----------------------------------------------------------------------
    def test_39_ipv4_10_network_private(self):
        """10.0.0.1 is RFC-1918 private and must be blocked."""
        result = validate_url("http://10.0.0.1/page")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")

    # -----------------------------------------------------------------------
    # Test 40 — 172.16.x.x private address
    # -----------------------------------------------------------------------
    def test_40_ipv4_172_network_private(self):
        """172.16.0.1 is RFC-1918 private and must be blocked."""
        result = validate_url("https://172.16.0.1/admin")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")

    # -----------------------------------------------------------------------
    # Test 41 — URL with leading and trailing whitespace
    # -----------------------------------------------------------------------
    def test_41_whitespace_trimming(self):
        """Leading/trailing whitespace must be stripped; URL must be accepted."""
        result = validate_url("  https://example.com/page  ")
        self.assertTrue(result.is_valid, msg=f"Errors: {result.errors}")
        self.assertEqual(result.hostname, "example.com")

    # -----------------------------------------------------------------------
    # Test 42 — IPv4 link-local (169.254.x.x)
    # -----------------------------------------------------------------------
    def test_42_ipv4_link_local(self):
        """
        169.254.x.x is the APIPA link-local range.
        It must be blocked — it is a local network address and indicates
        DHCP failure; an attacker could use it to probe the host's config.
        """
        result = validate_url("https://169.254.169.254/latest/meta-data/")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.destination_status, "blocked")
        self.assertTrue(
            any("link-local" in e.lower() for e in result.errors),
            msg=f"Expected link-local error. Errors: {result.errors}",
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
