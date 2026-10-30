"""
examples/test_url_validation.py
================================
Demonstration script for Module 1 — URL Input and Validation.

Run this from the project root:
    py examples/test_url_validation.py

This script demonstrates how an analyst or downstream module can call
validate_url() and inspect the structured result.  It is NOT part of
the automated test suite (which uses unittest).

Security reminder
-----------------
This script passes URLs to validate_url() only.
No HTTP connections are made.
No DNS queries are made (except in the DNS section below, which uses
  check_dns_destination() explicitly and is clearly labelled).
"""

import sys
import os

# Allow running from project root without installing as a package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.url_validator import validate_url, check_dns_destination


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _print_result(result) -> None:
    """Print a ValidationResult in a readable, analyst-friendly format."""
    status = "✓  VALID" if result.is_valid else "✗  INVALID"
    print(f"  Status         : {status}")
    print(f"  Scheme         : {result.scheme or '(none)'}")
    print(f"  Hostname       : {result.hostname or '(none)'}")
    print(f"  Port           : {result.port}")
    print(f"  Path           : {result.path!r}")
    # Query string may contain sensitive tokens — shown here only for demo.
    print(f"  Query          : {result.query!r}  [⚠ may contain sensitive data]")
    print(f"  Fragment       : {result.fragment!r}")
    print(f"  Uses HTTPS     : {result.uses_https}")
    print(f"  Is IP address  : {result.is_ip_address}")
    print(f"  Dest. status   : {result.destination_status}")
    if result.errors:
        for e in result.errors:
            print(f"  ERROR          : {e}")
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING        : {w}")


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Example URLs
# ---------------------------------------------------------------------------

examples = [
    # (label, url_string)
    ("Valid HTTPS URL",                    "https://example.com/login"),
    ("Valid HTTP URL (unencrypted)",       "http://example.com/page"),
    ("URL with path, query, fragment",     "https://example.com/search?q=phishing&page=1#results"),
    ("URL with explicit port",             "https://example.com:8443/api/v1"),
    ("Bare hostname (no scheme)",          "example.com"),
    ("Internationalized domain (IDN)",     "https://例え.jp/page"),
    ("Hostname with trailing dot",         "https://example.com./page"),
    ("Empty input",                        ""),
    ("Whitespace only",                    "   "),
    ("Missing scheme + no dot in input",   "notahostname"),
    ("Unsupported scheme: ftp://",         "ftp://files.example.com/data.zip"),
    ("Unsupported scheme: javascript:",    "javascript:alert('xss')"),
    ("Unsupported scheme: file://",        "file:///etc/passwd"),
    ("Missing hostname: https://",         "https://"),
    ("Embedded credentials",               "https://admin:secret@example.com/"),
    ("Invalid port (letters)",             "https://example.com:abc/path"),
    ("Port 0 (out of range)",              "https://example.com:0/path"),
    ("Port 65536 (out of range)",          "https://example.com:65536/path"),
    ("IPv4 loopback (SSRF risk)",          "https://127.0.0.1/admin"),
    ("IPv4 private RFC-1918",              "https://192.168.1.1/router"),
    ("IPv4 private 10.x.x.x",             "https://10.0.0.1/internal"),
    ("IPv4 link-local (AWS metadata)",     "https://169.254.169.254/latest/meta-data/"),
    ("IPv6 loopback [::1]",               "https://[::1]/admin"),
    ("IPv6 link-local [fe80::1]",          "https://[fe80::1]/path"),
    ("IPv4-mapped IPv6 (bypass attempt)", "https://[::ffff:192.168.1.1]/path"),
    ("Public IP (routable, unverified)",   "https://8.8.8.8/check"),
    ("Leading/trailing whitespace",        "  https://example.com/page  "),
]


def run_examples() -> None:
    _section("Module 1 — URL Validation Examples")
    print("  NOTE: No network requests are made by these examples.")
    print("  The query field may contain sensitive data — do not log in production.")

    for label, url in examples:
        _section(label)
        print(f"  Input  : {url!r}")
        result = validate_url(url)
        _print_result(result)

    # -----------------------------------------------------------------------
    # DNS check examples (uses mocked / public DNS — clearly labelled)
    # -----------------------------------------------------------------------
    _section("DNS Destination Check — check_dns_destination()")
    print("  NOTE: These examples DO make real DNS requests.")
    print("  They require network access.  They are separate from syntax validation.")
    print("  In production, call check_dns_destination() only when intentional.")
    print()

    dns_targets = [
        ("example.com",              "Expected: verified_for_policy (if DNS works)"),
        ("nonexistent.thisdoesnotexistever.invalid",
                                     "Expected: destination_unverified (NXDOMAIN)"),
    ]
    for hostname, note in dns_targets:
        print(f"  Hostname : {hostname}")
        print(f"  Note     : {note}")
        dns_result = check_dns_destination(hostname)
        print(f"  Result   : {dns_result['destination_status']}")
        print(f"  IPs      : {dns_result['resolved_ips']}")
        for e in dns_result.get("errors", []):
            print(f"  ERROR    : {e}")
        for w in dns_result.get("warnings", []):
            print(f"  WARNING  : {w}")
        print()

    _section("End of examples")


if __name__ == "__main__":
    run_examples()
