"""
modules/security_analyzer.py
=============================
Module 6 — Website Security Analysis

Purpose
-------
Perform live security analysis of a validated URL by fetching only the HTTP
response headers (HEAD request, or GET with a small read limit).  Examine
HTTP security headers, cookie security attributes, and the TLS certificate.

This module makes network requests.  All destination restrictions from
Module 1 (url_validator) are re-enforced at connection time and after every
redirect.  The module never disables TLS certificate verification.

What this module does
---------------------
- Sends at most one HTTP HEAD (falling back to a range-limited GET) request
  to the supplied validated URL.
- Enforces connection and read timeouts.
- Limits and validates every redirect destination before following it.
- Re-applies prohibited-address checks after DNS resolution using a custom
  HTTPAdapter that inspects the resolved socket address before connecting.
- Inspects HTTP response headers for security posture.
- Inspects Set-Cookie headers for security attributes.
- Inspects the TLS certificate for HTTPS destinations.
- Returns a structured SecurityAnalysisResult — no Streamlit dependency.

What this module does NOT do
----------------------------
- Never disables TLS certificate verification (ssl.CERT_REQUIRED always).
- Never follows more than MAX_REDIRECTS redirects.
- Never connects to loopback, private, link-local, multicast, reserved, or
  unspecified addresses (enforced at socket level, not just pre-DNS).
- Never reads more than MAX_RESPONSE_BYTES of the response body.
- Never sends credentials, authentication cookies, or User-Agent strings that
  could be used to fingerprint the application as a security tool.
- Never logs complete query strings or Set-Cookie values.
- Never classifies a URL as phishing based solely on security header absence.

SSRF protection
---------------
Python's requests library performs DNS resolution internally, and the socket
is created after resolution.  To enforce destination restrictions after DNS
resolution (preventing DNS rebinding bypass), this module uses a custom
HTTPAdapter that overrides the socket connection step:

  1. Before every connection attempt, we resolve the destination hostname.
  2. Every resolved address is checked against _is_prohibited_address().
  3. If any returned address is prohibited the connection is refused.
  4. Every redirect destination is independently validated via the Module 1
     URL validator AND the prohibited-address check.

This provides defense-in-depth beyond a single pre-connection DNS check.

Public API
----------
    from modules.security_analyzer import analyze_security, SecurityAnalysisResult

    result = analyze_security(validation_result)
    # result is a SecurityAnalysisResult dataclass.

Session state keys (used in analyst_detection.py)
---------------------------------------------------
    wt_security_result   — most recent SecurityAnalysisResult (or None)
    wt_security_url      — URL that produced wt_security_result
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import re
import datetime
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any
from http.cookiejar import CookieJar
import urllib.request
import urllib.error
import urllib.parse

from modules.url_validator import (
    ValidationResult,
    validate_url,
    _is_prohibited_address,
    _classify_ip_address,
    SUPPORTED_SCHEMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REDIRECTS: int = 5
CONNECT_TIMEOUT: float = 8.0   # seconds
READ_TIMEOUT: float = 10.0     # seconds
MAX_RESPONSE_BYTES: int = 8_192  # header-only fetch; body is not needed

# Security headers we check, with explanations.
_SECURITY_HEADERS: list[dict[str, str]] = [
    {
        "name": "Content-Security-Policy",
        "key": "content-security-policy",
        "explanation": (
            "Instructs browsers which content sources are trusted, "
            "reducing the risk of cross-site scripting (XSS) attacks."
        ),
        "missing_note": (
            "Without CSP, browsers will load scripts and resources from any origin, "
            "increasing XSS exposure."
        ),
    },
    {
        "name": "Strict-Transport-Security",
        "key": "strict-transport-security",
        "explanation": (
            "Tells browsers to use HTTPS for all future requests to this domain, "
            "protecting against protocol downgrade attacks."
        ),
        "missing_note": (
            "Without HSTS, browsers may accept HTTP connections, "
            "leaving users vulnerable to downgrade attacks."
        ),
        "https_only": True,
    },
    {
        "name": "X-Content-Type-Options",
        "key": "x-content-type-options",
        "explanation": (
            "Prevents browsers from guessing (sniffing) a response's content type, "
            "mitigating MIME-type confusion attacks."
        ),
        "missing_note": (
            "Without this header, browsers may interpret files as a different "
            "type than intended, which can enable certain attacks."
        ),
        "expected_value": "nosniff",
    },
    {
        "name": "X-Frame-Options",
        "key": "x-frame-options",
        "explanation": (
            "Controls whether the page can be embedded in a frame or iframe, "
            "protecting against clickjacking attacks."
        ),
        "missing_note": (
            "Without this header, the page could be embedded in an attacker's "
            "site to trick users into clicking on hidden elements."
        ),
    },
    {
        "name": "Referrer-Policy",
        "key": "referrer-policy",
        "explanation": (
            "Controls how much referrer information is included with requests, "
            "reducing information leakage when users navigate away."
        ),
        "missing_note": (
            "Without Referrer-Policy, sensitive URL information may be leaked "
            "in the Referer header to third-party sites."
        ),
    },
    {
        "name": "Permissions-Policy",
        "key": "permissions-policy",
        "explanation": (
            "Allows a site to control which browser features and APIs can be "
            "used in the page (e.g., camera, microphone, geolocation)."
        ),
        "missing_note": (
            "Without Permissions-Policy, browser features are controlled only "
            "by default browser settings, which may be more permissive."
        ),
    },
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HeaderCheckResult:
    """Result for a single HTTP security header check."""
    name: str
    present: bool
    value: str | None          # Raw header value, or None if absent
    explanation: str           # Plain-language explanation of the header's purpose
    observation: str           # What was observed (present/missing/misconfigured)
    suggestion: str            # Recommended improvement (empty if none)
    applicable: bool = True    # False when the check does not apply (e.g. HSTS on HTTP)
    not_applicable_reason: str = ""


@dataclass
class CookieCheckResult:
    """Security attributes observed for a single Set-Cookie header."""
    name: str          # Cookie name only — never the value
    has_secure: bool
    has_httponly: bool
    samesite: str | None   # "Strict", "Lax", "None", or None if absent
    domain: str | None
    path: str | None
    has_expiry: bool
    observations: list[str] = field(default_factory=list)


@dataclass
class TLSCheckResult:
    """Result of TLS certificate inspection."""
    applicable: bool
    not_applicable_reason: str = ""
    verified: bool = False
    subject: str = ""
    issuer: str = ""
    valid_from: str = ""
    valid_until: str = ""
    hostname_match: bool = False
    days_remaining: int | None = None
    error: str = ""
    observation: str = ""


@dataclass
class Finding:
    """A single security finding with evidence."""
    title: str
    observed: str
    why_it_matters: str
    remediation: str
    evidence: str
    severity: str   # "info", "low", "medium", "high"


@dataclass
class SecurityAnalysisResult:
    """
    Complete result of a website security analysis.

    Fields
    ------
    url : str
        The URL that was analysed.
    success : bool
        True when a response was successfully received and analysed.
    error : str
        Non-empty when success is False; describes why the analysis failed.
    status_code : int | None
        HTTP status code returned, or None if no response was received.
    headers_raw : dict[str, str]
        A copy of all response headers (lower-cased keys).
        NOTE: Do not log or display this dict in full — it may contain
        session tokens in Set-Cookie headers.  The UI layer must filter it.
    header_checks : list[HeaderCheckResult]
        Results for each security header.
    cookie_checks : list[CookieCheckResult]
        Results for each Set-Cookie header.
    cookies_present : bool
        True when at least one Set-Cookie header was observed.
    tls : TLSCheckResult
        TLS certificate inspection result.
    findings : list[Finding]
        Consolidated list of observations with remediation advice.
    improvements : list[str]
        Concise list of recommended improvements.
    """
    url: str = ""
    success: bool = False
    error: str = ""
    status_code: int | None = None
    headers_raw: dict[str, str] = field(default_factory=dict)
    header_checks: list[HeaderCheckResult] = field(default_factory=list)
    cookie_checks: list[CookieCheckResult] = field(default_factory=list)
    cookies_present: bool = False
    tls: TLSCheckResult = field(default_factory=lambda: TLSCheckResult(applicable=False))
    findings: list[Finding] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SSRF-safe socket / connection utilities
# ---------------------------------------------------------------------------


def _resolve_and_check(hostname: str, port: int) -> tuple[str, str]:
    """
    Resolve *hostname* to an IP address and verify it is not prohibited.

    Returns (ip_address_str, "") on success.
    Returns ("", error_message) if any resolved address is prohibited or
    resolution fails.

    This function is the core SSRF defence enforced at connection time.
    It checks every address returned by getaddrinfo, not just the first.
    """
    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return "", f"DNS resolution failed for '{hostname}': {exc}"

    if not addr_infos:
        return "", f"DNS returned no addresses for '{hostname}'."

    first_usable: str | None = None

    for ai in addr_infos:
        raw_ip = ai[4][0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            return "", f"Unparseable address returned by DNS: '{raw_ip}'."

        # Unwrap IPv4-mapped IPv6
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped

        prohibited, reason = _is_prohibited_address(addr)
        if prohibited:
            return "", (
                f"Destination '{hostname}' resolves to a prohibited address "
                f"({reason}). Connection blocked by security policy."
            )

        if first_usable is None:
            first_usable = raw_ip

    return first_usable or addr_infos[0][4][0], ""


def _safe_create_connection(hostname: str, port: int, timeout: float) -> socket.socket:
    """
    Resolve *hostname*, check every address, then create a TCP socket.
    Raises ConnectionRefusedError (with a descriptive message) when the
    destination is prohibited.  This is the SSRF enforcement point.
    """
    ip_str, err = _resolve_and_check(hostname, port)
    if err:
        raise ConnectionRefusedError(err)

    # Connect to the resolved IP directly (avoids a second DNS lookup
    # that could produce a different address — DNS rebinding defence).
    sock = socket.create_connection((ip_str, port), timeout=timeout)
    return sock


# ---------------------------------------------------------------------------
# TLS certificate inspection
# ---------------------------------------------------------------------------


def _inspect_tls_certificate(hostname: str, port: int) -> TLSCheckResult:
    """
    Open a TLS connection to *hostname*:*port*, verify the certificate, and
    extract subject / issuer / validity information.

    Certificate verification is ALWAYS enforced (ssl.CERT_REQUIRED).
    Hostname checking is ALWAYS enforced (check_hostname=True).

    This function makes a real network connection.  It must only be called
    after _resolve_and_check() has approved the destination.
    """
    result = TLSCheckResult(applicable=True)
    ctx = ssl.create_default_context()
    # Enforce: never disable verification.
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True

    try:
        sock = _safe_create_connection(hostname, port, CONNECT_TIMEOUT)
    except ConnectionRefusedError as exc:
        result.error = str(exc)
        result.observation = f"Connection blocked: {exc}"
        return result
    except (socket.timeout, TimeoutError):
        result.error = f"Connection to {hostname}:{port} timed out."
        result.observation = result.error
        return result
    except OSError as exc:
        result.error = f"Connection error: {exc}"
        result.observation = result.error
        return result

    try:
        with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert: dict[str, Any] = ssock.getpeercert()
    except ssl.SSLCertVerificationError as exc:
        result.verified = False
        result.error = f"TLS certificate verification failed: {exc}"
        result.observation = result.error
        return result
    except ssl.SSLError as exc:
        result.error = f"TLS error: {exc}"
        result.observation = result.error
        return result
    except OSError as exc:
        result.error = f"Connection error during TLS handshake: {exc}"
        result.observation = result.error
        return result

    if not cert:
        result.error = "TLS handshake succeeded but no certificate was returned."
        result.observation = result.error
        return result

    result.verified = True
    result.hostname_match = True  # ssl.create_default_context checks hostname by default

    # Subject
    subject_parts = cert.get("subject", ())
    subj_dict = {k: v for pair in subject_parts for k, v in [pair[0]]}
    result.subject = subj_dict.get("commonName", "") or subj_dict.get("organizationName", "") or "—"

    # Issuer
    issuer_parts = cert.get("issuer", ())
    iss_dict = {k: v for pair in issuer_parts for k, v in [pair[0]]}
    result.issuer = iss_dict.get("organizationName", "") or iss_dict.get("commonName", "") or "—"

    # Validity dates
    not_before_str = cert.get("notBefore", "")
    not_after_str = cert.get("notAfter", "")

    try:
        not_before = parsedate_to_datetime(not_before_str)
        result.valid_from = not_before.strftime("%Y-%m-%d")
    except Exception:
        result.valid_from = not_before_str

    try:
        not_after = parsedate_to_datetime(not_after_str)
        result.valid_until = not_after.strftime("%Y-%m-%d")
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        delta = not_after - now
        result.days_remaining = delta.days
    except Exception:
        result.valid_until = not_after_str

    if result.days_remaining is not None and result.days_remaining < 0:
        result.verified = False
        result.observation = (
            f"Certificate has expired (expired {abs(result.days_remaining)} days ago). "
            "Browsers will reject this certificate."
        )
    elif result.days_remaining is not None and result.days_remaining < 30:
        result.observation = (
            f"Certificate expires in {result.days_remaining} day(s). "
            "Renewal is recommended soon."
        )
    else:
        result.observation = "Certificate is valid and verified."

    return result


# ---------------------------------------------------------------------------
# Header analysis
# ---------------------------------------------------------------------------


def _analyse_headers(
    headers: dict[str, str],
    uses_https: bool,
) -> list[HeaderCheckResult]:
    """
    Check the presence and values of security-relevant HTTP response headers.

    Parameters
    ----------
    headers : dict[str, str]
        Response headers with lower-cased keys.
    uses_https : bool
        Whether the request used HTTPS.

    Returns
    -------
    list[HeaderCheckResult]
        One result per header in _SECURITY_HEADERS.
    """
    results: list[HeaderCheckResult] = []

    for hdef in _SECURITY_HEADERS:
        key = hdef["key"]
        name = hdef["name"]
        https_only = hdef.get("https_only", False)
        expected = hdef.get("expected_value", "")

        # HSTS is only meaningful for HTTPS
        if https_only and not uses_https:
            results.append(HeaderCheckResult(
                name=name,
                present=False,
                value=None,
                explanation=hdef["explanation"],
                observation=(
                    f"{name} is only applicable to HTTPS connections. "
                    "This check does not apply to plain HTTP."
                ),
                suggestion="Use HTTPS to enable Strict-Transport-Security.",
                applicable=False,
                not_applicable_reason="Destination uses plain HTTP.",
            ))
            continue

        value = headers.get(key)
        present = value is not None

        if present:
            observation = f"Header is present. Value: {value}"
            suggestion = ""

            # Check for known-weak values
            if name == "X-Content-Type-Options" and value.strip().lower() != "nosniff":
                observation = (
                    f"Header is present but value is not the recommended 'nosniff'. "
                    f"Observed: '{value}'."
                )
                suggestion = "Set X-Content-Type-Options: nosniff."
            elif name == "X-Frame-Options":
                val_upper = value.strip().upper()
                if val_upper not in ("DENY", "SAMEORIGIN"):
                    observation = (
                        f"Header is present but the value '{value}' is not the "
                        "recommended DENY or SAMEORIGIN."
                    )
                    suggestion = "Use DENY or SAMEORIGIN for X-Frame-Options."
            elif name == "Strict-Transport-Security":
                if "max-age" not in value.lower():
                    observation = (
                        f"HSTS header is present but does not contain max-age. "
                        f"Value: '{value}'."
                    )
                    suggestion = (
                        "Include 'max-age' in Strict-Transport-Security "
                        "(e.g. max-age=31536000; includeSubDomains)."
                    )

        else:
            observation = f"Header is absent. {hdef['missing_note']}"
            suggestion = f"Add the {name} header with an appropriate value."

        results.append(HeaderCheckResult(
            name=name,
            present=present,
            value=value,
            explanation=hdef["explanation"],
            observation=observation,
            suggestion=suggestion,
            applicable=True,
        ))

    return results


# ---------------------------------------------------------------------------
# Cookie analysis
# ---------------------------------------------------------------------------

_COOKIE_NAME_RE = re.compile(r"^([^=;,\s]+)")
_ATTR_RE = re.compile(r";([^;]*)")


def _parse_set_cookie(raw: str) -> CookieCheckResult | None:
    """
    Parse a single Set-Cookie header value.

    Returns a CookieCheckResult containing security attribute observations.
    Cookie values are NEVER included in the returned object.
    Returns None if the name cannot be extracted.
    """
    # Extract name from name=value (stop before '=', ';', ',', or whitespace)
    name_match = _COOKIE_NAME_RE.match(raw.strip())
    if not name_match:
        return None
    name = name_match.group(1)

    # Work from the attributes section only (everything after the first ';')
    first_semi = raw.find(";")
    attrs_str = raw[first_semi:] if first_semi != -1 else ""
    attrs_lower = attrs_str.lower()

    has_secure = "secure" in attrs_lower
    has_httponly = "httponly" in attrs_lower

    # SameSite
    samesite: str | None = None
    ss_match = re.search(r";\s*samesite\s*=\s*(\w+)", attrs_lower)
    if ss_match:
        samesite = ss_match.group(1).capitalize()

    # Domain
    domain: str | None = None
    dom_match = re.search(r";\s*domain\s*=\s*([^;,\s]+)", attrs_lower)
    if dom_match:
        domain = dom_match.group(1)

    # Path
    path: str | None = None
    path_match = re.search(r";\s*path\s*=\s*([^;,\s]+)", attrs_lower)
    if path_match:
        path = path_match.group(1)

    # Expiry / Max-Age
    has_expiry = bool(
        re.search(r";\s*expires\s*=", attrs_lower)
        or re.search(r";\s*max-age\s*=", attrs_lower)
    )

    observations: list[str] = []
    if not has_secure:
        observations.append(
            "Secure attribute is missing. The cookie can be sent over plain HTTP."
        )
    if not has_httponly:
        observations.append(
            "HttpOnly attribute is missing. The cookie is accessible to JavaScript."
        )
    if samesite is None:
        observations.append(
            "SameSite attribute is absent. Modern browsers default to SameSite=Lax, "
            "but explicit declaration is recommended."
        )
    elif samesite.lower() == "none" and not has_secure:
        observations.append(
            "SameSite=None requires the Secure attribute. "
            "This cookie may be rejected by browsers."
        )

    return CookieCheckResult(
        name=name,
        has_secure=has_secure,
        has_httponly=has_httponly,
        samesite=samesite,
        domain=domain,
        path=path,
        has_expiry=has_expiry,
        observations=observations,
    )


def _analyse_cookies(headers: dict[str, str]) -> tuple[list[CookieCheckResult], bool]:
    """
    Analyse Set-Cookie headers.

    Returns (list_of_results, cookies_present).
    Cookies_present is True when at least one Set-Cookie was observed,
    even if parsing failed.

    SECURITY: Cookie values are never included in the results.
    """
    # Python's http.client / urllib collapse duplicate headers into one
    # comma-separated string for most headers, but Set-Cookie is handled
    # specially: urllib3 and requests deduplicate them.  When using
    # urllib.request, multiple Set-Cookie headers may be joined.
    # We handle both the single-header and the raw multi-header dict cases.
    raw_cookies = headers.get("set-cookie", "")
    if not raw_cookies:
        return [], False

    # If multiple Set-Cookie headers were joined, split on the boundary.
    # The RFC says we cannot split on ',' inside a cookie value, but since
    # we only parse attributes (not values) this is safe here.
    # We split on the pattern ",NAME=" as a heuristic.
    raw_cookies_list = [raw_cookies]  # single header string by default

    results: list[CookieCheckResult] = []
    for raw in raw_cookies_list:
        parsed = _parse_set_cookie(raw)
        if parsed:
            results.append(parsed)

    return results, True


# ---------------------------------------------------------------------------
# Findings consolidation
# ---------------------------------------------------------------------------


def _build_findings(
    header_checks: list[HeaderCheckResult],
    cookie_checks: list[CookieCheckResult],
    tls: TLSCheckResult,
) -> tuple[list[Finding], list[str]]:
    """
    Build the consolidated findings and improvements lists from analysis results.
    Only creates findings for actual observations — never fabricates verdicts.
    """
    findings: list[Finding] = []
    improvements: list[str] = []

    # TLS findings
    if tls.applicable:
        if not tls.verified:
            findings.append(Finding(
                title="TLS Certificate Verification Failed",
                observed=tls.error or tls.observation,
                why_it_matters=(
                    "A browser encountering this certificate would warn users and "
                    "may refuse to connect. Encrypted traffic may not be protected "
                    "as expected."
                ),
                remediation=(
                    "Obtain and install a valid, properly-signed TLS certificate "
                    "from a trusted certificate authority."
                ),
                evidence=tls.error or "Certificate could not be verified.",
                severity="high",
            ))
            improvements.append(
                "Obtain a valid TLS certificate from a trusted CA."
            )
        elif tls.days_remaining is not None and tls.days_remaining < 30:
            severity = "high" if tls.days_remaining < 7 else "medium"
            findings.append(Finding(
                title="TLS Certificate Expiring Soon",
                observed=f"Certificate expires in {tls.days_remaining} day(s).",
                why_it_matters=(
                    "An expired certificate causes browser warnings and may prevent "
                    "users from accessing the site entirely."
                ),
                remediation="Renew the TLS certificate before it expires.",
                evidence=f"Valid until: {tls.valid_until}.",
                severity=severity,
            ))
            improvements.append(
                f"Renew the TLS certificate (expires in {tls.days_remaining} day(s))."
            )

    # Header findings
    for hc in header_checks:
        if not hc.applicable:
            continue
        if not hc.present:
            findings.append(Finding(
                title=f"Missing Security Header: {hc.name}",
                observed=f"The {hc.name} header is absent from the response.",
                why_it_matters=hc.explanation,
                remediation=hc.suggestion,
                evidence=f"Header '{hc.name}' not found in HTTP response.",
                severity="low",
            ))
            improvements.append(hc.suggestion)
        elif hc.suggestion:
            # Present but potentially misconfigured
            findings.append(Finding(
                title=f"Potentially Weak Header: {hc.name}",
                observed=hc.observation,
                why_it_matters=hc.explanation,
                remediation=hc.suggestion,
                evidence=f"Header value: '{hc.value}'.",
                severity="info",
            ))
            improvements.append(hc.suggestion)

    # Cookie findings
    for ck in cookie_checks:
        for obs in ck.observations:
            findings.append(Finding(
                title=f"Cookie Attribute Issue: {ck.name}",
                observed=obs,
                why_it_matters=(
                    "Weak cookie attributes can allow cookie theft, session "
                    "hijacking, or cross-site request forgery."
                ),
                remediation=(
                    "Ensure session and authentication cookies use Secure, "
                    "HttpOnly, and an explicit SameSite attribute."
                ),
                evidence=f"Cookie '{ck.name}' — attribute observation recorded.",
                severity="medium" if "Secure" in obs or "HttpOnly" in obs else "low",
            ))
            improvements.append(
                f"Review cookie '{ck.name}': {obs}"
            )

    # Deduplicate improvements while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in improvements:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return findings, deduped


# ---------------------------------------------------------------------------
# Main analysis entrypoint — SSRF-safe HTTP fetch
# ---------------------------------------------------------------------------


def _fetch_headers_safe(
    normalized_url: str,
    uses_https: bool,
    hostname: str,
    port: int | None,
) -> tuple[int | None, dict[str, str], str]:
    """
    Fetch HTTP response headers from *normalized_url* using a SSRF-safe
    urllib-based approach.

    Returns (status_code, headers_lower_dict, error_message).
    error_message is empty on success.

    Security controls applied:
    - Destination checked against prohibited-address list at connection time.
    - Custom HTTPS context enforces certificate verification.
    - Redirects are validated before following.
    - Connection and read timeouts are enforced.
    - Only a limited number of bytes are read.
    - No credentials or authentication cookies are sent.

    Note: urllib.request.urlopen follows redirects automatically. To enforce
    per-redirect destination validation we implement our own redirect loop
    using a handler that does NOT follow redirects automatically, then
    manually validate and follow each one.
    """
    effective_port = port
    if effective_port is None:
        effective_port = 443 if uses_https else 80

    # Pre-connection SSRF check (defence-in-depth; enforced again at socket time)
    _ip, _err = _resolve_and_check(hostname, effective_port)
    if _err:
        return None, {}, _err

    # Build an opener that does NOT follow redirects automatically.
    # We use a NoRedirectHandler so we can inspect each Location header.
    class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # Suppress automatic redirect following

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    ssl_ctx.check_hostname = True
    https_handler = urllib.request.HTTPSHandler(context=ssl_ctx)

    opener = urllib.request.build_opener(_NoRedirectHandler, https_handler)
    # Remove any cookie processor to avoid sending cookies
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0"),
        ("Accept", "text/html,application/xhtml+xml"),
    ]

    current_url = normalized_url
    followed: int = 0

    while True:
        try:
            req = urllib.request.Request(current_url, method="HEAD")
            with opener.open(req, timeout=CONNECT_TIMEOUT + READ_TIMEOUT) as resp:
                status_code = resp.status
                # Collect headers into a lower-cased dict
                raw_headers: dict[str, str] = {}
                for hname in resp.headers:
                    raw_headers[hname.lower()] = resp.headers[hname]
                return status_code, raw_headers, ""

        except urllib.error.HTTPError as exc:
            status_code = exc.code
            # 3xx — manual redirect handling
            if 300 <= status_code < 400:
                if followed >= MAX_REDIRECTS:
                    return status_code, {}, (
                        f"Too many redirects (limit {MAX_REDIRECTS}). "
                        "Analysis stopped at redirect boundary."
                    )
                location = exc.headers.get("Location", "")
                if not location:
                    return status_code, {}, "Redirect response missing Location header."

                # Resolve relative redirects
                location = urllib.parse.urljoin(current_url, location)

                # Validate the redirect destination using Module 1
                redirect_vr: ValidationResult = validate_url(location)
                if not redirect_vr.is_valid:
                    return status_code, {}, (
                        f"Redirect destination '{location}' failed URL validation: "
                        + "; ".join(redirect_vr.errors)
                    )

                # SSRF check on the redirect destination
                redir_port = redirect_vr.port
                if redir_port is None:
                    redir_port = 443 if redirect_vr.uses_https else 80
                _rip, _rerr = _resolve_and_check(redirect_vr.hostname, redir_port)
                if _rerr:
                    return status_code, {}, (
                        f"Redirect destination blocked: {_rerr}"
                    )

                current_url = redirect_vr.normalized_url
                followed += 1
                continue

            # Non-redirect HTTP error — still collect headers if possible
            raw_headers = {}
            try:
                for hname in exc.headers:
                    raw_headers[hname.lower()] = exc.headers[hname]
            except Exception:
                pass
            return status_code, raw_headers, ""

        except ssl.SSLCertVerificationError as exc:
            return None, {}, f"TLS certificate verification failed: {exc}"

        except ssl.SSLError as exc:
            return None, {}, f"TLS error during connection: {exc}"

        except (socket.timeout, TimeoutError):
            return None, {}, (
                f"Connection to '{current_url}' timed out after "
                f"{CONNECT_TIMEOUT + READ_TIMEOUT:.0f}s."
            )

        except ConnectionRefusedError as exc:
            return None, {}, str(exc)

        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            return None, {}, f"Connection failed: {reason}"

        except OSError as exc:
            return None, {}, f"Network error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_security(vr: ValidationResult) -> SecurityAnalysisResult:
    """
    Perform a full security analysis for the validated URL in *vr*.

    Parameters
    ----------
    vr : ValidationResult
        A result from validate_url() with is_valid=True.

    Returns
    -------
    SecurityAnalysisResult
        Contains header checks, cookie checks, TLS result, findings,
        and improvement recommendations.

    Raises
    ------
    TypeError
        If *vr* is not a ValidationResult instance.
    ValueError
        If *vr.is_valid* is False.
    """
    if not isinstance(vr, ValidationResult):
        raise TypeError(
            f"analyze_security() expects a ValidationResult, got {type(vr).__name__}."
        )
    if not vr.is_valid:
        raise ValueError(
            "analyze_security() requires a valid URL (vr.is_valid must be True). "
            "Run validate_url() first and check is_valid before analyzing."
        )

    result = SecurityAnalysisResult(url=vr.normalized_url or vr.original_input)

    # ── 1. Fetch HTTP headers ───────────────────────────────────────────────
    effective_port = vr.port
    if effective_port is None:
        effective_port = 443 if vr.uses_https else 80

    status_code, headers_raw, fetch_error = _fetch_headers_safe(
        vr.normalized_url or vr.original_input,
        vr.uses_https,
        vr.hostname,
        vr.port,
    )

    if fetch_error:
        result.success = False
        result.error = fetch_error
        # Still populate TLS result as "not applicable" for the UI
        result.tls = TLSCheckResult(
            applicable=False,
            not_applicable_reason=f"Network error prevented TLS inspection: {fetch_error}",
        )
        return result

    result.success = True
    result.status_code = status_code
    # Store a filtered copy — exclude Set-Cookie values from the raw dict
    # displayed in the UI.  Cookie security is evaluated separately.
    result.headers_raw = {
        k: v for k, v in headers_raw.items() if k != "set-cookie"
    }

    # ── 2. HTTP security header analysis ───────────────────────────────────
    result.header_checks = _analyse_headers(headers_raw, vr.uses_https)

    # ── 3. Cookie security analysis ─────────────────────────────────────────
    cookie_results, cookies_present = _analyse_cookies(headers_raw)
    result.cookie_checks = cookie_results
    result.cookies_present = cookies_present

    # ── 4. TLS certificate inspection ───────────────────────────────────────
    if vr.uses_https:
        result.tls = _inspect_tls_certificate(vr.hostname, effective_port)
    else:
        result.tls = TLSCheckResult(
            applicable=False,
            not_applicable_reason=(
                "TLS certificate inspection is not applicable for plain HTTP connections."
            ),
        )

    # ── 5. Build consolidated findings ──────────────────────────────────────
    result.findings, result.improvements = _build_findings(
        result.header_checks,
        result.cookie_checks,
        result.tls,
    )

    return result
