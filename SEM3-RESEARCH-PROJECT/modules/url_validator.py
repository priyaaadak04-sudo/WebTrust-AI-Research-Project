"""
modules/url_validator.py
========================
Module 1 — URL Input and Validation

Purpose
-------
Accept a URL submitted by an analyst, check its syntax, extract its
components, and perform preliminary checks to identify destinations that
the application must NOT access.

This module does NOT classify URLs as phishing or legitimate.
This module does NOT make any network requests.
This module does NOT follow redirects.

Concepts kept deliberately separate
------------------------------------
1. URL syntax validation          — is the string a well-formed URL?
2. URL normalisation              — produce a canonical form for later use.
3. Preliminary destination checks — is this destination safe to attempt?
4. Website feature extraction     — Module 2 (not yet implemented).
5. Phishing classification        — Module 4 (not yet implemented).

Public API
----------
    from modules.url_validator import validate_url, check_dns_destination

    result = validate_url("https://example.com/login")
    # Returns a dict — see ValidationResult field descriptions below.

    dns_result = check_dns_destination("example.com")
    # Returns a dict with 'destination_status', 'resolved_ips', 'errors'.
    # Only call this when you intentionally want DNS resolution.

Normalisation policy
--------------------
- Leading / trailing whitespace is stripped before any other processing.
- If the input contains no scheme separator (://), the string is treated as
  a possible bare hostname and 'https://' is prepended ONLY when what
  remains is a plausible hostname (contains at least one dot or is a valid
  IPv6 literal in brackets, and contains no spaces).
- Schemes are lowercased.
- Hostnames are lowercased and IDNA-encoded to ASCII (ACE form) so that
  比较.com and xn--fiqs8sirgfmh.com are treated as the same destination.
  The 'hostname' field in the result always contains the ACE form.
- Ports are preserved as integers when present.

Destination status values
--------------------------
"not_checked"         — no check has been attempted (default after syntax
                        validation only).
"blocked"             — the destination is prohibited by security policy
                        (e.g. loopback, private, link-local, multicast,
                        reserved, or unspecified address).
"destination_unverified" — a DNS check was attempted but failed, or the
                           result cannot be confirmed safe.
"verified_for_policy" — DNS resolution succeeded and no returned address
                        was in a prohibited range.  NOTE: this status does
                        NOT mean the site is legitimate or safe to access.
                        SSRF protections MUST be re-applied at connection
                        time and after every redirect.

Security notes
--------------
- A valid URL is not necessarily legitimate.
- An HTTPS URL is not necessarily trustworthy.
- A URL that fails validation is not automatically malicious.
- An unverified destination must never be reported as safe.
- DNS rebinding: the address that DNS returns during this check may differ
  from the address the server uses when a connection is actually opened.
  A one-time DNS check is not a sufficient SSRF defence.
- Time-of-check / time-of-use (TOCTOU): even 'verified_for_policy' does
  not guarantee the destination will be reachable or appropriate when the
  connection is eventually made.
- Future modules that make HTTP requests MUST enforce destination
  restrictions independently at connection time AND after every redirect,
  regardless of what this module returned.
- IPv4-mapped IPv6 addresses (e.g. ::ffff:192.168.1.1) are classified by
  their embedded IPv4 address, not by their IPv6 outer form, so that a
  private IPv4 address cannot bypass the block by being expressed in IPv6
  notation.
"""

from __future__ import annotations

import ipaddress
import socket
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Schemes that are always rejected regardless of other structure.
BLOCKED_SCHEMES: frozenset[str] = frozenset(
    {
        "file",
        "ftp",
        "ftps",
        "javascript",
        "data",
        "vbscript",
        "ldap",
        "ldaps",
        "dict",
        "gopher",
        "tftp",
        "sftp",
        "ssh",
        "telnet",
        "smtp",
        "imap",
        "pop3",
    }
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_scheme(scheme: str) -> str:
    return scheme.lower().strip()


def _encode_hostname_to_ace(hostname: str) -> tuple[str, list[str]]:
    """
    Convert a hostname to its ASCII-Compatible Encoding (ACE / IDNA) form.

    Returns (ace_hostname, warnings).

    Rules:
    - Empty string → returned as-is (caller must handle).
    - Trailing dot is preserved (it is legal in DNS and some parsers use it).
    - Each label is encoded independently; if encoding fails the original
      label is kept and a warning is added.
    - ASCII-only hostnames that are already valid pass through unchanged.
    """
    warnings: list[str] = []

    if not hostname:
        return hostname, warnings

    # Preserve a single trailing dot (FQDN notation).
    has_trailing_dot = hostname.endswith(".")
    work = hostname.rstrip(".")

    labels = work.split(".")
    ace_labels: list[str] = []

    for label in labels:
        if not label:
            # Empty label (e.g. double dot) — keep and let validation catch it.
            ace_labels.append(label)
            continue
        try:
            # IDNA encode via the 'idna' codec — works for both ASCII and
            # non-ASCII labels.
            ace_label = label.encode("idna").decode("ascii")
            ace_labels.append(ace_label)
        except (UnicodeError, UnicodeDecodeError):
            # Keep the original label; validation will flag it if illegal.
            ace_labels.append(label)
            warnings.append(
                f"Hostname label '{label}' could not be IDNA-encoded; "
                "the original form is retained."
            )

    result = ".".join(ace_labels)
    if has_trailing_dot:
        result += "."

    return result, warnings


def _classify_ip_address(
    raw: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address | None, list[str]]:
    """
    Attempt to parse *raw* as an IP address.

    IPv6 literals in URLs are wrapped in square brackets; strip them first.
    Returns (address_object, errors).  address_object is None if *raw* is
    not an IP address at all (i.e. it is a hostname).
    """
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    try:
        return ipaddress.ip_address(text), []
    except ValueError:
        return None, []


def _is_prohibited_address(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> tuple[bool, str]:
    """
    Return (True, reason) when *addr* is in a range that the application
    must never connect to.

    Checked ranges:
        loopback      — 127.0.0.0/8, ::1
        private       — RFC 1918 and IPv6 ULA (fc00::/7)
        link-local    — 169.254.0.0/16, fe80::/10
        multicast     — 224.0.0.0/4, ff00::/8
        reserved      — addresses reserved by IANA
        unspecified   — 0.0.0.0, ::

    IPv4-mapped IPv6 addresses (::ffff:x.x.x.x) are unwrapped and
    classified as their embedded IPv4 address.  This prevents a private
    IPv4 address from bypassing the block by being expressed in IPv6 form.
    """
    # Unwrap IPv4-mapped IPv6.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    if addr.is_loopback:
        return True, f"loopback address ({addr})"
    if addr.is_link_local:
        return True, f"link-local address ({addr})"
    if addr.is_private:
        return True, f"private address ({addr})"
    if addr.is_multicast:
        return True, f"multicast address ({addr})"
    if addr.is_reserved:
        return True, f"reserved address ({addr})"
    if addr.is_unspecified:
        return True, f"unspecified address ({addr})"

    return False, ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """
    Structured result returned by validate_url().

    Fields
    ------
    original_input : str
        The exact string passed to validate_url(), before any processing.
        Retained for audit / later analysis.  Do not display this in the UI
        because it may contain sensitive query parameters.

    normalized_url : str
        The canonical form of the URL after whitespace trimming, scheme
        lower-casing, and hostname IDNA-encoding.  Empty string on failure.

    is_valid : bool
        True only when the URL passed all syntax checks.  A valid URL is
        not necessarily legitimate or safe to fetch.

    scheme : str
        The URL scheme in lower case (e.g. "https").  Empty string on failure.

    hostname : str
        The hostname in ACE (IDNA) form, lower case.  Empty on failure.
        Trailing dots are preserved as supplied.

    port : int | None
        The explicitly stated port number, or None when no port was given.
        Implicit standard ports (80 for HTTP, 443 for HTTPS) are NOT added.

    path : str
        The URL path component.

    query : str
        The raw query string (everything after '?').
        WARNING — query strings can contain tokens, session IDs, and other
        sensitive data.  Do not log or display this field unnecessarily.

    fragment : str
        The fragment identifier (everything after '#').

    uses_https : bool
        True when the scheme is exactly "https".

    is_ip_address : bool
        True when the hostname is an IP address literal (IPv4 or IPv6).

    destination_status : str
        One of:
          "not_checked"          — default; no destination check performed.
          "blocked"              — prohibited by IP-address safety policy.
          "destination_unverified" — DNS check failed or inconclusive.
          "verified_for_policy"  — DNS check passed policy; NOT a safety cert.

    errors : list[str]
        Reasons the URL was rejected.  Empty when is_valid is True.

    warnings : list[str]
        Non-fatal observations (e.g. HTTP instead of HTTPS, IDN encoding
        notes, bare IP addresses).  Does not affect is_valid.
    """

    original_input: str = ""
    normalized_url: str = ""
    is_valid: bool = False
    scheme: str = ""
    hostname: str = ""
    port: int | None = None
    path: str = ""
    query: str = ""
    fragment: str = ""
    uses_https: bool = False
    is_ip_address: bool = False
    destination_status: str = "not_checked"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary — convenient for JSON serialisation."""
        return {
            "original_input": self.original_input,
            "normalized_url": self.normalized_url,
            "is_valid": self.is_valid,
            "scheme": self.scheme,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "query": self.query,
            "fragment": self.fragment,
            "uses_https": self.uses_https,
            "is_ip_address": self.is_ip_address,
            "destination_status": self.destination_status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _failed(original: str, errors: list[str], warnings: list[str] | None = None) -> ValidationResult:
    """Return a consistently structured failure result."""
    return ValidationResult(
        original_input=original,
        errors=errors,
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Public API — validate_url()
# ---------------------------------------------------------------------------


def validate_url(url_input: str) -> ValidationResult:
    """
    Validate a URL string submitted by an analyst.

    This function performs:
      1. Input sanitisation (empty / whitespace checks, trimming).
      2. Scheme inference for bare-hostname inputs.
      3. Scheme validation (allow HTTP/HTTPS; block others).
      4. URL parsing via urllib.parse.urlparse.
      5. Hostname extraction and IDNA normalisation.
      6. Credential (username / password) rejection.
      7. Port syntax and range validation.
      8. IP-address identification and prohibited-range checks.
      9. Construction of a structured ValidationResult.

    No network requests are made.
    No DNS resolution is performed.
    The result is NOT a statement about legitimacy or safety.

    Parameters
    ----------
    url_input : str
        The raw URL string from the analyst.

    Returns
    -------
    ValidationResult
        Always returns a ValidationResult — never raises an unhandled
        exception on bad input.
    """
    # ------------------------------------------------------------------
    # 1. Empty / whitespace check
    # ------------------------------------------------------------------
    if not isinstance(url_input, str):
        return _failed("", ["Input must be a string."])

    original_input = url_input  # keep the exact original before any change

    stripped = url_input.strip()

    if not stripped:
        return _failed(
            original_input,
            ["Input is empty or contains only whitespace."],
        )

    # ------------------------------------------------------------------
    # 2. Scheme inference
    # ------------------------------------------------------------------
    # If the string has no "://" it may be a bare hostname.
    # We add "https://" ONLY when the remainder looks like a plausible
    # hostname: contains at least one dot (or is an IPv6 literal in []),
    # and contains no spaces or illegal characters that would turn it
    # into an arbitrary text string after prepending a scheme.
    # "localhost" is intentionally not treated as a bare hostname because
    # it is a loopback alias; analysts are unlikely to mean it as a target.

    working = stripped

    scheme_inferred = False
    if "://" not in working:
        # Check for scheme-like prefix without '://' (e.g. "javascript:alert")
        # to catch schemeless-but-blocked cases early.
        colon_pos = working.find(":")
        if colon_pos != -1:
            potential_scheme = working[:colon_pos].lower()
            if potential_scheme in BLOCKED_SCHEMES:
                return _failed(
                    original_input,
                    [f"URL scheme '{potential_scheme}' is not supported. "
                     "Only 'http' and 'https' are accepted."],
                )
            # If it looks like an unsupported scheme without '//', reject.
            if (
                potential_scheme.isalpha()
                and potential_scheme not in SUPPORTED_SCHEMES
                # Allow port-only colons like "example.com:8080" — no alpha-only
                # segment before the colon that would look like a scheme.
                and not working[:colon_pos].replace(".", "").replace("-", "").isdigit()
            ):
                # Could be "example.com:8080" — don't reject yet; fall through.
                pass

        # Bare hostname / path — attempt to infer https://
        candidate = working
        # A plausible bare hostname must contain a dot OR be an IPv6 literal.
        has_dot = "." in candidate.split("/")[0]
        is_ipv6_literal = candidate.lstrip().startswith("[")
        has_space = " " in candidate

        if has_space:
            return _failed(
                original_input,
                ["Input contains spaces and is not a valid URL."],
            )

        if has_dot or is_ipv6_literal:
            working = "https://" + working
            scheme_inferred = True
        else:
            return _failed(
                original_input,
                [
                    "No URL scheme found (e.g. 'https://') and the input does "
                    "not look like a bare hostname. "
                    "Please provide a full URL including the scheme."
                ],
            )

    # ------------------------------------------------------------------
    # 3. Scheme validation
    # ------------------------------------------------------------------
    scheme_end = working.find("://")
    raw_scheme = working[:scheme_end].lower() if scheme_end != -1 else ""

    if raw_scheme in BLOCKED_SCHEMES:
        return _failed(
            original_input,
            [f"URL scheme '{raw_scheme}' is not supported. "
             "Only 'http' and 'https' are accepted."],
        )

    if raw_scheme not in SUPPORTED_SCHEMES:
        return _failed(
            original_input,
            [f"URL scheme '{raw_scheme}' is not supported. "
             "Only 'http' and 'https' are accepted."],
        )

    # ------------------------------------------------------------------
    # 4. Parse with urllib.parse
    # ------------------------------------------------------------------
    try:
        parsed = urlparse(working)
    except Exception as exc:  # pragma: no cover — urlparse rarely raises
        return _failed(original_input, [f"URL could not be parsed: {exc}"])

    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 5. Credential check (username / password in URL)
    # ------------------------------------------------------------------
    if parsed.username is not None or parsed.password is not None:
        errors.append(
            "URLs containing embedded credentials (username or password) "
            "are not accepted."
        )

    # ------------------------------------------------------------------
    # 6. Hostname extraction and IDNA normalisation
    # ------------------------------------------------------------------
    raw_hostname = parsed.hostname or ""  # lower-cased by urlparse

    if not raw_hostname:
        errors.append("URL is missing a hostname.")
        return _failed(original_input, errors, warnings)

    # Reject hostnames with spaces (urlparse may split weirdly on some inputs).
    if " " in raw_hostname:
        errors.append(f"Hostname '{raw_hostname}' contains spaces and is invalid.")
        return _failed(original_input, errors, warnings)

    # IDNA / ACE encoding.
    ace_hostname, idn_warnings = _encode_hostname_to_ace(raw_hostname)
    warnings.extend(idn_warnings)

    # ------------------------------------------------------------------
    # 7. Port validation
    # ------------------------------------------------------------------
    port: int | None = None
    raw_netloc = parsed.netloc  # e.g. "example.com:99999"

    # Extract the port from the netloc string to detect syntax errors
    # that urlparse might swallow.
    try:
        port = parsed.port  # None if absent; raises ValueError if malformed
    except ValueError:
        errors.append(
            f"The URL contains an invalid port specification in '{raw_netloc}'."
        )

    if port is not None:
        if not (1 <= port <= 65535):
            errors.append(
                f"Port {port} is outside the valid range 1–65535."
            )

    # ------------------------------------------------------------------
    # 8. IP address identification and prohibited-range check
    # ------------------------------------------------------------------
    ip_addr, _ = _classify_ip_address(ace_hostname)
    is_ip_address = ip_addr is not None

    destination_status = "not_checked"

    if is_ip_address:
        warnings.append(
            f"The hostname is a direct IP address ({ace_hostname}). "
            "Direct IP addresses are unusual for legitimate services."
        )
        prohibited, reason = _is_prohibited_address(ip_addr)
        if prohibited:
            errors.append(
                f"The destination IP address is prohibited ({reason}). "
                "Requests to this address are blocked by security policy."
            )
            destination_status = "blocked"
        else:
            warnings.append(
                "The IP address is publicly routable but has not been "
                "verified as a legitimate destination."
            )
            # Public IP — not blocked, but not verified either.
            # Leave destination_status as "not_checked".

    # ------------------------------------------------------------------
    # 9. HTTP vs HTTPS warning
    # ------------------------------------------------------------------
    uses_https = (raw_scheme == "https")
    if not uses_https and raw_scheme == "http":
        warnings.append(
            "The URL uses plain HTTP. Communications will not be encrypted. "
            "Prefer HTTPS where possible."
        )

    # ------------------------------------------------------------------
    # 10. Scheme-inferred warning
    # ------------------------------------------------------------------
    if scheme_inferred:
        warnings.append(
            "No scheme was provided; 'https://' was assumed. "
            "If a different scheme was intended, supply the full URL."
        )

    # ------------------------------------------------------------------
    # 11. Assemble result
    # ------------------------------------------------------------------
    if errors:
        return ValidationResult(
            original_input=original_input,
            is_valid=False,
            scheme=raw_scheme,
            hostname=ace_hostname,
            port=port,
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment,
            uses_https=uses_https,
            is_ip_address=is_ip_address,
            destination_status=destination_status,
            errors=errors,
            warnings=warnings,
        )

    # Rebuild a normalised URL using the ACE hostname.
    normalized_netloc = ace_hostname
    if port is not None:
        normalized_netloc = f"{ace_hostname}:{port}"
    normalized_url = urlunparse(
        (raw_scheme, normalized_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )

    return ValidationResult(
        original_input=original_input,
        normalized_url=normalized_url,
        is_valid=True,
        scheme=raw_scheme,
        hostname=ace_hostname,
        port=port,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
        uses_https=uses_https,
        is_ip_address=is_ip_address,
        destination_status=destination_status,
        errors=[],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public API — check_dns_destination()
# ---------------------------------------------------------------------------


def check_dns_destination(hostname: str) -> dict[str, Any]:
    """
    Perform a DNS lookup for *hostname* and check each returned IP address
    against the prohibited-address policy.

    This function is SEPARATE from validate_url() and must be called
    explicitly when destination verification is required.  It is NOT called
    automatically during syntax validation.

    Security limitations (documented deliberately)
    -----------------------------------------------
    DNS rebinding:
        The IP address returned now may differ from the address the server
        uses when a connection is actually opened.  An attacker who controls
        a DNS server can return a public address for the first query and a
        private address for subsequent queries.  This function is therefore
        NOT a complete SSRF defence.

    Time-of-check / time-of-use (TOCTOU):
        Even a successful result from this function does not guarantee the
        destination will be appropriate when the connection is made.

    Future modules must re-enforce destination restrictions at connection
    time and after every redirect, independently of what this function returns.

    Parameters
    ----------
    hostname : str
        The ACE-form hostname from a ValidationResult (result.hostname).

    Returns
    -------
    dict with keys:
        destination_status : str   — "verified_for_policy",
                                     "blocked", or "destination_unverified"
        resolved_ips       : list  — IP address strings that were returned.
        errors             : list  — reasons for 'blocked' or 'unverified'.
        warnings           : list  — non-fatal notes.
    """
    # Strip trailing dot for DNS lookup (some resolvers accept it; some don't).
    lookup_hostname = hostname.rstrip(".")

    resolved_ips: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    try:
        addr_infos = socket.getaddrinfo(lookup_hostname, None)
    except socket.gaierror as exc:
        return {
            "destination_status": "destination_unverified",
            "resolved_ips": [],
            "errors": [f"DNS resolution failed for '{lookup_hostname}': {exc}"],
            "warnings": [],
        }

    seen: set[str] = set()
    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        if ip_str not in seen:
            seen.add(ip_str)
            resolved_ips.append(ip_str)

    if not resolved_ips:
        return {
            "destination_status": "destination_unverified",
            "resolved_ips": [],
            "errors": [f"DNS returned no addresses for '{lookup_hostname}'."],
            "warnings": [],
        }

    # Check every returned address — all must pass.
    prohibited_found = False
    for ip_str in resolved_ips:
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            errors.append(f"Could not parse resolved address '{ip_str}'.")
            prohibited_found = True
            continue

        # Unwrap IPv4-mapped IPv6.
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
            warnings.append(
                f"Address '{ip_str}' is an IPv4-mapped IPv6 address; "
                f"classified as its embedded IPv4 address ({addr})."
            )

        prohibited, reason = _is_prohibited_address(addr)
        if prohibited:
            errors.append(
                f"Resolved address '{ip_str}' is prohibited ({reason})."
            )
            prohibited_found = True

    if prohibited_found:
        return {
            "destination_status": "blocked",
            "resolved_ips": resolved_ips,
            "errors": errors,
            "warnings": warnings,
        }

    return {
        "destination_status": "verified_for_policy",
        "resolved_ips": resolved_ips,
        "errors": [],
        "warnings": warnings + [
            "DNS check passed policy. This is NOT a safety certificate. "
            "Destination restrictions must be re-enforced at connection time "
            "and after every redirect."
        ],
    }
