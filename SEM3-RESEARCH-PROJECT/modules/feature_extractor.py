"""
modules/feature_extractor.py
============================
Module 5 — Traditional URL, Domain and Webpage Feature Extraction.

Purpose
-------
Extract a deterministic, machine-learning-compatible set of traditional URL
and hostname features from a validated URL.  These features supply the inputs
for Model 1 in the research experiment comparing:

  Model 1 — traditional URL / domain / webpage features only.
  Model 2 — traditional features PLUS website-security configuration features.

This module does NOT make any network requests.
This module does NOT classify URLs as phishing or legitimate.
This module does NOT train or invoke a machine-learning model.
This module does NOT perform DNS lookups, WHOIS queries, or reputation checks.
This module does NOT fetch or render webpage content.

Calling convention
------------------
Call validate_url() first.  Pass the resulting ValidationResult to
extract_features().  Do not call extract_features() on a URL that failed
validation.

    from modules.url_validator import validate_url
    from modules.feature_extractor import extract_features

    vr = validate_url("https://sub.example.com/login?user=a")
    if vr.is_valid:
        features = extract_features(vr)

Output
------
extract_features() returns a FeatureResult dataclass with two attribute groups:

  descriptive  — human-readable metadata (strings, lists); not intended as
                  direct ML inputs.
  numeric      — a flat dict[str, int | float | bool] suitable for a pandas
                  DataFrame row.  Feature names are stable across calls.

Public suffix determination
---------------------------
A URL hostname such as "sub.example.co.uk" has the public suffix "co.uk",
the registered domain "example.co.uk", and one subdomain label "sub".

This module bundles a small, documented set of common two-label public
suffixes (e.g. co.uk, com.au, co.in).  When the hostname matches a known
two-label suffix the module uses it; otherwise it falls back to a single
final label as the TLD.  When the result is ambiguous the module sets
registered_domain to the sentinel string "unknown" and subdomain_count
to -1 to make the uncertainty explicit rather than silently wrong.

This approach avoids a hard external dependency on the Mozilla Public Suffix
List while being honest about its limitations.  Future versions should use
tldextract or a similar library for production accuracy.

Keyword features
----------------
Keywords are matched case-insensitively against the full normalised URL
(scheme + hostname + path + query + fragment).  A match is a research
observation, not a phishing verdict.  Legitimate websites frequently use
terms such as "login" and "account".

URL-shortener detection
-----------------------
The module maintains a small, documented list of well-known URL shortener
hostnames.  Matching is exact (full hostname, case-insensitive).
A URL-shortener match is an observation, not a verdict.

Research notes
--------------
- All features are deterministic: the same URL always produces the same
  numeric vector when the module configuration is unchanged.
- Feature names follow the pattern <group>_<name> for easy column selection
  in a DataFrame.
- Boolean features are stored as Python bool (JSON-serialisable; pandas
  treats them as numpy bool_ when constructing a DataFrame).
- Missing or uncomputable values are represented as -1 (int features) or
  False (bool features), never as None, NaN, or silently invented values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

from modules.url_validator import ValidationResult

# ---------------------------------------------------------------------------
# Configuration — keyword list
# ---------------------------------------------------------------------------
# These terms appear frequently in phishing URLs but also in legitimate sites.
# They are research observations, not verdicts.

PHISHING_KEYWORDS: tuple[str, ...] = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "signin",
    "sign-in",
    "wallet",
    "payment",
    "confirm",
    "banking",
    "netbanking",
    "portal",
    "authenticate",
    "credential",
    "password",
    "passwd",
    "reactivate",
    "recover",
    "unlock",
)

# ---------------------------------------------------------------------------
# Configuration — URL shortener hostnames
# ---------------------------------------------------------------------------
# Exact match on the full hostname (after ACE normalisation, lower-case).
# This list is intentionally small and documented.

URL_SHORTENERS: frozenset[str] = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "buff.ly",
        "is.gd",
        "rb.gy",
        "cutt.ly",
        "shorte.st",
        "tiny.cc",
        "bl.ink",
        "rebrand.ly",
    }
)

# ---------------------------------------------------------------------------
# Configuration — known two-label public suffixes (partial list)
# ---------------------------------------------------------------------------
# This is NOT a complete public suffix list.  It covers suffixes common in
# Indian and other South/Southeast Asian ccTLDs plus a few widely-used ones.
# When a hostname's last two labels match an entry here the module uses the
# two-label suffix to compute the registered domain.  Otherwise it falls back
# to using just the single final label as the TLD.
#
# Limitation: this heuristic is imprecise for unusual or newer suffixes.
# For production use, replace with tldextract + the Mozilla PSL.

_TWO_LABEL_SUFFIXES: frozenset[str] = frozenset(
    {
        # India
        "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
        "ac.in", "edu.in", "res.in", "nic.in", "gov.in", "mil.in",
        # UK
        "co.uk", "org.uk", "me.uk", "net.uk", "ltd.uk", "plc.uk",
        "sch.uk", "ac.uk", "gov.uk", "mod.uk", "nhs.uk",
        # Australia
        "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au",
        "id.au",
        # New Zealand
        "co.nz", "net.nz", "org.nz", "school.nz", "govt.nz",
        # Other common
        "com.br", "org.br", "net.br", "gov.br", "edu.br",
        "com.mx", "org.mx", "net.mx", "gob.mx", "edu.mx",
        "com.ar", "net.ar", "org.ar", "gov.ar", "edu.ar",
        "com.sg", "edu.sg", "gov.sg", "net.sg", "org.sg",
        "com.hk", "org.hk", "net.hk", "gov.hk", "edu.hk",
        "com.my", "net.my", "org.my", "gov.my", "edu.my",
        "com.pk", "net.pk", "org.pk", "gov.pk", "edu.pk",
        "co.za", "net.za", "org.za", "gov.za", "edu.za",
        "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp",
    }
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_char(text: str, char: str) -> int:
    """Count occurrences of a single character in *text*."""
    return text.count(char)


def _has_punycode(hostname: str) -> bool:
    """
    Return True when any hostname label starts with the ACE prefix 'xn--',
    which indicates an internationalised domain name encoded as punycode.
    """
    return any(label.startswith("xn--") for label in hostname.split("."))


def _split_registered_domain(hostname: str) -> tuple[str, str, int]:
    """
    Split *hostname* into (registered_domain, public_suffix, subdomain_count).

    Uses _TWO_LABEL_SUFFIXES for a limited two-label suffix lookup.
    Falls back to the single final label as the TLD.
    Returns ("unknown", "unknown", -1) when the result is ambiguous
    (e.g. fewer labels than needed, IP address, or empty input).

    subdomain_count is the number of labels that precede the registered domain.
    For "www.sub.example.co.in": suffix="co.in", rd="example.co.in",
    subdomain_count=2 (["www", "sub"]).
    """
    if not hostname or hostname == "unknown":
        return "unknown", "unknown", -1

    # IP addresses (IPv4 or IPv6) have no registered domain.
    import ipaddress as _ip
    try:
        _ip.ip_address(hostname.strip("[]"))
        return "unknown", "unknown", -1
    except ValueError:
        pass  # Not an IP address — proceed with label splitting.

    # Strip trailing dot (FQDN notation).
    clean = hostname.rstrip(".")
    labels = clean.split(".")

    if len(labels) < 2:
        # Single-label hostname — cannot determine registered domain.
        return "unknown", "unknown", -1

    # Try two-label suffix first.
    candidate_two = f"{labels[-2]}.{labels[-1]}"
    if candidate_two in _TWO_LABEL_SUFFIXES:
        if len(labels) < 3:
            # e.g. bare "co.in" — no registered domain label present.
            return "unknown", candidate_two, -1
        public_suffix = candidate_two
        registered_domain = f"{labels[-3]}.{public_suffix}"
        subdomain_count = len(labels) - 3  # labels before the registered part
        return registered_domain, public_suffix, subdomain_count

    # Fall back to single-label TLD.
    public_suffix = labels[-1]
    registered_domain = f"{labels[-2]}.{public_suffix}"
    subdomain_count = len(labels) - 2
    return registered_domain, public_suffix, subdomain_count


def _match_keywords(url_lower: str) -> list[str]:
    """
    Return the subset of PHISHING_KEYWORDS found in *url_lower* (case-
    insensitive matching is handled by pre-lowercasing the caller's input).
    Each keyword is returned at most once even if it appears multiple times.
    """
    found: list[str] = []
    for kw in PHISHING_KEYWORDS:
        if kw in url_lower:
            found.append(kw)
    return found


def _non_default_port(scheme: str, port: int | None) -> bool:
    """
    Return True when *port* is explicitly set AND differs from the default
    for the scheme (80 for HTTP, 443 for HTTPS).
    """
    if port is None:
        return False
    defaults = {"http": 80, "https": 443}
    return port != defaults.get(scheme, -1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    """
    Structured output from extract_features().

    Attributes
    ----------
    url : str
        The normalised URL that was analysed (from ValidationResult).
        Contains no credentials; do not log the original_input.

    descriptive : dict[str, Any]
        Human-readable metadata — strings and lists.  Not intended as direct
        ML inputs.

    numeric : dict[str, int | float | bool]
        Flat dictionary of numeric/boolean features for ML use.  Every key is
        always present; missing values are -1 (int) or False (bool).
        Suitable for direct use as a pandas DataFrame row.

    keywords_matched : list[str]
        The subset of PHISHING_KEYWORDS found in the URL (lower-case).

    warnings : list[str]
        Non-fatal notes about the extraction (e.g. registered-domain unknown).
    """
    url: str = ""
    descriptive: dict[str, Any] = field(default_factory=dict)
    numeric: dict[str, Any] = field(default_factory=dict)
    keywords_matched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict merging descriptive and numeric fields."""
        return {
            "url": self.url,
            **self.descriptive,
            **self.numeric,
            "keywords_matched": list(self.keywords_matched),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(vr: ValidationResult) -> FeatureResult:
    """
    Extract traditional URL and hostname features from a validated URL.

    Parameters
    ----------
    vr : ValidationResult
        A result object returned by validate_url().  Must have is_valid=True.
        Passing an invalid result raises ValueError immediately.

    Returns
    -------
    FeatureResult
        Contains descriptive metadata and a numeric feature dict suitable
        for a pandas DataFrame row.

    Raises
    ------
    TypeError
        If *vr* is not a ValidationResult instance.
    ValueError
        If *vr.is_valid* is False.

    Notes
    -----
    - No network requests are made.
    - Extraction is deterministic: the same ValidationResult always produces
      the same FeatureResult when module configuration is unchanged.
    - boolean features are stored as Python bool.
    - Integer features missing a computable value are stored as -1.
    """
    if not isinstance(vr, ValidationResult):
        raise TypeError(
            f"extract_features() expects a ValidationResult, got {type(vr).__name__}."
        )
    if not vr.is_valid:
        raise ValueError(
            "extract_features() requires a valid URL (vr.is_valid must be True). "
            "Run validate_url() first and check is_valid before extracting features."
        )

    warnings: list[str] = []

    # ── Build the full normalised URL string for whole-URL measurements ──────
    # Use the normalised_url from the validator; fall back to reconstructing
    # from components if it is empty.
    full_url = vr.normalized_url or (
        f"{vr.scheme}://{vr.hostname}"
        + (f":{vr.port}" if vr.port is not None else "")
        + vr.path
        + (f"?{vr.query}" if vr.query else "")
        + (f"#{vr.fragment}" if vr.fragment else "")
    )

    url_lower = full_url.lower()

    # ── A. General URL features ──────────────────────────────────────────────

    url_length: int = len(full_url)
    hostname_length: int = len(vr.hostname)
    path_length: int = len(vr.path)
    query_length: int = len(vr.query)
    fragment_length: int = len(vr.fragment)

    # Count query parameters (parse_qs splits on & and ;).
    # An empty query string ("?") yields 0 parameters but has_query=True.
    num_query_params: int = 0
    has_query: bool = False
    if vr.query:
        has_query = True
        num_query_params = len(parse_qs(vr.query, keep_blank_values=True))

    has_fragment: bool = bool(vr.fragment)

    # Character counts across the full URL.
    char_count_dot: int        = _count_char(full_url, ".")
    char_count_hyphen: int     = _count_char(full_url, "-")
    char_count_underscore: int = _count_char(full_url, "_")
    char_count_slash: int      = _count_char(full_url, "/")
    char_count_percent: int    = _count_char(full_url, "%")
    char_count_at: int         = _count_char(full_url, "@")

    # Presence of @ — in a validated URL this should never appear in the
    # authority (the validator rejects embedded credentials), but it may
    # appear in the path or query.
    has_at_symbol: bool = char_count_at > 0

    # Non-default port.
    has_non_default_port: bool = _non_default_port(vr.scheme, vr.port)

    # ── B. Hostname and domain features ─────────────────────────────────────

    hostname = vr.hostname  # ACE form, lower-case, from the validator.

    hostname_labels = [lb for lb in hostname.rstrip(".").split(".") if lb]
    num_hostname_labels: int = len(hostname_labels)

    # Registered domain and subdomain count.
    registered_domain, public_suffix, subdomain_count = _split_registered_domain(hostname)

    if registered_domain == "unknown":
        warnings.append(
            f"Registered domain could not be determined for hostname '{hostname}'. "
            "This may be an IP address, a single-label host, or an unrecognised "
            "public suffix. Registered-domain features are set to sentinel values."
        )
        registered_domain_length: int = -1
    else:
        registered_domain_length = len(registered_domain)

    has_punycode: bool = _has_punycode(hostname)
    is_ip_address: bool = vr.is_ip_address

    is_url_shortener: bool = hostname in URL_SHORTENERS

    # ── C. Keyword features ──────────────────────────────────────────────────

    matched_keywords = _match_keywords(url_lower)
    num_keywords_matched: int = len(matched_keywords)
    has_keyword_match: bool = num_keywords_matched > 0

    # ── D. Scheme feature ────────────────────────────────────────────────────

    uses_https: bool = vr.uses_https

    # ── Assemble descriptive dict ────────────────────────────────────────────

    descriptive: dict[str, Any] = {
        "hostname":           hostname,
        "scheme":             vr.scheme,
        "registered_domain":  registered_domain,
        "public_suffix":      public_suffix,
        "keywords_found":     list(matched_keywords),
    }

    # ── Assemble numeric dict ────────────────────────────────────────────────
    # Key naming convention: <group>_<name>
    #   url_*      — whole-URL measurements
    #   host_*     — hostname measurements
    #   domain_*   — registered-domain measurements
    #   char_*     — character count features
    #   kw_*       — keyword features
    #   flag_*     — boolean indicator features

    numeric: dict[str, Any] = {
        # URL length features
        "url_length":               url_length,
        "url_hostname_length":      hostname_length,
        "url_path_length":          path_length,
        "url_query_length":         query_length,
        "url_fragment_length":      fragment_length,
        "url_num_query_params":     num_query_params,
        # Character count features
        "char_count_dot":           char_count_dot,
        "char_count_hyphen":        char_count_hyphen,
        "char_count_underscore":    char_count_underscore,
        "char_count_slash":         char_count_slash,
        "char_count_percent":       char_count_percent,
        "char_count_at":            char_count_at,
        # Hostname features
        "host_num_labels":          num_hostname_labels,
        "host_subdomain_count":     subdomain_count,
        "domain_registered_length": registered_domain_length,
        # Keyword features
        "kw_num_matched":           num_keywords_matched,
        # Boolean / flag features
        "flag_uses_https":          uses_https,
        "flag_is_ip_address":       is_ip_address,
        "flag_has_non_default_port":has_non_default_port,
        "flag_has_at_symbol":       has_at_symbol,
        "flag_has_query":           has_query,
        "flag_has_fragment":        has_fragment,
        "flag_has_punycode":        has_punycode,
        "flag_has_keyword":         has_keyword_match,
        "flag_is_url_shortener":    is_url_shortener,
    }

    return FeatureResult(
        url=full_url,
        descriptive=descriptive,
        numeric=numeric,
        keywords_matched=list(matched_keywords),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convenience — canonical feature names list
# ---------------------------------------------------------------------------

def numeric_feature_names() -> list[str]:
    """
    Return the ordered list of keys that every numeric dict will contain.

    Use this to build a consistent DataFrame schema before any URLs have
    been processed.
    """
    return [
        "url_length",
        "url_hostname_length",
        "url_path_length",
        "url_query_length",
        "url_fragment_length",
        "url_num_query_params",
        "char_count_dot",
        "char_count_hyphen",
        "char_count_underscore",
        "char_count_slash",
        "char_count_percent",
        "char_count_at",
        "host_num_labels",
        "host_subdomain_count",
        "domain_registered_length",
        "kw_num_matched",
        "flag_uses_https",
        "flag_is_ip_address",
        "flag_has_non_default_port",
        "flag_has_at_symbol",
        "flag_has_query",
        "flag_has_fragment",
        "flag_has_punycode",
        "flag_has_keyword",
        "flag_is_url_shortener",
    ]
