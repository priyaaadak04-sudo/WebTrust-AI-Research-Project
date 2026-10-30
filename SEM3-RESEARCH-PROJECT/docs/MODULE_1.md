# Module 1 — URL Input and Validation

**Project:** Impact of Website Security Features on ML-Based Phishing Detection  
**Module:** 1 of N  
**Status:** Complete  
**Author:** MCA Semester 3 Research Team  
**Last updated:** 2025

---

## Table of contents

1. [Purpose](#1-purpose)
2. [What this module does NOT do](#2-what-this-module-does-not-do)
3. [Public API](#3-public-api)
4. [ValidationResult fields](#4-validationresult-fields)
5. [Destination status values](#5-destination-status-values)
6. [Normalisation policy](#6-normalisation-policy)
7. [Prohibited destination ranges](#7-prohibited-destination-ranges)
8. [DNS destination checking](#8-dns-destination-checking)
9. [Security limitations and future module responsibilities](#9-security-limitations-and-future-module-responsibilities)
10. [How to run](#10-how-to-run)
11. [How to test](#11-how-to-test)
12. [Integration guidance for future modules](#12-integration-guidance-for-future-modules)
13. [Design decisions](#13-design-decisions)

---

## 1. Purpose

Module 1 accepts a URL string submitted by an analyst, verifies that it is
syntactically valid, extracts its components into a structured result, and
performs preliminary checks to identify destinations that the application
**must not** access.

The module prepares structured output that later modules (feature extraction,
model inference, reporting) can consume without repeating basic parsing logic.

---

## 2. What this module does NOT do

| Action | Reason |
|--------|--------|
| Make HTTP requests | Only Module 3 (feature extraction) should fetch pages. |
| Perform DNS lookups during syntax validation | DNS is optional and must be called explicitly via `check_dns_destination()`. |
| Classify URLs as phishing or legitimate | That is the responsibility of the ML modules. |
| Guarantee that a valid URL is safe to access | Validity means the URL is well-formed, not that the destination is trustworthy. |
| Guarantee that HTTPS means the site is legitimate | Phishing sites routinely use HTTPS and valid certificates. |

---

## 3. Public API

```python
from modules.url_validator import validate_url, check_dns_destination

# Syntax validation (no network requests)
result = validate_url("https://example.com/login")

# Optional DNS check (makes a real DNS query — call explicitly)
dns_result = check_dns_destination(result.hostname)
```

### `validate_url(url_input: str) -> ValidationResult`

Performs:
- Input sanitisation (empty/whitespace rejection, trimming).
- Scheme inference for bare-hostname inputs (see [Normalisation policy](#6-normalisation-policy)).
- Scheme allow-list enforcement.
- URL parsing using `urllib.parse.urlparse`.
- Hostname IDNA normalisation.
- Embedded-credential rejection.
- Port syntax and range validation (1–65535).
- IP address identification and prohibited-range checking.
- Returns a `ValidationResult` dataclass (see §4).

**Never raises an unhandled exception on bad input.** Always returns a
`ValidationResult` with `is_valid=False` and a populated `errors` list.

### `check_dns_destination(hostname: str) -> dict`

- Resolves the hostname using `socket.getaddrinfo`.
- Checks **every** returned IP address against the prohibited-range policy.
- Returns `"verified_for_policy"`, `"blocked"`, or `"destination_unverified"`.
- **Must be called explicitly.** It is never called automatically by `validate_url()`.
- See §8 for details and security limitations.

---

## 4. ValidationResult fields

| Field | Type | Description |
|-------|------|-------------|
| `original_input` | `str` | Exact string passed to `validate_url()` before any processing. Retained for audit. **Do not display in UI or log unnecessarily.** |
| `normalized_url` | `str` | Canonical URL after whitespace trimming, scheme lowercasing, and IDNA hostname encoding. Empty string on failure. |
| `is_valid` | `bool` | `True` only when all syntax checks pass. A valid URL is **not** necessarily legitimate or safe to fetch. |
| `scheme` | `str` | Lowercased scheme (`"http"` or `"https"`). Empty on failure. |
| `hostname` | `str` | Hostname in ACE (IDNA) form, lowercased. Trailing dot preserved. Empty on failure. |
| `port` | `int \| None` | Explicitly stated port, or `None`. Implicit ports (80/443) are **not** added. |
| `path` | `str` | URL path component. |
| `query` | `str` | Raw query string after `?`. **May contain tokens, session IDs, or other sensitive data.** Do not log or display unnecessarily. |
| `fragment` | `str` | Fragment identifier after `#`. |
| `uses_https` | `bool` | `True` when scheme is exactly `"https"`. |
| `is_ip_address` | `bool` | `True` when the hostname is an IP address literal (IPv4 or IPv6). |
| `destination_status` | `str` | See §5. |
| `errors` | `list[str]` | Reasons for rejection. Empty when `is_valid` is `True`. |
| `warnings` | `list[str]` | Non-fatal observations. Does not affect `is_valid`. |

---

## 5. Destination status values

| Status | Meaning |
|--------|---------|
| `"not_checked"` | Default. No destination check has been attempted. Validation passed but no claim is made about safety. |
| `"blocked"` | The destination is prohibited by security policy (loopback, private, link-local, multicast, reserved, or unspecified address). |
| `"destination_unverified"` | A DNS check was attempted but failed or was inconclusive. The destination must **not** be treated as safe. |
| `"verified_for_policy"` | DNS resolution succeeded and no returned address was in a prohibited range. This is **not** a safety certificate — see §8. |

---

## 6. Normalisation policy

1. **Whitespace** — Leading and trailing whitespace is stripped before any other processing.
2. **Scheme inference** — If the input has no `://`, the module attempts to treat the input as a bare hostname by prepending `https://`, subject to these conditions:
   - The input must contain at least one dot in the hostname part (or be an IPv6 literal in brackets).
   - The input must contain no spaces.
   - The input must not begin with a known blocked scheme followed by `:`.
   - If these conditions are not met, the input is rejected.
3. **Scheme casing** — The scheme is lowercased.
4. **Hostname casing** — The hostname is lowercased by `urllib.parse`.
5. **IDNA encoding** — Each hostname label is encoded to its ASCII-Compatible Encoding (ACE) form using the `idna` codec. For example, `例え.jp` → `xn--r8jz45g.jp`. The `hostname` field always contains the ACE form.
6. **Trailing dot** — A single trailing dot on the hostname (FQDN notation) is preserved.
7. **Ports** — The explicit port is extracted and stored as an integer. Implicit standard ports are **not** added.

---

## 7. Prohibited destination ranges

The module uses Python's `ipaddress` module to classify IP-address hostnames.
The following ranges are blocked (`destination_status = "blocked"`, `is_valid = False`):

| Range | Category | Risk |
|-------|----------|------|
| `127.0.0.0/8`, `::1` | Loopback | SSRF — accesses the application server itself. |
| RFC 1918: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`; IPv6 ULA `fc00::/7` | Private | SSRF — accesses internal network services. |
| `169.254.0.0/16`, `fe80::/10` | Link-local | SSRF — APIPA, AWS/GCP metadata endpoints. |
| `224.0.0.0/4`, `ff00::/8` | Multicast | Not a reachable service. |
| IANA-reserved ranges | Reserved | Not publicly routable. |
| `0.0.0.0`, `::` | Unspecified | Not a valid destination. |

**IPv4-mapped IPv6 addresses** (e.g., `::ffff:192.168.1.1`) are unwrapped to
their embedded IPv4 address before classification. This prevents a private
IPv4 address from bypassing the block by being expressed in IPv6 notation.

A **publicly routable IP address** is not blocked but receives a warning.
It is not verified as a legitimate destination.

---

## 8. DNS destination checking

`check_dns_destination(hostname)` resolves the hostname and checks every
returned address.

### Usage

```python
result = validate_url("https://example.com/login")
if result.is_valid:
    dns_result = check_dns_destination(result.hostname)
    print(dns_result["destination_status"])
    print(dns_result["resolved_ips"])
```

### Return structure

```python
{
    "destination_status": "verified_for_policy",  # or "blocked" / "destination_unverified"
    "resolved_ips": ["93.184.216.34"],
    "errors": [],
    "warnings": ["DNS check passed policy. This is NOT a safety certificate. ..."]
}
```

### Security limitations

**DNS rebinding:**  
The IP address returned during this check may differ from the address the
server uses when a connection is actually opened. An attacker who controls a
DNS server can return a public address for the first query and a private
address when the application server makes the actual connection. A one-time
DNS check is therefore **not** a complete SSRF defence.

**Time-of-check / time-of-use (TOCTOU):**  
A `"verified_for_policy"` result does not guarantee the destination will be
appropriate when the connection is eventually made.

**All addresses are checked:**  
`check_dns_destination()` examines every address returned by
`getaddrinfo()`, not just the first one. If any address is in a prohibited
range, the result is `"blocked"`.

**DNS failure → unverified, not safe:**  
If DNS resolution fails, the result is `"destination_unverified"`. This must
**not** be treated as permission to proceed.

---

## 9. Security limitations and future module responsibilities

### Module 1 limitations

- Module 1 never accesses the destination URL.
- A `"not_checked"` or `"verified_for_policy"` result is **not** permission to
  fetch the URL.
- Module 1 does not detect every possible form of URL manipulation (e.g.,
  percent-encoded characters in paths that may be interpreted differently by
  servers).
- Module 1 does not analyse page content, check certificates, or evaluate the
  reputation of a domain.

### Responsibilities of the future fetching module (Module 3)

When Module 3 is implemented, it **must** independently enforce:

| Protection | Detail |
|-----------|--------|
| Destination validation | Re-check the resolved IP before connecting — do not rely on Module 1's result. |
| Redirect safety | Re-validate and re-check the destination after every redirect. |
| Private IP block | Enforce the same prohibited-range policy at connection time. |
| Connection timeout | Set a strict timeout to prevent hanging on slow or malicious servers. |
| Response size limit | Limit the maximum response body size. |
| TLS verification | Verify the server's TLS certificate unless deliberately disabled for testing. |

Not doing so leaves the application vulnerable to SSRF attacks regardless of
what Module 1 validated.

---

## 10. How to run

### Confirm Python is installed (Windows PowerShell)

```powershell
py --version
```

### (Optional) Create and activate a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

Module 1 uses only the Python standard library. No packages need to be installed.

```powershell
pip install -r requirements.txt
# (no packages will be installed for Module 1)
```

### Run the example script

```powershell
py examples/test_url_validation.py
```

---

## 11. How to test

```powershell
# Run all tests with verbose output
py -m unittest discover -s tests -v

# Run a single test class
py -m unittest tests.test_url_validator -v

# Run a single test method
py -m unittest tests.test_url_validator.TestValidateUrl.test_01_valid_https_url -v
```

A passing run will show each test method followed by `ok` and end with:

```
Ran 42 tests in 0.XXXs

OK
```

---

## 12. Integration guidance for future modules

```python
from modules.url_validator import validate_url, check_dns_destination

def process_analyst_submission(raw_url: str) -> dict:
    # Step 1 — syntax validation (no network)
    result = validate_url(raw_url)
    if not result.is_valid:
        return {"status": "rejected", "errors": result.errors}

    # Step 2 — optional DNS check (explicit, intentional)
    dns = check_dns_destination(result.hostname)
    if dns["destination_status"] == "blocked":
        return {"status": "blocked", "errors": dns["errors"]}

    # Step 3 — pass structured result to Module 2 (feature extraction)
    return {
        "status": "ready_for_extraction",
        "validation": result.to_dict(),
        "dns": dns,
    }
```

Future modules must use `result.hostname` (ACE form) and `result.normalized_url`
rather than re-parsing the raw input string.

---

## 13. Design decisions

| Decision | Rationale |
|----------|-----------|
| `urllib.parse.urlparse` for parsing | Standard library; battle-tested; avoids third-party parser inconsistencies. |
| IDNA encoding of hostnames | Ensures `比较.com` and `xn--fiqs8sirgfmh.com` are treated as the same destination. |
| Trailing dot preserved | Legal FQDN notation; stripping it silently could change the semantic. |
| Port 0 rejected | Although technically parseable, port 0 is not a valid service port. |
| Credentials rejected | URL-embedded credentials are a credential-leakage risk and are not needed for this use case. |
| IPv4-mapped IPv6 unwrapped | Prevents private IPv4 bypass via `::ffff:` notation. |
| DNS check is opt-in | Syntax validation must work offline. DNS introduces latency, failure modes, and privacy implications; it should only be called when intentional. |
| `dataclass` for result | Provides field documentation, type hints, and easy conversion to `dict`. |
| `unittest` framework | Built-in; no dependencies; familiar to MCA students; sufficient for this module. |
