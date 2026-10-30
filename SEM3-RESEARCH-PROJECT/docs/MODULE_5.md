# Module 5 — Traditional URL, Domain and Webpage Feature Extraction

**Project:** Impact of Website Security Features on ML-Based Phishing Detection  
**Module:** 5  
**Status:** Complete (version 1)  
**Author:** MCA Semester 3 Research Team  
**Last updated:** 2025

---

## Table of contents

1. [Purpose](#1-purpose)
2. [Research relevance](#2-research-relevance)
3. [What this module does NOT do](#3-what-this-module-does-not-do)
4. [Public API](#4-public-api)
5. [Input and integration with URL validation](#5-input-and-integration-with-url-validation)
6. [Output — FeatureResult](#6-output--featureresult)
7. [Feature reference](#7-feature-reference)
   - [A. URL length and structure features](#a-url-length-and-structure-features)
   - [B. Character count features](#b-character-count-features)
   - [C. Hostname and domain features](#c-hostname-and-domain-features)
   - [D. Keyword features](#d-keyword-features)
   - [E. Boolean flag features](#e-boolean-flag-features)
8. [Descriptive fields](#8-descriptive-fields)
9. [Numeric feature schema](#9-numeric-feature-schema)
10. [Public suffix determination](#10-public-suffix-determination)
11. [Keyword list](#11-keyword-list)
12. [URL shortener list](#12-url-shortener-list)
13. [Missing and sentinel values](#13-missing-and-sentinel-values)
14. [How to run the application](#14-how-to-run-the-application)
15. [How to run the tests](#15-how-to-run-the-tests)
16. [Current limitations](#16-current-limitations)
17. [Features intentionally not implemented](#17-features-intentionally-not-implemented)

---

## 1. Purpose

Module 5 extracts a deterministic, machine-learning-compatible set of traditional
URL and hostname features from a validated URL.

These features will supply the inputs for **Model 1** in the research experiment:

| Model | Feature set |
|-------|-------------|
| Model 1 | Traditional URL, domain, and webpage features (this module) |
| Model 2 | Traditional features **plus** website-security configuration features (future module) |

The module produces a consistent numeric feature vector that can be exported to a
pandas DataFrame and saved as a CSV file for model training and evaluation.

---

## 2. Research relevance

The research question is:

> Does adding website-security configuration features (HTTP security headers, cookie
> attributes, TLS certificate properties) improve ML-based phishing detection over a
> model trained on traditional URL and domain features alone?

Module 5 provides the **baseline traditional features** that both models share.
It does not train a model, produce predictions, or evaluate accuracy.

The extracted features are **structural observations** only.
They do not classify any URL as phishing or legitimate.

---

## 3. What this module does NOT do

| Action | Reason |
|--------|--------|
| Make HTTP requests | No network activity is permitted during feature extraction. |
| Perform DNS lookups | DNS is handled separately by `check_dns_destination()` in Module 1. |
| Fetch or render webpage content | Webpage content features require safe HTTP fetching (out of scope for v1). |
| Classify URLs as phishing or legitimate | Classification is the responsibility of the ML modules. |
| Use WHOIS, domain-age, or reputation APIs | No external API calls are made. |
| Train or invoke a model | The module only produces features. |
| Fabricate results | All values are computed from the validated URL; none are invented. |

---

## 4. Public API

```python
from modules.url_validator import validate_url
from modules.feature_extractor import extract_features, numeric_feature_names

# Step 1 — validate the URL (Module 1)
vr = validate_url("https://sub.example.co.in/login?session=abc")

# Step 2 — extract features (Module 5), only if valid
if vr.is_valid:
    fr = extract_features(vr)
    print(fr.numeric)          # dict[str, int | float | bool]
    print(fr.descriptive)      # dict[str, Any] — human-readable metadata
    print(fr.keywords_matched) # list[str]
    print(fr.warnings)         # list[str] — non-fatal notes

# Feature schema (list of all numeric key names)
keys = numeric_feature_names()
```

### Errors raised by `extract_features()`

| Exception | Condition |
|-----------|-----------|
| `TypeError` | The argument is not a `ValidationResult` instance. |
| `ValueError` | The `ValidationResult` has `is_valid=False`. |

---

## 5. Input and integration with URL validation

`extract_features()` accepts a `ValidationResult` returned by `validate_url()`.
It does **not** accept a raw URL string.

**Calling contract:**
1. Call `validate_url(raw_url)` — this is Module 1.
2. Check `result.is_valid`. If `False`, display the validation errors and stop.
3. Call `extract_features(result)` — this is Module 5.
4. Use `feature_result.numeric` as a DataFrame row.

This design keeps validation and feature extraction cleanly separated and prevents
feature extraction from being called on an invalid or prohibited URL.

---

## 6. Output — FeatureResult

`extract_features()` returns a `FeatureResult` dataclass:

```python
@dataclass
class FeatureResult:
    url:              str               # normalised URL (no credentials)
    descriptive:      dict[str, Any]    # human-readable metadata
    numeric:          dict[str, Any]    # ML-compatible features
    keywords_matched: list[str]         # matched keyword subset
    warnings:         list[str]         # non-fatal notes
```

`to_dict()` returns a flat merged dictionary of all fields — convenient for
building a pandas DataFrame.

---

## 7. Feature reference

### A. URL length and structure features

| Key | Type | Description |
|-----|------|-------------|
| `url_length` | int | Total length of the normalised URL string in characters. |
| `url_hostname_length` | int | Length of the hostname component (ACE form). |
| `url_path_length` | int | Length of the path component. |
| `url_query_length` | int | Length of the raw query string (after `?`). 0 when no query. |
| `url_fragment_length` | int | Length of the fragment identifier (after `#`). 0 when none. |
| `url_num_query_params` | int | Number of distinct query parameter names. 0 when no query. |

### B. Character count features

Character counts are measured across the **full normalised URL** string.

| Key | Type | Character counted |
|-----|------|------------------|
| `char_count_dot` | int | `.` (period) |
| `char_count_hyphen` | int | `-` (hyphen / minus) |
| `char_count_underscore` | int | `_` (underscore) |
| `char_count_slash` | int | `/` (forward slash) |
| `char_count_percent` | int | `%` (percent, signals encoding) |
| `char_count_at` | int | `@` (at-sign) |

### C. Hostname and domain features

| Key | Type | Description |
|-----|------|-------------|
| `url_hostname_length` | int | Length of the hostname string. |
| `host_num_labels` | int | Total number of dot-separated labels in the hostname. |
| `host_subdomain_count` | int | Labels before the registered domain. -1 when unknown. |
| `domain_registered_length` | int | Length of the registered domain string. -1 when unknown. |

### D. Keyword features

| Key | Type | Description |
|-----|------|-------------|
| `kw_num_matched` | int | Number of keywords from the keyword list found in the URL. |
| `flag_has_keyword` | bool | True when at least one keyword matched. |

Matching is **case-insensitive** and covers the entire normalised URL string.

### E. Boolean flag features

| Key | Type | True when… |
|-----|------|-----------|
| `flag_uses_https` | bool | Scheme is `https`. |
| `flag_is_ip_address` | bool | Hostname is a direct IP address literal (IPv4 or IPv6). |
| `flag_has_non_default_port` | bool | Port is explicitly set and differs from the scheme default (80/443). |
| `flag_has_at_symbol` | bool | `@` appears anywhere in the URL. |
| `flag_has_query` | bool | A non-empty query string is present. |
| `flag_has_fragment` | bool | A fragment identifier is present. |
| `flag_has_punycode` | bool | At least one hostname label begins with `xn--` (ACE/punycode encoding). |
| `flag_is_url_shortener` | bool | Hostname exactly matches a known URL-shortening service. |

---

## 8. Descriptive fields

These fields are returned in `FeatureResult.descriptive`.  They are not intended
as direct ML inputs.

| Key | Type | Description |
|-----|------|-------------|
| `hostname` | str | The ACE-encoded hostname from the validator. |
| `scheme` | str | The URL scheme (`http` or `https`). |
| `registered_domain` | str | The registered domain, or `"unknown"` when indeterminate. |
| `public_suffix` | str | The public suffix used (e.g. `co.in`), or `"unknown"`. |
| `keywords_found` | list[str] | Same as `FeatureResult.keywords_matched`. |

---

## 9. Numeric feature schema

All 25 numeric keys are always present.  Use `numeric_feature_names()` to get
the canonical ordered list:

```python
from modules.feature_extractor import numeric_feature_names
print(numeric_feature_names())
# ['url_length', 'url_hostname_length', 'url_path_length', ...]
```

This is the **stable DataFrame column schema** for Model 1.

---

## 10. Public suffix determination

Accurate registered-domain extraction requires knowledge of public suffixes (e.g.
`co.uk`, `co.in`, `com.au`), which cannot be inferred from the hostname labels
alone.

**This module bundles a small, documented partial list** (`_TWO_LABEL_SUFFIXES`)
covering suffixes common in Indian ccTLDs, UK, Australia, and several others.

Rules:
1. If the last two labels of the hostname match a known two-label suffix, that
   suffix is used and the label before it is the registered-domain name.
2. Otherwise the single final label is treated as the TLD.
3. IP addresses, single-label hostnames, and bare suffixes return `"unknown"`.

**Limitation:** This heuristic is imprecise for uncommon or newer suffixes.
When the result is ambiguous, `registered_domain` is set to `"unknown"`,
`domain_registered_length` is set to `-1`, and a warning is added to
`FeatureResult.warnings`.

**Future improvement:** Replace with `tldextract` + the Mozilla Public Suffix
List for production-grade accuracy.

---

## 11. Keyword list

The following terms are matched case-insensitively against the full normalised URL:

```
login, verify, secure, account, update, signin, sign-in, wallet, payment,
confirm, banking, netbanking, portal, authenticate, credential, password,
passwd, reactivate, recover, unlock
```

**Important:** These keywords appear in phishing URLs but also in many legitimate
websites.  A keyword match is a structural observation, not a phishing verdict.
The list is defined in `PHISHING_KEYWORDS` in `modules/feature_extractor.py`.

---

## 12. URL shortener list

The following hostnames are matched exactly (case-insensitive, full hostname):

```
bit.ly, tinyurl.com, t.co, goo.gl, ow.ly, buff.ly, is.gd, rb.gy,
cutt.ly, shorte.st, tiny.cc, bl.ink, rebrand.ly
```

Defined in `URL_SHORTENERS` in `modules/feature_extractor.py`.

---

## 13. Missing and sentinel values

| Situation | Integer features | Boolean features |
|-----------|-----------------|-----------------|
| Value cannot be computed (e.g. IP-address hostname has no registered domain) | `-1` | `False` |
| Value is zero (e.g. no query parameters) | `0` | `False` |

`None` and `NaN` are **never used** in the numeric dict.  This ensures safe
DataFrame construction without unexpected NaN propagation.

---

## 14. How to run the application

```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Start the Streamlit application
py -m streamlit run app.py
```

Expected local address: `http://localhost:8501`

---

## 15. How to run the tests

```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Run Module 5 tests only
py -m pytest tests/test_feature_extractor.py -v

# Run all tests (Module 1 + Module 5)
py -m pytest tests/test_url_validator.py tests/test_feature_extractor.py -v

# Run all tests using unittest discover
py -m unittest discover -s tests -v
```

Test counts:
- Module 1 (URL validation): 42 tests
- Module 5 (feature extraction): 33 tests
- **Total: 75 tests**

---

## 16. Current limitations

| Limitation | Notes |
|------------|-------|
| Public suffix list is partial | Only common two-label suffixes are included. Unusual or new suffixes may be misclassified. Use `tldextract` for accuracy. |
| No webpage content features | Features derived from rendered page content (title text, form count, external links) are not extracted in v1. |
| No WHOIS or domain-age data | Registration date, registrar, and expiry are not extracted (require external API). |
| No DNS features | DNS record types (A, MX, NS) are not extracted (require live DNS lookup). |
| No redirect features | Redirect count and final destination are not assessed (require HTTP fetch). |
| URL shortener list is small | Only 13 well-known shorteners are detected. Uncommon shorteners are missed. |
| Keyword list is configurable but not exhaustive | The list covers common phishing terms; domain-specific terms may be missing. |

---

## 17. Features intentionally not implemented

The following features are **deliberately excluded** from v1:

| Feature | Reason for exclusion |
|---------|---------------------|
| Webpage title and body content | Requires safe HTTP fetching with SSRF protection (planned for a future module). |
| Number of external links | Requires page fetching. |
| Number of form elements | Requires page fetching. |
| Domain registration date / age | Requires WHOIS API — external dependency. |
| DNS record information | Requires live DNS resolution — kept separate (Module 1). |
| TLS certificate details | Planned for Module 6 (website security configuration features). |
| HTTP security headers | Planned for Module 6. |
| Cookie attributes | Planned for Module 6. |
| Brand/logo similarity detection | Requires image analysis — out of scope. |
| ML prediction | This module extracts features only; prediction is a separate module. |
