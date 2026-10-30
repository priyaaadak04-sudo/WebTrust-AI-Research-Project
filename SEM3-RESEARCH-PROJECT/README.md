# WebTrust AI — Financial Website Risk Intelligence

**Repository:** https://github.com/Garima2005-joshi/SEM3-RESEARCH-PROJECT  
**Team:** MCA Semester 3 — 3-person research team  
**Platform:** Python 3 · Streamlit · Windows (PowerShell)  
**Current development status:** Modules 1, 2, 3, 4, and 5 complete

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Research Objective](#3-research-objective)
4. [Research Question](#4-research-question)
5. [Proposed Research Methodology](#5-proposed-research-methodology)
6. [Application Users and Intended Use](#6-application-users-and-intended-use)
7. [Current Application Features](#7-current-application-features)
8. [Research Models](#8-research-models)
9. [Planned Project Modules and Their Status](#9-planned-project-modules-and-their-status)
10. [Technology Stack](#10-technology-stack)
11. [Repository Structure](#11-repository-structure)
12. [Installation and Setup](#12-installation-and-setup)
13. [How to Run the Application](#13-how-to-run-the-application)
14. [How to Run Tests](#14-how-to-run-tests)
15. [Team Collaboration and Git Workflow](#15-team-collaboration-and-git-workflow)
16. [Security, Ethical Considerations, and Limitations](#16-security-ethical-considerations-and-limitations)
17. [Current Development Status](#17-current-development-status)
18. [Future Work](#18-future-work)
19. [Disclaimer](#19-disclaimer)

---

## 1. Project Overview

WebTrust AI is a Streamlit-based web application built as part of an MCA Semester 3
research project. It provides banking and fintech security analysts with a structured
interface for investigating suspicious financial website URLs — particularly URLs that
may be impersonating Indian banking and fintech brands.

The application currently supports URL structural validation and traditional URL feature
extraction. It is designed to grow into a full ML-assisted phishing-detection platform
as additional modules are implemented and integrated.

The research component of the project investigates whether adding website
security-configuration features improves the accuracy of machine-learning-based
phishing detection compared to a model that relies solely on traditional URL and
domain features.

---

## 2. Problem Statement

Phishing attacks targeting banking and fintech customers often involve websites that
closely mimic legitimate financial institutions. These sites obtain valid HTTPS
certificates, use plausible domain names, and can pass basic URL checks that do not
examine the security configuration of the destination.

Security analysts investigating suspected phishing links need structured, repeatable
tools to examine the technical properties of a suspicious URL before committing to
further investigation steps. They also need clear distinctions between what has been
checked (URL structure) and what has not (domain legitimacy, ML classification,
destination safety).

---

## 3. Research Objective

To determine whether incorporating website security-configuration features — HTTP
security headers, cookie security attributes, and TLS certificate properties — into
a machine-learning model improves phishing detection accuracy over a baseline model
trained on traditional URL and domain features alone.

---

## 4. Research Question

> Does adding website security-configuration features (HTTP security headers,
> cookie security attributes, and TLS certificate properties) improve machine-learning-
> based phishing detection accuracy for financial websites compared to a model trained
> on traditional URL and domain features only?

The outcome of this comparison is not assumed in advance. Both models will be trained
and evaluated objectively on the same dataset.

---

## 5. Proposed Research Methodology

1. **Dataset collection** — Gather labelled URLs representing phishing and legitimate
   financial websites, with a focus on Indian banking and fintech brands.
2. **Feature extraction** — Extract two feature sets from each URL in the dataset:
   - Traditional URL, domain, and webpage features (Module 5).
   - Website security-configuration features: HTTP headers, cookie attributes, TLS
     certificate properties (Module 6, planned).
3. **Model training** — Train two separate classifiers (Model 1 and Model 2) on
   their respective feature sets.
4. **Evaluation** — Compare classification performance using standard metrics
   (accuracy, precision, recall, F1 score, AUC-ROC). Evaluate whether Model 2 offers
   a statistically meaningful improvement over Model 1.
5. **Integration** — Integrate the working model into the analyst-facing application
   so analysts can submit a URL and receive a structured assessment.
6. **Reporting** — Document findings, limitations, and recommendations.

---

## 6. Application Users and Intended Use

**Primary users:** Junior security analysts and fraud/security teams at fintech
companies and banks.

**Intended use:** Initial investigation and triage of suspicious financial website
URLs. The platform surfaces structured technical observations to help analysts
decide whether further investigation is warranted.

### Important distinctions

| Concept | What the platform does |
|---|---|
| URL structural validity | Checks that the URL is syntactically well-formed. **Implemented.** |
| Destination status | Optionally checks whether DNS resolves the hostname to a non-prohibited address. Not a safety certificate. |
| Domain legitimacy | **Not assessed.** A well-formed hostname is not evidence of legitimate ownership. |
| ML phishing classification | **Not yet available.** Will be shown only when a validated model is integrated. |

A URL passing structural validation is not a verdict on whether the website is
legitimate or malicious. The platform supports analyst investigation — it does not
replace professional judgement.

---

## 7. Current Application Features

### 7.1 Homepage (`pages/home.py`)

- Branded hero section with product description.
- Platform Capabilities overview (six feature cards).
- "How It Works" five-step workflow.
- "Explore the Platform" button scrolls to the Platform Capabilities section.
- "Analyze a Suspicious URL" button navigates to the analyst interface or sign-in page.
- About section.
- Professional footer with navigation, disclaimer, and copyright.
- Navigation bar adapts to authentication state (shows Sign In or Analyze URL / Sign Out).

### 7.2 Login and Role-Based Access Control (`pages/login.py`, `modules/authentication.py`, `modules/access_control.py`)

- Email and password login form.
- Credentials verified using PBKDF2-HMAC-SHA256 with per-account salts.
- Constant-time comparison to resist timing attacks.
- Two roles: `analyst` and `admin`.
- Session state managed via Streamlit session state (per-session, not persistent).
- Unauthenticated users redirected to sign-in before any protected page renders.
- Accounts loaded from environment variables at startup; default demo accounts
  are available for local development only.

> **Security note:** Session-state authentication is suitable for local demonstration.
> It is not a production-grade authentication system.

### 7.3 URL Structural Validation (`modules/url_validator.py`)

- Accepts a URL string and returns a structured `ValidationResult`.
- Checks: scheme allow-list, hostname presence, port range (1–65535), embedded
  credentials, IP address classification, IDNA/punycode hostname encoding.
- Blocks SSRF-risk destinations: loopback, RFC 1918 private, link-local, multicast,
  reserved, and unspecified address ranges.
- Bare-hostname inputs (without `https://`) are inferred if they contain a dot.
- Optional DNS destination check via `check_dns_destination()` — never called
  automatically.
- No network requests during structural validation.

### 7.4 Traditional URL Feature Extraction (`modules/feature_extractor.py`)

- Accepts a valid `ValidationResult` and returns a `FeatureResult` containing 25
  numeric features suitable for a machine-learning DataFrame row.
- Feature groups: URL length and structure, character counts, hostname and domain
  features, keyword indicators, boolean flags.
- Registered domain extraction uses a bundled partial public suffix list covering
  common Indian, UK, and Australian ccTLDs. Returns `"unknown"` when ambiguous.
- Keyword matching is case-insensitive and covers the full normalised URL. A match
  is a structural observation, not a phishing verdict.
- URL-shortener detection for 13 well-known shortener hostnames.
- No network requests during feature extraction.
- See [`docs/MODULE_5.md`](docs/MODULE_5.md) for the full feature reference.

### 7.5 Analyst Detection Interface (`pages/analyst_detection.py`)

The primary analyst-facing page. Requires authentication (analyst or admin role).

- URL input field with an Analyze button.
- Runs URL validation and, if valid, traditional feature extraction.
- Results persist across tab switches via Streamlit session state.
- Summary strip showing URL Validation status, Destination Status (from actual
  validator result), Domain Legitimacy (Not Assessed), and ML Classification
  (Not Available).
- Four analysis tabs:

  | Tab | Content |
  |---|---|
  | **URL Components** | Scheme, hostname, port, path, encryption, host type, destination status, domain legitimacy notice |
  | **Traditional Features** | URL length and structure, character counts, hostname and domain features, keyword indicators |
  | **Website Security** | Placeholder — not yet integrated |
  | **ML Detection** | Placeholder — not yet integrated |

- Analyst reminder that findings are structural observations and do not determine
  phishing or legitimacy.
- The administrator/researcher dashboard is not accessible from this page.

### 7.6 URL Validation Page (`pages/url_analysis.py`)

A standalone, access-controlled view of URL validation results. Shows parsed URL
components, validation status, errors, warnings, and destination status. No feature
extraction output is shown on this page.

---

## 8. Research Models

Two machine-learning models will be trained and compared. Neither model has been
trained yet. No experimental results are available at this stage.

### Model 1 — Traditional Features

Trained on URL structure and character patterns, domain-related features, and
webpage features where implemented and supported by the dataset.

The feature schema is defined in `modules/feature_extractor.py` and documented
in [`docs/MODULE_5.md`](docs/MODULE_5.md). The 25-feature numeric vector produced
by `extract_features()` is the planned DataFrame column schema for Model 1.

### Model 2 — Security-Aware Features

Trained on all applicable Model 1 features plus:

- HTTP security headers (HSTS, Content-Security-Policy, X-Frame-Options, etc.)
- Cookie security attributes (Secure, HttpOnly, SameSite)
- TLS certificate properties (issuer, validity period, subject alternative names)

These additional features will be extracted by Module 6 (planned).

### Evaluation

Both models will be evaluated on the same labelled dataset using accuracy,
precision, recall, F1 score, and AUC-ROC. The research objective is to determine
objectively whether Model 2 outperforms Model 1. No result is assumed in advance.

---

## 9. Planned Project Modules and Their Status

| # | Module Name | Status | Notes |
|---|---|---|---|
| 1 | Home Page and Navigation | ✅ Complete | Branded homepage, hero section, workflow, features, footer, nav |
| 2 | Login and Role-Based Access Control | ✅ Complete | PBKDF2 auth, analyst/admin roles, session state, access guards |
| 3 | URL Input and Validation | ✅ Complete | Full structural validation, SSRF protection, optional DNS check; 42 tests |
| 4 | Analyst Detection Interface | ✅ Complete | Four-tab interface, session-state result caching, status labels |
| 5 | Traditional URL, Domain, and Webpage Feature Extraction | ✅ Complete | 25-feature numeric vector, no network requests; 33 tests |
| 6 | Website Security Feature Extraction: HTTP Headers, Cookies, and TLS | 🔲 Planned | Tab 3 placeholder exists; module not yet implemented |
| 7 | Machine Learning Model Development and Integration | 🔲 Planned | Tab 4 placeholder exists; no model trained or integrated |
| 8 | Risk Assessment and Prediction Explanation | 🔲 Planned | Depends on Module 7 |
| 9 | Administrator/Researcher Dashboard for Model Comparison and Evaluation | 🔲 Planned | Admin role defined; dashboard not yet built |
| 10 | Report Export, System Integration, and Testing | 🔲 Planned | Cross-cutting; includes integration tests for future modules |

> **Note on Module 10:** Integration and system-level testing are ongoing
> responsibilities. Unit tests for Modules 3 and 5 are complete (96 tests total).
> Integration tests covering the full pipeline from URL input to ML output will be
> added as later modules are completed.

---

## 10. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Web framework | Streamlit ≥ 1.32.0 |
| URL parsing | `urllib.parse` (standard library) |
| IP address classification | `ipaddress` (standard library) |
| Hostname encoding | `unicodedata` / IDNA codec (standard library) |
| Password hashing | `hashlib.pbkdf2_hmac` / `hmac` (standard library) |
| Test framework | `pytest` (preferred) / `unittest` (built-in) |
| Theme | Custom light theme via `.streamlit/config.toml` |
| ML libraries (planned) | scikit-learn, pandas, numpy |
| HTTP fetching (planned) | `requests` or `httpx` with SSRF safeguards |

---

## 11. Repository Structure

```
SEM3-RESEARCH-PROJECT/
│
├── app.py                          # Application entry point (Streamlit navigation)
│
├── modules/
│   ├── __init__.py                 # Makes modules/ a Python package
│   ├── authentication.py           # Module 2 — credential verification, session management
│   ├── access_control.py           # Module 2 — role enforcement guards
│   ├── url_validator.py            # Module 3 — URL structural validation
│   ├── feature_extractor.py        # Module 5 — traditional URL feature extraction
│   └── styles.py                   # Shared CSS injected into every page
│
├── pages/
│   ├── home.py                     # Module 1 — homepage and navigation
│   ├── login.py                    # Module 2 — sign-in form
│   ├── analyst_detection.py        # Module 4 — analyst URL analysis interface
│   └── url_analysis.py             # Module 3 — standalone URL validation view
│
├── tests/
│   ├── test_url_validator.py       # 42 tests for Module 3 (URL validation)
│   ├── test_feature_extractor.py   # 33 tests for Module 5 (feature extraction)
│   └── test_authentication.py      # 21 tests for Module 2 (authentication)
│
├── examples/
│   └── test_url_validation.py      # Runnable demonstration script for Module 3
│
├── docs/
│   ├── MODULE_1.md                 # Design and security documentation for Module 3
│   └── MODULE_5.md                 # Feature reference documentation for Module 5
│
├── .streamlit/
│   └── config.toml                 # Light theme configuration
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Excludes .venv, __pycache__, secrets, datasets
```

> `.venv/`, `__pycache__/`, `.pytest_cache/`, and other generated directories
> are excluded from version control by `.gitignore`.

---

## 12. Installation and Setup

### Prerequisites

- Python 3.12 or later installed and available as `py` on Windows.
- Git installed.

### Step 1 — Confirm Python is available

```powershell
py --version
# Expected: Python 3.12.x or later
```

### Step 2 — Clone the repository

```powershell
git clone https://github.com/Garima2005-joshi/SEM3-RESEARCH-PROJECT.git
cd SEM3-RESEARCH-PROJECT
```

### Step 3 — Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4 — Install dependencies

```powershell
py -m pip install -r requirements.txt
```

The current dependencies are:

```
streamlit>=1.32.0
```

All other current functionality uses the Python standard library. Additional
dependencies (scikit-learn, pandas, numpy, requests) will be added to
`requirements.txt` as later modules are implemented.

---

## 13. How to Run the Application

Ensure the virtual environment is activated before running.

```powershell
py -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

**Demo accounts (local development only):**

| Role | Email | Password |
|---|---|---|
| Security Analyst | `analyst@webtrust.local` | `AnalystDemo2025!` |
| Researcher / Admin | `admin@webtrust.local` | `AdminDemo2025!` |

These credentials are the defaults when the corresponding environment variables are
not set. Do not use these accounts or passwords in any deployed or shared environment.

To override credentials before starting the app:

```powershell
$env:WEBTRUST_ANALYST_EMAIL    = "your-email@example.com"
$env:WEBTRUST_ANALYST_PASSWORD = "YourSecurePassword"
$env:WEBTRUST_ADMIN_EMAIL      = "admin@example.com"
$env:WEBTRUST_ADMIN_PASSWORD   = "YourAdminPassword"
py -m streamlit run app.py
```

---

## 14. How to Run Tests

Ensure the virtual environment is activated.

### Run the full test suite (recommended)

```powershell
py -m pytest tests/ -v
```

Expected result: **96 tests passed**.

| Test file | Module | Tests |
|---|---|---|
| `tests/test_url_validator.py` | Module 3 — URL validation | 42 |
| `tests/test_feature_extractor.py` | Module 5 — Feature extraction | 33 |
| `tests/test_authentication.py` | Module 2 — Authentication | 21 |
| **Total** | | **96** |

### Run tests for a single module

```powershell
py -m pytest tests/test_url_validator.py -v
py -m pytest tests/test_feature_extractor.py -v
py -m pytest tests/test_authentication.py -v
```

### Run tests using unittest (alternative)

```powershell
py -m unittest discover -s tests -v
```

### Run the demonstration script (Module 3 examples)

```powershell
py examples/test_url_validation.py
```

> **Note:** The demonstration script includes an optional DNS section that makes
> real DNS queries. All other tests run without network access.

---

## 15. Team Collaboration and Git Workflow

### Accepting a repository invitation

1. Go to **Settings → Collaborators and teams** in the GitHub repository.
2. Click **Add people** and enter each teammate's GitHub username.
3. They will receive an email invitation and must accept it before pushing.

### Teammate workflow

```powershell
# Clone the repository (first time only)
git clone https://github.com/Garima2005-joshi/SEM3-RESEARCH-PROJECT.git
cd SEM3-RESEARCH-PROJECT

# Create a virtual environment and install dependencies
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

# Create a personal feature branch
git checkout -b feature/your-feature-name

# Make changes, then stage only the files you intend to commit
git add <specific-files>
git commit -m "feat: short description of what was changed"

# Push your branch to GitHub
git push -u origin feature/your-feature-name

# Open a pull request on GitHub
# → Compare your branch against main
# → Ask a teammate to review before merging
```

### Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code only. Never commit directly to `main`. |
| `feature/<your-feature>` | Each team member's work in progress |

### Pull request rules

- All changes to `main` must go through a pull request.
- At least one other team member must review before merging.
- Pull requests should be small and focused — one feature or fix per PR.
- The automated test suite must pass before merging.

### Keeping your branch up to date

```powershell
git fetch origin
git rebase origin/main
```

Communicate with teammates about which files you are editing to avoid conflicts.

### What must NEVER be committed

| Type | Risk |
|---|---|
| `.env` files or any file containing passwords or API keys | Credential exposure — permanent in Git history |
| Virtual environment directories (`.venv/`) | Large, machine-specific, not portable |
| Python `__pycache__/` and `*.pyc` files | Generated, not portable |
| Datasets containing personal data (`.csv`, `.pkl`, etc.) | May be large, private, or subject to data-protection obligations |
| Trained model files (`.pkl`, `.h5`, `.joblib`, etc.) | May be large; should be shared separately |
| Personal configuration files or IDE settings | Machine-specific |

The `.gitignore` file already excludes all of the above categories.

---

## 16. Security, Ethical Considerations, and Limitations

### Handling suspicious URLs

- **Never visit a suspicious URL from your own machine** during development or testing.
- The validator and feature extractor process URLs as strings only — they make no
  network connections to the target URL.
- DNS checks are opt-in and must be called explicitly via `check_dns_destination()`.
- Query strings may contain tokens, session identifiers, or other sensitive data —
  do not log or display them unnecessarily.

### What URL validation does and does not establish

| This is checked | This is NOT established |
|---|---|
| URL syntax is well-formed | The destination is legitimate or safe |
| Scheme is http or https | The website is not phishing |
| Hostname does not resolve to a private/loopback address (if DNS check is run) | The domain is owned by the claimed brand |
| No embedded credentials | The TLS certificate is valid or meaningful |

An HTTPS scheme and a valid certificate do not indicate that a website is
trustworthy. Phishing sites routinely obtain valid certificates.

### Authentication limitations

The session-state authentication in the current application is suitable for local
demonstration only. It is not hardened for production deployment. Passwords are
never logged or displayed; PBKDF2-HMAC-SHA256 with per-account salts is used for
credential verification.

### Dataset and research ethics

- Do not include personally identifiable information in any dataset committed to
  this repository.
- Any dataset containing real phishing URLs should be obtained through a responsible,
  licensed source (e.g. PhishTank, OpenPhish, or an academic dataset).
- Do not access or store data from suspicious URLs encountered during research without
  appropriate institutional approval.

### DNS rebinding limitation

The optional DNS check (`check_dns_destination()`) is a one-time, best-effort
policy check. It is not a complete SSRF defence. An attacker controlling a DNS server
can return different addresses for successive queries (DNS rebinding). A
`"verified_for_policy"` result does not guarantee the destination is safe to access.

---

## 17. Current Development Status

| Area | Status |
|---|---|
| Homepage and navigation | ✅ Complete |
| Login and role-based access control | ✅ Complete |
| URL structural validation | ✅ Complete — 42 tests passing |
| Traditional URL feature extraction | ✅ Complete — 33 tests passing |
| Authentication unit tests | ✅ Complete — 21 tests passing |
| Analyst detection interface | ✅ Complete — four tabs, session-state caching |
| Website security feature extraction | 🔲 Not started — Tab 3 is a placeholder |
| ML model training and integration | 🔲 Not started — Tab 4 is a placeholder |
| Risk assessment and explanation | 🔲 Not started |
| Administrator/researcher dashboard | 🔲 Not started — admin role is defined |
| Report export | 🔲 Not started |

**Total automated tests: 96 (all passing)**

---

## 18. Future Work

The following items are planned for upcoming development phases:

1. **Module 6 — Website security feature extraction:** Implement HTTP request logic
   with SSRF protection to extract HTTP security headers, cookie security attributes,
   and TLS certificate properties from the target URL.

2. **Module 7 — ML model development:** Collect and label a dataset of financial
   website URLs. Train Model 1 (traditional features) and Model 2 (traditional +
   security features). Evaluate both models and record results.

3. **Module 7 — ML integration:** Integrate the selected model into the analyst
   interface so that Tab 4 displays a real classification result with feature
   contribution information.

4. **Module 8 — Risk assessment:** Present a structured risk summary to the analyst
   based on validation, feature extraction, and ML classification results.

5. **Module 9 — Researcher dashboard:** Build the administrator-facing view for
   comparing Model 1 and Model 2 performance metrics. This view must never be
   accessible to analyst-role users.

6. **Module 10 — Integration testing and export:** Add system-level integration tests,
   implement report export (PDF or CSV), and harden the application for evaluation.

7. **Public suffix list accuracy:** Replace the bundled partial suffix list in
   `modules/feature_extractor.py` with `tldextract` and the Mozilla Public Suffix
   List for production-grade registered-domain extraction.

---

## 19. Disclaimer

WebTrust AI is a student research project developed for academic purposes.

Analysis results are structured technical observations from currently integrated
modules. They support investigation and **do not independently establish that a
website is legitimate or malicious.**

URL structural validation, DNS policy checks, and traditional feature extraction
are distinct from phishing classification. A URL that passes structural validation
has not been determined to be safe, legitimate, or free of phishing intent.

No machine-learning model has been trained or integrated at this stage. No risk
scores, phishing labels, or confidence values are produced by the current application.

Do not use this application as the sole basis for any security, legal, or compliance
decision.
