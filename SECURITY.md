# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository (e.g. a way
credentials could leak, an unsafe deserialization path, or a dependency
with a known CVE), please open a private security advisory via the
repository's **Security** tab ("Report a vulnerability") rather than a
public issue, so it can be addressed before public disclosure.

## Credential handling in this repository

Q-TRACE's real-hardware quantum experiments (`quantum/ibm_runtime_job.py`)
require an IBM Quantum API token and CRN instance identifier. To avoid
accidental credential leakage:

- **Never hardcode your token or CRN into any file in this repository.**
  All example code uses `"my_api_key"` / `"my_crn"` as literal
  placeholders (see `README.md`, "Running on real IBM Quantum hardware").
  Paste your real credentials only into your local Python session /
  notebook cell at runtime -- never into a file you intend to commit.
- `.gitignore` excludes `qiskit_ibm_runtime_account.json`, `.qiskit/`,
  and `*.token` by default.
- If you accidentally commit a real token, treat it as compromised:
  revoke/regenerate it immediately from
  [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com/) (Account →
  API keys) rather than relying on removing it from git history, since
  the token may already be cached by anyone who cloned the repo before
  the fix, and git history rewriting does not retroactively invalidate
  a leaked credential -- only revocation does.

## Supported versions

This is a research-code repository accompanying a specific paper
submission; there is no ongoing security-patch policy beyond the active
development branch (`main`).
