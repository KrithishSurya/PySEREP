# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 3.x.x   | ✅ Current release |
| < 3.0   | ❌ Unsupported     |

## Reporting a Vulnerability

If you discover a security vulnerability in `pyserep`, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please send an email to the maintainer with:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact assessment
4. Any suggested fixes (if available)

The maintainer will acknowledge receipt within 72 hours and provide a timeline
for a fix.

## Scope

`pyserep` is a scientific computing library for structural dynamics.
Its primary dependencies (NumPy, SciPy, Matplotlib) are widely audited.
Security issues most likely to be relevant include:

- **Malicious matrix files** — the matrix loader parses external files
  (`.mtx`, `.npz`, `.h5`). While NumPy/SciPy loaders are generally safe,
  crafted files could potentially cause unexpected behaviour.
- **Pickle-based formats** — `pyserep` does not use pickle for any output
  format. All exports use NumPy's NPZ/NPY formats and JSON, which do not
  execute arbitrary code on load.
- **Dependency vulnerabilities** — please also report known CVEs in
  NumPy/SciPy/Matplotlib that affect `pyserep`'s use of those libraries.

## Disclosure Policy

After a fix is released, vulnerabilities will be disclosed in the
`CHANGELOG.md` under a `### Security` heading with the CVE number if assigned.
