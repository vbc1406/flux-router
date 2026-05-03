# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Flux, please report it responsibly:

- **Email:** security@flux.dev
- **Do NOT open a public GitHub issue** for security vulnerabilities
- **Subject line:** [SECURITY] Brief description

## What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Any suggested fixes (optional)

## Response Timeline

- Initial response: within 48 hours
- Status update: within 7 days
- Patch timeline: depends on severity (critical: 7 days, high: 30 days, medium: 90 days)
- Public disclosure: coordinated with reporter, typically 90 days after fix

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | Yes       |
| < 1.0   | No        |

## Out of Scope

- Issues in third-party dependencies (report to upstream)
- Self-XSS requiring user to paste code into console
- Theoretical attacks without practical impact
- Issues requiring physical access to a developer's machine

## Security Architecture

For details on Flux's data handling guarantees, see SECURITY_ARCHITECTURE.md.
