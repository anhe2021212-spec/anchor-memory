# Security policy

## Reporting a vulnerability

Use the repository Security tab to submit a private vulnerability report. If private reporting is unavailable, open a minimal issue asking the maintainers for a private contact channel. Do not include exploit details in a public issue.

## Sensitive data

Never commit credentials, memory databases, vector indexes, graph databases, prompts, provider responses, chat histories, review queues, logs, or private deployment configuration.

Credentials must come from the process environment or an explicitly configured external secret file. Network services bind to loopback by default and should not be exposed without authentication and a reverse proxy.

Before submitting a change, run:

```bash
python scripts/check_public_tree.py .
pytest
```
