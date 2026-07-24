# Security policy

Use the repository Security tab to submit a private vulnerability report. If
private vulnerability reporting is unavailable, open a minimal issue asking
the maintainers for a private channel; do not include exploit details. Never
put live credentials, memory text, prompts, provider responses, or private
deployment topology in an issue.

Credentials may come only from process environment or an external secret
manager. Example values and `.invalid` endpoints fail closed. Logs and health
reports contain status and counts, never memory bodies or full filesystem paths.

Before a public release:

1. Run `scripts/verify_release_tree.py` on source and built trees.
2. Scan dependency licenses and generate an SBOM.
3. Scan the complete new Git history.
4. Rotate and revoke every credential that ever appeared in a private source
   tree or its history.
