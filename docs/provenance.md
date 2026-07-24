# Source provenance

This release was built from an allowlisted mirror of the current live implementation, not from a clean-room rewrite. The private audit bundle retains original bytes and paths; it is intentionally excluded from the release tree.

Public `lineage.tsv` exposes only filenames and cryptographic/structural measurements. A release module must retain its live functions/classes unless an explicit public-boundary exclusion is documented. At present, only the production-only push adapter is excluded.

A green functional suite is not sufficient by itself. Release approval also requires lineage retention, security scanning, no-provider degradation, real Chroma/Kuzu tests, and agent-ownership wording checks.
