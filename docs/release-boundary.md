# Release boundary

`source-allowlist.txt` is the complete positive list of live implementation
files admitted to this extraction. Files outside that list are not copied and
cannot enter the release by a broad directory operation.

The public transformation classes are deliberately narrow:

1. replace deployment paths, ports, service names, identities, and secret-file
   assumptions with platform directories or environment configuration;
2. remove the private production push adapter and no other functional module;
3. make Chroma, Kuzu, and model providers optional while retaining the live
   Chroma recall branch and the SQLite FTS degradation branch;
4. make empty installs valid by creating an empty belief state and by avoiding
   listener startup during module import;
5. correct fail-open or false-green release behavior, including absent gateway
   keys and inconsistent graph health;
6. add release-only verification, projection rebuilding, tests, packaging, and
   documentation.

`generate_lineage.py` compares the private allowlisted mirror to the public
tree. The generated TSV identifies files only by basename. The private mirror,
source paths, raw findings, and adapted live tests are audit evidence and must
never be copied into the public tree.
