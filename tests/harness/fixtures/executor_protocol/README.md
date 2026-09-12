# Executor protocol fixtures

These are synthetic, deterministic payloads for parser tests.  They are
shaped from the vendor documentation and are not captured model executions.
The parser tests do not contact a model, executor service, or provider.

The supported adapter versions are Codex CLI `0.154.x` (the half-open range
`[0.154.0, 0.155.0)`), Claude Code `2.1.259`, and Cursor Agent build
`2026.09.10-fd3934a` on Linux x86_64.  Cursor status fixtures are synthetic
shapes statically derived from the published `status` command implementation
(`dist-package/8519.index.js`, module `./src/commands/status.ts`), not account
output.  The published Cursor package used for that static audit was obtained
from the official installer URL and had captured SHA-256
`27997c8391ad853a5a732b1845db8ef82a8ba6afb0f7829cc739464f8966e96e`; this is
artifact provenance rather than a vendor signature or checksum.  The captured
package URL was
`https://downloads.cursor.com/lab/2026.09.10-fd3934a/linux/x64/agent-cli-package.tar.gz`.

The documented formats used to shape the fixtures are:

- [Codex non-interactive JSONL](https://developers.openai.com/codex/noninteractive)
- [Claude Code headless JSON output](https://code.claude.com/docs/en/headless)
- [Cursor Agent JSON output](https://cursor.com/docs/cli/reference/output-format)
- [Cursor Agent installation and published build](https://cursor.com/install)
