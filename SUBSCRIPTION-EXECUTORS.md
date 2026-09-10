# Subscription-backed local executors

Local executors run GDPval task workspaces on the user's machine while using an authenticated coding-agent CLI for model access. They are intentionally separate from the NeMo Gym / Stirrup provider path and from judging.

These runs are **not** GDPval-AA v2 reproduction runs. `./gdpval aa-v2` remains the Stirrup + AA-v2 judge/reference path.

All subscription-backed runs require an explicit `--limit` so an accidental invocation cannot consume the full 220-task benchmark. Local executors are serial for now (`--parallel 1`).

## Codex CLI

Official references:

- [Codex authentication](https://developers.openai.com/codex/auth/)
- [Codex CLI reference](https://developers.openai.com/codex/cli/reference/)
- [Codex configuration reference](https://developers.openai.com/codex/config-reference/)

Codex supports ChatGPT sign-in for subscription access and API-key sign-in for usage-based access. This harness accepts only the former for `--executor codex`.

Authenticate interactively before running the harness:

```bash
codex login
codex login status
```

`./gdpval check --executor codex` verifies that the installed CLI reports ChatGPT authentication. API-key or access-token authentication fails closed. The executor also removes OpenAI API routing credentials from the Codex child environment; it never silently falls back to API billing.

Run one task:

```bash
./gdpval run --executor codex --limit 1 --out runs/codex-smoke
```

The harness invokes `codex exec` non-interactively, sets the task workspace with `--cd`, uses the `workspace-write` sandbox, sets `approval_policy=never`, disables outbound network for model-generated commands by default, uses an ephemeral session, and captures structured stdout plus the final message. The GDPval task prompt is supplied on stdin rather than argv.

## Claude Code

Official references:

- [Claude Code authentication](https://code.claude.com/docs/en/authentication)
- [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)
- [Claude Code settings](https://code.claude.com/docs/en/settings)

Claude Code supports stored Claude.ai subscription OAuth, Console/API credentials, cloud providers, gateway credentials, and setup-token OAuth. This harness deliberately accepts only a stored first-party Claude.ai Pro, Max, Team, or Enterprise login for `--executor claude-code`.

Authenticate before running:

```bash
claude auth login
claude auth status
```

Preflight requires `loggedIn=true`, `authMethod=claude.ai`, `apiProvider=firstParty`, and a supported subscription type. The executor removes API, bearer-token, setup-token, custom-base-URL, Bedrock, Vertex, and Foundry routing variables before both the auth check and task execution. This matters because Claude Code gives environment API credentials higher precedence than stored subscription OAuth in non-interactive `-p` mode.

Run one task:

```bash
./gdpval run --executor claude-code --limit 1 --out runs/claude-smoke
```

The harness uses `claude -p`, JSON output, disabled session persistence, `--safe-mode`, a restricted built-in tool set, and a Bash sandbox with `failIfUnavailable=true` and `allowUnsandboxedCommands=false`. Model-generated network access uses a strict empty allowlist by default. `GDPVAL_EXECUTOR_MAX_TURNS` controls Claude Code's `--max-turns` and defaults to 250.

## Cursor Agent CLI

Official references:

- [Cursor CLI overview](https://cursor.com/docs/cli/overview)
- [Cursor CLI authentication](https://cursor.com/docs/cli/reference/authentication)
- [Cursor CLI permissions](https://cursor.com/docs/cli/reference/permissions)
- [Cursor sandbox configuration](https://cursor.com/docs/reference/sandbox)

Cursor documents browser/account login (`agent login`) separately from API-key authentication (`CURSOR_API_KEY` / `--api-key`). This harness accepts only the stored account-login path for `--executor cursor`.

Authenticate first:

```bash
agent login
agent status
```

Preflight runs `agent status` after removing `CURSOR_API_KEY` and `CURSOR_AUTH_TOKEN`. It requires status output that positively identifies an authenticated account and rejects output that reports API/token authentication. The task path never supplies `--api-key` or `--auth-token`.

Run one task:

```bash
./gdpval run --executor cursor --limit 1 --out runs/cursor-smoke
```

The harness uses non-interactive `agent -p`, explicitly selects the isolated workspace, requests JSON output, enables Cursor's sandbox, and uses `--trust` only to prevent a headless workspace-trust prompt. It does **not** use `--force` / `--yolo`, which would broaden automatic approvals. Project permissions allow the shell/read/write operations needed to produce a work product while denying MCP access, WebFetch when executor networking is disabled, direct Write-tool changes to references, and `.env` access.

For tasks with reference files, the executor moves the reference tree outside the writable workspace before Cursor starts. The workspace sees it through a directory symlink, while `.cursor/sandbox.json` exposes the target only through `additionalReadonlyPaths`. The reference tree is SHA-256 verified after execution and restored before the harness copies it into judge-compatible deliverables. If the tree changes, the task fails closed.

Cursor's project `sandbox.json` also disables temp-directory writes and denies network by default. `--executor-network enabled` explicitly switches the sandbox network default to allow; this should be used only for tasks that require network access.

The task itself is stored in `GDPVAL_TASK.md`; only a fixed launcher instruction is placed on argv. The launcher explicitly prohibits Cloud Agent handoff, and the harness does not use `agent worker`, Cloud Agent APIs, `--worktree`, or a background/cloud execution mode.

Cursor account usage is not equivalent to unlimited flat-rate execution. Current Cursor plans expose different Agent usage limits, and Cursor's pricing policy allows subscription fees, usage fees, and other applicable fees depending on plan/account configuration. Check account usage and spend controls before repeated benchmark runs.

Cursor's current Terms of Service require third-party sharing of service benchmark results to include enough information for others to reproduce the tests. Preserve `run-metadata.json`, task/run settings, CLI versions, model selection, and other relevant configuration when publishing Cursor-backed results.

## Workspace and deliverables

For all three local executors, reference-file materialization by the harness is separate from the executor tool-network policy and may require network access/Hugging Face authentication before the coding agent starts.

```text
<out>/
  run-metadata.json
  tasks/<task-id>/
    workspace/
      reference_files/
      deliverables/
    executor/
      stdout.log
      stderr.log
      prompt.txt
      metadata.json
  deliverables/task_<task-id>/repeat_0/
```

Only the judge-compatible `deliverables/` tree is submitted to later GDPval grading. Executor logs and scratch workspace files are not copied into it. Nested deliverables are preserved; deliverable symlinks are rejected.

Enable model-generated network access only when a task genuinely requires it:

```bash
./gdpval run --executor codex --executor-network enabled --limit 1
```

The same flag applies to Claude Code and Cursor, using each runtime's own sandbox/network controls.

## Local subscription-backed judging

A local policy executor does not automatically judge its own output. Two completed deliverable sets can instead be compared with a separate subscription-backed judge:

```bash
./gdpval compare-runs \
  --a runs/plain/deliverables \
  --b runs/intervention/deliverables \
  --label-a plain \
  --label-b intervention \
  --judge-executor codex \
  --judge-trials 2 \
  --limit 10 \
  --out runs/comparison
```

Use `--judge-executor claude-code` for the Claude Code judge path. The same subscription-authentication checks and API credential scrubbing rules apply to judging as to policy execution.

The local judge path requires an explicit positive `--limit`; requires both candidates to contain the same GDPval task set; verifies identical reference files; rejects symlinked candidate, repeat, reference, and artifact paths; creates a fresh anonymous judge workspace for each task/trial; excludes executor bookkeeping; deterministically randomizes the first A/B placement and alternates later trials; and accepts only a standalone final `BOXED[A]`, `BOXED[B]`, or `BOXED[TIE]` verdict.

Codex judging uses an ephemeral read-only sandbox. Claude Code judging uses a fail-closed sandbox with an empty network allowlist and an OS-enforced `filesystem.denyWrite` rule for the anonymous judge workspace. Judge stdout/stderr and per-trial metadata are kept outside the anonymous submissions.

Local judge results are always labeled `official_gdpval_aa_v2=false`; they are not placed on the AA-v2 Elo scale.

If a candidate was generated by the same executor family used for judging, `local-judge-summary.json` records `same_executor_as_judge=true`. Exact model equality, when identifiable, is recorded separately. These are provenance warnings rather than bias corrections; prefer cross-family judging and/or human validation for stronger claims.

`./gdpval aa-v2` explicitly rejects `--judge-executor`. The original Stirrup + API judge panel + pinned reference Elo path is unchanged.

## Billing and comparability

Subscription/account-backed means that the harness deliberately uses the coding product's stored account authentication path instead of an API key. It does not mean usage is unlimited or has zero marginal effect: plan quotas, rate limits, usage pools, fair-use rules, on-demand settings, and product terms still apply.

The local executor environment also differs from NVIDIA's GDPval Apptainer environment. Use this path for repeated experiments and paired comparisons, not for claiming an official GDPval-AA v2 score.
