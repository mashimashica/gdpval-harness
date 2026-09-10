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
./gdpval run \
  --executor codex \
  --limit 1 \
  --out runs/codex-smoke
```

The harness invokes `codex exec` non-interactively, sets the task workspace with `--cd`, uses the `workspace-write` sandbox, sets `approval_policy=never`, disables outbound network for model-generated commands by default, uses an ephemeral session, and captures JSONL stdout plus the final message. The GDPval task prompt is supplied on stdin rather than argv.

Enable model-generated network access only when the task genuinely requires it:

```bash
./gdpval run --executor codex --executor-network enabled --limit 1
```

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

`claude auth status` is non-billable and returns JSON. Preflight requires `loggedIn=true`, `authMethod=claude.ai`, `apiProvider=firstParty`, and a supported subscription type. The executor removes API, bearer-token, setup-token, custom-base-URL, Bedrock, Vertex, and Foundry routing variables before both the auth check and task execution. This matters because Claude Code gives environment API credentials higher precedence than stored subscription OAuth in non-interactive `-p` mode.

Run one task:

```bash
./gdpval run \
  --executor claude-code \
  --limit 1 \
  --out runs/claude-smoke
```

The harness uses `claude -p`, JSON output, disabled session persistence, `--safe-mode`, a restricted built-in tool set, and a Bash sandbox with `failIfUnavailable=true` and `allowUnsandboxedCommands=false`. Model-generated network access uses a strict empty allowlist by default. `GDPVAL_EXECUTOR_MAX_TURNS` controls Claude Code's `--max-turns` and defaults to 250.

```bash
GDPVAL_EXECUTOR_MAX_TURNS=100 \
  ./gdpval run --executor claude-code --limit 1
```

The harness does not pass `--cloud` or `--environment`; those flags are Claude Code cloud-session paths and are intentionally outside this executor.

## Workspace and deliverables

Reference-file download by the harness is separate from the executor tool-network policy and may require network access/Hugging Face authentication before the local coding agent starts.

Per task, the working layout is:

```text
<out>/
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

## Billing and comparability

Subscription-backed means that the harness deliberately uses the coding product's account/subscription authentication path instead of an API key. It does not mean usage is unlimited or has zero marginal effect: plan quotas, rate limits, product usage policies, and fair-use rules still apply.

The local executor environment also differs from NVIDIA's GDPval Apptainer environment. Use this path for repeated experiments and paired comparisons, not for claiming an official GDPval-AA v2 score.
