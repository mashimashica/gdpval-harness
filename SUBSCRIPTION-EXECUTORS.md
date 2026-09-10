# Subscription-backed local executors

Local executors run GDPval task workspaces on the user's machine while using an authenticated coding-agent CLI for model access. They are intentionally separate from the NeMo Gym / Stirrup provider path and from judging.

These runs are **not** GDPval-AA v2 reproduction runs. `./gdpval aa-v2` remains the Stirrup + AA-v2 judge/reference path.

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

An explicit `--limit` is required for subscription-backed executors so an accidental invocation cannot consume the full 220-task benchmark. Local executors are serial for now (`--parallel 1`).

The harness invokes `codex exec` non-interactively, sets the task workspace with `--cd`, uses the `workspace-write` sandbox, disables approval prompts, disables outbound network for model-generated commands by default, uses an ephemeral session, and captures JSONL stdout plus the final message. The GDPval task prompt is supplied on stdin rather than argv.

Enable model-generated network access only when the task genuinely requires it:

```bash
./gdpval run --executor codex --executor-network enabled --limit 1
```

Reference-file download by the harness is separate from this tool-level network policy and may require network access/Hugging Face authentication before the Codex subprocess starts.

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
      final-message.txt
      metadata.json
  deliverables/task_<task-id>/repeat_0/
```

Only the judge-compatible `deliverables/` tree is submitted to later GDPval grading. Executor logs and scratch workspace files are not copied into it.

## Billing and comparability

Subscription-backed means that the harness deliberately uses the coding product's account/subscription authentication path instead of an API key. It does not mean usage is unlimited or has zero marginal effect: plan quotas, rate limits, product usage policies, and fair-use rules still apply.

The local executor environment also differs from NVIDIA's GDPval Apptainer environment. Use this path for repeated experiments and paired comparisons, not for claiming an official GDPval-AA v2 score.
