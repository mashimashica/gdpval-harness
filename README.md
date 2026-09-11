# gdpval-harness

A reproducible harness for running GDPval with configurable agent executors, model providers, and judges.

This repository is a focused fork of [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). It keeps the existing Stirrup, artifact handling, GDPval judge panel, pairwise comparison, and Elo machinery while adding local subscription/account-backed execution for repeated experiments.

## Execution modes

| Mode | Policy execution | Judge | Intended use |
| --- | --- | --- | --- |
| AA-v2-compatible reproduction | NeMo Gym / Stirrup + model API | existing AA-v2-style API panel | reference-scale benchmarking |
| Local experiment | Codex CLI / Claude Code / Cursor Agent CLI | separate | low-cost repeated artifact generation |
| Local blind comparison | existing artifact sets | Codex CLI | low-cost A/B experiments |
| Human validation | any artifact sets | human | HITL calibration and final review |

Local execution or local judging does **not** produce an official GDPval-AA v2 score.

## CLI

```text
./gdpval run [options]
./gdpval check [options]
./gdpval aa-v2 --refs DIR [options]
./gdpval compare-runs --a DIR --b DIR [options]
./gdpval executors
./gdpval providers
```

`executor` is the agent runtime that performs the task. `provider` is the model backend used only by `executor=stirrup`. `judge` / `judge-executor` is independent of both.

## Subscription/account-backed local executors

List supported runtimes:

```bash
./gdpval executors
```

Current local executors:

- `codex`: local Codex CLI with ChatGPT account authentication only;
- `claude-code`: local Claude Code with first-party Claude subscription authentication only;
- `cursor`: local Cursor Agent CLI with stored Cursor account authentication;
- `stirrup`: existing NeMo Gym / Stirrup API-backed path.

Check authentication and local prerequisites without issuing a model task:

```bash
./gdpval check --executor codex
./gdpval check --executor claude-code
./gdpval check --executor cursor
```

Run a single GDPval task:

```bash
./gdpval run --executor codex --limit 1 --out runs/codex-smoke
./gdpval run --executor claude-code --limit 1 --out runs/claude-smoke
./gdpval run --executor cursor --limit 1 --out runs/cursor-smoke
```

An explicit `--limit` is required for subscription/account-backed executors and execution is serial by default. Local adapters remove documented API credential/routing environment variables and fail closed rather than silently falling back to API billing. They do not intentionally invoke vendor cloud-agent execution paths.

The local workspace is isolated per task. Reference files are materialized before execution, and only files placed by the agent under `workspace/deliverables/` are copied into the judge-compatible GDPval deliverables tree.

See [SUBSCRIPTION-EXECUTORS.md](SUBSCRIPTION-EXECUTORS.md) for authentication, sandbox, network, usage, and vendor-specific details.

## Low-cost blind A/B comparison

Generate two conditions with the same executor/model/runtime, then judge the resulting artifacts separately:

```bash
./gdpval run \
  --executor codex \
  --limit 10 \
  --out runs/plain

./gdpval run \
  --executor codex \
  --limit 10 \
  --out runs/intervention

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

The verified subscription-backed blind-judge path currently uses `--judge-executor codex`. It pre-validates the selected task pairs, constructs per-trial anonymous workspaces, normalizes filesystem metadata, removes candidate provenance from the judge environment, probes Codex read confinement without a model call, alternates A/B positions, and reports win/tie/loss results.

`--judge-executor claude-code` is intentionally fail-closed for blind judging at present. Claude Code remains fully supported as a policy executor, but the harness does not issue a Claude judge model call unless filesystem read confinement can first be verified through a documented non-model runtime probe. Use Codex or human validation for blind judging meanwhile.

Using the same executor/model family for generation and judging can introduce evaluator dependence. The comparison summary records when the candidate run metadata identifies the same executor as the local judge. Cross-family judging or human validation is recommended for stronger claims.

Local judge results always identify themselves as non-AA-v2 results.

## GDPval-AA v2-compatible reproduction

The existing reproduction path remains Stirrup-based:

```bash
./gdpval aa-v2 \
  --refs ./refs \
  --executor stirrup \
  --provider openai \
  --model '<model-id>'
```

The command explicitly rejects local policy executors and local judge executors. It retains the existing API judge panel and the pinned reference Elo manifest at [`config/gdpval-aa-v2-references.tsv`](config/gdpval-aa-v2-references.tsv).

A locally produced rating remains an independent reproduction result, not an official Artificial Analysis score.

## Stirrup / provider-backed execution

The original NeMo Gym route remains available for benchmark reproduction and API-backed runs:

```bash
OPENAI_API_KEY='<policy-key>' \
GDPVAL_JUDGE_API_KEY='<judge-key>' \
./gdpval run \
  --executor stirrup \
  --provider openai \
  --model '<model-id>' \
  --limit 1
```

Available provider presets can be listed with:

```bash
./gdpval providers
```

The Stirrup path retains the GDPval Apptainer sandbox, model-server abstraction, existing judge configuration, and reference/Elo machinery. See [UPSTREAM.md](UPSTREAM.md) for the retained upstream scope.

## Outputs and reproducibility

Local policy runs use this structure:

```text
<out>/
  run-metadata.json
  executor-summary.json
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

Local blind comparisons additionally write:

```text
<out>/
  run-metadata.json
  local-judge-results.jsonl
  local-judge-summary.json
  judge/tasks/<task-id>/trial_<n>/...
```

Metadata records non-secret provenance including repository commit, executor/judge executor and versions, invocation/auth modes, selected model, workspace isolation, network/tool policy, timestamps, task/run limits, reference configuration, and exit state. API keys, OAuth/session credentials, and account tokens are not recorded.

## Safety and scope

- Subscription/account-backed execution is not the same as unlimited or zero-cost use. Plan limits, usage pools, rate limits, on-demand billing settings, fair-use policies, and vendor terms still apply.
- The local machine is not NVIDIA's GDPval Apptainer environment. Local runs are most useful for controlled relative experiments rather than claiming protocol-identical AA-v2 results.
- `aa-v2` remains the compatibility/reproduction path and is intentionally isolated from local executor shortcuts.
- The repository is independent and is not an official implementation or service of Artificial Analysis, OpenAI, Anthropic, Cursor, or NVIDIA.
