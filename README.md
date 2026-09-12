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

## Generic evaluator CLI

The `./eval` CLI selects the default evaluator registered for each benchmark key and keeps evaluation separate from task execution:

```bash
./eval benchmarks
./eval run gdpval --executor codex --limit 1 --out runs/eval-gdpval
```

AIME26 reports `accuracy` after a preflight requiring `math-verify==0.8.0`. BigCodeBench reports `pass_rate` through its separate grader virtual environment and process. For generic GDPval, each task's **evaluation status** is `external`; a run status of `completed` means execution and handoff completed and is not a successful evaluation or score. The CLI exposes this distinction through `evaluation_status_counts`. Generic GDPval makes no rubric, pairwise, model, or judge call, and exports submitted deliverables to `<out>/deliverables/task_<id>/repeat_0` for the existing `./gdpval compare-runs` path. The official rubric and AA-v2 reproduction remain on the existing Stirrup/API routes.

Programmatic callers can inject an explicit evaluator into the generic runner when its `validate_plan` accepts the benchmark's candidate layout. The pairwise adapter is opt-in, requires an injected `JudgeExecutor`, and evaluates exactly two candidates; it is not the GDPval CLI default.

The generic runner composes the registered `Benchmark × Executor × Intervention × Evaluator` axes. The default intervention is identity, and source-backed interventions are selected explicitly:

```bash
./eval run aime26 --executor codex --limit 1 --intervention none
./eval run aime26 --executor codex --limit 1 --intervention prompt-overlay --intervention-source ./overlay.txt
./eval run bigcodebench --executor codex --limit 1 --intervention files --intervention-source ./reviewed-files
./eval run aime26 --executor codex --limit 1 --intervention agent-skill --intervention-source ./skills/reviewed-skill
```

`TaskSpec` contains only the benchmark task id and canonical prompt. Intervention application changes the executor task or workspace and records reviewed hashes, logical materialized files, and static application evidence separately. External source paths and outer condition labels are excluded from the derived `TaskSpec`, materialized workspace, executor argv, and child environment. Intervention content crosses those boundaries only through its explicitly selected application method (prompt overlay, workspace files, or workspace-reference Agent Skill); hashes, revisions, and application evidence are persisted outside `TaskSpec`. Generic runs do not resume and refuse to overwrite an existing output directory. Registered benchmark, intervention, evaluator, and executor combinations still enforce their own preflight and plan compatibility; the axes do not imply arbitrary cross-product support.

The `agent-skill` intervention accepts the exact Skill directory, whose frontmatter `name` must match the directory name, and applies the portable `workspace-reference` method. The canonical external source directory/reference is recorded only in outer run metadata; bundle, manifest, and file hashes are the stable evidence, and the reference is absent from per-task application records. It validates `SKILL.md` frontmatter and the Agent Skills `name` rules from the [official specification](https://agentskills.io/specification); the harness separately requires strict UTF-8, regular files only, and rejects symlinks, special files, path escapes, collisions, overwrites, and source tampering. It materializes a neutral `.gdpval/interventions/<name>/SKILL.md` path and all validated resources in the executor workspace. The current Codex, Claude, and Cursor workspace-reading adapters follow the [executor request contract](gdpval_harness/executors/base.py): the workspace is their working directory and the derived `TaskSpec` prompt is supplied to them, so the same reference works portably across those adapters. This path does not discover native `.agents/`, `.claude/`, or `.cursor/` directories and has no silent fallback. Generic GDPval evaluation remains external.

### Programmatic Builder pre-stage

The experiment architecture remains `Benchmark × Executor × Intervention × Evaluator`. A programmatic Builder is a pre-stage that creates an intervention before an application run: `Builder -> sealed InterventionBundle`, then `AgentSkillIntervention(bundle) -> fresh application run`. Builder is not a fifth experiment axis. `TaskSpec` still contains only `task_id` and the canonical prompt. The generic `ExecutorSkillBuilder` receives that `TaskSpec` plus explicit allowlisted creation-time `BuilderInputBundle`s; it never receives evaluation metadata, a rubric, or a reference answer.

`ExecutorSkillBuilder` stages those inputs under neutral paths, runs the builder executor, and requires exactly one generated Skill directory whose name matches the `name` in `SKILL.md`. The Agent Skill loader validates the generated tree. The validated tree is then copied into a separately sealed artifact root. Only that sealed bundle crosses into a distinct application workspace and executor invocation through `AgentSkillIntervention`; no Builder `ExecutionResult`, conversation, session, or resume state is handed off.

Builder logs, workspace files, and generated deliverables live under the caller-supplied persistent build runtime and remain durable. For application runs, `run_benchmark(..., runtime_root=...)` can put task workspaces, executor logs, and task result records in a fresh external neutral root while judge-compatible deliverables and run metadata remain under `out_dir`. With no `runtime_root`, the default runtime layout is unchanged, preserving `./gdpval compare-runs` discovery.

Builds and runs refuse to overwrite roots and keep source, build runtime, sealed artifact, application runtime, and output roots separate. Creation inputs use an explicit allowlist; the Builder does not copy arbitrary `HOME`, authentication, history, or profile directories.

These controls mechanically separate harness-provided prompt, environment, session, workspace, and file handoff. They do not prove semantic non-leakage within model-generated Skill prose or establish OS-level filesystem read confinement for Codex, Claude, or Cursor. Native read confinement remains unverified, so callers should place application runtime in a dedicated neutral root. This flow does not add a generic sandbox.

The pinned [ALPS creation-time profile](EXPERIMENTS.md#pinned-alps-creation-time-profile) shows how to supply a work-design reference to the Builder through the generic profile API. Its baseline arm receives a caller-supplied self-contained five-file `skill-creator` source root (`SKILL.md`, `references/openai_yaml.md`, `scripts/generate_openai_yaml.py`, `scripts/init_skill.py`, and `scripts/quick_validate.py`), matching OpenAI's bundled [`skills/.system/skill-creator`](https://github.com/openai/skills/tree/main/skills/.system/skill-creator) distribution. Its revision remains unavailable in this profile; the mutable `main` link identifies the inspected distribution and is not a pin. The treatment arm receives that same sealed input plus a 14-file English ALPS creation-time closure pinned to commit `cf31ca93a1b5379e2ddbd430f9ea416192fb6797`. The Skill Creator UI metadata, icons, and license packaging are excluded. The ALPS bundle includes the optional worked example and is checked by bundle SHA-256 `1feb183fb6e01f7b48469c3669968df0741d93ddde428cfaad6e05350c2f28c4`.

This profile describes creation-time input to the Builder, not native ALPS plugin installation or application runtime behavior. The application receives only the newly sealed generated Skill; ALPS source is not handed to it. The worked example's `baseline.csv` and `candidate.csv` are static files from the pinned ALPS source, not experiment outputs or arm conditions. Host UI, icons, locales, plugin/root installation and provenance files are excluded from the allowlist, and the external `agentskills.io` references remain unpinned. A different Skill Creator distribution needs its own generic profile with a full explicit allowlist; no core special case exists. No ALPS-specific branch exists in the harness core.

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

Codex paths accept an optional requested reasoning effort from the allowlist `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`:

```bash
./eval run gdpval --executor codex --reasoning-effort high --limit 1
./eval experiment PROFILE --builder-reasoning-effort high --application-reasoning-effort max \
  --input-root INPUT_ID=DIR --limit 1 --order-seed 0 --out runs/experiment --runtime-root runs/runtime
./gdpval run --executor codex --reasoning-effort high --limit 1
./gdpval compare-runs --a runs/plain/deliverables --b runs/intervention/deliverables \
  --judge-executor codex --judge-reasoning-effort high --judge-trials 2
```

The legacy policy flag maps to `GDPVAL_REASONING_EFFORT`; the local compare-runs judge flag maps to
`GDPVAL_JUDGE_REASONING_EFFORT`. Executor and judge result records call these values `reasoning_effort_requested`;
the experiment run configuration also keeps the independent Builder and application values under their role-qualified
configuration keys and descriptors. This records the requested configuration and does not assert that a runtime
honored it. If the installed Codex runtime or model rejects the requested value, including literal `max`, the harness
records the failure and stops remaining work without retrying or falling back to another effort or default. The flags
are rejected for non-Codex executors and judges.

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

### Generic run reproducibility record

Generic runs write `run-metadata.json` with schema version 4. The record keeps task identity separate from prompt
evidence. `task_id` identifies the benchmark row, while `task_sha256` is the SHA-256 of the exact UTF-8 bytes in that
task's canonical `task-prompt.txt`, never a hash of `task_id`.

The schema records benchmark revision and revision status; typed executor identity, version, model, invocation mode,
authentication mode, and runtime; intervention identity and type, revision/status, reviewed hashes, logical files, and
application method; evaluator identity, version, and revision, with explicit judge
applicability; the actual application run ID;
repository commit and worktree status; `configuration_sha256` for the allowlisted configuration and a stable
`run_fingerprint_sha256`; and start time, finish time, and status. Unknown optional descriptor values are `null`;
revision availability and judge applicability use explicit companion fields, and outer experiment application IDs use
`application_run_id_status`. Existing metrics and outcomes are unchanged.

For compatibility, every execution record retains `execution.metadata` as an empty mapping (`{}`). Arbitrary
`ExecutionResult.metadata`, environment values, commands, credentials, output payloads, and preflight details are not
copied into provenance JSON or hashes. Raw executor logs remain separate and may contain model or tool output, so these
records do not provide blanket secret detection.

API keys, OAuth/session credentials, and account tokens are not recorded in the reproducibility metadata.

## Safety and scope

- Subscription/account-backed execution is not the same as unlimited or zero-cost use. Plan limits, usage pools, rate limits, on-demand billing settings, fair-use policies, and vendor terms still apply.
- The local machine is not NVIDIA's GDPval Apptainer environment. Local runs are most useful for controlled relative experiments rather than claiming protocol-identical AA-v2 results.
- `aa-v2` remains the compatibility/reproduction path and is intentionally isolated from local executor shortcuts.
- The repository is independent and is not an official implementation or service of Artificial Analysis, OpenAI, Anthropic, Cursor, or NVIDIA.
