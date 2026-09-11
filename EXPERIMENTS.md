# Experiment conditions

`gdpval-harness` treats an experiment condition as an external intervention applied to a policy run. It is deliberately separate from the GDPval task, executor, model provider, and judge.

The harness does not know what `plain`, `alps`, `prompt-v2`, or any other condition means. A condition label is provenance only. Optional condition instructions are supplied explicitly through a text file and are supported only by local policy executors.

## Paired local experiment

Run the control condition with the same executor and model/runtime settings:

```bash
./gdpval run \
  --executor codex \
  --condition plain \
  --limit 10 \
  --out runs/plain
```

Run the intervention condition:

```bash
./gdpval run \
  --executor codex \
  --condition intervention \
  --condition-file ./intervention.md \
  --limit 10 \
  --out runs/intervention
```

Then compare the completed deliverables independently with the verified local blind-judge path:

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

Claude Code remains supported as a policy executor, but its local blind-judge path currently fails closed before any model call because the harness cannot verify read confinement with a documented non-model probe.

For an ALPS experiment, `intervention.md` can contain the ALPS-derived instructions or work-design material. ALPS is not a built-in GDPval mode and no ALPS-specific behavior is hard-coded into the harness.

## Semantics

`--condition LABEL` records an arbitrary label in run provenance. The label itself is not inserted into the model prompt.

`--condition-file FILE` loads UTF-8 text, validates it before the first model task, and inserts it into the local executor prompt as an explicitly external experiment-condition section. The file must be non-empty and no larger than 1 MiB. Its path and SHA-256 are recorded in `run-metadata.json`; secrets should never be placed in a condition file.

Each local task also records the canonical GDPval prompt separately as `tasks/<task-id>/executor/task-prompt.txt`. This file is not sent to the executor as an extra instruction; it is provenance for later blind comparison. Condition text can therefore contain its own templates or `Task:` headings without shadowing the base GDPval prompt used for comparison.

`--condition-file` is currently rejected for `executor=stirrup` rather than being silently ignored. `aa-v2` rejects all condition overrides so the reproduction path remains protocol-separated. `compare-runs` uses `--label-a` and `--label-b`; it does not accept policy-run condition options.

## Resume safety

A local policy run may use `--resume`, but a conditioned resume must match the existing run provenance. Before overwriting run metadata or skipping completed tasks, the harness compares the requested condition label, condition-file SHA-256, and whether condition instructions were applied with the existing `run-metadata.json`.

If those values differ, resume fails closed and the caller must use the original condition or a new output directory. A conditioned run created with `--no-metadata` cannot be safely resumed with condition options because there is no run-level condition provenance to verify.

This prevents a partial output tree from mixing artifacts generated under different interventions while presenting them as one condition.

## Reproducibility

A conditioned local run records:

- condition label;
- condition file path and SHA-256 when present;
- whether condition instructions were applied to the policy prompt;
- the canonical GDPval task prompt separately from the executor wrapper;
- executor, version, authentication mode, selected model, workspace isolation, network policy, and tool-permission mode;
- repository commit and other existing harness provenance.

Keep the condition file together with the run metadata when publishing or reviewing an experiment. The label is descriptive metadata, not evidence that a particular intervention was actually applied; `condition_file_sha256` and `condition_applied_to_prompt` provide the machine-recorded link to the applied instructions.

## Codex subscription boundary

For Codex local policy execution and Codex local judging, the harness pins ChatGPT subscription authentication with `forced_login_method="chatgpt"`. When model-generated network access is disabled, policy execution also sets `web_search="disabled"`; the local Codex blind judge disables web search and additionally enforces the root-deny permission profile from the local-judge threat model. These controls supplement API-credential scrubbing, positive login-status checks, and the non-model read-confinement probe so the subscription-backed path fails closed instead of silently switching authentication modes.
