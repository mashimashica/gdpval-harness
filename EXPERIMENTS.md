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

Then compare the completed deliverables independently using the verified local blind-judge path:

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

For an ALPS experiment, `intervention.md` can contain the ALPS-derived instructions or work-design material. ALPS is not a built-in GDPval mode and no ALPS-specific behavior is hard-coded into the harness.

## Pinned ALPS creation-time profile

`config/experiments/alps-skill-creation.json` records a generic Builder experiment. The `skill-creator-only` arm is the baseline: the Builder receives the caller-supplied self-contained `skill-creator` source root with the five-file creation-time closure `SKILL.md`, `references/openai_yaml.md`, `scripts/generate_openai_yaml.py`, `scripts/init_skill.py`, and `scripts/quick_validate.py`. Those five semantic files match OpenAI's bundled [`skills/.system/skill-creator`](https://github.com/openai/skills/tree/main/skills/.system/skill-creator) distribution. Its UI metadata, icons, and license packaging are excluded. The `skill-creator-plus-alps` arm is the treatment: it receives the same exact `skill-creator` bundle plus the pinned `alps-work-design` input. The hypothesis concerns ALPS as a creation-time Builder input; it does not treat ALPS as an application-time plugin or a native runtime mode.

The `skill-creator` revision is intentionally unavailable and has no expected bundle hash in this profile; the linked mutable `main` branch is descriptive source identification, not a pin. Loader and outer run provenance capture the bound file and bundle hashes, and the caller must bind the same loaded five-file source root to both arms. A different Skill Creator distribution requires its own generic profile with its full explicit allowlist; the harness core has no Skill Creator-specific path or exception.

Run the profile by binding both input roots explicitly and keeping the Builder and application settings common:

```bash
./eval experiment config/experiments/alps-skill-creation.json \
  --input-root skill-creator=/absolute/path/to/skill-creator-source \
  --input-root alps-work-design=/absolute/path/to/alps-checkout \
  --builder-executor codex \
  --executor codex \
  --builder-model '<common-model-id>' \
  --model '<common-model-id>' \
  --builder-timeout 12600 \
  --executor-timeout 12600 \
  --limit 10 \
  --order-seed 7 \
  --out /absolute/path/to/alps-results \
  --runtime-root /absolute/path/to/neutral-runtime
```

The ALPS checkout must have HEAD `cf31ca93a1b5379e2ddbd430f9ea416192fb6797`, and its 14-file English creation-time closure must validate to bundle SHA-256 `1feb183fb6e01f7b48469c3669968df0741d93ddde428cfaad6e05350c2f28c4`. The closure includes both ALPS Skills, their English design references, and the optional worked example. The worked example's `baseline.csv` and `candidate.csv` are static files shipped in that pinned ALPS source; they are neither outputs of this GDPval experiment nor another arm or condition. The closure deliberately excludes host UI and icons, Japanese locales, plugin/root installation and provenance files, and other packaging material. The external `agentskills.io` URLs referenced by ALPS remain unpinned dependencies.

After the Builder seals a generated Skill, the application stage receives only that newly sealed Skill. It does not receive the ALPS source or plugin. The profile and core remain generic: ALPS-specific behavior is carried by the input data and profile arms, not by a conditional in the harness. Use a condition-neutral runtime root such as `/absolute/path/to/neutral-runtime`; its name and all ancestors should omit profile, input, arm, source, and evaluation labels. The `--out` path is outer provenance and may remain experiment-specific. Shared task selection, executor/model/timeout settings, order seed, and input hashes are outer provenance. They do not prove semantic non-leakage inside generated Skill prose, and OS-level read confinement remains unverified; use the dedicated neutral runtime path for that reason.

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

For Codex local policy execution and Codex local judging, the harness sets `forced_login_method="chatgpt"`. When model-generated network access is disabled, policy execution also sets `web_search="disabled"`; the local Codex judge always disables web search and retains the verified root-deny/workspace-read permission profile. These controls supplement the existing API-credential scrubbing and login-status preflight so the subscription-backed path fails closed instead of silently switching authentication modes.

Claude Code remains available as a subscription-backed policy executor, but its local blind-judge path currently fails closed before `claude -p` because the harness cannot verify read confinement through a documented non-model probe. Use Codex or human validation for blind comparison until that boundary can be established.
