# gdpval-harness

A one-command harness for running and evaluating GDPval with configurable policy models, judge configurations, and pairwise comparisons.

This repository is a focused fork of [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). It keeps the upstream GDPval execution, Stirrup agent, artifact handling, judge panel, and Elo machinery intact while exposing a smaller command-line surface for reproducible experiments.

## What it supports

- run GDPval tasks against local or hosted models;
- use OpenAI, Gemini, OpenRouter, LiteLLM, vLLM, or a generic OpenAI-compatible endpoint as the policy backend;
- score with the GDPval rubric or the existing multi-judge panel;
- reproduce the pinned GDPval-AA v2 reference-scale comparison path;
- blind-compare two existing deliverable sets without rerunning the policy model; and
- record non-secret run metadata for reproducibility.

## CLI

```text
./gdpval run [options]
./gdpval check [options]
./gdpval aa-v2 --refs DIR [options]
./gdpval compare-runs --a DIR --b DIR [options]
./gdpval providers
```

`run` is optional, so the original short form still works:

```bash
./gdpval --limit 1
```

## Setup

Use the normal NeMo Gym environment and GDPval sandbox. The harness currently embeds rather than reimplements those components.

1. Install/sync the NeMo Gym Python environment from this repository.
2. Copy the existing GDPval recipe configuration to the repository root as `env.yaml`, or provide equivalent CLI/environment overrides.
3. For a policy rollout, set `GDPVAL_CONTAINER_PATH` to the GDPval Apptainer image and provide `TAVILY_API_KEY`.
4. Configure judge access. The default `aa-v2` panel expects one gateway that can route the configured GPT-5.5, Gemini 3.1 Pro, and Claude Opus 4.8 judge model IDs. Use `--judge-panel single` when you have only one judge endpoint.

Check the environment without starting an evaluation:

```bash
./gdpval check --provider openai --model '<model-id>'
```

`check` exits before dataset preparation, model calls, and the optional source pin.

## Run a model

OpenAI:

```bash
OPENAI_API_KEY='<key>' \
GDPVAL_JUDGE_API_KEY='<judge-key>' \
./gdpval run \
  --provider openai \
  --model '<model-id>' \
  --judge-panel single \
  --judge-model '<judge-model-id>' \
  --judge-base-url '<judge-base-url>' \
  --limit 1
```

Gemini:

```bash
GEMINI_API_KEY='<key>' \
./gdpval run --provider gemini --model '<model-id>' --limit 1
```

Claude can currently be used as the policy model through an OpenAI-compatible gateway such as OpenRouter or LiteLLM. This harness does not add a new native Anthropic Messages API model server.

List the built-in policy presets:

```bash
./gdpval providers
```

Explicit `--model-type`, `--base-url`, and `GDPVAL_API_KEY` values take precedence over provider defaults.

## GDPval-AA v2-compatible comparison

Place rated reference deliverables under one directory using the names in [`config/gdpval-aa-v2-references.tsv`](config/gdpval-aa-v2-references.tsv), then run:

```bash
./gdpval aa-v2 \
  --refs ./refs \
  --provider openai \
  --model '<model-id>'
```

The reference Elo anchors are deliberately pinned. With two or more available references, the upstream adaptive flow first places the candidate on 45 tasks and then spends the full task budget against the nearest up to four references.

You can supply another rated reference set with:

```bash
./gdpval run \
  --comparison ./refs \
  --reference-manifest ./my-references.tsv
```

A locally produced rating is not an official Artificial Analysis score. The profile reproduces the public/open implementation path as closely as this fork permits, but provider behavior, serving details, and other non-public conditions can differ.

## Compare two experiment conditions

If the same GDPval tasks have already been executed under two conditions, compare the artifacts directly:

```bash
GDPVAL_JUDGE_API_KEY='<judge-key>' \
./gdpval compare-runs \
  --a runs/plain/deliverables \
  --b runs/alps/deliverables \
  --label-a plain \
  --label-b alps \
  --judge-panel single \
  --judge-model '<judge-model-id>' \
  --judge-base-url '<judge-base-url>'
```

This is judge-only: it needs neither the GDPval sandbox nor the search key. The labels are written only to metadata and are not shown to the pairwise judge. The upstream comparison grader alternates submission positions across trials to reduce A/B position bias.

Candidate B is assigned an arbitrary Elo of 1000 only so the existing aggregation path can be reused. For two-condition experiments, use pairwise win/tie/loss and preference as the primary result rather than interpreting this as an AA-v2 Elo placement.

## Outputs and reproducibility

A normal run writes under `./results/gdpval` unless `--out` is supplied. The harness records `run-metadata.json` before execution with:

- repository commit and dirty state;
- hash of `env.yaml` when present;
- policy provider/model/backend selection;
- judge mode and non-secret judge settings;
- evaluation mode/reference manifest and its hash; and
- output, concurrency, limit, resume, and pin settings.

API-key values are never written to metadata. Use `--no-metadata` only when this provenance record is intentionally unwanted.

## Reproduction pin

Normal runs never rewrite tracked source files. The NVIDIA recipe's historical source pin remains available explicitly:

```bash
./gdpval run --pin-gym --limit 1
```

That mode uses `git restore` to reproduce the pinned Gym source revision while leaving `nemotron_recipes` untouched. Use it only on a clean working tree.

## Scope

The public surface of this fork is GDPval-specific, but the NeMo Gym engine remains embedded because the benchmark currently depends on its environment lifecycle, model servers, Stirrup agent, resources server, artifact conversion, and aggregation code. See [`UPSTREAM.md`](UPSTREAM.md).

The repository is independent and is not an official implementation or service of Artificial Analysis, OpenAI, or NVIDIA.
