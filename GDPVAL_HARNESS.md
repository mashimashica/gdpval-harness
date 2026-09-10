# GDPval Harness

This fork exposes NVIDIA NeMo Gym's GDPval recipe through a small root-level runner while keeping the underlying benchmark implementation intact.

## Prerequisites

Use the existing NeMo Gym GDPval setup. In particular, a normal agent run requires:

- an active Gym Python environment;
- `env.yaml` configured for the policy model and judge endpoint;
- the API keys required by that configuration;
- `GDPVAL_CONTAINER_PATH` pointing to the GDPval Apptainer sandbox; and
- `TAVILY_API_KEY` for the Stirrup agent's web search.

Judge-only runs reuse existing deliverables and therefore do not require the sandbox or search key.

## Smoke test

From the repository root:

```bash
./gdpval --limit 1
```

The runner changes only invocation ergonomics. Model endpoint and model ID configuration remain in `env.yaml`.

## Common modes

Run a small sample with controlled concurrency:

```bash
./gdpval --limit 3 --parallel 3
```

Re-score existing deliverables:

```bash
./gdpval --judge-only
```

Run pairwise comparison against rated reference deliverables:

```bash
./gdpval --comparison ./refs --limit 45
```

Forward a different Gym response model type:

```bash
./gdpval --model-type <gym-model-type> --limit 1
```

The default remains `vllm_model`. vLLM-specific Nemotron overrides are applied only for that model type.

## Reproducibility pin

Normal harness runs use the current checkout and do not rewrite tracked source files.

The upstream recipe also contains a pinned-revision reproduction mode. It is now opt-in:

```bash
./gdpval --pin-gym --limit 1
```

That mode restores tracked Gym source files outside `nemotron_recipes` from the pinned commit, so use it only in a clean working tree.

## Scope

This repository is an independent harness built from the NeMo Gym implementation. It is not an official Artificial Analysis implementation and does not by itself make a score an official GDPval-AA score.
