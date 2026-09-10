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

Without model override flags, the policy endpoint, API key, and model ID continue to come from `env.yaml`.

## Policy model selection

The harness can override the model under test without editing `env.yaml`:

```bash
GDPVAL_API_KEY='<policy-api-key>' \
  ./gdpval \
  --model-type openai_model \
  --base-url '<openai-compatible-base-url>' \
  --model '<model-id>' \
  --limit 1
```

`--base-url` and `--model` override only the policy model. Set `GDPVAL_API_KEY` to override only its API key; the runner passes an OmegaConf environment interpolation rather than the secret itself on the command line.

The shared overrides work with Gym model backends that consume `policy_base_url`, `policy_api_key`, and `policy_model_name`, including `vllm_model`, `openai_model`, and `litellm_model`. The default model type remains `vllm_model`, and its Nemotron-specific overrides are applied only in that mode.

Judge selection and judge credentials are intentionally separate and remain configured in `env.yaml`.

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

## Reproducibility pin

Normal harness runs use the current checkout and do not rewrite tracked source files.

The upstream recipe also contains a pinned-revision reproduction mode. It is opt-in:

```bash
./gdpval --pin-gym --limit 1
```

That mode restores tracked Gym source files outside `nemotron_recipes` from the pinned commit, so use it only in a clean working tree.

## Scope

This repository is an independent harness built from the NeMo Gym implementation. It is not an official Artificial Analysis implementation and does not by itself make a score an official GDPval-AA score.
