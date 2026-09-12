# Upstream relationship

`gdpval-harness` is a focused fork of [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym), licensed under Apache-2.0.

The fork currently retains the upstream engine instead of copying GDPval code into a new standalone implementation. The GDPval path depends on several NeMo Gym components together:

- `benchmarks/gdpval` for task preparation and benchmark configuration;
- `responses_api_agents/stirrup_agent` for the agentic task execution path;
- `resources_servers/gdpval` for rubric scoring, blind pairwise comparison, judge sampling, artifact conversion, and Elo aggregation;
- `responses_api_models` for policy and judge backends; and
- `nemo_gym` for environment lifecycle and evaluation orchestration.

The root `./gdpval` command is intentionally a thin orchestration layer over those components.

The generated [upstream environment inventory](UPSTREAM-ENVIRONMENTS.md) remains available separately from the GDPval-focused root README.

## Fork policy

Harness-specific behavior should stay at the root CLI, `scripts/gdpval_*`, and `config/` where possible. Changes to embedded upstream GDPval code should be kept small and justified by a harness requirement.

Large-scale deletion of non-GDPval NeMo Gym code is deferred until the full policy-run, judge-only, AA-v2 comparison, and artifact-conversion paths have been exercised end-to-end with real credentials. Preserving a known-good engine is currently more important than minimizing repository size.

When syncing from upstream, review changes to the retained GDPval dependency path before accepting them; benchmark behavior is part of the evaluation protocol.
