# Baseline Fix Plan

This plan is ordered by what most affects FYP defensibility for the LeCun-AMI Breakout comparison.

## P0: Make Results Comparable

Progress started:

- DQN, DDQN, and PER DDQN now emit structured config, episode metrics,
  training logs, evaluation CSV, final evaluation JSON, and final summary JSON.
- Dreamer V2 and Dreamer V3 still need this structured reporting upgrade.

Add a shared reporting pattern to DQN, DDQN, PER DDQN, Dreamer V2, and Dreamer V3:

- write `config.json` with CLI args, environment, device, and timestamp
- write `eval_metrics.csv` with `global_step`, reward mean/std/min/max, episode count, and max steps
- write `final_eval_metrics.json`
- write `final_summary.json`
- expose `--seed`, `--eval-episodes`, `--eval-interval`, `--eval-max-steps`, `--no-gif`, and `--no-plots`

Acceptance criteria:

- each baseline can produce one report row without parsing scheduler logs
- each baseline can run seeds `1, 2, 3`
- final report can compare AMI and baselines from structured files only

## P1: Fix Dreamer V3 Run Reliability

Dreamer V3 currently has a critical runtime risk because an existing scheduler run failed with CUDA OOM.

Add one safe preset before using Dreamer V3 in the final comparison:

- lower `--batch-size`
- lower `--seq-len`
- lower `--prefill`
- optionally lower model depth/hidden size through CLI flags
- add a clear error message when GPU memory is likely insufficient

Acceptance criteria:

- Dreamer V3 smoke run completes
- Dreamer V3 reaches evaluation without CUDA OOM
- `baseline_availability.csv` no longer marks the selected Dreamer V3 result as failed

## P2: Standardize Evaluation Protocol

For all baselines:

- evaluate with full-game episodes, not life-loss terminals
- use `5` evaluation episodes by default
- report the same max evaluation step limit as AMI or document a deliberate difference
- avoid relying only on GIF reward as the final score

For DQN/DDQN/PER:

- keep training life-loss terminals because that is already explicit and common for Atari training
- keep evaluation with `terminal_on_life_loss=False`

For Dreamer V2/V3:

- either keep the compact raw-RGB 64x64 protocol and label it clearly
- or add an optional standardized Breakout evaluation wrapper that matches AMI/DQN evaluation more closely

Acceptance criteria:

- the audit report can mark evaluation protocol as `match` or documented `acceptable simplification`
- report tables do not mix scheduler-log estimates with final structured metrics

## P3: Documentation Cleanup

Update baseline READMEs and experiment report notes:

- DQN/DDQN/PER replay capacity is `100000`, not paper-scale `1000000`
- DQN/DDQN/PER are architecture-faithful but resource-scaled
- Dreamer V2 is a compact Dreamer V2 Atari baseline
- Dreamer V3 is a compact Dreamer V3-style baseline unless expanded to the full official recipe
- Dreamer V2/V3 preprocessing differs from the DQN/AMI grayscale frame-stack pipeline

Acceptance criteria:

- no baseline README implies strict paper reproduction
- the final FYP report distinguishes architecture fidelity from exact benchmark reproduction

## P4: Optional Architecture Tightening

Only do this after P0-P3:

- add optional DQN/DDQN/PER checkpoint evaluation every fixed frame count
- add optional baseline manifests matching `run_experiments.py`
- add Dreamer V2/V3 `config.json` and checkpoint intervals
- add a small `baseline_architecture_audit.py` generator if the audit should be regenerated automatically

Acceptance criteria:

- audit artifacts can be refreshed after code changes
- fix status can be tracked without manually rewriting tables
