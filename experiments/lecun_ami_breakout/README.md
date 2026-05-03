# LeCun-AMI Breakout Experiments

This folder implements the FYP experiment plan for testing whether adaptive
configurator-controlled planning improves the reward/compute tradeoff on
`ALE/Breakout-v5`.

## 1. Smoke Test

Run this before launching the long jobs:

```bash
python experiments/lecun_ami_breakout/run_experiments.py --phase smoke
```

It runs the three AMI conditions for `1000` steps with plots/GIFs disabled:

- `adaptive`: `--planning-mode adaptive --eval-mode adaptive`
- `always`: `--planning-mode always --eval-mode planner`
- `actor`: `--planning-mode actor --eval-mode actor`

Use `--dry-run` to print the exact commands without launching training.

## 2. Main AMI Matrix

```bash
python experiments/lecun_ami_breakout/run_experiments.py --phase main
```

This runs seeds `1 2 3` for each condition with the planned `500000` agent-step
configuration, full-game evaluation every `50000` steps, and final evaluation
over `5` episodes.

To run one condition at a time:

```bash
python experiments/lecun_ami_breakout/run_experiments.py --phase main --conditions adaptive
python experiments/lecun_ami_breakout/run_experiments.py --phase main --conditions always
python experiments/lecun_ami_breakout/run_experiments.py --phase main --conditions actor
```

Each invocation writes a manifest CSV under:

```text
experiments/lecun_ami_breakout/runs/
```

## 3. Baselines

The Breakout baselines used for the comparison are:

Model-free:

```bash
python baselines/model-free/DQN/dqn.py
python baselines/model-free/DDQN/double_dqn.py
python baselines/model-free/PER/per_ddqn.py
```

Model-based:

```bash
python baselines/model-based/Dreamer-v2/dreamer_v2.py --env ALE/Breakout-v5
python baselines/model-based/Dreamer-v3/dreamer_v3.py --env ALE/Breakout-v5
```

You can run the baseline set from this experiment folder:

```bash
python experiments/lecun_ami_breakout/run_baselines.py --dry-run
python experiments/lecun_ami_breakout/run_baselines.py --family model-based
python experiments/lecun_ami_breakout/run_baselines.py --family model-free
```

The other model-based implementations in this repo are not used as Breakout
baselines because their current environments differ: Dreamer V1 uses
`dm_control/walker-walk-v0`, MuZero uses `CartPole-v1`, MPC uses `Pendulum-v1`,
and Dyna-Q uses `CliffWalking-v1`.

The current baseline scripts produce plots/GIFs and scheduler logs, while AMI
produces structured `eval_metrics.csv` and final JSON summaries.

The model-free Breakout baselines now also write structured outputs:

- `config.json`
- `metrics.csv`
- `training_log.csv`
- `eval_metrics.csv`
- `final_eval_metrics.json`
- `final_summary.json`

Example smoke command:

```bash
python baselines/model-free/DQN/dqn.py \
  --max-frames 1000 \
  --learning-starts 2000 \
  --eval-interval 0 \
  --eval-episodes 1 \
  --eval-max-steps 500 \
  --no-plots \
  --no-gif
```

Dreamer V2/V3 still need the same structured reporting upgrade. Until then,
the summary script reports them from artifact availability and scheduler logs.

## 4. Report Tables

After runs finish:

```bash
python experiments/lecun_ami_breakout/summarize_results.py
```

Outputs are written to:

```text
experiments/lecun_ami_breakout/reports/
```

The important files are:

- `ami_runs.csv`: one row per AMI run.
- `ami_condition_summary.csv`: mean/std summary over seeds.
- `baseline_availability.csv`: DQN/DDQN/PER/Dreamer V2/Dreamer V3 artifact and
  log summary.
- `model_based_out_of_scope.csv`: model-based baselines excluded because they do
  not run Breakout in this repo.
- `report.md`: compact report table and conclusion template.
- `ami_eval_reward_curve.png`, `ami_eval_planning_rate_curve.png`, and
  `ami_efficiency_bars.png` when enough completed AMI data exists.

By default, incomplete AMI runs are excluded from the headline tables. To inspect
partial or interrupted runs too:

```bash
python experiments/lecun_ami_breakout/summarize_results.py --include-incomplete
```

Tiny structured baseline smoke runs are ignored by default in the baseline
availability table. To include shorter baseline runs:

```bash
python experiments/lecun_ami_breakout/summarize_results.py --baseline-min-steps 1
```
