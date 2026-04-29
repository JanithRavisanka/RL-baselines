# MuZero (CartPole) Baseline

This implementation trains MuZero on `CartPole-v1` using MCTS-guided action selection and unrolled latent-model training.

## Run

From repository root:

```bash
python baselines/model-based/MuZero/muzero.py
```

Research-style custom run:

```bash
python baselines/model-based/MuZero/muzero.py \
  --num-games 5000 \
  --num-simulations 100 \
  --batch-size 128 \
  --unroll-steps 10 \
  --td-steps 10 \
  --lr 1e-3 \
  --replay-capacity 5000 \
  --warmup-games 20 \
  --train-steps-per-game 8
```

## Inputs / Arguments

- `--num-games` (default: `5000`)
- `--num-simulations` (default: `100`)
- `--batch-size` (default: `128`)
- `--unroll-steps` (default: `10`)
- `--td-steps` (default: `10`)
- `--lr` (default: `1e-3`)
- `--replay-capacity` (default: `5000`)
- `--warmup-games` (default: `20`)
- `--train-steps-per-game` (default: `8`)

Fixed in-code search constants:
- `pb_c_init=1.25`, `pb_c_base=19652`
- root Dirichlet: `alpha=0.25`, `eps=0.25`

## Inputs

- Environment: `CartPole-v1` (inferred dims from Gymnasium).
- No external datasets or config files required.

## Outputs

On each run, the script writes artifacts to:

- `results/muzero/run_<timestamp>/`

Produced files:

- `muzero_network.pth` — trained model weights (`state_dict`)
- `training_curve.png` — reward curve across games
- `muzero_agent.gif` — evaluation rollout rendered as GIF

Console output includes device selection, periodic average reward logs every 10 games, solve signal (`Solved!`) when threshold is met, and save paths.

## Result directory layout

```text
results/
  muzero/
    run_YYYYMMDD_HHMMSS/
      muzero_network.pth
      training_curve.png
      muzero_agent.gif
```

`run_...` is created automatically from current timestamp at script start.
