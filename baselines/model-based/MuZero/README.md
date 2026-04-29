# MuZero (CartPole) Baseline

This implementation trains MuZero on `CartPole-v1` using MCTS-guided action selection and unrolled latent-model training.

## Run

From repository root:

```bash
python baselines/model-based/MuZero/muzero.py
```

Or from this directory:

```bash
python muzero.py
```

## Key defaults (from code)

- Environment: `CartPole-v1`
- Network hidden size: `128`
- Search:
  - `num_simulations=25`
  - `pb_c_init=1.25`
  - `pb_c_base=19652`
  - root Dirichlet noise: `alpha=0.25`, `eps=0.25`
- Training:
  - `discount=0.99`
  - `td_steps=5`
  - `unroll_steps=5`
  - `batch_size=64`
  - `num_games=200`
  - optimizer: `Adam(lr=0.002, weight_decay=1e-4)`
  - update schedule: 2 gradient updates per episode once replay has more than 10 games
- Replay buffer capacity: `500`
- Action sampling temperature by episode:
  - `<50`: `1.0`
  - `<100`: `0.5`
  - otherwise: `0.25`

## Inputs and outputs

## Inputs

- Environment observations from Gym (`obs_dim` inferred from env).
- Discrete action space size (`action_dim` inferred from env).
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
