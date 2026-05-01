# `double_dqn.py` Implementation Notes

## Pipeline overview

`double_dqn.py` follows the same Atari training pipeline as the DQN baseline:

1. Create `ALE/Breakout-v5` with wrappers:
   - `AtariPreprocessing` (grayscale, 84x84 resize, frame skip, life-loss terminals)
   - `FrameStackObservation` with 4 frames
2. Build two CNN Q-networks (`q_network`, `target_network`) with DQN-style conv + MLP head.
3. Collect transitions with epsilon-greedy exploration into replay memory.
4. Sample mini-batches, normalize pixels to `[0,1]`, and compute TD targets.
5. Optimize with Smooth L1 (Huber) loss + gradient clipping.
6. Periodically copy online weights to the target network.
7. Save artifacts: model checkpoint, training curve, and evaluation GIF.

## What is different from `DQN/dqn.py`

Almost everything is intentionally the same (env setup, architecture, optimizer, hyperparameters, logging cadence, saving flow).  
The optimizer is the DeepMind DQN RMSProp variant (`lr=2.5e-4`, `alpha=0.95`, `momentum=0.95`, `eps=0.01` inside the square root), not `torch.optim.RMSprop(..., eps=0.01)`.
The core algorithmic difference is the target calculation inside training:

- **DQN (`dqn.py`)**:  
  `max_next_q_values = target_network(next_states).max(...)`
- **Double DQN (`double_dqn.py`)**:
  - `best_next_actions = q_network(next_states).argmax(...)`  (selection)
  - `max_next_q_values = target_network(next_states).gather(..., best_next_actions)` (evaluation)

This selection/evaluation split is the defining DDQN change.

## Output location and naming

Running `double_dqn.py` writes under:

- `results/ddqn/run_<timestamp>/model.pth`
- `results/ddqn/run_<timestamp>/training_curve.png`
- `results/ddqn/run_<timestamp>/breakout_ddqn_agent.gif`

This mirrors DQN output structure, but uses `ddqn` paths and filenames.
