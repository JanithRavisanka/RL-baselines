# DQN Implementation Mapping (`dqn.py`)

This maps the current implementation in `dqn.py` to concrete components and behavior.

## File Structure Overview

- `QNetwork`: CNN-based Atari Q-function (3 conv layers + 2 linear layers).
- `ReplayBuffer`: deque-backed uniform replay buffer with random sampling.
- `make_env()`: Atari environment creation + preprocessing wrappers.
- `train()`: full online training loop with epsilon-greedy behavior and target network updates.
- `plot_rewards()`: saves a training curve PNG.
- `evaluate_and_record()`: runs one deterministic episode and writes a GIF.
- `__main__`: creates timestamped results directory, trains, saves model/plots/GIF.

## Environment + Wrappers

`make_env(env_name="ALE/Breakout-v5", render_mode=None)`:

- Creates env with `frameskip=1`.
- Wraps with `AtariPreprocessing(..., frame_skip=4, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)`.
  - grayscale 84x84 frames,
  - no in-wrapper normalization (pixels remain 0-255),
  - life loss treated as terminal for training transitions.
- Wraps with `FrameStackObservation(stack_size=4)` to provide temporal context.

## Training Logic

### Action Selection

- Epsilon-greedy:
  - random action with probability `epsilon`;
  - else `argmax` from `q_network`.
- State tensors are normalized in-model-input path via division by `255.0`.

### Replay and Learning

- Optimizer: DeepMind-style RMSProp (`lr=2.5e-4`, `alpha=0.95`, `momentum=0.95`, `eps=0.01`, eps inside the square root).
- Replay capacity: `100_000`.
- Batch size: `32`.
- Starts gradient updates after `50_000` replay transitions.
- Sampled tensors:
  - states/next_states converted to float and normalized to `[0,1]`,
  - actions gathered via `gather(1, actions_tensor)`.

### Target and Loss

- Discount: `gamma = 0.99`.
- Target computed with **target network** max over next actions:
  - `target = r + gamma * max_a' Q_target(s', a')` for non-terminal;
  - terminal masked via `(~dones_tensor)`.
- Reward clipping used before storage: `np.sign(reward)`.
- Loss: `F.smooth_l1_loss` (Huber loss).
- Gradient clipping: `clip_grad_norm_(..., max_norm=10.0)`.

### Target Network Update

- Hard update every `10000` frames:
  - `target_network.load_state_dict(q_network.state_dict())`.
- Initialized as a full copy of online network before training begins.

## Evaluation Behavior

`evaluate_and_record(model, save_dir)` characteristics:

- Uses wrapped env with `render_mode="rgb_array"`.
- Runs one episode with deterministic greedy policy (`argmax` only, no epsilon).
- Forces one initial `env.step(1)` to trigger Breakout "FIRE" launch.
- Appends rendered frames and writes `breakout_dqn_agent.gif` at `fps=30`.
- Includes safety cap `len(frames) > 500` to prevent huge/stuck GIF generation.
- Prints final episode reward.
