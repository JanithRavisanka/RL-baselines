# MuZero Implementation Notes (`muzero.py`)

This file maps the current code to MuZero concepts and calls out where the implementation is intentionally simplified versus canonical MuZero.

## Component mapping

## Network: representation / dynamics / prediction

- `MuZeroNetwork.representation`: `obs -> hidden_state` (`h`).
- `MuZeroNetwork.dynamics_state` + `MuZeroNetwork.dynamics_reward`: `(hidden_state, action_one_hot) -> (next_hidden_state, reward)` (`g`).
- `MuZeroNetwork.prediction_policy` + `MuZeroNetwork.prediction_value`: `hidden_state -> (policy_logits, value)` (`f`).
- `initial_inference` implements `h` then `f`.
- `recurrent_inference` implements `g` then `f`.
- Hidden state normalization uses per-sample min-max scaling after representation and dynamics transitions.

## Search tree, min-max stats, and PUCT

- `Node`: stores `visit_count`, `value_sum`, `prior`, `reward`, `children`, `hidden_state`.
- `MinMaxStats`: tracks running min/max and normalizes Q-values during selection.
- `ucb_score`:
  - prior bonus: `pb_c * prior`,
  - value term: normalized `q = reward + discount * child.value()` when child has visits.
- `select_child`: greedy argmax over UCB score.
- `run_mctx`:
  - root expansion from `softmax(policy_logits)`,
  - Dirichlet root noise (`dirichlet_alpha`, `dirichlet_eps`),
  - repeated selection -> recurrent expansion -> backup for `num_simulations`,
  - target policy from root child visit counts.

## Replay sampling, targets, and masks

- `GameHistory` stores per-step:
  - `observations`, `actions`, `rewards`, `target_policies`, `target_values`.
- `ReplayBuffer.sample` returns tensors for:
  - observations, actions, rewards, target policies, target values,
  - `reward_mask` and `policy_mask` to exclude padded tail positions from losses.
- Padding behavior beyond game end:
  - random actions, zero rewards, uniform policy, zero value,
  - masks set to `0.0` so padded targets do not contribute to loss.
- `make_value_target` computes n-step bootstrapped value target with `td_steps` and `discount`, bootstrapping from stored search root values.

## Unroll training losses

- Initial step (`k=0`, from `initial_inference`):
  - value MSE vs `val_batch[:, 0]` (masked by `pol_mask[:, 0]`),
  - policy cross-entropy vs `pol_batch[:, 0]` (masked).
- Recurrent steps (`k in [0, unroll_steps-1]`, from `recurrent_inference`):
  - reward MSE vs `rew_batch[:, k]` (masked by `rew_mask[:, k]`),
  - value MSE vs `val_batch[:, k+1]` (masked by `pol_mask[:, k+1]`),
  - policy cross-entropy vs `pol_batch[:, k+1]` (masked).
- Recurrent losses are scaled by `1 / unroll_steps`.
- Gradient stabilization includes hidden-state gradient scaling hook (`* 0.5`) and global grad clipping (`max_norm=5.0`).

## Simplified vs canonical MuZero

Compared with the original MuZero (board-game/Atari setups), this file is a compact educational variant:

- **Observation/modeling**
  - Uses low-dimensional CartPole vectors (`Linear` MLP), not image stacks with deep residual towers.
- **Support encoding**
  - Predicts scalar reward/value directly (MSE), not categorical support transform with cross-entropy.
- **Search scope**
  - Single-process, per-step search with fixed simulation count; no large-scale distributed self-play workers.
- **Replay/targets**
  - Uses stored root values plus n-step bootstrap; no reanalysis with a separate target network.
- **Environment setting**
  - Single-agent episodic Gym task (`CartPole-v1`) rather than board games or Atari benchmark protocol.
- **Action space**
  - Discrete one-hot action encoding in dynamics; no continuous-control or action-embedding variants.

Even with these simplifications, the structure still follows MuZero’s core loop: learned latent model + MCTS-generated policy targets + unrolled multi-head training.

## Saved checkpoint

`muzero_network.pth` stores both `model_state_dict` and the training search config. `play_model.py` remains backward-compatible with older raw `state_dict` checkpoints, but new runs preserve the MCTS settings used during training.
