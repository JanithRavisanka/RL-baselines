# `dreamer_v2.py` Implementation Notes

This file is a Dreamer V2-style Atari trainer that imports shared model utilities from `baselines/model-based/dreamer_common.py`.

## Module split: `dreamer_v2.py` vs `dreamer_common.py`
- `dreamer_v2.py` contains:
  - replay buffer, training loop, evaluation loop, CLI args, and output saving.
- `dreamer_common.py` provides reusable building blocks:
  - `ConvEncoder`, `ConvDecoder`, `MLPHead`,
  - `DiscreteRSSM`, `DreamerConfig`,
  - `free_nats_loss`, `lambda_return`.

## Replay chunks and sequence training
- Replay stores single transitions `(obs, action, reward, done)`.
- `ReplayBuffer.sample(batch, seq)` returns contiguous chunks:
  - `obs: [B, seq+1, C, H, W]`
  - `act/rew/done: [B, seq]`
- The extra observation frame (`t+1`) is used as the reconstruction target while unrolling RSSM over `seq` steps.
- This is critical for recurrent latent dynamics; training is not per-transition IID.

## Episode-boundary masking
- `mask_state(state, done_prev)` zeroes latent state whenever the previous step was terminal.
- During world-model unroll, the code applies masking for `t>0` using `done_b[:, t-1]`.
- Effect: sampled sequences can cross episode boundaries in replay without leaking latent memory across episodes.

## KL balancing in this implementation
- For each step, the code computes:
  - posterior/prior from `DiscreteRSSM.observe_step(...)`,
  - balanced KL via `DiscreteRSSM.kl_balanced(post, prior, alpha=0.8)`.
- It then stacks KL across time and applies `free_nats_loss`.
- Final world-model objective:
  - `world_loss = recon_loss + reward_loss + kl_scale * kl_loss`.

## Actor/value imagined updates
- Behavior learning starts from the last posterior state of replay unroll.
- `imagine_behavior(...)`:
  - samples policy actions from categorical logits,
  - converts to one-hot with straight-through gradient trick,
  - rolls `rssm.imagine_step(...)` for `horizon`,
  - predicts imagined reward/value and computes lambda-return targets.
- Actor update:
  - REINFORCE-style objective with imagined advantage and entropy bonus.
- Value update:
  - MSE regression to imagined lambda-return targets (features detached for critic update path).

## End-to-end outputs from this script
- Trains world model + actor + value.
- Saves model weights, loss plot, and evaluation GIF under:
  - `results/dreamer_v2/run_<timestamp>/`.

