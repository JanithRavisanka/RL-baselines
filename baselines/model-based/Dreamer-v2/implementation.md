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
- Replay stores aligned transitions `(obs, action, reward, done, next_obs)`.
- `ReplayBuffer.sample(batch, seq)` returns contiguous chunks:
  - `obs: [B, seq, C, H, W]`
  - `next_obs: [B, seq, C, H, W]`
  - `act/rew/done: [B, seq]`
- `next_obs_t` is used both as the posterior observation for transition `t` and as the reconstruction target.
- This is critical for recurrent latent dynamics; training is not per-transition IID.

## Episode-boundary masking
- Sequence sampling rejects chunks that cross episode boundaries before the final transition.
- Effect: RSSM carry-over remains within a real episode; terminal next observations can still be learned at the final transition.

## KL balancing in this implementation
- For each step, the code computes:
  - posterior/prior from `DiscreteRSSM.observe_step(...)`,
  - balanced KL via `DiscreteRSSM.kl_balanced(post, prior, alpha=0.8)`.
- It then stacks KL across time and applies `free_nats_loss`.
- Final world-model objective:
  - `world_loss = recon_loss + reward_loss + cont_loss + kl_scale * kl_loss`.
- The continuation head is trained with BCE against `1 - done` and provides
  imagined discounts as `sigmoid(cont) * gamma`.

## Actor/value imagined updates
- Behavior learning samples nonterminal posterior states from across the replay unroll.
- `imagine_behavior(...)`:
  - samples policy actions from categorical logits,
  - feeds sampled one-hot actions to the RSSM prior,
  - rolls `rssm.imagine_step(...)` for `horizon`,
  - predicts imagined reward/value and computes lambda-return targets.
- Actor update:
  - REINFORCE-style objective with imagined advantage and entropy bonus, matching the Atari `rho=1` setting.
- Value update:
  - MSE regression to imagined lambda-return targets (features detached for critic update path).

## Data collection

After random replay prefill, training periodically rolls out the current actor and appends fresh policy transitions to replay. This keeps the world model and actor training distribution aligned with the improving policy instead of training only on random data.

## End-to-end outputs from this script
- Trains world model + actor + value.
- Saves model weights, loss plot, and evaluation GIF under:
  - `results/dreamer_v2/run_<timestamp>/`.
