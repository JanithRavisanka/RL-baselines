# Dreamer V1 Implementation Notes (`dreamer_v1.py` + `dreamer_common.py`)

This compact implementation follows the Dreamer V1 structure with a **continuous RSSM** and pixel observations resized to `64x64`.

## Module Split

- `dreamer_v1.py`
  - training/eval loop, replay buffer, preprocessing, logging, plotting, GIF recording.
- `dreamer_common.py`
  - shared model parts and math utilities: `ConvEncoder`, `ConvDecoder`, `ContinuousRSSM`, `MLPHead`, `lambda_return`, `free_nats_loss`, `DreamerConfig`.

## Model Components Used Here

- `ConvEncoder(3, depth=32)` -> frame embedding.
- `ContinuousRSSM(action_dim, deter=200, stoch=30, hidden=200, embed_dim=enc_dim)`.
- Feature used by heads is `feat = concat(deter, stoch)` (size `230`).
- Heads:
  - decoder (`ConvDecoder`) for reconstruction,
  - reward head (`MLPHead`) for reward prediction,
  - continuation head (`MLPHead`) for predicted nonterminal discount,
  - value head (`MLPHead`) for critic,
  - actor head (`MLPHead`) outputting mean/std parameters for a squashed Normal policy.

Actor distribution details:
- actions are sampled with a tanh-squashed Normal wrapper,
- std uses `softplus(std_logits + 0.54) + 0.1`,
- deterministic evaluation uses the squashed mean.

## Data + Replay

- Replay stores aligned transitions `(obs, action, reward, done, next_obs)`.
- Sequence sampling returns current observations and true next observations:
  - `obs` and `next_obs`: `seq_len`,
  - actions/rewards/dones: `seq_len`.
- Sampling rejects chunks that cross an episode boundary before the final transition, so RSSM state does not leak across unrelated episodes.
- Prefill uses random actions before gradient updates, then training periodically collects fresh policy experience with the current actor.
- For dm_control-style dict observations, pixels come from `env.render()` through `to_pixel_observation`.

## World Model Training in Code

For each sampled sequence:

1. Infer the initial posterior from `obs[:, 0]` with a zero previous action.
2. For each transition, apply `action_t` and condition the posterior on the true `next_obs_t`.
3. Compute losses:
   - `recon_loss`: MSE between `sigmoid(decoder(feat_{t+1}))` and `next_obs_t`,
   - `reward_loss`: MSE reward head vs replay rewards,
   - `cont_loss`: BCE continuation head vs `1 - done`,
   - `kl_loss`: per-step `KL(post || prior)` via `rssm.kl`, then `free_nats_loss`.
4. `world_loss = recon_loss + reward_loss + cont_loss + kl_scale * kl_loss`.
5. Optimize encoder + RSSM + decoder + reward/continuation heads with Adam, grad clip `100`.

## Imagined Behavior Learning

`sample_state_batch(...)` picks nonterminal posterior states from across the replay sequence, then `imagine_behavior(...)` does latent rollouts from those detached posterior starts:

- loop horizon steps:
  - sample action from actor at current latent feature,
  - transition with `rssm.imagine_step` (prior-only dynamics),
  - predict imagined reward/value.
- use predicted continuation multiplied by `gamma`,
- bootstrap final value,
- compute lambda-returns with shared `lambda_return`.

Then:
- **actor loss**: `-(imag_target.mean() + 1e-3 * entropy.mean())`,
- **value loss**: MSE of critic predictions vs imagined targets (detached),
- both optimized with Adam and grad clip `100`.

## Evaluation Path

`evaluate_and_record` runs a real episode with:

- online RSSM filtering (`observe_step`) using current frame embedding and previous action,
- deterministic action selection (`dist.mean`),
- frame capture via `render_mode="rgb_array"` and GIF export.

This keeps latent state grounded by real observations during eval, while training behavior itself still comes from imagination.
