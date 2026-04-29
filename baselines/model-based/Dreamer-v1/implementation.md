# Dreamer V1 Implementation Notes (`dreamer_v1.py` + `dreamer_common.py`)

This implementation follows Dreamer V1 with a **continuous RSSM** and pixel observations resized to `64x64`.

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
  - value head (`MLPHead`) for critic,
  - actor head (`MLPHead`) outputting mean/std parameters for a Normal policy.

Actor distribution details:
- mean is squashed with `tanh`,
- std uses `softplus(std_logits + 0.54) + 0.1`,
- sampled actions are clamped to `[-1, 1]`.

## Data + Replay

- Replay stores `(obs, action, reward, done)` and samples sequences:
  - observations: `seq_len + 1`,
  - actions/rewards/dones: `seq_len`.
- Prefill uses random actions before gradient updates.
- For dm_control-style dict observations, pixels come from `env.render()` through `to_pixel_observation`.

## World Model Training in Code

For each sampled sequence:

1. Encode each frame.
2. RSSM `observe_step` computes posterior/prior each step.
3. Episode boundaries in sequence chunks are handled by `mask_state(state, done_{t-1})`.
4. Compute losses:
   - `recon_loss`: MSE between `sigmoid(decoder(feat_t))` and next-frame target,
   - `reward_loss`: MSE reward head vs replay rewards,
   - `kl_loss`: per-step `KL(post || prior)` via `rssm.kl`, then `free_nats_loss`.
5. `world_loss = recon_loss + reward_loss + kl_scale * kl_loss`.
6. Optimize encoder + RSSM + decoder + reward head with Adam, grad clip `100`.

## Imagined Behavior Learning

`imagine_behavior(...)` does latent rollouts from detached posterior starts:

- loop horizon steps:
  - sample action from actor at current latent feature,
  - transition with `rssm.imagine_step` (prior-only dynamics),
  - predict imagined reward/value.
- use constant discount `gamma`,
- bootstrap final value,
- compute lambda-returns with shared `lambda_return`.

Then:
- **actor loss**: `-(imag_target.mean() + 1e-3 * entropy.mean())`,
- **value loss**: MSE of critic predictions vs imagined targets (detached),
- both optimized with Adam and grad clip `100`.

## Evaluation Path

`evaluate_and_record` runs a real episode with:

- online RSSM filtering (`observe_step`) using current frame embedding and previous action,
- deterministic action selection (`dist.mean`, then clamp),
- frame capture via `render_mode="rgb_array"` and GIF export.

This keeps latent state grounded by real observations during eval, while training behavior itself still comes from imagination.
