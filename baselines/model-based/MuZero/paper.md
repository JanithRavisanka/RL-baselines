# MuZero (Paper-Oriented Overview)

MuZero learns a model that is only as detailed as needed for planning. Instead of reconstructing observations, it learns latent dynamics that support accurate policy/value/reward prediction under tree search.

## Core architecture

MuZero uses three learned functions:

- **Representation** `h(o_t) -> s_t^0`: converts an observation into a latent state used for planning.
- **Dynamics** `g(s_t^k, a_t^k) -> (r_t^{k+1}, s_t^{k+1})`: rolls the latent state forward under an action and predicts immediate reward.
- **Prediction** `f(s_t^k) -> (p_t^k, v_t^k)`: outputs policy logits and value from each latent state.

Planning alternates these functions: start from `h`, then repeatedly apply `g` and `f` while traversing the tree.

## MCTS in MuZero

At each real environment step, MuZero runs MCTS rooted at the current observation:

1. **Root expansion** from policy priors `p` produced by `f(h(o_t))`.
2. **Selection** with a PUCT-style score combining:
   - prior-driven exploration, and
   - normalized value estimates (`Q`) for exploitation.
3. **Expansion/evaluation** by one recurrent model step (`g` then `f`) at the leaf.
4. **Backup** of value estimates along the visited path.
5. **Improved policy target** from child visit-count distribution.

Action selection for self-play is sampled from visit counts (with temperature), while evaluation typically uses argmax.

## Training targets

For each unrolled step `k`, MuZero trains on three targets:

- **Policy target**: MCTS visit-count distribution at that position.
- **Value target**: bootstrapped return (discounted rewards plus `n`-step bootstrap value).
- **Reward target**: observed environment reward for recurrent steps.

Training unrolls `K` dynamics steps from sampled starting points and applies losses at the initial and recurrent steps (value + policy; plus reward for recurrent steps). This couples search supervision (policy target) with return prediction (value/reward) in the learned latent model.
