# Canonical Actor-Critic (High Level)

Actor-critic combines two ideas:
- **Actor (policy)** learns what action to take.
- **Critic (value)** learns how good a state is, then provides a lower-variance learning signal to the actor.

This keeps policy gradients from being as noisy as pure REINFORCE, while staying directly policy-based.

## Core objects

- Policy: `pi_theta(a|s)` (discrete) or a parameterized density over actions (continuous).
- Value: `V_w(s)` approximates expected discounted return from state `s`.
- Return: `G_t = sum_{k>=0} gamma^k r_{t+k}`.
- Advantage (canonical baseline form): `A_t = G_t - V_w(s_t)` (or bootstrapped variants like TD-error).

## Canonical update equations

Objective:
- `J(theta) = E[log pi_theta(a_t|s_t) * A_t]`

Actor gradient step:
- `theta <- theta + alpha_pi * grad_theta log pi_theta(a_t|s_t) * A_t`

Critic regression step:
- minimize `L_V(w) = (target_t - V_w(s_t))^2`
- with target often `G_t` (Monte Carlo) or `r_t + gamma * V_w(s_{t+1})` (TD)

Interpretation:
- If `A_t > 0`, increase probability of sampled action.
- If `A_t < 0`, decrease it.

## Discrete vs continuous action variants

### Discrete actions
- Actor outputs logits/probabilities over finite actions.
- Policy is categorical: `a_t ~ Categorical(pi_theta(.|s_t))`.
- `log pi_theta(a_t|s_t)` comes directly from chosen class probability.

### Continuous actions
- Actor outputs distribution parameters, commonly Gaussian `N(mu_theta(s), sigma_theta(s))`.
- Action sampled from that density, often clipped/squashed into environment bounds.
- Policy term uses continuous log-density: `log pi_theta(a_t|s_t)`.

Same actor-critic principle applies in both: policy gradient weighted by advantage, plus value-function learning.
