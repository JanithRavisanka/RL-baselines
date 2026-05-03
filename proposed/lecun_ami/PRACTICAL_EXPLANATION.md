# LeCun-AMI Practical Explanation

This document explains the proposed LeCun-AMI prototype in practical terms. It
focuses on the Atari/Breakout version because that is the main FYP experiment,
but the same ideas also apply to the simpler Pendulum version.

The short version:

```text
The agent learns a small internal world.
It uses that world to imagine short futures.
It chooses actions that lead to low predicted cost.
It trains a fast actor to copy the expensive planner.
It uses a configurator to decide when planning is worth the compute.
```

This is not a full implementation of Yann LeCun's complete autonomous machine
intelligence proposal. It is a bounded reinforcement-learning prototype that
maps the proposal into modules we can actually train and evaluate.

## 1. The Basic Idea

A normal DQN agent looks at a game screen and directly predicts action values:

```text
screen -> Q-values -> action
```

Your LeCun-AMI agent does something more layered:

```text
screen -> latent state
latent state + possible action -> imagined next latent state
imagined next latent state -> predicted cost
many imagined futures -> choose action
planner decisions -> train fast actor
```

So the AMI agent does not only ask:

```text
Which action has the highest learned value right now?
```

It asks:

```text
If I take this action, then this next action, then this next action,
what future do I predict, and how costly does it look?
```

That is the practical difference. DQN reacts from learned values. AMI tries to
use a learned world model to look ahead.

## 2. One Breakout Step, Intuitively

Imagine the game is at one frame in Breakout. The paddle is near the bottom, the
ball is moving downward, and the agent must choose an action.

The AMI action-selection path is:

1. The current stacked frames go into the CNN encoder.
2. The encoder compresses the pixels into a latent state.
3. The configurator decides whether to use the cheap actor or the slower
   planner.
4. If it uses the actor, the actor gives action logits immediately.
5. If it uses the planner, the planner samples many short action sequences.
6. The world model imagines what each sequence might do in latent space.
7. The cost model and critic score those imagined futures.
8. The agent executes the first action from the best sequence.
9. If the planner was used, that planner action is stored as a training target
   for the actor.

So a planner step is like a tiny internal rehearsal:

```text
What if I go left, left, fire, right?
What if I stay, left, left, left?
What if I follow the actor's first guess but mutate one action?
```

The agent does not render predicted images. It imagines future **latent states**,
which is much cheaper than predicting pixels.

## 3. Main Modules

The Atari implementation is in `lecun_ami_atari.py`.

### Perception Encoder

The encoder is the visual front-end.

Input:

```text
4 x 84 x 84 stacked grayscale frames
```

Output:

```text
latent vector z_t
```

The encoder answers:

```text
What is the important compact representation of this screen?
```

In Breakout, the raw screen has thousands of pixels. The agent does not want to
plan directly over pixels. It wants a smaller representation that captures the
useful game state: paddle position, ball position, ball movement, brick layout,
and other features the network learns.

Code: `AtariEncoder` in `lecun_ami_atari.py`.

### JEPA-Style World Model

The world model predicts the next latent state.

Input:

```text
current latent z_t
action a_t
```

Output:

```text
predicted next latent z_{t+1}
```

This is JEPA-style because the target is not the next raw image. The target is
the next encoded representation:

```text
target_encoder(next_screen) -> target latent
```

The learning objective is roughly:

```text
make predicted_next_latent close to target_next_latent
```

This matters because predicting pixels is hard and expensive. For planning, the
agent mostly needs a useful future representation, not a perfect picture of the
future screen.

Practical intuition:

```text
The model does not need to draw the next Breakout frame.
It only needs to predict enough about the future state for action choice.
```

The Atari version uses an ensemble of world models. If the ensemble members
predict different futures, the system treats that as uncertainty.

### Cost Model

The cost model predicts immediate cost:

```text
cost = -reward
```

If reward is good, cost is low. If reward is bad or missing, cost may be higher.

The cost model answers:

```text
How bad does this immediate transition look?
```

In Breakout, rewards are sparse. The agent only gets reward when bricks are hit,
so the cost model alone is not enough. That is why the critic is also needed.

### Critic

The critic predicts longer-term cost-to-go:

```text
critic(z_t) -> expected future cost
```

The planner only imagines a short horizon. For example, it may test 12 future
actions, but the episode lasts much longer than 12 actions. The critic estimates
what lies beyond the planning horizon.

Planner scoring is roughly:

```text
total score =
    predicted immediate costs
  + uncertainty penalty
  + terminal critic cost
```

Lower score is better.

Practical intuition:

```text
The cost model judges the next few imagined steps.
The critic says whether the final imagined state still looks promising.
```

### Actor

The actor is the fast policy.

Input:

```text
latent state z_t
```

Output:

```text
action logits
```

The actor is cheap. It can choose an action quickly without simulating many
future action sequences.

But in this implementation, the actor is not the main source of deliberate
decision-making. It learns from planner demonstrations:

```text
planner selected action -> supervised target for actor
```

So the actor gradually learns:

```text
When the screen looks like this, the planner usually chooses this action.
```

This is important because planning every step is expensive. The actor is how
the system amortizes planning. Planning creates examples, and the actor learns
to imitate them.

### Configurator

The configurator decides whether to plan.

It answers:

```text
Should I spend compute on planning right now,
or is the fast actor good enough?
```

In your code, the configurator is implemented by `should_plan()`.

It can plan because:

- `--planning-mode always` forces planning.
- early warmup forces planning.
- `--planning-interval` forces periodic planning.
- model uncertainty is above `--uncertainty-threshold`.

It can avoid planning because:

- `--planning-mode actor` disables planning.
- uncertainty is low.
- no periodic planning step is due.

This is the core research idea in your experiments. The agent should not always
pay the full planning cost. It should plan when planning is useful.

## 4. How Planning Works

The Atari planner uses random shooting over discrete action sequences.

Suppose:

```text
num_sequences = 512
horizon = 12
```

Then at one planning step, the planner samples 512 possible action sequences,
each 12 actions long.

Example:

```text
sequence 1: left, left, stay, right, ...
sequence 2: stay, stay, left, left, ...
sequence 3: right, fire, right, stay, ...
```

For each sequence:

1. Start with current latent state.
2. Apply action 1 through the world model.
3. Predict cost.
4. Apply action 2 through the world model.
5. Predict cost.
6. Continue until the horizon ends.
7. Add critic value for the final latent.
8. Add uncertainty penalty.

Then choose the sequence with the lowest predicted total cost.

Only the first action is executed:

```text
best sequence = [left, left, stay, right, ...]
execute left
```

After the real environment responds, the agent gets a new observation and can
plan again later. This is model-predictive control.

## 5. Why Use Actor-Seeded Plans?

Not all sampled sequences are purely random. Some are sampled near the actor's
current policy and then mutated.

This helps because pure random shooting wastes many candidates on obviously bad
sequences. The actor provides a useful starting bias:

```text
The actor thinks these actions are plausible.
The planner tests variations around them.
```

This makes planning more focused without becoming fully greedy.

## 6. How Training Works

Training has three broad phases.

### Phase A: Seed Replay

Before the world model can predict anything useful, the agent needs real
transitions.

So it first collects random environment steps:

```text
state, action, reward, next_state, done
```

These go into replay.

### Phase B: Initial Model Updates

The agent trains the encoder, world model, cost model, and critic on the seed
data before serious planning starts.

This avoids asking a completely untrained world model to guide planning.

### Phase C: Online Training

During the main loop:

1. The agent observes the current state.
2. Epsilon is computed for exploration.
3. The configurator chooses actor or planner.
4. The selected action is executed in the real environment.
5. The transition is stored in replay.
6. If the planner chose the action, the replay item stores `planner_action`.
7. The predictive modules update from replay.
8. The actor updates from planner demonstrations.

The replay buffer therefore serves two purposes:

```text
ordinary transitions -> train world/cost/critic
planner-labeled transitions -> train actor imitation
```

## 7. The Losses In Plain Language

The model update combines several losses.

### World Loss

```text
Did the world model predict the next latent correctly?
```

This trains the JEPA-style latent dynamics.

### Cost Loss

```text
Did the cost model predict -reward correctly?
```

This teaches immediate consequences.

### Critic Loss

```text
Did the critic predict future cost correctly?
```

For Atari, the critic uses n-step targets so sparse rewards can affect more
than a single transition.

### Representation Variance Loss

This discourages the latent representation from collapsing into an uninformative
constant vector.

Practical intuition:

```text
If every screen maps to almost the same latent vector,
the world model cannot learn useful futures.
```

### Actor Loss

```text
Can the actor copy planner-selected actions?
```

For Atari, this is cross-entropy over discrete planner actions.

## 8. Why The Ensemble Matters

The world model ensemble gives uncertainty.

If all models predict similar next latents:

```text
low uncertainty
```

If they disagree:

```text
high uncertainty
```

The configurator uses this uncertainty to decide whether to plan. The planner
also adds uncertainty as a penalty when scoring candidate futures.

Practical intuition:

```text
If the internal world model is unsure what happens next,
the agent should be more careful.
```

In your experiment, this is controlled by:

```bash
--uncertainty-threshold
--uncertainty-cost
```

## 9. Planning Modes

Your experiments compare three AMI modes.

### Adaptive

```bash
--planning-mode adaptive
```

The configurator decides when to plan. This is the proposed mode.

Expected behavior:

```text
Plan sometimes.
Use actor sometimes.
Trade reward against compute cost.
```

### Always Planner

```bash
--planning-mode always
```

The agent plans at every step.

Expected behavior:

```text
Potentially better decisions.
Much higher compute cost.
```

This is the expensive upper reference.

### Actor Only

```bash
--planning-mode actor
```

The agent uses only the fast actor.

Expected behavior:

```text
Fastest action selection.
No deliberate lookahead.
Depends heavily on whether actor learned good planner behavior.
```

This tests whether planning is actually helping.

## 10. Why This Is A Good FYP Experiment

The main question is not:

```text
Does this beat state-of-the-art Atari agents?
```

The better question is:

```text
Does adaptive planning improve the reward/compute tradeoff?
```

That is why you track:

- reward
- evaluation reward over time
- planning rate
- wall-clock time
- reward per planning rate
- reward per wall-clock hour
- model uncertainty
- actor loss
- critic loss
- world loss

If adaptive planning gets close to always-planner reward with much lower
planning rate, that supports the research claim.

If actor-only performs similarly, then planning may not be adding much.

If always-planner is much better but too slow, then the architecture works but
the configurator may need improvement.

## 11. Difference From The Pendulum Version

The Pendulum prototype uses the same conceptual modules:

```text
encoder
latent world model
cost model
critic
actor
configurator
planner
```

But the details differ:

| Part | Pendulum | Atari |
|---|---|---|
| Observation | Low-dimensional state vector | Stacked image frames |
| Encoder | MLP | CNN |
| Action | Continuous torque | Discrete joystick action |
| Planner | CEM over continuous action sequences | Random shooting over discrete action sequences |
| Actor loss | MSE against planner action | Cross-entropy against planner action |
| Main comparison | DDPG/TD3/SAC/MPC | DQN/DDQN/PER/Dreamer |

The Atari version is harder because it must learn from pixels and sparse
rewards.

## 12. What To Say In The Dissertation

A clean dissertation description could be:

```text
The proposed LeCun-AMI prototype implements a modular latent model-based RL
agent. A perception encoder maps observations to compact latent states. A
JEPA-style dynamics model predicts future latent states without reconstructing
pixels. A cost model and critic estimate immediate and long-term costs. A
configurator decides whether to act with a fast learned actor or invoke
cost-guided latent planning. Planner-selected actions are stored as
demonstrations and used to train the actor, amortizing planning into reactive
control. The Atari extension uses a CNN encoder, discrete latent dynamics, an
ensemble-based uncertainty estimate, and sampled discrete action-sequence
planning.
```

Then the key experimental claim:

```text
The central hypothesis is that adaptive configurator-controlled planning can
improve the reward/compute tradeoff compared with actor-only behavior and
always-on planning.
```

## 13. Limitations

Important limitations:

- This is not a complete AMI or AGI system.
- The world model predicts latent states, not exact future pixels.
- Planning is expensive on Atari.
- The actor only learns good behavior if planner demonstrations are useful.
- The cost model is currently tied to environment reward via `cost = -reward`.
- The configurator is heuristic, not a learned high-level controller.
- The method is research-prototype scale, not Atari SOTA scale.

These limitations are not failures. They define the scope of the FYP prototype.

## 14. Mental Model

The easiest way to remember the architecture is:

```text
Encoder:      What situation am I in?
World model:  What might happen next?
Cost model:   Is that immediately good or bad?
Critic:       Does the future after that look good or bad?
Planner:      Which short action sequence looks best?
Actor:        What action can I choose quickly?
Configurator: Is this a moment worth planning?
Replay:       What real experience and planner decisions can I learn from?
```

The whole system is built around one practical idea:

```text
Use planning when it is worth it.
Learn from planning so you do not always need it.
```
