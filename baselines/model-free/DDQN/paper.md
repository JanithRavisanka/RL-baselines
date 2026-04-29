# Double DQN: Principle vs DQN

Deep Q-Network (DQN) uses the same values to both **choose** and **evaluate** the next action in its target:

\[
y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
\]

This can create optimistic bias, because the max operator tends to pick overestimated actions.

Double DQN keeps the same overall architecture (online network + target network + replay), but changes the target to decouple roles:

1. **Selection (argmax)** with the online network  
   \[
   a^* = \arg\max_{a'} Q_{\text{online}}(s', a')
   \]
2. **Evaluation** of that selected action with the target network  
   \[
   y = r + \gamma Q_{\text{target}}(s', a^*)
   \]

So DQN does "max over target values", while Double DQN does "argmax by online, value from target".  
This simple decoupling reduces overestimation and usually improves training stability and policy quality.
