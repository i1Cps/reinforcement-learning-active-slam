# UNDER CONSTRUCTION (Research Project) 

## Rough working

### Current progress

Robots can be seen fully exploring environment in search for a goal after 4 hours of training (400 episodes) using naive ddpg

At around 8 hours (700 episodes) robot learns to sequentially find goals (Everytime robot finds a goal, number of steps is reset) Allowing it to chain big score increases per episode.

At around 11 hours (900-1000) robot policy divereges to a local minimum which no longer chains together goal finding. ( Most likely reaching the limit of what ddpg can do, state overestimation bias which is solved in td3)

### Next steps:

Add TD3 + PPO

Multi-Robot

### Bonus steps:

Explore CNN with 2D slam map (potentially use local + global feature maps suggested by Erik)

MADDPG/MAPPO usually has centralised decentralised learning, can we introduce decentralised centralised decentralised learning by each agent interacting with the environment independently in parallel with other agents

### Considerations:

More steps per episode
