# UNDER CONSTRUCTION (Research Project) 

## Rough working

### Current progress

### TD3, works (Unfortunately it looks as if it needs more time to train maybe 3 days.)

TD3 has been implemented, starts exploring the map around 900 episodes, Can be seen chaining goal together at 1500 episodes (40 hours of training)
The plot within the TD3 folder shows this

I added Weight initialisation plus deeper concatenation of actions to critic network

### DDPG, Proably broken from TD3 changes, Looking to implement TD3 paper version of DDPG

Robots can be seen fully exploring environment in search for a goal after 4 hours of training (400 episodes) using naive ddpg

At around 8 hours (700 episodes) robot learns to sequentially find goals (Everytime robot finds a goal, number of steps is reset) Allowing it to chain big score increases per episode.

At around 11 hours (900-1000) robot policy divereges to a local minimum which no longer chains together goal finding. ( Most likely reaching the limit of what ddpg can do, state overestimation bias which is solved in td3)

### Next steps:

Prioritized Experience Replay,
Better Gazebo worlds,
PPO,
Multi-Robot

### Bonus steps:

Explore CNN with 2D slam map (potentially use local + global feature maps suggested by Erik)

MADDPG/MAPPO usually has centralised decentralised learning, can we introduce decentralised centralised decentralised learning by each agent interacting with the environment independently in parallel with other agents

### Considerations:

