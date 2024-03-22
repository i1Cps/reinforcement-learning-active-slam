# UNDER CONSTRUCTION (Research Project) 

## Rough working

### ERROR:

I introduced a goal for the robot to find whilst avoiding collision,

this introduced the /spawn_entity and /delete_entity which are not compatible with /reset_simulation, and /reset_world and ROS packages dont go together.

I'm going to refer to my project from last year using /set_entity_state. StandBY

---- Below is currently invalid as this iteration dosnt work

### Current progress
Robots gains collision avoidance + exploration after around 2 hours of training (200 episodes) using ddpg

While (from visual observation) the agent learns a satisfactory exploration strategy using the reward function , the displayed rewards per episode dont reflect this

### Next steps:

Spawn goal entity so that robot gains positive rewards, and scale reward for appropriate intuitive understanding of agents learning progress.

Add TD3 + PPO

Multi-Robot

### Bonus steps:

Explore CNN with 2D slam map (potentially use local + global feature maps suggested by Erik)

MADDPG/MAPPO usually has centralised decentralised learning, can we introduce decentralised centralised decentralised learning by each agent interacting with the environment independently in parallel with other agents

### Considerations:

More steps per episode
