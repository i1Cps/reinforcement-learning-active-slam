import numpy as np

# THIS FILE IS VERY MESSY, SORRY FOR NOW

#########################################################
#                   ROBOT SETTINGS                      #
#########################################################

MAX_LINEAR_SPEED = 0.22
MAX_ANGULAR_SPEED = 2.0
NUMBER_OF_SCANS = 90


#########################################################
#                   ENVIRONMENT SETTINGS                #
#########################################################

INITIAL_POSE = np.array([-2.0, -0.5])
GOAL_PAD_RADIUS = 0.7
MAX_CONTINUOUS_ACTIONS = np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])


# Scans + Pose_x + Pose_y + D_Optimality
ENVIRONMENT_OBSERVATION_SPACE = NUMBER_OF_SCANS + 3
ENVIRONMENT_ACTION_SPACE = 2  # Linear and Angular Speed

#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# GLOBAL SETTINGS
MAX_STEPS = 2_000_000
MAX_MEMORY_SIZE = 1_000_000  # Adjust according to your system, I have 32GB RAM
FRAME_BUFFER_DEPTH = 3
FRAME_BUFFER_SKIP = 10
RANDOM_STEPS = 5000
TRAINING_EPISODES = 2000

# DDPG SETTINGS:
ALPHA_DDPG = 0.0001
BETA_DDPG = 0.003
TAU = 0.005
GAMMA_DDPG = 0.99
BATCH_SIZE_DDPG = 100
ACTOR_DDPG_FC1 = 400
ACTOR_DDPG_FC2 = 512
CRITIC_DDPG_FC1 = 512
CRITIC_DDPG_FC2 = 512

# PPO SETTINGS:
ALPHA_MAAPO = 0.0001
BETA_MAPPO = 0.003
TAU = 0.005
ACTOR_PPO_FC1 = 512
ACTOR_PPO_FC2 = 512
CRITIC_PPO_FC1 = 512
CRITIC_PPO_FC2 = 512
T = 512
POLICY_CLIP = 0.2
GAMMA_PPO = 0.99
BATCH_SIZE_PPO = 64
N_EPOCHS = 15
GAE_LAMBDA = 0.95
ENTROPHY_COEFFICIENT = 0.01
