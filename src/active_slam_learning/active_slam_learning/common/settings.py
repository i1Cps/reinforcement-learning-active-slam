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

MAX_CONTINUOUS_ACTIONS = np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])
# number of scans plus robot x and y coordinate predicted by robots local SLAM
ENVIRONMENT_OBSERVATION_SPACE = NUMBER_OF_SCANS + 2

#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# DQN SETTINGS:

# DDPG SETTINGS:
ALPHA = 0.001
BETA = 0.001
TAU = 0.001
BATCH_SIZE = 64
FC1_DIMS = 300
FC2_DIMS = 300
MAX_SIZE = 1000000  # Adjust according to your system, I have 32GB RAM

# TD3 SETTINGS:

# PPO SETTINGS:
