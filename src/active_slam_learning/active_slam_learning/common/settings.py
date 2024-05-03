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


# Scans + Pose_x + Pose_y + D_Optimality
ENVIRONMENT_OBSERVATION_SPACE = NUMBER_OF_SCANS + 3


#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# DDPG SETTINGS:
ALPHA = 0.0001
BETA = 0.003
TAU = 0.005
BATCH_SIZE = 100
FC1_DIMS = 512
FC2_DIMS = 512
MAX_SIZE = 1000000  # Adjust according to your system, I have 32GB RAM
RANDOM_STEPS = 50000
TRAINING_EPISODES = 2000

# PPO SETTINGS:
