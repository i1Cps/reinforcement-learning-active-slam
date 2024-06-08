import numpy as np

#########################################################
#                   ROBOT SETTINGS                      #
#########################################################

MAX_LINEAR_SPEED = 0.22
MAX_ANGULAR_SPEED = 2.0
MAX_SCAN_DISTANCE = 3.5
NUMBER_OF_SCANS = 90
COLLISION_DISTANCE = 0.18
ROBOT_NAME = "robot"


#########################################################
#                   ENVIRONMENT SETTINGS                #
#########################################################

# Scans + Pose_x + Pose_y
ENVIRONMENT_OBSERVATION_SPACE = NUMBER_OF_SCANS + 2
ENVIRONMENT_ACTION_SPACE = 2  # Linear and Angular Speed
MAX_CONTINUOUS_ACTIONS = np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])
EPISODE_LENGTH_SEC = 60
EPISODE_STEPS = 1000
GOAL_PAD_RADIUS = 0.7
REWARD_DEBUG = True

#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# GLOBAL SETTINGS
LOAD_MODEL = False
MODEL_PATH = "src/active_slam_learning/models"
FRAME_BUFFER_DEPTH = 3
FRAME_BUFFER_SKIP = 10
RANDOM_STEPS = 25000
TRAINING_EPISODES = 2000
TRAINING_STEPS = 1_000_000

# DDPG SETTINGS:
ALPHA_DDPG = 0.0001
BETA_DDPG = 0.0003
TAU = 0.005
GAMMA_DDPG = 0.99
BATCH_SIZE_DDPG = 100
ACTOR_DDPG_FC1 = 400
ACTOR_DDPG_FC2 = 512
CRITIC_DDPG_FC1 = 512
CRITIC_DDPG_FC2 = 512
MAX_MEMORY_SIZE = 1_000_000  # Adjust according to your system, I have 32GB RAM

# PPO SETTINGS:
ALPHA_PPO = 0.0001
BETA_PPO = 0.0003
ACTOR_PPO_FC1 = 512
ACTOR_PPO_FC2 = 512
CRITIC_PPO_FC1 = 512
CRITIC_PPO_FC2 = 512
POLICY_CLIP = 0.2
GAMMA_PPO = 0.99
TRAJECTORY = 2048
NUM_MINI_BATCHES = 64
N_EPOCHS = 15
GAE_LAMBDA = 0.95
ENTROPY_COEFFICIENT = 0.01
