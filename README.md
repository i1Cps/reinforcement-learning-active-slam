<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️--><h1 align="center">reinforcement-learning-active-slam</h1>
<p align="center">
  <img src="media/logo2.png" alt="Logo" width="550" height="auto" />
</p>


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#table-of-contents)

## ➤ Table of Contents

* [➤ :pencil: About The Project](#-pencil-about-the-project)
	* [Packages](#packages)
		* [active_slam_simulation:](#active_slam_simulation)
		* [active_slam_learning:](#active_slam_learning)
		* [active_slam_interfaces:](#active_slam_interfaces)
		* [slam_toolbox](#slam_toolbox)
* [➤ :hammer: Usage](#-hammer-usage)
	* [Terminal Commands](#terminal-commands)
	* [Navigation in TMUX:](#navigation-in-tmux)
* [➤ :floppy_disk: Key Project File Descriptions](#-floppy_disk-key-project-file-descriptions)
	* [`Custom Environment Files`](#custom-environment-files)
		* [active_slam_simulation package:](#active_slam_simulation-package)
		* [active_slam_learning package:](#active_slam_learning-package)
		* [active_slam_interface package:](#active_slam_interface-package)
	* [`Reinforcement Learning Files`](#reinforcement-learning-files)
		* [DDPG](#ddpg)
		* [PPO](#ppo)
		* [COMMON](#common)
	* [Robot Settings ](#robot-settings-)
	* [Environment Settings](#environment-settings)
	* [Reinforcement Learning Settings ](#reinforcement-learning-settings-)
		* [Global Settings ](#global-settings-)
		* [DDPG Settings ](#ddpg-settings-)
		* [PPO Settings](#ppo-settings)
* [➤ :hammer: Basic Installation](#-hammer-basic-installation)
* [➤ :rocket: Dependencies](#-rocket-dependencies)
* [➤ :coffee: Buy me a coffee](#-coffee-buy-me-a-coffee)
* [➤ :scroll: Credits](#-scroll-credits)
* [➤ License](#-license)


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#pencil-about-the-project)

## ➤ :pencil: About The Project
--------------------------

This repository explores and implements reinforcement learning strategies for active simultaneous localization and mapping (SLAM) using a single robot. The project integrates advanced reinforcement learning algorithms, specifically [Proximal Policy Optimization](https://arxiv.org/abs/1312.5602) (PPO) and [Deep Deterministic Policy Gradients](https://arxiv.org/abs/1509.02971) (DDPG), enabling a robotic agent to autonomously explore and map unknown environments effectively.

At its core, this project seeks to bridge the gap between theoretical reinforcement learning and practical robotic applications. It focuses on creating a robust learning environment where the robot iteratively updates and adjusts its policy based on real-world dynamics. This approach applies various state-of-the-art reinforcement learning algorithms to enhance both the precision of the spatial maps generated and the efficiency of goal-based exploration in True Unknown Environments (TUE).

Designed for researchers and developers with an interest in robotics and machine learning, this project provides a deep dive into how autonomous agents can learn to adapt and navigate independently, pushing the boundaries of robotic autonomy in exploration and mapping tasks.

<p align="center">
  <img src="media/example_video.gif" alt="Logo" width="550" height="auto" />
</p>

### Packages

#### active_slam_simulation:

To simulate our robotic environment, we create a custom training environment using gazebo classic, a physics engine simulator. This package is responsible for setting up Gazebo with custom maps and our robot model. It also starts up the SLAM algorithm from the slam toolbox.

#### active_slam_learning:

To train our agent in our custom training environment we handle the logic in this package, It creates several nodes which communicate with each other as well as hosts the main reinforcement learning algorithms including the training loop

#### active_slam_interfaces:

To allow all our nodes to communicate with each other, we use this package to establish msg and srv files which establish a protocol for sending and receiving information amongst the nodes

#### slam_toolbox

The slam_toolbox is managed and distributed by [Steve Macenski](https://www.steve.macenski.com/) and its GitHub repository can be found [here](https://github.com/SteveMacenski/slam_toolbox). It allows us to localise and map the robots environment which essentially is the basis behind this research





[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#hammer-usage)

## ➤ :hammer: Usage

### Terminal Commands

I highly recommend running this with [TMUX](https://github.com/tmux/tmux/wiki). Tmux allows us to split the terminal into panes to view all the essential ROS2 processes simultaneously. `apt install tmux`

<p align="center">
  <img src="media/tmux_image.png" alt="Logo" width="550" height="auto" />
</p>

After following the [basic installation](#-hammer-basic-installation)

Start tmux with:

```
tmux
```

First source ROS2 Humble with the command:

```
source /opt/ros/humble/setup.bash

```

Change into the workspace directory:

```
cd reinforcement-learning-active-slam
```

and run the following lines to split the panes:

```
tmux split-window -v -p 30
tmux split-window -h -p 50
tmux split-window -h -p 50 -t 0
tmux split-window -v -p 50 -t 1
tmux split-window -v -p 66
tmux split-window -v -p 50
tmux select-pane -t 6
```

***If you dont have tmux, you may create 5 seperate terminal windows instead***


Launch the gazebo simulation physics engine with our robot model:

```
source install/setup.bash
tmux select-pane -t 4
ros2 launch active_slam_simulations main_world.launch.py
```

Next launch the SLAM algorithm from the slam toolbox:

```
source install/setup.bash
tmux select-pane -t 5
ros2 launch active_slam_simulations slam.launch.py
```

Next run the Gazebo Bridge node:

```
source install/setup.bash
tmux select-pane -t 1
ros2 run active_slam_learning gazebo_bridge
```
Next run the Learning Environment node:
```
source install/setup.bash
tmux select-pane -t 3
ros2 run active_slam_learning learning_environment
```

Startup RViz to see the map the robot generates per episode:
```
source install/setup.bash
tmux select-pane -t 0
ros2 launch active_slam_simulations view_map.launch.py
```


Lastly start the Learning node (**DDPG or PPO**):
```
source install/setup.bash
ros2 run active_slam_learning learning_ddpg
```

    or

```
ros2 run active_slam_learning learning_ppo
```

### Navigation in TMUX:

Please refer to this [cheatsheet](https://tmuxcheatsheet.com/) for more information but two heplful commands are:

`ctrl+b o` ~ To switch to the next pane

and 

`ctrl+b z` ~ To zoom in and out of a pane









 




[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#floppy_disk-key-project-file-descriptions)

## ➤ :floppy_disk: Key Project File Descriptions

### `Custom Environment Files`

#### active_slam_simulation package:

* [`main_world.launch.py`](src/active_slam_simulations/launch/main_world.launch.py): This file is a ROS2 launch file that currently starts up Gazebo Classic, our physics simulation engine. It also runs the robot state publisher, which handles broadcasting the robot's state and spawns the robot model in the simulation environment.

* [`slam.launch.py`](src/active_slam_simulations/launch/slam.launch.py): This file launches the **SLAM toolbox**, which we use to perform Active SLAM. This allows us to calculate map certainty and, importantly, view estimated mappings of the environment while the robot explores. The SLAM_toolbox is managed and distributed by [Steve Macenski](https://www.steve.macenski.com/) and its GitHub repository can be found [here](https://github.com/SteveMacenski/slam_toolbox).

* [`view_map.launch.py`](src/active_slam_simulations/launch/view_map.launch.py): This file launches RViz, enabling the visualisation of grid maps generated by the SLAM algorithm.

* [`models folder`](src/active_slam_simulations/models): This folder contains the model files for our simulation, including the [Turtlebot3_burger](https://www.turtlebot.com/turtlebot3/) robot and the goal pad.

* [`worlds folder`](src/active_slam_simulations/worlds): This folder contains the different custom SDF world files which took a very very very long time to create :thumbsup:

#### active_slam_learning package:

* [`learning_environment.py`](src/active_slam_learning/active_slam_learning/learning_environment/learning_environment.py): This file defines a Learning Environment node for simulating a single-robot system that learns to autonomously explore True Unknown Environments using reinforcement learning through ROS2. It manages state updates, action processing, and reward calculations necessary for RL experiments. Key components include handling robot velocities, calculating map uncertainty using D-Optimality and computing rewards. The environment interacts with the robot and simulated Gazebo environment to facilitate the training and evaluation of learning agents primarily through the custom Gazebo Bridge.

* [`gazebo_bridge.py`](src/active_slam_learning/active_slam_learning/gazebo_bridge/gazebo_bridge.py): This file implements a custom Gazebo Bridge node, which handles direct communication with Gazebo services and the training environment. It manages the spawning and movement of goal and robot models, resets the SLAM algorithm provided by the [**SLAM toolbox**](https://github.com/SteveMacenski/slam_toolbox) and provides services for managing simulation states such as pausing and unpausing the physics engine.

* [`reward_function.py`](src/active_slam_learning/active_slam_learning/learning_environment/reward_function.py): This file implements the reward function for the reinforcement learning (RL) agent:

     
    * **Initial Reward**: The reward calculation's starting point is -0.4. 
    * **Linear Velocity Penalty**: Encourages the robot to maintain a higher linear velocity, calculated as -3 times the difference between the maximum speed and the current linear velocity. 
        
        ~ This ranges from -1.2 to 0 for linear velocities between -0.2 and 0.2. 
        
    * **Angular Velocity Penalty**: Penalises higher angular velocities, calculated as -0.2 times the square of the angular velocity. 
    
        ~ This ranges from -0.968 to 0 for angular velocities between -2.2 and 2.2. 
    
    * **Collision Penalty**: Imposes a significant penalty of -1000 if the robot collides with an obstacle. 
    
    * **Goal Reward**: Rewards the robot with 1000 if it successfully finds the goal.  
    * **Map Uncertainty Reward**: Provides a positive intrinsic reward based on the map uncertainty (D-Optimality). If D-Optimality is not provided, the reward is 0. Otherwise, it is calculated as the hyperbolic tangent of 0.01 divided by the D-Optimality value. 

      ~ This ranges from 0 to 1. 

#### active_slam_interface package:

* [`srv`](src/active_slam_interfaces/srv): This folder contains service types used by individual Nodes to communicate with each other, an essential foundation of the ROS framework.

* [`msg`](src/active_slam_interfaces/msg): This folder contains msg types used by individual Nodes to communicate with each other, an essential foundation of the ROS framework.


### `Reinforcement Learning Files`

#### DDPG

* [`learning_ddpg.py`](src/active_slam_learning/active_slam_learning/learning/learning_ddpg.py): Facilitates the main training loop of the [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm, including Frame stacking, frame skipping, reward normalisation, initialising the model and the memory buffer, handling the scoring metrics per episode and lastly saving the training data for evaluation later.

* [`agent.py`](src/active_slam_learning/active_slam_learning/learning/ddpg/agent.py): Defines the main agent interacting with the environment. It encapsulates the logic for selecting actions, applying noise, learning and updating the policy.

* [`replay_memory.py`](src/active_slam_learning/active_slam_learning/learning/ddpg/replay_memory.py): This file implements the replay memory buffer, storing the agent's experiences during training. It allows for random sampling of experiences to stabilise training by breaking the correlation between consecutive experiences.

* [`networks.py`](src/active_slam_learning/active_slam_learning/learning/ddpg/networks.py): This file defines the neural network architectures used for the actor and critic models in the DDPG algorithm. These networks are responsible for approximating the policy and value functions.


#### PPO


* [`learning_ppo.py`](src/active_slam_learning/active_slam_learning/learning/learning_ppo.py): Facilitates the main training loop for the [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) algorithm, including frame stacking, frame skipping, reward clipping, model and memory buffer initialiasation, episode scoring metrics, and training data saving for later evaluation. It handles the overall flow of the PPO training process.

* [`agent.py`](src/active_slam_learning/active_slam_learning/learning/ppo/agent.py): Defines the main agent interacting with the environment. It encapsulates the logic for selecting actions, learning, and updating the policy using actor and critic networks. The agent implements the PPO update rule, which involves clipping probability ratios to ensure stable policy updates and maintaining entropy for exploration.

* [`replay_memory.py`](src/active_slam_learning/active_slam_learning/learning/ddpg/replay_memory.py):  Implements the PPO memory buffer, storing experiences and enabling random sampling to stabilise training. It manages states, actions, rewards, next states, done flags, and action probabilities, facilitating mini-batch updates during learning.

* [`networks.py`](src/active_slam_learning/active_slam_learning/learning/ddpg/networks.py): Defines the neural network architectures for the actor and critic models in PPO. The actor network uses Beta distributions for action sampling, ensuring a stochastic policy, while the critic network estimates state values to provide advantage estimates during training.


#### COMMON

* [`utilities.py`](src/active_slam_learning/active_slam_learning/common/utilities.py): Includes helper functions for communicating with the [learning environment node](src/active_slam_learning/active_slam_learning/learning_environment/learning_environment.py) and plotting functions.

* [`settings.py`](src/active_slam_learning/active_slam_learning/common/settings.py): Contains all the configuration settings for the training process. Users can very much use this project by only ever changing this file

The following settings and options are exposed to you:

### Robot Settings 
* `MAX_LINEAR_SPEED`: Maximum linear speed (0.22) 
* `MAX_ANGULAR_SPEED`: Maximum angular speed (2.0) 
* `MAX_SCAN_DISTANCE`: Maximum scan distance (3.5) 
* `NUMBER_OF_SCANS`: Number of scans (90) 
* `COLLISION_DISTANCE`: Collision distance (0.18) 
### Environment Settings
* `ENVIRONMENT_OBSERVATION_SPACE`: Observation space for the environment (NUMBER\_OF\_SCANS + 2) 
* `ENVIRONMENT_ACTION_SPACE`: Action space for the environment (2) 

* `EPISODE_LENGTH_SEC`: Episode length in seconds (60) 
* `EPISODE_STEPS`: Number of steps in an episode (1000) 
* `GOAL_PAD_RADIUS`: Radius of the goal pad (0.7) 
* `REWARD_DEBUG`: Debug mode for rewards (True) 

### Reinforcement Learning Settings 

#### Global Settings 
* `LOAD_MODEL`: Whether to load a pre-trained model (False) 
* `MODEL_PATH`: Path to the model ("training\_data/models/single\_robot\_exploration") 
* `TRAINING_STEPS`: Number of training steps (1\_000\_000) 
* `RANDOM_STEPS`: Number of random steps (25000) 
* `MAX_MEMORY_SIZE`: Maximum memory size (1\_000\_000) 
* `FRAME_BUFFER_DEPTH`: Frame buffer depth (3) 
* `FRAME_BUFFER_SKIP`: Frame buffer skip (10) 

* `TRAINING_EPISODES`: Number of training episodes (2000) 

#### DDPG Settings 
* `ALPHA_DDPG`: Learning rate for the actor (0.0001) 
* `BETA_DDPG`: Learning rate for the critic (0.0003) 
* `ACTOR_DDPG_FC1`: Number of units in the first fully connected layer of the actor (400) 
* `ACTOR_DDPG_FC2`: Number of units in the second fully connected layer of the actor (512) 
* `CRITIC_DDPG_FC1`: Number of units in the first fully connected layer of the critic (512) 
* `CRITIC_DDPG_FC2`: Number of units in the second fully connected layer of the critic (512) 
* `TAU`: Soft update parameter (0.005) 
* `GAMMA_DDPG`: Discount factor for future rewards (0.99)
* `BATCH_SIZE_DDPG`: Training batch size (100)


#### PPO Settings
* `ALPHA_MAAPO`: Learning rate for the actor (0.0001) 
* `BETA_MAPPO`: Learning rate for the critic (0.003)
* `ACTOR_PPO_FC1`: Number of units in the first fully connected layer of the actor (512)
* `ACTOR_PPO_FC2`: Number of units in the second fully connected layer of the actor (512)
* `CRITIC_PPO_FC1`: Number of units in the first fully connected layer of the critic (512)
* `CRITIC_PPO_FC2`: Number of units in the second fully connected layer of the critic (512)
* `POLICY_CLIP`: Clipping parameter for policy (0.2)
* `GAMMA_PPO`: Discount factor for future rewards (0.99)
* `TRAJECTORY`: Number of steps per trajectory (2048)
* `NUM_MINI_BATCHES`: Number of mini-batches for training (64)
* `N_EPOCHS`: Number of epochs per update (15)
* `GAE_LAMBDA`: Generalized Advantage Estimation lambda (0.95)
* `ENTROPY_COEFFICIENT`: Coefficient for entropy regularization (0.01)




[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#hammer-basic-installation)

## ➤ :hammer: Basic Installation

 **Install Ubuntu 22.04 OS**

**Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)**

**Download workspace**
```
git clone https://github.com/i1Cps/reinforcement-learning-active-slam.git

cd reinforcement-learning-active-slam
```

**Build workspace (Could take a few minutes)**
```
colcon build --symlink-install
```

**Change ROS2 DDS implementation**
```
sudo apt install ros-humble-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
```

**Install package dependencies**
```
sudo rosdep init
rosdep update
rosdep install -i --from-path src --rosdistro humble -y
pip install setuptools==58.2.0
colcon build --symlink-install
```



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#rocket-dependencies)

## ➤ :rocket: Dependencies

**This project is developed using; ROS2 and Gazebo for simulation and coordination of robotic agents and Pytorch for Reinforcement Learning**


[![ROS Badge](https://img.shields.io/badge/ROS-22314E?logo=ros&logoColor=fff&style=for-the-badge)](https://docs.ros.org/en/humble/index.html)[![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge)](https://www.python.org/) [![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=for-the-badge)](https://pytorch.org/) [![NumPy Badge](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge)](https://numpy.org/)[![Gazebo Badge](https://custom-icon-badges.demolab.com/badge/-GazeboSim-FFBF00?style=for-the-badge&logo=package&logoColor=black)](https://gazebosim.org/home)



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#coffee-buy-me-a-coffee)

## ➤ :coffee: Buy me a coffee
Whether you use this project, have learned something from it, or just like it, please consider supporting it by buying me a coffee, so I can dedicate more time on open-source projects like this (҂⌣̀_⌣́)

<a href="https://www.buymeacoffee.com/i1Cps" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-violet.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#scroll-credits)

## ➤ :scroll: Credits

Theo Moore-Calters 


[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/i1Cps) [![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/theo-moore-calters)



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/cloudy.png)](#license)

## ➤ License
	
Licensed under [MIT](https://opensource.org/licenses/MIT).



