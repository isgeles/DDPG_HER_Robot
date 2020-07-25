# Fetch Robotic Gym Environment solved in Pytorch (DDPG+HER)

Used Algorithms:
- Deep Deterministic Policy Gradient (DDPG)
- DDPG with Hindsight Experience Replay (DDPG+HER)


### Environments

There are 4 different tasks to be solved in the openai robotic Fetch gym (arm with 7 DOF), with the mujoco physics simulator:
 > - Reach  (observation space: 10 observations, 3 achieved goal, 3 desired goal)
 > - Push  (observation space: 25 observations, 3 achieved goal, 3 desired goal)
 > - PickAndPlace  (observation space: 25 observations, 3 achieved goal, 3 desired goal)
 > - Slide  (observation space: 25 observations, 3 achieved goal, 3 desired goal)

in a episode of 50 timesteps, where the target position is always visualised in red.

For every timestep where the target position is not reached, the agent receives an reward of -1. If the agent fails to reach the goal until the end of an episode it is considered unsuccessful.

The action-space is a vector of 4 values (x,y,z, gripper), the gripper value is irrelevant for all environements except for PickAndPlace.

The algorithm DDPG with standard experience replay fails to learn in these environements, however with Hindsight Experience Replay (HER) the tasks can be solved.


### Results

Below you can see the success-rate over 200 epochs (1 epoch = 50 cycles = 16 episodes) for all the different Fetch environments (and random seed = 0).

FetchReach-v1| FetchPush-v1
-----------------------|-----------------------|
![](./trained/Reach/seed0/scores_FetchReach-v1_0.png)| ![](./trained/Push/seed0/scores_FetchPush-v1_0.png)

FetchPickAndPlace-v1| FetchSlide-v1
-----------------------|-----------------------|
![](./trained/PickAndPlace/seed0/scores_FetchPickAndPlace-v1_0.png)| ![](./trained/Slide/seed0/scores_FetchSlide-v1_0.png)


### Watch trained agents:

FetchReach-v1| FetchPush-v1
-----------------------|-----------------------|
![](./trained/Reach/reach_HER.gif)| ![](./trained/Push/push_HER.gif)

FetchPickAndPlace-v1| FetchSlide-v1
-----------------------|-----------------------|
![](./trained/PickAndPlace/pickandplace_HER.gif)| ![](./trained/Slide/slide_HER.gif)

### Files in this Repository
                    
    .
    ├── tmp_results/                       # folder for storing new results
    ├── trained/                           # stored gifs, weights for trained networks and results for different seeds 
    ├── agent_demo.ipynb
    ├── ddpg.py
    ├── her_sampler.py
    ├── main.py
    ├── model.py
    ├── parallelEnvironment.py 
    ├── replay_buffer.py
    ├── rollout.py
    ├── utils.py
    └── README.md


### Python Packages
 - abc
 - collections
 - copy
 - gym
 - IPython  (for displaying environment in notebook)
 - matplotlib
 - multiprocessing
 - mujoco
 - numpy
 - random
 - threading
 - torch
 - progressbar   (for tracking time during training)


