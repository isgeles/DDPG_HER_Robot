# Fetch Robotic Gym Environment solved in Pytorch (DDPG+HER)

Used Algorithms:
- Deep Deterministic Policy Gradient (DDPG)
- DDPG with Hindsight Experience Replay (DDPG+HER)


### Results

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


