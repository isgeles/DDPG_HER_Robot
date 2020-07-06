"""
Code structure from openai/baselines
"""

from parallelEnv import SubprocVecEnv
from parallelEnv import DummyVecEnv
from gym.wrappers import FlattenDictWrapper

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
import numpy as np
import random


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    np.random.seed(myseed)
    random.seed(myseed)


def make_vec_env(env_id, num_env, seed,
                 start_index=0,
                 flatten_dict_observations=True):
    """
    Create a wrapped, monitored SubprocVecEnv MuJoCo.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            flatten_dict_observations=flatten_dict_observations
        )

    set_global_seeds(seed)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id,
             mpi_rank=0,
             subrank=0,
             seed=None,
             flatten_dict_observations=True):

    env = gym.make(env_id)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    return env

