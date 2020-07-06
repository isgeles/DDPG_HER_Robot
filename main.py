import numpy as np
import random
import torch
import gym
import mujoco_py
from mpi4py import MPI

import progressbar as pb           # tracking time while training
import matplotlib.pyplot as plt    # plotting scores

from ddpg import ddpgAgent
from her_sampler import make_sample_her_transitions
from parallelEnv.cmd_util import make_vec_env
from rollout import RolloutWorker


DEFAULT_PARAMS = {
    # environment
    'env_name': 'FetchPush-v1',               # 'FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1'
    'seed': 0,                                # random seed for environment, torch, numpy, random packages
    'T': 50,                                  # maximum episode length

    # training setup
    'replay_strategy': 'future',              # 'none' for vanilla ddpg, 'future' for HER
    'num_workers': 16,                        # number of parallel workers with mpi
    'max_timesteps': 8000000,                 # maximum number of total timesteps, HER paper: 8mil
    #'n_epochs': 50,                          # number of epochs, HER paper: 200 epochs
    'n_cycles': 10,                           # number of cycles per epoch, HER paper: 50 cycles TODO
    'n_optim': 40,                            # number of optimization steps every cycle
    'n_test_rollouts': 10,                    # number of rollouts for testing, rollouts are episodes from num_workers

    # Agent hyper-parameters
    'lr_actor': 0.001,                        # learning rate actor network
    'lr_critic': 0.001,                       # learning rate critic network
    'buffer_size': int(1e6),                  # replay-buffer size
    'tau': 0.05,                              # soft update of target network, 1-tau = decay coefficient
    'batch_size': 256,                        # batch size per mpi thread
    'gamma': 0.98,                            # discount factor
    'clip_return': 50.,                       # return clipping
    'clip_obs': 200.,                         # observation clipping
    'clip_action': 1.,                        # action clipping

    # exploration
    'random_eps': 0.3,                        # probability of random action in hypercube of possible actions
    'noise_eps': 0.2,                        # std of gaussian noise added actions

    # normalization
    'norm_eps': 0.01,                         # eps for observation normalization
    'norm_clip': 5,                           # normalized observations are clipped to this values

    # location of files for report
    'results_path': './tmp_results'
}


def set_seeds(seed=0):
    """Set the random seed to all packages. Note: MPI workers will have different seeds in parallel environments."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def dims_and_reward_fun(env_name):
    """Get dimensions of observations, action, goal and the used reward function."""
    env = gym.make(env_name)
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'info_is_success': 1,
    }
    DEFAULT_PARAMS['dims'] = dims
    DEFAULT_PARAMS['reward_fun'] = env.compute_reward


def train(agent, rollout_worker, evaluation_worker):
    """Train DDPG with multiple workers"""
    scores = []


    n_epochs = int(DEFAULT_PARAMS['max_timesteps'] // DEFAULT_PARAMS['n_cycles'] //
                   DEFAULT_PARAMS['T'] // DEFAULT_PARAMS['num_workers'])

    # widget bar to display progress
    widget = ['training loop: ', pb.Percentage(), ' ',
              pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=n_epochs).start()

    for epoch in range(n_epochs):

        for _ in range(DEFAULT_PARAMS['n_cycles']):
            episode = rollout_worker.generate_rollouts()  # generate episodes with every parallel environment
            agent.store_episode(episode)                  # store experiences as whole episodes
            for _ in range(DEFAULT_PARAMS['n_optim']):    # optimize target network
                agent.train()
            agent.update_target_net()                     # update target network

        # testing agent for report
        test_scores = []
        for _ in range(DEFAULT_PARAMS['n_test_rollouts']):
            evaluation_worker.generate_rollouts()
            test_scores.append(evaluation_worker.mean_success)
        print('\n \tEpoch: {} / {}, Success: {:.4f}'.format(epoch, n_epochs, np.mean(test_scores)))
        scores.append(np.mean(test_scores))

        # different threads use different seeds
        MPI.COMM_WORLD.Bcast(np.random.uniform(size=(1,)), root=0)
        timer.update(epoch)
    agent.save_checkpoint(DEFAULT_PARAMS['results_path'], DEFAULT_PARAMS['env_name'])
    timer.finish()
    return scores


def main():
    seed = set_seeds(DEFAULT_PARAMS['seed'])
    env = make_vec_env(DEFAULT_PARAMS['env_name'], DEFAULT_PARAMS['num_workers'], seed=seed,
                       flatten_dict_observations=False)

    dims_and_reward_fun(DEFAULT_PARAMS['env_name'])
    DEFAULT_PARAMS['sample_her_transitions'] = make_sample_her_transitions(
        replay_strategy=DEFAULT_PARAMS['replay_strategy'], replay_k=4, reward_fun=DEFAULT_PARAMS['reward_fun'])

    agent = ddpgAgent(DEFAULT_PARAMS)

    rollout_worker = RolloutWorker(env, agent, DEFAULT_PARAMS)
    evaluation_worker = RolloutWorker(env, agent, DEFAULT_PARAMS, evaluate=True)

    scores = train(agent, rollout_worker, evaluation_worker)

    # save stats for report
    np.savetxt(DEFAULT_PARAMS['results_path']+'/scores_'+DEFAULT_PARAMS['env_name']+'.csv', scores, delimiter=',')
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.savefig(DEFAULT_PARAMS['results_path']+'/scores_'+DEFAULT_PARAMS['env_name']+'.png')
    plt.show()


if __name__ == '__main__':
    main()
