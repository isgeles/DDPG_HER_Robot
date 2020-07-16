import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic
from collections import OrderedDict
from normalizer import Normalizer
from replay_buffer import ReplayBuffer


class ddpgAgent(object):
    def __init__(self, params):
        """Implementation of DDPG with Hindsight Experience Replay (HER).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dims = params['dims']
        self.buffer_size = params['buffer_size']
        self.tau = params['tau']
        self.batch_size = params['batch_size']
        self.critic_lr = params['lr_critic']
        self.actor_lr = params['lr_actor']
        self.norm_eps = params['norm_eps']
        self.norm_clip = params['norm_clip']
        self.clip_obs = params['clip_obs']
        self.clip_action = params['clip_action']

        self.T = params['T']
        self.rollout_batch_size = params['num_workers']
        self.clip_return = params['clip_return']
        self.sample_transitions = params['sample_her_transitions']
        self.gamma = params['gamma']
        self.replay_strategy = params['replay_strategy']

        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, self.input_dims[key])
        stage_shapes['o_2'] = stage_shapes['o']
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # normalizer
        self.obs_normalizer = Normalizer(size=self.dimo, eps=self.norm_eps, clip_range=self.norm_clip)
        self.goal_normalizer = Normalizer(size=self.dimg, eps=self.norm_eps, clip_range=self.norm_clip)

        # networks
        self.actor_local = Actor(self.input_dims).to(self.device)
        self.critic_local = Critic(self.input_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.critic_target = copy.deepcopy(self.critic_local)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr)

        # Configuring the replay buffer
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, self.input_dims[key])
                         for key, val in self.input_dims.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)


    def act(self, o, g, noise_eps=0., random_eps=0., testing=False):

        obs = self.obs_normalizer.normalize(o)
        goals = self.goal_normalizer.normalize(g)

        obs = torch.tensor(obs).to(self.device)
        goals = torch.tensor(goals).to(self.device)

        # for testing single environment
        if testing:
            with torch.no_grad():
                action = self.actor_local(torch.cat([obs, goals], dim=0)).cpu().data.numpy()
            return action

        actions = self.actor_local(torch.cat([obs, goals], dim=1))

        noise = (noise_eps * np.random.randn(actions.shape[0], 4)).astype(np.float32)
        actions += torch.tensor(noise).to(self.device)

        eps_greedy_noise = np.random.binomial(1, random_eps, actions.shape[0]).reshape(-1, 1)

        random_action = torch.tensor(np.random.uniform(
            low=-1., high=1., size=(actions.shape[0], self.dimu)).astype(np.float32)).to(self.device)

        actions += torch.tensor(eps_greedy_noise.astype(np.float32)).to(self.device) * (
                    random_action - actions)  # eps-greedy

        actions = torch.clamp(actions, -self.clip_action, self.clip_action)

        return actions

    def store_episode(self, episode_batch):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)

        # add transitions to normalizer
        episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
        shape = episode_batch['u'].shape
        num_normalizing_transitions = shape[0] * shape[1]  # num_rollouts * (T - 1), steps every cycle
        transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

        self.obs_normalizer.update(transitions['o'])
        self.obs_normalizer.recompute_stats()

        self.goal_normalizer.update(transitions['g'])
        self.goal_normalizer.recompute_stats()

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        return [transitions[key] for key in self.stage_shapes.keys()]

    def learn(self):

        batch = self.sample_batch()
        batch_dict = OrderedDict([(key, batch[i].astype(np.float32).copy())
                                  for i, key in enumerate(self.stage_shapes.keys())])
        batch_dict['r'] = np.reshape(batch_dict['r'], [-1, 1])

        # prepare state, action, reward, next state
        obs = torch.tensor(self.obs_normalizer.normalize(batch_dict['o'])).to(self.device)
        goal = torch.tensor(self.goal_normalizer.normalize(batch_dict['g'])).to(self.device)
        actions = torch.tensor(batch_dict['u']).to(self.device)
        rewards = torch.tensor(batch_dict['r'].astype(np.float32)).to(self.device)
        obs_2 = torch.tensor(self.obs_normalizer.normalize(batch_dict['o_2'])).to(self.device)

        # update critic --------------------------------------------------------------

        # compute predicted Q values
        next_actions = self.actor_target(torch.cat([obs_2, goal], dim=1))
        next_Q_targets = self.critic_target(torch.cat([obs_2, goal], dim=1), next_actions)

        # compute Q values for current states and clip them
        Q_targets = rewards + self.gamma * next_Q_targets          # Note: last experience of episode is not included
        Q_targets = torch.clamp(Q_targets, -self.clip_return, 0.)  # clipping

        # compute loss
        Q_expected = self.critic_local(torch.cat([obs, goal], dim=1), actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # update weights critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # update actor -------------------------------------------------------------

        # compute loss
        pred_actions = self.actor_local(torch.cat([obs, goal], dim=1))
        actor_loss = -self.critic_local(torch.cat([obs, goal], dim=1), pred_actions).mean()
        actor_loss += (pred_actions ** 2).mean()  # minimize action moments

        # update weights actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update_target_networks(self):
        """Soft update model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        # update critic net
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        # update actor net
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, path, name):
        torch.save(self.actor_local.state_dict(), path + '/'+name+'_checkpoint_actor_her.pth')
        torch.save(self.critic_local.state_dict(), path + '/'+name+'_checkpoint_critic_her.pth')
        self.obs_normalizer.save_normalizer(path + '/'+name+'_obs_normalizer.pth')
        self.goal_normalizer.save_normalizer(path + '/'+name+'_goal_normalizer.pth')


