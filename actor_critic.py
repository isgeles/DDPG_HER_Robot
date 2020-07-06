import torch
import torch.nn as nn

from model import Actor, Critic


class ActorCritic(nn.Module):
    def __init__(self, obs_normalizer, goal_normalizer, dims):
        super(ActorCritic, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(dims).to(self.device)
        self.critic = Critic(dims).to(self.device)

        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizer

    # def compute_all(self, obs, goal, actions):
    #     obs = self.obs_normalizer.normalize(obs)
    #     obs = torch.tensor(obs).to(self.device)
    #
    #     goal = self.goal_normalizer.normalize(goal)
    #     goal = torch.tensor(goal).to(self.device)
    #     policy_input = torch.cat([obs, goal], dim=1)
    #     policy_output = self.actor(policy_input)
    #     # temporary
    #     self.pi = policy_output  # action expected
    #     critic_input = torch.cat([obs, goal], dim=1)
    #     self.q_pi = self.critic(critic_input, policy_output)
    #     actions = torch.tensor(actions).to(self.device)
    #     critic_input = torch.cat([obs, goal], dim=1)
    #     self.q = self.critic(critic_input, actions)


    # def get_action(self, obs, goals):  # TODO remove
    #     obs = self.obs_normalizer.normalize(obs)
    #     obs = torch.tensor(obs).to(self.device)
    #
    #     goals = self.goal_normalizer.normalize(goals)
    #     goals = torch.tensor(goals).to(self.device)
    #     policy_input = torch.cat([obs, goals], dim=1)
    #
    #     return self.actor(policy_input)

    # def compute_q_values(self, obs, goals, actions):
    #     obs = self.obs_normalizer.normalize(obs)
    #     obs = torch.tensor(obs).to(self.device)
    #
    #     goals = self.goal_normalizer.normalize(goals)
    #     goals = torch.tensor(goals).to(self.device)
    #     input_tensor = torch.cat([obs, goals, actions], dim=1)
    #
    #     return self.critic(input_tensor)


