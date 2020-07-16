import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Actor (Policy) Model.
    """
    def __init__(self, dims, fc1_units=256, fc2_units=256, fc3_units=256):
        """
        Initialize parameters and build model.
        @param dims: dict containing dimensions for input layer ('o' and 'g' ) and output layer ('a')
        @param fc1_units: (int) number units in the first hidden layer
        @param fc2_units: (int) number units in the second hidden layer
        @param fc3_units: (int) number units in the third hidden layer
        """
        super(Actor, self).__init__()
        self.state_size = dims['o'] + dims['g']
        self.action_size = dims['u']

        self.fc1 = nn.Linear(self.state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, self.action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        resetting the weights of the network
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states to actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """
    Critic (Value) Model.
    """
    def __init__(self, dims, fcs1_units=256, fc2_units=256, fc3_units=256):
        """
        Initialize parameters and build model.
        @param dims: dict containing dimensions for input layer ('o' and 'g' ) and output layer ('a')
        @param fcs1_units: (int) number units in the first hidden layer
        @param fc2_units: (int) number units in the second hidden layer (actions added here to critic)
        @param fc3_units: (int) number units in the third hidden layer
        """
        super(Critic, self).__init__()
        self.state_size = dims['o'] + dims['g']
        self.action_size = dims['u']

        self.fcs1 = nn.Linear(self.state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+self.action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        resetting the weights of the network
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action) pairs to Q-values.
        """
        xs = F.relu(self.fcs1(state))
        action = action.float()
        x = torch.cat((action, xs), dim=1)
        x = F.relu(self.fc2(x))
        Q = F.relu(self.fc3(x))
        return self.fc4(Q)
