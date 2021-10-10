import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np
from noisynet import NoisyLinear
from torch_util import *

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hid_list):
        super(Actor, self).__init__()
        self.in_dim = in_dim[0]
        self.out_dim = out_dim[0]
        self.hid_list = hid_list
        self.n_hid_layers = len(hid_list)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # === Create Layers === #
        self.layers = nn.ModuleList()
        for i in range(self.n_hid_layers+1):
            if i == 0:
                #print("self.in_dim: {}".format(self.in_dim))
                #print("self.hid_list[i]: {}".format(self.hid_list[i]))
                self.layers.append(nn.Linear(self.in_dim, self.hid_list[i]))
            elif i == self.n_hid_layers:
                for j in range(self.out_dim):
                    self.layers.append(nn.Linear(self.hid_list[i-1], 1))
                    self.layers.append(nn.Linear(self.hid_list[i-1], 1))
            else: self.layers.append(nn.Linear(self.hid_list[i-1], self.hid_list[i]))

    def forward(self, x):
        for i in range(self.n_hid_layers+1):
            if i == self.n_hid_layers:
                x1 = self.layers[i](x)
                out1 = self.tanh(x1) # angular velocity
                x2 = self.layers[i+1](x)
                out2 = self.sigmoid(x2) # linear velocity
            else:
                x = self.layers[i](x)
                x = self.relu(x)

        return torch.cat([out1, out2], 1)

    def reset_noise(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], NoisyLinear):
                self.layers[i].reset_noise()

        

        

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, out_dim, hid_list):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim[0]
        self.act_dim = act_dim[0]
        self.out_dim = out_dim
        self.hid_list = hid_list
        self.n_hid_layers = len(hid_list)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # === Create Layers === #
        self.layers = nn.ModuleList()
        for i in range(self.n_hid_layers+1):
            if i == 0:
                self.layers.append(nn.Linear(self.obs_dim, self.hid_list[i]))
            elif i == 1:
                self.layers.append(nn.Linear(self.hid_list[i-1]+self.act_dim, self.hid_list[i]))
            elif i == self.n_hid_layers:
                self.layers.append(nn.Linear(self.hid_list[i-1], self.out_dim))
            else: self.layers.append(nn.Linear(self.hid_list[i-1], self.hid_list[i]))

    def forward(self, xs):
        x, a = xs
        print("x size: {}".format(x.shape))
        print("a size: {}".format(a.shape))
        if torch.is_tensor(x) == False:
            x = to_tensor(np.array(x))
        if torch.is_tensor(a) == False:            
            a = to_tensor(np.array(a))
        #print("x: {}, type: {}, size: {}".format(x, type(x), x.shape))
        #print("a: {}, type: {}, size: {}".format(a, type(a), a.shape))
        #x = self.layers[0](x)
        #x = self.relu(x)
        #x = self.layers[1](torch.cat((x,a), 1))
        #x = self.relu(x)

        for i in range(self.n_hid_layers+1):
            if i == 1:
                x = self.layers[i](torch.cat((x,a),1))
                x = self.relu(x)
            elif i == self.n_hid_layers:
                x = self.layers[i](x)
            else:
                #print("x: {}".format(x.shape))
                #print("layers[i]: {}".format(self.layers[i]))
                x = self.layers[i](x)
                x = self.relu(x)
                
        return x

    def reset_noise(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], NoisyLinear):
                self.layers[i].reset_noise()