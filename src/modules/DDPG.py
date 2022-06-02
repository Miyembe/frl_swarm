
import numpy as np
import os
import sys
print(sys.path)
import multiprocessing

import random
from collections import deque
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, HighlightReplayBuffer
from schedule import LinearSchedule
import tensorboard_logging
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from ActorCritic import Actor, Critic
from torch_util import *

HOME = os.environ['HOME']

def stack_samples(samples):

	'''
    Func: stacking samples so that it is used easily for training
    '''
	
	current_states = np.asarray(samples[0])
	actions = np.asarray(samples[1])
	rewards = np.asarray(samples[2])
	new_states = np.asarray(samples[3])
	dones = np.asarray(samples[4])
	weights = np.asarray(samples[5])
	batch_idxes = np.asarray(samples[6])


	return current_states, actions, rewards, new_states, dones, weights, batch_idxes




class DDPG:
	def __init__(self, env, replay_buffer, num_weights, num_layers,weight_path):
		self.env  = env
		self.num_robots = env.num_robots

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.eps_counter = 0
		self.gamma = .90
		self.tau   = .01


		self.buffer_size = 1000000
		self.batch_size = 512

		self.hyper_parameters_lambda3 = 0.2
		self.hyper_parameters_eps = 0.2
		self.hyper_parameters_eps_d = 0.4

		self.demo_size = 1000
		self.time_str = time.strftime("%Y%m%d-%H%M%S")
		if os.path.isdir(weight_path) == False:
			os.mkdir(weight_path)
		self.parent_dir = weight_path
		self.path = os.path.join(self.parent_dir, self.time_str)
		os.mkdir(self.path)

        # Replay buffer
		#self.memory = deque(maxlen=1000000)
		# Replay Buffer
		self.replay_buffer = replay_buffer
		# File name
		self.file_name ="reward_{}_{}_{}".format(self.time_str, self.num_robots, self.replay_buffer.type_buffer)
		
		self.hid_list = [num_weights for _ in range(num_layers)]

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.actor_model = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.hid_list)
		self.target_actor_model = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.hid_list)
		self.actor_optim = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_model = Critic(self.env.observation_space.shape, self.env.action_space.shape, 1, self.hid_list)
		self.target_critic_model = Critic(self.env.observation_space.shape, self.env.action_space.shape, 1, self.hid_list)
		self.critic_optim = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
		

		hard_update(self.target_actor_model, self.actor_model) # Make sure target is with the same weight
		hard_update(self.target_critic_model, self.critic_model)
		self.cuda()



	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		for i in range(self.num_robots):
			self.memory.append([cur_state[i], action[i], reward[i], new_state[i], done[i]])



	def _train_critic_actor(self, samples):

		Loss = nn.MSELoss()
 

		# 1, sample
		cur_states, actions, rewards, new_states, dones, weights, batch_idxes = stack_samples(samples) # PER version also checks if I need to use stack_samples
		target_actions = to_numpy(self.target_actor_model(to_tensor(new_states)))
        
        # Critic Update
		self.critic_model.zero_grad()
		Q_now = self.critic_model([cur_states, actions])
		next_Q = self.target_critic_model([new_states, target_actions])
		dones = dones.astype(bool)
		Q_target = to_tensor(rewards) + self.gamma*next_Q.reshape(next_Q.shape[0]) * to_tensor(1 - dones)	

		td_errors = Q_target - Q_now.reshape(Q_now.shape[0])

		value_loss = Loss(Q_target, Q_now.squeeze())
		value_loss.backward()
		self.critic_optim.step()

		# Actor Update
		self.actor_model.zero_grad()
		policy_loss = -self.critic_model([
			to_tensor(cur_states),
			self.actor_model(to_tensor(cur_states))
		])
		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.actor_optim.step()

		# NoisyNet noise reset
		self.actor_model.reset_noise()
		self.target_actor_model.reset_noise()

		return td_errors
		# print("grads*weights is %s", grads)
		
	def train(self, t):
		batch_size = self.batch_size
		#if len(self.memory) < batch_size: #batch_size: # uniform buffer 
		#	return
		
		if self.replay_buffer.replay_buffer.__len__() < batch_size: #per
			return
		#samples = random.sample(self.memory, batch_size)    # what is deque, what is random.sample? self.mempory begins with self.memory.append
		samples = self.replay_buffer.replay_buffer.sample(batch_size, beta=self.replay_buffer.beta_schedule.value(t))
		(obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = samples
		# samples = self.memory.sample(1, batch_size)
		self.samples = samples

		td_errors = self._train_critic_actor(samples)

		# priority updates
		#new_priorities = np.abs(td_errors) + self.replay_buffer.prioritized_replay_eps
		#self.replay_buffer.replay_buffer.update_priorities(batch_idxes, new_priorities)



	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		soft_update(self.target_actor_model, self.actor_model, self.tau)
				
	def _update_critic_target(self):
		soft_update(self.target_critic_model, self.critic_model, self.tau)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()
	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):  # this function returns action, which is predicted by the model. parameter is epsilon
		if self.eps_counter >= self.num_robots:
			self.epsilon *= self.epsilon_decay
			self.eps_counter = 0
		else:
			self.eps_counter += 1
		eps = self.epsilon
		cur_state = np.array(cur_state).reshape(1, self.env.state_num)
		action = to_numpy(self.actor_model(to_tensor(cur_state))).squeeze(0)
		action = action.reshape(1,2)
		if np.random.random() < self.epsilon:
			action[0][0] = action[0][0] + (np.random.random()-0.5)*0.4
			action[0][1] = action[0][1] + (np.random.random())*0.4
			return action, eps	
		else:
			action[0][0] = action[0][0] 
			action[0][1] = action[0][1]
			return action, eps
		

	# ========================================================================= #
	#                              save weights                                 #
	# ========================================================================= #

	def save_weight(self, num_trials, trial_len):
		torch.save(
			self.actor_model.state_dict(),
			self.path + '/actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) +'.pkl'
		)
		torch.save(
			self.critic_model.state_dict(),
			self.path + '/criticmodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.pkl'
		)
		#self.actor_model.save_weights(self.path + 'actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)
		#self.critic_model.save_weights(self.path + 'criticmodel' + '-' + str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)#("criticmodel.h5", overwrite=True)

	def load_weights(self, output):

		self.actor_model.load_state_dict(
			torch.load('{}.pkl'.format(output))
		)

		self.critic_model.load_state_dict(
			torch.load('{}.pkl'.format(output))
		)

	def play(self, cur_state):
		return to_numpy(self.actor_model(to_tensor(cur_state), volatile = True)).squeeze(0)

	def cuda(self):
		self.actor_model.cuda()
		self.target_actor_model.cuda()
		self.critic_model.cuda()
		self.target_critic_model.cuda()
		#return self.actor_model.predict(cur_state)