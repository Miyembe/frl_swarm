#! /usr/bin/env python

import turtlebot_env
import numpy as np
import os
import sys
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/frl_swarm/src/modules')
print(sys.path)
import multiprocessing

import random
from collections import deque
#import modules.replay_buffer
from replay_buffers import *

import tensorboard_logging
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
from ActorCritic import Actor, Critic
from torch_util import *
from DDPG import DDPG

import logger

def safemean(xs):
		return np.nan if len(xs) == 0 else np.mean(xs)

def get_averaged_weights(agents, num_agents, num_actor_layers, num_critic_layers):
	actor_mean_weights = [None]*num_actor_layers
	actor_mean_bias = [None]*num_actor_layers

	critic_mean_weights = [None]*num_critic_layers
	critic_mean_bias = [None]*num_critic_layers

    # Initialise mean weights and bias for actor and critic
	for i in range(num_actor_layers):
		actor_mean_weights[i] = torch.zeros(size=agents[0].actor_model.layers[i].weight.shape).cuda()
		actor_mean_bias[i] = torch.zeros(size=agents[0].actor_model.layers[i].bias.shape).cuda()

	for i in range(num_critic_layers):
		critic_mean_weights[i] = torch.zeros(size=agents[0].critic_model.layers[i].weight.shape).cuda()
		critic_mean_bias[i] = torch.zeros(size=agents[0].critic_model.layers[i].bias.shape).cuda()

	with torch.no_grad():
		for i in range(num_actor_layers):
			for j in range(num_agents):
				actor_mean_weights[i] += agents[j].actor_model.layers[i].weight.data.clone()
				actor_mean_bias[i] += agents[j].actor_model.layers[i].bias.data.clone()
			actor_mean_weights[i] = actor_mean_weights[i]/num_agents
			actor_mean_bias[i] = actor_mean_bias[i]/num_agents

		for i in range(num_critic_layers):
			for j in range(num_agents):
				critic_mean_weights[i] += agents[j].critic_model.layers[i].weight.data.clone()
				critic_mean_bias[i] += agents[j].critic_model.layers[i].bias.data.clone()
			critic_mean_weights[i] = critic_mean_weights[i]/num_agents
			critic_mean_bias[i] = critic_mean_bias[i]/num_agents

	return actor_mean_weights, actor_mean_bias, critic_mean_weights, critic_mean_bias

def update_agents_models(agents, models_dict, num_actor_layers, num_critic_layers):
	for i in range(len(agents)):
		for j in range(num_actor_layers):
			agents[i].actor_model.layers[j].weight.data = models_dict["actor_weight"][j].clone()
			agents[i].actor_model.layers[j].bias.data = models_dict["actor_bias"][j].clone()
		for j in range(num_critic_layers):
			agents[i].critic_model.layers[j].weight.data = models_dict["critic_weight"][j].clone()
			agents[i].critic_model.layers[j].bias.data = models_dict["critic_bias"][j].clone()

def soft_update_agents_models(agents, models_dict, num_actor_layers, num_critic_layers, tau):
	# Soft update individual models with the model input. 
	# Tau [0 - 1]
	for i in range(len(agents)):
		for j in range(num_actor_layers):
			agents[i].actor_model.layers[j].weight.data = tau * (agents[i].actor_model.layers[j].weight.data) + (1-tau) * models_dict["actor_weight"][j].clone()
			agents[i].actor_model.layers[j].bias.data = tau * (agents[i].actor_model.layers[j].bias.data) + (1-tau) * models_dict["actor_bias"][j].clone()
		for j in range(num_critic_layers):
			agents[i].critic_model.layers[j].weight.data = tau * (agents[i].critic_model.layers[j].weight.data) + (1-tau) * models_dict["critic_weight"][j].clone()
			agents[i].critic_model.layers[j].bias.data = tau * (agents[i].critic_model.layers[j].bias.data) + (1-tau) * models_dict["critic_bias"][j].clone()

def main(args):
	time_str = time.strftime("%Y%m%d-%H%M%S")
	logger_ins = logger.Logger(HOME + '/catkin_ws/src/frl_swarm/src/log/FLDDPG', output_formats=[logger.HumanOutputFormat(sys.stdout)])
	board_logger = tensorboard_logging.Logger(os.path.join(logger_ins.get_dir(), "tf_board", time_str))
	random.seed(args.random_seed)
	log_path = HOME + '/catkin_ws/src/frl_swarm/src/log/FLDDPG/csv/'
	weight_path = HOME + '/catkin_ws/src/frl_swarm/src/weights/FLDDPG/'
	load_path = HOME + '/catkin_ws/src/frl_swarm/src/evaluation/SNACDDPG/20210528-234257/'
	epoch_num = 990
	net_size = 256
	########################################################
	env=turtlebot_env.Env()
	replay_buffers = [None]*env.num_robots
	agents = [None]*env.num_robots

	for i in range(env.num_robots):
		replay_buffers[i] = ExperienceReplayBuffer(total_timesteps=5000*256, type_buffer="HER")
		agents[i] = DDPG(env, replay_buffers[i], weight_path + '{}'.format(i))

		# Load weight from trained model with SNDDPG
		agents[i].actor_model.load_state_dict(torch.load('{}'.format(load_path)+'actormodel'+'-{}-{}.pkl'.format(epoch_num, net_size)))
		agents[i].critic_model.load_state_dict(torch.load('{}'.format(load_path)+'criticmodel'+'-{}-{}.pkl'.format(epoch_num, net_size)))
	

	########################################################
	num_trials = 1000
	trial_len  = 256
	log_interval = 3
	weight_interval = args.averaging_freq
	train_indicator = 1
	tfirststart = time.time()
	firstSuccess = [0]*env.num_robots
	
	# Reward Logging
	with open(log_path+ '{}.csv'.format(agents[0].file_name), mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['Episode', 'Average Reward'])
	for i in range(env.num_robots):
		with open(log_path+ '{}_{}.csv'.format(agents[i].file_name, i), mode='w') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			csv_writer.writerow(['Episode', 'Average Reward'])

	# Double ended queue with max size 100 to store episode info
	epinfobuf = deque(maxlen=100)
	reward_buf = deque(maxlen=100)
	reward_idvl_buf = []


	# Initialisation
	num_robots = env.num_robots
	num_actor_layers = len(agents[0].actor_model.layers)
	num_critic_layers = len(agents[0].critic_model.layers)
	_, current_state = env.reset()
	step = 0

	if (train_indicator==2):
		for i in range(num_trials):
			print("trial:" + str(i))
			current_state = env.reset()
			##############################################################################################
			total_reward = 0
			
			for j in range(100):
				step = step +1
				#print("step is %s", step)


				###########################################################################################
				#print('wanted value is %s:', env.observation_space.shape[0])
				current_state = current_state.reshape((1, env.observation_space.shape[0]))
				action, eps = agent.act(current_state)
				action = action.reshape((1, env.action_space.shape[0]))
				print("action is speed: %s, angular: %s", action[0][1], action[0][0])
				_, new_state, reward, done, _ = env.step(0.1, action[0][1]*5, action[0][0]*5) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				

	if (train_indicator==1):

        

		for i in range(num_trials):
			print("trial:" + str(i))
			
			''' Get states of multiple robots (num_robots x num_states) '''
			_, current_states = env.reset() 

			##############################################################################################
			epinfos = []
			reward_infos = []
			reward_idvl = []
			update_times = []

			for j in range(trial_len):
				
				###########################################################################################
				current_states = current_states.reshape((num_robots, env.observation_space.shape[0]))
				actions = []
				for k in range(num_robots):
					action, eps = agents[k].act(current_states[k])
					action = action.reshape((1, env.action_space.shape[0]))
					actions.append(action)

				actions = np.squeeze(np.asarray(actions))
				_, new_states, rewards, dones, infos, is_stops, isEverSuccess = env.step(actions, 0.1) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value

				if j == (trial_len - 1):
					dones = np.array([True]*env.num_robots).reshape(env.num_robots, 1)

				step = step + 1
				epinfos.append(infos[0]['episode'])
				
				start_time = time.time()

				update_time = time.time()
				if (j % 5 == 0):
					for k in range(num_robots):
						agents[k].train(j)
						agents[k].update_target()
					update_end_time = time.time()
					print("time for update: {}".format(update_end_time - update_time))
					update_times.append(update_end_time - update_time) 

				end_time = time.time()
				new_states = new_states.reshape((num_robots, env.observation_space.shape[0]))

				##########################################################################################
				
				# Store only non-terminal transition samples.
					
				state_samples = []
				action_samples = []
				reward_samples = []
				new_state_samples = []
				done_samples = []
				for k in range(num_robots):
					state_samples.append([])
					action_samples.append([])
					reward_samples.append([])
					new_state_samples.append([])
					done_samples.append([])

				for k in range(num_robots):
					state_samples[k].append(current_states[k])
					action_samples[k].append(actions[k])
					reward_samples[k].append(rewards[k])
					new_state_samples[k].append(new_states[k])
					done_samples[k].append(dones[k])

				state_samples = np.asarray(state_samples)
				action_samples = np.asarray(action_samples)
				reward_samples = np.asarray(reward_samples)
				new_state_samples= np.asarray(new_state_samples)
				done_samples = np.asarray(done_samples)
		
				for k in range(num_robots):
				    agents[k].replay_buffer.add(state_samples[k], action_samples[k], 
											reward_samples[k], new_state_samples[k],
											done_samples[k])
				current_states = new_states
				reward_idvl.append(reward_samples)
				reward_infos.append(safemean(reward_samples))

				for k in range(num_robots):
					if firstSuccess[k] == 0:
						if isEverSuccess[k] == True:
							firstSuccess[k] = i

				
				##########################################################################################
			
			if (i % 10==0):
				for k in range(num_robots):
					agents[k].save_weight(i, trial_len)
			

			epinfobuf.extend(epinfos)
			reward_buf.extend(reward_infos)
			reward_idvl_avg = np.mean(reward_idvl, axis = 0)
			reward_idvl_buf.append(reward_idvl_avg)
			tnow = time.time()
			#fps = int(nbatch / (tnow - tstart))
			
			##################################################
            ##      Logging and saving model & weights      ##
            ##################################################

			if i % log_interval == 0 or i == 0:
				#ev = explained_variance(values, returns)
				#reward_mean = safemean([epinfo['r'] for epinfo in epinfobuf])
				reward_mean = safemean([reward_info for reward_info in reward_buf])
				reward_idvl_mean = np.mean([reward_idvl for reward_idvl in reward_idvl_buf], axis =0) # Add individual logging
				update_time_mean = np.mean(update_times)
				print("reward_idvl_mean: {}".format(reward_idvl_mean))
				logger_ins.logkv("serial_timesteps", i*trial_len)
				logger_ins.logkv("nupdates", i)
				logger_ins.logkv("total_timesteps", i*trial_len)
				logger_ins.logkv('eprewmean', reward_mean)
				logger_ins.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
				logger_ins.logkv('time_elapsed', tnow - tfirststart)
				for j in range(num_robots):
					logger_ins.logkv('eprewmean_{}'.format(j), reward_idvl_mean[j])
					board_logger.log_scalar('eprewmean_{}'.format(j), reward_idvl_mean[j], i)

				board_logger.log_scalar("eprewmean", reward_mean, i)
				board_logger.log_scalar("update_time_mean", update_time_mean, i)
				
				
				board_logger.flush()
				with open(log_path + '{}.csv'.format(agents[0].file_name), mode='a') as csv_file:
					csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['%i'%i, '%0.2f'%reward_mean])
				for j in range(env.num_robots):
					with open(log_path+ '{}_{}.csv'.format(agents[j].file_name, j), mode='a') as csv_file:
						csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
						csv_writer.writerow(['%i'%j, '%0.2f'%reward_idvl_mean[j][0]])
				reward_idvl_buf = []

            ##################################################
            ##      Averaging Model Weights & Return it     ##
            ##################################################
			if i % weight_interval == 0:
				actor_mean_weights, actor_mean_bias, critic_mean_weights, critic_mean_bias = get_averaged_weights(agents, env.num_robots, num_actor_layers, num_critic_layers)
				models_dict = dict(actor_weight = actor_mean_weights, actor_bias = actor_mean_bias,
									critic_weight = critic_mean_weights, critic_bias = critic_mean_bias)
				update_agents_models(agents, models_dict, num_actor_layers, num_critic_layers)

		with open(log_path + '{}.csv'.format(agents[0].file_name), mode='a') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			csv_writer.writerow(['{}'.format(fs) for fs in firstSuccess])
		                         
	

		

	if train_indicator==0:
		for i in range(num_trials):
			print("trial:" + str(i))
			current_state = env.reset()
			
			agent.actor_model.load_weights(path + "actormodel-2950-256.h5")
			agent.critic_model.load_weights(path + "criticrmodel-2950-256.h5")
			##############################################################################################
			total_reward = 0
			
			for j in range(trial_len):

				###########################################################################################
				current_state = current_state.reshape((1, env.observation_space.shape[0]))

				start_time = time.time()
				action = agent.play(current_state)  # need to change the network input output, do I need to change the output to be [0, 2*pi]
				action = action.reshape((1, env.action_space.shape[0]))
				end_time = time.time()
				print(1/(end_time - start_time), "fps for calculating next step")

				_, new_state, reward, done = env.step(0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				###########################################################################################

				if j == (trial_len - 1):
					done = 1
					print("this is reward:", total_reward)
					

				# if (j % 5 == 0):
				# 	agent.train()
				# 	agent.update_target()   
				
				new_state = new_state.reshape((1, env.observation_space.shape[0]))
				# agent.remember(cur_state, action, reward, new_state, done)   # remember all the data using memory, memory data will be samples to samples automatically.
				# cur_state = new_state

				##########################################################################################
				#agent.remember(current_state, action, reward, new_state, done)
				current_state = new_state

				##########################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")
	args.exp_name = "exp_random_seed"
	name_var1 = 'random_seed'
	list_var1 = [12, 233, 998]
	name_var2 = 'averaging_freq'
	list_var2 = [10]

	for var1 in list_var1:
		setattr(args, name_var1, var1)
		for var2 in list_var2:
			setattr(args, name_var2, var2)
			print(args)
			main(args)