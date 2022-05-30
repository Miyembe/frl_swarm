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

def main(args):
	time_str = time.strftime("%Y%m%d-%H%M%S")
	logger_ins = logger.Logger(HOME + '/catkin_ws/src/frl_swarm/src/log/SEACDDPG', output_formats=[logger.HumanOutputFormat(sys.stdout)])
	board_logger = tensorboard_logging.Logger(os.path.join(logger_ins.get_dir(), "tf_board", time_str))
	random.seed(args.random_seed)
	log_path = HOME + '/catkin_ws/src/frl_swarm/src/log/SEACDDPG/csv/'
	weight_path = HOME + '/catkin_ws/src/frl_swarm/src/weights/SEACDDPG/'

	########################################################
	env= turtlebot_env.Env()
	replay_buffer = ExperienceReplayBuffer(total_timesteps=5000*256, type_buffer="HER")
	agents = [None]*env.num_robots
	for i in range(env.num_robots):
		agents[i] = DDPG(env, replay_buffer, weight_path + '{}'.format(i))
	

	########################################################
	num_trials = 1000
	trial_len  = 256
	log_interval = 3
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
				
				#agent.remember(current_states, actions, rewards, new_states, dones) # For uniform replay buffer
					
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
	name_var = 'random_seed'
	list_var = [101,102,103]
	for var in list_var:
		setattr(args, name_var, var)
		print(args)
		main(args)