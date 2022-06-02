#! /usr/bin/env python

import turtlebot_env
import ppswarm_env
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
	logger_ins = logger.Logger(HOME + '/catkin_ws/src/frl_swarm/src/log/SNACDDPG', output_formats=[logger.HumanOutputFormat(sys.stdout)])
	board_logger = tensorboard_logging.Logger(os.path.join(logger_ins.get_dir(), "tf_board", time_str))
	log_path = HOME + '/catkin_ws/src/frl_swarm/src/log/SNACDDPG/csv/'
	weight_path = HOME + '/catkin_ws/src/frl_swarm/src/weights/SNACDDPG/'
	if not os.path.isdir(log_path):
		os.mkdir(log_path)
	if not os.path.isdir(weight_path):
		os.mkdir(weight_path)
	random.seed(args.random_seed)


	########################################################
	env= turtlebot_env.Env()
	replay_buffer = ExperienceReplayBuffer(total_timesteps=5000*256, type_buffer="HER")
	agent = DDPG(env, replay_buffer, args.num_weights, args.num_layers, weight_path)
	
	

	########################################################
	num_trials = args.num_epochs
	trial_len  = 256
	log_interval = 3
	train_indicator = 1
	tfirststart = time.time()
	firstSuccess = [0]*env.num_robots
	
	# Reward Logging
	with open(log_path + '{}.csv'.format(agent.file_name), mode='w') as csv_file:
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
	print("Initialisation Done")

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
				#print('wanted value is %s:', env.observation_space.shape[0])
				current_states = current_states.reshape((num_robots, env.observation_space.shape[0]))
				actions = []
				for k in range(num_robots):
					action, eps = agent.act(current_states[k])
					action = action.reshape((1, env.action_space.shape[0]))
					actions.append(action)
				actions = np.squeeze(np.asarray(actions))
				
				_, new_states, rewards, dones, infos, is_stops, isEverSuccess, _, _ = env.step(None, actions, 0.1) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value


				if j == (trial_len - 1):
					dones = np.array([True]*env.num_robots).reshape(env.num_robots, 1)

				
				step = step + 1
				epinfos.append(infos[0]['episode'])
				
				
				start_time = time.time()

				update_time = time.time()
				if (j % 5 == 0):
					agent.train(j)
					agent.update_target()   
					update_end_time = time.time()
					print("time for update: {}".format(update_end_time - update_time))
					update_times.append(update_end_time - update_time)

				end_time = time.time()
				#print("Train time: {}".format(end_time - start_time))
				#print("new_state: {}".format(new_state))
				new_states = new_states.reshape((num_robots, env.observation_space.shape[0]))

				# print shape of current_state
				#print("current_state is %s", current_state)
				##########################################################################################
				
				# Store only non-terminal transition samples.
				
				#agent.remember(current_states, actions, rewards, new_states, dones) # For uniform replay buffer
				state_sample = []
				action_sample = []
				reward_sample = []
				new_state_sample = []
				done_sample = []
				for k in range(num_robots):
					state_sample.append(current_states[k])
					action_sample.append(actions[k])
					reward_sample.append(rewards[k])
					new_state_sample.append(new_states[k])
					done_sample.append(dones[k])

				state_sample = np.asarray(state_sample)
				action_sample = np.asarray(action_sample)
				reward_sample = np.asarray(reward_sample)
				new_state_sample = np.asarray(new_state_sample)
				done_sample = np.asarray(done_sample)
				agent.replay_buffer.add(state_sample, action_sample, 
											   reward_sample, new_state_sample,
											   done_sample)
				current_states = new_states
				reward_idvl.append(reward_sample)
				reward_infos.append(safemean(reward_sample))
				for k in range(num_robots):
					if firstSuccess[k] == 0:
						if isEverSuccess[k] == True:
							firstSuccess[k] = i
				
				##########################################################################################
			if (i % 10==0):
				agent.save_weight(i, trial_len)
			

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
				logger_ins.logkv("serial_timesteps", i*trial_len)
				logger_ins.logkv("nupdates", i)                                                               
				logger_ins.logkv('eprewmean', reward_mean)
				logger_ins.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
				logger_ins.logkv('time_elapsed', tnow - tfirststart)
				logger_ins.logkv('update_time_mean', update_time_mean)
				for j in range(num_robots):
					logger_ins.logkv('eprewmean_{}'.format(j), reward_idvl_mean[j])
					board_logger.log_scalar('eprewmean_{}'.format(j), reward_idvl_mean[j], i)

				board_logger.log_scalar("eprewmean", reward_mean, i)
				board_logger.log_scalar("update_time_mean", update_time_mean, i)
				
				board_logger.flush()
				with open(log_path + '{}.csv'.format(agent.file_name), mode='a') as csv_file:
					csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['%i'%i, '%0.2f'%reward_mean])
				for j in range(env.num_robots):
					with open(log_path+ '{}_{}.csv'.format(agent.file_name, j), mode='a') as csv_file:
						csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
						csv_writer.writerow(['%i'%i, '%0.2f'%reward_idvl_mean[j]])
				reward_idvl_buf = []

		with open(log_path + '{}.csv'.format(agent.file_name), mode='a') as csv_file:
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
	parser.add_argument('--num_epochs', default=500, type=int)
	parser.add_argument('--num_weights', default=128, type=int)
	parser.add_argument('--num_layers', default=2, type=int)
	args = parser.parse_args('')
	args.exp_name = "exp_random_seed"
	name_var = 'random_seed'
	list_var = [101]
	for var in list_var:
		setattr(args, name_var, var)
		print(args)
		main(args)