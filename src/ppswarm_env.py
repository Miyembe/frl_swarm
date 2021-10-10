#! /usr/bin/env python

import rospy
import rospkg
import tf
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Quaternion
import math
from math import *


from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

from tf.transformations import quaternion_from_euler
#from coordinates_generation import Coordinates

import time
import threading
import gym
import numpy as np
import random
import csv
import pandas as pd
import copy

import scipy.io as sio



class InfoGetter(object):
    '''
    Get Information from rostopic. It reduces delay 
    '''
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        #"""Blocks until the data is rx'd with optional timeout
        #Returns the received message
        #"""
        self._event.wait(timeout)
        return self._msg


class Env:

    # ========================================================================= #
	#                                Env Class                                  #
	# ========================================================================= #

    '''
    This class define Env (identical concept of OpenAI gym Env).
    1. __init__() - define required variables
    2. reset()
    3. step()
    '''

    def __init__(self):

        # Settings
        self.num_robots = 8
        self.num_experiments = 1

        # Node initialisation
        self.node = rospy.init_node('turtlebot_env', anonymous=True)
        self.pose_ig = InfoGetter()
        
        self.pub_tb3 = [None]*self.num_robots
        self.laser_ig = [None]*self.num_robots
        self.sub_scan = [None]*self.num_robots
        for i in range(self.num_robots):
            self.pub_tb3[i] = rospy.Publisher('/tb3_{}/cmd_vel'.format(i), Twist, queue_size=1)
            self.laser_ig[i] = InfoGetter()
            self.sub_scan[i] = rospy.Subscriber('/tb3_{}/scan'.format(i), LaserScan, self.laser_ig[i])
        self.position = Point() # Do I need this position in this script? or just get pheromone value only?
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.rate = rospy.Rate(100)

        # Default Twist message
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.0 #linear_x
        self.move_cmd.angular.z = 0.0 #angular_z

        # Collision Bool State
        self.is_collided = False

        # Observation & action spaces
        self.num_lasers = 24
        self.state_num = self.num_lasers + 4 # 2 for pheromone grad, 2 for pheromone value, 2 for linear & angular vel, 2 for distance and angle diff to the target in polar coordinates
        self.action_num = 2 # linear_x and angular_z
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # Initialise positions and orientations
        #self.coordinates = Coordinates(self.num_robots, self.num_experiments)
        #self.robot_pos = [None]*self.num_robots
        #self.robot_quat = [None]*self.num_robots
        #self.target_pos = [None]*self.num_robots
        #self.target_quat = [None]*self.num_robots
        #for i in range(self.num_robots):
            #self.robot_pos[i], self.robot_quat[i], self.target_pos[i], self.target_quat[i] = self.coordinates.circular_reset(i)
        
        self.x_init = [5.15, 5.15, -5.15, -5.15, 3, -3, 3, -3]#[0.0, 2.0] # [0.0,4.0]
        self.y_init = [3, -3, 3, -3, 5.15, 5.15, -5.15, -5.15]  #[0.0, -2.0]  # [0.0,0.0]
        self.theta_init = [pi, pi, 0, 0, -pi/2, -pi/2, pi/2, pi/2]
        self.theta_quat = [quaternion_from_euler(0, 0, i) for i in self.theta_init]
        self.x_prev = [5.15, 5.15, -5.15, -5.15, 3, -3, 3, -3]#[0.0, 2.0] # [0.0,4.0]
        self.y_prev = [3, -3, 3, -3, 5.15, 5.15, -5.15, -5.15]  #[0.0, -2.0]  # [0.0,0.0]
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots
        self.target_index = 0

        # Set target position
        self.target = [[-5.15, -3.0], [-5.15, 3.0], [5.15, -3.0], [5.15, 3.0], [-3.0, -5.15], [3.0, -5.15], [-3.0, 5.15], [3.0, 5.15]]#[[4.0, 0.0], [2.0, 2.0]] # Two goal (crossing scenario) # [[4.0,0.0], [0.0,0.0]]

        # Set turtlebot index in Gazebo (to distingush from other models in the world)
        self.model_index = -1
        self.model_state = ModelStates()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Miscellanous
        self.ep_len_counter = 0
        self.just_reset = [False] * self.num_robots
        self.dones = [False] * self.num_robots
        self.isEverSuccess = [False] * self.num_robots
        self.grad_sensitivity = 20

        # Weight Averaging Related
        self.toBeUpdated = [[None] for i in range(self.num_robots)]
        self.partnerIDs = [[None] for i in range(self.num_robots)]
        self.partnerIDs_prev = [[None] for i in range(self.num_robots)]
        self.updateProb = 0.1
        self.weight_averaging = False

    def reset(self, model_state = None, id_bots = 999):

        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset the positions of robots and targets
        '''

        # ========================================================================= #
	    #                          1. TARGET UPDATE                                 #
	    # ========================================================================= #
        
        self.is_collided = False

        # Assigning index number of model_state for each robot
        tb3 = [None]*self.num_robots
        
        if model_state is not None:
            for i in range(len(model_state.name)):
                if 'tb3' in model_state.name[i]:
                    tb3[int(model_state.name[i][-1])] = i
        else:
            model_state = self.pose_ig.get_msg()
            for i in range(len(model_state.name)):
                if 'tb3' in model_state.name[i]:
                    tb3[int(model_state.name[i][-1])] = i

        # All reset, target_index is updated to change environment for each robots

        
        # for i in range(self.num_robots):
        #     if id_bots == tb3[i] or id_bots == 999:
        #         if self.coordinates.reset_index[i] < self.num_experiments-1:
        #             self.coordinates.reset_index[i] += 1
        #         else:
        #             self.coordinates.reset_index[i] = 0

        # for i in range(self.num_robots):
        #     self.robot_pos[i], self.robot_quat[i], self.target_pos[i], self.target_quat[i] = self.coordinates.circular_reset(i)



        # ========================================================================= #
	    #                                 2. RESET                                  #
	    # ========================================================================= #
        robot_msgs = [None]*self.num_robots
        target_msgs = [None]*self.num_robots


        for i in range(self.num_robots):

            # Robot Pose assignment
            robot_msgs[i] = ModelState()
            robot_msgs[i].model_name = 'tb3_{}'.format(i)
            robot_msgs[i].pose.position.x = self.x_init[i]
            robot_msgs[i].pose.position.y = self.y_init[i]
            robot_msgs[i].pose.position.z = 0.0#self.robot_pos[i][2]
            robot_msgs[i].pose.orientation.x = self.theta_quat[i][0]
            robot_msgs[i].pose.orientation.y = self.theta_quat[i][1]
            robot_msgs[i].pose.orientation.z = self.theta_quat[i][2]
            robot_msgs[i].pose.orientation.w = self.theta_quat[i][3]

            # Target Pose assignment
            target_msgs[i] = ModelState()
            target_msgs[i].model_name = 'target_{}'.format(i)
            target_msgs[i].pose.position.x = self.target[i][0]#self.target_pos[i][0]
            target_msgs[i].pose.position.y = self.target[i][1]#self.target_pos[i][1]
            target_msgs[i].pose.position.z = 0.0#self.target_pos[i][2]
            target_msgs[i].pose.orientation.x = 0.0#self.target_quat[i][0]
            target_msgs[i].pose.orientation.y = 0.0#self.target_quat[i][1]
            target_msgs[i].pose.orientation.z = 0.0#self.target_quat[i][2]
            target_msgs[i].pose.orientation.w = 0.0#self.target_quat[i][3]

        rospy.wait_for_service('gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        resp_robots = [None] * self.num_robots
        resp_targets = [None] * self.num_robots
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            for i in range(self.num_robots):
                if id_bots == tb3[i] or id_bots == 999:
                    resp_robots[i] = set_state(robot_msgs[i])
                    #resp_targets[i] = set_state(target_msgs[i])
                    self.dones[i] = False

        except rospy.ServiceException as e:
            print("Service Call Failed: %s"%e)

        initial_state = np.zeros((self.num_robots, self.state_num))
        
        self.just_reset = True

        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0

        for i in range(self.num_robots):
            if id_bots == tb3[i] or id_bots == 999:
                self.pub_tb3[i].publish(self.move_cmd)
        self.rate.sleep()

        print("Reset Done")
        return range(0, self.num_robots), initial_state


    def action_to_twist(self, action):
        t = Twist()

        # Rescale and clipping the actions
        t.linear.x = action[1]*0.3
        t.linear.x = min(1, max(-1, t.linear.x))
        
        t.angular.z = min(pi/2, max(-pi/2, action[0]))
        return t
    
    def posAngle(self, model_state):
        '''
        Get model_state from rostopic and
        return (1) x position of robots (2) y position of robots (3) angle of the robots (4) id of the robots
        '''
        pose = [None]*self.num_robots
        ori = [None]*self.num_robots
        x = [None]*self.num_robots
        y = [None]*self.num_robots
        angles = [None]*self.num_robots
        theta = [None]*self.num_robots
        tb3 = [None]*self.num_robots
        for i in range(len(model_state.name)):
            if 'tb3' in model_state.name[i]:
                tb3[int(model_state.name[i][-1])] = i
        tb3_pose = [model_state.pose[i] for i in tb3]
        for i in range(self.num_robots):
            # Write relationship between i and the index
            pose[i] = tb3_pose[i] # Need to find the better way to assign index for each robot
            ori[i] = pose[i].orientation
            x[i] = pose[i].position.x
            y[i] = pose[i].position.y
            angles[i] = tf.transformations.euler_from_quaternion((ori[i].x, ori[i].y, ori[i].z, ori[i].w))
            theta[i] = angles[i][2]
        
        return x, y, theta, tb3

    def angle0To360(self, angle):
        for i in range(self.num_robots):
            if angle[i] < 0:
                angle[i] = angle[i] + 2*math.pi
        return angle
    
    def anglepiTopi(self, angle):
        for i in range(self.num_robots):
            if angle[i] < -math.pi:
                angle[i] = angle[i] + 2*math.pi
            if angle[i] > math.pi:
                angle[i] = angle[i] - 2*math.pi
        return angle

    def swap2elements(self, array):
        assert len(array) == 2
        tmp = [None]*2
        tmp[0] = array[1]
        tmp[1] = array[0]
        return tmp

    def getLaser(self, scan):
        scan_range = []
        min_range = 0.25
        collision = False
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        if min_range > min(scan_range) > 0:
            collision = True
        
        return scan_range, collision
    def step(self, tBU, actions, time_step=0.1):
        '''
        Take a step with the given action from DRL in the Environment
        0. Initialisation
        1. Move Robot for given time step
        2. Read robot pose
        3. Calculation of distances
        4. Read Pheromone
        5. Reward Assignment
        6. Reset
        7. Other Debugging Related
        '''
        # 0. Initiliasation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0

        # NN weight update related init
        if self.weight_averaging == True:
            self.toBeUpdated = tBU
        else: 
            self.toBeUpdate = None

        # Check if the robots are terminated
        dones = self.dones
        is_stops = dones
        print("actions: {}".format(actions))
        twists = [self.action_to_twist(action) for action in np.asarray(actions)]

        # rescaling the action
        for i in range(len(twists)):
            twists[i].linear.x = (twists[i].linear.x+1) * 1/2  # only forward motion
            twists[i].angular.z = twists[i].angular.z
            if is_stops[i] == True:
                twists[i] = Twist()
        
        linear_x = [i.linear.x for i in twists]
        angular_z = [i.angular.z for i in twists]
        

        # position of turtlebot before taking steps & Update related

        x_prev = self.x_prev
        y_prev = self.y_prev
        distance_to_goals_prv = [None]*self.num_robots
        


        for i in range(self.num_robots):
            distance_to_goals_prv[i] = sqrt((x_prev[i]-self.target[i][0])**2+(y_prev[i]-self.target[i][1])**2)

        


        # 1. Move robot with the action input for time_step
        while (record_time_step < time_step):
            for i in range(self.num_robots):
                self.pub_tb3[i].publish(twists[i])

            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time
        
        # 2. Read the position and angle of robot
        model_state = self.pose_ig.get_msg()
        self.model_state = model_state

        x, y, theta, idx = self.posAngle(model_state)
        self.x_prev = x
        self.y_prev = y

        # 3. Calculate the distance & angle difference to goal \
        distance_to_goals = [None]*self.num_robots
        global_angle = [None]*self.num_robots
        #print("x : {}, y: {}".format(x,y))
        for i in range(self.num_robots):
            distance_to_goals[i] = sqrt((x[i]-self.target[i][0])**2+(y[i]-self.target[i][1])**2)
            global_angle[i] = atan2(self.target[i][1] - y[i], self.target[i][0] - x[i])

        theta = self.angle0To360(theta)
        global_angle = self.angle0To360(global_angle)
        angle_diff = [a_i - b_i for a_i, b_i in zip(global_angle, theta)]
        angle_diff = self.anglepiTopi(angle_diff)

        # Calculate distance between robots & weight update conditions
        distance_btw_robots = np.zeros([self.num_robots, self.num_robots])
        #self.partnerIDs_prev = copy.copy(self.partnerIDs)


        for i in range(self.num_robots):
            new_partner = False
            for j in range(self.num_robots):
                if j != i:
                    distance_btw_robots[i][j] = sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) # Python
                    if self.weight_averaging == True:
                        if distance_btw_robots[i][j] > 0 and distance_btw_robots[i][j] <= 1.5:
                            self.partnerIDs[i] = j
                            new_partner = True
                            # Make update condition only it is previously 
                            if self.toBeUpdated[i] != 1 and self.partnerIDs[i] != self.partnerIDs_prev[i]:
                                #print("Robot {} and {} met together, but did not share weights.".format(i, j))
                                if random.random() <= self.updateProb:
                                    print("Robot {} and {} shared weights.".format(i, j))
                                    self.toBeUpdated[i] = 1
                                    # How can I prevent double averaging? 
                                    self.toBeUpdated[j] = 1
            if new_partner == False and self.weight_averaging == True:
                self.partnerIDs[i] = None

        print("------------")
        #print("distance_btw_robots: {}".format(distance_btw_robots))
        #print("] ] ]")
        #print("tBD: {}".format(self.toBeUpdated))
        #print("] ] ]")
        #print("partner IDs: {}".format(self.partnerIDs))
        

        # 4. Read LaserSensor (state) from the robot's position

        state = []
        scan = [None]*self.num_robots
        collision = [None]*self.num_robots
        for i in range(self.num_robots):
            scan[i] = self.laser_ig[i].get_msg()
            laser, collision[i] = self.getLaser(scan[i])
            state.append(laser)
        #print("lasers: {}".format(state))
        

        
        # Concatenating the state array
        state_arr = np.asarray(state).reshape(self.num_robots, 24) # I have to debug this. 
        state_arr = np.hstack((state_arr, np.asarray(distance_to_goals).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(linear_x).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(angular_z).reshape(self.num_robots,1)))
        state_arr = np.hstack((state_arr, np.asarray(angle_diff).reshape(self.num_robots,1)))

        # 5. State reshape
        states = state_arr.reshape(self.num_robots, self.state_num)

        
        # 6. Reward assignment

        ## 6.0. Initialisation of rewards
        distance_rewards = [0.0]*self.num_robots
        collision_rewards = [0.0]*self.num_robots
        goal_rewards = [0.0]*self.num_robots
        angular_punish_rewards = [0.0]*self.num_robots
        linear_punish_rewards = [0.0]*self.num_robots
        time_rewards = [0.0]*self.num_robots       

        ## 6.1. Distance Reward
        goal_progress = [a - b for a, b in zip(distance_to_goals_prv, distance_to_goals)]

        time_step_factor = 4/time_step
        for i in range(self.num_robots):
            if abs(goal_progress[i]) < 0.1:
                if goal_progress[i] >= 0:
                        distance_rewards[i] = goal_progress[i] * 1.2
                else:
                        distance_rewards[i] = goal_progress[i]
            else:
                distance_rewards[i] = 0.0
            distance_rewards[i] *= time_step_factor
        
        self.just_reset == False
        
        
        ## 6.3. Goal reward
        ### Reset condition is activated when both two robots have arrived their goals 
        ### Arrived robots stop and waiting
        for i in range(self.num_robots):
            if distance_to_goals[i] <= 0.5:
                goal_rewards[i] = 100.0
                dones[i] = True
                self.isEverSuccess[i] = True
                self.reset(model_state, id_bots=idx[i])

            
        

        ## 6.4. Angular speed penalty
        for i in range(self.num_robots):
            if abs(angular_z[i])>0.8:
                angular_punish_rewards[i] = -1
                if dones[i] == True:
                    angular_punish_rewards[i] = 0.0
        
        ## 6.5. Linear speed penalty
        for i in range(self.num_robots):
            if linear_x[i] < 0.2:
                linear_punish_rewards[i] = -1.0
        for i in range(self.num_robots):
            if dones[i] == True:
                linear_punish_rewards[i] = 0.0
        ## 6.6. Collision penalty
        #   if it collides to walls, it gets penalty, sets done to true, and reset
        #   it needs to be rewritten to really detect collision
        

        for i in range(self.num_robots):
            if collision[i] == True:
                print("Collision! Robot: {}".format(i))
                collision_rewards[i] = -100.0
                dones[i] = True
                self.reset(model_state, id_bots=idx[i])
        
        ## 6.7. Time penalty
        #  constant time penalty for faster completion of episode
        for i in range(self.num_robots):
            time_rewards[i] = 0.0 # 20201217 I nullified the time_rewards
            if dones[i] == True:
                time_rewards[i] = 0.0

            
    
        test_time = time.time()
        
        
        ## 7.4. If all the robots are done with tasks, reset
        if all(flag == True for flag in dones) == True:
            self.reset(model_state, id_bots=999)
            for i in range(self.num_robots):
                dones[i] = False

        self.dones = dones
        

        rewards = [a+b+c+d+e+f for a, b, c, d, e, f in zip(distance_rewards, goal_rewards, angular_punish_rewards, linear_punish_rewards, collision_rewards, time_rewards)]
        test_time2 = time.time()
        rewards = np.asarray(rewards).reshape(self.num_robots)
        infos = [{"episode": {"l": self.ep_len_counter, "r": rewards}}]
        self.ep_len_counter = self.ep_len_counter + 1
        print("-------------------")
        #print("Infos: {}".format(infos))
        #print("Linear: {}, Angular: {}".format(linear_x, angular_z))
        #print("Reward: {}".format(rewards))
        print("Time for step: {}".format(time.time()-start_time))
        #print("Time diff: {}".format(test_time-test_time2))
        

        #print("state: {}, action:{}, reward: {}, done:{}, info: {}".format(state, action, reward, done, info))
        return range(0, self.num_robots), states, rewards, dones, infos, is_stops, self.isEverSuccess, self.toBeUpdated, self.partnerIDs

if __name__ == '__main__':
    try:
        env = Env()
    except rospy.ROSInterruptException:
        pass