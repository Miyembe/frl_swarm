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

import scipy.io as sio
import os
HOME = os.environ['HOME']

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

    def __init__(self, algo_name):

        # Settings
        self.num_robots = 1
        self.num_experiments = 100
        self.save_path = HOME + '/catkin_ws/src/frl_swarm/src/log/evaluation_result/'
        self.algo_name = algo_name


        # Node initialisation
        self.node = rospy.init_node('eval_env', anonymous=True)
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
        self.robot_pos = [None]*self.num_robots
        self.robot_quat = [None]*self.num_robots
        self.target_pos = [None]*self.num_robots
        self.target_quat = [None]*self.num_robots
        #for i in range(self.num_robots):
            #self.robot_pos[i], self.robot_quat[i], self.target_pos[i], self.target_quat[i] = self.coordinates.circular_reset(i)
        
        self.x_init = [0.0]#[0.0, 2.0] # [0.0,4.0]
        self.y_init = [0.0]  #[0.0, -2.0]  # [0.0,0.0]
        self.x_prev = [0.0]#[0.0, 2.0] # [0.0,4.0]
        self.y_prev = [0.0]  #[0.0, -2.0]  # [0.0,0.0]
        self.x = [0.0]*self.num_robots
        self.y = [0.0]*self.num_robots
        self.theta = [0.0]*self.num_robots
        self.target_index = 0

        # Set target position
        self.target_list = [[5.0, 5.0], [-5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]]#[[4.0, 0.0], [2.0, 2.0]] # Two goal (crossing scenario) # [[4.0,0.0], [0.0,0.0]]
        self.angle_list = [(1.0/4.0)*pi, (3.0/4.0)*pi, -(3.0/4.0)*pi, -(1.0/4.0)*pi]
        print("self.angle_list: {}".format(self.angle_list))

        self.target = [[]]
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

        # File name
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.file_name = "{}_{}".format(self.algo_name, self.time_str)
        self.traj_name = "{}_traj".format(self.file_name)
        print(self.file_name)

        # Experiments
        self.isExpDone = False
   
        self.counter_step = 0
        self.counter_collision = 0
        self.counter_success = 0
        self.counter_timeout = 0
        self.arrival_time = []
        
        self.is_reset = False
        self.is_collided = False
        self.is_goal = 0
        self.is_timeout = False
        self.done = False

        self.is_traj = True

        # Log related

        self.log_timer = time.time()
        self.reset_timer = time.time()
        self.positions = []
        #for i in range(self.num_robots):
        #    self.positions.append([])
        print("positions: {}".format(self.positions))
        self.traj_eff = list()

    def reset(self, model_state = None, id_bots = 999):

        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset the positions of robots and targets
        '''
        # ========================================================================= #
	    #                          1. COUNTER UPDATE                                #
	    # ========================================================================= #
        
        #Increment Collision Counter
        if self.is_collided == True:
            print("Collision!")
            self.counter_collision += 1
            self.counter_step += 1

        # Increment Arrival Counter and store the arrival time
        if self.is_goal == True:
            print("Arrived goal!")
            self.counter_success += 1
            self.counter_step += 1
            arrived_timer = time.time()
            art = arrived_timer-self.reset_timer
            self.arrival_time.append(art)
            print("Episode time: %0.2f"%art)

            # Compute trajectory efficiency (how can I add outlier removal?)
            total_distance = 0.0
            pure_distance = 0.0
            print("self.positions: {}, shape:{}".format(self.positions, np.array(self.positions).shape))
            
            for j in range(len(self.positions)-1):
                distance_t = sqrt((self.positions[j+1][0][0] - self.positions[j][0][0])**2 + (self.positions[j+1][1][0] - self.positions[j][1][0])**2)
                if distance_t <= 0.5:
                    total_distance += distance_t
            pure_distance = sqrt((self.positions[0][0][0] - self.positions[-1][0][0])**2 + (self.positions[0][1][0] - self.positions[-1][1][0])**2)

            avg_distance_traj = np.average(total_distance)
            avg_distance_pure = np.average(pure_distance)
            traj_efficiency = pure_distance/total_distance
            print("Step: {}, avg_distance_traj: {}".format(self.counter_step, avg_distance_traj))
            #print("self.positions: {}".format(self.positions))
            #print("Total Distance: {}".format(total_distance))
            print("avg_distance_pure: {}, traj_efficiency: {}".format(avg_distance_pure, traj_efficiency))
            #print("distance_t: {}".format(distance_t))

            self.traj_eff.append(traj_efficiency)

        if self.is_timeout == True:
            self.counter_timeout += 1
            self.counter_step += 1
            print("Timeout!")

        # Reset the flags
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False

        # ========================================================================= #
	    #                          2. TARGET UPDATE                                 #
	    # ========================================================================= #
        

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

        self.target = [self.target_list[self.target_index]]
        print("Target: {}".format(self.target))
        print("self.angle_list: {}".format(self.angle_list))
        theta = self.angle_list[self.target_index]
        print("Theta: {}".format(theta))
        if self.target_index < 3:
            self.target_index += 1
        else: self.target_index = 0

        
        
        quat = quaternion_from_euler(0,0,theta)
        print("quat: {}".format(quat))
        
        for i in range(self.num_robots):

            # Robot Pose assignment
            robot_msgs[i] = ModelState()
            robot_msgs[i].model_name = 'tb3_{}'.format(i)
            robot_msgs[i].pose.position.x = self.x_init[i]
            robot_msgs[i].pose.position.y = self.y_init[i]
            robot_msgs[i].pose.position.z = 0.0#self.robot_pos[i][2]
            robot_msgs[i].pose.orientation.x = quat[0]#self.robot_quat[i][0]
            robot_msgs[i].pose.orientation.y = quat[1]#self.robot_quat[i][1]
            robot_msgs[i].pose.orientation.z = quat[2]#self.robot_quat[i][2]
            robot_msgs[i].pose.orientation.w = quat[3]#self.robot_quat[i][3]

            # Target Pose assignment
            target_msgs[i] = ModelState()
            target_msgs[i].model_name = 'target_{}'.format(5)
            target_msgs[i].pose.position.x = self.target[0][0]#self.target_pos[i][0]
            target_msgs[i].pose.position.y = self.target[0][1]#self.target_pos[i][1]
            target_msgs[i].pose.position.z = 0.1#self.target_pos[i][2]
            target_msgs[i].pose.orientation.x = quat[0]#self.target_quat[i][0]
            target_msgs[i].pose.orientation.y = quat[1]#self.target_quat[i][1]
            target_msgs[i].pose.orientation.z = quat[2]#self.target_quat[i][2]
            target_msgs[i].pose.orientation.w = quat[3]#self.target_quat[i][3]

        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        resp_robots = [None] * self.num_robots
        resp_targets = [None] * self.num_robots
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            for i in range(self.num_robots):
                if id_bots == tb3[i] or id_bots == 999:
                    resp_robots[i] = set_state(robot_msgs[i])
                    resp_targets[i] = set_state(target_msgs[i])
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

        # ========================================================================= #
	    #                                 4. LOGGING                                #
	    # ========================================================================= #

        if self.counter_step == 0:
            with open(self.save_path + '{}.csv'.format(self.file_name), mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Episode', 'Success Rate', 'Average Arrival time', 'std_at', 'Collision Rate', 'Timeout Rate', 'Trajectory Efficiency', 'std_te'])

        if self.counter_step != 0:
            if (self.counter_collision != 0 or self.counter_success != 0):
                succ_percentage = 100*self.counter_success/(self.counter_success+self.counter_collision+self.counter_timeout)
                col_percentage = 100*self.counter_collision/(self.counter_success+self.counter_collision+self.counter_timeout)
                tout_percentage = 100*self.counter_timeout/(self.counter_success+self.counter_collision+self.counter_timeout)
            else:
                succ_percentage = 0
                col_percentage = 0
                tout_percentage = 0
            print("Counter: {}".format(self.counter_step))
            print("Success Counter: {}".format(self.counter_success))
            print("Collision Counter: {}".format(self.counter_collision))
            print("Timeout Counter: {}".format(self.counter_timeout))
            print("Trajectory Efficiency: {}".format(self.traj_eff))
            print("AVG TE: {}".format(np.average(np.array(self.traj_eff))))

        if (self.counter_step % 1 == 0 and self.counter_step != 0):
            print("Success Rate: {}%".format(succ_percentage))

        if (self.counter_step % self.num_experiments == 0 and self.counter_step != 0):
            avg_comp = np.average(np.asarray(self.arrival_time))
            std_comp = np.std(np.asarray(self.arrival_time))
            avg_traj = np.average(np.asarray(self.traj_eff))
            std_traj = np.std(np.asarray(self.traj_eff))
            print("{} trials ended. Success rate: {}, average completion time: {}, Standard deviation: {}, Collision rate: {}, Timeout Rate: {}".format(self.counter_step, succ_percentage, avg_comp, std_comp, col_percentage, tout_percentage))
            with open(self.save_path + '{}.csv'.format(self.file_name), mode='a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['%i'%self.counter_step, '%0.2f'%succ_percentage, '%0.2f'%avg_comp, '%0.2f'%std_comp, '%0.2f'%col_percentage, '%0.2f'%tout_percentage, '%0.4f'%avg_traj, '%0.4f'%std_traj])
                print("Successfully Logged.")
            self.arrival_time = list()
            self.traj_eff = list()
            self.counter_collision = 0
            self.counter_success = 0
            self.counter_timeout = 0
            self.target_index = 0
        time.sleep(1)
        self.reset_timer = time.time()
        self.log_timer = time.time()
        self.positions = []
        

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
        min_range = 0.15
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
    def step(self, actions, time_step=0.1):
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

        # Check if the robots are terminated
        dones = self.dones
        is_stops = dones
        
        twists = [self.action_to_twist(action) for action in np.asarray(actions)]

        # rescaling the action
        for i in range(len(twists)):
            twists[i].linear.x = (twists[i].linear.x+1) * 1/2  # only forward motion
            twists[i].angular.z = twists[i].angular.z
            if is_stops[i] == True:
                twists[i] = Twist()
        
        linear_x = [i.linear.x for i in twists]
        angular_z = [i.angular.z for i in twists]
        

        # position of turtlebot before taking steps
        x_prev = self.x_prev
        y_prev = self.y_prev
        distance_to_goals_prv = [None]*self.num_robots
        for i in range(self.num_robots):
            distance_to_goals_prv[i] = sqrt((x_prev[i]-self.target[i][0])**2+(y_prev[i]-self.target[i][1])**2)


        step_time = time.time()
        episode_time = step_time - self.reset_timer
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


        # Log Positions
        if time.time() - self.log_timer > 0.1 and self.is_traj == True:
            #for i in range(self.num_robots):
                #with open(self.save_path + '{}.csv'.format(self.traj_name), mode='a') as csv_file:
                #        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #       csv_writer.writerow(['%0.1f'%reset_time, '%i'%i, '%0.2f'%x, '%0.2f'%y])
                #print("positions[i]: {}".format(self.positions[i]))
            self.positions.append([x,y])
            self.log_timer = time.time()

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
            if distance_to_goals[i] <= 0.6:
                if self.reset_timer > 3.0:
                    goal_rewards[i] = 100.0
                    dones[i] = True
                    self.is_goal = True
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
                self.is_collided = True
                self.reset(model_state, id_bots=idx[i])

        if episode_time > 60:
            self.is_timeout = True
            self.done = True
        
        ## 6.7. Time penalty
        #  constant time penalty for faster completion of episode
        for i in range(self.num_robots):
            time_rewards[i] = 0.0 # 20201217 I nullified the time_rewards
            if dones[i] == True:
                time_rewards[i] = 0.0

            
    
        test_time = time.time()
        
        
        # ##7.4. If all the robots are done with tasks, reset
        # if all(flag == True for flag in dones) == True:
        #     self.reset(model_state, id_bots=999)
        #     for i in range(self.num_robots):
        #         dones[i] = False

        self.dones = dones
        

        rewards = [a+b+c+d+e+f for a, b, c, d, e, f in zip(distance_rewards, goal_rewards, angular_punish_rewards, linear_punish_rewards, collision_rewards, time_rewards)]
        test_time2 = time.time()
        rewards = np.asarray(rewards).reshape(self.num_robots)
        infos = [{"episode": {"l": self.ep_len_counter, "r": rewards}}]
        self.ep_len_counter = self.ep_len_counter + 1
        #print("-------------------")
        #print("Infos: {}".format(infos))
        #print("Linear: {}, Angular: {}".format(linear_x, angular_z))
        #print("Reward: {}".format(rewards))
        #print("Time for step: {}".format(time.time()-start_time))
        #print("Time diff: {}".format(test_time-test_time2))
        

        #print("state: {}, action:{}, reward: {}, done:{}, info: {}".format(state, action, reward, done, info))
        return range(0, self.num_robots), states, rewards, dones, infos, is_stops, self.isEverSuccess

if __name__ == '__main__':
    try:
        env = Env()
    except rospy.ROSInterruptException:
        pass