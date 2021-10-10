import numpy as np
from math import *
from tf.transformations import quaternion_from_euler


class Coordinates():
    '''
    Class containing robot and target pose when reset for each env
    '''
    def __init__(self, num_robots, num_exp):
        self.num_robots = num_robots
        self.num_exp = num_exp

        # 20210427 There is a need for initialising first poses. - no I don't need this? it just needs to return 
        self.robot_pose = [None]*self.num_robots
        self.target_pose = [None]*self.num_robots
        self.reset_index = [0]*self.num_robots

        self.radius = 4.0

    def circular_reset(self, idx):
        '''
        Input  - (1) index for the robot-env set 
        Output - (1) Robot position & orientation (x, y, z) & quaternion (x, y, z, w)
                 (2) Target position & orientation (x, y, z) & quaternion (x, y, z, w)
                
        '''
        
        angle_target = self.reset_index[idx]*2*pi/self.num_exp     

        target_x = self.radius*cos(angle_target)
        target_y = self.radius*sin(angle_target)
        
        if self.reset_index[idx] < self.num_exp-1:
            self.reset_index[idx] += 1
        else:
            self.reset_index[idx] = 0

        robot_pos = [0.0, 0.0, 0.0]
        robot_quat = quaternion_from_euler(0, 0, angle_target)
        
        target_pos  = [target_x, target_y, 0.0]
        target_quat = [0.0, 0.0, -0.2, 0.0]

        return robot_pos, robot_quat, target_pos, target_quat

        

