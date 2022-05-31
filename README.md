# frl_swarm
Federated Deep Reinforcement Learning for swarm robotics.

This is a ros package for Federated Deep Reinforcement learning for swarm robotics project.

For running the experiment, please follow the below instruction

## 1. Launch a launch file. 
To open a Gazebo world and generate Turtlebot3 robots, a launch file needs to be launch file. Use following command: roslaunch swarm_frl LAUNCHFILE_NAME (LAUNCHFILE_NAME is a file name you want to launch, frl_6.launch is the one used for swarm_frl experiment)

## 2. Run a script. 
In this stage, you will run a script for swarm_frl. There are several scripts in swarm_frl/src. IACDDPG, SEACDDPG, SNACDDPG are the baseline algorithms and FLDDPG is the federated deep reinforcement learning algorithm. To run the script, please open another terminal and enter the following command: rosrun swarm_frl SCRIPT_NAME (SCRIPT_NAME is a file name of a script you want to run, FLDDPG.py is the one used for swarm_frl training and FLDDPG_eval is the one for evaluation of the trained model.)

## 3. Check the training progress
The training script offers tensorboard logging. You can find the tensorboard directory in log/FLDDPG/tf_board.
 
