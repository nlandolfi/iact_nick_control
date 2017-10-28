#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco
so that it maintains a fixed distance to a target. 
Author: Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys, select, os
import thread
import argparse
import actionlib
import time
import trajopt_planner
#import planner
import traj
import ros_utils
import experiment_utils

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32
from sympy import Point, Line

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

epsilon = 0.10							# epislon for when robot think it's at goal
MAX_CMD_TORQUE = 40.0					# max command robot can send
INTERACTION_TORQUE_THRESHOLD = 8.0		# threshold when interaction is measured 

#HUMAN_TASK = 0
#COFFEE_TASK = 1 
#TABLE_TASK = 2
#LAPTOP_TASK = 3

FAM_TASK = 1
EXP_TASK = 2

ZERO_FEEDBACK = 'A'
HAPTIC_FEEDBACK = 'B'

ONE_FEAT = 1					# experiment where human has to correct only one feature (out of three)
TWO_FEAT = 2					# experiment where human has to correc two features

ALL = "ALL" 							# updates all features
MAX = "MAX"								# updates only feature that changed the most
LIKELY = "LIKELY"						# updates the most likely feature 

class PIDVelJaco(object): 
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint velocities are computed as:
		
		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)
	where:
		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change
	
	Subscribes to: 
		/j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles
		/j2s7s300_driver/out/joint_torques	- Jaco sensed joint torques
	
	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities 
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		sim_flag 				  - flag for if in simulation or not
	"""

	def __init__(self, ID, task, methodType, demo, record, featMethod, numFeat):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# task type - experimental (table) or familiarization (cup)
		if task == "FAM" or task == "fam":
			self.task = FAM_TASK
		elif task == "EXP" or task == "exp":
			self.task = EXP_TASK

		# method type - A=IMPEDANCE, B=LEARNING
		self.methodType = methodType

		# can be ALL, MAX, or LIKELY
		self.featMethod = featMethod

		# can be ONE_FEAT or TWO_FEAT
		self.numFeat = numFeat

		# optimal demo mode
		if demo == "F" or demo == "f":
			self.demo = False
		elif demo == "T" or demo == "t":
			self.demo = True
		else:
			print "Oopse - it is unclear if you want demo mode. Turning demo mode off."
			self.demo = False

		# record experimental data mode 
		if record == "F" or record == "f":
			self.record = False
		elif record == "T" or record == "t":
			self.record = True
		else:
			print "Oopse - it is unclear if you want to record data. Not recording data."
			self.record = False

		# start admittance control mode
		self.start_admittance_mode()

		#setup the interaction mode
		self.interaction = False
		self.no_interaction_count = 0
		self.interaction_count = 0
		self.admittance_mode = False
		self.gravity_comp_mode = False

		# ---- Trajectory Setup ---- #

		# total time for trajectory
		self.T = 20.0 	#TODO THIS IS EXPERIMENTAL - used to be 15.0

		# initialize trajectory weights

		# TODO THIS IS EXPERIMENTAL - CHANGE BACK TO ALL 0
		self.weights = [0.0, 0.0] 

		# if in demo mode, then set the weights to be optimal
		if self.demo:
			if self.task == FAM_TASK:
				self.weights = [1.0,0.0]
			elif self.task == EXP_TASK:
				if self.numFeat == ONE_FEAT:
					self.weights = [0.0,1.0]
				elif self.numFeat == TWO_FEAT:
					self.weights = [1.0,1.0]

		# initialize start/goal based on task 
		if self.task == FAM_TASK:
			pick = pick_shelf
		else:
			if self.numFeat == TWO_FEAT:
				# if two features are wrong, initialize the starting config badly (tilted cup)
				pick = pick_basic
				pick[-1] = 200.0
			else:
				pick = pick_basic 

		if self.task == FAM_TASK:
			place = place_higher
		else:
			place = place_lower
		
		start = np.array(pick)*(math.pi/180.0)
		goal = np.array(place)*(math.pi/180.0)
		self.start = start
		self.goal = goal

		# TODO THIS IS EXPERIMENTAL
		self.place_pose = place_pose
		
		self.curr_pos = None

		# create the trajopt planner and plan from start to goal

		# TODO THIS IS EXPERIMENTAL
		self.planner = trajopt_planner.Planner(self.task, self.demo, self.featMethod, self.numFeat)
		#self.planner = planner.Planner(self.task, self.demo, self.featMethod)
		
		# stores the current trajectory we are tracking, produced by planner
		self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=None)		
		#self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, goal_pose=self.place_pose)
		print "original traj: " + str(self.traj)

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# keeps running time since beginning of program execution
		self.process_start_T = time.time() 
		# keeps running time since beginning of path
		self.path_start_T = None 

		# ----- Controller Setup ----- #

		# stores maximum COMMANDED joint torques		
		self.max_cmd = MAX_CMD_TORQUE*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# P, I, D gains 
		p_gain = 50.0
		i_gain = 0.0
		d_gain = 20.0
		self.P = p_gain*np.eye(7)
		self.I = i_gain*np.eye(7)
		self.D = d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		# ---- Experimental Utils ---- #

		self.expUtil = experiment_utils.ExperimentUtils()
		# store original trajectory

		# TODO THIS IS EXPERIMENTAL
		original_traj = self.planner.get_waypts_plan()	
		#original_traj = self.traj.get_waypts_plan()

		self.expUtil.update_original_traj(original_traj)

		# ---- ROS Setup ---- #

		rospy.init_node("pid_trajopt")

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

		# publish to ROS at 100hz
		r = rospy.Rate(100) 

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"
		
		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()
		
		print "----------------------------------"

		# plot weight update over time
		#self.planner.plot_feature_update()
		#self.planner.plot_weight_update()

		# save experimental data (only if experiment started)
		if self.record and self.reached_start:
			print "Saving experimental data to file..."
			if featMethod == ALL:
				method = "A"
			elif featMethod == MAX:
				method = "B"
				
			weights_filename = "weights" + str(ID) + str(numFeat) + method
			force_filename = "force" + str(ID) + str(numFeat) + method		
			tracked_filename = "tracked" + str(ID) + str(numFeat) + method
			original_filename = "original" + str(ID) + str(numFeat) + method
			deformed_filename = "deformed" + str(ID) + str(numFeat) + method		
			self.expUtil.pickle_force(force_filename)	
			self.expUtil.pickle_tracked_traj(tracked_filename)
			self.expUtil.pickle_original_traj(original_filename)
			self.expUtil.pickle_deformed_traj(deformed_filename)
			self.expUtil.pickle_weights(weights_filename)

		# end admittance control mode
		self.stop_admittance_mode()

	def start_admittance_mode(self):
		"""
		Switches Jaco to admittance-control mode using ROS services
		"""
		service_address = prefix+'/in/start_force_control'	
		rospy.wait_for_service(service_address)
		try:
			startForceControl = rospy.ServiceProxy(service_address, Start)
			startForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def stop_admittance_mode(self):
		"""
		Switches Jaco to position-control mode using ROS services
		"""
		service_address = prefix+'/in/stop_force_control'	
		rospy.wait_for_service(service_address)
		try:
			stopForceControl = rospy.ServiceProxy(service_address, Stop)
			stopForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		return -self.controller.update_PID(error)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		#print "Current torque: " + str(torque_curr)
		self.interaction = False
		for i in range(7):
			THRESHOLD = INTERACTION_TORQUE_THRESHOLD
			if self.reached_start and i == 3:
				THRESHOLD = 2.5
			if self.reached_start and i > 3:
				THRESHOLD = 1.0
			if np.fabs(torque_curr[i][0]) > THRESHOLD:
				self.interaction = True
			else:
				# zero out torques below threshold for cleanliness
				torque_curr[i][0] = 0.0
		#print "Cleaned torque: " + str(torque_curr)

		if self.path_start_T is not None: # make sure we got to start before allowing interaction!
			if self.interaction:
				timestamp = time.time() - self.path_start_T
				self.expUtil.update_tauH(timestamp, torque_curr)
				self.interaction_count += 1
				self.no_interaction_count = 0
			else:
				self.interaction_count = 0
				self.no_interaction_count += 1


		"""
		if self.methodType == ZERO_FEEDBACK:
			if self.interaction_count > 0 and not self.gravity_comp_mode:
				self.controller.set_gains(0.0*self.P,0.0*self.I,0.0*self.D, 0, 0)
				self.gravity_comp_mode = True
				print "starting gravity compensation mode..."
			if self.no_interaction_count > 7 and self.gravity_comp_mode:
				self.controller.set_gains(self.P,self.I,self.D, 0, 0)
				self.gravity_comp_mode = False
				print "stopping gravity compensation mode..."
		"""

		"""
		if self.path_start_T is not None:
			# for tracing weight updates
			if self.planner.weight_update is None:
				self.planner.weight_update = np.array([self.planner.weights])
				self.planner.update_time = np.array([time.time() - self.path_start_T])
			else:
				self.planner.weight_update = np.append(self.planner.weight_update,np.array([self.planner.weights]),0)
				self.planner.update_time = np.append(self.planner.update_time,np.array([time.time() - self.path_start_T]))

			# more updates of tracking feature updates
			if self.planner.feature_update is None and self.planner.curr_features is not None:
				diff = self.planner.curr_features - self.planner.prev_features
				self.planner.feature_update = np.array([diff])
				self.planner.update_time2 = np.array([time.time() - self.path_start_T])
			elif self.planner.curr_features is not None:
				diff = self.planner.curr_features - self.planner.prev_features
				self.planner.feature_update = np.append(self.planner.feature_update,np.array([diff]),0)
				self.planner.update_time2 = np.append(self.planner.update_time2,np.array([time.time() - self.path_start_T]))
		"""

		# if experienced large enough interaction force, then deform traj
		if self.interaction:
			#print "--- INTERACTION ---"
			#print "u_h: " + str(torque_curr)
			if self.reached_start and not self.reached_goal:
				timestamp = time.time() - self.path_start_T
				self.expUtil.update_tauH(timestamp, torque_curr)

				"""
				# compute the optimal difference in configuration space
				deltaQDes = self.planner.computeDeltaQDes(self.start, self.goal, self.T, self.traj, timestamp)
				print "deltaQDes: " + str(deltaQDes)

				# compute argmax of dot product between actual_deltaQ and  des_deltaQ
				inner_prod = [0.0]*len(self.weights)
				for i in range(len(self.weights)):
					des_deltaQ = np.array(deltaQDes[i])

					# compute actual delta q at current time
					curr_q = self.curr_pos
					if self.traj is not None:
					desired_q = self.traj.interpolate(timestamp)
					actual_deltaQ = desired_q - curr_q
				
					#print "des_deltaQ: " + str(des_deltaQ)
					#print "actual_deltaQ: " + str(actual_deltaQ)
					# compute dot product between actual and desired changes in configuration
					inner_prod[i] = np.dot(actual_deltaQ.T,des_deltaQ)[0][0]
				print "(abs) inner prod: " + str(np.fabs(inner_prod))
				feat_idx_opt = np.argmax(np.fabs(inner_prod))
				print "feat_idx_opt: " + str(feat_idx_opt)
				"""

				# TODO THIS IS EXPERIMENTAL
				self.weights = self.planner.learnWeights(torque_curr)
				#self.weights = self.planner.learnWeights(torque_curr, self.traj) #feat_idx=feat_idx_opt
				#print "here are my new weights: ", self.weights

				print "in joint torques callback: going to plan..."
				# TODO THIS IS EXPERIMENTAL
				#self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5)				
				self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=self.traj)				
				#self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=self.traj, goal_pose=self.place_pose)
				print "in joint torques callback: finished planning -- self.traj = " + str(self.traj)

				# update the experimental data with new weights
				timestamp = time.time() - self.path_start_T
				self.expUtil.update_weights(timestamp, self.weights)

				# store deformed trajectory
				# TODO THIS IS EXPERIMENTAL
				deformed_traj = self.planner.get_waypts_plan()
				#deformed_traj = self.traj.get_waypts_plan()
				self.expUtil.update_deformed_traj(deformed_traj)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
	
		# read the current joint angles from the robot
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		self.curr_pos = self.curr_pos*(math.pi/180.0)	

		# update the OpenRAVE simulation 
		#self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(self.curr_pos)

		# update the experiment utils executed trajectory tracker
		if self.reached_start and not self.reached_goal:
			# update the experimental data with new position
			timestamp = time.time() - self.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

		# update cmd from PID based on current position
		self.cmd = self.PID_control(self.curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:

			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_start = np.fabs(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()

				# set start time and the original weights as experimental data
				self.expUtil.set_startT(self.path_start_T)
				timestamp = time.time() - self.path_start_T
				#self.expUtil.update_weights(timestamp, self.weights)
			else:
				#print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:
			#print "REACHED START --> EXECUTING PATH"

			t = time.time() - self.path_start_T
			#print "t: " + str(t)

			# get next target position from position along trajectory

			# TODO THIS IS EXPERIMENTAL
			self.target_pos = self.planner.interpolate(t + 0.1)
			#self.target_pos = self.traj.interpolate(t + 0.1)
			
			# check if the arm reached the goal, and restart path
			if not self.reached_goal:
			
				dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)			
				dist_from_goal = np.fabs(dist_from_goal)

				# check if every joint is close enough to goal configuration
				close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
			
				# if all joints are close enough, robot is at goal
				is_at_goal = all(close_to_goal)
			
				if is_at_goal:
					self.reached_goal = True
			else:
				print "REACHED GOAL! Holding position at goal."
				self.target_pos = self.goal_pos
				# TODO: this should only set it once!
				self.expUtil.set_endT(time.time())

if __name__ == '__main__':
	#if len(sys.argv) < 7:
	#	print "ERROR: Not enough arguments. Specify ID, task, methodType, demo, record, featMethod, numFeat"
	#else:	
	ID = int(sys.argv[1])
	task = sys.argv[2]
	methodType = 'B' #sys.argv[3]
	demo = sys.argv[4]
	record = sys.argv[5]
	featMethod = sys.argv[6]
	numFeat = int(sys.argv[7])

	PIDVelJaco(ID,task,methodType,demo,record,featMethod,numFeat)


