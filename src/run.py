import json
import math
import os
import select
import sys
import time

import numpy as np
import jaco
import openrave_utils
import pid
import planner
import ros_utils
import rospy
import matplotlib.pyplot as plt

from geometry_msgs.msg import Vector3

def load(filename):
    """
        Load a JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def degreesToRadians(a):
    """
        Degrees to radians conversion.
    """
    temp = np.asarray(a)*(math.pi/180.0)
    temp[2] += np.pi
    return temp

def parse(exp):
    """
        Parses experiment JSON. see ./experiments/*.json

        exp is a dictionary, normally from load.
    """

    if "start" not in exp:
        raise Exception("start position required!")
    start = exp["start"]
    start = degreesToRadians(start)

    if "goals" not in exp:
        raise Exception("end position required!")
    goals = exp["goals"]
    if len(goals) < 1:
        raise Exception("need at least one goal")
    goals = [degreesToRadians(goal) for goal in goals]

    if "prior" not in exp: # default to uniform
        prior = [1./len(goals) for g in goals]
    else:
        prior = exp["prior"]

    if "observation" not in exp:
        raise Exception("need observation model")

    obs = exp["observation"]
    if "name" not in obs:
        raise Exception("observation model needs name")

    return (start, goals, prior)

STATE_STARTING = 0
STATE_RUNNING  = 1
STATE_STOPPED  = 2

# error for pid between target pos and actual
def error(target, actual):
    return -((target - actual + math.pi)%(2*math.pi) - math.pi)

MAX_CMD_TORQUE = 40.0
# max joint torques
def cap(cmd, maxes=MAX_CMD_TORQUE*np.eye(7)):
    for i in range(7):
        if cmd[i][i] > maxes[i][i]:
            cmd[i][i] = maxes[i][i]
        if cmd[i][i] < -maxes[i][i]:
            cmd[i][i] = -maxes[i][i]
    return cmd

def estimate(goals, belief):
    x = np.array([goal*belief for (goal, belief) in zip(goals, belief)])
    return np.sum(x, axis=0)

# andreas implementation of all_close
def all_close(target, actual, epsilon=0.1):
    dist = -((actual - target + math.pi)%(2*math.pi) - math.pi)
    dist = np.fabs(dist)

    # check if every joint is close enough to target configuration
    return np.all([dist[i] < epsilon for i in range(7)])

def interpolate(plan, deltaT, start, end, step=0.5):
    sizeOfPlan = plan.shape[0]
    sizeOfTraj = int(math.ceil((end-start)/step)) + 1
    trajoptStep = (end-start)/sizeOfPlan

    if deltaT >= end - trajoptStep:
        return plan[sizeOfPlan - 1]

    i = int(deltaT/trajoptStep)
    u, v = plan[i], plan[i+1]
    ti, tf = i * trajoptStep, (i+1)*trajoptStep
    return (v - u)*((deltaT - ti)/(tf - ti)) + u

def gaussian(mu, var):
    cov = var * np.eye(mu.shape[0])
    return lambda x: (1./np.sqrt(2*math.pi*np.linalg.det(cov))) * np.exp(
            -(1./2.) * np.dot(np.dot((x - mu), np.linalg.inv(cov)), (x - mu))
            )

def xyz(robot, q):
    current_config = robot.GetActiveDOFValues()
    robot.SetDOFValues(np.append(q, [0, 0, 0]))
    coords = openrave_utils.robotToCartesian(robot)[6]
    robot.SetActiveDOFValues(current_config)
    return coords

def direction(x, y):
    return (y - x)/np.linalg.norm(y - x + 1e-12)

def normalize(beliefs):
    return np.asarray(beliefs)/np.sum(np.asarray(beliefs))

def update(robot, p, q, goals, beliefs):
    goals = [xyz(robot, goal) for goal in goals]
    goals_dirs = [direction(xyz(robot, p), goal) for goal in goals]
    interaction_dir = direction(xyz(robot, p), xyz(robot, q))
    print("INTERACTION DIR %s" % (interaction_dir))
    beliefs = np.array([b*gaussian(goal_dir, 1)(interaction_dir) for (b, goal_dir) in zip(beliefs, goals_dirs)])
    return normalize(beliefs)

def run(start, goals, prior):
    print("STARTING WITH")
    print("START: %s" % (start))
    print("GOALS: %s" % (goals))
    shared = {
        'p':          None,
        'q':          None,
        'state':      STATE_STARTING,
        'start_time': None,
        'latest_plan_time': None,
        'belief':     prior,
        'plan':       None,
    }
    model_filename = 'jaco_dynamics'
    env, robot = openrave_utils.initialize(model_filename)
    openrave_utils.plotTable(env)
    raw_input()
    p = planner.planner(env, 4)

    def target():
        if shared['state'] == STATE_STARTING:
            return start
        elif shared['state'] == STATE_RUNNING:
            if shared['start_time'] is None:
                raise Exception("start_time is none!")

            current_time = time.time()
            time_since_plan = current_time - shared['latest_plan_time']
            remaining_duration = 10 - (current_time - shared['start_time'])
            return interpolate(shared['plan'], time_since_plan, 0, remaining_duration, 0.5)

        elif shared['state'] == STATE_STOPPED:
            return estimate(goals, shared['belief'])
        else:
            raise Exception("unknown state")

    def angles(current):
        """
            Callback for the joint angles of the Jaco Arm.

            Sets the current configuration, q, and updates the state.
        """
        shared['p'] = shared['q']
        shared['q'] = current

        if shared['state'] == STATE_STARTING:
            if all_close(start, shared['q']):
                print("STARTING")
                shared['state'] = STATE_RUNNING
                shared['start_time'] = time.time()
                shared['latest_plan_time'] = shared['start_time']
                shared['plan'] = p(robot.GetActiveDOFValues()[:7], estimate(goals, shared['belief']))
        elif shared['state'] == STATE_RUNNING:
            if np.any([all_close(shared['q'], goal, epsilon=0.01) for goal in goals]):
#            if all_close(estimate(goals, shared['belief']), shared['q'], epsilon=0.01):
                print('COMPLETED.')
                print(np.linalg.norm(robot.GetActiveDOFValues()[:7] - goals[0]))
                shared['state'] = STATE_STOPPED
        elif shared['state'] == STATE_STOPPED:
            pass
        else:
            raise Exception('unknown state') 

    def perturbations(msg):
        """
            Callback for the joint perturbations of the Jaco Arm.
        """
        if shared['state'] != STATE_RUNNING:
            return

        perturbation = .05 * np.array([msg.x, msg.y, msg.z])
        J = robot.arm.CalculateJacobian()
        Jinv = np.linalg.pinv(J)
        delta_q = Jinv.dot(perturbation)
        #shared['q'] = shared['p'] + delta_q
        #robot.SetActiveDOFValues(np.append(shared['q'], [0, 0, 0]))
        with robot.GetEnv():
            current_config = robot.GetActiveDOFValues()[:7]
            shared['belief'] = update(robot, current_config, current_config + delta_q, goals, shared['belief'])
            print("beliefs -> %s" % (shared['belief']))
            shared['plan'] = p(current_config, estimate(goals, shared['belief']))
            shared['latest_plan_time'] = time.time()
    rospy.Subscriber('jaco_perturbations', Vector3, perturbations)

    print "----------------------------------"
    print "Moving robot, press ENTER to quit:"
    rospy.init_node("pid_trajopt")
    ticker = rospy.Rate(30)
    initial_ee = robot.arm.GetEndEffectorTransform()[:3,-1]
    goalA = xyz(robot, goals[0])
    goalB = xyz(robot, goals[1])
    openrave_utils.plotMug(env)
    mug = env.GetKinBody('mug')
    T = mug.GetTransform()
    T[:3,-1] = goalA
    mug.SetTransform(T)

    barplot = plt.bar([0, 1], [1., 1.])
    # plt.ion()
    plt.show(block=False)

    while not rospy.is_shutdown():
        barplot[0].set_height(shared['belief'][0])
        barplot[1].set_height(shared['belief'][1])
        plt.pause(.00001)
        plt.draw()
        angles(robot.GetActiveDOFValues()[:7])
        with robot.GetEnv():
            robot.SetActiveDOFValues(np.append(target(), [0,0,0]))
        ticker.sleep()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "ERROR: need an experiment to run"
    else:
        run(*parse(load(sys.argv[1])))
