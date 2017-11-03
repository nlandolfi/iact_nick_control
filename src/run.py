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
    return np.asarray(a)*(math.pi/180.0)

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
    robot.SetDOFValues(np.append(q, [0, 0, 0]))
    return openrave_utils.robotToCartesian(robot)[6]

def direction(x, y):
    return (y - x)/np.linalg.norm(y - x + 1e-12)

def normalize(beliefs):
    return np.asarray(beliefs)/np.sum(np.asarray(beliefs))

def update(robot, p, q, goals, beliefs):
    goals = [xyz(robot, goal) for goal in goals]
    goals_dirs = [direction(xyz(robot, p), goal) for goal in goals]
    interaction_dir = direction(xyz(robot, p), xyz(robot, q))
    print("INTERACTION DIR %s" % (interaction_dir))
    beliefs = np.array([b*gaussian(goal_dir, 1e-2)(interaction_dir) for (b, goal_dir) in zip(beliefs, goals_dirs)])
    return normalize(beliefs)

def run(start, goals, prior):
    print("STARTING WITH")
    print("START: %s" % (start))
    print("GOALS: %s" % (goals))
    raw_input()
    shared = {
        'p':          None,
        'q':          None,
        'state':      STATE_STARTING,
        'start_time': None,
        'belief':     prior,
        'plan':       None,
    }
    model_filename = 'jaco_dynamics'
    env, robot = openrave_utils.initialize(model_filename)
    p = planner.planner(env, 4)

    vel_pub = jaco.init()

    jaco.start_admittance_mode()

    def target():
        if shared['state'] == STATE_STARTING:
            return start
        elif shared['state'] == STATE_RUNNING:
            if shared['start_time'] is None:
                raise Exception("start_time is none!")

            return interpolate(shared['plan'], time.time() - shared["start_time"], 0, 10, 0.5)

        elif shared['state'] == STATE_STOPPED:
            return estimate(goals, shared['belief'])
        else:
            raise Exception("unknown state")

    def angles(msg):
        """
            Callback for the joint angles of the Jaco Arm.

            Sets the current configuration, q, and updates the state.
        """
        shared['p'] = shared['q']
        shared['q'] = jaco.parse_joint_angles(msg)
        robot.SetDOFValues(np.append(shared['q'], [0, 0, 0]))

        if shared['state'] == STATE_STARTING:
            if all_close(start, shared['q']):
                print("STARTING")
                shared['state'] = STATE_RUNNING
                shared['start_time'] = time.time()
                shared['plan'] = p(shared['q'], estimate(goals, shared['belief']))
        elif shared['state'] == STATE_RUNNING:
            if all_close(estimate(goals, shared['belief']), shared['q']):
                shared['state'] = STATE_STOPPED
        #elif STATE == STATE_STOPPED:
        else:
            raise Exception("unknown state")
    jaco.subscribe_joint_angles(angles)

    def torques(msg):
        """
            Callback for the joint torques of the Jaco Arm.
        """
        if not shared['state'] == STATE_RUNNING:
            return

        torques, interaction = jaco.interpret_torques(jaco.parse_torques(msg))
        if interaction:
            print("INTERACTION")
            shared['belief'] = update(robot, shared['p'], shared['q'], goals, shared['belief'])
            print("beliefs -> %s" % (shared['belief']))
            shared['plan'] = p(shared['q'], estimate(goals, shared['belief']))
    jaco.subscribe_joint_torques(torques)

    # P, I, D gains
    p_gain = 50.0
    i_gain = 0.0
    d_gain = 20.0
    P = p_gain*np.eye(7)
    I = i_gain*np.eye(7)
    D = d_gain*np.eye(7)
    controller = pid.PID(P,I,D,0,0)

    ticker = rospy.Rate(100)
    print "----------------------------------"
    print "Moving robot, press ENTER to quit:"
    while not rospy.is_shutdown():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = raw_input()
            break

        if shared['q'] is not None:
            #print([c*180./math.pi for c in shared['q']])
            cmd = cap(-controller.update_PID(error(target(), shared['q']).reshape([7, 1])))
            vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(cmd))
        ticker.sleep()
    print "----------------------------------"
    jaco.stop_admittance_mode()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "ERROR: need an experiment to run"
    else:
        run(*parse(load(sys.argv[1])))
