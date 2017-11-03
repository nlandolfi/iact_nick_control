import json

import numpy as np
import openrave_utils
import trajoptpy

DEFAULT_MODEL = 'jaco_dynamics'

def trajopt_request(num_waypts_plan, goal, init_waypts):
    """
        normal vals -> a dictionary request
    """
    request = {
        "basic_info": {
            "n_steps": num_waypts_plan,
            "manip" : "j2s7s300",
            "max_iter" : 40
        },
        "costs": [
        {
            "type": "joint_vel",
            "params": {"coeffs": [1.0]}
        }
        ],
        "constraints": [
        {
            "type": "joint",
            "params": {"vals": goal.tolist()}
        }
        ],
        "init_info": {
            "type": "given_traj",
            "data": init_waypts.tolist()
        }
    }
    return request

def trajopt_problem(num_waypts_plan, goal, init_waypts, env):
    """
        standard vals -> trajopt problem"
    """
    return trajoptpy.ConstructProblem(json.dumps(trajopt_request(num_waypts_plan, goal, init_waypts)), env)

def line_start(start, goal, num):
    """
        linear between start and goal.
    """
    wpts = np.zeros((num, 7))
    for count in range(num):
        wpts[count, :] = start[:7] + count/(num - 1.0)*(goal[:7] - start[:7])
    return wpts

def trajopt_traject(num_waypts_plan, start, goal, env, warm_start=None, additional_config=[]):
    """
        Get a trajectory from trajopt with the stuff
    """
    if warm_start is None:
        init = line_start(start, goal, num_waypts_plan)
    else:
        init = warm_start

    prob = trajopt_problem(num_waypts_plan, goal, init, env)

    for f in additional_config:
        f(prob)

    result = trajoptpy.OptimizeProblem(prob)

    return result.GetTraj()

def pad(dofs):
    """
        if only 7 dofs, set fingers to zero.
    """
    l = dofs.shape[0]
    if l == 10:
        return dofs
    elif l == 7:
        withfingers = np.zeros(10)
        withfingers[:7] = dofs
        return withfingers
    else:
        raise Exception("malformed dofs shape")

def planner(env, num_waypts_plan):
    """
        curry a planner with an env and a waypoints number
    """
    def plan(start, goal):
        return trajopt_traject(num_waypts_plan, np.asarray(start), np.asarray(goal), env)

    return plan
