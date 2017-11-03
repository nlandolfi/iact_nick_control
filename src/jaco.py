import math

import kinova_msgs.msg
import kinova_msgs.srv as ks
import numpy as np
import rospy

ROS_PREFIX = 'j2s7s300_driver'
INTERACTION_TORQUE_THRESHOLD = 8.0

def init():
    rospy.init_node("pid_trajopt")

    return rospy.Publisher(ROS_PREFIX + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

def start_admittance_mode():
    """
    Switches Jaco to admittance-control mode using ROS services
    """
    service_address = ROS_PREFIX+'/in/start_force_control'
    rospy.wait_for_service(service_address)
    try:
        startForceControl = rospy.ServiceProxy(service_address, ks.Start)
        startForceControl()
        return None
    except rospy.ServiceException, e:
        return "Service call failed: %s"%e

def stop_admittance_mode():
    """
    Switches Jaco to position-control mode using ROS services
    """
    service_address = ROS_PREFIX+'/in/stop_force_control'
    rospy.wait_for_service(service_address)
    try:
        stopForceControl = rospy.ServiceProxy(service_address, ks.Stop)
        stopForceControl()
        return None
    except rospy.ServiceException, e:
        return "Service call failed: %s"%e

def subscribe_joint_angles(func):
    rospy.Subscriber(ROS_PREFIX + '/out/joint_angles', kinova_msgs.msg.JointAngles, func, queue_size=1)

def subscribe_joint_torques(func):
    rospy.Subscriber(ROS_PREFIX + '/out/joint_torques', kinova_msgs.msg.JointTorque, func, queue_size=1)

def parse_joint_angles(msg):
    # read the current joint angles from the robot
    curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7])

    # convert to radians
    curr_pos = curr_pos*(math.pi/180.0)

    return curr_pos

def parse_torques(msg):
    torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7])
    return torque_curr

joint_thresholds = np.array([8, 8, 8, 2, 1, 1, 1, 1])

def interpret_torques(torque_curr):
    interaction = False

    for i in range(7):
        if np.fabs(torque_curr[i]) > joint_thresholds[i]:
            interaction = True
        else:
            # zero out torques below threshold for cleanliness
            torque_curr[i] = 0.0

    return torque_curr, interaction
