#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Vector3

def command():
    pub = rospy.Publisher('jaco_perturbations', Vector3, queue_size=10)
    rospy.init_node('pertruber', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        cmd = raw_input('Enter command:')
        if cmd == 'l':
            v = Vector3(0, 1, 0)
        else:
            v = Vector3(0, -1, 0)
        pub.publish(v)
        rate.sleep()

if __name__ == '__main__':
    try:
        command()
    except rospy.ROSInterruptException:
        pass