#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def callback(msg):
    
    try:
	rospy.loginfo("Received distance: %d", msg.data)
        distance = int(msg.data)
        rospy.loginfo("Received distance: %d", distance)
    except ValueError:
        rospy.logerr("Received invalid distance: %s", msg.data)

def distance_node_listener():
    rospy.init_node('distance_node_listener', anonymous=True)
    rospy.Subscriber('serial_data_distance', String, callback)
    rospy.spin()

if __name__ == '__main__':
    distance_node_listener()
