#!/usr/bin/env python
import rospy
from std_msgs.msg import String
#import serial

#ser = serial.Serial('/dev/ttyUSB1', 115200)  

def callback(data):

    if  data.data[0] == 'U':
        distance_str = data.data[1:-2]
        data = distance_str
	rospy.loginfo(data)
        pub.publish(data)
    else :
	pass



pub = rospy.Publisher('serial_data_recoding', String, queue_size=10)
rospy.init_node('only_distance_node', anonymous=True)
rospy.Subscriber('serial_data_distance', String, callback)
rospy.spin()




