#!/usr/bin/env python
import rospy
from std_msgs.msg import String
#import serial

#ser = serial.Serial('/dev/ttyUSB1', 115200)  

def callback(data):

    if  data.data[0] == '+':
        distance_str = data.data[15:-5]
        data = distance_str
	rospy.loginfo(data)
        pub.publish(data)
    else :
	
	pass



pub = rospy.Publisher('RF_data_recoding', String, queue_size=10)
rospy.init_node('only_RF_node', anonymous=True)
rospy.Subscriber('serial_data_distance', String, callback)
rospy.spin()




