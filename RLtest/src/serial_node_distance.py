#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import serial

ser = serial.Serial('/dev/ttyUSB1', 115200)  

def talker():

    pub = rospy.Publisher('serial_data_distance', String, queue_size=10)
    rospy.init_node('distance_node', anonymous=True)


    while not rospy.is_shutdown():
        if ser.in_waiting > 0:

            data = ser.readline().decode()

            rospy.loginfo(data)
            pub.publish(data)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

