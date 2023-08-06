#!/usr/bin/env python
import rospy
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)  

def send_serial_command():
   
    rospy.init_node('serial_sender', anonymous=True)

    while not rospy.is_shutdown():
    	ser.write(b'R')

if __name__ == '__main__':
    try:
        send_serial_command()
    except rospy.ROSInterruptException:
        pass

