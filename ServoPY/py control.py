# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:29:43 2023

@author: SandeepRaju
"""

#pip install pyserial

import serial
import time


ser = serial.Serial('COM7', 9600, timeout=1) 

ser.write(b'0')
print(str(ser.readline()))
time.sleep(5)

ser.write(b'0')
print(str(ser.readline()))
time.sleep(5)



ser = serial.Serial('COM7', 9600, timeout=1) 
for i in range(10):
    send = str.encode(str(i*20))
    ser.write(send)
    print(str(ser.readline()))
    time.sleep(5)   

ser.write(b'0')
print(str(ser.readline()))
ser.close()  
del(i,send,ser)
