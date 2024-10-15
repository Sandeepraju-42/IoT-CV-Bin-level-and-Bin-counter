# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:12:16 2023

@author: SandeepRaju
"""
###############################################################################
#set path
#Conda environment: conda activate C:/Users/Sandeep/anaconda3/envs/CV_ML2303
# conda activate C:/Users/Sandeep/anaconda3/envs/CV_ML2303
# conda update conda --all
# conda update anaconda

path = 'C:/Users/sandeep/Desktop/ML2303 Digitalisation for Sustainable Production/Group 7'
pathcopy = path
###############################################################################


###############################################################################
#Libraries
###############################################################################
#openCV and image
import cv2 #pip install opencv-python
import uuid
import os
import time

#pip install --upgrade pyqt5 lxml

#Arduino connection
import serial #pip install pyserial
import time
###############################################################################

###############################################################################
# Define Images to Collect and capture images into folder
###############################################################################
labels = ['zero', 'quarter', 'half', 'threequarter', 'full']
number_imgs = 10 #number of images per class


# Setup Folders
os.chdir(path)
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix': #os/linux machine?
        !mkdir -p {IMAGES_PATH}
    if os.name == 'nt': #windows?
         !mkdir {IMAGES_PATH}
         
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}

#reset Arduino servo position

ser = serial.Serial('COM7', 9600, timeout=1) #open connection to Arduino
ser.write(b'0') #sets servo position to zero
print(str(ser.readline()))
time.sleep(1)

ser.write(b'0') 
print(str(ser.readline()))
time.sleep(1)



#Capture Images
cap = cv2.VideoCapture(0)

for label in labels:  
    i = 0
    ser.write(b'0') #reset position of Arduino servo 
    print(str(ser.readline()))
    time.sleep(1) #end reset
    
    print('Collecting images for {}'.format(label))
    time.sleep(1)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        
        #begin Arduino control
        send = str.encode(str(i*(180/(number_imgs-1)))) #set position of DC servo
        ser.write(send) #send position to Arduino
        #print(str(ser.readline())) #read servo positio from Echo Arduino
        time.sleep(2) #wait for servo to reach position
        #end arduino control
        
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        i = i + 1

        if cv2.waitKey(1) & 0xFF == ord('q'): #pressing q stops process
            break
    print(input("Press any key to continue to the next label"))

cap.release()
cv2.destroyAllWindows()


ser.write(b'0') #reset Servo again
print(str(ser.readline()))
ser.close()  #close serial connection to Arduino
del(i,send,ser)

###############################################################################




###############################################################################
# Image Labelling
###############################################################################
#pip install --upgrade pyqt5 lxml --user
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

if not os.path.exists(LABELIMG_PATH):
    !mkdir {LABELIMG_PATH}
    !git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}
    
if os.name == 'posix':
    !cd {LABELIMG_PATH} && make qt5py3 #for mac
if os.name =='nt':
    #for windows
    !cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc 


!cd {LABELIMG_PATH} && python labelImg.py
###############################################################################



###############################################################################
#Move them into a Training and Testing Partition
# ****train and test folders added manually in the paths***
###############################################################################
import shutil, os, glob, random
from shutil import copy
from pathlib import Path

# List all files in a directory using os.listdir

dst = Path(pathcopy + "/" + 'Tensorflow/workspace/images/train')
dst_test = Path(pathcopy + "/" + 'Tensorflow/workspace/images/test')

for label in labels:  
    print(label)
    basepath = pathcopy + '/Tensorflow/workspace/images/collectedimages/' + label
    
    filenames = []
    filenamesunique = []

    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)
            filenames.append(entry)
        
    filenamesunique.append(list(set([i.split('.', 2)[1] for i in filenames])))


    indices=[i for i in range(len(filenamesunique[0]))]        
    filenamesunique[0].sort()
    random.seed(230)
    random.shuffle(indices) # shuffles the ordering of filenames (deterministic given the chosen seed)
    
    split = int(0.8 * len(filenamesunique[0]))
    file_train = [filenamesunique[0][idx] for idx in indices[:split]]
    file_test = [filenamesunique[0][idx] for idx in indices[split:]]

    #print(file_test)
    #print(file_train)
    
    src = Path(basepath)
    
    idx = 0
    for file in src.iterdir():
        if ([ele for ele in file_train if(ele in str(file))]):
            idx += 1
            copy(file, dst)
            
    idx = 0
    for file in src.iterdir():
        if ([ele for ele in file_test if(ele in str(file))]):
            idx += 1
            copy(file, dst_test)
###############################################################################























