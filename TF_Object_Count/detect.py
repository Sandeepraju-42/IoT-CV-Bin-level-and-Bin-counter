# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:20:31 2023

@author: SandeepRaju
"""

import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import pandas as pd
import datetime
import os

tracker_df = pd.DataFrame(columns=['Class','Confidence','TS','ObjectNo'])

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
threshold = 0.80
object_positions = [0,0]

pos=[(0,0)]
object_count=0

labels = [{'name':'Big_zero', 'id':1}, 
          {'name':'Big_quarter', 'id':2}, 
          {'name':'Big_half', 'id':3}, 
          {'name':'Big_threequarter', 'id':4},
          {'name':'Big_full', 'id':5},
          {'name':'Small_zero', 'id':6}, 
          {'name':'Small_quarter', 'id':7}, 
          {'name':'Small_half', 'id':8},
          {'name':'Small_threequarter', 'id':9},
          {'name':'Small_full', 'id':10}]

# def load_labels(path='labels.txt'):
#     """Loads the labels file. Supports files with or without index numbers."""
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         labels = {}
#         for row_number, content in enumerate(lines):
#             pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
#             if len(pair) == 2 and pair[0].strip().isdigit():
#                 labels[int(pair[0])] = pair[1].strip()
#             else:
#                 labels[row_number] = pair[0].strip()
#     return labels

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    # Get all output details
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def main():
    #labels = load_labels()
    interpreter = Interpreter('/home/pi/Desktop/Object_Detection/TFODRPi/detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0) #turn off auto focus
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, threshold)
 
        #object counting
        for result in res:
            # Get the class ID and detection score of the current object
            class_id = result['class_id']
            score = result['score']
            
        
            # Only consider objects that have a high enough detection score
            if score > threshold:
                # Get the position of the current object
                ymin, xmin, ymax, xmax = result['bounding_box']
                position = (round((xmin + xmax) / 2, 1), round((ymin + ymax) / 2, 1))
        
                # Check if the current object overlaps with any object in the previous frame
                object_found = False
                global pos
             
                if (abs(position[0]) == abs(pos[len(pos)-1][0]) and
                        abs(position[1]) == abs(pos[len(pos)-1][1])):
                    object_found = True
                    break
        
                # If the current object does not overlap with any object in the previous frame, count it as a new object
                if not object_found:
                    # Increment the object count for the class
                    global object_count
                    global object_positions
                    object_count +=1
                    object_positions.append(position)
                    pos.append(position)
                    
                # Remove positions that are too old                
                object_positions = object_positions[-50:]
                pos = pos[-50:]

        for result in res:
            # Get the class ID and detection score of the current object
            class_id = result['class_id']
            score = result['score']
            if score > threshold:
                #extrapolation
                global tracker_df
                tracker_df=pd.concat([tracker_df,
                   pd.DataFrame([[labels[int(class_id)]['name'] ,score, datetime.datetime.now(),object_count]], 
                     columns=['Class','Confidence','TS','ObjectNo'])])
        
                #retain latest 10 objects
                #tracker_df=tracker_df.drop(tracker_df[tracker_df['ObjectNo'] >= (max(tracker_df['ObjectNo'])-10)].index) 
                tracker_TS = tracker_df.groupby(['ObjectNo','Class']).agg({'TS': [np.min,np.max]})
                
                #retain last 4 rows
                tracker_TS=tracker_TS.iloc[-3:]
                tracker_TS = tracker_TS.droplevel(0, axis=1).rename_axis(index=(None, None), columns=None)
                tracker_TS['diff'] = (tracker_TS['amin'] - tracker_TS['amax']).dt.seconds/3600


        #res = detect_objects(interpreter, img, threshold)
        for result in res:
            # Get the class ID and detection score of the current object
            class_id = result['class_id']
            score = result['score']
            # Only consider objects that have a high enough detection score
            if score > threshold:
                # Get the position of the current object
                ymin, xmin, ymax, xmax = result['bounding_box']
                position = (round((xmin + xmax) / 2, 1), round((ymin + ymax) / 2, 1))                          

                #Camera feed
                ymin, xmin, ymax, xmax = result['bounding_box']
                xmin = int(max(1,xmin * CAMERA_WIDTH))
                xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                ymin = int(max(1, ymin * CAMERA_HEIGHT))
                ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                
                #adding textprompts into camera feed
                cv2.putText(frame,labels[int(class_id)]['name'] +" - Image Score:" + str(round(score,3)),(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4,cv2.LINE_AA)
                cv2.putText(frame,"Object count   :" + str(object_count),(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4,cv2.LINE_AA)
                cv2.putText(frame,"Current Time   :" + str(datetime.datetime.now()),(50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4,cv2.LINE_AA)
                cv2.putText(frame,"ETA completion :" + str(datetime.datetime.now() + datetime.timedelta(seconds=tracker_TS['diff'].mean()*4)),(50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4,cv2.LINE_AA)              
                                
                print("bin_name:",labels[int(class_id)]['name'],",",
                      "score:",score,",",
                      'time_stamp:',datetime.datetime.now(),",",
                      'object_count:',object_count,",",
                      'time_left:',round(tracker_TS['diff'].mean()*4,0))
                


                #adding retangle around detected image
                cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
  
        cv2.imshow('object detection', cv2.resize(frame,(1000,800)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()