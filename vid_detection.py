# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:29:54 2023

@author: SandeepRaju
"""

#set path
path = 'C:/Users/sandeep/Desktop/ML2303 Digitalisation for Sustainable Production/Group 7/TF_Object_Count'

###############################################################################
#Libraries
###############################################################################
import os
import datetime
import tensorflow as tf
import cv2 
import numpy as np
import pandas as pd
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
###############################################################################

###############################################################################
#Paths and locations
###############################################################################
os.chdir(path)
#os.system("conda run -n ML2303_PV_3_8_16")

#create directories for scripts
CUSTOM_MODEL_NAME = 'custom_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'GIT_TFOC_PATH' : os.path.join('Tensorflow', 'GIT_Repo'),
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
###############################################################################

###############################################################################
#load model
###############################################################################
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
###############################################################################


###############################################################################
#Ldetect images
#Real time from camera
###############################################################################


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

object_count = 0  # Initialize object counter
threshold = 0.8

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


# Dictionary to store object positions in previous frames
object_positions = [0,0]
pos = [(0,0)]
tracker_df = pd.DataFrame(columns=['Class','Confidence','TS','ObjectNo'])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=threshold,
        agnostic_mode=False)

    # Count objects based on detection scores
    object_scores = detections['detection_scores']
    object_boxes = detections['detection_boxes']
    object_classes = detections['detection_classes']
    object_count_screen = np.sum(object_scores > threshold)
    
    

    # Loop through all detected objects
    for i in range(len(object_scores)):
        # Get the class ID and detection score of the current object
        class_id = object_classes[i]
        score = object_scores[i]
        

        # Only consider objects that have a high enough detection score
        if score > threshold:
            # Get the position of the current object
            box = object_boxes[i]
            ymin, xmin, ymax, xmax = box
            position = (round((xmin + xmax) / 2, 1), round((ymin + ymax) / 2, 1))

            # Check if the current object overlaps with any object in the previous frame
            object_found = False
         
            if (abs(position[0]) == abs(pos[len(pos)-1][0]) and
                    abs(position[1]) == abs(pos[len(pos)-1][1])):
                object_found = True
                break

            # If the current object does not overlap with any object in the previous frame, count it as a new object
            if not object_found:
                # Increment the object count for the class
                object_count +=1
                object_positions.append(position)
                pos.append(position)
                
            # Remove positions that are too old
            object_positions = object_positions[-50:]
            pos = pos[-50:]


    # Print the total number of objects detected and the number of new objects counted
    #print('Total objects detected:', object_count_screen)
    #print('New objects:', object_count)
    
    #time.sleep(2)  # Sleep for 1 second
    if score > threshold:
        tracker_df=pd.concat([tracker_df,
                   pd.DataFrame([[labels[class_id]['name'],score, datetime.datetime.now(),object_count]], 
                     columns=['Class','Confidence','TS','ObjectNo'])])
        
        #retain latest 10 objects
        #tracker_df=tracker_df.drop(tracker_df[tracker_df['ObjectNo'] >= (max(tracker_df['ObjectNo'])-10)].index) 
        tracker_TS = tracker_df.groupby(['ObjectNo','Class']).agg({'TS': [np.min,np.max]})
        
        #retain last 4 rows
        tracker_TS=tracker_TS.iloc[-3:]
        tracker_TS = tracker_TS.droplevel(0, axis=1).rename_axis(index=(None, None), columns=None)
        tracker_TS['diff'] = (tracker_TS['amin'] - tracker_TS['amax']).dt.seconds/3600


    cv2.putText(image_np_with_detections,"Object count   :" + str(object_count),(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image_np_with_detections,"Current Time   :" + str(datetime.datetime.now()),(50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image_np_with_detections,"ETA completion :" + str(datetime.datetime.now() + datetime.timedelta(seconds=tracker_TS['diff'].mean()*4)),(50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)                         
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))


                
    print([labels[class_id]['name'],score, datetime.datetime.now(),object_count,tracker_TS['diff'].mean()*4])

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
###############################################################################

