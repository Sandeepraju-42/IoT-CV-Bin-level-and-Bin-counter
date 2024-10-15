# -*- coding: utf-8 -*-
"""
@author: SandeepRaju

Set up virtual env in the folder of your choosing 
open conda console, CD to desired folder

run the code: conda create --prefix=Custom_envs python=3.8
run code to activate and move to virtual env: conda activate ./Custom_envs
#install spyder: conda install spyder=5.4.3
install spyder kernel: pip install spyder-kernels=2.1.*
"""

#set path
path = 'C:/Users/sandeep/Desktop/ML2303 Digitalisation for Sustainable Production/Group 7/TF_Object_Count'
###############################################################################
#Libraries
###############################################################################
#dependencies:
# TensorFlow Object Detection API : https://github.com/tensorflow/models/tree/master/research/object_detection
# tensorflow
# keras
# opencv-python
# Protobuf #pip install --upgrade protobuf
# Python-tk
# Pillow
# lxml
# tf Slim (which is included in the "tensorflow/models/research/" checkout)
# Jupyter notebook
# Matplotlib
# Cython
# contextlib2
# cocoapi: pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# gin: pip install gin-config
#pip install wget


import os
import gin 
import wget

os.chdir(path)
###############################################################################

###############################################################################
#install TensorFlow 2 Detection Model Zoo
###############################################################################
CUSTOM_MODEL_NAME = 'custom_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

#create directories for scripts
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


for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            !mkdir -p {path}
        if os.name == 'nt':
            !mkdir {path}
            
#print(paths, files)
###############################################################################

###############################################################################
#Download TF Models Pretrained Models from Tensorflow Model Zoo and Install TFOD
###############################################################################
# https://www.tensorflow.org/install/source_windows

#using lib wget
#clone git from official TF
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}


#clone git from TF object counter
if not os.path.exists(os.path.join(paths['GIT_TFOC_PATH'],'GIT_Repo')):
    !git clone https://github.com/ahmetozlu/tensorflow_object_counting_api.git {paths['GIT_TFOC_PATH']}
    
# Install Tensorflow Object Detection protoc
if os.name=='posix':  
    !apt-get install protobuf-compiler
    !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 

#install official tensorflow
if os.name=='nt':
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    !move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}
    !cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    !cd models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python -m pip install --use-feature=2020-resolver && python setup.py build && python setup.py install
    !cd Tensorflow/models/research/slim && pip install -e . 
  

#verify installations
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
!python {VERIFICATION_SCRIPT}

#if the above throws error, install libraries from below and check again

# pip uninstall numpy scipy  protobuf matplotlib -y
# pip install  numpy scipy protobuf matplotlib
# pip install --upgrade "protobuf<=3.20.1"
# pip install pycocotools-windows
# pip install tensorflow-text~=2.10.0
# pip install tensorflow --upgrade --user

# pip install tensorflow_addons
# pip install pycocotools

# !pip install Cython>=0.29.12

###############################################################################

###############################################################################
import object_detection 
#may get error, try restarting kernel and run all steps except download section

#!pip list
#intall CUDA and cuDNN to enable GPU, if no GPU skip installation of CUDN and cuDNN
#follow version guide as per link https://www.tensorflow.org/install/source
#and pip install tensorflow-gpu

#download/clone from tensorflow zoo git
if os.name =='posix':
    !wget {PRETRAINED_MODEL_URL}
    !mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}
    !cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    !move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}
    !cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
###############################################################################


###############################################################################
#Create Label Map
###############################################################################
#make sure to use the same names as image collected list
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


with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
###############################################################################


###############################################################################
#Create TF records
###############################################################################
#clone from TF records
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    !git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}

#training dataset
!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} 

#testing dataset
!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}     


#Copy Model Config to Training Folder
if os.name =='posix':
    !cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}
if os.name == 'nt':
    !copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}
    

#Update Config For Transfer Learning
import tensorflow as tf
import yaml
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

#print(config)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  
    
    
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 10 #change to number of labels
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

###############################################################################'


###############################################################################
#Model
###############################################################################
#Train the model
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 
                               'research', 
                               'object_detection', 
                               'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(TRAINING_SCRIPT, 
                                                                                             paths['CHECKPOINT_PATH'],
                                                                                             files['PIPELINE_CONFIG'])

       
!{command}

#Evaluate the Model
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, 
                                                                                          paths['CHECKPOINT_PATH'],
                                                                                          files['PIPELINE_CONFIG'], 
                                                                                          paths['CHECKPOINT_PATH'])
!{command}

#tensorboard --logdir=.
###############################################################################



###############################################################################
#Load Train Model From Checkpoint  
###############################################################################
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
###############################################################################

###############################################################################
#Ldetect images
###############################################################################
import cv2 
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'zero.e40b9c4f-f002-11ed-b8f2-28f10e0aa7f1.jpg')


img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

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
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)



plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
###############################################################################


###############################################################################
#Real time from camera
###############################################################################
#!pip uninstall opencv-python-headless -y

import datetime
import time

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

threshold = 0.8

cap = cv2.VideoCapture(0)
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
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=threshold,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    #print(labels[detections['detection_classes'][0]]['name'],detections['detection_scores'][0],datetime.datetime.now())
    #time.sleep(1) # Sleep for 1 seconds
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

###############################################################################



import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import datetime
import time

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = 0  # Initialize object counter
threshold = 0.8

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
    object_count = np.sum(object_scores > threshold)
    
    # Display object count on camera feed
    cv2.putText(image_np_with_detections, 'Object Count in feed: {}'.format(object_count),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    print(labels[detections['detection_classes'][0]]['name'], detections['detection_scores'][0],
          datetime.datetime.now())
    print('Total objects detected:', object_count)
    time.sleep(1)  # Sleep for 1 second

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break








import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import datetime
import time

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = 0  # Initialize object counter
threshold = 0.8

# Keep track of detected object IDs and their presence
detected_objects = {}

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

    # Update object counts and track object presence
    object_scores = detections['detection_scores']
    object_classes = detections['detection_classes']
    object_boxes = detections['detection_boxes']

    object_count = np.sum(object_scores > threshold)

    for score, obj_class, box in zip(object_scores, object_classes, object_boxes):
        if score > threshold:
            class_name = labels[obj_class]['name']
            if obj_class not in detected_objects:
                detected_objects[obj_class] = {'count': 0, 'present': False, 'last_box': box}
                detected_objects[obj_class]['present'] = True
            elif not detected_objects[obj_class]['present']:
                detected_objects[obj_class]['count'] += 1
                detected_objects[obj_class]['present'] = True
            detected_objects[obj_class]['last_box'] = box
        else:
            if obj_class in detected_objects and detected_objects[obj_class]['present']:
                detected_objects[obj_class]['present'] = False

    # Display object count on camera feed
    for obj_class, data in detected_objects.items():
        class_name = labels[obj_class]['name']
        count = data['count']
        cv2.putText(image_np_with_detections, '{}: {}'.format(class_name, count),
                    (10, 60 + obj_class * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    print(labels[detections['detection_classes'][0]]['name'], detections['detection_scores'][0],
          datetime.datetime.now())
    print('Total objects detected:', object_count)
    time.sleep(1)  # Sleep for 1 second

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break












import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import datetime
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

object_count = 0  # Initialize object counter
threshold = 0.8

# Dictionary to store object positions in previous frames
object_positions = [0,0]
pos = [(0,0)]
tracker=[]
object_counts=0


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
            print(labels[class_id]['name'],score,datetime.datetime.now(),position)

            # Check if the current object overlaps with any object in the previous frame
            object_found = False
            
            if (abs(position[0]) == abs(pos[len(pos)-1][0]) and
                    abs(position[1]) == abs(pos[len(pos)-1][1])):
                object_found = True
                break

            # If the current object does not overlap with any object in the previous frame, count it as a new object
            if not object_found:
                # Increment the object count for the class
                if labels[class_id]['name'] == 'zero':
                    object_counts +=1
                    
                if labels[class_id]['name'] == 'quarter':
                    object_counts +=1                
                
                if labels[class_id]['name'] == 'half':
                    object_counts +=1

                if labels[class_id]['name'] == 'threequarter':
                    object_counts +=1
                    
                if labels[class_id]['name'] == 'full':
                    object_counts +=1

                tracker.append([labels[class_id]['name'],datetime.datetime.now(),object_counts])
                
                
                object_positions.append(position)
                
            # Remove positions that are too old
            object_positions = object_positions[-50:]
            
            pos.append(position)
            pos = pos[-50:]


    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    # Print the total number of objects detected and the number of new objects counted
    print('Total objects detected:', object_count_screen)
    print('New objects:', object_count)
    
    time.sleep(1)  # Sleep for 1 second

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


###############################################################################
#freezing the graph
###############################################################################
#!pip install tensorflowjs

FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')


command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])


print(command)
!{command}

#Conversion to TFJS
!pip install tensorflowjs

command = "tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default {} {}".format(os.path.join(paths['OUTPUT_PATH'], 'saved_model'), paths['TFJS_PATH'])

print(command)
!{command}

#Conversion to TFLite
TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')

command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['TFLITE_PATH'])

print(command)
!{command}

FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')

command = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )

print(command)
!{command}

#Zip and Export Models
!tar -czf models.tar.gz {paths['CHECKPOINT_PATH']}



###############################################################################


