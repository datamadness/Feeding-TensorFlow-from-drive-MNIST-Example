# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:49:35 2019

@author:    DATAmadness
Github:     https://github.com/datamadness
Blog:       ttps://datamadness.github.io

Converts MNIST dataset to TFRecords and reads it back using Data API


"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%% LOAD DATA  
# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
#Scale data
train_data = train_data / np.float32(255)

#%% Function to parse a single record from the orig MNIST data into dict suitable for 

def get_observation(observation_id=0):
    img_data = train_data[observation_id,:,:]
    shape = img_data.shape
    observation_data = {
            'height': shape[0],
            'width': shape[1],
            #Flatten 2D array into 1D arrye so it can be stored as a list
            'img_string': img_data.flatten(order='C'),
            'label': train_labels[observation_id]
            }
    return observation_data

#%% # Create an example object with features stored in lists
    # This is the format for adding data into TensorFlow TFRecords
def get_example_object(single_record):
    
    record = tf.train.Example(features=tf.train.Features(feature={
        'height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['height']])),
        'width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['width']])),
        'img_string': tf.train.Feature(
            #float_list=tf.train.FloatList(value=single_record['img_string'])), 
            bytes_list=tf.train.BytesList(value=[single_record['img_string'].tobytes(order='C')])), 
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['label']]))
    }))
    return record

#%% Write data into TFrecord files

#Number of TFR files to save the data into
numFiles = 10
records_per_file = int(len(train_data)/numFiles)

for file_id in range(numFiles):
    with tf.python_io.TFRecordWriter('MNIST_train_data_strings_' + str(file_id) + '.tfrecord') as tfwriter:
        
        # Iterate through all records
        for observation_id in range(records_per_file):
            observation_data = get_observation(observation_id + file_id * records_per_file)
            example = get_example_object(observation_data)
    
            # Append each record into TFRecord
            tfwriter.write(example.SerializeToString())

#%% An example how to read the dataset from TFR

#Create record parsing function
def extract_fn(data_record):
    features = {
        'height': tf.FixedLenFeature([], dtype=tf.int64),
        'width': tf.FixedLenFeature([], dtype=tf.int64),
        'img_string': tf.VarLenFeature(dtype=tf.float32), #if using floatlist
        #'img_string': tf.FixedLenFeature([], dtype=tf.string), #if using bytelist
        'label': tf.FixedLenFeature([], dtype=tf.int64)}
    parsed_sample = tf.parse_single_example(data_record, features)
    
    return parsed_sample

#Create TFRecord files iterator (using single file for demonstration)
filenames = ['MNIST_train_data_0.tfrecord']
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

#Get first ten handwritten digits from the MNIST example in TFRecords and plot them
with tf.Session() as sess:
    try:
        #while True:
        for i in range(10):
            data_record = sess.run(next_element)
            digit = data_record['img_string'][1].reshape((data_record['height'],data_record['width']), order='C') #if using floatlist
            #digit = np.frombuffer(data_record['img_string'], dtype = np.uint8).reshape((data_record['height'],data_record['width']), order='C') #if using bytelist
            print(type(digit))
            plt.imshow(digit, cmap='gray')
            plt.axis('off')
            plt.show()
            print(data_record['label'])
    except:
        print('Ran out of records')
 
