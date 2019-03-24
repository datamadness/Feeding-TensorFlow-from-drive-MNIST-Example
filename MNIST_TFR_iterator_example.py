# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:16:02 2019

@author:    DATAmadness
Github:     https://github.com/datamadness
Blog:       ttps://datamadness.github.io
"""

import tensorflow as tf
import os
#%% Specify parameters 
batch_size = 100                    #Note that large batch sized is linked to sharp gradients
training_steps = 100                #Number of batches to train on (100 for a quick test, 1000 or more for results)
num_epochs = None                   #None to repeat dataset until all steps are executed
eval_folder = 'MNIST_TFRs_eval'     #Subfolder containing TFR files with evaluation data
train_folder = 'MNIST_TFRs_train'   #Subfolder containing TFR files with training data
#%% Building the CNN MNIST Classifier
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  input_layer = tf.reshape(features["image_data"], [-1, 28, 28, 1])
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) 
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#%% CREATE ESTIMATOR

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp2/mnist_convnet_model")

#%% Set Up a Logging Hook

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

#%% Input function for training data

def dataset_input_fn(subfolder, batch_size, train = False, num_epochs=None):
         
    filenames = [file for file in os.listdir(os.path.join(os.getcwd(), subfolder)) if file.endswith('.tfrecord')]
    filenames = [os.path.join(subfolder, file) for file in filenames]
    dataset = tf.data.TFRecordDataset(filenames)

    #Create record extraction function
    def parser(record):
        features = {
            'height': tf.FixedLenFeature([], dtype=tf.int64),
            'width': tf.FixedLenFeature([], dtype=tf.int64),
            'img_string': tf.VarLenFeature(dtype=tf.float32),
            'label': tf.FixedLenFeature([], dtype=tf.int64)}
        parsed = tf.parse_single_example(record, features)
        
        # Perform additional preprocessing on the parsed data.
        image = parsed['img_string']
        image = tf.sparse.to_dense(image,default_value = 0)
        label = tf.cast(parsed["label"], tf.int32)
    
        return {"image_data": image, "width": parsed["width"]}, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    
    #Shuffle data if in training mode
    if train:
        dataset = dataset.shuffle(buffer_size=batch_size*2)  #Shuffles along first dimension(rows)(!)  and selects from buffer
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    
    # Each element of `dataset` is tuple containing a dictionary of features
    # (in which each value is a batch of values for that feature), and a batch of
    # labels.
    return dataset

#%% Playground for understanding the dataset workflow / experimenting - not for actual CNN
#sess = tf.InteractiveSession()
#dataset = dataset_input_fn(train_folder, train = True, batch_size = 1, num_epochs=None)
#iterator = dataset.make_one_shot_iterator()
#batch = iterator.get_next()
#image_tensor = tf.reshape(d[0]["image_data"], [-1, 28, 28])
#image = image_tensor.eval()
#%% Train the clasifier
mnist_classifier.train(
    input_fn=lambda : dataset_input_fn(train_folder, train = True, batch_size = batch_size, num_epochs=num_epochs),
    #input_fn=dataset_train_input_fn,
    steps=training_steps,
    hooks=[logging_hook])
#%% Evaluate the model
eval_results = mnist_classifier.evaluate(
        input_fn=lambda : dataset_input_fn(eval_folder, train = False, batch_size = batch_size, num_epochs=1))
        #input_fn=dataset_eval_input_fn)
print(eval_results)