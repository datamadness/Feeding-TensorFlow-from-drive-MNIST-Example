# TensorFlow Estimators API - Out-of-the-core feeding of large datasets from drive via TFRecords (MNIST example)
It is easy to hit resource limits when working with large datasets. The available memory in particular becomes quickly a limiting factor when training your neural networks on swaths of data. The solution is to create a continuous stream of data that will sequentially read batch data from drive(s). Using this approach, the memory needs to hold only one batch of data while pre-loading the data for the next batch, allowing us to operate with datasets of virtually unlimited size.
This article demonstrates the approach on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using TensorFlow Estimators API, TFRecords and Data API.

## Detailed Documentation and Practical examples:
[Visit my blog post](https://datamadness.github.io/tensorflow_estimator_large_dataset_feed) for detailed documentation and practical examples

#### High Level Workflow Overview
1. Load MNIST dataset via Keras
1. Serialize the data into list(s)
1. Save the data on drive in TFRecord format
1. Create the Estimator API input function that builds dataset from TFRecords using an iterator
1. Train the Convolutional Neural Network streaming the data via a custom input function


#### Saving Data in TFRecord format
The MNIST training dataset consists of 60000 28x28 images of hand written digits such as this one:
![image post](/assets/images/tf_file_feed/MNIST_digit.png)

To save this data into TFRecord format, a couple of things need to happen: 

- The data must be represented in a dictionary format
- Individual dictionary values should be flattened into scalars or 1D arrays suitable for storing as a series

The following function transforms a single 28x28 MNIST digit into the desired format. Notice that the dictionary can have an arbitrary number of keys which is useful for storing any metadata such as original data dimension or training labels:
```python
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
```

The next step in generating the TFRecords is to transform the dictionary into an object where all values from the dictionary are saved into one of the three TensorFlow datatype lists:

- Int64List
- FloatList
- ByteList

Each datatype can offer different advantages, so it is worth experimenting with them. For example, I was able to significantly reduce the size required for storing the files by encoding Floats/Integers into ByteList, but it increases the complexity of your code and it benefits only certain data structures.

```python 
def get_example_object(single_record):
    
    record = tf.train.Example(features=tf.train.Features(feature={
        'height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['height']])),
        'width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['width']])),
        'img_string': tf.train.Feature(
            float_list=tf.train.FloatList(value=single_record['img_string'])), 
            #bytes_list=tf.train.BytesList(value=[single_record['img_string'].tobytes(order='C')])), 
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['label']]))
    }))
    return record
```

Using the transformation functions above, we can finally write the MNIST training dataset into TFRecord files. The following piece of code save the dataset into 10 files with 6000 records per file:

```python
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
```


#### Building an input function for Estimators API using TensorFlow Data API
Once we have the data saved into TFRecords on a drive, the next step is to write an input function for the Estimators API that will stream the data into your model. 
The easiest way to accomplish this is via the Data API, specifically the TFRecordDataset class that will directly create a dataset from a list of TFRecord files. 

The Estimators and Data APIs are designed to work together, so you do not even have to initialize the iterator for your dataset.
The only trick to is that you have to write a parser that will transform the serialized data in lists back into a format suitable for your model. As such this is also the perfect place to build your pipeline for further data preprocessing, transformation, and feature engineering.

Here is the input function for our MNIST dataset:

```python
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
        dataset = dataset.shuffle(buffer_size=batch_size*2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    
    # Each element of `dataset` is tuple containing a dictionary of features
    # (in which each value is a batch of values for that feature), and a batch of
    # labels.
    return dataset
```

#### Plugging it all together: Training your CNN on a data stream from TFR files

Now when we have all the pieces ready, we can put it all together and train our MNIST CNN on a data streaming from TFRecord files saved on your drive. Here is the complete modified [MNIST CNN example from TensorFlow page](https://www.tensorflow.org/tutorials/estimators/cnn):

```python
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
         
    filenames = [file for file in os.listdir(os.path.join(os.getcwd(), 
		subfolder)) if file.endswith('.tfrecord')]
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
        dataset = dataset.shuffle(buffer_size=batch_size*2)
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
        input_fn=lambda : dataset_input_fn(eval_folder, train = False, batch_size = batch_size,
			num_epochs=1))
        #input_fn=dataset_eval_input_fn)
print(eval_results)

```
