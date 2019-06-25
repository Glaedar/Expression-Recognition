import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import threading

#Adding Seed so that there is a random place the image comes form and it is notstarting at the same point each iteration
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#setting the variable batch_size to 32
batch_size = 32

#Setting the names of the classes and putting classes into an array
classes = ['Angry','Disgusted','Fear','Happy','Neutral','Sad','Surprised']

#Getting the Length of the array classes
num_classes = len(classes)

# 20% of the data will automatically be used for validation and setting the rest of the variables
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='training_data'

# run data set python script, runs read_train_sets function, sending previously set variables as arguments
data = dataset.read_train_sets(train_path, img_size, classes, validation_size)

#outputs to console
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


#starts the tensorflow session
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

#set up the initial weights by specifying the shape
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#set up the initial biases byt specifiying the size
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    #use the create_weights function to define the weights
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    #create the biases
    biases = create_biases(num_filters)

    #create the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    #apply the max_pooling layer
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    #fed to the reLUs to change negative values to 0
    layer = tf.nn.relu(layer)

    return layer

    

def create_flatten_layer(layer):
    # Get the shape of the layer from the previous layer.
    layer_shape = layer.get_shape()

    #calculate number of features
    num_features = layer_shape[1:4].num_elements()

    #flatten the layer and reshape to the number of features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

#runs through the convolutional function 3 times using the multiple layers
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
			   
			   
 #runs through the flatten layer         
layer_flat = create_flatten_layer(layer_conv3)

#runs through the fully connected layer using the true and false ReLUs
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

#get the probability of each class by applying the tensorflow softmax
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
#contains the predicted probability for each input class, the prediction of the network will be the class with the higher probability
y_pred_cls = tf.argmax(y_pred, dimension=1)
#runs the tensorflow session
session.run(tf.global_variables_initializer())
#calculates the cost using tensorflow function which takes output of the last fully connected layer and actual labels to reach the optimum value of weights
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#use the adam optimizer for gradient calculation and weight optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 

#gets all the values required to print out to console
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
			#after each epoch report the accuracy numbers and save the model using saver object in tensorflow
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, 'C:\\Users\\Andrew\\Desktop\\AI Project\\Artificial Intelligence Coursework\\emotions_model') 


    total_iterations += num_iteration

train(num_iteration=3000)

