import tensorflow as tf
import numpy as np
import pandas as pd
import os



#tf.contrib.layers.xavier_initializer()

def initialize_parameters():

    tf.set_random_seed(1)


    parameters = {
    "encoder_w1" : tf.get_variable("encoder_w1", shape= [5,5,5,1,16], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "encoder_w2" : tf.get_variable("encoder_w2", shape= [5,5,5,16,32], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "encoder_w3" : tf.get_variable("encoder_w3", shape= [5,5,5,32,64], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "encoder_w4" : tf.get_variable("encoder_w4", shape= [5,5,5,64,64], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),


    "decoder_w4" : tf.get_variable("decoder_w4", shape= [5,5,5,1,16], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "decoder_w3" : tf.get_variable("decoder_w3", shape= [5,5,5,16,32], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "decoder_w2" : tf.get_variable("decoder_w2", shape= [5,5,5,32,64], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),
    "decoder_w1" : tf.get_variable("decoder_w1", shape= [5,5,5,64,64], dtype= tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(factor = 0.5)),

    }

    return parameters

def encoder_layers(X,parameters,num_HL):


    # Reshape (note sure if this is necessary)
    enc = tf.reshape(X,[-1,192,288,288,1])


    enc = tf.nn.conv3d(enc, filter= parameters["encoder_w1"] ,strides=[1,2,2,2,1],padding="SAME")
    enc = tf.nn.leaky_relu(enc, alpha=0.2)

    enc = tf.nn.conv3d(enc, filter= parameters["encoder_w2"] ,strides=[1,2,2,2,1],padding="SAME")
    enc = tf.nn.leaky_relu(enc, alpha=0.2)

    enc = tf.nn.conv3d(enc, filter= parameters["encoder_w3"] ,strides=[1,2,2,2,1],padding="SAME")
    enc = tf.nn.leaky_relu(enc, alpha=0.2)

    enc = tf.nn.conv3d(enc, filter= parameters["encoder_w4"] ,strides=[1,2,2,2,1],padding="SAME")
    enc = tf.nn.leaky_relu(enc, alpha=0.2)

    enc = tf.reshape(enc, shape = [-1, 64*18*18*12])
    encoder_output = tf.layers.dense(enc, num_HL, activation = tf.nn.sigmoid, kernel_initializer= tf.contrib.layers.xavier_initializer())

    return encoder_output

def decoder_layers(encoder_output, parameters, num_HL):

    # Reshape input
    dec = tf.reshape(encoder_output, shape = [-1, num_HL])


    dec = tf.layers.dense(dec, 64*18*18*12)
    dec = tf.reshape(dec, shape=(-1,12,18,18,64))
    temp_batch_size = tf.shape(dec)[0]

    dec = tf.nn.conv3d_transpose(dec, filter = parameters["decoder_w1"], output_shape =[temp_batch_size,24,36,36,64] , strides=[1,2,2,2,1], padding = 'SAME')
    dec = tf.nn.relu(dec)

    dec = tf.nn.conv3d_transpose(dec, filter = parameters["decoder_w2"], output_shape =[temp_batch_size,48,72,72,32] , strides=[1,2,2,2,1], padding = 'SAME')
    dec = tf.nn.relu(dec)

    dec = tf.nn.conv3d_transpose(dec, filter = parameters["decoder_w3"], output_shape =[temp_batch_size,96,144,144,16] , strides=[1,2,2,2,1], padding = 'SAME')
    dec = tf.nn.relu(dec)

    dec = tf.nn.conv3d_transpose(dec, filter = parameters["decoder_w4"], output_shape =[temp_batch_size,192,288,288,1] , strides=[1,2,2,2,1], padding = 'SAME')
    dec = tf.nn.relu(dec)

    return dec

def read_npy(filename):
    #temp = np.load(filename.decode())
    #print(temp.shape)
    data = np.expand_dims(np.load(filename.decode()), axis = -1)
    return data.astype(np.float32)

def normalize(image):
    MIN_BOUND  = 0
    MAX_BOUND = 255
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = image.astype(np.float32)
    return image
