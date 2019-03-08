from lib_1.support_function import *


import tensorflow as tf
import numpy as np
import pandas as pd
import math

#import matplotlib.pyplot as plt



def feature_extraction_model(training_filelist,test_filelist,num_HL  , model_path,training_files, test_files, training_output, test_output):


    tf.reset_default_graph()

    # input data pipline
    num_files= len(training_filelist)

    training_dataset = tf.data.Dataset.from_tensor_slices(training_filelist)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_filelist)

    training_dataset = training_dataset.map(lambda item: tf.py_func(read_npy, [item], tf.float32))
    test_dataset = test_dataset.map(lambda item: tf.py_func(read_npy, [item], tf.float32))

    training_dataset = training_dataset.batch(1)
    test_dataset = test_dataset.batch(1)

    handle = tf.placeholder(tf.string, shape = [])
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)

    next_element = iterator.get_next()

    training_iterator = training_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()


    # Get file ID
    training_ID=np.empty(shape=[len(training_filelist),1])
    for i, file in enumerate(training_files):
        training_ID[i] = int(file.split('_')[0])
    training_ID=np.transpose(training_ID)

    test_ID=np.empty(shape=[len(test_filelist),1])
    for i, file in enumerate(test_files):
        test_ID[i] = int(file.split('_')[0])
    test_ID=np.transpose(test_ID)
    # create place holder
    X=tf.placeholder(tf.float32, shape= (None, 192,288,288,1))

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward_propagation
    encoder_output = encoder_layers(X, parameters, num_HL)

    init = tf.global_variables_initializer()

    # Start tf.Session
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        training_handle = sess.run(training_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        training_feature_matrix=np.empty(shape= [len(training_filelist),num_HL + 1])

        # Restore model
        saver.restore(sess, model_path)

        for batch in range(len(training_filelist)):

            sample = sess.run(next_element, feed_dict={handle: training_handle})
            print(sample.shape, batch)
            output_features = sess.run([encoder_output], feed_dict={X: sample})
            print(output_features[0].shape)
            training_feature_matrix[batch,1:]=output_features[0]
        training_feature_matrix[:,0]=training_ID

        np.savetxt(training_output,training_feature_matrix, delimiter=',')

        test_feature_matrix=np.empty(shape= [len(test_filelist),num_HL + 1])
        for batch in range(len(test_filelist)):
            sample = sess.run(next_element, feed_dict={handle: test_handle})
            print(sample.shape, batch)
            output_features = sess.run([encoder_output], feed_dict={X: sample})
            print(output_features[0].shape)
            test_feature_matrix[batch,1:]=output_features[0]
        test_feature_matrix[:,0]=test_ID

        np.savetxt(test_output,test_feature_matrix, delimiter=',')




    return

