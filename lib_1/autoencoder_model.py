from lib_1.support_function import *



import tensorflow as tf
import numpy as np
import pandas as pd
import math

#import matplotlib.pyplot as plt




def autoencoder_model_tf(training_filelist,validation_filelist,num_HL, learning_rate,lambda_value, num_epochs, minibatch_size, restore_model, model_path,output_file,print_cost):

    tf.reset_default_graph()

    # input data pipline
    training_numfiles= len(training_filelist)
    validation_numfiles = len(validation_filelist)

    training_dataset = tf.data.Dataset.from_tensor_slices(training_filelist)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_filelist)

    training_dataset = training_dataset.map(lambda item: tf.py_func(read_npy, [item], tf.float32))
    validation_dataset = validation_dataset.map(lambda item: tf.py_func(read_npy, [item], tf.float32))

    training_dataset = training_dataset.shuffle(buffer_size=len(training_filelist)).batch(minibatch_size).repeat(num_epochs)
    validation_dataset = validation_dataset.shuffle(buffer_size=len(validation_filelist)).batch(minibatch_size).repeat()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_one_shot_iterator()



    # create place holder
    X=tf.placeholder(tf.float32, shape= (None, 192,288,288,1))

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward_propagation
    encoder_output = encoder_layers(X, parameters, num_HL)
    decoder_output = decoder_layers(encoder_output, parameters, num_HL)

    # Model output and ground truth
    y_pred = decoder_output
    y_true = X

    # cost function
    l2 = sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if not 'biases' in var.name)
    cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
    cost_reg = cost + lambda_value * l2

    # optimize function
    optimize = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)


    init = tf.global_variables_initializer()

    # Start tf.Session
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        costs=[]
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        # Restore model
        if restore_model == True:
            saver.restore(sess, model_path)
            f = open(output_file, 'a')
            f.write('Path to restored model'+ str(model_path)+ '\n')
            f.close()
            print('Restore model from',model_path)
        n_batch = math.ceil(training_numfiles/ minibatch_size)
        for epoch in range(num_epochs):
            f = open(output_file, 'a')
            cum_minibatch_cost = 0.0
            for batch in range(n_batch):

                sample = sess.run(next_element, feed_dict={handle: training_handle})
                print(sample.shape, lambda_value)
                f.write(str(sample.shape) + '\n')
                _, minibatch_cost = sess.run([optimize, cost_reg], feed_dict={X: sample})
                print(minibatch_cost)
                f.write('minibatch cost ='+str(minibatch_cost) + '\n')
                cum_minibatch_cost += minibatch_cost

            epoch_cost = cum_minibatch_cost / n_batch
            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                f.write('Cost after epoch ' + str(epoch) + ': ' + str(epoch_cost) + '\n')
                f.write('lambda value ='+str(lambda_value) + '\n')
                f.close()

            # Validation cost evaluation
            validation_total_loss = 0.0
            nv_batch = math.ceil(validation_numfiles/ minibatch_size)
            for batch in range(nv_batch):
                sample = sess.run(next_element, feed_dict={handle: validation_handle})
                f = open(output_file, 'a')

                validation_cost = sess.run(cost, feed_dict={X: sample})
                print(sample.shape)
                print("Validation cost: ", validation_cost," Lambda value used for training: ",lambda_value)
                f.write( 'validation loss' + str(validation_cost) + '\n')
                validation_total_loss =+ validation_cost
            f.write( 'Average cost on validation set' + str(validation_total_loss/nv_batch) + '\n')
            print('Average cost on validation set: ',validation_total_loss/nv_batch)
            f.close()

        # Save model
        saver_path= saver.save(sess, model_path)
        print ("Parameters have been trained!")
        print(" Model saved in path %s" % saver_path)
        # plot the cost
        #plt.plot(np.squeeze(costs))
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()





    return

