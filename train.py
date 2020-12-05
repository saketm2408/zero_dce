import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto

import matplotlib.pyplot as plt
import cv2

import os
import sys
import random
import pickle
import collections
from collections import defaultdict
from pprint import pprint
import math
import time
import random
import pickle as p

import mllib
from graph import *
from model import *

# ### 4.6. Train the model

# Finally we will create a session and run a for loop  for num_epochs, get the mini-batches, and then for each mini-batch you will optimize the function.

def minimize_model(X_train, X_val, save_dir, learning_rate=0.06,
                   num_epochs=100, start_epoch=0, minibatch_size=64, pretrained=None,
                   print_cost=True, show_result=True, show_inferences=True, compute_time=True):
    """
    Implements a SR using ConvNet in Tensorflow. Trains the model inside the sessionand validates the model
    
    Arguments:
        X_train -- training set, of shape (None, 64, 64, 3)
        X_val -- validation set, of shape (None, 64, 64, 3) or None
        save_dir -- str | dir to save weights
        num_epochs -- number of epochs of the optimization loop
        start_epoch -- int starting epoch
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        show_result -- True to show image after every 5 epoch
        show_inferences --
        compute_time -- 
        train -- str | default 'full' | one of full, content, bicubic, perceptual
    
    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        Y_val_pred -- predicted lebels for the validation set
        Y_val_test -- predicted lebels for the test set
    """
    # create the tensorflow graph for the model
    graph = model_graph(X_train, X_val, learning_rate=learning_rate, pretrained=pretrained, get_pd=False)
    
    # store the dimenssions of the training set
    if type(X_train) != dict:
        if type(X_train) != list:
            (m, n_H0, n_W0, n_C0) = X_train.shape
        else:
            m = len(X_train)
    else:
        if type(X_train) == list:
            m = len(X_train)
    n_C0 = n_CY = 3

    costs = []
    time_diff_bp = []
    time_diff_fp = []
    original_learning_rate = learning_rate
    Y_val_pred = []
    Y_pred = []
    val_cost = []
    train_psnr = []
    val_psnr = []
    
    # unroll graph
    X = graph[0]
    parameters = graph[1]
    Z_final = graph[2]
    cost = graph[3]
    optimizer = graph[4]
    learning_rate = graph[5]

    
    
    # create directory to save the model
    if not os.path.exists(save_dir+'/'):
        os.makedirs(save_dir+'/')
    
    # number of minibatches of size minibatch_size in the train set
    if type(X_train) != dict:
        num_minibatches_train = int(m / minibatch_size)
    
    saver = tf.train.Saver()
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    
    # start the session
    sess = tf.Session(config=config)

    # Run the initialization
    sess.run(init)
    
    # Do the training loop
    print('Training on images of same size.')
    for epoch in range(start_epoch, num_epochs):
        minibatch_cost = 0
            
        # divide the training set into 'num_minibatches' minibatches
        # minibatches = mllib.random_mini_batches(X_train, Y_train, minibatch_size)
        minibatches = mllib.random_mini_batches_from_SSD_no_gt(X_train, minibatch_size)
        minibatches_val = mllib.random_mini_batches_no_gt(X_val, minibatch_size)
        minibatch_processed = 0
        # iterate over the minibaches
        for minibatch in minibatches:
            # Select a minibatch
            minibatch_X = minibatch
            minibatch_X = mllib.load_images_into_array_from_list_of_dirs(minibatch_X)
                
            # Run the session to execute the optimizer and the cost, the feedict should contain a 
            # minibatch for (X,Y). (Run the graph of a single minibatch)
            start_time = time.time()
            # _  temp_cost_bic = sess.run([optimizer[0], cost_bic], feed_dict={X:minibatch_X, Y:minibatch_Y})
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X})
            end_time = time.time()
            # temp_cost = sess.run(cost, feed_dict={X:minibatch_X, Y:minibatch_Y})
            
            # calculate time taken for back propagation
            time_diff_bp.append(end_time - start_time)
            
            # update minibatch cost
            minibatch_cost += temp_cost / num_minibatches_train
            if minibatch_processed > 0:
                sys.stdout.write("\033[K")
                    
            print(f'{minibatch_processed}/{num_minibatches_train} minibatches processed | cost - {temp_cost} | image - {minibatch[0][0].split("/")[-1]}', end="\r")

            minibatch_processed += 1
            
        if print_cost == True :
            costs.append(minibatch_cost)
            sys.stdout.write("\033[K")
            print ("Cost after epoch %i: %f | lr: %f" % (epoch, minibatch_cost, sess.run(learning_rate))) 
        #
        if epoch == (num_epochs-1) or (epoch % 1 == 0):
            # save the model
            print('Saving Progress...', end='\r')
            Y_pred = sess.run(Z_final, feed_dict={X:minibatch_X})
            print(f'max = {np.max(Y_pred[0])} | min = {np.min(Y_pred[0])} | mean = {np.mean(Y_pred[0])} | unique = {np.unique(Y_pred[0])}')
            cv2.imwrite(save_dir+f'/demo1_'+str(epoch)+'.png', Y_pred[0] * 255)
            cv2.imwrite(save_dir+f'/demo2_'+str(epoch)+'.png', Y_pred[-1] * 255)
            #tf.train.write_graph(sess.graph_def, './',
            #         'cnn_NAS'+str(epoch)+'.pb', as_text=False)
            save_path = saver.save(sess, save_dir+f'/10ksr_'+str(epoch)+'.ckpt')
            parameters_dict = sess.run(parameters) 
            with open(save_dir+f'/zdce_'+str(epoch)+'.pickle', 'wb') as handle:
                p.dump(parameters_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Progress saved in '+save_path, end='\r')
        
    print()
    parameters = sess.run(parameters) 
    # Y_pred = sess.run(Z_final, feed_dict={X:X_train[:10]})
    # Y_val_pred, val_cost = sess.run([Z_final, cost], feed_dict={X:X_val[:10], Y:Y_val[:10]})    
    # close the session
    sess.close()
                
    return  parameters

if __name__ == '__main__':

    # ll_dir_train = "/home/ubuntu/efs/saket/datasets/SICE/train/"
    ll_dir_train = '/home/ubuntu18/Projects/low_light/metric/LOLdataset/our485/low/'

    # ll_dir_val = "/home/ubuntu/efs/saket/datasets/SICE/val/"
    ll_dir_val = '/home/ubuntu18/Projects/low_light/metric/LOLdataset/our485/val/high/'


    # ### 2.2. Load the dataset

    print('Train set...')
    # Y_train = mllib.load_image_names_into_list(ol_dir_train, f=True)
    X_train = mllib.load_image_names_into_list(ll_dir_train, f=True)


    print('Val set...')
    # Y_val = mllib.load_images_into_array(ol_dir_val)
    X_val = mllib.load_images_into_array(ll_dir_val)


    # ### 4. Check the dim of the dataset


    print ("number of training examples = " + str(len(X_train)))
    print ("number of validation examples = " + str(X_val.shape[0]))
    print ("X_val shape: " + str(X_val.shape))
    # print ("Y_val shape: " + str(Y_val.shape))

    # with open("./saved/4/10ksr_32.pickle", "rb") as input_file:i
    #   pretrained = p.load(input_file)
    pretrained = None

    parameter = minimize_model(X_train, X_val, save_dir='./exdark',
                                start_epoch=0, num_epochs=501, minibatch_size=1, learning_rate=0.001, pretrained=pretrained,
                                print_cost=True, show_result=True, show_inferences=True)
