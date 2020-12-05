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

def minimize_model(X_train, X_val, Y_val, outputs=[4, 6, 8], it=8, save_dir='./save/', learning_rate=0.06,
                   num_epochs=100, start_epoch=0, minibatch_size=64,
                   print_cost=True):
    """
    Implements a SR using ConvNet in Tensorflow. Trains the model inside the sessionand validates the model
    
    Arguments:
        X_train -- training set, of shape (None, h, w, 3)
        X_val -- validation set, of shape (None, h, w, 3) 
        X_val -- validation set, of shape (None, h, w, 3) 
        outputs ,it -- model hyperparameters | see model.py for more info
        save_dir -- str | dir to save weights
        num_epochs -- number of epochs of the optimization loop
        start_epoch -- int starting epoch
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs    
    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.

    """
    # create the tensorflow graph for the model
    graph = model_graph(outputs=[4, 6, 8], it=8, learning_rate=learning_rate)
    
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
    X, parameters, Z_final, cost, optimizer, learning_rate = graph
   
    
    # create directory to save the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # number of minibatches of size minibatch_size in the train set
    m = len(X_train)
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
    for epoch in range(start_epoch, num_epochs):
        minibatch_cost = 0
            
        # divide the training set into 'num_minibatches' minibatches
        minibatches = mllib.random_mini_batches_from_SSD_no_gt(X_train, minibatch_size)
        minibatches_val = mllib.random_mini_batches(X_val, Y_val, minibatch_size)
        minibatch_processed = 0

        # iterate over the minibaches
        for minibatch in minibatches:
            # Select a minibatch
            minibatch_X = mllib.load_images_into_array_from_list_of_dirs(minibatch)
                
            # Run the session to execute the optimizer and the cost, the feedict should contain a 
            # minibatch for (X,Y). (Run the graph of a single minibatch)
            start_time = time.time()
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X})
            end_time = time.time()
            
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
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost)) 
        #
        if epoch == (num_epochs-1) or (epoch % 1 == 0):
            # save the model
            print('Saving Progress...', end='\r')

            # save one of the train inferences 
            Y_pred = sess.run(Z_final[-1], feed_dict={X:minibatch_X})
            print(f'max = {np.max(Y_pred[0])} | min = {np.min(Y_pred[0])} | mean = {np.mean(Y_pred[0])} | unique = {np.unique(Y_pred[0])}')
            cv2.imwrite(save_dir+f'/demo1_'+str(epoch)+'.png', Y_pred[0] * 255)

            #tf.train.write_graph(sess.graph_def, './',
            #         'cnn_NAS'+str(epoch)+'.pb', as_text=False)
            save_path = saver.save(sess, os.path.join(save_dir, f'zdce_{epoch}.ckpt'))
            parameters_dict = sess.run(parameters) 
            with open(os.path.join(save_dir, f'zdce_{epoch}.pickle'), 'wb') as handle:
                p.dump(parameters_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Progress saved in {save_path}', end='\r')

        # evaluate the model after every 5 epochs
        if epoch == 0 and epoch % 5 == 0:
            val_psnr = [0 for i in range(len(Z_final))]
            for minibatch in minibatches_val:
                # Select a minibatch
                minibatch_X, minibatch_Y = minibatch
                # val_enhances = sess.run(Z_final, feed_dict={X:minibatch_X})

                # calculate PSNR
                for i in range(len(Z_final)):
                    val_psnr[i] += sess.run(tf.reduce_mean(tf.image.psnr(Z_final[i], minibatch_Y, 1.0)), feed_dict={X:minibatch_X})

            val_pasn = [val_psnr[i]/len(minibatches_val) for i in range(len(val_psnr))]
            print()
            print(f'Dev set PSNR after {epoch} epochs as (iteration, value) pairs = {list(zip(outputs, val_psnr))}')
    
    print()
    parameters = sess.run(parameters) 
    # Y_pred = sess.run(Z_final, feed_dict={X:X_train[:10]})
    # Y_val_pred, val_cost = sess.run([Z_final, cost], feed_dict={X:X_val[:10], Y:Y_val[:10]})    
    # close the session
    sess.close()
                
    return  parameters

if __name__ == '__main__':

    # ll_dir_train = "/home/ubuntu/efs/saket/datasets/SICE/train/"
    ll_dir_train = '../LOLdataset/our485/mix/'

    X_val_dir = '../LOLdataset/our485/val/low/'
    Y_val_dir = '../LOLdataset/our485/val/high/'


    # Load the dataset
    print('Train set...')
    X_train = mllib.load_image_names_into_list(ll_dir_train, f=True)


    print('Val set...')
    X_val = mllib.load_images_into_array(X_val_dir)
    Y_val = mllib.load_images_into_array(Y_val_dir)


    # Check the dim of the dataset
    print ("number of training examples = " + str(len(X_train)))
    print ("number of validation examples = " + str(X_val.shape[0]))
    print ("X_val shape: " + str(X_val.shape))

    parameter = minimize_model(X_train, X_val,Y_val, outputs=[4, 6, 8], it=8, save_dir='./exdark',
                                start_epoch=0, num_epochs=101, minibatch_size=12, learning_rate=0.001,
                                print_cost=True)
