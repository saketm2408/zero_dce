import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def create_placeholder(n_H0=None, n_W0=None, n_C0=None, n_HY=None, n_WY=None, n_CY=None, m=None):
    '''
    Creates the placeholders for the tensorflow session
    
    Arguments :
    
    Returns:
        X -- placeholder for the input data of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y -- placeholder for the input landmark of shape [None, n_HY, n_WY, n_CY]
    '''
    X = tf.placeholder(tf.float32, [m, n_H0, n_W0, n_C0], name='x')
    Y = tf.placeholder(tf.float32, [m, n_HY, n_WY, n_CY], name='y_')
    
    return X, Y


# ### 4.2. Initialize parameters

def initialize_parameters(parameter_dims):
    '''
    Initializes weight parameters to build a neural network with tensorflow. eg.
    Arguments :
        parameter_dims -- list containing the dimenssions of the weight matrics
                
    Returns:
        parameters --  a dictionary of tensors containing weights
                    parameter = {
                                    'conv2D-0' : <weight tensor for  conv>,
                                    'conv2D-1' : <weight tensor for  conv>
                                }
    '''
    parameters = {}
    parameters['conv2D-0'] = tf.get_variable('conv2D-0', parameter_dims[0], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-1'] = tf.get_variable('conv2D-1', parameter_dims[1], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-2'] = tf.get_variable('conv2D-2', parameter_dims[2], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-3'] = tf.get_variable('conv2D-3', parameter_dims[3], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-4'] = tf.get_variable('conv2D-4', parameter_dims[4], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-5'] = tf.get_variable('conv2D-5', parameter_dims[5], initializer=tf.contrib.layers.xavier_initializer())
    parameters['conv2D-6'] = tf.get_variable('conv2D-6', parameter_dims[6], initializer=tf.contrib.layers.xavier_initializer())
   
    return parameters

# ### 4.3. Forward propagation

def forward_propagation(X, parameters, it=8, outputs=[4, 6, 8]):
    '''
    Implements the forward propagation for the model. This function is inspired by Zero DCE proposed by 
    Guo1 et. al.
    
    Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters the shapes are given 
                        in initialize_parameters
        it -- int | default 8 | number of times to apply the curve transformation
    Returns:
        Z_final -- the output of the last unit
    '''         
    Z1 = tf.nn.conv2d(X, parameters['conv2D-0'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-0')
    Z1 = tf.nn.relu(Z1, name='conv2D-0-relu')
    
    Z2 = tf.nn.conv2d(Z1, parameters['conv2D-1'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-1')
    Z2 = tf.nn.relu(Z2, name='conv2D-1-relu')

    Z3 = tf.nn.conv2d(Z2, parameters['conv2D-2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-2')
    Z3 = tf.nn.relu(Z3, name='conv2D-2-relu')

    Z4 = tf.nn.conv2d(Z3, parameters['conv2D-3'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-3')
    Z4 = tf.nn.relu(Z4, name='conv2D-3-relu')

    Z5 = tf.concat((Z4, Z3), axis=-1, name='conv2D-4-concat')
    Z5 = tf.nn.conv2d(Z5, parameters['conv2D-4'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-4')
    Z5 = tf.nn.relu(Z5, name='conv2D-4-relu')

    Z6 = tf.concat((Z5, Z2), axis=-1, name='conv2D-5-concat')
    Z6 = tf.nn.conv2d(Z6, parameters['conv2D-5'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-5')
    Z6 = tf.nn.relu(Z6, name='conv2D-5-relu')

    Z7 = alpha_stacked = tf.concat((Z6, Z1), axis=-1, name='conv2D-6-concat-alpha_stacked')
    Z7 = alpha_stacked = tf.nn.conv2d(Z7, parameters['conv2D-6'], strides=[1, 1, 1, 1], padding='SAME', name='conv2D-6-alpha_stacked')
    Z7 = alpha_stacked = tf.nn.tanh(Z7, name='conv2D-6-tanh-alpha_stacked')

    alphas = tf.split(alpha_stacked, num_or_size_splits=it, axis=-1)

    Z_final = X
    Z_final_list = []
    for i in range(it):
        Z_final = tf.math.add(Z_final, alphas[i] * (tf.math.pow(Z_final, 2) - Z_final), name=f'y{i+1}')
        if it in outputs:
            Z_final_list.append(Z_final)

    return alpha_stacked, Z_final_list

def alpha_total_variation(A):
    '''
    1. This function calculates total variation loss. 
    2. Total Variation or TV loss is a no reference loss function originally implemented by Guo1 et. al. in the
        paper Zero DCE.
    3. The intuition behind this loss is to preserve the monotonicity relations between neighboring pixels.

    Args ::
        A - tensor | image transformation coefs
    Return ::
        loss - float | the value of the loss
    '''
    delta_h = A[:, 1:, :, :] - A[:, :-1, :, :]
    delta_w = A[:, :, 1:, :] - A[:, :, :-1, :]


    tv = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(delta_h), 2), 1) + tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(delta_w), 2), 1)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(tv, 1)) # / (tf.shape(A)[1] / 3))

    return loss

def exposure_control_loss(enhances, rsize=16, E=0.6):
    '''
    1. This loss helps restraining under-/over-exposed regions of the input image.
    2. This is also a no reference loss function originally implemented by Guo1 et. al. in the paper Zero DCE.
    N.B. - I have found that this loss function can be implented in other image generation tasks and
        it has the ability to get rid of artifacts (blocking artifacts to be specific)

    Args ::
        enhances - tensor | reformed image (output) at various iterations
        rsize - int | default 16 | denotes the size of the window to adjust the intensity | an hyperparameter
                            that can be tuned
        E - float | default 0.6 | the av. intensity of the ouput image | this hypeerparameter can also be tuned
                            but I have found that 0.6 give the best results as suggested by the authors
    '''
    avg_intensity = tf.reduce_mean(tf.nn.avg_pool(enhances, [1, rsize, rsize, 1], [1, 1, 1, 1], padding='VALID', name='avg_pool_for_exposure_control_loss'), 1)
    exp_loss = tf.reduce_mean(tf.math.abs(avg_intensity - E))

    return exp_loss

def color_constency_loss(enhances):
    '''
    This loss function is inspired by the Gray-World color constancy hypothesis which states that
    color in each sensor channel averages to gray over the entire image.

    Args ::
        enhances - tensor | reformed image (output) at various iterations
    '''
    plane_avg = tf.reduce_mean(tf.reduce_mean(enhances, 1), 2)
    col_loss = ((plane_avg[:, 0] - plane_avg[:, 1]) ** 2 ) + ((plane_avg[:, 1] - plane_avg[:, 2]) ** 2 ) + ((plane_avg[:, 2] - plane_avg[:, 0]) ** 2 )
    col_loss = tf.reduce_mean(col_loss)

    return col_loss

def spatial_consistency_loss(enhances, originals, rsize=4):
    '''
    This loss function encourages spatial coherence of the enhanced image through preserving the difference 
    of neighboring regions between the input image and its enhanced version. 

    Arguments : 
        enhances - tensor | reformed image (output) at various iterations
        originals - tensor | original image taken in suboptimal lighting condition
        rsize - int | default 16 | denotes the size of the window to copare the difference
    ''' 
    consistency_kernel = tf.constant(np.array([[[[ 0.,  0.,  0.,  0.]],
                [[-1.,  0.,  0.,  0.]],
                [[ 0.,  0.,  0.,  0.]]],

             [[[ 0.,  0., -1.,  0.]],
                [[ 1.,  1.,  1.,  1.]],
                [[ 0.,  0.,  0., -1.]]],

            [[[ 0.,  0.,  0.,  0.]],
                [[ 0., -1.,  0.,  0.]],
                [[ 0.,  0.,  0.,  0.]]]]), dtype=tf.dtypes.float32)
    
    ## N.B. the array [0.3, 0.59, 0.1] is also a hyper-parameter, which can be tuned.
    to_gray = tf.constant(np.array([0.3, 0.59, 0.1]).reshape((1, 1, 3, 1)), dtype=tf.dtypes.float32)

    enh_gray =  tf.nn.conv2d(enhances, to_gray, strides=[1, 1, 1, 1], padding='VALID', name='conv2D-scl1')
    orig_gray =  tf.nn.conv2d(originals, to_gray, strides=[1, 1, 1, 1], padding='VALID', name='conv2D-scl2')
    
    enh_pool = tf.nn.avg_pool(enh_gray, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    orig_pool = tf.nn.avg_pool(orig_gray, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    
    cost = tf.reduce_mean((tf.nn.conv2d(enh_pool, consistency_kernel, strides=[1, 1, 1, 1], padding='VALID') - tf.nn.conv2d(orig_pool, consistency_kernel, strides=[1, 1, 1, 1], padding='VALID')) ** 2)
    
    return cost
