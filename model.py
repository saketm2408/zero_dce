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
                                    'subpixel' : <weight tensor for subpixel conv>,
                                    'tail_0' : <weight tensor for tail>
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

def forward_propagation(X, parameters, it=8):
    '''
    Implements the forward propagation for the model:
    e.g. CONV2D -> RELU -> RESBLOCK -> RESBLOCK -> CONV2D -> SUBPIXEL_UPSAMPLING
    
    Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters the shapes are given 
                        in initialize_parameters
        train -- str | default full | can be one of (full, bicubic, div2k)
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
    for i in range(it):
        Z_final = tf.math.add(Z_final, alphas[i] * (tf.math.pow(Z_final, 2) - Z_final), name=f'y{i+1}')

    return alpha_stacked, Z_final

def alpha_total_variation(A):
    '''
    '''
    delta_h = A[:, 1:, :, :] - A[:, :-1, :, :]
    delta_w = A[:, :, 1:, :] - A[:, :, :-1, :]


    tv = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(delta_h), 2), 1) + tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(delta_w), 2), 1)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(tv, 1)) # / (tf.shape(A)[1] / 3))

    return loss

def exposure_control_loss(enhances, rsize=16, E=0.6):
    '''
    '''
    avg_intensity = tf.reduce_mean(tf.nn.avg_pool(enhances, [1, 16, 16, 1], [1, 1, 1, 1], padding='VALID', name='avg_pool_for_exposure_control_loss'), 1)
    exp_loss = tf.reduce_mean(tf.math.abs(avg_intensity - E))

    return exp_loss

def color_constency_loss(enhances):
    '''
    '''
    plane_avg = tf.reduce_mean(tf.reduce_mean(enhances, 1), 2)
    col_loss = ((plane_avg[:, 0] - plane_avg[:, 1]) ** 2 ) + ((plane_avg[:, 1] - plane_avg[:, 2]) ** 2 ) + ((plane_avg[:, 2] - plane_avg[:, 0]) ** 2 )
    col_loss = tf.reduce_mean(col_loss)

    return col_loss

def spatial_consistency_loss(enhances, originals, rsize=4):
    '''
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
    
    to_gray = tf.constant(np.array([0.3, 0.59, 0.1]).reshape((1, 1, 3, 1)), dtype=tf.dtypes.float32)

    enh_gray =  tf.nn.conv2d(enhances, to_gray, strides=[1, 1, 1, 1], padding='VALID', name='conv2D-scl1')
    orig_gray =  tf.nn.conv2d(originals, to_gray, strides=[1, 1, 1, 1], padding='VALID', name='conv2D-scl2')
    
    enh_pool = tf.nn.avg_pool(enh_gray, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    orig_pool = tf.nn.avg_pool(orig_gray, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    
    cost = tf.reduce_mean((tf.nn.conv2d(enh_pool, consistency_kernel, strides=[1, 1, 1, 1], padding='VALID') - tf.nn.conv2d(orig_pool, consistency_kernel, strides=[1, 1, 1, 1], padding='VALID')) ** 2)
    
    return cost

# ### 4.4. Cost function

# We will use simmple squared error cost function for this task. <br>
# Assumption : Every observation is independant of each other.


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A)) # '*' is elementwise mul in numpy
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_H*n_W, n_C)
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(tf.transpose(a_S)) #notice that the input of gram_matrix is A: matrix of shape (n_C, n_H*n_W)
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss
    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)
    
    
    return J_style_layer

def compute_style_cost(model_Z, model_Y, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model_Z -- vgg19 model having input layer as the SR image
    model_Y -- vgg19 model having input layer as the HR image
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        J_style_layer = compute_layer_style_cost( model_Y[layer_name],  model_Z[layer_name])

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style# cost function

def compute_content_cost(Z_final, Y):
    '''
    Computes the cost
    
    Arguments:
        Z_final -- output of forward propagation SR image
        Y -- "true" labels vector placeholder, same shape as Z_final
    
    Returns:
        cost - Tensor of the cost function
    '''
    
    J_content = tf.losses.absolute_difference(Y, Z_final)
        
    return J_content

def compute_total_cost(J_style, J_content):
    '''
    '''
    return 0.1 * J_style + J_content


def compute_bic_cost(Z_bic, Y, shape):
    '''
    '''
    Y_down = tf.image.resize_images(Y, shape, method=tf.image.ResizeMethod.BICUBIC)

    J = tf.losses.absolute_difference(Y_down, Z_bic)
        
    return J
