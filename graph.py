import tensorflow as tf
from matplotlib import pyplot as plt
from pprint import pprint
import os
from model import *

# ### 4.5. Build the tensorflow graph for the model

# Finally we will merge the helper functions implemented above to build a model. 
# 
# The model below should:
# 
# - create placeholders
# - initialize parameters
# - forward propagate
# - compute the cost
# - create an optimizer
# 
# We will use the adam optimizer for faster convergence
def model_graph(learning_rate=0.003, it=8):
    """
    Creates tensorflow graph for the model with cost and optimizer
    
    Arguments:
        learning_rate -- float | default 0.003 | learning rate of the optimization
        it -- int | default 8 | number of times to apply the curve transformation
    Returns:
        X -- placeholder for input data
        parameters -- a dictionary of tensors containing initiallized parameters
        Z_final -- predicted landmarks
        cost -- cost of the model
        optimizer -- adam optimizer
        learning_rate -- learnning rate of the model
        init -- tensorflow variable initializer
    
    """

    # run the graph without overwritting the variables
    tf.reset_default_graph()
    
    n_H0 = None
    n_W0 = None
    n_C0 = None
    n_HY = None
    n_WY = None
    n_CY = None
  
    # RGB images
    n_CY = n_C0 = 3
    
    # Create Placeholders of the correct shape
    X, _ = create_placeholder(n_H0, n_W0, n_C0, n_HY, n_WY, n_CY)
    
    # Initialize parameters
    parameter_dims = []
    parameter_dims.append([3, 3, 3, 32])    # 0
    parameter_dims.append([3, 3, 32, 32])   # 1
    parameter_dims.append([3, 3, 32, 32])   # 2
    parameter_dims.append([3, 3, 32, 32])   # 3
    parameter_dims.append([3, 3, 64, 32])   # 4
    parameter_dims.append([3, 3, 64, 32])   # 5
    parameter_dims.append([3, 3, 64, 3*it]) # 6
    parameters = initialize_parameters(parameter_dims)

    pprint(parameters)
        
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A, Z_final = forward_propagation(X, parameters)

    cost = exposure_control_loss(Z_final) + 1.25 * color_constency_loss(Z_final) +  (1/8) * alpha_total_variation(A) + 0.75 * spatial_consistency_loss(Z_final, X)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    global_step = tf.Variable(0, trainable=False, dtype=tf.float64)
    starter_learning_rate = learning_rate
    learning_rate = learning_rate * tf.math.pow(0.8, tf.dtypes.cast(tf.math.divide(1000, global_step), 
                                                                        dtype=tf.float32))
    # Passing global_step to minimize() will increment it at each step.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                                                                                cost, global_step=global_step)
        
    return X, parameters, Z_final, cost, optimizer, learning_rate