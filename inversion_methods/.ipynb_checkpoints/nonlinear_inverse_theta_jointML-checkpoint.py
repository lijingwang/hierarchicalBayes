# Author: Lijing Wang (lijing52@stanford.edu), 2021
# Solve the non-linear inversion problem for global variable theta
# Method: 
# # estimate the joint distribution: p(S(d), theta) and then take the conditional distribution p(theta|S(d_obs))
# # S(.) is the machine learning-assisted dimension reduction method

# Forward modeling: 
# # d = Gm + epsilon or d = g(m) + epsilon 
# # No matter what the forward model is (linear or nonlinear), the relationship between d and theta is always non-linear

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback


# Customize loss function to meet the constrain of anisotropy: 0 and 180 are the same.
# Please customize your loss function for your global variable dimension reduction
def custom_loss_function_case1(y_true,y_pred):
    num_theta = 5
    # mean squared error except anisotropy, anisotropy is the last global variable
    loss1 = K.sum(K.square(y_pred[:,:-1] - y_true[:,:-1]), axis=-1)
    
    # mean squared error for anisotropy (note here we rescale all global variables in between of -1 and 1. Suitable for activation tanh)
    # (you can also rescale them between 0 and 1. Suitable for activation sigmoid)
    
    diff_angle = K.abs(y_true[:,-1:]-y_pred[:,-1:])
    loss2 = K.sum((K.square(K.abs(tf.math.minimum(2 - diff_angle,diff_angle)))), axis=-1)
    
    return (loss1+loss2)/num_theta


def custom_loss_function_case2(y_true,y_pred):
    num_theta = 6
    # mean squared error except anisotropy, anisotropy is the last global variable
    loss1 = K.sum(K.square(y_pred[:,:4] - y_true[:,:4]), axis=-1)
    
    # mean squared error for anisotropy (note here we rescale all global variables in between of -1 and 1. Suitable for activation tanh)
    # (you can also rescale them between 0 and 1. Suitable for activation sigmoid)
    
    diff_angle = K.abs(y_true[:,4:5]-y_pred[:,4:5])
    loss2 = K.sum((K.square(K.abs(tf.math.minimum(2 - diff_angle,diff_angle)))), axis=-1)
    
    loss3 = K.sum(K.square(y_pred[:,5:] - y_true[:,5:]), axis=-1)
    
    return (loss1+loss2+loss3)/num_theta

def custom_loss_function_case3(y_true,y_pred):
    num_theta = 13
    # mean squared error except anisotropy, anisotropy is the last global variable
    loss1 = K.sum(K.square(y_pred[:,:-2] - y_true[:,:-2]), axis=-1)
    
    # mean squared error for anisotropy (note here we rescale all global variables in between of -1 and 1. Suitable for activation tanh)
    # (you can also rescale them between 0 and 1. Suitable for activation sigmoid)
    
    diff_angle = K.abs(y_true[:,-2:]-y_pred[:,-2:])
    loss2 = K.sum((K.square(K.abs(tf.math.minimum(2 - diff_angle,diff_angle)))), axis=-1)

    return (loss1+loss2)/num_theta


# Construct neural network model for ML dimension reduction
def model_tanh(num_input, num_output, custom_loss_function, l1_reg = 1e-4, l2_reg = 1e-3,learning_rate = 2.5e-3,activation = 'relu'):
    model = keras.Sequential()
    
    model.add(keras.layers.Dense(num_input, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=regularizers.l2(l2_reg), 
                                 activity_regularizer=regularizers.l2(l2_reg), activation=activation))
    model.add(keras.layers.Dense(num_input, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=regularizers.l2(l2_reg), 
                                 activity_regularizer=regularizers.l2(l2_reg), activation=activation))
    model.add(keras.layers.Dense(num_input, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=regularizers.l2(l2_reg), 
                                 activity_regularizer=regularizers.l2(l2_reg), activation=activation))
    model.add(keras.layers.Dense(num_input, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=regularizers.l2(l2_reg), 
                                 activity_regularizer=regularizers.l2(l2_reg), activation=activation))

    model.add(keras.layers.Dense(num_output,activation = 'tanh'))
    
    learning_rate = learning_rate

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.99)
    
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
   
    model.compile(optimizer=opt,
                  loss=custom_loss_function,
                  metrics=[custom_loss_function])
    return model

# ML-assisted dimension reduction
def ML_dimension_reduction(d, d_obs, theta, prior_min_theta, prior_max_theta, custom_loss_function, l1_reg = 1e-4, l2_reg = 1e-3,learning_rate = 2.5e-3, num_input = None, num_epoch = 2000, batch_size = 125):
    # d: number of sample x number of features
    # theta: number of sample x number of features
    num_d = d.shape[1]
    num_theta = theta.shape[1]
    
    # rescale d and theta
    ## remove the mean of d
    d_zero_mean = d-np.mean(d,axis = 0)
    
    ## rescale theta into [-1,1]
    theta_scaled = ((theta - prior_min_theta)/(prior_max_theta-prior_min_theta))*2 -1
    
    # split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(d-np.mean(d,axis = 0), theta_scaled, test_size=0.1, random_state = 3)
    
    # construct nn model: S()
    if num_input is None: 
        num_input = num_d
    S = model_tanh(num_input, num_theta, custom_loss_function,l1_reg = l1_reg, l2_reg = l2_reg, learning_rate = learning_rate)
    
    # train model: S()
    history = S.fit(X_train, y_train, epochs=num_epoch, batch_size=batch_size, validation_data=(X_test, y_test), 
                    verbose = 0, callbacks=[TqdmCallback(verbose=0)])
    
    # ML dimension 
    S_d = S.predict(d_zero_mean)
    S_d_obs = S.predict(d_obs.reshape(1,num_d)-np.mean(d,axis = 0))
    
    return S_d, S_d_obs, S.predict(X_train), y_train, S.predict(X_test), y_test, history

def KernelDensityEstimation(S_d, theta):
    kde = stats.gaussian_kde(np.hstack((S_d,theta)).T)
    return kde

def posterior_global_samples_jointML(S_d, S_d_obs, theta, prior_min_theta, prior_max_theta, num_conditional_sample = 100000, num_pos = 2000):
    kde = KernelDensityEstimation(S_d, theta)
    num_theta = theta.shape[1]
    
    # Conditional samples density p(theta|S(d_obs)) = p(theta, S(d_obs))/p(S(d_obs))
    kde_obs_sample = np.add(np.zeros((num_conditional_sample,num_theta)),S_d_obs.reshape(1,-1)).T
    for i in range(num_theta):
        kde_obs_sample = np.vstack((kde_obs_sample,
                                    np.random.uniform(prior_min_theta[i],prior_max_theta[i],num_conditional_sample)))
    density = kde(kde_obs_sample)
    density = density/np.sum(density)
    
    # sample posterior based on the conditional probability p(theta|S(d_obs))
    posterior_idx = np.random.choice(num_conditional_sample, 
                                     num_pos,
                                     p = density)
    theta_pos_MC = kde_obs_sample[num_theta:,posterior_idx]
    
    return theta_pos_MC
