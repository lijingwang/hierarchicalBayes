# Author: Lijing Wang (lijing52@stanford.edu), 2021
# Solve the linear inversion problem for spatial variable m
# d = Gm + epsilon

import numpy as np
import pandas as pd
import gstools as gs


num_x = 70
num_y = 30

# Given global variable theta, obtain the covariance matrix of spatial variable m
def covariance_m(theta):
    mu = np.zeros(num_x*num_y)+theta[0]
    model = gs.Gaussian(dim=2, 
                    var= theta[1], 
                    len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3)],
                    angles = theta[4]*np.pi/180)

    x = np.arange(-num_x,num_x)
    y = np.arange(-num_y,num_y)

    xx,yy = np.meshgrid(x,y)
    cov = model.cov_spatial([xx.reshape(-1),yy.reshape(-1)])
    cov = cov.reshape(num_y*2,num_x*2).T

    C_m = np.zeros((num_x*num_y,num_x*num_y))
    for i in range(num_x):
        for j in range(num_y):
            idx = i*num_y+j
            C_m[i*num_y+j,:] = cov[(num_x-i):(num_x*2-i),(num_y-j):(num_y*2-j)].reshape(-1)
    
    return mu, C_m


# Bayes-linear-Gauss, update the posterior mean and covariance matrix
## Directly derived covariance matrix m based on theta
## Covariance matrix of epsilon are also known to us
def bayes_linear_gauss_closed_form(theta,G,d_obs,cov_e = None):
    mu, cov_m = covariance_m(theta)
    if cov_e:  # with measurement error
        K = np.linalg.multi_dot((cov_m, G.T,np.linalg.inv(np.linalg.multi_dot((G,cov_m,G.T))+cov_e)))
    else: # no measurement error
        K = np.linalg.multi_dot((cov_m, G.T,np.linalg.inv(np.linalg.multi_dot((G,cov_m,G.T)))))
    mu_pos = mu + np.dot(K,d_obs.reshape(-1)-np.dot(G,mu))
    cov_m_pos = cov_m-np.linalg.multi_dot((K,G,cov_m))
    return mu_pos,cov_m_pos

# Bayes-linear-Gauss, update the posterior mean and covariance matrix
## Estimate covariance matrix for m and epsilon from Monte Carlo samples
def bayes_linear_gauss_MC(m,d,G,d_obs):
    mu = np.mean(m, axis = 1)
    cov_m = np.cov(m)
    cov_e = np.cov(d-G.dot(m))
    K = np.linalg.multi_dot((cov_m, G.T,np.linalg.inv(np.linalg.multi_dot((G,cov_m,G.T))+cov_e)))
    mu_pos = mu + np.dot(K,d_obs.reshape(-1)-np.dot(G,mu))
    cov_m_pos = cov_m-np.linalg.multi_dot((K,G,cov_m))
    return mu_pos,cov_m_pos

# Sample from the posterior
def sample_mvn(mu, cov, num_sample = 10):
    m_sampled = np.random.multivariate_normal(mu.reshape(-1), 
                                              cov, num_sample)
    return m_sampled
