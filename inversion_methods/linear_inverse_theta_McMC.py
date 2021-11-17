# Author: Lijing Wang (lijing52@stanford.edu), 2021
# Inverse global variable theta with linear forward modeling (McMC)
# Method: 
## Derive the likelihood directly and use McMC to sample

import numpy as np
import pymc3 as pm
import gstools as gs
from theano import *
import theano
import theano.tensor as tt

# Initialize random number generator
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)

def rotation_matrix(anisotropy):
    angle = anisotropy*theano.shared(np.pi/180)
    r_matrix = tt.stacklists([[pm.math.cos(angle),pm.math.sin(angle)],[-pm.math.sin(angle),pm.math.cos(angle)]])
    return r_matrix

def McMC(G, d_obs):
    # domain size
    num_x = 70
    num_y = 30

    # x,y range, for covariance matrix calculation
    x = np.arange(-num_x,num_x)
    y = np.arange(-num_y,num_y)
    xx,yy = np.meshgrid(x,y)
    xx = xx.T
    yy = yy.T

    # linear forward modeling, smooth filter
    ## linear forward modeling: G
    idx_m_list = np.where(np.sum(G,axis = 0)>0)[0] # idx of m impacts d

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        mean = pm.Uniform("mean", lower = 28, upper=32)
        variance = pm.Uniform("variance", lower = 9, upper=16)
        max_range = pm.Uniform("max_range", lower = 15, upper=30)
        min_range = pm.Uniform("min_range", lower = 5, upper=15) 
        anisotropy = pm.Uniform("anisotropy", lower = 0, upper=180)

        # Expected value of outcome
        mu = pm.math.dot(G,np.zeros(G.shape[1])+mean)

        ## calculate the covariance matrix 
        h = pm.math.dot(rotation_matrix(anisotropy),
                        tt.stacklists([xx.reshape(-1),yy.reshape(-1)]))/tt.stacklists([max_range,min_range]).dimshuffle(0,'x')

        cov_entire = variance*tt.exp(-(np.pi/4)*(pm.math.sqrt(h[0,:]**2+h[1,:]**2)*np.sqrt(3))**2) #np.sqrt(3)
        cov_entire = tt.reshape(cov_entire,[num_x*2,num_y*2])
        C_m = tt.zeros((num_x*num_y,num_x*num_y))
        for i in range(num_x):
            for j in range(num_y):
                idx = i*num_y+j
                if idx in idx_m_list:
                    C_m = tt.inc_subtensor(C_m[i*num_y+j,:],tt.reshape(cov_entire[(num_x-i):(num_x*2-i),(num_y-j):(num_y*2-j)],[-1]))

        cov = pm.math.dot(pm.math.dot(G,C_m),G.T)

        # Likelihood
        Y_obs = pm.MvNormal("d_obs", mu=mu, cov=cov, observed=d_obs.reshape(-1))

    with basic_model:
        trace = pm.sample(2000, chains = 1,cores = 1, tune = 1500, 
                          start={'mean': np.array(30.), 'variance': np.array(12.5),
                                 'max_range': np.array(22.5), 'min_range': np.array(10.),
                                 'anisotropy': np.array(90.)}, return_inferencedata=False)
    
    return trace