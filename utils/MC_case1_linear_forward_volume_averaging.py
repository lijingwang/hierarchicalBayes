#Author: Lijing Wang (lijing52@stanford.edu), 2021
## linear forward modeling: volume averaging
import numpy as np
import pandas as pd
import gstools as gs
import os
from tqdm import tqdm

def generate_m(theta, seed = None):
    model = gs.Gaussian(dim=2, 
                var= theta[1], 
                len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3)],
                angles = theta[4]*np.pi/180)
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
        
    field = srf.structured([x, y]) + theta[0]
    return field

def generate_m_MC(theta_prior):
    m_prior = np.zeros((num_sample,num_x,num_y))
    for MC_num in tqdm(range(num_sample)):
        m_prior[MC_num,:,:] = generate_m(theta_prior[MC_num,:])
    return m_prior

if __name__ == '__main__':
    path = os.path.dirname(os.getcwd()) 
    subpath = '/examples/case1_linear_forward_volume_averaging/'
    subpath = '/examples/'

    # domain size
    num_x = 70
    num_y = 30
    # x, y range
    x = range(num_x)
    y = range(num_y)

    # linear forward modeling, smooth filter
    ## linear forward modeling: G
    G = np.load(path+subpath+'G.npy')

    ## prior Monte Carlo sampling: theta
    np.random.seed(2021)
    num_sample = 5
    prior_min = np.array([28,9,15,5,0]) #np.array([28,9,15,5,0])
    prior_max = np.array([32,16,30,15,180]) #np.array([32,16,30,15,180])
    theta_prior = np.vstack((np.random.uniform(prior_min[0],prior_max[0],num_sample),
                             np.random.uniform(prior_min[1],prior_max[1],num_sample),
                             np.random.uniform(prior_min[2],prior_max[2],num_sample),
                             np.random.uniform(prior_min[3],prior_max[3],num_sample),
                             np.random.uniform(prior_min[4],prior_max[4],num_sample))).T

    ## prior Monte Carlo sampling: corresponding m

    m_prior = generate_m_MC(theta_prior)

    ## prior Monte Carlo sampling: linear forward modeling d
    d = np.dot(G,m_prior.reshape(num_sample,-1).T)


    # save
    np.save(path+subpath+'d.npy',d)
    np.save(path+subpath+'m.npy',m_prior)
    np.save(path+subpath+'theta.npy',theta_prior)


    # true model parameter
    theta_true = np.array([29,12,25,8,20])
    m_true = generate_m(theta_true,seed = 45)
    d_obs = np.dot(G,m_true.reshape(-1,1))

    np.save(path+subpath+'d_obs.npy',d_obs)
    np.save(path+subpath+'m_true.npy',m_true)
    np.save(path+subpath+'theta_true.npy',theta_true)

