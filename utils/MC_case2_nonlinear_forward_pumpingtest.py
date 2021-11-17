# Author: Lijing Wang (lijing52@stanford.edu), 2021
# # nonlinear forward modeling: pumping test

import numpy as np
import pandas as pd
import gstools as gs
from tqdm import tqdm
import flopy
import flopy.utils.binaryfile as bf
import os

## mfnwt path 
modflowpath = os.getcwd()+'/mfnwt'
path = os.path.dirname(os.getcwd()) 
subpath = '/examples/case2_nonlinear_forward_pumping_test/'


# domain size
nrow = 100
ncol = 100

# x, y range
x = range(nrow)
y = range(ncol)

# pumping test info
pumping_rate = -10.
well_location = [49,49]
observed_wells = [[41,41],[57,57],[41,57],[57,41]]


# -

def run_modflow(fname, nrow, ncol, hwest, loghk, pumping_rate, well_location, observed_wells, path = modflowpath):
    modelname = fname
    mf = flopy.modflow.Modflow(modelname, exe_name=path)
    
    # Define the model grid
    # A horizontal confined aquifer (1000 x 1000 x 50 m) with constant
    # head on the western and eastern boundaries (hwest = hwest m, heast = 0
    # m), no flow condition on northern and southern boundaries.
    # Horizontal and vertical hydraulic conductivity are given by 10 m/d.

    Lx = 1000.
    Ly = 1000.
    ztop = 0.
    zbot = -50.

    nlay = 1
    nrow = nrow
    ncol = ncol
    delr = Lx/ncol # spacings along a row, can be an array
    delc = Ly/nrow # spacings along a column, can be an array
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)
    
    # Time step parameters
    nper = 3 # number of stress periods
    perlen = [1, 2, 4] # stress period lengths
    nstp = [10, 20, 40] # time steps per period
    steady = [True, False, False]
    
    # Create the discretization object
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:], nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    
    # Variables for the BAS package
    # active > 0, inactive = 0, or constant head < 0
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    
    # intial head value also serves as boundary conditions
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] =  hwest
    strt[:, :, -1] = 10.
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    # Add LPF package to the MODFLOW model
    # hk array of horizontal hydraulic conductivity. vka array of vertical hydraulic conductivity, ipakcb file number writing for cell-by-cell budget(need to be defined for the current version)
    hk = np.zeros((nlay, nrow, ncol))+np.power(10,loghk)
    hk[:,0,:] = 1e-18
    hk[:,-1,:] = 1e-18
    
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=1, ipakcb=53)
    
    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)
    
    # Add the well package
    # Remember to use zero-based layer, row, column indices!
    pumping_rate = pumping_rate
    
    # Define source terms - e.g. a pumping well
    wel_sp1 = [[0, well_location[0], well_location[1], 0.]]
    wel_sp2 = [[0, well_location[0], well_location[1], pumping_rate]]
    wel_sp3 = [[0, well_location[0], well_location[1], 0]]
    wel_stress_period_data = {0:wel_sp1, 1:wel_sp2, 2:wel_sp3}
    oc_stress_period_data = {} # Output control
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            oc_stress_period_data[(kper, kstp)] = ['save head', 'save drawdown', 'save budget']

    wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_stress_period_data)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=oc_stress_period_data, compact=True)

    # Write input files and run
    mf.write_input()
    success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
    
    if success:
        # Read output files
        headobj = bf.HeadFile(modelname + '.hds')
        times = headobj.get_times()
        before_pumping = np.squeeze(headobj.get_data(totim = times[0])) # squeeze to remove z dimension in 2D model
        after_pumping = np.squeeze(headobj.get_data(totim = times[-1]))
        time_series_at_pumping_well = headobj.get_ts((0, well_location[0], well_location[1]))

        time_series_observed = []
        for well_loc in observed_wells:
            time_series_observed.append(headobj.get_ts((0, well_loc[0], well_loc[1])))
        drawdown_curves = np.array([time_series_at_pumping_well[:,1],
                                    time_series_observed[0][:,1],
                                    time_series_observed[1][:,1],
                                    time_series_observed[2][:,1],
                                    time_series_observed[3][:,1]]).T.reshape(-1)
    else: 
        drawdown_curves = np.nan
    return drawdown_curves


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
    m_prior = np.zeros((num_sample,nrow,ncol))
    for MC_num in tqdm(range(num_sample)):
        m_prior[MC_num,:,:] = generate_m(theta_prior[MC_num,:])
    return m_prior


def generate_m_MC_same_theta(theta, num_sample):
    m = np.zeros((num_sample,nrow,ncol))
    for MC_num in tqdm(range(num_sample)):
        m[MC_num,:,:] = generate_m(theta)
    return m


if __name__ == '__main__':
    fname = 'pumping_test'
    
    ## prior Monte Carlo sampling: theta
    np.random.seed(0)
    num_sample = 5000
    
    # mean, variance, max_range, min_range, anisotropy, head_west
    prior_min = np.array([-3,0.1,15,5,0, 8])
    prior_max = np.array([-1,1,30,15,180,12]) 
    theta_prior = np.vstack((np.random.uniform(prior_min[0],prior_max[0],num_sample),
                             np.random.uniform(prior_min[1],prior_max[1],num_sample),
                             np.random.uniform(prior_min[2],prior_max[2],num_sample),
                             np.random.uniform(prior_min[3],prior_max[3],num_sample),
                             np.random.uniform(prior_min[4],prior_max[4],num_sample),
                             np.random.uniform(prior_min[5],prior_max[5],num_sample))).T

    # prior Monte Carlo sampling: corresponding m
    loghk_prior = generate_m_MC(theta_prior)
    
    drawdown_prior = np.zeros((70*5,num_sample))
    
    # forward modeling
    for i in tqdm(range(num_sample)):
        hwest = theta_prior[i,-1]
        loghk = loghk_prior[i,:,:]
        drawdown_curves = run_modflow(fname, nrow, ncol, 
                                      hwest, loghk, pumping_rate, 
                                      well_location, observed_wells, 
                                      path = modflowpath)
        drawdown_prior[:,i] = drawdown_curves
    
    # save
    np.save(path+subpath+'d.npy',drawdown_prior)
    np.save(path+subpath+'m.npy',loghk_prior)
    np.save(path+subpath+'theta.npy',theta_prior)

    # true model parameter
    
    theta_true = np.array([-1.6,0.3,20,7,120,11])
    loghk_true = generate_m(theta_true,seed = 90)
    drawdown_obs = run_modflow(fname, nrow, ncol, 
                               theta_true[-1], loghk_true, pumping_rate, 
                               well_location, observed_wells, 
                               path = modflowpath).reshape(-1,1)
    
    np.save(path+subpath+'d_obs.npy',drawdown_obs)
    np.save(path+subpath+'m_true.npy',loghk_true)
    np.save(path+subpath+'theta_true.npy',theta_true)

    



