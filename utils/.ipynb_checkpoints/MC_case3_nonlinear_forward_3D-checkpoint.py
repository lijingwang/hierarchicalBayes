# Author: Lijing Wang (lijing52@stanford.edu), 2022
# # nonlinear forward modeling: test

import numpy as np
import gstools as gs
from tqdm import tqdm
import flopy
import flopy.utils.binaryfile as bf
import os

## mfnwt path 
modflowpath = os.getcwd()+'/mfnwt'
path = os.path.dirname(os.getcwd()) 
subpath = '/examples/case3_nonlinear_forward_3D/'


# domain size
nrow = 200
ncol = 200
nlay = 10

# x, y range
x = range(nrow)
y = range(ncol)
z = range(nlay)

# Boundary condition: 
# North: constant head (river)
# South: no flow boundary
# West and east: general head boundary condition

# Model
# Global: mean, variance, range*2, anisotropy, vka_ratio_fines, vka_ratio_gravel
# Spatial: 200*200*10
# # Fines: top 4 layers, Gravel bottom 6 layers, in total 10. 

# Data
# D: 6 indirect head data (vertical*4, same location), 2 direct logK measurement

head_wells = np.array([[1,1,1,26,26,26],[75,100,125,75,100,125]])
direct_k_wells = np.array([[26,76],[100,50]])

def run_modflow(fname, nrow, ncol, nlay, loghk, logvka, stageleft, path = modflowpath):
    modelname = fname
    mf = flopy.modflow.Modflow(modelname, exe_name=path)

    Lx = 100.
    Ly = 100.
    ztop = 0.
    zbot = -5.

    nlay = nlay # 4 fines, 6 gravel
    nrow = nrow
    ncol = ncol
    delr = Lx/ncol # spacings along a row, can be an array
    delc = Ly/nrow # spacings along a column, can be an array
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)
    
    
    #  Dis package 
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, 
                                   delr=delr, delc=delc,
                                   top=ztop,
                                   botm=botm[1:],nper=1,perlen=1,
                                   nstp=1, steady=True)
    
    # Variables for the BAS package
    # active > 0, inactive = 0, or constant head < 0
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[0,0,:] = -1
    
    # intial head value also serves as boundary conditions
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:] = 0.
    strt[0,0,:] =  0.
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    # General boundary condition
    L = 20
    K1 = 3e-4*(3600*24)
    K2 = 2e-3*(3600*24)
    
    A = delv
    condleft = (np.hstack([[K1]*4,[K2]*6]).reshape(-1,1)*A)/L
    condright = (np.hstack([[K1]*4,[K2]*6]).reshape(-1,1)*A)/L

    stageleft = stageleft
    stageright =  -0.5
    bound_sp = []

    for il in range(4):
        for ir in range(nrow):
            bound_sp.append([il, ir, 0, stageleft, condleft[il]])
            bound_sp.append([il, ir, ncol-1, stageright, condright[il]])
    
    for il in np.arange(4)+6:
        for ir in range(nrow):
            bound_sp.append([il, ir, 0, stageleft, condleft[il]])
            bound_sp.append([il, ir, ncol-1, stageright, condright[il]])
            
    stress_period_data = {0: bound_sp}
    
    
    # Add LPF package to the MODFLOW model
    hk = np.zeros((nlay, nrow, ncol))+np.power(10,np.swapaxes(loghk,0,2))
    vka = np.zeros((nlay, nrow, ncol))+np.power(10,np.swapaxes(logvka,0,2))
    
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk*(3600*24), vka=vka*(3600*24), ipakcb=53)
    
    
    spd = {(0, 0): ['print head', 'print budget','save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)
    
    # add GHB
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

    # Write the MODFLOW model input files
    mf.write_input()

    success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
    
    if success:
        hds = bf.HeadFile(modelname+'.hds')
        times = hds.get_times() # simulation time, steady state
        head = hds.get_data(totim=times[-1])
        indirect_head = head[:,head_wells[0],head_wells[1]][[1,3,5,7]]
        direct_k = hk[:,direct_k_wells[0],direct_k_wells[1]][[1,5]]
    else: 
        indirect_head = np.zeros((4,6))
        indirect_head[:] = np.nan
        direct_k = np.zeros(4)
        direct_k[:] = np.nan

    return indirect_head,direct_k

def generate_m_3D(theta, x, y, z, seed = None):
    model = gs.Gaussian(dim=3, 
                var= theta[1], 
                len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3),2/np.sqrt(3)],
                angles = [np.pi-theta[4]*np.pi/180,0])
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
    field = srf.structured([x, y, z]) + theta[0]
    return field

def generate_m_MC(theta_prior, seed = None):
    fines_prior = np.zeros((theta_prior.shape[0],nrow,ncol,4))
    gravel_prior = np.zeros((theta_prior.shape[0],nrow,ncol,6))
    x = range(nrow)
    y = range(ncol)
    
    for MC_num in range(theta_prior.shape[0]):
        
        fines_theta = theta_prior[MC_num,:5]
        gravel_theta = theta_prior[MC_num,5:10]
        
        fines_prior[MC_num,:,:,:]  = generate_m_3D(fines_theta, x, y, range(4),seed)
        gravel_prior[MC_num,:,:,:] = generate_m_3D(gravel_theta,x, y, range(6),seed)
    
    m_prior = np.concatenate([fines_prior,gravel_prior],axis = 3)
    
    return m_prior


def generate_m_MC_same_theta(theta, num_sample):
    fines_prior = np.zeros((num_sample,nrow,ncol,4))
    gravel_prior = np.zeros((num_sample,nrow,ncol,6))
    x = range(nrow)
    y = range(ncol)
    fines_theta = theta[:5]
    gravel_theta = theta[5:10]
    
    for MC_num in tqdm(range(num_sample)):
        fines_prior[MC_num,:,:,:]  = generate_m_3D(fines_theta, x, y, range(4))
        gravel_prior[MC_num,:,:,:] = generate_m_3D(gravel_theta,x, y, range(6))
    
    m = np.concatenate([fines_prior,gravel_prior],axis = 3)
    return m


if __name__ == '__main__':
    fname = '3D_head'

    # 13 global variables
    # fines: mean, variance, rangex, rangey, anisotropy_xy
    # gravel: mean, variance, rangex, rangey, anisotropy_xy
    # vka_ratio_in_fines (1)
    # vka_ratio_in_gravel (1)
    # boundary condition of the upstream
    
    # true model parameter
    theta_true = np.array([-3.5, 0.2, 90., 50., 30., -1.5, 0.4, 60., 60., 20., -1.5, -1., 0.1]).reshape(1,-1)
    loghk_true = generate_m_MC(theta_true, seed = 100)
    
    vka_ratio = np.zeros((1,nlay))
    vka_ratio[:,:4] = theta_true[:,-3] 
    vka_ratio[:,4:] = theta_true[:,-2]  
    logvka_true = loghk_true+np.expand_dims(vka_ratio, axis=(1, 2))
    
    stageleft = theta_true[:,-1]
    
    loghk = loghk_true[0,:,:,:]
    logvka = logvka_true[0,:,:,:]
    indirect_head_true, direct_k_true = run_modflow(fname, nrow, ncol, nlay, loghk, logvka, stageleft, path = modflowpath)
    d_obs = np.hstack([indirect_head_true.reshape(-1),direct_k_true.reshape(-1)])
    
    
    np.save(path+subpath+'d_obs.npy',d_obs)
    np.save(path+subpath+'m_true.npy',loghk_true)
    np.save(path+subpath+'theta_true.npy',theta_true)
    
    
    ## prior Monte Carlo sampling: theta
    np.random.seed(0)
    num_sample = 1000
    
    prior_min = np.array([-5., 0.1, 40., 40., 0., -3., 0.1, 40., 40., 0., -3., -3., -0.25])
    prior_max = np.array([-3., 1., 100., 100., 180., -1., 1, 100., 100., 180., 0., 0., 0.25])
    
    theta_prior = np.vstack(np.random.uniform(prior_min[i],prior_max[i],num_sample) for i in range(len(prior_min))).T
    
    np.save(path+subpath+'theta.npy',theta_prior)
    
    # prior Monte Carlo sampling: corresponding m
    loghk_all = np.zeros((num_sample,nrow,ncol,nlay))
    data = np.zeros((4*6+4,num_sample))

    
    # forward modeling
    for i in tqdm(range(num_sample)):
        loghk_prior = generate_m_MC(theta_prior[i:(i+1),:])
        
        vka_ratio = np.zeros((1,nlay))
        vka_ratio[:,:4] = theta_prior[i:(i+1),-3].reshape(-1,1)
        vka_ratio[:,4:] = theta_prior[i:(i+1),-2].reshape(-1,1)
        
        logvka_prior = loghk_prior+np.expand_dims(vka_ratio, axis=(1, 2))

        stageleft = theta_prior[i:(i+1),-1]
        
        loghk = loghk_prior[0,:,:,:]
        logvka = logvka_prior[0,:,:,:]
        indirect_head, direct_k = run_modflow(fname, nrow, ncol, nlay, loghk, logvka, stageleft, path = modflowpath)
        data[:-4,i] = indirect_head.reshape(-1)
        data[-4:,i] = direct_k.reshape(-1)
        loghk_all[i,:,:,:] = loghk
        
        np.save(path+subpath+'d.npy',data[:,:(i+1)])
        np.save(path+subpath+'m.npy',loghk_all[:(i+1),:,:,:])


