# Author: Lijing Wang (lijing52@stanford.edu), 2021

import numpy as np
import pandas as pd
import gstools as gs
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
plt.rcParams.update({'font.size': 15})

import os
path = os.path.dirname(os.getcwd())
subpath = '/examples/case1_linear_forward_volume_averaging/'

# linear forward modeling: smooth filter 
G = np.load(path+subpath+'G.npy')

num_prior_sample = 5000
num_x = 70
num_y = 30
num_m = G.shape[1]
num_d = G.shape[0]
num_grid = 12
start_loc_x = 17-0.5
start_loc_y = 8-0.5

def print_theta(theta, name = 'theta'):
    theta_pd = pd.DataFrame(theta.reshape(1,-1), index = [name], columns = ['mean','variance','max_range','min_range','anisotropy'])
    print(theta_pd)

def visualize_d_2D(d):
    num_block = 3
    d_vis = np.zeros(num_m)
    d_vis[:] = np.nan
    for i in range(num_block*num_block*2):
        d_vis[np.where(G[i,:]>0)[0]] = d[i]
    d_vis = d_vis.reshape(num_x,num_y)
    return d_vis

def visualize_one_d(d, cmap = 'viridis',vmin = 20, vmax = 40):
    fig, ax = plt.subplots(figsize = [6,6])
    d_obs_show = ax.imshow(visualize_d_2D(d).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title('Measured electrical conductivity (mS/m)',fontsize = 13)
    ax.set_title('Observed data',fontsize = 13)
    fig.colorbar(d_obs_show, ax = ax, shrink = 0.3)

def visualize_one_m(m, vmin = 20, vmax = 40, cmap = 'viridis',title = 'True spatial field, m'):
    fig, ax = plt.subplots(figsize = [6,6])
    m_show = ax.imshow(m.T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title,fontsize = 13)
    fig.colorbar(m_show, ax = ax, shrink = 0.3)

def print_theta_multiple(theta, name = 'theta',head = 8):
    theta_pd = pd.DataFrame(theta, index = ['theta_'+str(i) for i in np.arange(1,theta.shape[0]+1)], columns = ['mean','variance','max_range','min_range','anisotropy'])
    print(theta_pd.head(head))

def visualize_multiple_m(m, head = 8, vmin = 20, vmax = 40, cmap = 'viridis', theta = None):
    plt.figure(figsize = [18,4])
    for i in np.arange(head):
        ax = plt.subplot(2, 4, i+1)
        ax.imshow(m[i,:].reshape(num_x,num_y).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
        rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        if theta is not None: 
            ax.set_title('\u03B8 = '+str(tuple(np.round(theta[i,:],1))))
        ax.set_xticks([])
        ax.set_yticks([])

def visualize_multiple_d(d, head = 8, vmin = 20, vmax = 40, cmap = 'viridis'):
    plt.figure(figsize = [18,4])
    for i in np.arange(head):
        ax = plt.subplot(2, 4, i+1)
        ax.imshow(visualize_d_2D(d[:,i]).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
        rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])

def visualize_multiple_d_pcs(d, head = 6, vmin = -0.5, vmax = 0.5, cmap = 'RdBu_r'):
    plt.figure(figsize = [14,5])
    for i in np.arange(head):
        ax = plt.subplot(2, 3, i+1)
        ax.imshow(visualize_d_2D(d[:,i]).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
        rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
        ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('d: PC'+str(i+1))

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def visualize_mean_var(mu, covariance, vmin = 20, vmax = 40, cmap = 'viridis'):
    var = np.diag(covariance)
    plt.figure(figsize = [18,4])
    ax = plt.subplot(2, 4, 1)
    ax.imshow(mu.reshape(num_x,num_y).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(2, 4, 2)
    ax.imshow(var.reshape(num_x,num_y).T, origin = 'lower', cmap = cmap, vmin = 0, vmax = 16)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_mean_var_MC(m, vmin = 20, vmax = 40, cmap = 'viridis'):
    mu = np.mean(m,axis = 0)
    var = np.var(m,axis = 0)
    plt.figure(figsize = [18,4])
    ax = plt.subplot(2, 4, 1)
    ax.imshow(mu.reshape(num_x,num_y).T, origin = 'lower', cmap = cmap, vmin = vmin, vmax = vmax)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(2, 4, 2)
    ax.imshow(var.reshape(num_x,num_y).T, origin = 'lower', cmap = cmap, vmin = 0, vmax = 16)
    rect = patches.Rectangle((start_loc_x,start_loc_y),num_grid, num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    rect = patches.Rectangle((start_loc_x+num_grid*2,start_loc_y),num_grid,num_grid, linewidth=2,linestyle = 'dashed', edgecolor='black',facecolor='None', label = 'pilot area')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])

    

# Visualization: updating theta
def pos_pairplot(theta_pos, theta_name):
    sns.pairplot(pd.DataFrame(theta_pos.T,columns = theta_name),kind="hist")

def prior_pos_theta(theta, theta_pos, theta_true, theta_name):
    num_theta = theta.shape[1]
    plt.figure(figsize=[25,10])
    for i in np.arange(num_theta): 
        ax = plt.subplot(2, 3, i+1)
        ax.hist(theta[:,i],density=True, bins = 1,label = 'prior',alpha = 0.7)
        y_, _, _ = ax.hist(theta_pos[i,:],density=True, bins = 20,label = 'posterior',alpha = 0.7)
        ax.vlines(x = theta_true[i], ymin = 0, ymax = np.max(y_),linestyles='--',label = 'true',color = 'black')
        ax.legend()
        ax.set_title(theta_name[i])
        ax.set_ylabel('pdf')

def ML_dimension_reduction_vis(pred_train, y_train, pred_test, y_test, S_d_obs, theta_name):
    fig = plt.figure(figsize=[24,10])
    num_theta = len(theta_name)
    for i in np.arange(num_theta): 
        ax = plt.subplot(2, 3, i+1)
        ax.plot(pred_train[:,i], y_train[:,i],'.',label = 'train')
        ax.plot(pred_test[:,i], y_test[:,i],'.',label = 'test')
        ax.vlines(x = S_d_obs[0,i],ymin = -1, ymax = 1, linestyles='--',color = 'black',zorder = 100)
        ax.plot([-1.2,1.2],[-1.2,1.2])
        ax.legend()
        ax.set_xlabel('$S(d)_'+str(i+1)+'$')
        ax.set_ylabel(theta_name[i]+'_rescaled')
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)

def history_plot(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
