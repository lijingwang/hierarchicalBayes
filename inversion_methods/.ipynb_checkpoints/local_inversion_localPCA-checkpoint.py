# Author: Lijing Wang
# Contact: lijing52@stanford.edu
# Created: Feb 28, 2019

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def PCA_fast(input_array, n_component = None, name = 'model', scale = True):
	'''
	Dimension Reduction with Principal Component Analysis by fast EVD
	Before Perform PCA, we need standardize the input
	
	Args: 
		input_array: (np.array) the array you want to do dimension reduction, # grids * # realizations
		n_component: (int) # component you would like to keep

	Output:
		PCA_output
	'''
	input_array = np.transpose(input_array)

	num_realizations = input_array.shape[0]

	if n_component is None:
		n_component = np.min(input_array.shape)

	# Perform Standard Scaler:
	# Standardize features by removing the mean and scaling to unit variance

	if scale:
		scaler = StandardScaler()
		scaler.fit(input_array)
		input_array_after_scale = scaler.transform(input_array)
	else: 
		scaler = StandardScaler(with_std=False)
		scaler.fit(input_array)
		input_array_after_scale = scaler.transform(input_array)


	## EVD for XtX
	XtX = np.matmul(input_array_after_scale,input_array_after_scale.T)

	eigVal, eigVector_v = np.linalg.eigh(XtX)
	order = eigVal.argsort()[::-1]
	eigVal = eigVal[order]
	eigVector_v = eigVector_v[:,order]
	
	# Output: 
	PCA_output = {}

	## Explained Variance
	PCA_output['explained_variance'] = eigVal/np.sum(eigVal)


	## Eigen vectors/images/arrays
	eigVector_u = np.matmul(input_array_after_scale.T, eigVector_v)
	eigVector_u_norm = np.linalg.norm(eigVector_u,axis = 0,keepdims=True)
	PCA_output['eigVector_u_norm'] = eigVector_u_norm
	PCA_output['eigen_vectors'] = eigVector_u/eigVector_u_norm
	PCA_output['eigVector_v'] = eigVector_v

	## PC scores (# pcs * # realizations)
	PCA_output['pc_scores'] = np.matmul(PCA_output['eigen_vectors'].T, input_array_after_scale.T)
	PCA_output['pc_scores'] 

	if n_component == num_realizations:
		PCA_output['eigen_vectors'][:, num_realizations-1] = 0
		PCA_output['pc_scores'][num_realizations-1,:] = 0

	if scale:
		## Mean before standardization
		PCA_output['standard_mean'] = scaler.mean_

		## Var before standardization
		PCA_output['standard_var'] = scaler.var_
		
	else: 
		PCA_output['standard_mean'] = scaler.mean_

	return PCA_output

def projection(PCA_output, new_array,scale = True):
	'''
	Project our realizations given existing PCA fit. 
	
	Args: 
		PCA_output: (np.array) the array you want to do dimension reduction, # grids * # realizations
		new_array: (np.array) # grids * # realizations, new arrays project to existing PC spaces

	Output:
		new_PC_scores(after_PCA): (np.array) # pcs * # realizations
	'''

	## Standard Scaler
	if scale:
		after_scaler = (new_array-PCA_output['standard_mean'].reshape(-1,1))/np.sqrt(PCA_output['standard_var'].reshape(-1,1))
	else:
		after_scaler = (new_array-PCA_output['standard_mean'].reshape(-1,1))
	## PCA 
	after_PCA = np.matmul(PCA_output['eigen_vectors'].T, after_scaler)

	return after_PCA


def local_PCA(input_array, local_indicator, project = True):
	ind_all = np.zeros(input_array.shape[0]).reshape(-1,1)
	ind_all[np.isnan(input_array[:,0])] = np.nan

	nan_value_local = np.where(~np.isnan(local_indicator.reshape(-1)))
	nan_value = np.where(~np.isnan(ind_all.reshape(-1)))

	array_no_nan_local = input_array[nan_value_local[0],:]
	array_no_nan = input_array[nan_value[0],:]

	global_PCA = PCA_fast(array_no_nan)

	local_PCA  = PCA_fast(array_no_nan_local)

	scaler = StandardScaler()
	scaler.fit(array_no_nan.T)
	input_array_after_scale = scaler.transform(array_no_nan.T)

	eigVector_u_all = np.matmul(input_array_after_scale.T,local_PCA['eigVector_v'])

	eigVector_u_norm = local_PCA['eigVector_u_norm']

	local_PCA['eigen_vectors_all'] = eigVector_u_all/eigVector_u_norm

	return  (global_PCA,local_PCA,nan_value,nan_value_local)

def reconstruction(PCA_output, pc_scores=None, n_component=None):
	'''
	Reconstruct our realizations given pc_scores
	
	Args: 
		PCA_output: (np.array) the array you want to do dimension reduction, # grids * # realizations
		pc_scores: (np.array) # pcs * # realizations, the prior/posterior pc scores you would like to do reconstruction 
		n_component: (int) # component you would like to do reconstruction

	Output:
		reconstructed_arrays(before_scaler): # grids * # realizations
	'''
	if pc_scores is None:
		# use the pc_scores from PCA
		pc_scores = PCA_output['pc_scores']

	if n_component is None:
		n_component = pc_scores.shape[0]

	## inverse PCA
	before_PCA = np.matmul(PCA_output['eigen_vectors'][:,:n_component],pc_scores[:n_component,:])

	## inverse Standard Scaler
	before_scaler = before_PCA*np.sqrt(PCA_output['standard_var'].reshape(-1,1)) + PCA_output['standard_mean'].reshape(-1,1)

	return before_scaler

def reconstruction_localPCA(global_PCA, local_PCA, pc_scores=None, n_component=None):
	'''
	Reconstruct our realizations given pc_scores
	
	Args: 
		PCA_output: (np.array) the array you want to do dimension reduction, # grids * # realizations
		pc_scores: (np.array) # pcs * # realizations, the prior/posterior pc scores you would like to do reconstruction 
		n_component: (int) # component you would like to do reconstruction

	Output:
		reconstructed_arrays(before_scaler): # grids * # realizations
	'''
	if pc_scores is None:
		# use the pc_scores from PCA
		pc_scores = local_PCA['pc_scores']

	if n_component is None:
		n_component = pc_scores.shape[0]

	## inverse PCA
	before_PCA = np.matmul(local_PCA['eigen_vectors_all'][:,:n_component],pc_scores[:n_component,:])

	## inverse Standard Scaler
	before_scaler = before_PCA*np.sqrt(global_PCA['standard_var'].reshape(-1,1)) + global_PCA['standard_mean'].reshape(-1,1)

	return before_scaler



