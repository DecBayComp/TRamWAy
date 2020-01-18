import numpy as np 
import pandas as pd

import sys
sys.path.append("..")
import argparse
from   os.path import abspath
import gc
import os
####################################################################
####################################################################
####################################################################

####################################################################
####################################################################
####################################################################
def load_xyt(path_xyt):


	colonne_names = ['x', 'y','t'] 
	xyt_pandas     = pd.read_csv(path_xyt, header=None,sep='\t', names=colonne_names)
	xyt            = xyt_pandas.values


	return xyt, xyt_pandas


####################################################################
####################################################################
####################################################################
def convert_to_list(xyt):

	t_unique        = np.unique(xyt[:,2])
	n_unique        = t_unique.shape[0]
	movie_per_frame = []
	for i in range(n_unique):

		II = xyt[:,2]==t_unique[i]
		movie_per_frame.append(xyt[II,:])

	return movie_per_frame, n_unique	

####################################################################
####################################################################
####################################################################
def get_dt(movie_per_frame, n_unique):

	ddtt    = []
	dt_theo = movie_per_frame[1][0,2] - movie_per_frame[0][0,2]

	for i in range(n_unique - 1):
		ddtt.append(movie_per_frame[i+1][0,2] - movie_per_frame[i][0,2])

	dt_theo = min(ddtt)
	t_init  = movie_per_frame[0][0,2]
	t_end   = movie_per_frame[n_unique - 1][0,2]

	return dt_theo, t_init, t_end

####################################################################
####################################################################
####################################################################
def get_local_parameters_accessible_elsewhere_in_tramway(dt_theo):

	sigma        = 0.025
	D_high       = 0.5
	length_high  = 2.*np.sqrt(2*D_high*dt_theo)


	return sigma, D_high, length_high
####################################################################
####################################################################
####################################################################
def get_cost_function(number_movie, movie_per_frame):


	x2 = movie_per_frame[number_movie+1][:,0]
	x1 = movie_per_frame[number_movie][:,0]

	y2 = movie_per_frame[number_movie+1][:,1]
	y1 = movie_per_frame[number_movie][:,1]

	dx = x2[:,np.newaxis] - x1
	dy = y2[:,np.newaxis] - y1

	C  = dx**2 + dy**2


	return C


####################################################################
####################################################################
####################################################################
def give_optimal_assigment(C, length_high):
	##. cutoff
	(M,N)            = C.shape
	l2               = length_high**2
	C[C>l2]          = np.inf
	## where there is a possibility to do a match
	non_inf          = ~np.isinf(C)
	num_col          = np.sum(non_inf,axis=0)
	num_row          = np.sum(non_inf,axis=1)

	row_reduced      = np.where(num_row != 0);
	col_reduced      = np.where(num_col != 0);
	CC               = np.squeeze(C[row_reduced,:])
	## reduced matrix with non-inf value
	CC[np.isinf(CC)] = np.amax(CC , where=~np.isinf(CC) , initial=-1)
	## optimal assingment
	row_ind, col_ind = linear_sum_assignment(CC)

	assingment       = np.zeros((M,N))
	assingment[row_reduced,col_ind]     = 1

	return C

####################################################################
####################################################################
####################################################################





