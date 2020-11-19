import numpy as np 
import pandas as pd

import argparse
from   os.path import abspath
import gc
import os

from scipy.optimize      import linear_sum_assignment
from .file_processing_loc import *


####################################################################
####################################################################
####################################################################
def traj_from_assigment(liste_assingment,Min_length_traj, number_init):



	return 1
####################################################################
####################################################################
####################################################################
def position_and_displacement_from_assignment(liste_assingment, liste_row, liste_col,movie_per_frame,filename):


	
	liste = []
	indice = 0
	for (row, col) in zip(liste_row,liste_col):
		if np.size(row) == 0:
			1
		else:
			x  = movie_per_frame[indice][row,0]
			y  = movie_per_frame[indice][row,1]
			t  = movie_per_frame[indice][row,2]
			dx = movie_per_frame[indice+1][col,0] - movie_per_frame[indice][row,0]
			dy = movie_per_frame[indice+1][col,1] - movie_per_frame[indice][row,1]
			dt = movie_per_frame[indice+1][col,2] - movie_per_frame[indice][row,2]
			#f.write("%e,%e,%e,%e,%e,%e\n" % (x,y,t,dx,dy,dt))
			out = np.vstack((x,y,t,dx,dy,dt))
			out = np.transpose(out)
			liste.append(out)
			del out
		indice = indice + 1
			#x = 
	liste_array = np.vstack(liste)

	pd.DataFrame(liste_array).to_csv(filename, header=None, index=0, sep="\t", float_format='%e')


	return liste_array 


####################################################################
####################################################################
####################################################################
def get_all_assigment_stack(path_xyt):

	## load files and provide para
	xyt, xyt_pandas            = load_xyt(path_xyt)
	movie_per_frame, n_unique  = convert_to_list(xyt)
	dt_theo, t_init, t_end     = get_dt(movie_per_frame, n_unique)
	sigma, D_high, length_high = get_local_parameters_accessible_elsewhere_in_tramway(dt_theo)

	liste_assingment  = []
	liste_row         = []
	liste_col         = []

	for indice in range(n_unique-1):
		#if (indice>18000):
		#	print((indice,n_unique))
		#print(indice)
		if (indice%1000)==0:
			print((indice,n_unique))

		#try:
		C        = get_cost_function(indice, movie_per_frame)
		C_eff, _, _, _, row_eff, col_eff,M,N, n_row_eff, n_col_eff,anomaly = correct_cost_function(C,length_high)
		assingment, global_row, global_col = get_assigment_matrix_from_reduced_cost(C_eff,row_eff,col_eff,M,N,n_col_eff,n_row_eff,anomaly)
		liste_assingment.append(assingment)
		liste_row.append(global_row)
		liste_col.append(global_col)
		#except IndexError:
		#	print(indice)

	return liste_assingment, liste_row, liste_col

####################################################################
####################################################################
####################################################################



####################################################################
####################################################################
####################################################################
if __name__ == "__main__":

	warnings.filterwarnings('ignore')

    # parse the input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--server-name', type=str, help="server name", required=True)
	args = parser.parse_args()




####################################################################
####################################################################
####################################################################



####################################################################
####################################################################
####################################################################



####################################################################
####################################################################
####################################################################


####################################################################
####################################################################
####################################################################



