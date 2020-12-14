import numpy as np 
import pandas as pd

import sys
import argparse
from   os.path import abspath
import gc
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

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
	D_high       = 0.3
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
	C = C.transpose()

	return C


####################################################################
####################################################################
####################################################################
def correct_cost_function(C,length_high):


	(M,N)            = C.shape
	# cut the links
	l2               = length_high**2
	C[C>l2]          = np.inf
	# positions whith no link possible
	non_inf          = ~np.isinf(C)
	num_col          = np.sum(non_inf,axis=0)
	num_row          = np.sum(non_inf,axis=1)
	## portion that need to be optimised
	row_reduced      = np.where(num_row != 0);
	col_reduced      = np.where(num_col != 0);
	n_row_reduced    = np.size(row_reduced)
	n_col_reduced    = np.size(col_reduced)

	nn               = np.maximum(n_row_reduced,n_col_reduced)

	if (n_row_reduced>1)&(n_col_reduced > 1):  
		CC               = np.squeeze(C[row_reduced,:])
		CC               = np.squeeze(CC[:,col_reduced])
		#square the matrix
		nn               = np.maximum(n_row_reduced,n_col_reduced)
		C_reduced        = np.zeros((nn,nn))
		C_reduced[0:n_row_reduced,0:n_col_reduced] = CC[:,:]
		d_max            = np.amax(C_reduced , where=~np.isinf(C_reduced) , initial=-1)

		# adjust the isze of the matrix to ensure reasonnable assugments
		edge             = np.zeros((nn,nn)) 
		edge[:,:]        = C_reduced[:,:]
		non_inf          = ~np.isinf(C_reduced)
		edge[non_inf]    = 0
		n_add            = correct_deficiencies(edge)
		nn               = nn + n_add
		## corrected matric cost function 

		C_reduced_corrected  = np.ones((nn,nn))*d_max
		C_reduced_corrected[0:n_row_reduced,0:n_col_reduced] =  CC[:,:]
		##elements usefull for final assigment 

		row_reduced      = np.squeeze(np.array(row_reduced))
		col_reduced      = np.squeeze(np.array(col_reduced))

		inf_loc          = np.isinf(C_reduced_corrected)
		C_reduced_corrected[inf_loc] = d_max*1e10
		anomaly          = 0

	elif (n_row_reduced==0)&(n_col_reduced==0):

		C_reduced_corrected = np.zeros(1)
		C_reduced           = np.zeros(1)
		edge                = np.zeros(1)
		d_max               = np.inf
		row_reduced         = np.zeros(1)
		col_reduced         = np.zeros(1)
		n_row_reduced       = 0
		n_col_reduced       = 0
		anomaly             = 2

	elif (n_row_reduced==1):
		#print("here row")
		#print()
		CC               = np.squeeze(C[row_reduced,:])
		CC               = np.squeeze(CC[col_reduced])
		#square the matrix
		nn               = np.maximum(n_row_reduced,n_col_reduced)
		C_reduced        = np.zeros((n_row_reduced,nn))
		#C_reduced        = np.zeros((nn,nn))
		#print((n_row_reduced, n_col_reduced))
		if nn == 1:
			C_reduced[0,0] = CC
		else:
			C_reduced[0,0:n_col_reduced] = CC[:]
			#C_reduced[0:n_row_reduced,0:n_col_reduced] = CC[:]		
			#print(C_reduced)

		d_max            = np.amax(C_reduced , where=~np.isinf(C_reduced) , initial=-1)

		# adjust the isze of the matrix to ensure reasonnable assugments
		edge = []
		## corrected matric cost function 
		if nn==1:
			C_reduced_corrected = d_max
		else:
			C_reduced_corrected  = C_reduced
		
		##elements usefull for final assigment 
		row_reduced      = np.squeeze(np.array(row_reduced))
		col_reduced      = np.squeeze(np.array(col_reduced))
		if nn>1:
			inf_loc          = np.isinf(C_reduced_corrected)
			C_reduced_corrected[inf_loc] = d_max*1e10
		anomaly          = 1

	elif (n_col_reduced==1):
		#print("here col")

		CC               = np.squeeze(C[row_reduced,:])
		CC               = np.squeeze(CC[:,col_reduced])
		#square the matrix
		nn               = np.maximum(n_row_reduced,n_col_reduced)
		C_reduced        = np.zeros((n_row_reduced,n_col_reduced))
		#C_reduced        = np.full((nn, nn), np.inf)
		C_reduced[0:n_row_reduced,0] = CC[:]
		d_max            = np.amax(C_reduced , where=~np.isinf(C_reduced) , initial=-1)

		# adjust the isze of the matrix to ensure reasonnable assugments
		edge = []
		## corrected matric cost function 
		if nn==1:
			C_reduced_corrected = d_max
		else:
			C_reduced_corrected  = C_reduced

		##elements usefull for final assigment 

		row_reduced      = np.squeeze(np.array(row_reduced))
		col_reduced      = np.squeeze(np.array(col_reduced))

		if nn>1:
			inf_loc          = np.isinf(C_reduced_corrected)
			C_reduced_corrected[inf_loc] = d_max*1e10
		anomaly          = 1



	return C_reduced_corrected, C_reduced, edge, d_max, row_reduced, col_reduced,M,N, n_row_reduced, n_col_reduced, anomaly

####################################################################
####################################################################
####################################################################
def get_assigment_matrix_from_reduced_cost(C_reduced_corrected,row_reduced,col_reduced,M,N,n_col_reduced, n_row_reduced,anomaly):
	## most cases
	if (anomaly == 0):

		row_ind, col_ind = linear_sum_assignment(C_reduced_corrected)
		global_row       = []
		global_col       = []
		assingment       = np.zeros((M,N))

		for i in range(len(row_reduced)):
			if col_ind[i] < n_col_reduced:
				coll    = col_reduced[col_ind[i]]
				index   = row_reduced[i]
					
				global_row.append(index)
				global_col.append(coll)
				assingment[index,coll]  = 1 
		global_row = np.array(global_row)
		global_col = np.array(global_col)

	# cases where only one column or one row survived
	elif (anomaly ==1):

		if (n_col_reduced==1)&(n_row_reduced==1):

			row_ind     = 0
			col_ind     = 0
			global_col  = col_reduced
			global_row  = row_reduced
			assingment  = np.zeros((M,N))
			assingment[global_row,global_col] = 1

		elif (n_col_reduced==1):
			row_ind, col_ind = linear_sum_assignment(C_reduced_corrected)

			#row_ind     = 0
			#col_ind     = 0
			global_col  = col_reduced
			global_row  = row_reduced[row_ind]
			assingment  = np.zeros((M,N))
			assingment[global_row,global_col] = 1

		elif(n_row_reduced==1):
			row_ind, col_ind = linear_sum_assignment(C_reduced_corrected)

			#row_ind     = 0
			#col_ind     = 0
			global_col  = col_reduced[col_ind]
			global_row  = row_reduced
			assingment  = np.zeros((M,N))
			assingment[global_row,global_col] = 1

		else:
		
			row_ind, col_ind = linear_sum_assignment(C_reduced_corrected)
			global_row       = []
			global_col       = []
			assingment       = np.zeros((M,N))

			for i in range(len(row_reduced)):
				if col_ind[i] < n_col_reduced:
					coll    = col_reduced[col_ind[i]]
					index   = row_reduced[i]
					
					global_row.append(index)
					global_col.append(coll)
					assingment[index,coll]  = 1 
			global_row = np.array(global_row)
			global_col = np.array(global_col)



	# no link can be made evrything is impossible : zero assigment
	elif anomaly == 2:
		assingment       = np.zeros((M,N))
		global_row       = np.array([])
		global_col       = np.array([])



	return assingment, global_row, global_col

####################################################################
####################################################################
####################################################################
def plot_linking_two_images(movie_per_frame, indice, row_ind, col_ind):
	## red first frame
	## blue second frame

	xyt1 = movie_per_frame[indice]
	xyt2 = movie_per_frame[indice+1]

	plt.scatter(xyt1[:,0],xyt1[:,1], s=5,  c='r')
	plt.scatter(xyt2[:,0],xyt2[:,1], s=10, c='b')

	dx      = np.squeeze( xyt2[col_ind,0] - xyt1[row_ind,0] )
	dy      = np.squeeze( xyt2[col_ind,1] - xyt1[row_ind,1] )
	x_start = np.squeeze( xyt1[row_ind,0] )
	y_start = np.squeeze( xyt1[row_ind,1] )
	#print(x_start, y_start, dx, dy)

	try:
		for idx in range( len(dx) ):
			plt.arrow(x_start[idx],y_start[idx], dx[idx] ,dy[idx] , fc='k', ec='k',head_width=0.5 )
	except TypeError:
			plt.arrow(x_start,y_start, dx ,dy , fc='k', ec='k',head_width=0.5 )




	#plt.axis('scaled')
	plt.axis('scaled')
	plt.show()


	return 1
####################################################################
####################################################################
####################################################################
def plot_linking_from_list(movie_per_frame, indice, liste_row, liste_col ):
	## red first frame
	## blue second frame

	xyt1 = movie_per_frame[indice]
	xyt2 = movie_per_frame[indice+1]

	row_ind  = liste_row[indice]
	col_ind  = liste_col[indice]	

	plt.scatter(xyt1[:,0],xyt1[:,1], s=5,  c='r')
	plt.scatter(xyt2[:,0],xyt2[:,1], s=10, c='b')

	dx      = np.squeeze( xyt2[col_ind,0] - xyt1[row_ind,0] )
	dy      = np.squeeze( xyt2[col_ind,1] - xyt1[row_ind,1] )
	x_start = np.squeeze( xyt1[row_ind,0] )
	y_start = np.squeeze( xyt1[row_ind,1] )
	#print(x_start, y_start, dx, dy)

	try:
		for idx in range( len(dx) ):
			plt.arrow(x_start[idx],y_start[idx], dx[idx] ,dy[idx] , fc='k', ec='k',head_width=0.5 )
	except TypeError:
			plt.arrow(x_start,y_start, dx ,dy , fc='k', ec='k',head_width=0.5 )




	#plt.axis('scaled')
	plt.axis('scaled')
	plt.show()



	return 1
####################################################################
####################################################################
####################################################################
def plot_linking_3_from_list(movie_per_frame, indice, liste_row, liste_col ):
	## red first frame
	## blue second frame

	xyt1 = movie_per_frame[indice]
	xyt2 = movie_per_frame[indice+1]
	xyt3 = movie_per_frame[indice+2]


	row_ind  = liste_row[indice]
	col_ind  = liste_col[indice]	

	plt.scatter(xyt1[:,0],xyt1[:,1], s=5,  c='r')
	plt.scatter(xyt2[:,0],xyt2[:,1], s=10, c='b')

	dx      = np.squeeze( xyt2[col_ind,0] - xyt1[row_ind,0] )
	dy      = np.squeeze( xyt2[col_ind,1] - xyt1[row_ind,1] )
	x_start = np.squeeze( xyt1[row_ind,0] )
	y_start = np.squeeze( xyt1[row_ind,1] )
	#print(x_start, y_start, dx, dy)

	try:
		for idx in range( len(dx) ):
			plt.arrow(x_start[idx],y_start[idx], dx[idx] ,dy[idx] , fc='k', ec='k',head_width=0.5 )
	except TypeError:
			plt.arrow(x_start,y_start, dx ,dy , fc='k', ec='k',head_width=0.5 )




	#plt.axis('scaled')
	plt.axis('scaled')
	plt.show()



	return 1


####################################################################
####################################################################
####################################################################
def correct_deficiencies(edge):


	(K_edge, L_edge) = edge.shape
	#print((K_edge, L_edge))
	row              = np.zeros((K_edge,1))
	col              = np.zeros((K_edge,1))
	edge_eff         = np.zeros((K_edge,K_edge))



	for ii in range(K_edge):
		for jj in range(K_edge):
			if (edge[ii,jj]==0)&(row[ii]==0)&(col[jj]==0):
				edge_eff[ii,jj] = 1
				row[ii]=1
				col[jj]=1
				break


	row = np.zeros((K_edge,1))
	indicateur_1 = 1

	while indicateur_1:
		row_loc = -1
		col_loc = -1
		indicateur_2 = 1
		ii      = 0
		jj      = 0
		#print(indicateur_1)
		while indicateur_2:
			if (edge[ii,jj]==0)&(row[ii]==0)&(col[jj]==0):
				row_loc = ii
				col_loc = jj
				indicateur_2 = 0

			jj = jj + 1
			if (jj>=K_edge):
 #				print("here jj\n")
				jj=0
				ii=ii+1
				#print(ii)
			if (ii>=K_edge):
 #				print("here ii \n")
				indicateur_2=0

		if row_loc ==-1:
			#print("here")
			indicateur_1 = 0
		else:
			#print((row_loc,col_loc))
			edge_eff[row_loc,col_loc] = 2
			III = edge_eff[row_loc,:] ==1
			#print(III)
			if (np.sum(III) !=0):
				row[row_loc]  = 1
				col_col2      = edge_eff[row_loc,:]==1
				col[col_col2] = 0
			else:
				#print("here\n")
				indicateur_1 = 0
		#print(indicateur_1)




	n_add = K_edge - np.sum(col) - np.sum(row)
	return int(n_add)


####################################################################
####################################################################
####################################################################
#def give_optimal_assigment(C, length_high):
	##. cutoff
#	(M,N)            = C.shape
#	l2               = length_high**2
	#print(np.sqrt(l2))
#	C[C>l2]          = np.inf
	## where there is a possibility to do a match
#	non_inf          = ~np.isinf(C)
#	num_col          = np.sum(non_inf,axis=0)
#	num_row          = np.sum(non_inf,axis=1)

#	row_reduced      = np.where(num_row != 0);
#	col_reduced      = np.where(num_col != 0);
#	CC               = np.squeeze(C[row_reduced,:])

#	row_reduced      = np.squeeze(np.array(row_reduced))
	#print(row_reduced.shape)
	## reduced matrix with non-inf value
#	CC[np.isinf(CC)] = np.amax(CC , where=~np.isinf(CC) , initial=-1)

	## optimal assingment
#	row_ind, col_ind = linear_sum_assignment(CC)
	#print(row_ind.shape)
#	index            = row_reduced[row_ind]
#	assingment       = np.zeros((M,N))
#	assingment[index,col_ind]     = 1

#	return assingment, index, col_ind
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





####################################################################
####################################################################
####################################################################


