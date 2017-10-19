
import os
import sys
from tramway.inference import DiffusivityWarning, distributed
from tramway.helper import *
import matplotlib.pyplot as plt
import warnings
#warnings.simplefilter('error')
import numpy as np
import pandas as pd
import numpy.linalg as la
from math import *

short_description = 'temporal regularization with predefined temporal mesh'


name = 'regular_temporal_mesh'

method = 'grid'
localization_error = 0.001
priorD = 0.01
#priorV = 0.1
minD = -localization_error


D0 = 0.1 # um2.s-1
D = 0.05 # um2.s-1

R = .2 # um

dim = 2
x0 = y0 = t0 = 0.
width = height = 3. * R # um
duration = 10 # s
time_step = .05 # s


_box = (x0, y0, width, height)

def diffusivity_map(xy, t=None, box=None):
	if box is None:
		box = _box
	x0, y0, width, height = box
	if t is None:
		t = np.arange(t0, duration, time_step)
	x_radius = y_radius = R
	t_radius = duration * .3
	r1 = np.asarray((x0 + width * .5, y0 + height * .5, t0 + duration * .5))
	r05 = np.asarray((x_radius, y_radius, t_radius))
	if np.isscalar(t_radius):
		xyt = np.r_[xy, t]
	else:
		raise NotImplementedError
	return D0 - (D-D0) * np.exp(-np.sqrt(np.sum(( (xyt - r1) / r05 )**2.)) * -np.log(.05))

def force_map(xy, t=None, box=None):
	return np.zeros((dim,))

def grad_force_map(xy, t=None, box=None):
	return np.zeros((dim,))

def truth(cells, ts=None):
	if ts is None:
		ts = np.arange(t0, duration, time_step)
	I, DF = [], []
	for i in cells.cells:
		cell = cells.cells[i]
		I.append(i)
		df = np.concatenate(([diffusivity_map(cell.center, ts)], force_map(cell.center, ts)))
		DF.append(df)
	DF = np.vstack(DF)
	return pd.DataFrame(index=I, data=DF, columns = [ 'diffusivity' ] + \
		[ 'force x' + str(col+1) for col in range(dim) ])


map_lower_bound = np.asarray((x0, y0))
map_upper_bound = np.asarray((x0 + width, y0 + height))


def random_walk(trajectory_mean_count=100, trajectory_count_sd=3, turn_over=.1):
	dim = map_lower_bound.size
	T = int(round(float(duration) / time_step))
	K = np.round(np.random.randn(T) * trajectory_count_sd + trajectory_mean_count)
	T = np.arange(time_step, duration + time_step, time_step)
	if K[1] < K[0]: # switch so that K[0] <= K[1]
		tmp = K[0]
		K[0] = K[1]
		K[1] = tmp
	if K[-2] < K[-1]: # switch so that K[end-1] >= K[end]
		tmp = K[-1]
		K[-1] = K[-2]
		K[-2] = tmp
	xs = []
	X = np.array([])
	knew = n0 = 0
	for t, k in zip(T, K):
		k = int(k)
		if t == duration:
			kupdate = k
		elif X.size:
			kupdate = max(knew, int(round(min(k, X.shape[0]) * (1.0 - turn_over))))
		else:
			kupdate = 0
		knew = k - kupdate
		if knew == 0:
			if not X.size:
				raise RuntimeError
			Xnew = np.zeros((knew, dim), dtype=X.dtype)
			nnew = np.zeros((knew, ), dtype=n.dtype)
		else:
			Xnew = np.random.rand(knew, dim) * (map_upper_bound - map_lower_bound) + map_lower_bound
			nnew = np.arange(n0, n0 + knew)
			n0 += knew
		if X.size:
			X = X[:kupdate]
			D, gradV = zip(*[ (diffusivity_map(x, t), grad_force_map(x, t)) for x in X ])
			D, gradV = np.array(D), np.array(gradV)
			dX = -gradV + D.reshape(D.size, 1) * np.random.randn(*X.shape)
			X = np.concatenate((Xnew, X + dX))
			n = np.concatenate((nnew, n[:kupdate]))
		else:
			X = Xnew
			n = nnew
		if np.unique(n).size < n.size:
			print((n, kupdate, knew, t))
			raise RuntimeError
		xs.append(np.concatenate((n.reshape(n.size, 1), X, np.full((n.size, 1), t)), axis=1))
	columns = 'xyz'
	if dim <= 3:
		columns = [ d for d in columns[:dim] ]
	else:
		columns = [ 'x'+str(i) for i in range(dim) ]
	columns = ['n'] + columns + ['t']
	data = np.concatenate(xs, axis=0)
	data = data[np.lexsort((data[:,-1], data[:,0]))]
	return pd.DataFrame(data=data, columns=columns)


def main():
	output_basename = name
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	xyt_file = out('', 'trxyt')
	tesselation_file = out(method, '.rwa')
	new_xyt = not os.path.exists(xyt_file)
	new_tesselation = not os.path.isfile(tesselation_file)

	## define the ground truth (xyt_file)
	if new_xyt:
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk()
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		## mesh regularly to sample ground truth for illustrative purposes
		#grid = tesselate(df, method='grid', min_cell_count=10)
		#cells = distributed(grid)
		#true_map = cells.run(truth)
		#print('ploting ground truth maps: {}'.format(out('*.truth', '.png')))
		#map_plot((cells, 'true', true_map), output_file=out('truth', '.png'), show=True, aspect='equal')
		if not new_tesselation:
			print("WARNING: tesselation will overwrite file '{}'".format(tesselation_file))
			new_tesselation = True

	## tesselate (tesselation_file)
	if new_tesselation:
		tesselate(xyt_file, method, output_file=tesselation_file, \
			verbose=True, strict_min_cell_size=10)
		cell_plot(tesselation_file, output_file=out(method, '.mesh.png'), \
			show=True, aspect='equal')

	## infer and plot
	# capture negative diffusivity warnings and turn them into exceptions
	warnings.filterwarnings('error', '', DiffusivityWarning)

	print("running D inference mode...")
	D_ = infer(tesselation_file, mode='D', localization_error=localization_error, \
		min_diffusivity=minD)
	map_plot(D_, output_file=out(method, '.d.png'), show=True, aspect='equal')

	#print("running DF inference mode...")
	#DF = infer(tesselation_file, mode='DF', localization_error=localization_error)
	#map_plot(DF, output_file=out(method, '.df.png'), show=True, aspect='equal')

	print("running DD inference mode...")
	DD = infer(tesselation_file, mode='DD', localization_error=localization_error, \
		priorD=priorD, min_diffusivity=minD)
	map_plot(DD, output_file=out(method, '.dd.png'), show=True, aspect='equal')

	#print("running DV inference mode...")
	#DV = infer(tesselation_file, mode='DV', localization_error=localization_error, \
	#	priorD=priorD, priorV=priorV)
	#map_plot(DV, output_file=out(method, '.dv.png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()

