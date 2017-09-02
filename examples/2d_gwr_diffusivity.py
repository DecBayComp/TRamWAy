
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

short_description = 'generate trajectories and infer diffusivity and potential maps'


method = 'gwr'
localization_error = 0.001
priorD = 0.01
priorV = 0.1
minD = -localization_error


dim = 2
D0 = 0.2
D = 1
normGradV0 = 0
normGradV = .2
name = '2d'


def random_walk(map_lower_bound, map_upper_bound, diffusivity_map, grad_force_map,
		duration=60, time_step=.05, trajectory_count=100, turn_over=.1, count_sd=3):
	dim = map_lower_bound.size
	T = int(round(float(duration) / time_step))
	K = np.round(np.random.randn(T) * count_sd + trajectory_count)
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
			kupdate = max(knew, round(min(k, X.shape[0]) * (1.0 - turn_over)))
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
			D, gradV = zip(*[ (diffusivity_map(x), grad_force_map(x)) for x in X ])
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
		gradV0 = np.full(dim, normGradV0, dtype=float)
		random_dir = np.random.randn(dim)
		random_dir /= la.norm(random_dir)
		gradV = normGradV * random_dir
		map_lower_bound = np.zeros(dim)
		map_upper_bound = np.full((dim,), 20.0)
		d_area_center = np.full((dim,), 12.0)
		d_area_radius = 3
		def diffusivity_map(x):
			d = x - d_area_center
			d = np.dot(d, d)
			return D if d <= d_area_radius * d_area_radius else D0
		v_area_center= np.full((dim,), 8.0)
		v_area_radius = 2
		out_of_bounds = 20 * np.sum(np.abs(gradV))
		def grad_force_map(x):
			if any(x < map_lower_bound) or any(map_upper_bound < x):
				# pull the particules back within the bounds
				gv = np.zeros(dim)
				gv[x < map_lower_bound] = -1
				gv[map_upper_bound < x] = 1
				return out_of_bounds * gv
			v = x - v_area_center
			s = -np.sign(np.dot(v, gradV))
			d = np.dot(v, v)
			return s * gradV if d <= v_area_radius * v_area_radius else gradV0
		def force_map(x):
			u = x - v_area_center
			d2 = np.dot(u, u)
			r2 = v_area_radius * v_area_radius
			if d2 < r2:
				e = np.dot(u, gradV) / la.norm(gradV)
				e2 = e * e
				f2 = d2 - e2
				y = sqrt(r2 - f2) - sqrt(e2)
				return gradV * y
			else:
				return gradV0 # should be 0
		def truth(cells):
			I, DF = [], []
			for i in cells.cells:
				cell = cells.cells[i]
				I.append(i)
				df = np.concatenate(([diffusivity_map(cell.center)], force_map(cell.center)))
				DF.append(df)
			DF = np.vstack(DF)
			return pd.DataFrame(index=I, data=DF, columns = [ 'diffusivity' ] + \
				[ 'force x' + str(col+1) for col in range(dim) ])
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk(map_lower_bound, map_upper_bound, diffusivity_map, grad_force_map)
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		# mesh regularly to sample ground truth for illustrative purposes
		grid = tesselate(df, method='grid', min_cell_count=10)
		cells = distributed(grid)
		true_map = cells.run(truth)
		print('ploting ground truth maps: {}'.format(out('*.truth', '.png')))
		map_plot((cells, 'true', true_map), output_file=out('truth', '.png'), show=True, aspect='equal')
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

	print("running DF inference mode...")
	DF = infer(tesselation_file, mode='DF', localization_error=localization_error)
	map_plot(DF, output_file=out(method, '.df.png'), show=True, aspect='equal')

	print("running DD inference mode...")
	DD = infer(tesselation_file, mode='DD', localization_error=localization_error, \
		priorD=priorD, min_diffusivity=minD)
	map_plot(DD, output_file=out(method, '.dd.png'), show=True, aspect='equal')

	print("running DV inference mode...")
	DV = infer(tesselation_file, mode='DV', localization_error=localization_error, \
		priorD=priorD, priorV=priorV)
	map_plot(DV, output_file=out(method, '.dv.png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()

