
import os
import sys
from tramway.inference import DiffusivityWarning, distributed
from tramway.helper import *
from tramway.helper.simulation import *
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


def main():
	output_basename = name
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	xyt_file = out('', 'trxyt')
	tessellation_file = out(method, '.rwa')
	new_xyt = not os.path.exists(xyt_file)
	new_tessellation = not os.path.isfile(tessellation_file)

	## define the ground truth (xyt_file)
	if new_xyt:
		gradV0 = np.full(dim, normGradV0, dtype=float)
		random_dir = np.random.randn(dim)
		random_dir /= la.norm(random_dir)
		print(random_dir)
		gradV = normGradV * random_dir
		map_lower_bound = np.zeros(dim)
		map_upper_bound = np.full((dim,), 20.0)
		d_area_center = np.full((dim,), 12.0)
		d_area_radius = 3
		def diffusivity_map(x, *args):
			d = x - d_area_center
			d = np.dot(d, d)
			return D if d <= d_area_radius * d_area_radius else D0
		v_area_center= np.full((dim,), 8.0)
		v_area_radius = 2
		out_of_bounds = 20 * np.sum(np.abs(gradV))
		def force_map(x, *args):
			if any(x < map_lower_bound) or any(map_upper_bound < x):
				# pull the particules back within the bounds
				f = np.zeros(dim)
				f[x < map_lower_bound] = 1
				f[map_upper_bound < x] = -1
				return out_of_bounds * f
			v = x - v_area_center
			s = np.sign(np.dot(v, -gradV))
			d = np.dot(v, v)
			return -s * -gradV if d <= v_area_radius * v_area_radius else -gradV0
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk(diffusivity_map, force_map, \
			duration = 10, \
			box = np.r_[map_lower_bound, map_upper_bound - map_lower_bound])
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		# mesh regularly to sample ground truth for illustrative purposes
		grid = tessellate(df, method='grid', min_location_count=10)
		cells = distributed(grid)
		true_map = cells.run(truth, diffusivity=diffusivity_map, force=force_map)
		print('ploting ground truth maps: {}'.format(out('*.truth', '.png')))
		map_plot((cells, 'true', true_map), output_file=out('truth', '.png'), show=True, aspect='equal')
		if not new_tessellation:
			print("WARNING: tessellation will overwrite file '{}'".format(tessellation_file))
			new_tessellation = True

	## tessellate (tessellation_file)
	if new_tessellation:
		tessellate(xyt_file, method, output_file=tessellation_file, \
			verbose=True, strict_min_location_count=10)
		cell_plot(tessellation_file, output_file=out(method, '.mesh.png'), \
			show=True, aspect='equal')

	## infer and plot
	# capture negative diffusivity warnings and turn them into exceptions
	warnings.filterwarnings('error', '', DiffusivityWarning)

	print("running D inference mode...")
	D_ = infer(tessellation_file, mode='D', localization_error=localization_error, \
		min_diffusivity=minD)
	map_plot(D_, output_file=out(method, '.d.png'), show=True, aspect='equal')

	print("running DF inference mode...")
	DF = infer(tessellation_file, mode='DF', localization_error=localization_error)
	map_plot(DF, output_file=out(method, '.df.png'), show=True, aspect='equal')

	print("running DD inference mode...")
	DD = infer(tessellation_file, mode='DD', localization_error=localization_error, \
		priorD=priorD, min_diffusivity=minD)
	map_plot(DD, output_file=out(method, '.dd.png'), show=True, aspect='equal')

	print("running DV inference mode...")
	DV = infer(tessellation_file, mode='DV', localization_error=localization_error, \
		priorD=priorD, priorV=priorV)
	map_plot(DV, output_file=out(method, '.dv.png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()

