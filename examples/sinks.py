
import os
import sys
from tramway.inference import DiffusivityWarning, distributed
from tramway.helper import *
from tramway.helper.simulation import *
import warnings
#warnings.simplefilter('error')
import numpy as np
from math import *

short_description = 'generate trajectories and infer diffusivity and potential maps'


method = 'gwr'
localization_error = 0.001
priorD = 0.01
priorV = 0.01
minD = -localization_error


dim = 2
D0 = .5
D = .2
normGradV = 5.
name = 'sinks'


def main():
	output_basename = name
	def out(label, extension):
		return '{}.{}.{}.{}'.format(output_basename, method, label, extension)

	xyt_file = output_basename + '.trxyt'
	rwa_file = output_basename + '.rwa'
	new_xyt = not os.path.exists(xyt_file)
	new_tessellation = not os.path.isfile(rwa_file)

	## define the ground truth (xyt_file)
	d_area_center = np.full((dim,), .8)
	d_area_radius = .15
	def diffusivity_map(x, *args):
		d = x - d_area_center
		d = np.dot(d, d)
		return D if d <= d_area_radius * d_area_radius else D0
	v_area_center= np.full((dim,), .4)
	v_area_radius = .2
	v_ring_width = .06
	def force_map(x, *args):
		f = v_area_center - x
		d = np.sqrt(np.dot(f, f))
		f *= normGradV / d
		return f if abs(d - v_area_radius) <= v_ring_width else np.zeros(dim)
	if new_xyt:
		map_lower_bound = np.zeros(dim)
		map_upper_bound = np.full((dim,), 1.)
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk(diffusivity_map, force_map, \
			trajectory_mean_count = 200, turnover = .3, duration = 10, \
			box = np.r_[map_lower_bound, map_upper_bound - map_lower_bound])
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		## mesh regularly to sample ground truth for illustrative purposes
		#grid = tessellate(df, method='grid', min_location_count=10)
		if not new_tessellation:
			print("WARNING: tessellation will overwrite file '{}'".format(rwa_file))
			new_tessellation = True

	## tessellate (tessellation_file)
	if new_tessellation:
		cells = tessellate(xyt_file, method, output_file=rwa_file, \
			verbose=True, strict_min_location_count=10, force=True)
		cell_plot(rwa_file, output_file=out('mesh', 'png'), \
			show=True, aspect='equal')
		# show ground truth
		true_map = distributed(cells).run(truth, diffusivity=diffusivity_map, force=force_map)
		print('ploting ground truth maps: {}'.format(out('truth', 'png')))
		map_plot(true_map, cells=cells, output_file=out('truth', 'png'), show=True, aspect='equal')

	## infer and plot
	# capture negative diffusivity warnings and turn them into exceptions
	warnings.filterwarnings('error', '', DiffusivityWarning)

	#print("running D inference mode...")
	#D_ = infer(rwa_file, mode='D', localization_error=localization_error, \
	#	min_diffusivity=minD, output_label='D')
	#map_plot(D_, output_file=out('d', 'png'), show=True, aspect='equal')

	print("running DF inference mode...")
	DF = infer(rwa_file, mode='DF', localization_error=localization_error, output_label='DF')
	map_plot(DF, output_file=out('df', 'png'), show=True, aspect='equal')

	#print("running DD inference mode...")
	#DD = infer(rwa_file, mode='DD', localization_error=localization_error, \
	#	priorD=priorD, min_diffusivity=minD, output_label='DD')
	#map_plot(DD, output_file=out('dd', 'png'), show=True, aspect='equal')

	print("running DV inference mode...")
	DV = infer(rwa_file, mode='DV', localization_error=localization_error, \
		priorD=priorD, priorV=priorV, output_label='DV')
	map_plot(DV, output_file=out('dv', 'png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()
