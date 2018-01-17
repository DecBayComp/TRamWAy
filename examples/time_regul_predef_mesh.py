
import os
import sys
from tramway.inference import DiffusivityWarning, distributed
from tramway.tessellation.time import *
from tramway.helper import *
from tramway.helper.simulation import *
import matplotlib.pyplot as plt
import warnings
#warnings.simplefilter('error')
import numpy as np
import pandas as pd
import numpy.linalg as la
from math import *
import copy

short_description = 'temporal regularization with predefined temporal mesh'


name = 'regular_temporal_mesh'

method = 'grid'
localization_error = 0.001
priorD = 0.01
#priorV = 0.1
minD = -localization_error


D0 = .1 # um2.s-1
D = .01 # um2.s-1

R = .2 # um

dim = 2
x0 = y0 = t0 = 0.
width = height = 1. # um
duration = 10. # s
time_step = .05 # s
tessellation_dt = 2. # s

min_count = 10 # number of points per cell


_box = (x0, y0, width, height)


def diffusivity_map(xy, t):
	x0, y0, width, height = _box
	x_radius = y_radius = R
	t_radius = duration * .1
	r0 = np.asarray((x0 + width * .5, y0 + height * .5, t0 + duration * .5)) # where diffusivity is D
	radius = np.asarray((x_radius, y_radius, t_radius)) # within which diffusivity is half-way from D0
	if np.isscalar(t_radius):
		r = np.r_[xy, t]
	else:
		raise NotImplementedError
	return D0 + (D-D0) * np.exp(-np.sqrt(np.sum(( (r - r0) / radius )**2)) * -np.log(.5))



def main():
	output_basename = name
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	xyt_file = out('', 'trxyt')
	tessellation_file = out(method, '.rwa')
	new_xyt = not os.path.exists(xyt_file)
	new_tessellation = not os.path.isfile(tessellation_file)

	time = np.arange(t0, t0 + duration + tessellation_dt * .1, tessellation_dt)
	nsegments = time.size - 1

	## define the ground truth (xyt_file)
	if new_xyt:
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk(diffusivity_map, None, 1000, 30, box=_box)
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		# mesh regularly to sample ground truth for illustrative purposes
		grid = tessellate(df, method='grid', min_location_count=10)
		cells = distributed(grid)
		for i in range(nsegments):
			t = (time[i] + time[i+1]) * .5 # bin center
			true_map = cells.run(truth, t, diffusivity_map)
			subext = 'truth.{}'.format(i)
			print('plotting ground truth maps at time {}: {}'.format(t, \
				out(subext, '.png')))
			map_plot(true_map, cells=cells, mode='true', \
				output_file=out(subext, '.png'), aspect='equal', clim=[D, D0])
		if not new_tessellation:
			print("WARNING: tessellation will overwrite file '{}'".format(tessellation_file))
			new_tessellation = True

	## tessellate (tessellation_file)
	if new_tessellation:
		tessellate(xyt_file, method, min_location_count=min_count * nsegments, \
			output_file=tessellation_file, verbose=True)
		cell_plot(tessellation_file, output_file=out(method, '.mesh.png'), \
			show=True, aspect='equal')

	_, static_cells = find_imt(tessellation_file)
	if static_cells is None:
		raise EnvironmentError("cannot load file: {}".format(tessellation_file))

	frames = np.c_[time[:-1], time[1:]] # time segments

	# `exclude_cells_by_location_count` provides a mechanism to blacklist space cells
	# based on their point counts over time
	def exclude(sizes):
		# `sizes` is a (#cells * #frames) matrix
		excl = np.zeros(sizes.shape, dtype=bool)
		for cell in range(sizes.shape[0]):
			if np.any(sizes[cell] < min_count):
				excl[cell] = True
		# `exclude` returns a boolean matrix similar to `sizes`
		return excl

	dynamic_cells = with_time_lattice(static_cells, frames, \
		exclude_cells_by_location_count=exclude)

	## infer and plot
	# capture negative diffusivity warnings and turn them into exceptions
	warnings.filterwarnings('error', '', DiffusivityWarning)

	print("running D inference mode...")
	D_ = infer(dynamic_cells, mode='D', localization_error=localization_error, \
		min_diffusivity=minD)
	Dlim = np.r_[0, D_.quantile(.95).values]
	D_ = dynamic_cells.tessellation.split_frames(D_)
	for t, frame_map in enumerate(D_):
		map_plot(frame_map, cells=static_cells, mode='D', \
			output_file=out(method, '.d.{}.png'.format(t)), \
			aspect='equal', clim=Dlim)

	#print("running DF inference mode...")
	#DF = infer(tessellation_file, mode='DF', localization_error=localization_error)
	#map_plot(DF, output_file=out(method, '.df.png'), show=True, aspect='equal')

	#print("running DD inference mode...")
	#DD = infer(dynamic_cells, mode='DD', localization_error=localization_error, \
	#	priorD=priorD, min_diffusivity=minD)
	#Dlim = np.r_[0, DD.quantile(.95).values]
	#DD = dynamic_cells.tessellation.split_frames(DD)
	#for t, frame_map in enumerate(DD):
	#	map_plot(frame_map, cells=static_cells, mode='DD', \
	#		output_file=out(method, '.dd.{}.png'.format(t)), \
	#		aspect='equal', clim=Dlim)

	#print("running DV inference mode...")
	#DV = infer(tessellation_file, mode='DV', localization_error=localization_error, \
	#	priorD=priorD, priorV=priorV)
	#map_plot(DV, output_file=out(method, '.dv.png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()

