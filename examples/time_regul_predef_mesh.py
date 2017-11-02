
import os
import sys
from tramway.inference import DiffusivityWarning, distributed
from tramway.tesselation.time import TimeLattice
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
tesselation_dt = 2. # s


time = np.arange(t0, t0 + duration + tesselation_dt * .1, tesselation_dt)
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
	tesselation_file = out(method, '.rwa')
	new_xyt = not os.path.exists(xyt_file)
	new_tesselation = not os.path.isfile(tesselation_file)

	## define the ground truth (xyt_file)
	if new_xyt:
		# simulate random walks
		print('generating trajectories: {}'.format(xyt_file))
		df = random_walk(diffusivity_map, None, 1000, 30, box=_box)
		#print(df)
		df.to_csv(xyt_file, sep="\t", header=False)
		# mesh regularly to sample ground truth for illustrative purposes
		grid = tesselate(df, method='grid', min_cell_count=10)
		cells = distributed(grid)
		for i in range(time.size - 1):
			t = (time[i] + time[i+1]) * .5
			true_map = cells.run(truth, t, diffusivity_map)
			subext = 'truth.{}'.format(i)
			print('ploting ground truth maps at time {}: {}'.format(t, \
				out('*.'+subext, '.png')))
			map_plot((cells, 'true', true_map), output_file=out(subext, '.png'), \
				aspect='equal', clim=[D, D0])
		if not new_tesselation:
			print("WARNING: tesselation will overwrite file '{}'".format(tesselation_file))
			new_tesselation = True

	## tesselate (tesselation_file)
	if new_tesselation:
		tesselate(xyt_file, method, min_cell_count=100, \
			output_file=tesselation_file, verbose=True)
		cell_plot(tesselation_file, output_file=out(method, '.mesh.png'), \
			show=True, aspect='equal')

	_, static_cells = find_imt(tesselation_file)
	spatial_mesh = static_cells.tesselation
	cells = copy.deepcopy(static_cells)

	frames = np.c_[time[:-1], time[1:]]
	cells.tesselation = TimeLattice(spatial_mesh, frames)

	def exclude(sizes):
		excl = np.zeros(sizes.shape, dtype=bool)
		for cell in range(sizes.shape[0]):
			if np.any(sizes[cell] < frames.shape[0]):
				excl[cell] = True
		return excl
	cells.cell_index = cells.tesselation.cellIndex(cells.points, \
		exclude_by_cell_size=exclude)

	## infer and plot
	# capture negative diffusivity warnings and turn them into exceptions
	warnings.filterwarnings('error', '', DiffusivityWarning)

	print("running D inference mode...")
	D_ = infer(cells, mode='D', localization_error=localization_error, \
		min_diffusivity=minD)
	Dbounds = np.r_[0, D_.quantile(.95).values]
	D_ = cells.tesselation.reshapeFrames(D_)
	for t, _D in enumerate([ d for _, d in D_ ]):
		_map = (static_cells, 'D', _D)
		map_plot(_map, output_file=out(method, '.d.{}.png'.format(t)), \
			aspect='equal', clim=Dbounds) # show=True, 

	#print("running DF inference mode...")
	#DF = infer(tesselation_file, mode='DF', localization_error=localization_error)
	#map_plot(DF, output_file=out(method, '.df.png'), show=True, aspect='equal')

	print("running DD inference mode...")
	DD = infer(cells, mode='DD', localization_error=localization_error, \
		priorD=priorD, min_diffusivity=minD)
	Dbounds = np.r_[0, DD.quantile(.95).values]
	DD = cells.tesselation.reshapeFrames(DD)
	for t, _D in enumerate([ d for _, d in DD ]):
		_map = (static_cells, 'DD', _D)
		map_plot(_map, output_file=out(method, '.dd.{}.png'.format(t)), \
			aspect='equal', clim=Dbounds) # show=True, 

	#print("running DV inference mode...")
	#DV = infer(tesselation_file, mode='DV', localization_error=localization_error, \
	#	priorD=priorD, priorV=priorV)
	#map_plot(DV, output_file=out(method, '.dv.png'), show=True, aspect='equal')

	sys.exit(0)


if __name__ == '__main__':
	main()

