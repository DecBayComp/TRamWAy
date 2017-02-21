
import os
from inferencemap.io import *
from inferencemap.helper import tesselate, find_imt
from inferencemap.inference import *
from inferencemap.plot.map import scalar_map_2d
import matplotlib.pyplot as plt
#import cProfile

short_description = 'infer and plot kmeans-based diffusivity and potential maps for the glycine receptor dataset'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

def main():
	local = os.path.join(data_dir, data_file)

	# tesselate
	method = 'kmeans'
	_, kmeans = find_imt(local, method)
	if kmeans is None:
		kmeans = tesselate(local, method, verbose=True, strict_min_cell_size=10)

	# prepare the data for the inference
	detailled_map = distributed(kmeans)

	# infer diffusivity (D mode)
	diffusivity = detailled_map.run(inferD, \
		localization_error=0.0)

	# plot
	scalar_map_2d(detailled_map, diffusivity)
	plt.title('D (D mode)')
	#plt.show()


	# infer diffusivity and potential (DV mode)
	dvfile = os.path.splitext(local)[0] + '.dv.h5' # store result in file
	if os.path.isfile(dvfile): # if result file exists, load it
		store = HDF5Store(dvfile, 'r')
		dv = store.peek('DV')
	else: # otherwise compute

		localization_error = 0.0
		priorD = 0.01
		priorV = 0.1

		multiscale_map = detailled_map.group(max_cell_count=20)
		#bold_map = multiscale_map.flatten()
		dv = multiscale_map.run(inferDV, localization_error, priorD, priorV)

		# store the result
		store = HDF5Store(dvfile, 'w')
		store.poke('DV', dv)
		store.poke('localization_error', localization_error)
		store.poke('priorD', priorD)
		store.poke('priorV', priorV)
	
	store.close()

	# and plot
	scalar_map_2d(detailled_map, dv['D'])
	plt.title('D (DV mode)')
	scalar_map_2d(detailled_map, dv['V'])
	plt.title('V (DV mode)')
	plt.show()


if __name__ == '__main__':
	main()

