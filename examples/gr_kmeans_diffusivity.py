
import os
import sys
from inferencemap.io import *
from inferencemap.helper import tesselate, find_imt
from inferencemap.inference import *
from inferencemap.plot.map import scalar_map_2d, field_map_2d
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('error')
from time import time

short_description = 'infer and plot kmeans-based diffusivity and potential maps for the glycine receptor dataset'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''


localization_error = 0.001
priorD = 0.01
priorV = 0.1


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
	t = time()
	diffusivity = detailled_map.run(inferD, \
		localization_error=localization_error)
	print('D mode: elapsed time: {}ms'.format(int(round((time()-t)*1e3))))
	# plot
	scalar_map_2d(detailled_map, diffusivity)
	plt.title('D (D mode)')
	#plt.show()


	# infer diffusivity and force (DF mode)
	t = time()
	df = detailled_map.run(inferDF, \
		localization_error=localization_error)
	print('DF mode: elapsed time: {}ms'.format(int(round((time()-t)*1e3))))
	# plot
	scalar_map_2d(detailled_map, df['diffusivity'])
	plt.title('D (DF mode)')
	field_map_2d(detailled_map, df[[ col for col in df.columns if col.startswith('force') ]])
	plt.title('F (DF mode)')
	#plt.show()
	#sys.exit(0)


	# DD and DV require cell grouping
	multiscale_map = detailled_map.group(max_cell_count=20)

	# infer diffusivity (DD mode)
	ddfile = os.path.splitext(local)[0] + '.dd.h5' # store result in file
	if os.path.isfile(ddfile): # if result file exists, load it
		store = HDF5Store(ddfile, 'r')
		dd = store.peek('DD')
		assert store.peek('localization_error') == localization_error \
			and store.peek('priorD') == priorD
		runtime = store.peek('runtime')
	else: # otherwise compute

		runtime = time()
		dd = multiscale_map.run(inferDD, localization_error, priorD)
		runtime = time() - runtime

		# store the result
		store = HDF5Store(ddfile, 'w')
		store.poke('DD', dd)
		store.poke('localization_error', localization_error)
		store.poke('priorD', priorD)
		store.poke('runtime', runtime)
	store.close()
	print('DD mode: elapsed time: {}ms'.format(int(round(runtime*1e3))))

	# and plot
	scalar_map_2d(detailled_map, dd) # only 'diffusivity' in dd, scalar_map_2d will find alone
	plt.title('D (DD mode)')


	# infer diffusivity and potential (DV mode)
	dvfile = os.path.splitext(local)[0] + '.dv.h5' # store result in file
	if os.path.isfile(dvfile): # if result file exists, load it
		store = HDF5Store(dvfile, 'r')
		dv = store.peek('DV')
		assert store.peek('localization_error') == localization_error \
			and store.peek('priorD') == priorD \
			and store.peek('priorV') == priorV
		runtime = store.peek('runtime')
	else: # otherwise compute

		runtime = time()
		dv = multiscale_map.run(inferDV, localization_error, priorD, priorV)
		runtime = time() - runtime

		# store the result
		store = HDF5Store(dvfile, 'w')
		store.poke('DV', dv)
		store.poke('localization_error', localization_error)
		store.poke('priorD', priorD)
		store.poke('priorV', priorV)
		store.poke('runtime', runtime)
	store.close()
	print('DV mode: elapsed time: {}ms'.format(int(round(runtime*1e3))))

	# and plot
	scalar_map_2d(detailled_map, dv['diffusivity'])
	plt.title('D (DV mode)')
	scalar_map_2d(detailled_map, dv['potential'])
	plt.title('V (DV mode)')


	plt.show()


if __name__ == '__main__':
	main()

