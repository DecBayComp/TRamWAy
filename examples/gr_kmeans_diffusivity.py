
import os
from inferencemap.helper import tesselate, find_imt
from inferencemap.inference import *
from inferencemap.plot.map import scalar_map_2d
import matplotlib.pyplot as plt
#import cProfile

short_description = 'infer and plot a kmeans-based diffusivity map for glycine receptor dataset'

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
		kmeans = tesselate(local, method, verbose=True)

	# infer diffusivity
	localization_error = 0.21
	pre_map = distributed(kmeans)
	#diffusivity = inferD(pre_map, localization_error)

	# plot
	#scalar_map_2d(diffusivity)
	#plt.show()

	# diffusivity+potential
	priorD = 0.1
	priorV = 0.1
	dv = inferDV(pre_map, localization_error, priorD, priorV, options=dict(maxiter=1, disp=True))
	# and plot
	scalar_map_2d(dv, 'D')
	scalar_map_2d(dv, 'V')
	plt.show()


if __name__ == '__main__':
	main()

