
import os
from inferencemap.helper import tesselate
from inferencemap.inference import *
from inferencemap.plot.map import scalar_map_2d
import matplotlib.pyplot as plt

short_description = 'infer and plot a kmeans-based diffusivity map for glycine receptor dataset'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

if __name__ == '__main__':
	main()

def main():
	local = os.path.join(data_dir, data_file)

	# tesselate
	method = 'kmeans'
	kmeans = tesselate(local, method, verbose=True)

	# infer
	localization_error = 0.0
	pre_map = Distributed(kmeans)
	diffusivity = inferD(pre_map, localization_error)

	# plot
	scalar_map_2d(diffusivity)
	plt.show()

