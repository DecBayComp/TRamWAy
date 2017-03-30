
import os
import sys
from tramway.helper import tesselate, find_imt, infer, map_plot
import matplotlib.pyplot as plt
import warnings
#warnings.simplefilter('error')

short_description = 'infer and plot kmeans-based diffusivity and potential maps for the glycine receptor dataset'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''


method = 'kmeans'
localization_error = 0.001
priorD = 0.01
priorV = 0.1


def main():
	local = os.path.join(data_dir, data_file)

	output_basename, _ = os.path.splitext(local)
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	# tesselate
	tesselation_file = out(method, '.rwa')
	if not os.path.isfile(tesselation_file):
		tesselate(local, method, output_file=tesselation_file, \
			verbose=True, strict_min_cell_size=10)

	# infer and plot
	D = infer(tesselation_file, mode='D', localization_error=localization_error)
	map_plot(D, output_file=out(method, '.d.png'), show=True)

	DF = infer(tesselation_file, mode='DF', localization_error=localization_error)
	map_plot(DF, output_file=out(method, '.df.png'), show=True)

	DD = infer(tesselation_file, mode='DD', localization_error=localization_error, \
		priorD=priorD)
	map_plot(DD, output_file=out(method, '.dd.png'), show=True)

	DV = infer(tesselation_file, mode='DV', localization_error=localization_error, \
		priorD=priorD, priorV=priorV)
	map_plot(DV, output_file=out(method, '.dv.png'), show=True)

	sys.exit(0)


if __name__ == '__main__':
	main()

