
from tramway.helper import *
from tramway.inference.nontracking import *


short_description = "demonstrate nontracking-based inference on generated data"


def main(**kwargs):
	# the `main` function takes any number of keyworded arguments;
	# see *examples/glycine_receptor* for an example of how to prototype input arguments;
	# arguments defined in the `arguments` module variable can be supplied from the command-line
	# ``python -m examples nontracking --key1 val1 --key2 val2 ...`` following the :mod:`argparse`
	# logic

	# generate data here;
	# a `pandas.DataFrame` is expected here, with columns 'x', 'y' and 't' (optionally 'z')

	# split the set of locations in pairs of successive frames (or timestamps);
	# the first "cell" will contain the locations of the first two frames,
	# the second will contain the locations of the second and third frames, and so on;
	# note that this `tessellate` step is necessary even if `data` contains only one or two frames
	cells = tessellate(data, method='window', duration=2, shift=1, frames=True)

	# the `tessellate` step below is optional;
	# it tessellates the space and partitions the locations in each cell into smaller cells (in space)
	# the space cells may be different between time cells;
	# for a temporal expansion of a static spatial tessellations, please use the 
	# :mod:`~tramway.tessellation.time` module instead of the above windowing
	#cells = tessellate(cells, method='kmeans', avg_location_count=5, knn=20)

	# call a nontracking plugin and generate diffusivity (and potential or force) maps
	maps = infer(cells, mode='nontracking.d')

	# extract whatever statistics from the maps and plot them;
	# alternatively the maps can be plotted the following way:
	#map_plot(maps, output_file='nontracking.png')
	# if maps are labelled, the output image files will carry the label in their name, between the
	# basename ('nontracking') and the extension ('.png')


if __name__ == '__main__':
	main()

