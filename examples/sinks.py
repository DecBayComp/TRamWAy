
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


method = 'hexagon'

dim = 2
D0 = .05
D = .05
dV = 2.
#ref_distance = .25
name = 'sinks'


sigma = 1. / sqrt(-log(.05))

def main():
        output_basename = name
        def out(label, extension):
                return '{}.{}.{}.{}'.format(output_basename, method, label, extension)

        xyt_file = output_basename + '.txt'
        rwa_file = output_basename + '.rwa'
        new_xyt = not os.path.exists(xyt_file)
        new_tessellation = not os.path.isfile(rwa_file)

        ## define the ground truth (xyt_file)
        d_area_center = np.full((dim,), .5)
        d_area_radius = .2
        def diffusivity_map(x, *args):
                r = x - d_area_center
                r = np.sqrt(np.dot(r, r))
                return D0 + (D - D0) * exp(-(r / d_area_radius / sigma)**2)
        v_area_center= np.full((dim,), .5)
        v_area_radius = .2
        def force_map(x, *args):
                f = x - v_area_center
                r2 = np.dot(f, f)
                s2 = (v_area_radius * sigma) ** 2
                f *= -2. * dV / s2 * exp(-r2/s2)
                return f
        map_lower_bound = np.zeros(dim)
        map_upper_bound = np.full((dim,), 1.)
        if new_xyt:
                # simulate random walks
                print('generating trajectories: {}'.format(xyt_file))
                df = random_walk(diffusivity_map, force_map, \
                        trajectory_mean_count = 100, turnover = .7, duration = 10, \
                        box = np.r_[map_lower_bound, map_upper_bound - map_lower_bound], \
                        full = True, count_outside_trajectories = True)
                #print(df)
                df.to_csv(xyt_file, sep="\t", header=False)
                ## mesh regularly to sample ground truth for illustrative purposes
                #grid = tessellate(df, method='grid', min_location_count=10)
                if not new_tessellation:
                        print("WARNING: tessellation will overwrite file '{}'".format(rwa_file))
                        new_tessellation = True

        ## tessellate (tessellation_file)
        if new_tessellation:
                print("partitioning: {}".format(rwa_file))
                #cells = tessellate(xyt_file, method, output_file=rwa_file, \
                #       verbose=True, strict_min_location_count=1, force=True, \
                #       ref_distance=ref_distance, label='{}(d={})'.format(method, ref_distance))
                cells = tessellate(xyt_file, method, output_file=rwa_file, \
                        verbose=True, strict_min_location_count=10, force=True, label=method)
                # show ground truth
                true_map = distributed(cells).run(truth, diffusivity=diffusivity_map, force=force_map)

        ## infer
        # capture negative diffusivity warnings and turn them into exceptions
        warnings.filterwarnings('error', '', DiffusivityWarning)

        print("running DF inference mode...")
        DF = infer(rwa_file, mode='DF', output_label='DF')

        print("running DV inference mode...")
        DV = infer(rwa_file, mode='DV', output_label='DV', max_iter=1000, verbose=True)

        ## plot
        if new_tessellation:
                output_file = out('mesh', 'png')
                print('plotting the partition: {}'.format(output_file))
                cell_plot(rwa_file, output_file=output_file, show=True, aspect='equal')
                output_file = out('truth', 'png')
                print('plotting ground truth maps: {}'.format(output_file))
                map_plot(true_map, cells=cells, output_file=output_file, show=True, aspect='equal')

        #
        bb = cells.bounding_box
        bb[cells.tessellation.scaler.columns] = \
                np.c_[map_lower_bound, map_upper_bound - map_lower_bound]
        cells.bounding_box = bb
        output_file = out('df', 'png')

        print('plotting DF maps: {}'.format(output_file))
        map_plot(DF, output_file=output_file, show=True, aspect='equal')
        output_file = out('dv', 'png')
        print('plotting DV maps: {}'.format(output_file))
        map_plot(DV, output_file=output_file, show=True, aspect='equal')

        sys.exit(0)


if __name__ == '__main__':
        main()
