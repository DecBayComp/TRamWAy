
import os
import sys
from tramway.core import find_artefacts
from tramway.core.hdf5 import load_rwa
from tramway.tessellation import CellStats
from tramway.helper import tessellate, infer, map_plot, cell_plot
import warnings

short_description = 'infer and plot gwr-based diffusivity and potential maps for the glycine receptor dataset'

data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

verbose = True


methods = ['grid', 'kdtree', 'kmeans', 'gwr']
default_localization_error = 0.03
default_diffusivity_prior = 0.
default_potential_prior = 1.


arguments = [
        ('--grid',      dict(help='regular grid mesh', action='store_true')),
        ('--kdtree',    dict(help='kd-tree mesh', action='store_true')),
        ('--kmeans',    dict(help='k-means mesh', action='store_true')),
        ('--gwr',       dict(help='growing-when-required-based mesh (default)', action='store_true', default=True)),
        ('-ll', dict(help='plot locations (low resolution image)', action='store_true')),
        ('-lh', dict(help='plot locations (high resolution image)', action='store_true')),
        ('-ml', dict(help='plot mesh (low resolution image)', action='store_true')),
        ('-mh', dict(help='plot mesh (high resolution image)', action='store_true')),
        ('-d',  dict(help='D inference mode', action='store_true')),
        ('-dd', dict(help='DD inference mode', action='store_true')),
        ('-df', dict(help='DF inference mode (default)', action='store_true')),
        ('-dv', dict(help='DV inference mode', action='store_true')),
        (('--localization-error', '-le'), dict(help='localization error', type=float, default=default_localization_error)),
        (('--prior-diffusivity', '-pd'), dict(help='prior diffusivity', type=float, default=default_diffusivity_prior)),
        (('--prior-potential', '-pv'), dict(help='prior potential', type=float, default=default_potential_prior)),
        ('-nc', dict(help='do NOT plot colorbars', action='store_true'))]


def main(**kwargs):

        #warnings.simplefilter('error')

        local = os.path.join(data_dir, data_file)
        if not os.path.isfile(local):
                print("cannot find file: {}".format(local))
                sys.exit(1)

        output_basename, _ = os.path.splitext(local)

        # determine which tessellation method
        method = None
        for m in methods:
                if kwargs.get(m, False):
                        method = m
                        break
        if not method:
                method = methods[-1]

        # tessellate
        rwa_file = output_basename + '.rwa'
        _tessellate = True
        if os.path.isfile(rwa_file):
                try:
                        _, = find_artefacts(load_rwa(rwa_file), CellStats, labels=method)
                except:
                        pass
                else:
                        _tessellate = False
        #if verbose:
        #       print('glycine receptor demo (verbose mode: on)')
        if _tessellate :
                tessellate(local, method, output_file=rwa_file, \
                        label=method, verbose=verbose, strict_min_location_count=10)

        # plot locations and meshes
        pt_size_low, pt_size_high = 8, 6
        res_low, res_high = 100, 200
        other_plots = False
        if kwargs.get('ll', False):
                other_plots = True
                cell_plot(rwa_file, label=method, output_file='gr_locations_low.png',
                        verbose=verbose, voronoi=False,
                        locations=dict(color=['k']*9999, size=pt_size_low), dpi=res_low)
        if kwargs.get('lh', False):
                other_plots = True
                cell_plot(rwa_file, label=method, output_file='gr_locations_high.png',
                        verbose=verbose, voronoi=False,
                        locations=dict(color=['k']*9999, size=pt_size_high), dpi=res_high)
        if kwargs.get('ml', False):
                other_plots = True
                #cell_plot(rwa_file, label=method, output_file='gr_{}_low.png'.format(method),
                #       verbose=verbose, voronoi=dict(centroid_style=None, color='k'),
                #       locations=dict(size=pt_size_low), dpi=res_low)
                cell_plot(rwa_file, label=method, output_file='gr_{}_low.png'.format(method),
                        verbose=verbose, voronoi=False,
                        delaunay=dict(color='kkk', linewidth=2, centroid_style=None),
                        locations=dict(color='light', size=pt_size_low), dpi=res_low)
        if kwargs.get('mh', False):
                other_plots = True
                cell_plot(rwa_file, label=method, output_file='gr_{}_high.png'.format(method),
                        verbose=verbose,
                        delaunay=dict(color='kkk', linewidth=1, centroid_style=None),
                        voronoi=dict(color='yyyy', linewidth=.5, centroid_style=None,
                                negative=True),
                        locations=dict(color='light', size=pt_size_high), dpi=res_high)

        # determine which inference modes
        _d  = kwargs.get('d',  False)
        _dd = kwargs.get('dd', False)
        _df = kwargs.get('df', False)
        _dv = kwargs.get('dv', False)
        if not any((_d, _dd, _df, _dv)) and not other_plots:
                _df = True

        localization_error = kwargs.get('locatization_error', default_localization_error)
        diffusivity_prior = kwargs.get('diffusivity_prior', default_diffusivity_prior)
        potential_prior = kwargs.get('potential_prior', default_potential_prior)

        def img(mode):
                return '{}.{}.{}.png'.format(output_basename, method, mode)

        # infer and plot maps
        map_plot_args = dict(show=True, point_style=dict(alpha=.01), clip=4.,
                        colorbar=not kwargs.get('nc', False))
        if _d:
                D, cells = infer(rwa_file, mode='d', input_label=method, output_label='D',
                        localization_error=localization_error, return_cells=True)
                map_plot(D, output_file=img('d'), cells=cells, **map_plot_args)

        if _df:
                DF, cells = infer(rwa_file, mode='df', input_label=method, output_label='DF',
                        localization_error=localization_error, return_cells=True)
                map_plot(DF, output_file=img('df'), cells=cells, **map_plot_args)

        if _dd:
                DD, cells = infer(rwa_file, mode='dd', input_label=method, output_label='DD',
                        localization_error=localization_error, return_cells=True)
                map_plot(DD, output_file=img('dd'), cells=cells, **map_plot_args)

        if _dv:
                DV, cells = infer(rwa_file, mode='dv', input_label=method, output_label='DV',
                        localization_error=localization_error, diffusivity_prior=diffusivity_prior,
                        potential_prior=potential_prior, return_cells=True)
                map_plot(DV, output_file=img('dv'), cells=cells, **map_plot_args)

        sys.exit(0)


if __name__ == '__main__':
        main()

