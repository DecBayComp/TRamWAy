# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


"""
As of version 0.4.7, the tramway command is frozen and is only affected by minor fixes.
"""

import argparse
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
import sys
from tramway.core.plugin import *
import tramway.tessellation as tessellation
import tramway.inference as inference
import tramway.feature as feature
from .helper import *
#import tramway.core.hdf5.compat
import tramway.utils.inferencemap as inferencemap


def _parse_args(args):
    kwargs = dict(args.__dict__)
    del kwargs['func']
    input_files = []
    try:
        input_files = kwargs.pop('input')
    except KeyError:
        pass
    try:
        more_input_files = kwargs.pop('input_file')
    except KeyError:
        pass
    else:
        if more_input_files:
            if isinstance(more_input_files, list):
                input_files += more_input_files
            else:
                input_files.append(more_input_files)
    if not input_files:
        print('please specify input file(s)')
        sys.exit(1)
    for input_file in input_files:
        if not os.path.isfile(input_file):
            print("cannot find file: {}".format(input_file))
            sys.exit(1)
    try:
        comment = kwargs.pop('comment')
    except KeyError:
        pass
    else:
        if comment is not None:
            kwargs['comment'] = comment.replace('\\n', '\n')
    seed = kwargs.pop('seed', False)
    if seed is None:
        import time
        seed = int(time.time())
        print('random generator seed: {}'.format(seed))
    if seed is not False:
        seed = int(seed)
        import random, numpy
        random.seed(seed)
        numpy.random.seed(seed)
    return input_files, kwargs


def _render_cells(args):
    input_file, kwargs = _parse_args(args)
    output_file = kwargs.pop('output', None)
    fig_format = kwargs.pop('print', None)
    delaunay = kwargs.pop('delaunay', False)
    hist = kwargs.pop('histogram', '')
    if hist is None: hist = ''
    if delaunay:
        kwargs['xy_layer'] = 'delaunay'
    #del kwargs['min_location_count']
    cell_plot(input_file, output_file=output_file, fig_format=fig_format, figsize=True, \
        location_count_hist='c' in hist, cell_dist_hist='d' in hist, \
        location_dist_hist='p' in hist, **kwargs)
    sys.exit(0)


def _single_molecule(args):
    input_file, kwargs = _parse_args(args)
    output_file = kwargs.pop('output', None)
    if output_file is None:
        output_file = input_file[0]
    xyt = load_xyt(input_file, columns=list('xyt'))
    xyt['n'] = 1
    xyt = xyt[list('nxyt')]
    xyt.to_csv(output_file, sep='\t', index=False)


def _sample(method, parse_extra=None):
    def sample(args):
        input_file, kwargs = _parse_args(args)
        output_file = kwargs.pop('output', None)
        scaling = kwargs.pop('w', None)
        if scaling and not kwargs['scaling']:
            kwargs['scaling'] = 'whiten'
        min_nn = kwargs.pop('knn', None)
        max_nn = kwargs.pop('max_nn', None)
        if not (min_nn is None and max_nn is None):
            kwargs['knn'] = (min_nn, max_nn)
        if kwargs.pop('enable_time_regularization', False):
            kwargs['time_window_options'] = dict(time_dimension=True)
        elif kwargs.get('time_window_duration', None):
            kwargs['time_window_options'] = dict(time_dimension=False)
        if parse_extra:
            for extra_arg, parse_arg in parse_extra:
                kwargs[extra_arg] = parse_arg(**kwargs)
        kwargs = { kw: arg for kw, arg in kwargs.items() if arg is not None }
        tessellate(input_file, output_file=output_file, method=method, **kwargs)
        sys.exit(0)
    return sample

def _infer(mode, parse_extra=None):
    def __infer(args):
        input_file, kwargs = _parse_args(args)
        output_file = kwargs.pop('output', None)
        kwargs['mode'] = mode
        if kwargs.get('profile', False) is None:
            kwargs['profile'] = True
        if parse_extra:
            for extra_arg, parse_arg in parse_extra:
                kwargs[extra_arg] = parse_arg(**kwargs)
        kwargs = { kw: arg for kw, arg in kwargs.items() if arg is not None }
        infer(input_file[0], output_file=output_file, **kwargs)
        # kwargs: mode, localization_error, diffusivity_prior, potential_prior, jeffreys_prior
        sys.exit(0)
    return __infer

def _render_map(args):
    input_file, kwargs = _parse_args(args)
    output_file = kwargs.pop('output', None)
    fig_format = kwargs.pop('print', None)
    points = kwargs.pop('points')
    if points is not False:
        point_style = dict(alpha=.01)
        if points is not None:
            for prm in points.split(','):
                key, val = prm.split('=')
                if key in ('a', 'alpha'):
                    point_style['alpha'] = float(val)
                elif key in ('c', 'color'):
                    if val[0] == "'" and val[-1] == "'":
                        val = val[1:-1]
                    point_style['color'] = val
                elif key in ('s', 'size'):
                    point_style['size'] = int(val)
        kwargs['point_style'] = point_style
    if kwargs['delaunay'] is None:
        kwargs['delaunay'] = True
    elif kwargs['delaunay'] is False:
        del kwargs['delaunay']
    else:
        delaunay = {}
        for prm in kwargs['delaunay'].split(','):
            key, val = prm.split('=')
            if key in ('a', 'alpha'):
                delaunay['alpha'] = float(val)
            elif key in ('c', 'color'):
                if val[0] == "'" and val[-1] == "'":
                    val = val[1:-1]
                delaunay['color'] = val
        kwargs['delaunay'] = delaunay
    if kwargs['clip'] is None:
        kwargs['clip'] = 4.
    elif kwargs['clip'] == 0.:
        del kwargs['clip']
    if kwargs['title'] is None:
        kwargs.pop('title')
    map_plot(input_file[0], output_file=output_file, fig_format=fig_format, figsize=True, show=True, **kwargs)
    sys.exit(0)

def _dump_rwa(args):
    input_files, kwargs = _parse_args(args)
    verbose = kwargs.pop('verbose', False)
    label = kwargs.pop('label')
    if label and input_files:
        labels = kwargs.pop('input_label', None)
        export = False
        if not input_files[1:]:
            cluster = kwargs.pop('cluster')
            vmesh = kwargs.pop('vmesh')
            export = cluster or vmesh
        if export:
            kwargs['cluster_file'] = cluster
            kwargs['vmesh_file'] = vmesh
            kwargs = { k: v for k, v in kwargs.items() if v is not None }
            inferencemap.export_file(input_files[0], label=labels, **kwargs)
        else:
            for input_file in input_files:
                print(' -> '.join(['in '+input_file] + [ str(l) for l in labels ]) + ':')
                analyses = load_rwa(input_file, lazy=True)
                for label in labels:
                    analyses = analyses[label]
                print('\t' + str(analyses.data).replace('\n', '\n\t'))
    else:
        for input_file in input_files:
            print('in {}:'.format(input_file))
            analyses = load_rwa(input_file, lazy=True)
            print(format_analyses(analyses, global_prefix='\t', node=lazytype, metadata=kwargs.get('metadata', False)))

def _curl(args):
    import copy
    import tramway.feature.curl
    input_file, kwargs = _parse_args(args)
    input_label = kwargs.get('input_label', kwargs.get('label', None))
    if input_file[1:]:
        raise NotImplementedError('cannot handle multiple input files')
    input_file = input_file[0]
    analyses = load_rwa(input_file, lazy=True)
    cells, maps, leaf = find_artefacts(analyses, (CellStats, Maps), input_label, return_subtree=True)
    curl = tramway.feature.curl.Curl(cells, maps)
    vector_fields = { f: vs for f, vs in curl.variables.items() if len(vs) == 2 }
    curl_name = kwargs.get('output_label', None)
    if curl_name:
        curl_name = curl_name.replace('*', '')
    else:
        curl_name = 'curl'
    distance = kwargs.get('radius', 1)
    curl_maps = copy.copy(maps)
    curl_maps.maps = None
    for f in vector_fields:
        _name = '{}<{}>_{}'.format(curl_name, f, distance)
        curl_map = curl.extract(_name, f, distance)
        if curl_maps.maps is None:
            curl_maps.maps = curl_map
        else:
            curl_maps.maps = curl_maps.maps.join(curl_map)
    if curl_maps.extra_args is None:
        curl_maps.extra_args = {}
    else:
        curl_maps.extra_args = dict(curl_maps.extra_args) # copy
    curl_maps.extra_args['radius'] = distance
    # insert `curl_maps` into `analyses`
    leaf.add(Analyses(curl_maps), label=kwargs.get('output_label', None))
    output_file = kwargs.get('output', None)
    if output_file is None:
        output_file = input_file
    save_rwa(output_file, analyses, force=output_file == input_file)

def _animate_trajectories(args):
    import matplotlib
    matplotlib.use('Agg')
    from tramway.helper.animation import animate_trajectories_2d_helper
    input_file, kwargs = _parse_args(args)
    input_label = kwargs.get('input_label', kwargs.get('label', None))
    if input_file[1:]:
        raise NotImplementedError('cannot handle multiple input files')
    input_file = input_file[0]
    output_file = kwargs.pop('output', None)
    write_only = kwargs.pop('write_only', False)
    kwargs = { kw: arg for kw, arg in kwargs.items() if arg is not None }
    if 'play' not in kwargs:
        kwargs['play'] = not write_only
    animate_trajectories_2d_helper(input_file, output_file, **kwargs)

def _animate_map(args):
    import matplotlib
    matplotlib.use('Agg')
    from tramway.helper.animation import animate_map_2d_helper
    input_file, kwargs = _parse_args(args)
    input_label = kwargs.pop('input_label', kwargs.get('label', None))
    kwargs['label'] = input_label
    if input_file[1:]:
        raise NotImplementedError('cannot handle multiple input files')
    input_file = input_file[0]
    output_file = kwargs.pop('output', None)
    write_only = kwargs.pop('write_only', False)
    kwargs = { kw: arg for kw, arg in kwargs.items() if arg is not None }
    if 'play' not in kwargs:
        kwargs['play'] = not write_only
    animate_map_2d_helper(input_file, output_file, **kwargs)


def _add_subparsers(sub, subcommand, aliases=None, help=None, title=None, target='targets'):
    if aliases:
        try:
            _parser = sub.add_parser(subcommand, aliases=aliases, help=help)
        except TypeError: # Py2
            _parser = sub.add_parser(subcommand, help=help)
    else:
        _parser = sub.add_parser(subcommand, help=help)
    if help:
        _subsub = _parser.add_subparsers(title=subcommand if title is None else title,
            description='{}; the available {} are:'.format(help, target))
    else:
        _subsub = None
    return _subsub, _parser


def main():
    verbose = '--verbose' in sys.argv
    if not verbose:
        simple = None
        for k, a in enumerate(sys.argv):
            if a[0] == '-' and a[1:] and all(c == 'v' for c in a[1:]):
                simple = not a[2:]
                break
        if simple is True:
            # note that '-v' may not be the verbose flag
            # if it appears after the first command (i.e. tessellate, infer);
            # exclude one such known case by testing the argument that comes next:
            try:
                float(sys.argv[k+1])
            except (ValueError, IndexError):
                verbose = True
        elif simple is False:
            verbose = True
    if verbose:
        tessellation.plugins.verbose = inference.plugins.verbose = True

    parser = argparse.ArgumentParser(prog='tramway',
        description='TRamWAy central command.',
        epilog='See also https://github.com/DecBayComp/TRamWAy',
        conflict_handler='resolve')
    global_arguments = [
        ('-v', '--verbose', dict(action='count', help='increase verbosity')),
        ('-i', '--input', dict(action='append', default=[],
            metavar='INPUT_FILE', help='path to input file')),
        ('-o', '--output', dict(metavar='OUTPUT_FILE', help='path to output file'))]
    for arg1, arg2, kwargs in global_arguments:
        parser.add_argument(arg1, arg2, dest=arg1[1]+'pre', **kwargs)
    sub = parser.add_subparsers(title='commands', \
        description="type '%(prog)s command --help' for additional help")


    # track
    track_name = 'track'
    track_aliases = ['preprocess']
    ksub, _ = _add_subparsers(sub, track_name, track_aliases,
            'track or preprocess location data', target='preprocesses')
    single_molecule_parser = ksub.add_parser('single')
    for short_arg, long_arg, kwargs in global_arguments:
        dest = short_arg[1:] + 'post'
        single_molecule_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
    single_molecule_parser.set_defaults(func=_single_molecule)


    # tessellate
    tessellate_name = 'tessellate'
    tessellate_aliases = ['sample']
    load_tessellate_plugins = tessellate_name in sys.argv
    if not load_tessellate_plugins:
        load_tessellate_plugins = any(alias in sys.argv for alias in tessellate_aliases)
    tsub, _ = _add_subparsers(sub, tessellate_name, tessellate_aliases,
            'spatial and temporal binning of (trans-)locations data', target='binning methods')
    if load_tessellate_plugins:
        for method in tessellation.plugins:
            method_parser = tsub.add_parser(method)
            setup, _ = tessellation.plugins[method]
            short_args = short_options(setup.get('make_arguments', {}))
            for short_arg, long_arg, kwargs in global_arguments:
                dest = short_arg[1:] + 'post'
                if short_arg in short_args:
                    method_parser.add_argument(long_arg, dest=dest, **kwargs)
                else:
                    method_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
            method_parser.add_argument('-l', '--label', '--output-label', help='output label')
            method_parser.add_argument('--comment', help='description message (newlines must be escaped)')
            method_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
            method_parser.add_argument('--inplace', action='store_true', \
                help='replace the input sampling by the output one (only when --input-label is defined)')
            method_parser.add_argument('-n', '--knn', '--min-nn', '--knn-min', type=int, \
                help='minimum number of nearest neighbours; cells can overlap')
            method_parser.add_argument('-N', '--max-nn', '--knn-max', type=int, \
                help='maximum number of nearest neighbours')
            method_parser.add_argument('-r', '--radius', type=float, \
                help='selection radius for locations around the cell centers; cells can overlap')
            method_parser.add_argument('-d', '--distance', type=float, help='reference distance (default is the average translocation distance)')
            method_group = method_parser.add_mutually_exclusive_group()
            method_group.add_argument('-w', action='store_true', help='whiten the input data')
            method_group.add_argument('--scaling', choices=['whiten', 'unit'])
            method_parser.add_argument('-s', '--min-location-count', type=int, \
                help='minimum number of locations per cell; this affects the tessellation only and not directly the partition; see --knn and -ss for partition-related parameters')
            method_parser.add_argument('-ss', '--strict-min-location-count', '--min-n', type=int, \
                metavar='MIN_LOCATION_COUNT', \
                help='minimum number of locations per cell; this is enforced at partition time; cells with insufficient locations are discarded and not compensated for')
            method_parser.add_argument('--seed', nargs='?', default=False, \
                help='random generator seed (for testing purposes)')
            method_parser.add_argument('--disable-metadata', action='store_true', help="do not record additional metadata in the output file")
            translations = add_arguments(method_parser, setup.get('make_arguments', {}), name=method)
            try:
                method_parser.add_argument('input_file', nargs='*', help='path to input file(s)')
            except:
                pass
            if setup.get('window_compatible', True):
                method_parser.add_argument('--time-window-duration', type=float, help="time window duration in seconds")
                method_parser.add_argument('--time-window-shift', type=float, help="time window shift in seconds (default is equal to --time-window-duration so that there is no overlap between successive segments)")
                method_parser.add_argument('--enable-time-regularization', action="store_true", help="successive segments are connected; this enables the further inference steps to regularize across time; beware that most inference modes are NOT compatible with this option")
            method_parser.set_defaults(func=_sample(method, translations))


    # infer
    isub, _ = _add_subparsers(sub, 'infer', help='infer local parameters', target='inference modes')
    if 'infer' in sys.argv:
        for mode in inference.plugins:
            mode_parser = isub.add_parser(mode)
            setup, _ = inference.plugins[mode]
            short_args = short_options(setup.get('arguments', {}))
            for short_arg, long_arg, kwargs in global_arguments:
                dest = short_arg[1:] + 'post'
                if short_arg in short_args:
                    mode_parser.add_argument(long_arg, dest=dest, **kwargs)
                else:
                    mode_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
            mode_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
            mode_parser.add_argument('-l', '--output-label', help='output label')
            mode_parser.add_argument('--comment', help='description message for the output artefact (newlines must be escaped)')
            # shouldn't `inplace` be optional?
            mode_parser.add_argument('--inplace', action='store_true', \
                help='replace the input maps by the output ones')
            try:
                translations = add_arguments(mode_parser, setup['arguments'], name=mode)
            except KeyError:
                translations = None
            mode_parser.add_argument('--seed', nargs='?', default=False, help='random generator seed (for testing purposes)')
            mode_parser.add_argument('--profile', nargs='?', default=False, help='profile each individual child process if any')
            mode_parser.add_argument('--disable-metadata', action='store_true', help="do not record additional metadata in the output file")
            try:
                mode_parser.add_argument('input_file', nargs='?', help='path to input file')
            except:
                pass
            mode_parser.set_defaults(func=_infer(mode, translations))


    # dump analysis tree
    dump_parser = sub.add_parser('dump', help='print information from an .rwa file')
    dump_parser.set_defaults(func=_dump_rwa)
    for arg1, arg2, kwargs in global_arguments:
        if arg1 in ['-c', '-v', '-L']:
            dump_parser.add_argument(arg2, dest=arg1[1]+'post', **kwargs)
        else:
            dump_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    dump_parser.add_argument('-c', '--cluster', metavar='FILE', help='path to cluster file')
    dump_parser.add_argument('-v', '--vmesh', metavar='FILE', help='path to vmesh file')
    dump_parser.add_argument('-L', '--label', help='comma-separated list of labels')
    dump_parser.add_argument('-m', '--metadata', action='store_true', help='print metadata as @key=val')
    dump_parser.add_argument('--eps', '--epsilon', metavar='EPSILON', type=float, help='margin for half-space gradient calculation (cluster file exports only; see also `tramway.inference.base.neighbours_per_axis`)')
    dump_parser.add_argument('--auto', action='store_true', help='infer default values for missing parameters')
    try:
        dump_parser.add_argument('input_file', nargs='?', help='path to input file')
    except:
        pass


    # extract features
    fsub, _ = _add_subparsers(sub, 'extract', help='extract features from maps', target='features')

    # extract curl
    curl_parser = fsub.add_parser('curl')
    curl_parser.set_defaults(func=_curl)
    for arg1, arg2, kwargs in global_arguments:
        if arg1 in ['-v']:
            curl_parser.add_argument(arg2, dest=arg1[1]+'post', **kwargs)
        else:
            curl_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    curl_parser.add_argument('-L', '--label', '--input-label', help='comma-separated list of input labels')
    curl_parser.add_argument('-l', '--output-label', help='output label')
    curl_parser.add_argument('-r', '--radius', '-d', '--distance', type=int, default=1, help='radius in number of cells')
    try:
        curl_parser.add_argument('input_file', nargs='?', help='path to input file')
    except:
        pass


    # plot artefacts
    psub, _ = _add_subparsers(sub, 'draw', aliases=['show'], help='plot static data',
            target='types of data')

    # plot cells
    cells_parser = psub.add_parser('cells')
    cells_parser.set_defaults(func=_render_cells)
    for arg1, arg2, kwargs in global_arguments:
        cells_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    cells_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
    cells_parser.add_argument('--segment', type=int, help='time segment index (indices range from 0)')
    cells_parser.add_argument('-s', '--min-location-count', type=int, \
        help='minimum number of locations per cell')
    #cells_parser.add_argument('--min-distance', type=float, \
    #    help='minimum distance between neighbour cells (for histograms)')
    #cells_parser.add_argument('--avg-distance', type=float, \
    #    help='average distance between neighbour cells (for histograms)')
    cells_parser.add_argument('-D', '--delaunay', action='store_true', help='plot the Delaunay graph instead of the Voronoi')
    #cells_parser.add_argument('-H', '--histogram', help="plot/print additional histogram(s); any combination of 'c' (cell count histogram), 'd' (distance between neighbouring centers) and 'p' (distance between any pair of locations from distinct neighbouring centers)")
    cells_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')
    try:
        cells_parser.add_argument('input_file', nargs='?', help='path to input file')
    except:
        pass

    # plot map(s)
    _, map_parser = _add_subparsers(psub, 'map', ['maps'])
    map_parser.set_defaults(func=_render_map)
    for arg1, arg2, kwargs in global_arguments:
        map_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    map_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
    map_parser.add_argument('-f', '--feature', '--variable', help='mapped feature name')
    map_parser.add_argument('--segment', type=int, help='time segment index (indices range from 0)')
    map_parser.add_argument('-P', '--points', nargs='?', default=False, help='plot the points; options can be specified as "c=\'r\',a=0.1" (no space, no double quotes)')
    map_parser.add_argument('-D', '--delaunay', nargs='?', default=False, help='plot the Delaunay graph; options can be specified as "c=\'r\',a=0.1" (no space, no double quotes)')
    map_parser.add_argument('-cm', '--colormap', help='colormap name (see https://matplotlib.org/users/colormaps.html)')
    map_parser.add_argument('-c', '--clip', type=float, nargs='?', default=0., help='clip map by absolute values; clipping threshold can be specified as a number of interquartile distances above the median')
    map_parser.add_argument('-cb', '--colorbar', action='store_false', help='do not plot colorbar')
    map_parser.add_argument('--xlim', type=float, nargs='+', help='space-separated couple of limit values for the x axis')
    map_parser.add_argument('--ylim', type=float, nargs='+', help='space-separated couple of limit values for the y axis')
    map_parser.add_argument('--zlim', type=float, nargs='+', help='space-separated couple of limit values for the z axis (plots the landscape in 3D)')
    map_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')
    map_parser.add_argument('--dpi', type=int, help='dots per inch')
    map_parser.add_argument('--inferencemap', action='store_true', help='for field maps, size the arrows wrt cell size only')
    map_parser.add_argument('--title', help='figure title')
    map_parser.add_argument('--unit', help="add units to colorbars, in latex maths; if 'std', set diffusivity to \mu\\rm{m}^2\\rm{s}^{-1}, and potential and force amplitude to k_{\\rm{B}}T")
    try:
        map_parser.add_argument('input_file', nargs='?', help='path to input file')
    except:
        pass


    # animate artefacts
    asub, _ = _add_subparsers(sub, 'animate', aliases=['grab'], help='make a movie')

    # animate trajectories
    try:
        traj_parser = asub.add_parser('trajectories', aliases=['traj'])
    except TypeError: # Py2
        traj_parser = asub.add_parser('trajectories')
    traj_parser.set_defaults(func=_animate_trajectories)
    for arg1, arg2, kwargs in global_arguments:
        traj_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    traj_parser.add_argument('-r', '--frame-rate', '--fps', type=float, help="frames per second")
    traj_parser.add_argument('-c', '--codec', help="the codec to use")
    traj_parser.add_argument('-b', '--bit-rate', '--bitrate', type=int, help="movie bitrate")
    traj_parser.add_argument('--dots-per-inch', '--dpi', type=int, help="dots per inch")
    traj_parser.add_argument('--write-only', action='store_true', default=False, help="do not play the movie")
    traj_parser.add_argument('-s', '--time-step', type=float, help="time step between successive frames")
    traj_parser.add_argument('-u', '--time-unit', type=str, default='s', help="time unit for time display")

    # animate a map
    map_parser = asub.add_parser('map')
    map_parser.set_defaults(func=_animate_map)
    for arg1, arg2, kwargs in global_arguments:
        map_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
    map_parser.add_argument('-l', '-L', '--label', help="comma-separated list of labels to the map")
    map_parser.add_argument('-f', '--feature', help="mapped feature to be rendered")
    map_parser.add_argument('-r', '--frame-rate', '--fps', type=float, help="frames per second")
    map_parser.add_argument('-c', '--codec', help="the codec to use")
    map_parser.add_argument('-b', '--bit-rate', '--bitrate', type=int, help="movie bitrate")
    map_parser.add_argument('--dots-per-inch', '--dpi', type=int, help="dots per inch")
    map_parser.add_argument('--write-only', action='store_true', default=False, help="do not play the movie")
    map_parser.add_argument('-s', '--time-step', type=float, help="time step between successive frames")
    map_parser.add_argument('-u', '--time-unit', type=str, default='s', help="time unit for time display")


    # parse
    args = parser.parse_args()
    args.verbose = args.vpre
    try:
        args.vpost
    except AttributeError:
        args.input = args.ipre
        args.output = args.opre
    else:
        if args.vpost:
            if args.verbose is None:
                args.verbose = args.vpost
            else:
                args.verbose += args.vpost
        args.input = args.ipre + args.ipost
        args.output = args.opre if args.opre else args.opost
        del args.vpost
        del args.ipost
        del args.opost
    del args.vpre
    del args.ipre
    del args.opre
    if args.verbose is None:
        args.verbose = False
    labels = None
    try:
        labels = args.input_label
    except AttributeError:
        try:
            labels = args.label
        except AttributeError:
            pass
    if labels:
        if labels[0] in "'\"" and labels[0] == labels[-1]:
            labels = labels[1:-1]
        if ';' in labels:
            labels = labels.split(';')
        else:
            labels = labels.split(',')
        args.input_label = []
        for label in labels:
            try:
                label = int(label)
            except (TypeError, ValueError):
                pass
            args.input_label.append(label)
    try:
        args.output_label = int(args.output_label)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        args.func(args)
    except AttributeError as e:
        if e.args and 'Namespace' in e.args[0]:
            parser.print_help()
        else:
            raise



if __name__ == '__main__':
    main()

