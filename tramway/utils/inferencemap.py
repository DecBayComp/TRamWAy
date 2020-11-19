#!/usr/bin/env python

from tramway.core import *
from tramway.core.hdf5 import *
from tramway.tessellation import CellStats
from tramway.inference import Distributed, Translocations, Maps, distributed, neighbours_per_axis

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from collections import namedtuple, OrderedDict

import os
import argparse
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser


# cluster
__zone__ = 'ZONE:'
__ntr__ = 'NUMBER_OF_TRANSLOCATIONS:'
__x__ = 'X-CENTRE:'
__y__ = 'Y-CENTRE:'
__area__ = 'AREA:'
__convhull__ = 'AREA_CONVHULL:'
__nleft__ = 'NUMBER_OF_LEFT_NEIGHBOURS:'
__nright__ = 'NUMBER_OF_RIGHT_NEIGHBOURS:'
__ntop__ = 'NUMBER_OF_TOP_NEIGHBOURS:'
__nbottom__ = 'NUMBER_OF_BOTTOM_NEIGHBOURS:'
__left__ = 'LEFT_NEIGHBOURS:'
__right__ = 'RIGHT_NEIGHBOURS:'
__top__ = 'TOP_NEIGHBOURS:'
__bottom__ = 'BOTTOM_NEIGHBOURS:'
__coord__ = ['DX', 'DY', 'DT']


def import_cluster_file(input_file, return_areas=False):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    if return_areas:
        areas = []
    cells = OrderedDict()
    ncells = 0
    adjacency = []
    i = None
    ntr_defined = x_defined = y_defined = area_defined = convhull_defined = False
    C = OrderedDict
    neighbours_prefixes, neighbours, nneighbours, neighbours_code = C(), C(), C(), C()
    for side, prefixes, code in (
            ('left', (__left__, __nleft__), -1),
            ('right', (__right__, __nright__), 1),
            ('top', (__top__, __ntop__), 2),
            ('bottom', (__bottom__, __nbottom__), -2),
        ):
        neighbours_prefixes[side] = prefixes # static
        neighbours[side] = [] # = undefined
        nneighbours[side] = None # = undefined
        neighbours_code[side] = code # static
    coord_section = False
    for l, line in enumerate(lines+['ZONE: -1']):
        line = line.strip()
        if not line or line[0] in '#;':
            continue
        if line.split() == __coord__:
            if i is None:
                raise IOError('not a cluster file')
            coord_section = True
            continue
        elif coord_section:
            try:
                tr = [ float(k) for k in line.split() ]
            except ValueError:
                if ntr_defined and len(coord) != ntr:
                    print('cell {}: wrong number of translocations'.format(i))
                coord_section = False
            else:
                coord.append(tr)
                continue
        if line.startswith(__zone__):
            if i is not None:
                # finalize current cell
                cols = ['x', 'y', 't']
                cell = Translocations(i - 1, pd.DataFrame(np.array(coord), columns=cols))
                cell.space_cols = cols[:2]
                cell.time_col = cols[-1]
                if x_defined and y_defined:
                    cell.center = np.array([x, y])
                if area_defined and return_areas:
                    areas.append((i - 1, area))
                if convhull_defined:
                    cell.volume = convhull
                cells[i - 1] = cell
                ncells = max(i, ncells)
                all_neighbours = []
                for side in neighbours:
                    if neighbours[side]:
                        if not (nneighbours[side] is None or len(neighbours[side]) == nneighbours[side]):
                            print('cell {}: wrong number of {} neighbours'.format(i, side))
                        all_neighbours += [ (j, neighbours_code[side]) for j in neighbours[side] ]
                    elif 0 < nneighbours[side]:
                        print('cell {}: {} neighbours are not defined'.format(i, side))
                if all_neighbours:
                    ncells = max(ncells, *[ j for j,_ in all_neighbours ])
                    adjacency += [ (i-1,j-1,k) for j,k in all_neighbours ]
            # initialize new cell parsing
            i = int(line[len(__zone__):])
            ntr_defined = x_defined = y_defined = area_defined = convhull_defined = False
            for side in neighbours:
                neighbours[side], nneighbours[side] = [], None
            coord = []
        elif line.startswith(__ntr__):
            ntr = int(line[len(__ntr__):])
            ntr_defined = True
        elif line.startswith(__x__):
            x = float(line[len(__x__):])
            x_defined = True
        elif line.startswith(__y__):
            y = float(line[len(__y__):])
            y_defined = True
        elif line.startswith(__area__):
            area = float(line[len(__area__):])
            area_defined = True
        elif line.startswith(__convhull__):
            convhull = float(line[len(__convhull__):])
            convhull_defined = True
        else:
            cont = False
            for side in neighbours_prefixes:
                prefix, prefix_n = neighbours_prefixes[side]
                if line.startswith(prefix_n):
                    if nneighbours[side] is not None:
                        print('cell {}: duplicate field for the number of {} neighbours'.format(i, side))
                    try:
                        nneighbours[side] = int(line[len(prefix_n):])
                    except ValueError:
                        print('cell {}: wrong format for the number of {} neighbours'.format(i, side))
                    else:
                        cont = True
                        break
                elif line.startswith(prefix):
                    if neighbours[side]:
                        print('cell {}: duplicate field for {} neighbours'.format(i, side))
                    try:
                        neighbours[side] = [ int(k) for k in line[len(prefix):].split() ]
                    except ValueError:
                        print('cell {}: wrong format for {} neighbours'.format(i, side))
                    else:
                        cont = True
                        break
            if not cont:
                print('skipping line {}:'.format(l))
                print(line)
    # make the Distributed object
    i, j, k = zip(*adjacency)
    adjacency = sparse.coo_matrix((k, (i, j)), shape=(ncells, ncells)).tocsr()
    for i in cells:
        cell = cells[i]
        neighbours = np.vstack([ cells[j].center for j in adjacency[i].indices ])
        cell.span = neighbours - cell.center[np.newaxis,:]
    distr = Distributed(cells, adjacency)

    if return_areas:
        index, areas = zip(*areas)
        areas = pd.Series(areas, index=index)
        return distr, areas
    else:
        return distr


def import_vmesh_file(input_file):
    maps = pd.read_table(input_file, skiprows=7, index_col=0)
    maps.index -= 1
    maps = maps.rename(columns={'x-Force': 'force x', 'y-Force': 'force y',
        'Translocations': 'n', 'x-Centre': 'x', 'y-Centre': 'y',
        'Diffusion': 'diffusivity', 'Potential': 'potential'})
    maps = Maps(maps)
    with open(input_file, 'r') as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            key, value = line.split(': ')
            if key == 'Input File':
                pass
            elif key == 'Inference Mode':
                maps.mode = value
            elif key == 'Localization Error':
                value, unit = value.split()
                assert unit == '[nm]'
                value = float(value) * 1e3
                maps.localization_error = value
            elif key == 'Number of Zones':
                pass
            elif key == 'Diffusion Prior':
                maps.diffusivity_prior = float(value)
            elif key == 'Potential Prior':
                maps.potential_prior = float(value)
            elif key == "Jeffreys' Prior":
                maps.jeffreys_prior = bool(value)
    return maps


def import_file(root_file, label, cluster_file=None, vmesh_file=None, output_file=None, **kwargs):
    clabel = vlabel = None
    if isinstance(label, (tuple, list)):
        if label:
            if label[1:]:
                try:
                    clabel, vlabel = label
                except ValueError:
                    raise ValueError('too many labels')
            else:
                clabel = label[0]
    else:
        clabel = label # can be None
    try:
        tree = load_rwa(root_file, verbose=False)
    except:
        trajectory_file = root_file
        xyt = load_xyt(trajectory_file)
        tree = Analyses(xyt)
        if not output_file:
            output_file = os.path.splitext(root_file)[0] + '.rwa'
    else:
        if not output_file:
            output_file = root_file
    if cluster_file:
        distr = import_cluster_file(cluster_file)
        clabel = tree.autoindex(clabel)
        tree.add(distr, label=clabel)
    if vmesh_file:
        maps = import_vmesh_file(vmesh_file)
        tree[clabel].add(maps, label=vlabel)
    save_rwa(output_file, tree, force=True, **kwargs)


def export_to_cluster_file(cells, cluster_file, neighbours=None, neighbours_kwargs={}, new_cell=Translocations, distributed_kwargs={}, **kwargs):
    not_2D = ValueError('data are not 2D')
    if isinstance(cells, Distributed):
        if cells.dim != 2:
            raise not_2D
        zones = cells
    else:
        if cells.tessellation._cell_centers.shape[1] != 2:
            raise not_2D
        zones = distributed(cells, new_cell=new_cell, **distributed_kwargs)
    if neighbours is None:
        neighbours = neighbours_per_axis
    with open(cluster_file, 'w') as f:
        for i in zones:
            zone = zones[i]
            f.write('ZONE: {}\n'.format(i+1))
            f.write('NUMBER_OF_TRANSLOCATIONS: {}\n'.format(len(zone)))
            f.write('X-CENTRE: {}\n'.format(zone.center[0]))
            f.write('Y-CENTRE: {}\n'.format(zone.center[1]))
            #area = np.prod(np.max(zone.span, axis=0) - np.min(zone.span, axis=0))
            area = zone.volume
            f.write('AREA: {}\n'.format(area))
            f.write('AREA_CONVHULL: {}\n'.format(area))
            below, above = neighbours(i, zones, **neighbours_kwargs)
            _neighbours = zones.adjacency[i].indices
            _neighbours = OrderedDict(
                left = _neighbours[below[0]],
                right = _neighbours[above[0]],
                top = _neighbours[above[1]],
                bottom = _neighbours[below[1]],
                )
            for side in _neighbours:
                f.write('NUMBER_OF_{}_NEIGHBOURS: {}\n'.format(
                    side.upper(), _neighbours[side].size))
                if _neighbours[side].size:
                    f.write('{}_NEIGHBOURS: '.format(side.upper()))
                    for j in _neighbours[side]:
                        f.write('{}\t'.format(j+1))
                    f.write('\n')
            f.write('DX\tDY\tDT\n')
            for dx, dy, dt in np.hstack((zone.dr, zone.dt[:,np.newaxis])):
                f.write('{}\t{}\t{}\n'.format(dx, dy, dt))
            f.write('\n\n')


def export_to_vmesh_file(cells, maps, vmesh_file, cluster_file=None, auto=False, new_cell=Translocations, distributed_kwargs={}, **kwargs):
    not_2D = ValueError('data are not 2D')
    if isinstance(cells, Distributed):
        if cells.dim != 2:
            raise not_2D
        zones = cells
    else:
        if cells.tessellation._cell_centers.shape[1] != 2:
            raise not_2D
        zones = distributed(cells, new_cell=new_cell, **distributed_kwargs)
    with open(vmesh_file, 'w') as f:
        # Input File
        if cluster_file is None and auto:
            dirname, basename = os.path.split(vmesh_file)
            cluster_files = [ fn
                for fn in os.listdir(dirname if dirname else '.')
                if fn.endswith('.cluster') and basename.startswith(fn[:8]) ]
            if cluster_files:
                if cluster_files[1:]:
                    print('multiple cluster files match:')
                    for fn in cluster_files:
                        print('\t'+fn)
                cluster_file = cluster_files[0]
            else:
                cluster_file = ''
        if cluster_file is not None:
            f.write('Input File: {}\n'.format(cluster_file))
        # Inference Mode
        mode = maps.mode.upper()
        if mode is None and auto:
            if 'potential' in maps.features:
                mode = 'DV'
            elif 'force' in maps.features:
                mode = 'DF'
            elif 'drift' in maps.features:
                mode = 'DD'
            else:
                mode = 'D'
        if mode is not None:
            f.write('Inference Mode: {}\n'.format(mode))
        # Localization Error
        err = maps.localization_error
        if err is None and auto:
            err = .03
        if err is not None:
            f.write('Localization Error: {:f} [nm]\n'.format(err*1e3))
        # Number of Zones
        f.write('Number of Zones: {}\n'.format(len(zones)))
        # Diffusion Prior
        dp = maps.diffusivity_prior
        if dp is None and auto:
            dp = 0.
        if dp is not None:
            f.write('Diffusion Prior: {:f}\n'.format(dp))
        # Potential Prior
        vp = maps.potential_prior
        if vp is None and auto:
            vp = 0.
        if vp is not None:
            f.write('Potential Prior: {:f}\n'.format(vp))
        # Jeffreys' Prior
        jp = maps.jeffreys_prior
        if jp is None and auto:
            jp = 0
        if jp is not None:
            f.write("Jeffreys' Prior: {:d}\n".format(jp))
        # table header
        f.write('\nZone ID\tTranslocations\tx-Centre\ty-Centre')
        na = np.isnan(maps.maps.values) | np.isinf(maps.maps.values)
        ok = ~np.all(na, axis=1)
        data0 = [ [i, len(zones[i]), zones[i].center] for i in maps.maps.index[ok] ]
        data1 = []
        nvars = 0
        if 'diffusivity' in maps.features:
            f.write('\tDiffusion')
            data1.append(maps['diffusivity'].values[ok])
            nvars += 1
        if 'force' in maps.features:
            f.write('\tx-Force\ty-Force')
            data1.append(maps['force'].values[ok])
            nvars += 2
        if 'potential' in maps.features:
            f.write('\tPotential')
            data1.append(maps['potential'].values[ok])
            nvars += 1
        # table data
        data1 = np.hstack(data1)
        fmt0 = '\n{:d}\t{:d}\t{:f}\t{:f}'
        for r, s in zip(data0, data1):
            i, n, r = r
            s = [ 'NA' if np.isnan(x) or np.isinf(x) else x for x in s ]
            fmt1 = '\t{' + '}\t{'.join('' if x == 'NA' else ':f' for x in s) + '}'
            f.write((fmt0+fmt1).format(i, n, *(r.tolist() + s)))


def export_file(rwa_file, label, cluster_file=None, vmesh_file=None, eps=None, **kwargs):
    if not (cluster_file or vmesh_file):
        raise ValueError('output filename(s) not defined')
    if not label:
        raise ValueError('labels are required')
    vlabel = None
    if isinstance(label, (tuple, list)):
        if label[1:]:
            clabel, vlabel = label
        else:
            clabel = label[0]
    else:
        clabel = label
    if vlabel is None and vmesh_file:
        raise ValueError('label for vmesh file is required')

    analyses = load_rwa(rwa_file)

    if vmesh_file:

        cells, maps = find_artefacts(analyses, ((CellStats, Distributed), Maps), (clabel, vlabel))

    elif cluster_file:

        cells, = find_artefacts(analyses, ((CellStats, Distributed), ), (clabel, ))

    if cluster_file:

        if eps is not None:
            neighbours_kwargs = kwargs.get('neighbours_kwargs', {})
            neighbours_kwargs['eps'] = eps
            kwargs['neighbours_kwargs'] = neighbours_kwargs

        export_to_cluster_file(cells, cluster_file, **kwargs)

    if vmesh_file:

        export_to_vmesh_file(cells, maps, vmesh_file, cluster_file, **kwargs)



def main():
    parser = argparse.ArgumentParser(prog='inferencemap',
        description='InferenceMAP-TRamWAy file converter.')
    parser.add_argument('-t', '--trajectories', metavar='FILE', help='path to trajectory file')
    parser.add_argument('-c', '--cluster', metavar='FILE', help='path to cluster file')
    parser.add_argument('-v', '--vmesh', metavar='FILE', help='path to vmesh file')
    parser.add_argument('-l', '-L', '--label', metavar='LABEL', action='append', default=[], help='labels; the first one identifies the cluster data in the rwa file, the second one identifies the vmesh data') # ideally '-l' is to import, '-L' to export
    import_or_export = parser.add_mutually_exclusive_group()
    import_or_export.add_argument('-o', '--output', metavar='FILE', help='path to the rwa file which to import InferenceMAP files to')
    import_or_export.add_argument('-i', '--input', metavar='FILE', help='path to the rwa file which to export from')
    parser.add_argument('--eps', '--epsilon', metavar='EPSILON', type=float, help='margin for half-space gradient calculation (cluster file exports only; see also `tramway.inference.base.neighbours_per_axis`)')
    parser.add_argument('--auto', action='store_true', help='infer default values for missing parameters')
    args = parser.parse_args()
    kwargs = {}
    if args.input:
        export_file(args.input, args.label, args.cluster, args.vmesh,
            eps=args.eps, auto=args.auto)
    else:
        if args.trajectories:
            root = args.trajectories
        elif args.output and os.path.exists(args.output):
            root = args.output
        else:
            raise ValueError('trajectory file not defined')
        import_file(root, args.label, args.cluster, args.vmesh, args.output)


if __name__ == '__main__':
    main()

