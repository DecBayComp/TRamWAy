#!/usr/bin/env python

from tramway.core import *
from tramway.inference import Distributed, Translocations, Maps

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from collections import namedtuple, OrderedDict

import os.path
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


Area = namedtuple('Area', ['volume'])

def import_cluster_file(input_file):
        with open(input_file, 'r') as f:
                lines = f.readlines()
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
                                if convhull_defined:
                                        cell.boundary = Area(volume=convhull)
                                #if area_defined or convhull_defined:
                                #       cell.cache = dict()
                                #if area_defined:
                                #       cell.cache['area'] = np.full((2,), area)
                                #if convhull_defined:
                                #       cell.cache['convhull'] = np.full((2,), convhull)
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


def import_file(root_file, cluster_file=None, vmesh_file=None, output_file=None, clabel=None, vlabel=None, **kwargs):
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
        save_rwa(output_file, tree, **kwargs)


def main():
        parser = argparse.ArgumentParser(prog='imap',
                description='InferenceMAP to TRamWAy file converter.')
        parser.add_argument('-t', '--trajectories', metavar='FILE', help='path to trajectory file')
        parser.add_argument('-c', '--cluster', metavar='FILE', help='path to cluster file')
        parser.add_argument('-l1', '--clabel', metavar='LABEL', help='label for the imported cluster data')
        parser.add_argument('-v', '--vmesh', metavar='FILE', help='path to vmesh file')
        parser.add_argument('-l2', '--vlabel', metavar='LABEL', help='label for the imported vmesh data')
        parser.add_argument('-o', '--output', metavar='FILE', help='path to output file')
        args = parser.parse_args()
        kwargs = {}
        if args.trajectories:
                root = args.trajectories
        elif args.output and os.path.exists(args.output):
                root = args.output
        else:
                raise ValueError('trajectory file not defined')
        import_file(root, args.cluster, args.vmesh, args.output, args.clabel, args.vlabel)


if __name__ == '__main__':
        main()

