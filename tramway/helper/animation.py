# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core.analyses.base import Analyses
from tramway.plot.animation import *
import os.path

def animate_trajectories_2d_helper(input_data, *args, **kwargs):
    """
    Animate 2D trajectories.

    Arguments:

        input_data (pandas.DataFrame or tramway.core.analyses.base.Analyses or str):
            nxyt data, or analysis tree, or path to xyt file.

        columns (str or list): (comma-separated) list of column names if input data
            are to be loaded; keyworded-only; see also :func:`~tramway.core.xyt.load_xyt`.

    The other arguments are passed to :func:`~tramway.plot.animation.xyt.animate_trajectories_2d`.
    """
    from tramway.plot.animation.xyt import animate_trajectories_2d
    import pandas as pd

    columns = kwargs.pop('columns', None)

    if isinstance(input_data, pd.DataFrame):
        xyt = input_data
    elif isinstance(input_data, Analyses):
        xyt = input_data.data
    else:
        input_file = os.path.expanduser(input_data)
        if not os.path.isfile(input_file):
            raise OSError("file '{}' not found".format(input_file))
        load_kwargs = {}
        if columns is not None:
            if isinstance(columns, str):
                columns = columns.split(',')
            load_kwargs['columns'] = columns
        from tramway.core.xyt import load_xyt
        try:
            xyt = load_xyt(input_file, **load_kwargs)
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            from tramway.core.hdf5 import load_rwa
            xyt = load_rwa(input_file, lazy=True).data

    animate_trajectories_2d(xyt, *args, **kwargs)


def animate_map_2d_helper(input_data, output_file=None, label=None, feature=None, variable=None, **kwargs):
    """
    Animate 2D maps.

    Arguments:

        input_data (tramway.core.analyses.base.Analyses or str): analysis tree or path to rwa file.

        output_file (str): path to .mp4 file; if None, the video stream is dumped into
            a temporary file that is deleted after being played.

        label (str or list): path to the maps as a (comma-separated) list of labels.

        feature/variable (str): name of the mapped feature to be rendered.

    The other keyword arguments are passed to :func:`~tramway.plot.animation.map.animate_map_2d`.
    """
    from tramway.core.analyses import find_artefacts
    from tramway.tessellation.base import Partition
    from tramway.inference.base import Maps
    from tramway.plot.animation.map import animate_map_2d

    if isinstance(input_data, Analyses):
        analyses = input_data
    else:
        from tramway.core.hdf5 import load_rwa
        input_file = os.path.expanduser(input_data)
        if not os.path.isfile(input_file):
            raise OSError("file '{}' not found".format(input_file))
        analyses = load_rwa(input_file, lazy=True)

    if isinstance(label, str):
        _label = []
        for _l in label.split(','):
            try:
                _l = int(_l)
            except (TypeError, ValueError):
                pass
            _label.append(_l)
        label = _label

    cells, maps = find_artefacts(analyses, (Partition, Maps), label)

    if feature is None:
        feature = variable
    if feature is None:
        if maps.features[1:]:
            raise ValueError('multiple mapped features found: {}'.format(maps.features))
        feature = maps.features[0]
    _map = maps[feature]

    animate_map_2d(_map, cells, output_file, **kwargs)


__all__ = ['animate_trajectories_2d_helper', 'animate_map_2d_helper', 'VideoReader', 'VideoWriter', 'VideoWriterReader']

