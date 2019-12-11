# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
from collections import OrderedDict
import traceback
from warnings import warn


class SpatialOperatorFailure(UserWarning):
    pass


def gradient_map(cells, feature=None, verbose=False, **kwargs):
    grad_kwargs = get_grad_kwargs(**kwargs)
    if feature:
        if isinstance(feature, str):
            feature = [f.strip() for f in feature.split(',')]
    else:
        raise ValueError('please define `feature`')
    grads = None
    for f in feature:
        I, X = [], []
        for i in cells:
            try:
                x = getattr(cells[i], f)
            except (SystemExit, KeyboardInterrupt):
                raise
            except:
                pass
            else:
                if x is not None:
                    I.append(i)
                    X.append(x)
        #I = np.array(I)
        X = np.hstack(X) # hstack instead of array just in the case elements in X are single-element arrays
        #print(X, X.dtype)
        reverse_index = np.full(max(cells.keys())+1, -1, dtype=int)
        reverse_index[I] = np.arange(len(I))
        index, gradX = [], []
        for i in I:
            try:
                g = cells.grad(i, X, reverse_index, **grad_kwargs)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                if verbose:
                    traceback.print_exc()
                else:
                    warn(e.args[0], SpatialOperatorFailure)
                g = None
            if g is not None:
                index.append(i)
                gradX.append(g)
        if not index:
            return None
        gradX = np.vstack(gradX)
        assert gradX.shape[1:] and gradX.shape[1] == len(cells.space_cols)
        columns = [ '{} {}'.format(f, col) for col in cells.space_cols ]
        gradX = pd.DataFrame(gradX, index=index, columns=columns)
        if grads is None:
            grads = gradX
        else:
            grads.join(gradX)
    return grads


default_selection_angle = .9

def get_grad_kwargs(_kwargs=None, gradient=None, grad_epsilon=None, grad_selection_angle=None, compatibility=None, grad=None, epsilon=None, **kwargs):
    """Parse :func:`grad1` keyworded arguments.

    Identifies 'grad_'- or 'gradient_'-prefixed keyword arguments.

    Arguments:

        _kwargs (dict):
            mutable keyword arguments; all the keywords below are popped out.

        gradient (str): either *grad1* or *gradn*.

        grad_epsilon (float): `eps` argument for :func:`grad1` (or :func:`neighbours_per_axis`).

        grad_selection_angle (float): `selection_angle` argument for :func:`grad1` (or :func:`neighbours_per_axis`).

        compatibility (bool): backward compatibility with InferenceMAP.

        grad (str): alias for `gradient`.

        epsilon (float): alias for `grad_epsilon`.

    Returns:

        dict: keyworded arguments to :meth:`~tramway.inference.base.FiniteElements.grad`.

    Note: `grad` and `gradient` are currently ignored.
    """
    grad_kwargs = {}

    if _kwargs is not None:
        assert isinstance(_kwargs, dict)
        try:
            _epsilon = _kwargs.pop('epsilon')
        except KeyError:
            pass
        else:
            if epsilon is None:
                epsilon = _epsilon
            elif epsilon != _epsilon:
                # TODO: warn or raise an exception
                pass
        try:
            _grad_epsilon = _kwargs.pop('grad_epsilon')
        except KeyError:
            pass
        else:
            if grad_epsilon is None:
                grad_epsilon = _grad_epsilon
            elif grad_epsilon != _grad_epsilon:
                # TODO: warn or raise an exception
                pass
        try:
            _compatibility = _kwargs.pop('compatibility')
        except KeyError:
            pass
        else:
            if compatibility is None:
                compatibility = _compatibility
            elif compatibility != _compatibility:
                # TODO: warn or raise an exception
                pass
        try:
            _grad_selection_angle = _kwargs.pop('grad_selection_angle')
        except KeyError:
            pass
        else:
            if grad_selection_angle is None:
                grad_selection_angle = _grad_selection_angle
            elif grad_selection_angle != _grad_selection_angle:
                # TODO: warn or raise an exception
                pass

        for kw in dict(_kwargs):
            if kw.startswith('grad_'):
                grad_kwargs[kw[5:]] = _kwargs.pop(kw)
            if kw.startswith('gradient_'):
                grad_kwargs[kw[9:]] = _kwargs.pop(kw)

    for kw in dict(kwargs):
        if kw.startswith('grad_'):
            grad_kwargs[kw[5:]] = kwargs.get(kw)
        elif kw.startswith('gradient_'):
            grad_kwargs[kw[9:]] = kwargs.get(kw)

    if epsilon is not None:
        if grad_epsilon is None:
            grad_epsilon = epsilon
        elif epsilon != grad_epsilon:
            raise ValueError('`epsilon` is an alias for `grad_epsilon`; these arguments do not admit distinct values')
    if grad_epsilon is not None:
        if compatibility:
            warn('grad_epsilon breaks backward compatibility with InferenceMAP', RuntimeWarning)
        if grad_selection_angle:
            warn('grad_selection_angle is not compatible with epsilon and will be ignored', RuntimeWarning)
        grad_kwargs['eps'] = grad_epsilon
    else:
        if grad_selection_angle:
            if compatibility:
                warn('grad_selection_angle breaks backward compatibility with InferenceMAP', RuntimeWarning)
        else:
            grad_selection_angle = default_selection_angle
        if grad_selection_angle:
            grad_kwargs['selection_angle'] = grad_selection_angle

    return grad_kwargs

def setup_with_grad_arguments(setup):
    """Add :meth:`~tramway.inference.base.FiniteElements.grad` related arguments
    to inference plugin setup.

    Input argument `setup` is modified inplace.
    """
    args = setup.get('arguments', OrderedDict())
    if not ('gradient' in args or 'grad' in args):
        args['gradient'] = ('--grad', dict(help="spatial gradient implementation; any of 'grad1', 'gradn'"))
    if not ('grad_epsilon' in args or 'grad' in args):
        args['grad_epsilon'] = dict(args=('--eps', '--epsilon'), kwargs=dict(type=float, help='if defined, every spatial gradient component can recruit all of the neighbours, minus those at a projected distance less than this value'), translate=True)
    if 'grad_selection_angle' not in args:
        if 'compatibility' in args:
            compatibility_note = 'if not -c, '
        else:
            compatibility_note = ''
        args['grad_selection_angle'] = ('-a', dict(type=float, help='top angle of the selection hypercone for neighbours in the spatial gradient calculation (1= pi radians; {}default is: {})'.format(compatibility_note, default_selection_angle)))
    setup['arguments'] = args


setup = dict(
        infer='gradient_map',
        arguments=OrderedDict((
            ('feature', ('-f', dict(help='feature or comma-separated list of features which gradient is expected'))),
            ('verbose',             ()),
        )))
setup_with_grad_arguments(setup)


__all__ = ['default_selection_angle', 'get_grad_kwargs', 'setup_with_grad_arguments', 'setup', 'gradient_map']

