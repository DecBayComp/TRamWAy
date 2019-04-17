# -*- coding: utf-8 -*-

# Copyright © 2017,2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
from .namedcolumns import *


class Scaler(object):
    """:class:`Scaler` scales data points, point differences (vectors) or distances.

    It initializes itself with the first provided sample (in :meth:`scale_point`), and then applies
    the same transformation to the next samples.

    A default ``Scaler()`` instance does not scale. However, initialization still takes place
    so that :meth:`scaled` properly works.

    It manages a constraint in the calculation of the scaling parameters, forcing a common factors
    over a subset of dimensions. Attribute :attr:`euclidean` controls the selection of this subset.
    Distances are scaled and unscaled only in this subspace, if it is defined.

    Beware that when possible data are scaled in place, but `scaledonly` optional argument, when
    available, never operates in place.

    Attributes:
        init (bool):
            ``True`` as long as `Scaler` has not been initialized.
        center (array or pandas.Series):
            Vector that is substracted to each row of the data matrix to be scaled.
        factor (array or pandas.Series):
            Vector by which each row of the data matrix to be scaled is divided.
            Applies after `center`.
        columns (list or pandas.Index):
            Sequence of column names along which scaling applies. This applies only to
            structured data. `columns` is determined even if `Scaler` is set to do nothing,
            so that :meth:`scaled` can still apply.
            `columns` can be manually set after the first call to :meth:`scale_point` if data
            are not structured (do not have named columns).
        function (callable):
            A function that takes a data matrix as input and returns `center` and `factor`.
            `function` is called once during the first call to :meth:`scale_point`.
        euclidean (list):
            Sequence of names or indices of the columns to be scaled by a common factor.
    """
    __slots__ = ['init', 'center', 'factor', 'columns', 'function', 'euclidean']

    def __init__(self, scale=None, euclidean=None):
        """
        Arguments:
            scale (callable):
                A function that takes a data matrix as input and returns `center` and
                `factor`. `scale` becomes the :attr:`function` attribute.
            euclidean (list):
                Sequence of names or indices of the columns to be scaled by a common
                factor.
        """
        self.init   = True
        self.center = None
        self.factor = None
        self.columns = []
        self.function = scale
        if euclidean and not \
            (isinstance(euclidean, list) and euclidean[1:]):
            raise TypeError('`euclidean` should be a multi-element list')
        self.euclidean = euclidean

    @property
    def ready(self):
        """Returns `True` if scaler is initialized."""
        return not self.init

    def scaled(self, points, asarray=False):
        """Discard columns that are not recognized by the initialized scaler.

        Applies to points and vectors, not distances, surface areas or volumes."""
        if len(self.columns):
            if isstructured(points):
                cols = columns(points)
                coltype = type(cols[0])
                if type(self.columns[0]) is not coltype:
                    if coltype is bytes:
                        coerce = lambda s: s.encode('utf-8')
                    else:
                        coerce = lambda s: s.decode('utf-8')
                    self.columns = [ coerce(c) for c in self.columns ]
                    self.euclidean = [ coerce(c) for c in self.euclidean ]
                points = points[self.columns]
            else:
                try:
                    points = np.asarray(points)[:, self.columns]
                except IndexError:
                    cols = list(self.columns)
                    raise TypeError("the input data array does not feature the following columns: {}".format(cols))
        elif isstructured(points):
            raise ValueError("input data are structured whereas scaler's internal data are not")
        else:
            scaler_data = self.center
            if scaler_data is None:
                scaler_data = self.factor
            if scaler_data is None:
                if self.function:
                    raise RuntimeError('scaler has not been initialized')
            else:
                if isinstance(points, (tuple, list)):
                    points = np.asarray(points)
                if scaler_data.shape[1] != points.shape[1]:
                    raise ValueError('number of columns does not match')
        if asarray:
            points = np.asarray(points)
        return points

    def scale_point(self, points, inplace=True, scaledonly=False, asarray=False):
        """
        Scale data.

        When this method is called for the first time, the `Scaler` instance initializes itself
        for further call of any of its methods.

        Arguments:
            points (array-like):
                Data matrix to be scaled. When :meth:`scale_point` is called for the
                first time, `points` can be structured or not, without the unnecessary
                columns, if any.
                At further calls of any (un-)scaling method, the input data should be in
                the same format with at least the same column names, and may feature
                extra columns.
            inplace (bool):
                Per default, scaling is performed in-place. With ``inplace=False``,
                `points` are first copied.
            scaledonly (bool):
                If ``True``, undeclared columns are stripped away out of the returned
                data.
            asarray (bool):
                If ``True``, the returned data is formatted as a :class:`numpy.array`.

        Returns:
            array-like: With default optional input arguments, the returned variable will be
                a pointer to `points`, not otherwise.
        """
        if isinstance(points, (tuple, list)):
            points = np.asarray(points)
        if self.init:
            # define named columns
            if self.columns:
                raise AttributeError('remove data columns at initialization instead of defining `columns`')
            try:
                self.columns = list(columns(points)) # structured arrays do not support tuple indexing
            except:
                pass
            # backup predefined values
            if self.center is None:
                predefined_centers = []
            elif isinstance(self.center, list):
                predefined_centers = self.center
            if self.factor is None:
                predefined_factors = []
            elif isinstance(self.factor, list):
                predefined_factors = self.factor
            if self.function:
                # calculate centers and factors
                self.center, self.factor = self.function(points)
                # equalize factor for euclidean variables
                if self.euclidean:
                    if isinstance(points, (pd.Series, pd.DataFrame)):
                        xyz = points[self.euclidean].values
                    elif points.dtype.names:
                        xyz = np.asarray(points[list(self.euclidean)])
                    else:
                        xyz = np.asarray(points)[:,self.euclidean]
                    _, self.factor[self.euclidean] = self.function(xyz.flatten())
            # overwrite the coordinates that were actually predefined
            if predefined_centers:
                if self.center is None:
                    self.center = __get_row(points, 0.0)
                for col, val in predefined_centers:
                    self.center[col] = val
            if predefined_factors:
                if self.factor is None:
                    self.factor = __get_row(points, 1.0)
                for col, val in predefined_factors:
                    self.factor[col] = val
            self.init = False
        if not (self.center is None and self.factor is None):
            points = _format(points, inplace)
            if isinstance(points, np.ndarray):
                if self.center is not None:
                    points -= np.asarray(self.center)
                if self.factor is not None:
                    points /= np.asarray(self.factor)
            else:
                if self.center is not None:
                    points -= self.center
                if self.factor is not None:
                    points /= self.factor
        if scaledonly:
            points = self.scaled(points, asarray)
        elif asarray:
            points = np.asarray(points)
        return points

    def unscale_point(self, points, inplace=True):
        """
        Scale data back to original domain.

        The calling `Scaler` instance must have been initialized.

        Arguments:
            points (array-like):
                Scaled data matrix to be unscaled.
            inplace (bool):
                Per default, scaling is performed in-place. With ``inplace=False``,
                `points` are first copied.

        Returns:
            array-like: unscaled data matrix.
        """
        if self.init:
            raise AttributeError('scaler has not been initialized')
        if not (self.center is None and self.factor is None):
            points = _format(points, inplace)
            if isinstance(points, np.ndarray):
                if self.factor is not None:
                    points *= np.asarray(self.factor)
                if self.center is not None:
                    points += np.asarray(self.center)
            else:
                if self.factor is not None:
                    points *= self.factor
                if self.center is not None:
                    points += self.center
        return points


    def scale_vector(self, vect, inplace=True, scaledonly=False, asarray=False):
        """
        Scale vectors.

        The calling `Scaler` instance must have been initialized.

        Arguments:
            vect (array-like):
                Data matrix to be scaled.
            inplace (bool):
                Per default, scaling is performed in-place. With ``inplace=False``,
                `vect` is first copied.
            scaledonly (bool):
                If ``True``, undeclared columns are stripped away out of the returned
                data.
            asarray (bool):
                If ``True``, the returned data is formatted as a :class:`numpy.array`.

        Returns:
            array-like: scaled data matrix.
        """
        if self.init:
            raise AttributeError('scaler has not been initialized')
        if self.factor is not None:
            vect = _format(vect, inplace)
            vect /= self.factor
        if scaledonly:
            vect = self.scaled(vect, asarray)
        elif asarray:
            vect = np.asarray(vect)
        return vect

    def unscale_vector(self, vect, inplace=True):
        """
        Scale vectors back to original range.

        The calling `Scaler` instance must have been initialized.

        Arguments:
            vect (array-like):
                Scaled data matrix to be unscaled.
            inplace (bool):
                Per default, scaling is performed in-place. With ``inplace=False``,
                `points` are first copied.

        Returns:
            array-like: unscaled data matrix.
        """
        if self.init:
            raise AttributeError('scaler has not been initialized')
        if self.factor is not None:
            vect = _format(vect, inplace)
            vect *= self.factor
        return vect

    def scale_size(self, size, dim=None, inplace=True, _unscale=False):
        """
        Scale/unscale lengths, surface areas, volumes and other scalar sizes.

        The calling `Scaler` instance must have been initialized.

        Arguments:
            size (array-like):
                Values to be scaled, per element.
            dim (int):
                Number of characteristic dimensions, with 0 referring to all the
                euclidean dimensions (e.g. lengths: 1, areas: 2, volumes: 0).
            inplace (bool):
                Per default, scaling is performed in-place. With ``inplace=False``,
                `size` is first copied.
            _unscale (bool):
                If ``True``, unscales instead.

        Returns:
            array-like: scaled values.
        """
        if self.init:
            raise AttributeError('scaler has not been initialized')
        if self.factor is not None:
            _dim = len(self.euclidean)
            if not dim:
                dim = _dim
            if _dim < min(1, dim):
                raise ValueError('not enough euclidean dimensions')
            try:
                factor = self.factor[self.euclidean[0]]
            except KeyError:
                # on loading Py2-generated Series or DataFrame from an rwa file,
                # PyTables may convert Py2 str in index/columns into Py3 str;
                # as a consequence `columns` and `euclidean` should also be converted
                self.columns = [ name.decode('utf-8') for name in self.columns ]
                self.euclidean = [ name.decode('utf-8') for name in self.euclidean ]
                factor = self.factor[self.euclidean[0]]
            if self.euclidean[1:] and not np.all(self.factor[self.euclidean[1:]] == factor):
                raise ValueError('the scaling factors for the euclidean variables are not all equal')
            size = _format(size, inplace)
            if 1 < dim:
                factor **= dim
            if _unscale:
                size *= factor
            else:
                size /= factor
        return size

    def scale_distance(self, dist, inplace=True):
        return self.scale_size(dist, 1, inplace)

    def unscale_distance(self, dist, inplace=True):
        return self.scale_size(dist, 1, inplace, True)

    def scale_length(self, dist, inplace=True):
        return self.scale_size(dist, 1, inplace)

    def unscale_length(self, dist, inplace=True):
        return self.scale_size(dist, 1, inplace, True)

    def scale_surface_area(self, area, inplace=True):
        return self.scale_size(area, 2, inplace)

    def unscale_surface_area(self, area, inplace=True):
        return self.scale_size(area, 2, inplace, True)

    def scale_volume(self, vol, inplace=True):
        return self.scale_size(vol, 0, inplace)

    def unscale_volume(self, vol, inplace=True):
        return self.scale_size(vol, 0, inplace, True)


def _whiten(x):
    '''Scaling function for :class:`Scaler`. Performs ``(x - mean(x)) / std(x)``. Consider using
    :func:`whiten` instead.'''
    scaling_center = x.mean(axis=0)
    scaling_factor = x.std(axis=0, ddof=0)
    return (scaling_center, scaling_factor)

def whiten(): # should be a function so that each new instance is a distinct one
    """Returns a :class:`Scaler` that scales data ``x`` following: ``(x - mean(x)) / std(x)``."""
    return Scaler(_whiten)


def _unitrange(x):
    '''Scaling function for :class:`Scaler`. Performs ``(x - min(x)) / (max(x) - min(x))``.
    Consider using :func:`unitrange` instead.'''
    scaling_center = x.min(axis=0)
    scaling_factor = x.max(axis=0) - scaling_center
    return (scaling_center, scaling_factor)

def unitrange():
    """Returns a :class:`Scaler` that scales data ``x`` following:
    ``(x - min(x)) / (max(x) - min(x))``."""
    return Scaler(_unitrange)


def __get_row(points, fill=None):
    if isinstance(points, (pd.Series, pd.DataFrame)):
        row = points.iloc[0]
    else:
        row = points[0]
    if fill is not None:
        if isinstance(points, (tuple, list)):
            row = np.asarray(row)
        row.fill(fill)
    return row


def _format(points, inplace):
    if inplace:
        if isinstance(points, tuple):
            raise TypeError('cannot modify tuple inplace')
        elif isinstance(points, list):
            raise NotImplementedError('cannot modify list inplace')
    else:
        if isinstance(points, (tuple, list)):
            points = np.asarray(points)
        else:
            points = points.copy()
    return points


__all__ = [
    'Scaler',
    'whiten',
    'unitrange',
    ]

