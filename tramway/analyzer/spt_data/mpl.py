# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import AnalyzerNode
import warnings
from collections import deque
import numpy as np
import pandas as pd
from tramway.core.xyt import iter_frames, iter_trajectories
from tramway.plot.mesh import __colors__
from matplotlib import animation


class LineDealer(object):
    """
    Makes glyphs for trajectories to be drawn as lines and
    keeps track of the trajectories passed to `animate`
    so that glyphs are freed as soon as possible and
    can be assigned to new trajectories.
    """
    __slots__ = ('available_glyphs', 'active_glyphs')
    def __init__(self, ax, n=None, colors=__colors__, **kwargs):
        """
        Arguments:
        
            ax (Axes): matplotlib axes.
            
            n (int): number of lines to draw.
            
            colors (Sequence): sequence of colors.
            
        Extra keyword arguments are passed to `ax.plot`.
        """
        if n is None:
            n = len(colors)
        self.available_glyphs = deque()
        for k in range(n):
            glyph, = ax.plot([], [], '-', color=colors[k % len(colors)], **kwargs)
            self.available_glyphs.append(glyph)
        self.active_glyphs = {}
    def plot(self, trajs):
        """
        Updates the glyphs with the active trajectories.
        
        Arguments:
        
            trajs (list): list of active trajectories (dataframes).
            
        Returns:
        
            tuple: sequence of updated glyphs.
        """
        updated_glyphs = []
        previous_traj_ids = set(list(self.active_glyphs.keys()))
        current_traj_ids = set([ traj['n'].iloc[0] for traj in trajs ])
        for traj_ix in previous_traj_ids - current_traj_ids:
            glyph = self.active_glyphs.pop(traj_ix)
            glyph.set_data([], [])
            updated_glyphs.append(glyph)
            self.available_glyphs.append(glyph)
        for traj in trajs:
            if len(traj)==1:
                continue
            traj_ix = traj['n'].iloc[0]
            try:
                glyph = self.active_glyphs[traj_ix]
            except KeyError:
                if self.available_glyphs:
                    self.active_glyphs[traj_ix] = glyph = self.available_glyphs.popleft()
                else:
                    warnings.warn('not enough allocated glyphs; increase n')
                    continue
            else:
                # checks
                if True:
                    previous_data_len = len(glyph.get_xdata())
                    current_data_len = len(traj)
                    if current_data_len != previous_data_len+1:
                        print('x_prev=',glyph.get_xdata(), 'y_prev=',glyph.get_ydata())
                        print(traj)
                        warnings.warn('frames are skipped')
                #
            glyph.set_data(traj['x'].values, traj['y'].values)
            updated_glyphs.append(glyph)
        return tuple(updated_glyphs)
    def init_func(self):
        """
        To be passed as argument `init_func` to `FuncAnimation`.
        """
        return tuple(self.available_glyphs)
    def animate(self, trajs, dt):
        """
        Makes the function to be passed as second positional argument to `FuncAnimation`.
        
        Arguments:
        
            trajs (DataFrame): trajectories.
            
            dt (float): time step.
            
        Returns:
        
            callable: function that takes a list of index slices.
        """
        t0 = trajs['t'].min()
        k = int(np.round(t0/dt))
        k = [k] # wrapped in a mutable object (a list),
                # so that it is available from within _animate
        def _animate(_traj_ids):
            _t = k[0]*dt
            _trajs = []
            for _i,_j in _traj_ids:
                _traj = trajs.iloc[_i:_j]
                assert 0<_traj.shape[0]
                _until_t = _traj['t']<_t+dt/2
                if not np.any(_until_t) or _traj['t'].iloc[-1]<_t-dt/2:
                    raise RuntimeError('synchronization error\n\tt= {};\ttrajectory[\'t\']= {}'.format(_t, _traj['t'].tolist()))
                _trajs.append(_traj[_until_t])
            k[0] += 1
            return self.plot(_trajs)
        return _animate


class Mpl(AnalyzerNode):
    __slots__ = ()
    @property
    def animate_cls(self):
        return LineDealer
    def animate(self, fig, trajs=None, axes=None, xlim='auto', ylim=None, aspect='equal', **kwargs):
        """
        Arguments:

            fig (matplotlib.figure.Figure): figure.

            trajs (pandas.DataFrame): trajectories (and NOT translocations).

            axes (matplotlib.axes.Axes): figure axes.

            xlim (tuple or None): abscissa limits.

            ylim (tuple or None): ordinates limits.

            aspect (str or None): aspect ratio.

        Returns:

            matplotlib.animation.FuncAnimation: animation object.

        Extra input arguments are passed to :class:`~matplotlib.animation.FuncAnimation`
        or :class:`LineDealer` (and :meth:`~matplotlib.axes.Axes.plot`).

        Set ``xlim=None`` to let the axes as is, otherwise axis limits are adjusted to
        the data extent.
        """
        if trajs is None:
            trajs = self._parent.dataframe
        dt = self._parent.dt
        nframes = len(np.unique(np.round(trajs['t']/dt)))
        if axes is None:
            axes = fig.gca()
        #
        anim_kwargs = dict(blit=True, cache_frame_data=False, save_count=nframes, interval=1e3*dt,
                repeat=False)
        more_kwargs = dict(repeat_delay=None, fargs=None)
        more_kwargs.update(anim_kwargs)
        for kw in more_kwargs:
            try:
                arg = kwargs.pop(kw)
            except KeyError:
                pass
            else:
                if kw == 'interval':
                    if isinstance(arg, str) and arg.endswith('x'):
                        arg = float(arg[:-1]) * anim_kwargs[kw]
                    elif callable(arg):
                        arg = arg(anim_kwargs[kw])
                anim_kwargs[kw] = arg
        #
        line_kwargs = dict(linewidth=2)
        line_kwargs.update(kwargs)
        if 'lw' in kwargs:
            line_kwargs.pop('linewidth')
        #
        if isinstance(xlim, str) and xlim == 'auto':
            xlim, ylim = self._parent.bounds[['x','y']].values.T
        if xlim is not None:
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
        if aspect is not None:
            axes.set_aspect(aspect)
        #
        dt = self._parent.dt
        lines = self.animate_cls(axes, **line_kwargs)
        return animation.FuncAnimation(fig, lines.animate(trajs, dt), init_func=lines.init_func,
                frames=iter_frames(trajs, dt=dt, as_trajectory_slices=True, skip_empty_frames=False),
                **anim_kwargs)

    def plot(self, data=None, axes=None, aspect='equal', **kwargs):
        """
        Keyworded arguments are parsed based on prefixes:

        * arguments starting with '*loc_*' apply to locations drawn as points;
          `KeyError` is raised on line-styling arguments;
        * arguments starting with '*trj_*' apply to trajectories drawn as lines;
          if line style is `None` or line width is `0` or color is `None`,
          trajectories are not drawn, not even as points;
        * arguments starting with '*roi_*' apply to regions of interest drawn as
          polygons; if line style is `None` or line width is `0` or color is `None`,
          roi are not drawn, not even as points;

        All 3 types of elements are drawn with `matplotlib.axes.Axes.plot`.

        By default, trajectories are not drawn and locations are.
        If styling arguments are passed for trajectories, trajectories are drawn
        and locations are not.
        To draw both locations and trajectories as different glyphs, at least one
        styling argument should be passed for each type of elements.

        Missing elements, for example when no roi are defined, never raise an
        exception even with explicit styling arguments that would legally apply;
        these elements are simply not drawn.

        """
        loc_kwargs, trj_kwargs, roi_kwargs, sup_kwargs = {}, {}, {}, {}

        for kw in kwargs:
            if kw.startswith('loc_'):
                loc_kwargs[kw[4:]] = kwargs[kw]
            elif kw.startswith('trj_'):
                trj_kwargs[kw[4:]] = kwargs[kw]
            elif kw.startswith('roi_'):
                roi_kwargs[kw[4:]] = kwargs[kw]
            elif kw.startswith('sup_'):
                sup_kwargs[kw[4:]] = kwargs[kw]
            else:
                self._eldest_parent.logger.warning("ignoring argument: '{}'".format(kw))

        for kw in loc_kwargs:
            if kw.startswith('line'):
                raise KeyError("line styling argument ('{}') are not allowed for locations".format(kw))

        draw_loc = draw_trj = draw_roi = True
        draw_sup = bool(sup_kwargs)

        exclusions = ['color', 'marker', 'markersize']
        while draw_loc and exclusions:
            exclusion = exclusions.pop()
            draw_loc = bool(loc_kwargs.get(exclusion, True))

        exclusions = ['color', 'linestyle', 'linewidth']
        while draw_trj and exclusions:
            exclusion = exclusions.pop()
            draw_trj = bool(trj_kwargs.get(exclusion, True))

        exclusions = ['color', 'linestyle', 'linewidth']
        while draw_roi and exclusions:
            exclusion = exclusions.pop()
            draw_roi = bool(roi_kwargs.get(exclusion, True))

        exclusions = ['color', 'linestyle', 'linewidth']
        while draw_sup and exclusions:
            exclusion = exclusions.pop()
            draw_sup = bool(sup_kwargs.get(exclusion, True))

        if draw_loc and not trj_kwargs:
            draw_trj = False
        elif draw_trj and not loc_kwargs:
            draw_loc = False

        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt

        loc_glyphs = trj_glyphs = roi_glyphs = sup_glyphs = None

        if draw_loc:
            if data is None:
                data = self._parent.dataframe
            if isinstance(data, pd.DataFrame) and not data.empty:
                kwargs = dict(linestyle='none', marker='.', markersize=4, alpha=.2, color='r')
                kwargs.update(loc_kwargs)
                loc_glyphs, = axes.plot(data['x'], data['y'], **kwargs)

        if draw_trj:
            if data is None:
                data = self._parent.dataframe
            if isinstance(data, pd.DataFrame) and not data.empty:
                nan = np.full(1, np.nan, dtype=data.dtypes['x'])
                x, y = [], []
                for trj in iter_trajectories(data[list('nxyt')], asslice=False, asarray=True, order='start'):
                    x.append(trj[:,0])
                    y.append(trj[:,1])
                    x.append(nan)
                    y.append(nan)
                x = np.concatenate(x[:-1])
                y = np.concatenate(y[:-1])
                kwargs = dict(linestyle='-', marker=None, color='r')
                kwargs.update(trj_kwargs)
                trj_glyphs, = axes.plot(x, y, **kwargs)

        if draw_sup:
            x, y = [], []
            from tramway.analyzer.roi import FullRegion
            for r in self._parent.roi.as_support_regions():
                if isinstance(r, FullRegion):
                    break # or, equivalently, continue
                _min, _max = r.bounding_box
                x.append(np.r_[_min[0],_min[0],_max[0],_max[0],_min[0],np.nan])
                y.append(np.r_[_min[1],_max[1],_max[1],_min[1],_min[1],np.nan])
            if x:
                x = np.concatenate(x)
                y = np.concatenate(y)
                kwargs = dict(linestyle='-', linewidth=1, marker=None, color='b')
                kwargs.update(sup_kwargs)
                sup_glyphs, = axes.plot(x, y, **kwargs)

        if draw_roi:
            x, y = [], []
            from tramway.analyzer.roi import FullRegion, BoundingBox
            for r in self._parent.roi.as_individual_roi():
                if isinstance(r, FullRegion):
                    break # or, equivalently, continue
                assert isinstance(r, BoundingBox)
                _min, _max = r.bounding_box
                x.append(np.r_[_min[0],_min[0],_max[0],_max[0],_min[0],np.nan])
                y.append(np.r_[_min[1],_max[1],_max[1],_min[1],_min[1],np.nan])
            if x:
                x = np.concatenate(x)
                y = np.concatenate(y)
                kwargs = dict(linestyle='-', marker=None, color='g')
                kwargs.update(roi_kwargs)
                roi_glyphs, = axes.plot(x, y, **kwargs)

        if aspect is not None:
            axes.set_aspect(aspect)

        return dict(loc=loc_glyphs, trj=trj_glyphs, roi=roi_glyphs, sup=sup_glyphs)


__all__ = ['LineDealer', 'Mpl']

