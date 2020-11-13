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
from tramway.core.xyt import iter_frames
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
    def plotter(self):
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
        lines = self.plotter(axes, **line_kwargs)
        return animation.FuncAnimation(fig, lines.animate(trajs, dt), init_func=lines.init_func,
                frames=iter_frames(trajs, dt=dt, as_trajectory_slices=True, skip_empty_frames=False),
                **anim_kwargs)


__all__ = ['LineDealer', 'Mpl']

