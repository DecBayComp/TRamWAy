# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import *
from ..artefact import *
from .abc import *
import os.path
import numpy as np


class ImagesInitializer(Initializer):
    """
    Initial value for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.images` attribute.

    *from_...* methods alters the parent attribute which specializes
    into an initialized :class:`Images` object.
    """
    __slots__ = ()
    def from_tiff_file(self, filepath):
        """
        Defines the raw single molecule imaging file.

        Loading is not performed while calling this method.
        """
        self.specialize( StandaloneTiffFile, filepath )
    def from_tiff_files(self, filepattern):
        """
        Defines the raw single molecule imaging files.

        Loading is not performed while calling this method.
        """
        self.specialize( TiffFiles, filepattern )


class ImageParameters(object):
    __slots__ = ()
    # default implementation for children classes that define `_pixel_size` and `_loc_offset`
    def __init__(self):
        self._pixel_size = self._loc_offset = None
    @property
    def pixel_size(self):
        r"""
        *float*: Pixel size in :math:`\mu m`
        """
        return self._pixel_size
    @pixel_size.setter
    def pixel_size(self, pxsize):
        self._pixel_size = pxsize
    @property
    def loc_offset(self):
        """
        *numpy.ndarray*: Offset between coordinates and the image, in pixels
        """
        return self._loc_offset
    @loc_offset.setter
    def loc_offset(self, offset):
        self._loc_offset = offset
    # access to shared parameters
    @property
    def frame_interval(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.frame_interval`
        """
        return self._eldest_parent.spt_data.frame_interval
    @frame_interval.setter
    def frame_interval(self, dt):
        self._eldest_parent.spt_data.frame_interval = dt
    @property
    def dt(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.dt`
        """
        return self.frame_interval
    @dt.setter
    def dt(self, dt):
        self.frame_interval = dt
    @property
    def logger(self):
        return self._eldest_parent.logger


class _RawImage(AnalyzerNode, ImageParameters):
    __slots__ = ('_stack', '_pixel_size', '_loc_offset')
    def __init__(self, stack, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        ImageParameters.__init__(self)
        self._stack = stack
    @property
    def stack(self):
        if callable(self._stack):
            self._stack = self._stack()
        return self._stack
    @stack.setter
    def stack(self, stack):
        self._stack = stack
    @property
    def n_frames(self):
        return self.stack.shape[0]
    @property
    def width(self):
        return self.stack.shape[2]
    @property
    def height(self):
        return self.stack.shape[1]

    def as_frames(self, index=None, return_time=False):
        """
        Generator function; iterates over the image frames and yields
        NumPy 2D arrays, or pairs of (*float*, NumPy 2D array).

        Arguments:

            index (*int*, *Set*, *Sequence* or *callable*):
                frame filter; see also :func:`~tramway.analyzer.attribute.indexer`.

            return_time (bool): return time along with image frames, as first item.

        """
        for f in indexer(index, range(self.n_frames)):
            frame = self.stack[f,:,:]
            if return_time:
                t = (f+1)*self.dt
                yield t, frame
            else:
                yield frame

    def cropping_bounds(self, bounding_box):
        lb, ub = [ b / self.pixel_size for b in bounding_box ]
        lb = np.floor(lb) * self.pixel_size
        ub = np.ceil(ub) * self.pixel_size
        return lb, ub

    def crop_frames(self, bounding_box, index=None, return_time=False):
        """
        Generator function; iterates and crops the image frames, similarly to
        :meth:`as_frames`.

        Arguments:

            bounding_box (tuple): pair of NumPy arrays (lower bound, upper bound).

            index (*int*, *Set*, *Sequence* or *callable*):
                frame filter; see also :func:`~tramway.analyzer.attribute.indexer`.

            return_time (bool): return time along with cropped image frames, as first item.

        .. note::

            Time bounds are not supported yet.

        """
        lb, ub = bounding_box
        if 2<lb.size:
            self.logger.warning('time supports are not supported yet')
            lb, ub = lb[:2], ub[:2]
            bounding_box = (lb, ub)
        lb, ub = [ b / self.pixel_size - self.loc_offset for b in bounding_box ]
        lb = np.floor(lb).astype(int)
        ub = np.ceil(ub).astype(int)
        if not np.all(lb <= ub):
            raise ValueError('image cropping failed: lower bound > upper bound')
        i_min = max(0, self.height-1-ub[1]) # range start (included)
        i_max = min(self.height-lb[1], self.height) # range stop (excluded)
        j_min = max(0, lb[0]) # range start (included)
        j_max = min(ub[0]+1, self.width) # range stop (excluded)
        if return_time:
            for t, frame in self.as_frames(index, return_time):
                yield t, frame[i_min:i_max, j_min:j_max]
        else:
            for frame in self.as_frames(index, return_time):
                yield frame[i_min:i_max, j_min:j_max]

    def to_color_movie(self, output_file=None, fourcc='VP80', colormap='gray',
            locations=None, trajectories=None, frames=None, origin=None,
            markersize=2, linecolor='many', linewidth=1,
            magnification=None, playback_rate=None, light_intensity=1.):
        """
        Generates a movie of the images with overlaid locations or trajectories.

        Arguments:

            output_file (str): path of the movie file.

            fourcc (str): 4-character code string.

            colormap (*str* or *matplotlib.colors.ListedColormap*): Matplotlib colormap;
                see also https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html.

            locations (pandas.DataFrame): particle locations with columns :const:`'x'`,
                :const:`'y'` and :const:`'t'`.

            trajectories (pandas.DataFrame): trajectories with columns :const:`'n'`,
                :const:`'x'`, :const:`'y'` and :const:`'t'`;
                note this differs from translocations in that displacements are not
                encoded along with locations and trajectory terminations are independent rows.

            frames (sequence): iterable of time (*float*, in seconds) and frame (2D pixel array)
                pairs; if `origin` defines time, `frames` can be a sequence of frames only;
                :meth:`as_frames` is called instead if undefined.

            origin (*numpy.ndarray* or *pandas.Series*): data lower bound if `frames` is defined;
                implicit columns are :const:`'x'` and :const:`'y'`;
                to define time, `origin` must be a `Series`
                with indices :const:`'x'`, :const:`'y'` and :const:`'t'`;
                `origin` is useful only for overlaying locations or trajectories.

            markersize (int): location marker size in pixels (side).

            linecolor (*str* or 3-column float array): color for trajectories;
                value :const:`None` defaults to red.

            linewidth (float): trajectory line width

            magnification (*int* or *str*): the original image pixels can be represented as
                square-patches of `magnification` video pixel side;
                if *str*:
                :const:`'1x'` = `round(pixel_size/localization_precision)`,
                :const:`'2x'` = `round(2*pixel_size/localization_precision)`;
                :const:`'2x'` is adequate for overlaid trajectories, even with the over-compressing
                :const:`'MJPG'` encoder.

            playback_rate (float): default playback rate;
                1 is real-time, 0.5 is half the normal speed.

            light_intensity (float): scale the pixel intensity values;
                works best with the 'gray' colormap.

        """
        import cv2
        # TODO: support PIL as an optional replacement for skimage
        from skimage.util import img_as_ubyte
        from skimage import draw
        from matplotlib import cm, colors

        if output_file is None:
            try:
                output_file = os.path.splitext(self.filepath)[0]+'.avi'
            except AttributeError:
                raise ValueError('output_file is not defined')

        dt = self.dt
        pxsize = self.pixel_size
        if pxsize is None:
            pxsize = 1.

        if self.loc_offset is None:
            offset = np.zeros((1,2), dtype=np.float)
        else:
            offset = np.asarray(self.loc_offset)
            if not offset.shape[1:]:
                offset = offset[np.newaxis,:]
        #x_offset, y_offset = offset[0,0], offset[0,1]

        if origin is None:
            xy0 = None
            t0 = None
        elif isinstance(origin, np.ndarray):
            xy0 = origin.reshape((1,2)).astype(np.float)
            t0 = None
        else:#if isinstance(origin, pd.Series):
            xy0 = origin[list('xy')].values[np.newaxis,:].astype(np.float)
            try:
                t0 = origin['t'].astype(np.float)
            except KeyError:
                t0 = None

        if magnification is None:
            magnification = 1
        elif isinstance(magnification, str) and magnification.endswith('x'):
            try:
                magnification = float(magnification[:-1])
            except ValueError:
                raise ValueError('cannot parse the magnification factor')
            try:
                magnification *= self.pixel_size / self._eldest_parent.localization_precision
            except (AttributeError, TypeError):
                raise ValueError('failed to adjust magnification; pixel_side or localization_precision not defined')
            magnification = int(np.round(magnification))

        if playback_rate is None:
            playback_rate = 1.

        if light_intensity == 1:
            color_scale = None
        else:
            color_scale = int(np.round(1./light_intensity))

        marker_color = line_color = np.array([[1,0,0]], dtype=np.float) # red
        if locations is not None:
            if callable(locations):
                locations = locations()
            marker_size_delta = .5*float(markersize-1)
        if trajectories is not None:
            if callable(trajectories):
                trajectories = trajectories()
            if isinstance(linecolor, str):
                if linecolor == 'many':
                    from tramway.plot.mesh import __colors__
                    line_color = np.stack([colors.to_rgb(c) for c in __colors__], axis=0)
                else:
                    line_color = colors.to_rgb(linecolor)[np.newaxis,:]
            elif isinstance(linecolor, np.ndarray):
                line_color = linecolor
            lc = line_color
            def append_n(n, ijk):
                i, j, k = ijk
                return i, j, k, np.full(k.shape, n)
            trajectories_as_dict = {}
            #trajectories_as_dict = { n: trajectories[list('xyt')][trajectories['n']==n]
            #        for n in np.unique(trajectories['n']) }
        dt_max_err = (.5*dt)**2
        def isclose(ts, t):
            _dt = ts-t
            return _dt*_dt < dt_max_err

        vid = None # will be initialized later
        vid_pxsize = pxsize / magnification
        vid_offset = offset / magnification

        if isinstance(colormap, str):
            colormap = cm.get_cmap(colormap)
        elif not callable(colormap):#isinstance(colormap, colors.ListedColormap):
            colormap = cm.viridis

        if frames:
            if not (callable(frames) or t0 is None):
                def wrap(fs, t0, dt):
                    for i, f in enumerate(fs):
                        if isinstance(f, (tuple, list)):
                            t, f = f
                            if i == 0 and t != t0:
                                raise ValueError('start time does not match with first frame time: {} != {}'.format(t0, t))
                        else:
                            t = t0 + i * dt
                        yield t, f
                frames = wrap(frames, t0, dt)
        else:
            frames = self.as_frames

        t_prev = -1
        for t, frame in frames(return_time=True) if callable(frames) else frames:
            if t <= t_prev:
                raise ValueError('time does not strictly increase')
            t_prev = t

            if vid is None:
                # complete the initialization
                height, width = frame.shape
                ii_max = height * magnification - 1
                jj_max = width * magnification - 1

                vid = cv2.VideoWriter(os.path.expanduser(output_file),
                        cv2.VideoWriter_fourcc(*fourcc),
                        playback_rate/dt, (width*magnification, height*magnification), True)

            if color_scale:
                frame = frame // color_scale

            frame = colormap(frame)[:,:,:3]

            if magnification:
                frame = np.repeat(np.repeat(frame, magnification, axis=0), magnification, axis=1)

            if locations is not None:
                xy = locations[ isclose(locations['t'], t) ]
                if xy0 is None:
                    xy_f = xy[list('xy')].values / vid_pxsize - vid_offset
                else:
                    xy_f = (xy[list('xy')].values - xy0) / vid_pxsize
                for j,i in xy_f:
                    i = np.array([np.floor(i-marker_size_delta), np.ceil(i+marker_size_delta)], dtype=np.int)
                    j = np.array([np.floor(j-marker_size_delta), np.ceil(j+marker_size_delta)], dtype=np.int)
                    if np.any(i<0) or np.any(j<0):
                        continue
                    i,j = np.meshgrid(np.arange(i[0],i[1]), np.arange(j[0],j[1]), indexing='ij')
                    frame[ii_max-np.ravel(i),np.ravel(j),:] = marker_color

            if trajectories is not None:
                # get the ids of the active trajectories for frame f
                active_trajectories = trajectories['n'][ isclose(trajectories['t'], t) ]
                assert all_unique(active_trajectories)

                # extract the corresponding coordinate series truncated at time t
                trajs_f = []
                for n in active_trajectories:
                    try:
                        xyt_n = trajectories_as_dict[n]
                    except KeyError:
                        trajectories_as_dict[n] = xyt_n = trajectories[trajectories['n']==n]
                    xy_n = xyt_n[list('xy')][xyt_n['t']<t+.5*dt]
                    if len(xy_n)<2:
                        continue
                    if xy0 is None:
                        traj = xy_n.values / vid_pxsize - vid_offset
                    else:
                        traj = (xy_n.values - xy0) / vid_pxsize
                    if np.any(traj < 0): # this may occur with negative offsets
                        continue
                    traj = np.round(traj).astype(np.uint32)
                    trajs_f.append((n, traj))

                # overlay the truncated trajectories
                if trajs_f:
                    # TODO: draw a proper polyline with unique ii and jj and max(kk)
                    jj, ii, kk, nn = zip(*[ append_n(n,
                            draw.line_aa(traj[p,0], traj[p,1], traj[p+1,0], traj[p+1,1])
                        ) for n, traj in trajs_f for p in range(traj.shape[0]-1) ])
                    ii = np.concatenate(ii)
                    jj = np.concatenate(jj)
                    ok = (0<=ii)&(ii<=ii_max)&(0<=jj)&(jj<=jj_max)
                    ii, jj = ii[ok], jj[ok]
                    kk = np.concatenate(kk)[ok][:,np.newaxis]
                    if 1<len(line_color):
                        nn = np.concatenate(nn)[ok]
                        lc = line_color[nn % len(line_color)]
                    frame[ii_max-ii,jj,:] = (1.-kk) * frame[ii_max-ii,jj,:] + kk * lc

            vid.write(img_as_ubyte(frame)[:,:,::-1])
        vid.release()

    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.images.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)


class RawImage(_RawImage):
    __slots__ = ()

Image.register(RawImage)

class _ImageFile(_RawImage):
    __slots__ = ('_filepath',)
    def __init__(self, filepath, **kwargs):
        self._filepath = filepath
        RawImage.__init__(self, self.read, **kwargs)
    @property
    def filepath(self):
        return self._filepath
    @filepath.setter
    def filepath(self, fp):
        self._filepath = fp
    def read(self):
        from skimage import io
        return io.imread(os.path.expanduser(self.filepath))

class ImageFile(_ImageFile):
    __slots__ = ()

Image.register(ImageFile)

class _TiffFile(_ImageFile):
    __slots__ = ()

class TiffFile(ImageFile):
    __slots__ = ()

class StandaloneImage(object):
    __slots__ = ()
    def __len__(self):
        return 1
    def __iter__(self):
        yield self

class ImageIterator(AnalyzerNode, ImageParameters):
    """ Partial :class:`Images` implementation for multiple SPT data items.

    Children classes must implement the :meth:`__iter__` method."""
    __slots__ = ()
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        #ImageParameters.__init__(self)
    @property
    def pixel_size(self):
        it = iter(self)
        pxsize = next(it).pixel_size
        while True:
            try:
                _pxsize = next(it).pixel_size
            except StopIteration:
                break
            else:
                if _pxsize != pxsize:
                    raise AttributeError('not all the images share the same pixel size')
        return pxsize
    @pixel_size.setter
    def pixel_size(self, pxsize):
        for f in self:
            f.pixel_size = pxsize
    @property
    def loc_offset(self):
        it = iter(self)
        offset = next(it).loc_offset
        while True:
            try:
                _offset = next(it).loc_offset
            except StopIteration:
                break
            else:
                if not np.all(_offset == offset):
                    raise AttributeError('not all the images share the same localization offset')
        return offset
    @loc_offset.setter
    def loc_offset(self, offset):
        for f in self:
            f.loc_offset = offset

class StandaloneImageFile(_ImageFile, StandaloneImage):
    __slots__ = ()

Images.register(StandaloneImageFile)

class StandaloneTiffFile(_TiffFile, StandaloneImage):
    __slots__ = ()

Images.register(StandaloneTiffFile)


class RawImages(ImageIterator):
    __slots__ = ('_images',)
    def __init__(self, stacks, **kwargs):
        ImageIterator.__init__(self, **kwargs)
        self.images = stacks
    @property
    def images(self):
        return self._images
    @images.setter
    def images(self, stacks):
        self._images = tuple([ self._bear_child( RawImage, stack ) for stack in stacks ])
    @property
    def reified(self):
        return True
    def __len__(self):
        return len(self.images)
    def __iter__(self):
        yield from self.images

#Images.register(RawImages)

class ImageFiles(ImageIterator):
    __slots__ = ('_files', '_filepattern')
    def __init__(self, filepattern, **kwargs):
        ImageIterator.__init__(self, **kwargs)
        self._files = []
        if isinstance(filepattern, str):
            self._filepattern = os.path.expanduser(filepattern)
        else:
            self._filepattern = [os.path.expanduser(pattern) for pattern in filepattern]
    @property
    def filepattern(self):
        return self._filepattern
    @filepattern.setter
    def filepattern(self, fp):
        if self._files:
            if fp != self._filepattern:
                raise AttributeError('the files have already been listed; cannot set the file pattern anymore')
        else:
            self._filepattern = fp
    @property
    def files(self):
        if not self._files:
            self.list_files()
        return self._files
    @property
    def filepaths(self):
        return [ f.filepath for f in self.files ]
    @property
    def partially_reified(self):
        return self._files and any([ f.reified for f in self._files ])
    @property
    def fully_reified(self):
        return self._files and all([ f.reified for f in self._files ])
    @property
    def reified(self):
        return self.fully_reified
    def __len__(self):
        return len(self.files)
    def __iter__(self):
        yield from self.files
    def list_files(self, _element_cls=ImageFile):
        from glob import glob
        self._files = glob(self.filepattern)
        if not self._files:
            raise ValueError("no files found")
        self._files = [ self._bear_child( _element_cls, filepath ) for filepath in self._files ]

Images.register(ImageFiles)

class TiffFiles(ImageFiles):
    __slots__ = ()
    def list_files(self):
        ImageFiles.list_files(self, TiffFile)

def all_unique(values):
    return np.unique(values).size == values.size


__all__ = ['Images', 'Image', 'ImagesInitializer', 'ImageParameters', '_RawImage',
        'RawImage', 'ImageFile', 'TiffFile', 'StandaloneImageFile', 'StandaloneTiffFile',
        'RawImages', 'ImageFiles', 'TiffFiles']

