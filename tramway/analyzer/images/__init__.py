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
    initial value for the `RWAnalyzer.images` attribute.

    `from_...` methods alters the parent attribute which specializes
    into an initialized :class:`.abc.Images` object.
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
        return self._pixel_size
    @pixel_size.setter
    def pixel_size(self, pxsize):
        self._pixel_size = pxsize
    @property
    def loc_offset(self):
        return self._loc_offset
    @loc_offset.setter
    def loc_offset(self, offset):
        self._loc_offset = offset
    # access to shared parameters
    @property
    def dt(self):
        return self._eldest_parent.spt_data.dt
    @dt.setter
    def dt(self, dt):
        self._eldest_parent.spt_data.dt = dt
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

    def as_frames(self, return_time=False):
        for f in range(self.n_frames):
            frame = self.stack[f,:,:]
            if return_time:
                t = (f+1)*self.dt
                yield t, frame
            else:
                yield frame

    def to_color_movie(self, output_file=None, opencv_fourcc='MJPG', colormap='Greys',
            locations=None, trajectories=None, frames=None, linecolor='many'):
        """
        Generates a movie of the images with overlaid locations or trajectories.

        Arguments:

            output_file (str): path of the movie file.

            opencv_fourcc (str): 4-character code string.

            colormap (str or matplotlib.colors.ListedColormap): Matplotlib colormap;
                see also https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html.

            locations (pandas.DataFrame): particle locations with columns 'x', 'y' and 't'.

            trajectories (pandas.DataFrame): trajectories with columns 'n', 'x', 'y' and 't';
                note this differs from translocations in that displacements are not
                encoded along with locations and trajectory terminations are independent rows.

            frames (sequence): iterable of time (*float*, in seconds) and frame (2D pixel array)
                pairs; :meth:`as_frames` is called instead if undefined.

            linecolor (str or 3-column float array): color for trajectories.

        """
        import cv2
        from skimage.util import img_as_ubyte
        from skimage import draw
        from matplotlib import cm, colors

        if output_file is None:
            try:
                output_file = os.path.splitext(self.filepath)[0]+'.avi'
            except AttributeError:
                raise ValueError('output_file is not defined')

        dt = self.dt
        width, height = self.width, self.height
        pxsize = self.pixel_size
        if pxsize is None:
            pxsize = 1.
        if self.loc_offset is None:
            offset = np.zeros((1,2), dtype=np.float)
        else:
            offset = np.asarray(self.loc_offset)
            if not offset.shape[1:]:
                offset = offset[np.newaxis,:]
        x_offset, y_offset = offset[0,0], offset[0,1]

        marker_color = line_color = np.array([[1,0,0]], dtype=np.float) # red
        if trajectories is not None:
            if isinstance(linecolor, str):
                if linecolor == 'many':
                    from tramway.plot.mesh import __colors__
                    line_color = np.stack([colors.to_rgb(c) for c in __colors__], axis=0)
                else:
                    line_color = colors.to_rgb(linecolor)[np.newaxis,:]
            elif isinstance(linecolor, np.ndarray):
                line_color = linecolor
            lc = line_color
            def zip_n(n, ijk):
                i, j, k = ijk
                return i, j, k, np.full(k.shape, n)

        vid = cv2.VideoWriter(os.path.expanduser(output_file),
                cv2.VideoWriter_fourcc(*opencv_fourcc),
                1./dt, (width, height), True)

        if isinstance(colormap, str):
            colormap = cm.get_cmap(colormap)
        elif not callable(colormap):#isinstance(colormap, colors.ListedColormap):
            colormap = cm.viridis

        for t, frame in self.as_frames(return_time=True) if frames is None else frames:
            frame = colormap(frame)[:,:,:3]

            if locations is not None:
                xy = locations[ (locations['t']-t)**2 < (.5*dt)**2 ]
                xy_f = (xy[list('xy')].values + offset) / pxsize
                for j,i in xy_f:
                    i = np.array([np.floor(i), np.ceil(i)], dtype=np.int)
                    j = np.array([np.floor(j), np.ceil(j)], dtype=np.int)
                    i,j = np.meshgrid(i[0<=i], j[0<=j], indexing='ij')
                    frame[height-1-np.ravel(i),np.ravel(j),:] = marker_color

            if trajectories is not None:
                # get the locations for frame f
                xy = trajectories[ (trajectories['t']-t)**2 < (.5*dt)**2 ]

                # ...and the corresponding truncated trajectories
                trajs_f, n_f = [], []
                for n in xy['n']:
                    xy_n = trajectories[trajectories['n']==n]
                    xy_n = xy_n[xy_n['t']<t+.5*dt]
                    if 1<len(xy_n):
                        traj = (xy_n[list('xy')].values + offset) / pxsize
                        traj = np.round(traj).astype(np.int64)
                        if np.any(traj < 0):
                            continue
                        trajs_f.append(traj)
                        n_f.append(n)

                # overlay the trajectories
                if trajs_f:
                    jj, ii, kk, nn = zip(*[
                            zip_n(n, draw.line_aa(traj[pt,0], traj[pt,1], traj[pt+1,0], traj[pt+1,1]))
                            for pt in range(traj.shape[0]-1) for n, traj in zip(n_f, trajs_f) ])
                    ii = np.concatenate(ii)
                    jj = np.concatenate(jj)
                    kk = np.concatenate(kk)[:,np.newaxis]
                    if 1<len(line_color):
                        nn = np.concatenate(nn)
                        lc = line_color[nn % len(line_color)]
                    frame[height-1-ii,jj,:] = (1.-kk) * frame[height-1-ii,jj,:] + kk * lc

            vid.write(img_as_ubyte(frame)[:,:,::-1])
        vid.release()


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
    """ partial implementation for multiple SPT data items.

    Children classes must implement the `__iter__` method."""
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
        self._filepattern = filepattern
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

def TiffFiles(ImageFiles):
    __slots__ = ()
    def list_files(self):
        ImageFiles.list_files(self, TiffFile)

__all__ = ['Image', 'Images', 'ImagesInitializer', 'ImageParameters', 'RawImage', 'ImageFile', 'TiffFile', 'StandaloneImageFile', 'StandaloneTiffFile', 'RawImages', 'ImageFiles', 'TiffFiles']

