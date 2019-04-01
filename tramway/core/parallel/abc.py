# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from abc import *


class Workspace(metaclass=ABCMeta):
    """ Parameter singleton.
    """

    @abstractmethod
    def update(self, step):
        """ Update the workspace with a job step.

        Arguments:

            step (AbstractJobStep): completed job step.

        """
        raise NotImplementedError('abstract method')
    @abstractmethod
    def __len__(self):
        """ `int`: Number of resource items."""
        raise NotImplementedError('abstract method')


class JobStep(metaclass=ABCMeta):

    @abstractmethod
    def get_workspace(self):
        raise NotImplementedError('abstract method')
    @abstractmethod
    def set_workspace(self, ws):
        raise NotImplementedError('abstract method')
    @abstractmethod
    def unset_workspace(self):
        raise NotImplementedError('abstract method')
    @property
    @abstractmethod
    def workspace_set(self):
        """ bool: is workspace set?"""
        raise NotImplementedError('abstract method')
    @property
    @abstractmethod
    def step_id(self):
        """ int: job step ID.

        A job step ID is unique per sub-jobs
        """
        raise NotImplementedError('abstract ro property')
    @property
    @abstractmethod
    def resource_id(self):
        """ array-like: Indices of the required resources.

        Resource ids should be interpreted as indices in a contiguous array.
        """
        raise NotImplementedError('abstract property')


__all__ = [ 'Workspace', 'JobStep' ]

