# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute.abc import Attribute, abstractmethod

class Environment(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.env` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    @property
    @abstractmethod
    def script(self):
        """
        *str*: Path to the local script; usually a filename
        """
        pass
    @abstractmethod
    def setup(self):
        pass
    @abstractmethod
    def dispatch(self, **kwargs):
        """
        Input arguments must be keyworded.

        Arguments:

            stage_index (int): index of the pipeline stage which material has to be dispatched.

            source (str): source name of the spt dataset to be dispatched.

        """
        pass
    @abstractmethod
    def make_job(self, stage_index=None, source=None, region_label=None, segment_index=None):
        pass
    @abstractmethod
    def submit_jobs(self):
        pass
    @abstractmethod
    def wait_for_job_completion(self):
        pass
    @abstractmethod
    def collect_results(self):
        pass

