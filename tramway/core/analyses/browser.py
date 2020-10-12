# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


class AnalysisBrowser(object):
    __slots__ = ('_analyses', '_path', '_subtree')
    def __init__(self, analyses=None):
        self.analyses = analyses
    @property
    def analyses(self):
        return self._analyses
    @analyses.setter
    def analyses(self, analyses):
        if isinstance(analyses, str):
            import os.path
            if os.path.isfile(analyses):
                from tramway.core.hdf5.store import load_rwa
                analyses = load_rwa(analyses, lazy=True)
        self._analyses = analyses
        self._path = None
        self._subtree = None
    @property
    def subtree(self):
        if self._subtree is None and self.analyses is not None:
            self._subtree = self.analyses
            for label in self.path():
                self._subtree = self._subtree[label]
        return self._subtree
    @subtree.setter
    def subtree(self, analyses):
        self._subtree = analyses
    @property
    def artefact(self):
        return None if self.subtree is None else self._subtree.artefact
    def path(self):
        if self._path is not None:
            yield from self._path
    def labels(self):
        if self.subtree is not None:
            yield from self.subtree.labels
    def select_child(self, label):
        self.subtree = self.subtree[label]
        if self._path is None:
            self._path = [label]
        else:
            self._path.append(label)
    def select_parent(self):
        if self._path is None:
            self._subtree = self.analyses
        else:
            child = self.analyses
            path, label = [], None
            it = self.path()
            while True:
                try:
                    child_label = next(it)
                except StopIteration:
                    break
                self._subtree = child
                if label is not None:
                    path.append(label)
                label = child_label
                child = child[label]
            self._path = path

