# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from ..core import *
from ..core.hdf5 import *
import os.path
import six
import pandas as pd


class UseCaseWarning(UserWarning):
    pass
class IgnoredInputWarning(UserWarning):
    pass


class Helper(object):

    def __init__(self):
        """
        """
        self.input_file = None
        self.analyses = None
        #self.plugins = None
        self.module = None
        self.setup = None
        self.verbose = False
        self._label_is_output = None

    def are_multiple_files(self, files):
        return isinstance(files, (tuple, list, frozenset, set))

    def prepare_data(self, data, labels=None, types=None, verbose=None, **kwargs):
        """
        Load the data if the input argument is a file (can be rwa or txt).
        If the input file is a trajectory file, trailing keyworded arguments are passed
        to :func:`load_xyt`.

        Isolate the requested artefacts if the input is an analysis tree or an rwa file
        and `labels` or `types` is defined.
        """
        if verbose is None:
            verbose = self.verbose
        if labels is None:
            labels = self.input_label
        if isinstance(data, pd.DataFrame):
            self.analyses = Analyses(data)
        elif isinstance(data, Analyses):
            self.analyses = data
        elif (self.are_multiple_files(data) and data and all(os.path.exists(_d) for _d in data)) or \
                (isinstance(data, six.string_types) and os.path.isdir(data)):
            data, self.input_file = load_xyt(data, return_paths=True, **kwargs)
            self.analyses = Analyses(data)
        elif isinstance(data, six.string_types) and os.path.isfile(data):
            self.input_file = [data]
            try:
                self.analyses = data = load_rwa(data, lazy=True, verbose=verbose)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                if labels is None and types is None:
                    try:
                        data = load_xyt(data, **kwargs)
                    except (SystemExit, KeyboardInterrupt):
                        raise
                    except:
                        raise e
                    else:
                        if kwargs.get('return_paths', False):
                            data, = data
                        self.analyses = Analyses(data)
                else:
                    raise
        if isinstance(data, Analyses):
            if not (labels is None and types is None):
                data = find_artefacts(data, types, labels)
                if self.input_label is None:
                    self.input_label = self.find(data[-1])
        return data

    def find(self, artefact, labels=None, return_subtree=False):
        if labels is None:
            labels = self.input_label
        found = False
        _labels = []
        _label = None
        analysis = self.analyses
        if analysis._data is artefact:
            found = True
        elif self.label_is_absolute(labels):
            for _label in labels:
                _labels.append(_label)
                analysis = analysis[_label]
                if analysis._data is artefact:
                    found = True
                    break
        else:
            assert labels is None
            available_labels = list(analysis.labels)
            # TODO: recursive search
            # for now, `labels` is required
            while available_labels and not available_labels[1:]:
                _label = available_labels[0]
                _labels.append(_label)
                analysis = analysis[_label]
                if analysis._data is artefact:
                    found = True
                    break
                available_labels = list(analysis.labels)
        if not found:
            #raise RuntimeError('artefact not found')
            _labels = []
            analysis = None
        if return_subtree:
            return _labels, analysis
        else:
            return _labels

    def output_file(self, output_file=None, suffix=None, extension=None):
        basename = None
        if output_file is None:
            if self.input_file:
                if self.input_file[1:]:
                    warnings.warn('formatting the output filename on basis of the first input filename', RuntimeWarning)
                if suffix is None and extension is None:
                    return self.input_file[0]
                else:
                    basename, = os.path.splitext(self.input_file[0])
        else:
            if suffix is None and extension is None:
                return output_file
            else:
                basename, = os.path.splitext(output_file)
        output_file = basename
        if basename:
            if suffix:
                if suffix[0].isalpha():
                    suffix = '_'+suffix
                output_file += suffix
            if extension:
                if extension[0] != '.':
                    extension = '.'+extension
                output_file += extension
        return output_file

    def label_is_absolute(self, label):
        return isinstance(label, (tuple, list))

    def labels(self, label=None, input_label=None, output_label=None, inplace=False, comment=None):
        # store comment
        self.comment = comment
        # define whether `label` is input or output label
        exclusive_labels_error = ValueError("multiple different values in exclusive arguments 'label', 'input_label' and 'output_label'")
        if label is not None:
            if self._label_is_output is True:
                if output_label is None:
                    output_label = label
                else:
                    raise ValueError("'label' and 'output_label' are both defined and are different")
            elif self._label_is_output is False:
                if input_label is None:
                    input_label = label
                else:
                    raise ValueError("'label' and 'input_label' are both defined and are different")
            else:
                if output_label is None:
                    if input_label is None:
                        # arbitrarily choose
                        input_label = label
                    else:
                        output_label = label
                else:
                    if input_label is None:
                        input_label = label
                    else:
                        if label == input_label or label == output_label:
                            # safely ignore `label`
                            pass
                        else:
                            raise exclusive_labels_error
            label = None
        # input labels necessarily form a path
        if input_label is None:
            self.explicit_input_label = False
        else:
            self.explicit_input_label = True
            if not self.label_is_absolute(input_label):
                input_label = [input_label]
        # if inplace, all the defined labels should equal
        if inplace:
            if output_label is None:
                label = input_label
            else:
                if self.label_is_absolute(output_label):
                    label = output_label
                    if not (input_label is None or output_label == input_label):
                        raise exclusive_labels_error
                else:
                    if input_label is None:
                        label = [output_label]
                    elif output_label == input_label[-1]:
                        label = input_label
                    else:
                        raise exclusive_labels_error
            output_label = input_label = label
        # return
        self.input_label = input_label
        self.output_label = output_label
        return input_label, output_label

    @property
    def inplace(self):
        return self.input_label \
            and self.label_is_absolute(self.output_label) \
            and self.input_label == self.output_label

    def plugin(self, name, plugins=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if plugins is not None:
            self.plugins = plugins
        try:
            if self.plugins is None:    raise AttributeError
        except AttributeError:
            raise ValueError('no plugins defined')
        if verbose:
            self.plugins.verbose = True
        try:
            self.setup, self.module = self.plugins[name]
        except KeyError:
            raise KeyError('no such plugin: {}'.format(name))
        else:
            self.name = name
        return self.module, self.setup

    def insert_analysis(self, analysis_or_artefact, label=None, comment=None, anchor=None):
        labels = None
        if isinstance(analysis_or_artefact, Analyses):
            artefact = analysis_or_artefact.data
            analysis = analysis_or_artefact
        else:
            artefact = analysis_or_artefact
            analysis = Analyses(analysis_or_artefact)
        if self.analyses is None:
            raise RuntimeError('no root analysis')
        if anchor:
            labels, input_analysis = self.find(anchor, return_subtree=True,
                labels=label if self.label_is_absolute(label) else None)
            if input_analysis is None:
                raise ValueError('anchor not found')
            if label is None:
                label = self.output_label
            if self.label_is_absolute(label):
                if labels == label[:-1]:
                    label = label[-1]
                else:
                    raise ValueError('output labels do not match with anchor artefact')
        else:
            input_analysis = self.analyses
            if label is None:
                label = self.output_label
            if self.label_is_absolute(label):
                # output_label is absolute path; forget input_label
                labels = list(label)
                label = labels.pop() # terminal label
                for _label in labels:
                    input_analysis = input_analysis.instances[_label]
            else:
                labels = self.input_label
                if labels is not None:
                    # input_label is absolute path, even if it is not formatted correspondingly
                    if self.label_is_absolute(labels):
                        labels = list(labels)
                    else:
                        self.input_label = labels = [ labels ]
                    #
                    for _label in labels:
                        input_analysis = input_analysis.instances[_label]
        if label is not None and label in input_analysis.labels and self.inplace:
            # `input_analysis[label]` already exists
            if comment:
                input_analysis.comments[label] = comment
            input_analysis = input_analysis.instances[label]
            input_analysis.artefact = artefact
        else:
            label = input_analysis.autoindex(label)
            input_analysis.add(analysis, label=label, comment=comment)
        # self.analyses is implicitly updated
        if labels is None:
            return [label]
        else:
            labels.append(label)
            return labels

    def save_analyses(self, output_file=None, verbose=None, force=None, **kwargs):
        if verbose is None:
            verbose = self.verbose
        output_file = self.output_file(output_file)
        if output_file is None:
            return
        if force is None and bool(self.input_file):
            if self.are_multiple_files(self.input_file):
                input_files = list(self.input_file)
            else:
                input_files = [self.input_file]
            force = not input_files[1:] and input_files[0] == output_file
        save_rwa(output_file, self.analyses, verbose, force, **kwargs)

