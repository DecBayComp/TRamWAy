# -*- coding: utf-8 -*-

# Copyright © 2017-2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from . import abc
import itertools
import copy
import warnings


class AnalysesView(dict):
    __slots__ = ('__analyses__',)
    def __init__(self, analyses):
        self.__analyses__ = analyses
    def __nonzero__(self):
        #return self.__analyses__._instances.__nonzero__()
        return bool(self.__analyses__._instances) # Py2
    def __len__(self):
        return len(self.__analyses__._instances)
    def __missing__(self, label):
        raise KeyError('no such analysis instance: {}'.format(label))
    def __iter__(self):
        return self.__analyses__._instances.__iter__()
    def __contains__(self, label):
        return self.__analyses__._instances.__contains__(label)
    def keys(self):
        return self.__analyses__._instances.keys()

class InstancesView(AnalysesView):
    __slots__ = ()
    def __str__(self):
        return self.__analyses__._instances.__str__()
    def __getitem__(self, label):
        return self.__analyses__._instances[label]
    def __setitem__(self, label, analysis):
        if not isinstance(analysis, abc.Analyses):
            analysis = type(self.__analyses__)(analysis)
        self.__analyses__._instances[label] = analysis
    def __delitem__(self, label):
        self.__analyses__._instances.__delitem__(label)
        try:
            self.__analyses__._comments.__delitem__(label)
        except KeyError:
            pass
    def values(self):
        return self.__analyses__._instances.values()
    def items(self):
        return self.__analyses__._instances.items()
    def get(self, label, default=None):
        return self.__analyses__._instances.get(label, default)
    def pop(self, label, default=None):
        analysis = self.get(label, default)
        self.__delitem__(label)
        return analysis

class CommentsView(AnalysesView):
    __slots__ = ()
    def __str__(self):
        return self.__analyses__._comments.__str__()
    def __getitem__(self, label):
        try:
            return self.__analyses__._comments[label]
        except KeyError:
            if label in self.__analyses__._instances:
                return None
            else:
                self.__missing__(label)
    def __setitem__(self, label, comment):
        if label in self.__analyses__._instances:
            if comment:
                self.__analyses__._comments[label] = comment
            else:
                self.__delitem__(label)
        else:
            self.__missing__(label)
    def __delitem__(self, label):
        self.__analyses__._comments.__delitem__(label)
    def values(self):
        return self.__analyses__._comments.values()
    def items(self):
        return self.__analyses__._comments.items()


class Analyses(object):
    """
    Analysis tree - Generic container with labels, comments and other metadata to structure
    the analyses that apply to the same data.

    An :class:`Analyses` object is a node of a tree.
    In attribute `data` (or equivalently `artefact`)
    it contains the input data for the children analyses,
    and these children analyses can be accessed as subtrees using a dict-like interface.

    Labels of the children analyses can be listed with property `labels`.
    A label is a key in the dict-like interface.

    Comments associated to children analyses are also addressable with labels.

    Metadata are attached to each node, including the top node.

    Setting the `data` attribute unsets the other attributes.

    Example:

    .. code-block:: python

        ## let `my_input_data` and `my_output_data` be dataframes:
        #my_output_data = my_analysis(my_input_data)

        ## build the tree
        tree = Analyses(my_input_data) # root node

        tree.add(my_output_data, label='my analysis', comment='description of my analysis')
        # or equivalently (order matters):
        tree['my analysis'] = my_output_data
        tree.comments['my analysis'] = 'description of my analysis'

        ## print
        print(tree)
        #<class 'pandas.core.frame.DataFrame'>
        #        'my analysis' <class 'pandas.core.frame.DataFrame'>:    "description of my analysis"

        assert tree.data is my_input_data
        # note that `my_output_data` has been automatically wrapped into an `Analysis` object:
        assert isinstance(tree['my analysis'], Analyses)
        assert tree['my analysis'].data is my_output_data

        print(tree.labels) # or print(tree.keys())
        #dict_keys(['my analysis'])

        print(tree.comments['my analysis'])
        #description of my analysis


    Attributes:

        data/artefact (any): input data to the children instances.

        instances (dict): analyses on the data; keys are natural integers or string labels.

        comments (dict): comments associated to the analyses; keys are a subset of the keys
            in `instances`.

        metadata (dict): additional metadata associated to the input data; keys are attributes
            and are not related to children instances.

    """
    __slots__ = ('_data', '_instances', '_comments', '_metadata')

    def __init__(self, data=None, metadata=None):
        self._data = data
        self._instances = {}
        self._comments = {}
        self._metadata = {} if metadata is None else metadata

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._instances = {}
        self._comments = {}
        self._metadata = {}

    @property
    def artefact(self):
        return self.data

    @artefact.setter
    def artefact(self, a):
        self.data = a

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @metadata.setter
    def metadata(self, d):
        self._metadata = {} if d is None else d

    @property
    def instances(self):
        return InstancesView(self)

    @property
    def comments(self):
        return CommentsView(self)

    @property
    def labels(self):
        return self.instances.keys()

    def keys(self):
        return self.instances.keys()

    def autoindex(self, pattern=None):
        r"""
        Determine the lowest available natural integer for use as key in `instances` and `comments`.

        If `pattern` is an integer, `autoindex` returns the pattern unchanged.

        Arguments:

            pattern (str): label with a *'\*'* to be replaced by a natural integer.

        Returns:

            int or str: index or label.
        """
        if valid_label(pattern):
            try:
                pattern = int(pattern) # this also checks type
            except ValueError:
                pass
            if not isinstance(pattern, int) and '*' in pattern:
                f = lambda i: pattern.replace('*', str(i))
            else:
                return pattern
        else:
            f = lambda i: i
        i = 0
        if self.instances:
            while f(i) in self.instances:
                i += 1
        return f(i)

    def add(self, analysis, label=None, comment=None, raw=False):
        """
        Add an analysis.

        Adding an analysis at an existing label overwrites the existing analysis
        instance and deletes the associated comment if any.

        Arguments:

            analysis (any): analysis instance.

            label (any): key for the analysis; calls :meth:`autoindex` if undefined.

            comment (str): associated comment.

            raw (bool):
                if `analysis` is not an :class:`~tramway.core.analyses.base.Analyses`,
                it is wrapped into such a container object;
                set `raw` to ``True`` to prevent wrapping.

        """
        label = self.autoindex(label)
        if not (raw or isinstance(analysis, abc.Analyses)):
            analysis = type(self)(analysis)
        self.instances[label] = analysis
        if comment:
            self.comments[label] = comment
        else:
            try:
                del self.comments[label]
            except KeyError:
                pass

    def __nonzero__(self):
        return self.instances.__nonzero__()
    def __len__(self):
        return self.instances.__len__()
    def __missing__(self, label):
        self.instances.__missing__(label)
    def __iter__(self):
        return self.instances.__iter__()
    def __contains__(self, label):
        return self.instances.__contains__(label)
    def __getitem__(self, label):
        return self.instances.__getitem__(label)
    def __setitem__(self, label, analysis):
        self.instances.__setitem__(label, analysis)
    def __delitem__(self, label):
        self.instances.__delitem__(label)

abc.Analyses.register(Analyses)


valid_label = lambda label: label or (isinstance(label, int) and label == 0)


def map_analyses(fun, analyses, label=False, comment=False, metadata=False, depth=False,
        allow_tuples=False):
    with_label = label or valid_label(label)
    with_comment, with_metadata, with_depth = comment, metadata, depth
    def _fun(x, **kwargs):
        y = fun(x, **kwargs)
        if not allow_tuples and isinstance(y, tuple):
            raise ValueError('type conflict: function returned a tuple')
        return y
    def _map(analyses, label=None, comment=None, depth=0):
        kwargs = {}
        if with_label:
            kwargs['label'] = label
        if with_comment:
            kwargs['comment'] = comment
        if with_metadata:
            kwargs['metadata'] = analyses.metadata
        if with_depth:
            kwargs['depth'] = depth
        node = _fun(analyses._data, **kwargs)
        if analyses.instances:
            depth += 1
            tree = []
            for label in analyses.instances:
                child = analyses.instances[label]
                comment = analyses.comments[label]
                if isinstance(child, abc.Analyses):
                    tree.append(_map(child, label, comment, depth))
                else:
                    if with_label:
                        kwargs['label'] = label
                    if with_comment:
                        kwargs['comment'] = comment
                    if with_depth:
                        kwargs['depth'] = depth
                    tree.append(_fun(child, **kwargs))
            return node, tuple(tree)
        else:
            return node
    return _map(analyses)


def extract_analysis(analyses, labels):
    """
    Extract an analysis from a hierarchy of analyses.

    The elements of an :class:`~tramway.core.analyses.base.Analyses` instance can be other
    :class:`~tramway.core.analyses.base.Analyses` objects.
    As such, analyses are structured in a tree that exhibits as many logically-consecutive
    layers as there are processing steps.

    Arguments:

        analyses (tramway.core.analyses.base.Analyses):
            hierarchy of analyses, with `instances` possibly containing
            other :class:`~tramway.core.analyses.base.Analyses` instances.

        labels (int, str or sequence of int and str):
            analyses label(s); the first label addresses the first layer of
            analyses instances, the second label addresses the second layer of
            analyses and so on.

    Returns:

        tramway.core.analyses.base.Analyses: copy of the analyses along the path defined by `labels`.
    """
    analysis = type(analyses)(analyses._data, analyses.metadata)
    if labels:
        if not isinstance(labels, (tuple, list)):
            labels = [labels]
        label = labels[0]
        analysis.instances[label] = extract_analyses(analyses.instances[label], labels[1:])
        try:
            analysis.comments[label] = analyses.comments[label]
        except KeyError:
            pass
    return analysis


def _append(s, ls):
    """for internal use in :fun:`label_paths`."""
    if ls:
        ss = []
        for ok, _ls in ls:
            if isinstance(ok, bool):
                l, _ls = _ls, None
            else:
                ok, l = ok
            _s = list(s) # copy
            _s.append(l)
            if ok:
                ss.append([tuple(_s)])
            if _ls:
                ss.append(_append(_s, _ls))
        return itertools.chain(*ss)
    else:
        return []

def label_paths(analyses, filter):
    """
    Find label paths for analyses matching a criterion.

    Arguments:

        analyses (tramway.core.analyses.base.Analyses):
            hierarchy of analyses, with `instances` possibly containing
            other :class:`~tramway.core.analyses.base.Analyses` instances.

        filter (type or callable):
            criterion over analysis data.

    Returns:

        list of tuples:
            list of label paths to matching analyses.

    """
    if isinstance(filter, type):
        _type = filter
        _map = lambda node, label: (isinstance(node, _type), label)
    elif callable(filter):
        _map = lambda node, label: (filter(node), label)
    else:
        raise TypeError('`filter` is neither a type nor a callable')
    _, labels = map_analyses(_map, analyses, label=True, allow_tuples=True)
    return list(_append([], labels))


def find_artefacts(analyses, filters, labels=None, quantifiers=None, fullnode=False,
        return_subtree=False):
    """
    Find related artefacts.

    Filters are applied to find data elements (artefacts) along a single path specified by `labels`.

    Arguments:

        analyses (tramway.core.analyses.base.Analyses): hierarchy of analyses.

        filters (type or callable or tuple or list): list of criteria, a criterion being
            a boolean function or a type.

        labels (list): label path.

        quantifiers (str or tuple or list): list of quantifers, a quantifier for now being
            either '*first*', '*last*' or '*all*'; a quantifier should be defined for each
            filter; default is '*last*' (admits value ``None``).

        return_subtree (bool): return as extra output argument the analysis subtree corresponding
            to the deepest matching artefact.

    Returns:

        tuple: matching data elements/artefacts, and optionally analysis subtree.

    Examples:

    .. code-block:: python

        cells, maps = find_artefacts(analyses, (CellStats, Maps))

        maps, maps_subtree = find_artefacts(analyses, Maps, return_subtree=True)

    """
    # filters
    if not isinstance(filters, (tuple, list)):
        filters = (filters,)
    # quantifiers
    quantifier = 'last'
    if quantifiers:
        if not isinstance(quantifiers, (tuple, list)):
            quantifiers = (quantifiers,)
        if len(quantifiers) == len(filters):
            filters = zip(quantifiers, filters)
        elif quantifiers[1:]:
            warnings.warn('wrong number of quantifiers; ignoring them')
        else:
            quantifier = quantifiers[0]
    # labels
    if labels is None:
        labels = []
    elif isinstance(labels, (tuple, list)):
        labels = list(labels) # copy
    else:
        labels = [labels]
    labels_defined = bool(labels)
    if labels_defined:
        labels = labels[::-1]
    subtree, matches, lookup = None, [], True
    for i, _filter in enumerate(filters):
        if quantifiers:
            quantifier, _filter = _filter
        if isinstance(_filter, (type, tuple, list)):
            _type = _filter
            _filter = lambda a: isinstance(a.data, _type)
        elif callable(_filter):
            if not fullnode:
                f = _filter
                _fitler = lambda a: f(a.data)
        else:
            raise TypeError('invalid filter type: {}'.format(type(_filter)))
        match = []
        while True:
            if lookup:
                if labels:
                    label = labels.pop()
                elif labels_defined:
                    if match and i + 1 == len(filters):
                        break
                    raise ValueError('no match for {}{} filter'.format(i+1,
                        {1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')))
                else:
                    _labels = list(analyses.labels)
                    if _labels and not _labels[1:]:
                        label = _labels[0]
                    elif match and i + 1 == len(filters):
                        break
                    elif not _labels:
                        raise ValueError('no match for {}{} filter'.format(i+1,
                            {1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')))
                    else:#if _labels[1:]:
                        raise ValueError('multiple labels; argument `labels` required')
                try:
                    analyses = analyses.instances[label]
                except KeyError:
                    available = str(list(analyses.labels))[1:-1]
                    if available:
                        raise KeyError("missing label '{}'; available labels are: {}".format(label, available))
                    else:
                        raise KeyError("missing label '{}'; no labels available".format(label))
            lookup = True
            if _filter(analyses):
                match.append(analyses)
            elif match:
                lookup = False
                break
        if quantifier in ('first', ):
            subtree = match[0]
            match = subtree._data
        elif quantifier in ('last', None):
            subtree = match[-1]
            match = subtree._data
        elif quantifier in ('all', '+'):
            subtree = match
            match = [ a._data for a in match ]
        else:
            raise ValueError('invalid quantifier: {}'.format(quantifier))
        matches.append(match)
    if return_subtree:
        matches.append(subtree)
    return tuple(matches)


def coerce_labels_and_metadata(analyses):
    for key in analyses.metadata:
        val = analyses.metadata[key]
        if not isinstance(val, str):
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, unicode):
                val = val.encode('utf-8')
        analyses.metadata[key] = val
    for label in tuple(analyses.labels):
        if isinstance(label, (int, str)):
            coerced = label
        else:
            try: # Py2
                coerced = label.encode('utf-8')
            except AttributeError: # Py3
                try:
                    coerced = label.decode('utf-8')
                except AttributeError: # numpy.int64?
                    coerced = int(label)
            assert isinstance(coerced, (int, str))
        comment = analyses.comments[label]
        analysis = analyses.instances.pop(label)
        if isinstance(analysis, abc.Analyses):
            analysis = coerce_labels_and_metadata(analysis)
        analyses.instances[coerced] = analysis
        if comment:
            analyses.comments[coerced] = comment
    return analyses

coerce_labels = coerce_labels_and_metadata


def format_analyses(analyses, prefix='\t', node=type, global_prefix='', format_standalone_root=None,
        metadata=False, annotations={}):
    if isinstance(annotations, dict):
        root_annotation = annotations.get('', None)
    elif callable(annotations):
        root_annotation = annotations
    elif annotations:
        root_annotation = annotations.pop(0)
    if format_standalone_root is None:
        if metadata:
            if root_annotation is None:
                format_standalone_root = lambda r: '<Analyses {}>\n{}'.format(*r.split('\n',1))
            elif callable(root_annotation):
                def _format_standalone_root(r):
                    _type, _metadata = r.split('\n',1)
                    _annotation = root_annotation('', _type)
                    if _annotation:
                        return '<Analyses {}>\t<- {}\n{}'.format(_type, _annotation, _metadata)
                    else:
                        return '<Analyses {}>\n{}'.format(*r.split('\n',1))
                format_standalone_root = _format_standalone_root
            else:
                format_standalone_root = lambda r: '<Analyses {1}>\t<- {0}\n{2}'.format(root_annotation, *r.split('\n',1))
        else:
            if root_annotation is None:
                format_standalone_root = lambda r: '<Analyses {}>'.format(r)
            else:
                format_standalone_root = lambda r: '<Analyses {}>\t<- {}'.format(r, root_annotation)
    def _format(data, label=None, comment=None, metadata=None, depth=0):
        _prefix = global_prefix + prefix * depth
        s = [_prefix]
        t = []
        if label is None:
            assert comment is None
            if node:
                _node = node(data)
                desc = str(_node)
                s.append(desc)
            else:
                return None
        else:
            try:
                label + 0 # check numeric types
            except TypeError:
                s.append("'{}'")
            else:
                s.append('[{}]')
            t.append(label)
            if node:
                s.append(' {}')
                desc = _node = node(data)
                t.append(desc)
            if isinstance(annotations, dict):
                annotation = annotations.get(label, None)
            elif callable(annotations):
                annotation = annotations(label, _node)
            elif annotations:
                annotation = annotations.pop(0)
            if annotation is not None:
                s.append('\t<- {}')
                t.append(annotation)
            if comment:
                assert isinstance(comment, str)
                _comment = comment.split('\n')
                s.append('\n{}"{}')
                t.append(_prefix)
                t.append(_comment[0])
                for _line in _comment[1:]:
                    s.append('\n{} {}')
                    t.append(_prefix)
                    t.append(_line)
                s.append('"')
        if metadata:
            for key in metadata:
                val = metadata[key]
                s.append('\n{}@{}={}')
                t.append(_prefix)
                t.append(key)
                t.append(val)
        return ''.join(s).format(*t)
    def _flatten(_node):
        if _node is None:
            return []
        elif isinstance(_node, str):
            return [ _node ]
        try:
            _node, _children = _node
        except TypeError:
            return []
        else:
            assert isinstance(_node, str)
            return itertools.chain([_node], *[_flatten(c) for c in _children])
    entries = list(_flatten(map_analyses(_format, analyses, label=True, comment=True, metadata=metadata, depth=True)))
    if entries[1:]:
        if root_annotation:
            r = entries[0]
            if metadata:
                _type, _metadata = r.split('\n',1)
            else:
                _type = r
            if callable(root_annotation):
                _annotation = root_annotation('', _type)
            else:
                _annotation = root_annotation
            if _annotation:
                it = iter(entries[1:])
                _min_offset = None
                while _min_offset is None:
                    try:
                        s = next(it)
                        _min_offset = s.index('<-')
                    except ValueError:
                        pass
                _ntabs = s.count('\t', 0, _min_offset)
                _tab = lambda s: ' '*(_min_offset-_ntabs-len(s))+'\t'*_ntabs
                if metadata:
                    entries[0] = '{}{}<- {}\n{}'.format(_type, _tab(_type), _annotation, _metadata)
                else:
                    entries[0] = '{}{}<- {}'.format(r, _tab(r), _annotation)
        return '\n'.join(entries)
    else:
        return format_standalone_root(entries[0])


def append_leaf(analysis_tree, augmented_branch, overwrite=False):
    """
    Merge new analyses into an existing analysis tree.

    Only leaves and missing branches are appended.
    Existing nodes with children nodes are left untouched.

    Arguments:

        analysis_tree (tramway.core.analyses.base.Analyses): existing analysis tree.

        augmented_branch (tramway.core.analyses.base.Analyses): sub-tree with extra leaves.

    """
    if augmented_branch:
        for label in augmented_branch:
            if label in analysis_tree and (not overwrite or augmented_branch[label]):
                append_leaf(analysis_tree[label], augmented_branch[label])
            else:
                analysis_tree.add(augmented_branch[label], label=label)
    else:
        if analysis_tree:
            return
            #raise ValueError('the existing analysis tree has higher branches than the augmented branch')
        analysis_tree.data = augmented_branch.data



__all__ = [
    'AnalysesView',
    'InstancesView',
    'CommentsView',
    'Analyses',
    'map_analyses',
    'extract_analysis',
    'label_paths',
    'find_artefacts',
    'coerce_labels_and_metadata',
    'coerce_labels',
    'format_analyses',
    'append_leaf',
    'valid_label',
    ]

