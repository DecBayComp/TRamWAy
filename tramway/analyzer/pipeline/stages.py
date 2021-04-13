
"""
Standard analysis steps for RWAnalyzer.pipeline
"""

from ..roi import FullRegion
from . import PipelineStage
from ..attribute import single
from ..artefact import commit_as_analysis
import os.path
import numpy as np
from tramway.core import load_xyt
from collections import defaultdict


def _spt_source_name(f):
    try:
        alias = f.alias
    except AttributeError:
        alias = None
    return f.source if alias is None else alias

def _filters(kwargs):
    asr_filters = {}
    for k in kwargs:
        if k.startswith('spt_') or k.startswith('roi_'):
            asr_filters[k[4:]] = kwargs[k]
    return asr_filters

def _placeholder_for_root_data(tree):
    tree._data = None


def tessellate(label=None, roi_expected=False, spt_data=True, tessellation='freeze', **kwargs):
    """
    Returns a standard pipeline stage for SPT data sampling.

    Although the name 'tessellate' refers merely to the spatial
    segmentation, time segmentation is also handled, if defined.
    The name alludes to the standalone
    :func:`tramway.helper.tessellation.tessellate` function.

    The default granulariy is set to 'spt data',
    whereas in principle the finest valid granularity is 'roi'.
    The tessellation step is fast enough to a avoid the overhead
    of saving as many files as regions of interest.

    .. important::

        If the input SPT data is defined as ascii file(s),
        the corresponding *.rwa* files will be overwritten.

        If analysis trees already reference analysis artefacts
        for the specified label, these artefacts are also
        overwritten.

    This stage building function features two mechanisms to
    make the resulting *.rwa* files smaller in size:

    * with option `spt_data='placeholder'`, the SPT data can be
      omitted in the *.rwa* files;
      see also :func:`restore_spt_data`;
    * the spatial tessellation can be freezed (default);
      this implies the tesellation cannot be updated any longer
      with extra data.

    """

    asr_filters = _filters(kwargs)

    def _tessellate(self):

        dry_run = True

        for f in self.spt_data:

            # for logging
            source_name = _spt_source_name(f)
            # for data formatting
            any_full_region = False

            with f.autosaving() as tree:

                for r in f.roi.as_support_regions(**asr_filters):

                    if isinstance(r, FullRegion):
                        if roi_expected:
                            continue
                        any_full_region = True
                        msg = f"tessellating source: '{source_name}'..."
                    else:
                        roi_label = r.label
                        msg = f"tessellating roi: '{roi_label}' (in source '{source_name}')..."

                    # get the SPT data
                    df = r.crop()

                    # filter some translocations out
                    df = r.discard_static_trajectories(df)

                    # tessellate
                    self.logger.info(msg)
                    sampling = self.sampler.sample(df)

                    dry_run = False

                    # store
                    if tessellation=='freeze':
                        try:
                            sampling.tessellation.freeze()
                        except AttributeError:
                            pass
                    r.add_sampling(sampling, label=label)

                if spt_data == 'placeholder':
                    ## [does not work:] make the dataframe empty and keep the additional attributes
                    #df = f.dataframe
                    #f._dataframe = df[np.zeros(df.shape[0], dtype=bool)]
                    tree.hooks.append(_placeholder_for_root_data)
                elif not any_full_region:
                    # save the full dataset with reasonnable precision to save storage space
                    f.set_precision('single')
                    # the data in the roi is stored separately and will keep the original precision

        if dry_run:
            self.logger.info('stage skipped')
            diagnose(self)

    return PipelineStage(_tessellate, granularity='spt data')


def infer(map_label=None, sampling_label=None, roi_expected=False, overwrite=False,
        single_path=False, **kwargs):
    """
    Returns a standard pipeline stage for inferring model parameters on each region of interest,
    or SPT data item if no roi are defined.

    The default granularity is set to 'roi', which is suitable for computer-intensive
    inferences such as 'DV'.

    With default ``overwrite=False``, if the specified output label `map_label` already
    exists in analysis trees, the inference is skipped for these analysis trees and the
    corresponding artefact not overwritten.
    However, if `map_label` is :const:`None`, `overwrite` is ignored and the stage acts like if
    `overwrite` were :const:`True`.
    """

    asr_filters = _filters(kwargs)

    def _infer(self):

        dry_run = True

        for r in self.roi.as_support_regions(**asr_filters):

            # get the input data
            sampling = r.get_sampling(sampling_label)

            if not overwrite and map_label in sampling.subtree:
                # skip the already-processed data
                continue

            source_name = _spt_source_name(r._spt_data)
            if isinstance(r, FullRegion):
                if roi_expected:
                    continue
                msg = f"inferring on source: '{source_name}'..."
            else:
                roi_label = r.label
                msg = f"inferring on roi: '{roi_label}' (in source '{source_name}')..."

            with r.autosaving() as tree:

                self.logger.info(msg)
                maps = self.mapper.infer(sampling)

                dry_run = False

                # store
                maps.commit_as_analysis(map_label)

                if single_path:
                    for label in tree:
                        if label != sampling_label:
                            del tree[label]
                    for label in tree[sampling_label]:
                        if label != map_label:
                            del tree[sampling_label][label]

        if dry_run:
            self.logger.info('stage skipped')
            diagnose(self)

    return PipelineStage(_infer, granularity='roi')


def reload(skip_missing=True):
    """
    Returns a pipeline stage to reload the generated *.rwa* files.

    This is useful if a first computation - *e.g.* `tessellate` - was dispatched
    (including with the `environments.LocalHost` environment)
    and the *.rwa* files were generated and retrieved from the worker
    environments.
    In the case a second stage has to be dispatched - *e.g.* `infer`,
    the local pipeline must be updated with the content of these retrieved files.

    This may also be required on the worker side for the next stages,
    for example if the SPT data was initially defined from ascii files
    and the coming stages should take over from the corresponding *.rwa* files.
    As a consequence, this `reload` stage is set with `requires_mutability=True`.

    """

    def _reload(self):
        self.spt_data.reload_from_rwa_files(skip_missing=skip_missing)

    return PipelineStage(_reload, requires_mutability=True)

def restore_spt_data():
    """
    Reintroduces the original SPT dataframe into the *.rwa* files.

    This post-processing stage - typically post-infer stage - is expected to be
    called only when the `tessellate` stage was called before with option
    `spt_data='placeholder'`.

    This stage was tested with SPT data defined as :class:`~..spt_data.SPTAsciiFiles`
    or :class:`..spt_data.RWAFiles`.
    However, with :class:`SPTAsciiFiles`, the `alias` attribute should be defined.

    """
    from ..spt_data import RWAFile

    def _restore(self):
        for f in self.spt_data:
            if not isinstance(f, RWAFile):
                try:
                    alias = f.alias
                except AttributeError:
                    alias = None
                if alias is None:
                    raise NotImplementedError('alias is not defined')
                #
                analyzer = self._eldest_parent
                analyzer.spt_data.reload_from_rwa_files()
                f = single(analyzer.spt_data.filter_by_source(alias))
            #
            spt_ascii_file = f.analyses.metadata['datafile']
            rwa_file = f.filepath
            local_spt_file = os.path.join(os.path.dirname(rwa_file), os.path.basename(spt_ascii_file))
            if os.path.isfile(local_spt_file):
                df = load_xyt(local_spt_file)
                restore = no_dataframe = f.dataframe is None
                if not no_dataframe:
                    cols, cols_ = f.dataframe.columns, df.columns
                    restore = len(cols)==len(cols_) and all([ c==c_ for c, c_ in zip(cols, cols_) ])
                if restore:
                    with f.autosaving() as tree:
                        f._dataframe = df
                        tree.flag_as_modified()
                else:
                    self.logger.error('column names do not match with file: {}'.format(local_spt_file))
            else:
                self.logger.error('could not find file: {}'.format(local_spt_file))

    return PipelineStage(_restore, granularity='spt data')


def tessellate_and_infer(map_label=None, sampling_label=None, spt_data=True, overwrite=False,
        roi_expected=False, **kwargs):

    save_active_branches_only = not (overwrite or map_label is None)

    def _tessellate_and_infer(self):

        if not self._eldest_parent.env.initialized:
            save_active_branches_only = False

        dry_run = True

        for f in self.spt_data:

            source_dry_run = True

            # for logging
            source_name = _spt_source_name(f)
            # for data formatting
            any_full_region = False

            if save_active_branches_only:
                active_labels = defaultdict(set)

            with f.autosaving() as tree:

                for r in f.roi.as_support_regions():

                    if isinstance(r, FullRegion):
                        if roi_expected:
                            continue
                        any_full_region = True
                        msg = f"{{}} source: '{source_name}'..."
                    else:
                        roi_label = r.label
                        msg = f"{{}} roi: '{roi_label}' (in source '{source_name}')..."
                    def log(op):
                        self.logger.info(msg.format(op))

                    if sampling_label is None:
                        label = r.label
                    elif callable(sampling_label):
                        label = sampling_label(r.label)
                    else:
                        if not source_dry_run:
                            self.logger.warning("multiple roi bound to single sampling label; make sampling_label callable")
                        label = sampling_label

                    # control predicates
                    _tessellate = overwrite or label is None or label not in tree.labels
                    _infer = _tessellate or overwrite or map_label is None or \
                            map_label not in tree[label].labels

                    if _tessellate:

                        # get the SPT data
                        df = r.crop()

                        if df.empty:
                            raise ValueError(f'not a single displacement in roi {roi_label}')

                        # filter some translocations out
                        df = r.discard_static_trajectories(df)

                        if df.empty:
                            raise ValueError(f'all the trajectories are static in roi {roi_label}')

                        # tessellate
                        log('tessellating')
                        sampling = self.sampler.sample(df)

                    elif _infer:
                        sampling = r.get_sampling(sampling_label).data

                    if _infer:

                        # infer
                        log('inferring')
                        maps = self.mapper.infer(sampling)

                        dry_run = source_dry_run = False

                    # store
                    if _tessellate:
                        # ... the tessellation
                        try:
                            sampling.tessellation.freeze()
                        except AttributeError:
                            pass
                        sampling = commit_as_analysis(label, sampling, parent=r)
                        #
                    elif _infer:
                        sampling = r.get_sampling(sampling_label)
                    if _infer:
                        # ... and the inferred maps
                        maps     = commit_as_analysis(map_label, maps, parent=sampling)
                        #
                        assert tree.modified(recursive=True)
                        if save_active_branches_only:
                            active_labels[label].add(map_label)

                if save_active_branches_only:
                    for label0 in list(tree.labels):
                        if label0 in active_labels:
                            for label1 in list(tree[label0].labels):
                                if label1 not in active_labels[label0]:
                                    tree[label0].comments
                                    del tree[label0][label1]
                        else:
                            tree.comments
                            del tree[label0]

                if spt_data=='placeholder':
                    tree.hooks.append(_placeholder_for_root_data)

        if dry_run:
            self.logger.info('stage skipped')
            diagnose(self)

    return PipelineStage(_tessellate_and_infer,
            granularity='roi',
            update_existing_rwa_files=not overwrite)


def diagnose(self):
    """
    Checks for filter consistency.

    To be called only in the case of proven dry runs.
    """
    from ..spt_data import _normalize

    sources = self.env.selectors['source']
    if isinstance(sources, str):
        sources = (sources,)

    try:
        alias = all([ bool(f.alias) for f in self._eldest_parent.spt_data ])
    except AttributeError:
        alias = False
    if alias:
        requested = set(sources)
        available = set(self.spt_data.aliases)
    else:
        requested = set([ _normalize(_s) for _s in sources ])
        available = set([ _normalize(_f.source) for _f in self.spt_data ])

    if bool( requested - available ):
        self.logger.critical(f"""
The local worker was assigned the following SPT data source(s):
   {requested}
but listed other sources:
   {available}
This is known to happen when files are listed from the file system
and a subset of the listed files is selected based on their order
in the list.
Avoid doing so!
Filename ordering varies from a file system to another.
""")
        raise RuntimeError


__all__ = ['tessellate', 'infer', 'reload', 'restore_spt_data', 'tessellate_and_infer']

