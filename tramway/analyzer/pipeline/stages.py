
"""
Standard analysis steps for RWAnalyzer.pipeline
"""

from ..roi import FullRegion
from . import PipelineStage


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


def tessellate(label=None, roi_expected=False, **kwargs):
    """
    Returns a standard pipeline stage for SPT data sampling.

    The default granulariy is set to 'spt data',
    whereas in principle the finest valid granularity is 'roi'.
    The tessellation step is fast enough to a avoid the overhead of saving
    as many files as regions of interest.

    .. important::

        If the input SPT data is defined as ascii file(s),
        the corresponding *.rwa* files will be overwritten.

    """

    asr_filters = _filters(kwargs)

    def _tessellate(self):

        dry_run = True

        for f in self.spt_data:

            # for logging
            source_name = _spt_source_name(f)
            # for data formatting
            any_full_region = False

            with f.autosaving():

                for r in f.roi.as_support_regions(**asr_filters):

                    if isinstance(r, FullRegion):
                        if roi_expected:
                            continue
                        any_full_region = True
                        msg = f"tessellating source: '{source_name}'..."
                        self.logger.info()
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
                    r.add_sampling(sampling, label=label)

                if not any_full_region:
                    # save the full dataset with reasonnable precision to save storage space
                    f.set_precision('single')
                    # the data in the roi is stored separately and will keep the original precision

        if dry_run:
            self.logger.info('stage skipped')

    return PipelineStage(_tessellate, granularity='spt data')


def infer(map_label=None, sampling_label=None, roi_expected=False, overwrite=False, **kwargs):
    """
    Returns a standard pipeline stage for inferring model parameters on each region of interest,
    or SPT data item if no roi are defined.

    The default granularity is set to 'roi', which is suitable for computer-intensive
    inferences such as 'DV'.

    If `map_label` is :const:`None`, `overwrite` is ignored and the stage acts like if
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

            with r.autosaving():

                self.logger.info(msg)
                maps = self.mapper.infer(sampling)

                dry_run = False

                # store
                maps.commit_as_analysis(map_label)

        if dry_run:
            self.logger.info('stage skipped')

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


__all__ = ['tessellate', 'infer', 'reload']

