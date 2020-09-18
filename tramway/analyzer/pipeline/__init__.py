
from ..attribute import *
from ..roi import DecentralizedROIManager
import sys
import traceback


class PipelineStage(object):
    __slots__ = ('_run','_granularity','mutable','options')
    def __init__(self, run, granularity='coarsest', mutable=False, **options):
        self._run = run
        self._granularity = granularity
        self.mutable = mutable
        self.options = options
    @property
    def granularity(self):
        return self._granularity
    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class Pipeline(AnalyzerNode):
    """
    `pipeline` attribute of an :class:`~tramway.analyzer.RWAnalyzer` object.

    The main methods are `append_stage` and `run`.
    Note that the `run` method is called by :met:`~tramway.analyzer.RWAnalyzer.run`
    of :class:`~tramway.analyzer.RWAnalyzer`.
    """
    __slots__ = ('_stage',)
    def __init__(self, *args, **kwargs):
        AnalyzerNode.__init__(self, *args, **kwargs)
        self._stage = []
    @property
    def analyzer(self):
        return self._parent
    @property
    def logger(self):
        return self._parent.logger
    @property
    def spt_data(self):
        return self._parent.spt_data
    @property
    def roi(self):
        return self._parent.roi
    @property
    def time(self):
        return self._parent.time
    @property
    def tesseller(self):
        return self._parent.tesseller
    @property
    def sampler(self):
        return self._parent.sampler
    @property
    def mapper(self):
        return self._parent.mapper
    @property
    def env(self):
        return self._parent.env
    def reset(self):
        """
        Empties the pipeline processing chain.
        """
        self._stage = []
    def append_stage(self, stage, granularity='coarsest', mutable=False, **options):
        """
        Appends a pipeline stage to the processing chain.

        Arguments:

            stage (callable): function that takes an :class:`~tramway.analyzer.RWAnalyzer` object
                as unique input argument.

            granularity (str): smallest data item `stage` can independently process;
                any of *'coarsest'* or equivalently *'full dataset'*,
                *'source'* or equivalently *'spt data'*, *'data source'* or *'spt data source'*,
                *'roi'* or equivalently *'region of interest'* (case-insensitive).

            mutable (bool): callable object `stage` alters input argument `self`.
                Stages with `mutable` set to ``True`` are always run as dependencies.

        """
        self._stage.append(PipelineStage(stage, granularity, mutable, **options))
    def run(self, stages='all', verbose=False):
        """
        Sequentially runs the different stages of the pipeline.

        The input arguments are currently ignored.
        """
        if self.env.initialized:
            try:
                self.env.setup(*sys.argv)
                self.logger.info('setup complete')
                if self.env.worker_side:
                    # a single stage can apply
                    stage_index = self.env.selectors.get('stage_index', 0)
                    if isinstance(stage_index, (tuple,list)):
                        for i in stage_index[:-1]:
                            stage = self._stage[i]
                            stage(self)
                        stage_index = stage_index[-1]
                    stage = self._stage[stage_index]
                    # alter the iterators for spt_data
                    self.analyzer.spt_data.self_update(self.env.spt_data_selector)
                    # alter the iterators for roi
                    if isinstance(self.roi, DecentralizedROIManager):
                        for f in self.spt_data:
                            f.roi.self_update(self.env.roi_selector)
                    elif not isinstance(self.roi, Initializer):
                        self.analyzer.roi.self_update(self.env.roi_selector)
                    self.logger.info('stage {:d} ready'.format(stage_index))
                    try:
                        stage(self)
                    except:
                        self.logger.error('stage {:d} failed with t'.format(stage_index)+traceback.format_exc()[1:-1])
                        raise
                    else:
                        self.logger.info('stage {:d} done'.format(stage_index))
                    #
                    #self.env.save_analyses(self.spt_data)
                else:
                    assert self.env.submit_side
                    if self.env.dispatch():
                        self.logger.info('initial dispatch done')
                    mutable = []
                    for s, stage in enumerate(self._stage):
                        granularity = '' if stage.granularity is None else stage.granularity.lower()
                        if granularity in ('coarsest','full dataset'):
                            self.logger.info('stage {:d} ready'.format(s))
                            stage(self)
                            self.logger.info('stage {:d} done'.format(s))
                            if stage.mutable:
                                mutable.append(s)
                            continue
                        if stage.mutable:
                            raise NotImplementedError('cannot make a dispatched job modify the local analyzer')
                        if self.env.dispatch(stage_index=s, stage_options=stage.options):
                            self.logger.info('stage {:d} dispatched'.format(s))
                        if mutable:
                            s = mutable+[s]
                        if granularity.endswith('source') or granularity.startswith('spt data'):
                            for f in self.spt_data:
                                if f.source is None and 1<len(self.spt_data):
                                    raise NotImplementedError('undefined source identifiers')
                                if self.env.dispatch(source=f.source):
                                    self.logger.info('source "{}" dispatched'.format(f.source))
                                self.env.make_job(stage_index=s, source=f.source)
                        elif granularity in ('roi','region of interest'):
                            for f in self.spt_data:
                                if f.source is None and 1<len(self.spt_data):
                                    raise NotImplementedError('undefined source identifiers')
                                if self.env.dispatch(source=f.source):
                                    self.logger.info('source "{}" dispatched'.format(f.source))
                                for i, _ in f.roi.as_support_regions(return_index=True):
                                    self.env.make_job(stage_index=s, source=f.source, region_index=i)
                        else:
                            raise NotImplementedError('only roi-level granularity is currently supported')
                        self.logger.info('jobs ready')
                        try:
                            self.env.submit_jobs()
                            self.logger.info('jobs submitted')
                            self.env.wait_for_job_completion()
                        except KeyboardInterrupt:
                            self.logger.critical('interrupting jobs...')
                            if not self.env.interrupt_jobs():
                                raise
                        self.logger.info('jobs complete')
                        self.env.collect_results()
                        self.logger.info('results collected')
            finally:
                if self.env.submit_side and not self.env.debug:
                    self.env.delete_temporary_data()
        else:
            for stage in self._stage:
                stage(self)


__all__ = ['Pipeline']

