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
from ..roi import DecentralizedROIManager
import sys
import traceback


class PipelineStage(object):
    """
    Wrapper for callables to be run on calling :meth:`Pipeline.run`.

    `requires_mutability` defaults to :const:`False`.
    """
    __slots__ = ('_run','_granularity','_mutability','options')
    def __init__(self, run, granularity=None, requires_mutability=None, **options):
        if isinstance(run, PipelineStage):
            self._run = run._run
            self._granularity = run._granularity if granularity is None else granularity
            self._mutability = run._mutability if requires_mutability is None else requires_mutability
            self.options = dict(run.options)
            self.options.update(options)
        else:
            self._run = run
            self._granularity = granularity
            self._mutability = False if requires_mutability is None else requires_mutability
            self.options = options
    @property
    def granularity(self):
        """
        *str*: See :meth:`Pipeline.append_stage`
        """
        return self._granularity
    @property
    def requires_mutability(self):
        """
        *bool*: See :meth:`Pipeline.append_stage`
        """
        return self._mutability
    @property
    def name(self):
        """
        *str*: Callable's name
        """
        return self._run.__name__
    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class Pipeline(AnalyzerNode):
    """
    :attr:`~tramway.analyzer.RWAnalyzer.pipeline` attribute of an
    :class:`~tramway.analyzer.RWAnalyzer` object.

    The main methods are :meth:`append_stage` and :meth:`run`.
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
    def append_stage(self, stage, granularity=None, requires_mutability=None, **options):
        """
        Appends a pipeline stage to the processing chain.

        Arguments:

            stage (callable): function that takes an :class:`~tramway.analyzer.RWAnalyzer` object
                as unique input argument.

            granularity (str): smallest data item `stage` can independently process;
                any of :const:`'coarsest'` or equivalently :const:`'full dataset'`,
                :const:`'source'` or equivalently :const:`'spt data'`, :const:`'data source'`
                or :const:`'spt data source'`, :const:`'roi'` or equivalently
                :const:`'region of interest'`,
                :const:`'time'` or equivalently :const:`'segment'` or :const:`'time segment'`
                (case-insensitive).

            requires_mutability (bool): callable object `stage` alters input argument `self`.
                Stages with this argument set to :const:`True` are always run as dependencies.
                Default is :const:`False`.

        """
        self._stage.append(PipelineStage(stage, granularity, requires_mutability, **options))
    def early_setup(self):
        """
        Sets the `submit_side`/`worker_side` properties of the :attr:`~tramway.analyzer.RWAnalyzer.env`
        attribute.

        :meth:`early_setup` can be called once :attr:`~tramway.analyzer.RWAnalyzer.env` is initialized
        and before :meth:`run` is called.
        This allows to run some conditional code, specifically on the submit side or on the worker
        side.

        This method returns silently if :attr:`~tramway.analyzer.RWAnalyzer.env` is not initialized
        and both the :attr:`..env.EnvironmentInitializer.submit_side` and
        :attr:`..env.EnvironmentInitializer.worker_side` properties are :const:`False`.
        """
        if self.env.initialized:
            self.env.early_setup(*sys.argv)
    def run(self):
        """
        Sequentially runs the different stages of the pipeline.
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
                    # alter the iterators for time
                    if not isinstance(self.time, Initializer):
                        self.analyzer.time.self_update(self.env.time_selector)
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
                    stack = []
                    permanent_stack = []
                    for s, stage in enumerate(self._stage):
                        granularity = '' if stage.granularity is None \
                                else stage.granularity.lower().replace('-',' ').replace('_',' ')
                        if stage.requires_mutability:
                            if not granularity or granularity in ('coarsest','full dataset'):
                                # run locally
                                self.logger.info('stage {:d} ready'.format(s))
                                stage(self)
                                self.logger.info('stage {:d} done'.format(s))
                                # make the next dispatched stages run this stage again
                                permanent_stack.append(s)
                                continue
                            else:
                                raise NotImplementedError('cannot make a dispatched job modify the local analyzer')
                        if self.env.dispatch(stage_index=s, stage_options=stage.options):
                            self.logger.info('stage {:d} dispatched'.format(s))
                        if stack or permanent_stack:
                            s = sorted(permanent_stack+stack+[s])
                            stack = []
                        if not granularity or granularity in ('coarsest','full dataset'):
                            self.env.make_job(stage_index=s)
                        else:
                            try:
                                alias = all([ bool(f.alias) for f in self._eldest_parent.spt_data ])
                            except AttributeError:
                                alias = False
                            for f in self.spt_data:
                                source = f.alias if alias else f.source
                                if source is None:
                                    if 1<len(self.spt_data):
                                        raise NotImplementedError('undefined source identifiers')
                                elif self.env.dispatch(source=source):
                                    self.logger.info('source "{}" dispatched'.format(source))
                                if granularity.endswith('source') or granularity.startswith('spt data'):
                                    self.env.make_job(stage_index=s, source=source)
                                elif granularity in ('roi','region of interest'):
                                    for i, _ in f.roi.as_support_regions(return_index=True):
                                        self.env.make_job(stage_index=s, source=source, region_index=i)
                                elif granularity in ('time', 'segment', 'time segment'):
                                    for i, r in f.roi.as_support_regions(return_index=True):
                                        try:
                                            w = r.get_sampling()
                                        except ValueError:
                                            raise NotImplementedError('cannot iterate on multiple sampling per ROI yet') from None
                                        except KeyError:
                                            raise NotImplementedError('cannot autoload the sampling stage; please load the sampling in a separate stage with requires_mutability=True') from None
                                        for j, _ in self.time.as_time_segments(w, return_index=True, return_times=False):
                                            self.env.make_job(stage_index=s, source=source, region_index=i, segment_index=j)
                                else:
                                    raise NotImplementedError
                        self.logger.info('jobs ready')
                        try:
                            self.env.submit_jobs()
                            self.logger.info('jobs submitted')
                            self.env.wait_for_job_completion()
                        except KeyboardInterrupt as e:
                            self.logger.critical('interrupting jobs...')
                            try:
                                ret = self.env.interrupt_jobs()
                            except KeyboardInterrupt:
                                self.logger.debug('interrupt_jobs() did not return')
                                raise e from None
                            else:
                                if not ret:
                                    raise
                        self.logger.info('jobs complete')
                        if self.env.collect_results(stage_index=s):
                            self.logger.info('results collected')
            finally:
                if self.env.submit_side and not self.env.debug:
                    self.env.delete_temporary_data()
        else:
            for stage in self._stage:
                stage(self)

    def add_collectible(self, filepath):
        """
        Registers a file as to be collected after stage completion.
        """
        self.env.collectibles.add(filepath)

    def resume(self, **kwargs):
        """
        Looks for orphaned remote jobs and collect the generated files.

        Works as a replacement for the :meth:`run` method to recover
        after connection loss.

        Recovery procedures featured by the :attr:`~tramway.analyzer.RWAnalyzer.env`
        backend may fail or recover some of the generated files only.

        See also the :meth:`~tramway.analyzer.env.environments.SlurmOverSSH.resume` method
        of the :attr:`~tramway.analyzer.RWAnalyzer.env` attribute, if available.
        """
        try:
            proc = self.env.resume
        except AttributeError:
            self.logger.error('no recovery procedure available')
        else:
            proc(**kwargs)

Attribute.register(Pipeline)


from . import stages

__all__ = ['Pipeline', 'stages']

