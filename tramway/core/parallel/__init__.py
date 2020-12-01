# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue
import time
import numpy as np
from warnings import warn
try:
    from . import abc
except SyntaxError: # Py2
    import abc_py2 as abc
from traceback import format_exc

#import logging # DEBUG
#module_logger = logging.getLogger(__name__)
#module_logger.setLevel(logging.DEBUG)
#_console = logging.StreamHandler()
#_console.setFormatter(logging.Formatter('%(message)s\n'))
#module_logger.addHandler(_console)


class StarQueue(object):
    """
    A star queue is a multidirectional queue such that each message must be consumed
    by every processes but the sender.
    This is useful to send asynchronous updates between processes in a distributed setting.

    A :class:`StarQueue` is instanciated first, and then :class:`StarConn` objects are delt
    to children processes.
    The :class:`StarConn` objects are the actual queues.
    """
    __slots__ = 'deck',
    def __init__(self, n, variant=multiprocessing.Queue, **kwargs):
        self.deck = []
        queues = [ variant(**kwargs) for _ in range(n) ]
        others = []
        for _ in range(n):
            _queue = queues.pop()
            self.deck.append((_queue, queues+others))
            others.append(_queue)
    def deal(self):
        try:
            return StarConn(*self.deck.pop())
        except IndexError:
            raise RuntimeError('too many consumers')

class StarConn(queue.Queue):
    __slots__ = 'input', 'output'
    def __init__(self, input_queue=None, output_queues=None):
        queue.Queue.__init__(self)
        self.input = input_queue
        self.output = output_queues
    @property
    def joinable(self):
        return isinstance(self.input, multiprocessing.JoinableQueue)
    @property
    def variant(self):
        return type(self.input)
    def qsize(self):
        return sum( q.qsize() for q in self.output )
    def empty(self):
        return all( q.empty() for q in self.output )
    def full(self):
        return any( q.full()  for q in self.output)
    def put(self, obj, block=True, timeout=None):
        if timeout and 0 < timeout:
            t0 = time.time()
        _timeout = timeout
        for _queue in self.output:
            _queue.put(obj, block, _timeout)
            if timeout and 0 < timeout:
                _timeout = timeout - (time.time() - t0)
    def put_nowait(self, obj):
        for _queue in self.output:
            _queue.put_nowait(obj)
    def get(self, block=True, timeout=None):
        return self.input.get(block, timeout)
    def get_nowait(self):
        return self.input.get_nowait()
    def close(self):
        if self.output is None:
            raise RuntimeError('queue not yet `close` should not be called by the parent process')
        for _queue in self.output:
            _queue.close()
    def join_thread(self):
        if self.output is None:
            raise RuntimeError('`join_thread` should not be called by the parent process')
        for _queue in self.output:
            _queue.join_thread()
    def task_done(self):
        if self.input is None:
            raise RuntimeError('no task was received through the queue')
        try:
            self.input.task_done()
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            if self.joinable:
                raise
            else:
                raise RuntimeError('queue was not set as joinable at init')

class WorkerNearDeathException(Exception):
    __slots__ = '_id', '_name', '_type', '_msg'
    def __init__(self, _id, name, exc_type, traceback):
        self._id = _id
        self._name = name
        self._type = exc_type
        self._msg = traceback.split('\n',1)[1][:-1] # remove first and last lines
    def __str__(self):
        return 'Process {} died with error (most recent call last):\n{}'.format(self._name, self._msg)

class Worker(multiprocessing.Process):
    """ Worker that runs job steps.

    The :meth:`target` method may be implemented following the pattern below:

    .. code:: python

        class MyWorker(Worker):
            def target(self, *args, **kwargs):
                while True:
                    k, task = self.get_task()
                    status = dict()
                    try:
                        # modify `task` and `status`
                    finally:
                        self.push_update(task, status)

    The optional positional and keyword arguments come from the `args` and `kwargs` arguments
    to :meth:`Scheduler.__init__`, plus the extra keyword arguments to the latter constructor.

    """
    def __init__(self, _id, workspace, task_queue, return_queue, update_queue,
            name=None, args=(), kwargs={}, daemon=None, **_kwargs):
        # `daemon` is not supported in Py2; pass `daemon` only if defined
        if daemon is None:
            __kwargs = {}
        else:
            __kwargs = dict(daemon=daemon)
        multiprocessing.Process.__init__(self, name=name, **__kwargs)
        self._id = _id
        self.workspace = workspace
        self.tasks = task_queue
        self.update = update_queue
        self.feedback = return_queue
        self.args = args
        kwargs.update(_kwargs)
        self.kwargs = kwargs
    def get_task(self):
        """ Listen to the scheduler and get a job step to be run.

        The job step is loaded with the worker-local copy of the synchronized workspace.

        Returns:

            int, tramway.core.parallel.JobStep: step/iteration number, job step object.

        """
        #module_logger.debug('get_task: waiting...') # DEBUG
        k, task = self.tasks.get()
        #module_logger.debug('get_task: received {}'.format(k)) # DEBUG
        task.set_workspace(self.workspace)
        self.pull_updates()
        return k, task
    def push_update(self, update, status=None):
        """ Send a completed job step back to the scheduler and to the other workers.

        Arguments:

            update (tramway.core.parallel.JobStep): completed job step.

            status (any): extra information that :meth:`Scheduler.stop` will receive.

        """
        if isinstance(update, abc.VehicleJobStep):
            update.push_updates(self.workspace.pop_extension_updates())
        update.unset_workspace() # free memory space
        if self.update is not None:
            self.update.put(update)
        #module_logger.debug('push_update: sending back') # DEBUG
        self.feedback.put((update, status))
        #module_logger.debug('push_update: sent') # DEBUG
    def pull_updates(self):
        if self.update is None:
            return
        while True:
            try:
                update = self.update.get_nowait()
            except queue.Empty:
                break
            else:
                self.workspace.update(update) # `Workspace.update` reloads the workspace into the update
    def run(self):
        try:
            self.target(*self.args, **self.kwargs)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            self.feedback.put((None,
                WorkerNearDeathException(self._id, self.name, type(e), format_exc())))

class NormalTermination(Exception):
    pass

def _pseudo_worker(worker):
    class PseudoWorker(worker):
        def __init__(self, scheduler, args=(), kwargs={}, _id=0, name=None):
            worker.__init__(self, _id, scheduler.workspace,
                None, None, None, name=name, args=args, kwargs=kwargs)
            self._scheduler = scheduler
        def get_task(self):
            return self._scheduler.next_task()
        def push_update(self, update, status=None):
            i = update.resource_id
            self._scheduler.task[i] = update
            if self._scheduler.pseudo_stop(status):
                raise NormalTermination
        def run(self):
            self.target(*self.args, **self.kwargs)
    return PseudoWorker


class Scheduler(object):
    """ Scheduler that distributes job steps over a shared workspace.

    Workers are spawned and assigned job steps.
    Each worker maintains a copy of the common workspace that is synchronized
    by :meth:`Worker.push_update` calls.

    The :meth:`stop` method should be overloaded so that the distributed computation
    may complete on termination criteria.
    """
    def __init__(self, workspace, tasks, worker_count=None, iter_max=None,
            name=None, args=(), kwargs={}, daemon=None, max_runtime=None,
            task_timeout=None, **_kwargs):
        """
        Arguments:

            workspace (Workspace): workspace to be replicated

            ...

            max_runtime (float): timeout in seconds;
                runtime is absolute, not cumulated over all workers;
                the active workers are interrupted on reaching the timeout;
                *new in 0.5b2*.

            task_timeout (float): timeout in seconds;
                this timeout applies to each task and the corresponding
                worker only is interrupted;
                the other tasks keep on;
                *new in 0.5b5*.

        """
        self.workspace = workspace
        self.task = tasks
        self.active = dict()
        self.paused = dict()
        self.dead_workers = dict()
        self.k_eff = 0
        self.k_max = iter_max
        self.global_timeout = max_runtime
        self.task_timeout = task_timeout
        if worker_count is None:
            worker_count = multiprocessing.cpu_count() - 1
        elif worker_count < 0:
            worker_count = multiprocessing.cpu_count() + worker_count
        kwargs.update(_kwargs)
        if worker_count:
            self.task_queue = multiprocessing.Queue()
            self.return_queue = multiprocessing.Queue()
            if worker_count == 1:
                self.workers = { 0: self.worker(0, self.workspace,
                            self.task_queue, self.return_queue, None,
                            name=name, args=args, kwargs=kwargs, daemon=daemon) }
            else:
                def _name(w):
                    return '{}-{:d}'.format(name, w) if w else None
                update_queue = StarQueue(worker_count)
                self.workers = { i: self.worker(i, self.workspace,
                            self.task_queue, self.return_queue, update_queue.deal(),
                            name=_name(i), args=args, kwargs=kwargs, daemon=daemon)
                        for i in range(worker_count) }
        else:
            self.workers = self.pseudo_worker(self, args=args, kwargs=kwargs)
    def init_resource_lock(self):
        if 1 < self.worker_count:
            self.resource_lock = np.zeros(len(self.workspace), dtype=bool)
    @property
    def worker(self):
        return Worker
    @property
    def pseudo_worker(self):
        return _pseudo_worker(self.worker)
    @property
    def worker_count(self):
        return len(self.workers) if isinstance(self.workers, dict) else 0
    @property
    def timeout(self):
        if self.task_timeout is None:
            return self.global_timeout
        elif self.global_timeout is None:
            return self.task_timeout
        else:
            return min(self.global_timeout, self.task_timeout)
    def draw(self, k):
        return k
    def next_task(self):
        """ no-mp mode only """
        task = None
        while not self.iter_max_reached():
            i = self.draw(self.k_eff)
            self.k_eff += 1
            if i is not None:
                task = self.task[i]
                break
        return self.k_eff-1, task
    def locked(self, step):
        return 1 < self.worker_count and \
                (step.resource_id in self.active or \
                np.any(self.resource_lock[step.resources]))
    def lock(self, step):
        if 1 < self.worker_count:
            self.resource_lock[step.resources] = True
    def unlock(self, step):
        if 1 < self.worker_count:
            self.resource_lock[step.resources] = False
    @property
    def available_slots(self):
        """
        `int`: Number of available workers.
        """
        return self.worker_count - len(self.active)
    def send_task(self, k, step):
        """
        Send a job step to be assigned to a worker as soon as possible.

        Arguments:

            k (int): step/iteration number.

            step (tramway.core.parallel.JobStep): job step.

        """
        self.active[step.resource_id] = k
        self.lock(step)
        step.unset_workspace() # free memory
        #module_logger.debug('send_task: sending {}...'.format(k)) # DEBUG
        self.task_queue.put((k, step))
        #module_logger.debug('send_task: sent') # DEBUG
        self.k_eff += 1
    def get_processed_step(self):
        """
        Retrieve a processed job step and check whether stopping criteria are met.

        Calls the :meth:`stop` method.

        Returns :const:`False` if a stopping criterion has been met, :const:`True` otherwise.
        """
        while True:
            #module_logger.debug('get_processed_step: waiting...') # DEBUG
            # note: `get` does not raise TimeoutError
            step, status = self.return_queue.get(timeout=self.timeout)
            #module_logger.debug('get_processed_step: received {}'.format(self.active[step.resource_id])) # DEBUG
            if step is None:
                if isinstance(status, WorkerNearDeathException):
                    self.dead_workers[status._id] = self.workers.pop(status._id, None)
                    try:
                        self.logger.critical(str(status))
                    except AttributeError:
                        print('error: {}'.format(status))
                    if not self.workers:
                        return False
                else:
                    return self.stop(None, None, status)
            else:
                break
        #step.set_workspace(self.workspace) # reload workspace
        self.workspace.update(step) # `update` reloads the workspace into `step`
        assert step.get_workspace() is not None
        i = step.resource_id
        self.task[i] = step
        k = self.active.pop(i)
        try:
            return not self.stop(k, i, status)
        finally:
            self.unlock(step)
    def iter_max_reached(self):
        return self.k_max and self.k_max <= self.k_eff
    def workers_alive(self):
        self.workers = { i: w for i, w in self.workers.items() if w.is_alive() }
        return bool(self.workers)
    def fill_slots(self, k, postponed):
        """
        Send as many job steps as there are available workers.

        Arguments:

            k (int) : step/iteration number.

            postponed (dict): postponed job steps.

        Returns:

            int: new step/iteration number.

        """
        while 0 < self.available_slots:
            i = self.draw(k)
            if i is None:
                k += 1
            elif i in self.active or i in postponed:
                break
            else:
                task = self.task[i]
                if self.locked(task):
                    postponed[i] = k
                else:
                    self.send_task(k, task)
                k += 1 # increment anyway in the case i is None
                if self.iter_max_reached(): # can happen only after `send_task`
                    #assert False
                    break
        return k
    def run(self):
        """
        Start the workers, send and get job steps back and check for stop criteria.

        Returns :const:`True` on normal completion, :const:`False` on interruption
        (:class:`SystemExit`, :class:`KeyboardInterrupt`).
        """
        if isinstance(self.workers, Worker):
            # no multi-processing
            context = None
            if self.timeout:
                try:
                    import stopit
                except ImportError:
                    warn('limited timeout support; install package stopit to get full support', ImportWarning)
                else:
                    context = stopit.ThreadingTimeout(self.timeout)
            if context is None:
                class DummyContextManager(object):
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                context = DummyContextManager()
            #
            ret = True
            try:
                with context:
                    self.workers.run()
            except NormalTermination:
                pass
            except (SystemExit, KeyboardInterrupt):
                ret = False
            return ret

        for w in self.workers.values():
            w.start()
        self.init_resource_lock()
        if self.global_timeout:
            self.start_time = time.time()
        k = 0
        postponed = dict()
        try:
            k = self.fill_slots(k, postponed)
            while not self.iter_max_reached():
                if not self.workers_alive():
                    break
                if not self.get_processed_step():
                    break
                if self.global_timeout and self.global_timeout <= (time.time() - self.start_time):
                    #raise multiprocessing.TimeoutError
                    break
                if postponed:
                    for i in list(postponed):
                        task = self.task[i]
                        if not self.locked(task):
                            self.send_task(postponed.pop(i), task)
                            if self.available_slots == 0:
                                break
                            else:
                                assert 0 < self.available_slots
                if not postponed:
                    k = self.fill_slots(k, postponed)
            ret = True
        except (SystemExit, KeyboardInterrupt, multiprocessing.TimeoutError):
            ret = False
        for w in self.workers.values():
            try:
                w.terminate()
            except:
                pass
        for w in self.dead_workers.values():
            try:
                w.terminate()
            except:
                pass
        return ret
    def stop(self, k, i, status):
        """
        Default implementation returns ``False``.

        Arguments:

            k (int): step/interation number.

            i (int): step id.

            status (dict): status data returned by a job step.

        Returns:

            bool: ``True`` if a stopping criterion has been met, ``False`` otherwise.

        """
        return False
    def pseudo_stop(self, status):
        k = self.k_eff - 1
        return self.stop(k, self.draw(k), status)


class EpochScheduler(Scheduler):
    def __init__(self, workspace, tasks, epoch_length=None, soft_epochs=False, worker_count=None,
            iter_max=None, name=None, args=(), kwargs={}, daemon=None, **_kwargs):
        epoch_length = len(tasks) if epoch_length is None else epoch_length
        if not soft_epochs and not worker_count:
            worker_count = min(epoch_length, multiprocessing.cpu_count() - 1)
        Scheduler.__init__(self, workspace, tasks, worker_count=worker_count, iter_max=iter_max,
            name=name, args=args, kwargs=kwargs, daemon=daemon, **_kwargs)
        self.soft_epochs = soft_epochs
        self._task_epoch = np.arange(epoch_length)

    def draw(self, k):
        i = k % len(self._task_epoch)
        if i == 0:
            # wait for the active buffer to get empty
            if not self.soft_epochs and self.active:
                return None
            # permute the tasks
            self.start_new_epoch(self._task_epoch)
        return self._task_epoch[i]

    def start_new_epoch(self, task_order):
        """ must modify the `task_order` array inplace """
        np.random.shuffle(task_order)


class ProtoWorkspace(object):
    __slots__ = '_extensions',
    def __init__(self, args=()):
        self._extensions = {}
        if args:
            self.identify_extensions(args)
    def update(self, step):
        step.set_workspace(self)
        if isinstance(step, abc.VehicleJobStep):
            self.push_extension_updates(step.pop_updates())
    def resources(self, step):
        return step.resources
    def identify_extensions(self, args):
        for k, a in enumerate(args):
            if isinstance(a, abc.WorkspaceExtension):
                self._extensions[k] = a
    def push_extension_updates(self, updates):
        for k in updates:
            update = updates[k]
            extension = self._extensions[k]
            extension.push_workspace_update(update)
    def pop_extension_updates(self):
        updates = {}
        for k in self._extensions:
            extension = self._extensions[k]
            updates[k] = extension.pop_workspace_update()
        return updates

abc.ExtendedWorkspace.register(ProtoWorkspace)


class Workspace(ProtoWorkspace):
    """ Parameter singleton.

    Attributes:

        data_array (array-like): working copy of the parameter vector.

    """
    __slots__ = 'data_array',
    def __init__(self, data_array, *args):
        ProtoWorkspace.__init__(self, args)
        self.data_array = data_array
    def __len__(self):
        return len(self.data_array)


class JobStep(object):
    """ Job step data.

    A job step object contains all the necessary input data for a job step
    to be performed as well as the output data resulting from the step completion.

    A job step object merely contains a reference to a shared workspace.

    The `resource_id` attribute refers to a series of job steps that operate
    on the same subset of resource items.

    Multiple steps can operate simultaneously in the same workspace in a distributed fashion
    provided that they do not compete for the same resources.

    `resources` is an index array that designates the items of shared data to be accessed.
    This attribute is used by :class:`Scheduler` to lock the required items of data,
    which determines which steps can be run simultaneously.
    """
    __slots__ = '_id', '_workspace'
    def __init__(self, _id, workspace=None):
        self._id = _id
        self._workspace = workspace
    def get_workspace(self):
        return self._workspace
    def set_workspace(self, ws):
        self._workspace = ws
    def unset_workspace(self):
        self._workspace = None
    @property
    def workspace_set(self):
        return self._workspace is not None
    @property
    def resource_id(self):
        return self._id
    @property
    def resources(self):
        return self.get_workspace().resources(self)

abc.JobStep.register(JobStep)


class UpdateVehicle(object):
    """ Not instanciable! Introduced for __slots__-enabled multiple inheritance.

    Example usage, in the case class ``B`` implements (abc.) :class:`VehicleJobStep` and
    class ``A`` can only implement (abc.) :class:`JobStep` and not
    inherit from :class:`VehiculeJobStep`::

        class A:
            __slots__ = 'a',
        abc.JobStep.register(A)
        class B(A, UpdateVehicle):
            __slots__ = ('b', ) + VehicleJobStep.__slots__
            def __init__(self, a, b):
                A.__init__(self, a)
                UpdateVehicle.__init__(self)
                self.b = b
        abc.VehicleJobStep.register(B)

    :class:`VehicleJobStep` brings the slots, :class:`UpdateVehicle` brings the implementation
    (methods) and :class:`abc.VehicleJobStep` the typing required by :class:`Workspace` and
    :class:`Worker` to handle ``B`` as a :class:`VehicleJobStep`.
    """
    __slots__ = ()
    def __init__(self):
        self._updates = {}
    def pop_updates(self):
        try:
            return self._updates
        finally:
            self._updates = {}
    def push_updates(self, updates):
        self._updates = updates

class VehicleJobStep(JobStep, UpdateVehicle):
    __slots__ = '_updates',
    def __init__(self, _id, workspace=None):
        JobStep.__init__(self, _id, workspace)
        UpdateVehicle.__init__(self)

abc.VehicleJobStep.register(VehicleJobStep)


__all__ = [ 'StarConn', 'StarQueue', 'ProtoWorkspace', 'Workspace', 'JobStep', 'UpdateVehicle', 'VehicleJobStep', 'Worker', 'Scheduler', 'EpochScheduler', 'abc' ]

