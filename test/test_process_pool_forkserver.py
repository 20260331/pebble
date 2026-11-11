import gc
import os
import sys
import time
import weakref
import pickle
import signal
import asyncio
import unittest
import threading
import concurrent
import dataclasses
import multiprocessing

from unittest import mock
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import CancelledError, TimeoutError

import pebble
from pebble import ProcessPool, ProcessExpired
from pebble.pool.base_pool import PoolStatus


# set start method
supported = False
mp_context = None


methods = multiprocessing.get_all_start_methods()
if 'forkserver' in methods:
    try:
        mp_context = multiprocessing.get_context('forkserver')

        if mp_context.get_start_method() == 'forkserver':
            supported = True
        else:
            raise BaseException(mp_context.get_start_method())
    except RuntimeError:  # child process
        pass


initarg = 0


def initializer(value):
    global initarg
    initarg = value


def long_initializer():
    time.sleep(60)


def broken_initializer():
    raise BaseException("BOOM!")


def function(argument, keyword_argument=0):
    """A docstring."""
    return argument + keyword_argument


def initializer_function():
    return initarg


def error_function():
    raise BaseException("BOOM!")


def return_error_function():
    return BaseException("BOOM!")


def unpickleable_error_function():
    raise UnpickleableException()


class UnpickleableException(Exception):
    """This exception cannot be correctly re-constructed
    on the main side of the pool.
    """
    def __init__(self):
        self.message = "BOOM!"
        super().__init__(self.message)


@dataclasses.dataclass(frozen=True)
class FrozenError(Exception):
    pass


def frozen_error_function():
    raise FrozenError()


def pickle_error_function():
    return threading.Lock()


def long_function(value=1):
    time.sleep(value)
    return value


def lease_queue_function(queue):
    queue.put(1)
    time.sleep(0.3)
    return queue


def pid_function():
    time.sleep(0.1)
    return os.getpid()


def sigterm_function():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(10)


def suicide_function():
    os._exit(1)


def process_function():
    p = multiprocessing.Process(target=function, args=[1])
    p.start()
    p.join()

    return 1


def pool_function():
    signal.signal(signal.SIGTERM, lambda _n, _s: sys.exit(0))
    pool = multiprocessing.Pool(1)
    result = pool.apply(function, args=[1])
    pool.close()
    pool.join()

    return result


def pebble_function():
    signal.signal(signal.SIGTERM, lambda _n, _s: sys.exit(0))
    with ProcessPool(max_workers=1, context=mp_context) as pool:
        f = pool.schedule(function, args=[1])

    return f.result()


def dead_manager_loop(*args, **kwargs):
    """Stand-in for pool_manager_loop that crashes."""
    time.sleep(0.1)


@unittest.skipIf(not supported, "Start method is not supported")
class TestProcessPool(unittest.TestCase):
    def setUp(self):
        global initarg
        initarg = 0
        self.event = threading.Event()
        self.event.clear()
        self.result = None
        self.exception = None

    def callback(self, future):
        try:
            self.result = future.result()
        except BaseException as error:
            self.exception = error
        finally:
            self.event.set()

    def test_process_pool_single_future(self):
        """Process Pool Forkserver single future."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(function, args=[1],
                                   kwargs={'keyword_argument': 1})
        self.assertEqual(future.result(), 2)

    def test_process_pool_multiple_futures(self):
        """Process Pool Forkserver multiple futures."""
        futures = []
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            for _ in range(5):
                futures.append(pool.schedule(function, args=[1]))
        self.assertEqual(sum([f.result() for f in futures]), 5)

    def test_process_pool_callback(self):
        """Process Pool Forkserver result is forwarded to the callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(
                function, args=[1], kwargs={'keyword_argument': 1})
        future.add_done_callback(self.callback)
        self.event.wait()
        self.assertEqual(self.result, 2)

    def test_process_pool_error(self):
        """Process Pool Forkserver errors are raised by future get."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(error_function)
        self.assertRaises(BaseException, future.result)

    def test_process_pool_error_returned(self):
        """Process Pool Forkserver returned errors are returned by future get."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(return_error_function)
        self.assertIsInstance(future.result(), BaseException)

    def test_process_pool_error_callback(self):
        """Process Pool Forkserver errors are forwarded to callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(error_function)
        future.add_done_callback(self.callback)
        self.event.wait()
        self.assertTrue(isinstance(self.exception, BaseException))

    def test_process_pool_pickling_error_task(self):
        """Process Pool Forkserver task pickling errors
        are raised by future.result."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(function, args=[threading.Lock()])
            self.assertRaises((pickle.PicklingError, TypeError), future.result)

    def test_process_pool_pickling_error_result(self):
        """Process Pool Forkserver result pickling errors
        are raised by future.result."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(pickle_error_function)
            self.assertRaises((pickle.PicklingError, TypeError), future.result)

    def test_process_pool_broken_pickling_error(self):
        """Process Pool Forkserver pickling errors breaking the pool
        are raised by future.result."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(unpickleable_error_function)
            self.assertRaises(BrokenProcessPool, future.result)

    def test_process_pool_frozen_error(self):
        """Process Pool Forkserver frozen errors are raised by future get."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(frozen_error_function)
        self.assertRaises(FrozenError, future.result)

    def test_process_pool_timeout(self):
        """Process Pool Forkserver future raises TimeoutError if so."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(long_function, timeout=0.1)
        self.assertRaises(TimeoutError, future.result)

    def test_process_pool_timeout_callback(self):
        """Process Pool Forkserver TimeoutError is forwarded to callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(long_function, timeout=0.1)
        future.add_done_callback(self.callback)
        self.event.wait()
        self.assertTrue(isinstance(self.exception, TimeoutError))

    def test_process_pool_cancel(self):
        """Process Pool Forkserver future raises CancelledError if so."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(long_function)
            time.sleep(0.1)  # let the process pick up the task
            self.assertTrue(future.cancel())
        self.assertRaises(CancelledError, future.result)

    def test_process_pool_cancel_callback(self):
        """Process Pool Forkserver CancelledError is forwarded to callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(long_function)
            future.add_done_callback(self.callback)
            time.sleep(0.1)  # let the process pick up the task
            self.assertTrue(future.cancel())
        self.event.wait()
        self.assertTrue(isinstance(self.exception, CancelledError))

    @unittest.skipIf(sys.platform == 'darwin', "Not supported on MAC OS")
    def test_process_pool_different_process(self):
        """Process Pool Forkserver multiple futures are handled by different processes."""
        futures = []
        with ProcessPool(max_workers=2, context=mp_context) as pool:
            for _ in range(0, 5):
                futures.append(pool.schedule(pid_function))
        self.assertEqual(len(set([f.result() for f in futures])), 2)

    def test_process_pool_future_limit(self):
        """Process Pool Forkserver tasks limit is honored."""
        futures = []
        with ProcessPool(max_workers=1, max_tasks=2, context=mp_context) as pool:
            for _ in range(0, 4):
                futures.append(pool.schedule(pid_function))
        self.assertEqual(len(set([f.result() for f in futures])), 2)

    def test_process_pool_stop_timeout(self):
        """Process Pool Forkserver workers are stopped if future timeout."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future1 = pool.schedule(pid_function)
            pool.schedule(long_function, timeout=0.1)
            future2 = pool.schedule(pid_function)
        self.assertNotEqual(future1.result(), future2.result())

    def test_process_pool_stop_cancel(self):
        """Process Pool Forkserver workers are stopped if future is cancelled."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future1 = pool.schedule(pid_function)
            cancel_future = pool.schedule(long_function)
            time.sleep(0.1)  # let the process pick up the task
            cancel_future.cancel()
            future2 = pool.schedule(pid_function)
        self.assertNotEqual(future1.result(), future2.result())

    def test_process_pool_initializer(self):
        """Process Pool Forkserver initializer is correctly run."""
        with ProcessPool(initializer=initializer, initargs=[1], context=mp_context) as pool:
            future = pool.schedule(initializer_function)
        self.assertEqual(future.result(), 1)

    def test_process_pool_broken_initializer(self):
        """Process Pool Forkserver broken initializer is notified."""
        with self.assertRaises(RuntimeError):
            with ProcessPool(initializer=broken_initializer, context=mp_context) as pool:
                pool.active
                time.sleep(1)
                pool.schedule(function)

    def test_process_pool_broken_worker_launch(self):
        """Process Pool Forkserver is broken if a worker launch hits EOF."""
        launch_process = pebble.pool.process.launch_process
        launched = []

        def launch_once(*args, **kwargs):
            if launched:
                raise EOFError("unexpected EOF")
            launched.append(launch_process(*args, **kwargs))
            return launched[0]

        pool = ProcessPool(max_workers=1, max_tasks=1, context=mp_context)
        try:
            with mock.patch.object(
                    pebble.pool.process, "launch_process", launch_once
            ):
                first = pool.schedule(function, args=[1])
                second = pool.schedule(function, args=[1])
                self.assertEqual(first.result(timeout=5), 1)
                self.assertRaises(BrokenProcessPool, second.result, timeout=5)
            self.assertFalse(pool.active)
        finally:
            pool.stop()
            pool.join()

    def test_process_pool_running(self):
        """Process Pool Forkserver is active if a future is scheduled."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            pool.schedule(function, args=[1])
            self.assertTrue(pool.active)

    def test_process_pool_stopped(self):
        """Process Pool Forkserver is not active once stopped."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            pool.schedule(function, args=[1])
        self.assertFalse(pool.active)

    def test_process_pool_close_futures(self):
        """Process Pool Forkserver all futures are performed on close."""
        futures = []
        pool = ProcessPool(max_workers=1, context=mp_context)
        for index in range(10):
            futures.append(pool.schedule(function, args=[index]))
        pool.close()
        pool.join()
        map(self.assertTrue, [f.done() for f in futures])

    def test_process_pool_close_stopped(self):
        """Process Pool Forkserver is stopped after close."""
        pool = ProcessPool(max_workers=1, context=mp_context)
        pool.schedule(function, args=[1])
        pool.close()
        pool.join()
        self.assertFalse(pool.active)

    def test_process_pool_stop_futures(self):
        """Process Pool Forkserver not all futures are performed on stop."""
        futures = []
        pool = ProcessPool(max_workers=1, context=mp_context)
        for index in range(10):
            futures.append(pool.schedule(function, args=[index]))
        pool.stop()
        pool.join()
        self.assertTrue(len([f for f in futures if not f.done()]) > 0)

    def test_process_pool_stop_stopped(self):
        """Process Pool Forkserver is stopped after stop."""
        pool = ProcessPool(max_workers=1, context=mp_context)
        pool.schedule(function, args=[1])
        pool.stop()
        pool.join()
        self.assertFalse(pool.active)

    def test_process_pool_stop_stopped_callback(self):
        """Process Pool Forkserver is stopped in callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            def stop_pool_callback(_):
                pool.stop()

            future = pool.schedule(function, args=[1])
            future.add_done_callback(stop_pool_callback)
            with self.assertRaises(RuntimeError):
                for index in range(10):
                    time.sleep(0.1)
                    pool.schedule(long_function, args=[index])

        self.assertFalse(pool.active)

    def test_process_pool_large_data(self):
        """Process Pool Forkserver large data is sent on the channel."""
        data = "a" * 1098 * 1024 * 100  # 100 Mb

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(
                function, args=[data], kwargs={'keyword_argument': ''})

        self.assertEqual(data, future.result())

    def test_process_pool_stop_large_data(self):
        """Process Pool Forkserver is stopped if large data is sent on the channel."""
        data = "a" * 1098 * 1024 * 100  # 100 Mb
        pool = ProcessPool(max_workers=1, context=mp_context)
        pool.schedule(function, args=[data])
        time.sleep(1)
        pool.stop()
        pool.join()

        self.assertFalse(pool.active)

    def test_process_pool_join_workers(self):
        """Process Pool Forkserver no worker is running after join."""
        pool = ProcessPool(max_workers=4, context=mp_context)
        pool.schedule(function, args=[1])
        pool.stop()
        pool.join()
        self.assertEqual(len(pool._pool_manager.worker_manager.workers), 0)

    def test_process_pool_join_running(self):
        """Process Pool Forkserver RuntimeError is raised if active pool joined."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            pool.schedule(function, args=[1])
            self.assertRaises(RuntimeError, pool.join)

    def test_process_pool_join_futures_timeout(self):
        """Process Pool Forkserver TimeoutError is raised if join on long tasks."""
        pool = ProcessPool(max_workers=1, context=mp_context)
        for _ in range(2):
            pool.schedule(long_function)
        pool.close()
        self.assertRaises(TimeoutError, pool.join, 0.4)
        pool.stop()
        pool.join()

    def test_process_pool_callback_error(self):
        """Process Pool Forkserver does not stop if error in callback."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(function, args=[1],
                                   kwargs={'keyword_argument': 1})
            future.add_done_callback(self.callback)
            # sleep enough to ensure callback is run
            time.sleep(0.1)
            pool.schedule(function, args=[1],
                          kwargs={'keyword_argument': 1})

    def test_process_pool_exception_isolated(self):
        """Process Pool Forkserver an BaseException does not affect other futures."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(error_function)
            try:
                future.result()
            except BaseException:
                pass
            future = pool.schedule(function, args=[1],
                                   kwargs={'keyword_argument': 1})
        self.assertEqual(future.result(), 2)

    @unittest.skipIf(os.name == 'nt', "Test won't run on Windows'.")
    def test_process_pool_ignoring_sigterm(self):
        """Process Pool Forkserver ignored SIGTERM signal are handled on Unix."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(sigterm_function, timeout=0.2)
            with self.assertRaises(TimeoutError):
                future.result()

    def test_process_pool_expired_worker(self):
        """Process Pool Forkserver unexpect death of worker raises ProcessExpired."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(suicide_function)
            worker_pid = list(pool._pool_manager.worker_manager.workers)[0]
            with self.assertRaises(ProcessExpired) as exc_ctx:
                future.result()
            self.assertEqual(exc_ctx.exception.exitcode, 1)
            self.assertEqual(exc_ctx.exception.pid, worker_pid)

    def test_process_pool_map(self):
        """Process Pool Forkserver map simple."""
        elements = [1, 2, 3]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_process_pool_map_empty(self):
        """Process Pool Forkserver map no elements."""
        elements = []

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_process_pool_map_single(self):
        """Process Pool Forkserver map one element."""
        elements = [0]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_process_pool_map_multi(self):
        """Process Pool Forkserver map multiple iterables."""
        expected = (2, 4)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, (1, 2, 3), (1, 2))
            generator = future.result()
            self.assertEqual(tuple(generator), expected)

    def test_process_pool_map_one_chunk(self):
        """Process Pool Forkserver map chunksize 1."""
        elements = [1, 2, 3]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, elements, chunksize=1)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_process_pool_map_zero_chunk(self):
        """Process Pool Forkserver map chunksize 0."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            with self.assertRaises(ValueError):
                pool.map(function, [], chunksize=0)

    def test_process_pool_map_timeout(self):
        """Process Pool Forkserver map with timeout."""
        raised = []
        elements = [1, 2, 3]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(long_function, elements, timeout=0.1)
            generator = future.result()
            while True:
                try:
                    next(generator)
                except TimeoutError as error:
                    raised.append(error)
                except StopIteration:
                    break

        self.assertTrue(all((isinstance(e, TimeoutError) for e in raised)))

    def test_process_pool_map_timeout_chunks(self):
        """Process Pool Forkserver map timeout is assigned per chunk."""
        elements = [0.1]*20

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            # it takes 1s to process a chunk
            future = pool.map(
                long_function, elements, chunksize=5, timeout=1.8)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_process_pool_map_error(self):
        """Process Pool Forkserver errors do not stop the iteration."""
        raised = None
        elements = [1, 'a', 3]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            while True:
                try:
                    next(generator)
                except TypeError as error:
                    raised = error
                except StopIteration:
                    break

        self.assertTrue(isinstance(raised, TypeError))

    def test_process_pool_map_cancel(self):
        """Process Pool Forkserver cancel iteration."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(long_function, range(5))
            generator = future.result()

            self.assertEqual(next(generator), 0)

            future.cancel()

            for _ in range(4):
                with self.assertRaises(CancelledError):
                    next(generator)

    def test_process_pool_map_broken_pool(self):
        """Process Pool Forkserver Broken Pool."""
        elements = [1, 2, 3]

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(long_function, elements, timeout=1)
            generator = future.result()
            pool._context.status = PoolStatus.ERROR
            while True:
                try:
                    next(generator)
                except (ProcessExpired, TimeoutError):
                    self.assertFalse(pool.active)
                    future.cancel()
                    break
                except StopIteration:
                    break

    def test_process_pool_map_lease(self):
        """Process Pool Forkserver map lease releases the buffers once consumed."""
        elements = list(range(4))

        with ProcessPool(max_workers=2, context=mp_context) as pool:
            future = pool.map(function, elements, lease=1)
            generator = future.result()

            self.assertTrue(future.lease.active)
            self.assertEqual(list(generator), elements)
            self.assertTrue(future.lease.released)
            self.assertEqual(future.futures, ())
            self.assertIsNone(generator._results)

    def test_process_pool_map_lease_renew(self):
        """Process Pool Forkserver map lease can be renewed while consuming."""
        with ProcessPool(max_workers=2, context=mp_context) as pool:
            future = pool.map(function, range(5), lease=0.3)
            generator = future.result()
            first_deadline = future.renew_lease()
            time.sleep(0.2)
            second_deadline = generator.renew_lease()
            self.assertGreater(second_deadline, first_deadline)
            time.sleep(0.2)
            self.assertFalse(future.lease.released)
            self.assertEqual(list(generator), list(range(5)))

    def test_process_pool_map_lease_renew_timeout(self):
        """Process Pool Forkserver lease renewal accepts a new timeout."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, range(2), lease=10)
            generator = future.result()
            deadline = generator.renew_lease(0.2)
            self.assertLess(deadline - time.monotonic(), 0.25)
            time.sleep(0.4)
            with self.assertRaises(CancelledError):
                next(generator)
            self.assertTrue(future.lease.released)

    def test_process_pool_map_lease_expiration(self):
        """Process Pool Forkserver lease expiration cancels the batch and buffers."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(long_function, [1] * 5, lease=0.2)
            chunk_futures = list(future.futures)
            generator = future.result()

            self.assertTrue(any(f.cancelled() for f in chunk_futures))
            with self.assertRaises(CancelledError):
                next(generator)
            self.assertIsNone(generator._results)
            self.assertEqual(future.futures, ())
            self.assertTrue(future.lease.released)
            time.sleep(0.3)
            self.assertEqual(pool._context.leases._entries, [])

    def test_process_pool_map_lease_close(self):
        """Process Pool Forkserver abandoning the iterator cancels the batch."""
        with ProcessPool(max_workers=2, context=mp_context) as pool:
            future = pool.map(function, range(6), lease=5)
            generator = future.result()
            self.assertEqual(next(generator), 0)
            generator.close()
            with self.assertRaises(CancelledError):
                next(generator)
            self.assertIsNone(generator._results)
            self.assertEqual(future.futures, ())
            self.assertTrue(future.lease.released)
            time.sleep(0.3)
            self.assertEqual(pool._context.leases._entries, [])

    def test_process_pool_map_lease_cancel(self):
        """Process Pool Forkserver cancelling a leased map releases the lease."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(long_function, [1] * 5, lease=5)
            chunk_futures = list(future.futures)

            def abandon():
                time.sleep(0.2)
                future.cancel()

            thread = threading.Thread(target=abandon)
            thread.start()
            generator = future.result()
            thread.join()
            self.assertTrue(any(f.cancelled() for f in chunk_futures))
            with self.assertRaises(CancelledError):
                next(generator)
            self.assertTrue(future.lease.released)

    def test_process_pool_map_lease_context_manager(self):
        """Process Pool Forkserver map results support the context manager protocol."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.map(function, range(4), lease=5)
            with future.result() as generator:
                self.assertEqual(next(generator), 0)
            self.assertTrue(future.lease.released)
            with self.assertRaises(CancelledError):
                next(generator)

    def test_process_pool_map_lease_isolation(self):
        """Process Pool Forkserver leases of different maps do not affect each other."""
        with ProcessPool(max_workers=2, context=mp_context) as pool:
            abandoned = pool.map(function, range(10), lease=0.2)
            active = pool.map(function, range(5), lease=5)
            abandoned.result()
            time.sleep(0.5)
            self.assertTrue(abandoned.lease.released)
            self.assertFalse(active.lease.released)
            entries = pool._context.leases._entries
            self.assertEqual(len(entries), 1)
            self.assertIs(entries[0][0], active.lease)
            generator = active.result()
            active.renew_lease()
            self.assertEqual(list(generator), list(range(5)))

    def test_process_pool_map_lease_garbage_collection(self):
        """Process Pool Forkserver abandoned leased maps are collected and cancelled."""
        manager = mp_context.Manager()
        queue = manager.Queue()
        try:
            with ProcessPool(max_workers=1, context=mp_context) as pool:
                future = pool.map(
                    lease_queue_function, [queue] * 8, lease=10
                )
                reference = weakref.ref(future)
                del future
                gc.collect()
                self.assertIsNone(reference())
                deadline = time.time() + 3
                while time.time() < deadline and pool._context.leases._entries:
                    time.sleep(0.05)
                self.assertEqual(pool._context.leases._entries, [])
                self.assertLessEqual(queue.qsize(), 1)
        finally:
            manager.shutdown()

    def test_process_pool_map_lease_invalid(self):
        """Process Pool Forkserver map lease parameter is validated."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            with self.assertRaises(ValueError):
                pool.map(function, [1], lease=0)
            with self.assertRaises(TypeError):
                pool.map(function, [1], lease="1")
            future = pool.map(function, [1])
            with self.assertRaises(RuntimeError):
                future.renew_lease()

    def test_process_pool_map_lease_shutdown(self):
        """Process Pool Forkserver shutdown releases the map leases."""
        pool = ProcessPool(max_workers=1, context=mp_context)
        future = pool.map(function, range(3), lease=5)
        pool.close()
        pool.join()
        self.assertTrue(future.lease.released)
        self.assertEqual(future.futures, ())

    def test_process_pool_child_process(self):
        """Process Pool Forkserver worker starts process."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(process_function)
        self.assertEqual(future.result(), 1)

    def test_process_pool_child_pool(self):
        """Process Pool Forkserver worker starts multiprocessing.Pool."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(pool_function)
        self.assertEqual(future.result(), 1)

    def test_process_pool_child_pebble(self):
        """Process Pool Forkserver worker starts pebble.ProcessPool."""
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            future = pool.schedule(pebble_function)
        self.assertEqual(future.result(), 1)

    def test_process_pool_pickle_function(self):
        """Process Pool Forkserver picklable functions."""
        queue = mp_context.Manager().Queue()
        with ProcessPool(max_workers=1, context=mp_context) as pool:
            pool.schedule(queue.put, args=[1])
        self.assertEqual(queue.get(timeout=1), 1)

    def test_process_pool_active_error(self):
        """Process Pool Forkserver is not active if its manager thread dies."""
        with mock.patch(
            "pebble.pool.process.pool_manager_loop", dead_manager_loop
        ):
            with ProcessPool(max_workers=1, context=mp_context) as pool:
                self.assertTrue(pool.active)  # start the pool
                time.sleep(0.3)               # wait for manager to die
                self.assertFalse(pool.active)

    def test_process_pool_orphan_tasks(self):
        """Process Pool Fork orphan tasks are handled."""
        def long_callback(_):
            time.sleep(1)

        with ProcessPool(max_workers=2, context=mp_context) as pool:
            f1 = pool.schedule(long_function, args=[0.3])
            f2 = pool.schedule(long_function, args=[0.01])
            f2.add_done_callback(long_callback)
            pool.schedule(long_function, args=[1])
            f3 = pool.schedule(long_function, args=[1.5])

            while not f1.running():
                time.sleep(0.001)
            time.sleep(0.5)

            self.assertTrue(f1.cancel())
            with self.assertRaises(ProcessExpired):
                f3.result(timeout=3)


@unittest.skipIf(not supported, "Start method is not supported")
class TestAsyncIOProcessPool(unittest.TestCase):
    def setUp(self):
        self.event = None
        self.result = None
        self.exception = None

    def callback(self, future):
        try:
            self.result = future.result()
        # asyncio.exception.CancelledError does not inherit from BaseException
        except BaseException as error:
            self.exception = error
        finally:
            self.event.set()

    def test_process_pool_single_future(self):
        """Process Pool Forkserver single future."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, function, None, 1)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            self.assertEqual(asyncio.run(test(pool)), 1)

    def test_process_pool_multiple_futures(self):
        """Process Pool Forkserver multiple futures."""
        async def test(pool):
            futures = []
            loop = asyncio.get_running_loop()

            for _ in range(5):
                futures.append(loop.run_in_executor(pool, function, None, 1))

            return await asyncio.wait(futures)

        with ProcessPool(max_workers=2, context=mp_context) as pool:
            self.assertEqual(sum(r.result()
                                 for r in asyncio.run(test(pool))[0]), 5)

    def test_process_pool_callback(self):
        """Process Pool Forkserver result is forwarded to the callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, function, None, 1)
            future.add_done_callback(self.callback)

            await self.event.wait()

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))
            self.assertEqual(self.result, 1)

    def test_process_pool_error(self):
        """Process Pool Forkserver errors are raised by future get."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, error_function, None)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            with self.assertRaises(BaseException):
                asyncio.run(test(pool))

    def test_process_pool_error_returned(self):
        """Process Pool Forkserver returned errors are returned by future get."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, return_error_function, None)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            self.assertIsInstance(asyncio.run(test(pool)), BaseException)

    def test_process_pool_error_callback(self):
        """Process Pool Forkserver errors are forwarded to callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, error_function, None)
            future.add_done_callback(self.callback)

            await self.event.wait()

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))
            self.assertTrue(isinstance(self.exception, BaseException))

    def test_process_pool_timeout(self):
        """Process Pool Forkserver future raises TimeoutError if so."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, long_function, 0.1)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            with self.assertRaises(asyncio.TimeoutError):
                asyncio.run(test(pool))

    def test_process_pool_timeout_callback(self):
        """Process Pool Forkserver TimeoutError is forwarded to callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, long_function, 0.1)
            future.add_done_callback(self.callback)

            await asyncio.sleep(0.1) # let the process pick up the task

            await self.event.wait()

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))
            self.assertTrue(isinstance(self.exception, asyncio.TimeoutError))

    def test_process_pool_cancel(self):
        """Process Pool Forkserver future raises CancelledError if so."""
        async def test(pool):
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(pool, long_function, None)

            await asyncio.sleep(0.1) # let the process pick up the task

            self.assertTrue(future.cancel())

            return await future

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(test(pool))

    def test_process_pool_cancel_callback(self):
        """Process Pool Forkserver CancelledError is forwarded to callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, long_function, None)
            future.add_done_callback(self.callback)

            await asyncio.sleep(0.1) # let the process pick up the task

            self.assertTrue(future.cancel())

            await self.event.wait()

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))
            self.assertTrue(isinstance(self.exception, asyncio.CancelledError))

    def test_process_pool_stop_timeout(self):
        """Process Pool Forkserver workers are stopped if future timeout."""
        async def test(pool):
            loop = asyncio.get_running_loop()
            future1 = loop.run_in_executor(pool, pid_function, None)
            with self.assertRaises(asyncio.TimeoutError):
                await loop.run_in_executor(pool, long_function, 0.1)
            future2 = loop.run_in_executor(pool, pid_function, None)

            self.assertNotEqual(await future1, await future2)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))

    def test_process_pool_stop_cancel(self):
        """Process Pool Forkserver workers are stopped if future is cancelled."""
        async def test(pool):
            loop = asyncio.get_running_loop()
            future1 = loop.run_in_executor(pool, pid_function, None)
            cancel_future = loop.run_in_executor(pool, long_function, None)
            await asyncio.sleep(0.1)  # let the process pick up the task
            self.assertTrue(cancel_future.cancel())
            future2 = loop.run_in_executor(pool, pid_function, None)

            self.assertNotEqual(await future1, await future2)

        with ProcessPool(max_workers=1, context=mp_context) as pool:
            asyncio.run(test(pool))


# DEADLOCK TESTS
def install_broken_process_task_callback():
    """Since spawn does not inherit from parent,
    we need to patch Pebble in the child process.

    """
    pebble.pool.process.send_result = broken_worker_process_tasks
    pebble.CONSTS.channel_lock_timeout = 0.1


def broken_worker_process_tasks(channel, *_):
    """Process failing in receiving new tasks."""
    with channel.mutex.reader:
        os._exit(1)


def install_broken_process_result_callback():
    """Since spawn does not inherit from parent,
    we need to patch Pebble in the child process.

    """
    pebble.pool.process.send_result = broken_worker_process_result
    pebble.CONSTS.channel_lock_timeout = 0.1


def broken_worker_process_result(channel, *_):
    """Process failing in delivering result."""
    with channel.mutex.writer:
        os._exit(1)


@unittest.skipIf(not supported, "Start method not supported")
class TestProcessPoolDeadlockOnNewFutures(unittest.TestCase):
    def test_pool_deadlock_stop(self):
        """Process Pool ForkServer reading deadlocks are stopping the Pool."""
        with pebble.ProcessPool(max_workers=1,
                                initializer=install_broken_process_task_callback,
                                context=mp_context) as pool:
            # First task crashes the worker
            f = pool.schedule(function, args=[1])
            self.assertRaises(ProcessExpired, f.result)
            time.sleep(0.1)
            # Second task acknowledges the pool to be broken
            f = pool.schedule(function, args=[1])
            self.assertRaises(BrokenProcessPool, f.result)
            time.sleep(0.1)
            # Third task cannot be scheduled since the pool is broken
            with self.assertRaisesRegex(RuntimeError,
                                        "Unexpected error within the Pool"):
                pool.schedule(function, args=[1])


@unittest.skipIf(not supported or os.name == 'nt', "Start method/OS not supported")
class TestProcessPoolDeadlockOnResult(unittest.TestCase):
    def test_pool_deadlock(self):
        """Process Pool ForkServer no deadlock if writing worker dies locking channel."""
        with pebble.ProcessPool(max_workers=1,
                                initializer=install_broken_process_result_callback,
                                context=mp_context) as pool:
            # First task crashes the worker
            f = pool.schedule(function, args=[1])
            self.assertRaises(ProcessExpired, f.result)
            time.sleep(0.1)
            # Second task acknowledges the pool to be broken
            f = pool.schedule(function, args=[1])
            self.assertRaises(BrokenProcessPool, f.result)
            time.sleep(0.1)
            # Third task cannot be scheduled since the pool is broken
            with self.assertRaisesRegex(RuntimeError,
                                        "Unexpected error within the Pool"):
                pool.schedule(function, args=[1])


@unittest.skipIf(not supported, "Start method not supported")
class TestProcessPoolDeadlockOnCancelLargeData(unittest.TestCase):
    def test_pool_deadlock_stop_cancel(self):
        """Process Pool Forkserver is stopped when futures are cancelled on large data."""
        data = b'A' * 1024 * 1024 * 100

        with pebble.ProcessPool(context=mp_context) as pool:
            futures = [pool.schedule(function, args=[data]) for _ in range(10)]
            concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for f in futures:
                f.cancel()
            pool.stop()
