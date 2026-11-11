import gc
import time
import weakref
import asyncio
import unittest
import threading
import dataclasses

from unittest import mock
from concurrent.futures import CancelledError, TimeoutError

from pebble import ThreadPool
from pebble.pool.base_pool import PoolStatus

initarg = 0


def error_callback(future):
    raise BaseException("BOOM!")


def initializer(value):
    global initarg
    initarg = value


def broken_initializer():
    raise BaseException("BOOM!")


def function(argument, keyword_argument=0):
    """A docstring."""
    return argument + keyword_argument


def initializer_function():
    return initarg


def error_function():
    raise BaseException("BOOM!")


@dataclasses.dataclass(frozen=True)
class FrozenError(Exception):
    pass


def frozen_error_function():
    raise FrozenError()


def long_function(value=0):
    time.sleep(1)
    return value


def tid_function():
    time.sleep(0.1)
    return threading.current_thread()


def dead_manager_loop(*args, **kwargs):
    """Stand-in for pool_manager_loop that crashes."""
    time.sleep(0.1)


class TestThreadPool(unittest.TestCase):
    def setUp(self):
        global initarg
        initarg = 0
        self.event = threading.Event()
        self.event.clear()
        self.results = None
        self.exception = None


    def callback(self, future):
        try:
            self.results = future.result()
        except BaseException as error:
            self.exception = error
        finally:
            self.event.set()

    def test_thread_pool_single_future(self):
        """Thread Pool single future."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(function, args=[1],
                                   kwargs={'keyword_argument': 1})
        self.assertEqual(future.result(), 2)

    def test_thread_pool_multiple_futures(self):
        """Thread Pool multiple futures."""
        futures = []
        with ThreadPool(max_workers=1) as pool:
            for _ in range(5):
                futures.append(pool.schedule(function, args=[1]))
        self.assertEqual(sum([t.result() for t in futures]), 5)

    def test_thread_pool_callback(self):
        """Thread Pool results are forwarded to the callback."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(
                function, args=[1], kwargs={'keyword_argument': 1})
            future.add_done_callback(self.callback)

        self.event.wait()
        self.assertEqual(self.results, 2)

    def test_thread_pool_error(self):
        """Thread Pool errors are raised by future get."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(error_function)
        with self.assertRaises(BaseException):
            future.result()

    def test_thread_pool_error_callback(self):
        """Thread Pool errors are forwarded to callback."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(error_function)
            future.add_done_callback(self.callback)
        self.event.wait()
        self.assertTrue(isinstance(self.exception, BaseException))

    def test_process_pool_frozen_error(self):
        """Thread Pool frozen errors are raised by future get."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(frozen_error_function)
        self.assertRaises(FrozenError, future.result)

    def test_thread_pool_cancel_callback(self):
        """Thread Pool FutureCancelled is forwarded to callback."""
        with ThreadPool(max_workers=1) as pool:
            pool.schedule(long_function)
            future = pool.schedule(long_function)
            future.add_done_callback(self.callback)
            future.cancel()
        self.event.wait()

        self.assertTrue(isinstance(self.exception, CancelledError))

    def test_thread_pool_different_thread(self):
        """Thread Pool multiple futures are handled by different threades."""
        futures = []
        with ThreadPool(max_workers=2) as pool:
            for _ in range(0, 5):
                futures.append(pool.schedule(tid_function))
        self.assertEqual(len(set([t.result() for t in futures])), 2)

    def test_thread_pool_tasks_limit(self):
        """Thread Pool future limit is honored."""
        futures = []
        with ThreadPool(max_workers=1, max_tasks=2) as pool:
            for _ in range(0, 4):
                futures.append(pool.schedule(tid_function))
        self.assertEqual(len(set([t.result() for t in futures])), 2)

    def test_thread_pool_initializer(self):
        """Thread Pool initializer is correctly run."""
        with ThreadPool(initializer=initializer, initargs=[1]) as pool:
            future = pool.schedule(initializer_function)
        self.assertEqual(future.result(), 1)

    def test_thread_pool_broken_initializer(self):
        """Thread Pool broken initializer is notified."""
        with self.assertRaises(RuntimeError):
            with ThreadPool(initializer=broken_initializer) as pool:
                pool.active
                time.sleep(0.3)
                pool.schedule(function)

    def test_thread_pool_running(self):
        """Thread Pool is active if a future is scheduled."""
        with ThreadPool(max_workers=1) as pool:
            pool.schedule(function, args=[1])
            self.assertTrue(pool.active)

    def test_thread_pool_stopped(self):
        """Thread Pool is not active once stopped."""
        with ThreadPool(max_workers=1) as pool:
            pool.schedule(function, args=[1])
        self.assertFalse(pool.active)

    def test_thread_pool_close_futures(self):
        """Thread Pool all futures are performed on close."""
        futures = []
        pool = ThreadPool(max_workers=1)
        for index in range(10):
            futures.append(pool.schedule(function, args=[index]))
        pool.close()
        pool.join()
        map(self.assertTrue, [t.done() for t in futures])

    def test_thread_pool_close_stopped(self):
        """Thread Pool is stopped after close."""
        pool = ThreadPool(max_workers=1)
        pool.schedule(function, args=[1])
        pool.close()
        pool.join()
        self.assertFalse(pool.active)

    def test_thread_pool_stop_futures(self):
        """Thread Pool not all futures are performed on stop."""
        futures = []
        pool = ThreadPool(max_workers=1)
        for index in range(10):
            futures.append(pool.schedule(long_function, args=[index]))
        pool.stop()
        pool.join()
        self.assertTrue(len([t for t in futures if not t.done()]) > 0)

    def test_thread_pool_stop_stopped(self):
        """Thread Pool is stopped after stop."""
        pool = ThreadPool(max_workers=1)
        pool.schedule(function, args=[1])
        pool.stop()
        pool.join()
        self.assertFalse(pool.active)

    def test_thread_pool_stop_stopped_function(self):
        """Thread Pool is stopped in function."""
        with ThreadPool(max_workers=1) as pool:
            def function():
                pool.stop()

            pool.schedule(function)

        self.assertFalse(pool.active)

    def test_thread_pool_stop_stopped_callback(self):
        """Thread Pool is stopped in callback."""
        with ThreadPool(max_workers=1) as pool:
            def stop_pool_callback(_):
                pool.stop()

            future = pool.schedule(function, args=[1])
            future.add_done_callback(stop_pool_callback)
            with self.assertRaises(RuntimeError):
                for index in range(10):
                    time.sleep(0.1)
                    pool.schedule(long_function, args=[index])

        self.assertFalse(pool.active)

    def test_thread_pool_join_workers(self):
        """Thread Pool no worker is running after join."""
        pool = ThreadPool(max_workers=4)
        pool.schedule(function, args=[1])
        pool.stop()
        pool.join()
        self.assertEqual(len(pool._pool_manager.workers), 0)

    def test_thread_pool_join_running(self):
        """Thread Pool RuntimeError is raised if active pool joined."""
        with ThreadPool(max_workers=1) as pool:
            pool.schedule(function, args=[1])
            self.assertRaises(RuntimeError, pool.join)

    def test_thread_pool_join_futures_timeout(self):
        """Thread Pool TimeoutError is raised if join on long futures."""
        pool = ThreadPool(max_workers=1)
        for _ in range(2):
            pool.schedule(long_function)
        pool.close()
        self.assertRaises(TimeoutError, pool.join, 0.4)
        pool.stop()
        pool.join()

    def test_thread_pool_exception_isolated(self):
        """Thread Pool an BaseException does not affect other futures."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.schedule(error_function)
            try:
                future.result()
            except:
                pass
            future = pool.schedule(function, args=[1],
                                   kwargs={'keyword_argument': 1})
        self.assertEqual(future.result(), 2)

    def test_thread_pool_map(self):
        """Thread Pool map simple."""
        elements = [1, 2, 3]

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_thread_pool_map_empty(self):
        """Thread Pool map no elements."""
        elements = []

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_thread_pool_map_single(self):
        """Thread Pool map one element."""
        elements = [0]

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, elements)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_thread_pool_map_multi(self):
        """Thread Pool map multiple iterables."""
        expected = (2, 4)

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, (1, 2, 3), (1, 2))
            generator = future.result()
            self.assertEqual(tuple(generator), expected)

    def test_thread_pool_map_one_chunk(self):
        """Thread Pool map chunksize 1."""
        elements = [1, 2, 3]

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, elements, chunksize=1)
            generator = future.result()
            self.assertEqual(list(generator), elements)

    def test_thread_pool_map_zero_chunk(self):
        """Thread Pool map chunksize 0."""
        with ThreadPool(max_workers=1) as pool:
            with self.assertRaises(ValueError):
                pool.map(function, [], chunksize=0)

    def test_thread_pool_map_error(self):
        """Thread Pool errors do not stop the iteration."""
        raised = None
        elements = [1, 'a', 3]

        with ThreadPool(max_workers=1) as pool:
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

    def test_thread_pool_map_cancel(self):
        """Thread Pool cancel iteration."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.map(long_function, range(5))
            generator = future.result()

            self.assertEqual(next(generator), 0)

            future.cancel()

            # either gets computed or it gets cancelled
            try:
                self.assertEqual(next(generator), 1)
            except CancelledError:
                pass

            for _ in range(3):
                with self.assertRaises(CancelledError):
                    next(generator)

    def test_thread_pool_map_broken_pool(self):
        """Thread Pool Fork Broken Pool."""
        elements = [1, 2, 3]

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(long_function, elements, timeout=1)
            generator = future.result()
            pool._context.status = PoolStatus.ERROR
            while True:
                try:
                    next(generator)
                except TimeoutError as error:
                    self.assertFalse(pool.active)
                    future.cancel()
                    break
                except StopIteration:
                    break

    def test_thread_pool_map_lease(self):
        """Thread Pool map lease releases the buffers once consumed."""
        elements = list(range(4))

        with ThreadPool(max_workers=2) as pool:
            future = pool.map(function, elements, lease=1)
            generator = future.result()

            self.assertTrue(future.lease.active)
            self.assertEqual(list(generator), elements)
            self.assertTrue(future.lease.released)
            self.assertEqual(future.futures, ())
            self.assertIsNone(generator._results)

    def test_thread_pool_map_lease_renew(self):
        """Thread Pool map lease can be renewed while consuming."""
        with ThreadPool(max_workers=2) as pool:
            future = pool.map(function, range(5), lease=0.3)
            generator = future.result()
            first_deadline = future.renew_lease()
            time.sleep(0.2)
            second_deadline = generator.renew_lease()
            self.assertGreater(second_deadline, first_deadline)
            time.sleep(0.2)
            self.assertFalse(future.lease.released)
            self.assertEqual(list(generator), list(range(5)))

    def test_thread_pool_map_lease_renew_timeout(self):
        """Thread Pool lease renewal accepts a new timeout."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, range(2), lease=10)
            generator = future.result()
            deadline = generator.renew_lease(0.2)
            self.assertLess(deadline - time.monotonic(), 0.25)
            time.sleep(0.4)
            with self.assertRaises(CancelledError):
                next(generator)
            self.assertTrue(future.lease.released)

    def test_thread_pool_map_lease_expiration(self):
        """Thread Pool lease expiration cancels pending tasks and buffers."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.map(long_function, range(5), lease=0.2)
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

    def test_thread_pool_map_lease_close(self):
        """Thread Pool abandoning the iterator cancels the batch."""
        with ThreadPool(max_workers=2) as pool:
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

    def test_thread_pool_map_lease_cancel(self):
        """Thread Pool cancelling a leased map releases the lease."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.map(long_function, range(5), lease=5)
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

    def test_thread_pool_map_lease_context_manager(self):
        """Thread Pool map results support the context manager protocol."""
        with ThreadPool(max_workers=1) as pool:
            future = pool.map(function, range(4), lease=5)
            with future.result() as generator:
                self.assertEqual(next(generator), 0)
            self.assertTrue(future.lease.released)
            with self.assertRaises(CancelledError):
                next(generator)

    def test_thread_pool_map_lease_isolation(self):
        """Thread Pool leases of different maps do not affect each other."""
        with ThreadPool(max_workers=2) as pool:
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

    def test_thread_pool_map_lease_garbage_collection(self):
        """Thread Pool abandoned leased maps are collected and cancelled."""
        executed = []

        def recorder(value):
            executed.append(value)
            time.sleep(0.3)
            return value

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(recorder, range(8), lease=10)
            reference = weakref.ref(future)
            del future
            gc.collect()
            self.assertIsNone(reference())
            deadline = time.time() + 3
            while time.time() < deadline and pool._context.leases._entries:
                time.sleep(0.05)
            self.assertEqual(pool._context.leases._entries, [])
            self.assertLessEqual(len(executed), 1)

    def test_thread_pool_map_lease_invalid(self):
        """Thread Pool map lease parameter is validated."""
        with ThreadPool(max_workers=1) as pool:
            with self.assertRaises(ValueError):
                pool.map(function, [1], lease=0)
            with self.assertRaises(TypeError):
                pool.map(function, [1], lease="1")
            future = pool.map(function, [1])
            with self.assertRaises(RuntimeError):
                future.renew_lease()

    def test_thread_pool_map_lease_shutdown(self):
        """Thread Pool shutdown releases the map leases."""
        pool = ThreadPool(max_workers=1)
        future = pool.map(function, range(3), lease=5)
        pool.close()
        pool.join()
        self.assertTrue(future.lease.released)
        self.assertEqual(future.futures, ())

    def test_thread_pool_active_error(self):
        """Thread Pool is not active if its manager thread dies."""
        with unittest.mock.patch(
            "pebble.pool.thread.pool_manager_loop", dead_manager_loop
        ):
            with ThreadPool(max_workers=1) as pool:
                self.assertTrue(pool.active)  # start the pool
                time.sleep(0.3)               # wait for manager to die
                self.assertFalse(pool.active)


class TestAsyncIOThreadPool(unittest.TestCase):
    def setUp(self):
        global initarg
        initarg = 0
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

    def test_thread_pool_single_future(self):
        """Thread Pool single future."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, function, 1)

        with ThreadPool(max_workers=1) as pool:
            self.assertEqual(asyncio.run(test(pool)), 1)

    def test_thread_pool_multiple_futures(self):
        """Thread Pool multiple futures."""
        async def test(pool):
            futures = []
            loop = asyncio.get_running_loop()

            for _ in range(5):
                futures.append(loop.run_in_executor(pool, function, 1))

            return await asyncio.wait(futures)

        with ThreadPool(max_workers=2) as pool:
            self.assertEqual(sum(r.result()
                                 for r in asyncio.run(test(pool))[0]), 5)

    def test_thread_pool_callback(self):
        """Thread Pool results are forwarded to the callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, function, 1)
            future.add_done_callback(self.callback)

            await self.event.wait()

        with ThreadPool(max_workers=1) as pool:
            asyncio.run(test(pool))
            self.assertEqual(self.result, 1)

    def test_thread_pool_error(self):
        """Thread Pool errors are raised by future get."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(pool, error_function)

        with ThreadPool(max_workers=1) as pool:
            with self.assertRaises(BaseException):
                asyncio.run(test(pool))

    def test_thread_pool_error_callback(self):
        """Thread Pool errors are forwarded to callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, error_function)
            future.add_done_callback(self.callback)

            await self.event.wait()

        with ThreadPool(max_workers=1) as pool:
            asyncio.run(test(pool))
            self.assertTrue(isinstance(self.exception, BaseException))

    def test_thread_pool_cancel_callback(self):
        """Thread Pool FutureCancelled is forwarded to callback."""
        async def test(pool):
            loop = asyncio.get_running_loop()

            self.event = asyncio.Event()
            self.event.clear()

            future = loop.run_in_executor(pool, long_function)
            future.add_done_callback(self.callback)

            await asyncio.sleep(0.1) # let the process pick up the task

            self.assertTrue(future.cancel())

            await self.event.wait()

        with ThreadPool(max_workers=1) as pool:
            asyncio.run(test(pool))
            self.assertTrue(isinstance(self.exception, asyncio.CancelledError))
