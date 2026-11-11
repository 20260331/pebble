import time
import asyncio
import unittest
import threading
import dataclasses
import itertools

from unittest import mock
from concurrent.futures import CancelledError, TimeoutError

from pebble import ThreadPool, PoolFullError, concurrent
from pebble.pool.base_pool import PoolStatus

initarg = 0

map_gate = threading.Event()


def gated_map_function(value):
    """Map worker blocked until the test opens the gate."""
    map_gate.wait()
    return value


def gated_function(event, value=0):
    event.wait()
    return value


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


class TestThreadPoolTaskBudget(unittest.TestCase):
    """Pool-wide unfinished task budget."""

    def setUp(self):
        map_gate.clear()

    def test_invalid_budget(self):
        """Negative budgets are rejected."""
        with self.assertRaises(ValueError):
            ThreadPool(max_unfinished_tasks=-1)

    def test_unlimited_budget_by_default(self):
        """No budget is enforced unless explicitly configured."""
        with ThreadPool(max_workers=1) as pool:
            self.assertFalse(pool._context.budget.limited)

    def test_reject_when_full(self):
        """block=False raises PoolFullError once the budget is exhausted."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=2, block=False) as pool:
            pool.schedule(long_function)
            pool.schedule(long_function)
            with self.assertRaises(PoolFullError):
                pool.schedule(long_function)

    def test_slot_returned_after_completion(self):
        """A finished task frees a slot for a new admission."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=1, block=False) as pool:
            self.assertEqual(pool.schedule(function, args=[1]).result(), 1)
            self.assertEqual(pool.schedule(function, args=[2]).result(), 2)

    def test_reject_shared_by_submit(self):
        """submit accounts against the same budget as schedule."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=1, block=False) as pool:
            pool.submit(long_function)
            with self.assertRaises(PoolFullError):
                pool.submit(long_function)

    def test_block_waits_for_slot(self):
        """block=True waits instead of rejecting once the budget is full."""
        event = threading.Event()
        result = []

        def waiter():
            result.append(pool.schedule(function, args=[1]).result())

        with ThreadPool(max_workers=1, max_unfinished_tasks=1) as pool:
            first = pool.schedule(gated_function, args=(event,))
            thread = threading.Thread(target=waiter)
            thread.start()
            time.sleep(0.3)
            self.assertTrue(thread.is_alive())  # blocked on the budget
            event.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(result, [1])
        self.assertEqual(first.result(), 0)

    def test_concurrent_rejections_match_overflow(self):
        """Concurrent admissions never hand out more slots than the budget."""
        barrier = threading.Barrier(8)
        outcomes = []

        def client():
            barrier.wait()
            try:
                future = pool.schedule(tid_function)
                outcomes.append(("ok", future))
            except PoolFullError:
                outcomes.append(("full", None))

        with ThreadPool(max_workers=2, max_unfinished_tasks=3, block=False) as pool:
            threads = [threading.Thread(target=client) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len([o for o in outcomes if o[0] == "ok"]), 3)
        self.assertEqual(len([o for o in outcomes if o[0] == "full"]), 5)
        for kind, future in outcomes:
            if kind == "ok":
                future.result()

    def test_queued_cancel_returns_slot(self):
        """Cancelling a queued task returns its slot immediately."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=2, block=False) as pool:
            running = pool.schedule(long_function)
            queued = pool.schedule(long_function)
            with self.assertRaises(PoolFullError):
                pool.schedule(long_function)

            self.assertTrue(queued.cancel())
            admitted = pool.schedule(function, args=[1])
            self.assertEqual(admitted.result(), 1)
            running.result()

    def test_stop_returns_stranded_slots(self):
        """stop returns the slots of tasks no worker will ever consume."""
        pool = ThreadPool(max_workers=1, max_unfinished_tasks=3, block=False)
        for _ in range(3):
            pool.schedule(long_function)
        with self.assertRaises(PoolFullError):
            pool.schedule(long_function)

        pool.stop()
        pool.join()

        self.assertEqual(pool._context.budget._semaphore._value, 3)

    def test_close_join_returns_all_slots(self):
        """All slots are back once a closed pool is joined."""
        pool = ThreadPool(max_workers=2, max_unfinished_tasks=4, block=False)
        for index in range(4):
            pool.schedule(function, args=[index])

        pool.close()
        pool.join()

        self.assertEqual(pool._context.budget._semaphore._value, 4)

    def test_decorator_shares_budget(self):
        """concurrent.thread tasks created via a pool share its budget."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=1, block=False) as pool:

            @concurrent.thread(pool=pool)
            def decorated():
                return 1

            first = decorated()
            with self.assertRaises(PoolFullError):
                decorated()

            self.assertEqual(first.result(), 1)
            self.assertEqual(decorated().result(), 1)

    def test_map_processes_all_within_budget(self):
        """map completes inputs larger than the budget."""
        with ThreadPool(max_workers=2, max_unfinished_tasks=3) as pool:
            self.assertEqual(
                list(pool.map(function, range(20)).result()), list(range(20))
            )

    def test_map_on_rejecting_pool_backpressures(self):
        """map never rejects: the feeder waits for slots on blocking pools."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=2, block=False) as pool:
            self.assertEqual(
                list(pool.map(function, range(15)).result()), list(range(15))
            )

            first = pool.schedule(long_function)
            pool.schedule(long_function)
            with self.assertRaises(PoolFullError):
                pool.schedule(long_function)
            first.result()

    def test_map_does_not_prefetch_beyond_budget(self):
        """Lazy inputs are pulled only as budget slots allow."""
        pulled = []

        def tracking():
            for index in range(30):
                pulled.append(index)
                yield index

        with ThreadPool(max_workers=1, max_unfinished_tasks=2) as pool:
            iterator = pool.map(gated_map_function, tracking()).result()
            time.sleep(0.3)
            # two chunks admitted, the third is pulled while waiting for a slot
            self.assertEqual(pulled, [0, 1, 2])

            map_gate.set()
            self.assertEqual(list(iterator), list(range(30)))

        self.assertEqual(pulled, list(range(30)))

    def test_map_lazy_infinite_input(self):
        """An infinite iterable cannot be prefetched into memory."""
        with ThreadPool(max_workers=1, max_unfinished_tasks=2) as pool:
            map_future = pool.map(gated_map_function, itertools.count())
            iterator = map_future.result()
            time.sleep(0.3)

            map_gate.set()
            self.assertEqual([next(iterator) for _ in range(5)], [0, 1, 2, 3, 4])
            map_future.cancel()

    def test_map_cancel_stops_input_consumption(self):
        """Cancelling a map stops the feeder from pulling more input."""
        pulled = []

        def tracking():
            for index in range(50):
                pulled.append(index)
                yield index

        with ThreadPool(max_workers=1, max_unfinished_tasks=2) as pool:
            map_future = pool.map(gated_map_function, tracking())
            iterator = map_future.result()
            time.sleep(0.3)
            self.assertEqual(pulled, [0, 1, 2])

            map_future.cancel()
            map_gate.set()

            with self.assertRaises(CancelledError):
                list(iterator)

        self.assertLess(len(pulled), 50)
