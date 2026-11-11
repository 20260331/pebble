import time
import asyncio
import unittest
import threading
import dataclasses

from unittest import mock
from concurrent.futures import CancelledError, TimeoutError

from pebble import ThreadPool, concurrent
from pebble.pool.base_pool import PoolStatus
from pebble.pool.scheduler import ChannelConfig

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


class ConcurrencyRecorder:
    """Tracks the maximum amount of concurrent tasks per channel."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.max_seen = {}
        self.order = []

    def run(self, channel, gate=None, delay=0.001):
        with self.lock:
            current = self.active.get(channel, 0) + 1
            self.active[channel] = current
            self.max_seen[channel] = max(self.max_seen.get(channel, 0), current)
            self.order.append(channel)

        if gate is not None:
            gate.wait(5)

        time.sleep(delay)

        with self.lock:
            self.active[channel] -= 1

        return channel


def recorder_function(recorder, channel, gate=None, delay=0.001):
    return recorder.run(channel, gate, delay)


def flood(pool, recorder, channel, stop_event):
    while not stop_event.is_set():
        pool.schedule(recorder_function, args=(recorder, channel), channel=channel)


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


class TestThreadPoolChannels(unittest.TestCase):
    def test_thread_pool_channel_result(self):
        """Tasks scheduled on a channel are executed normally."""
        with ThreadPool(
            max_workers=2,
            channels={"a": ChannelConfig(max_concurrency=1)},
        ) as pool:
            future = pool.schedule(
                function, args=[1], kwargs={"keyword_argument": 1},
                channel="a",
            )

        self.assertEqual(future.result(), 2)

    def test_thread_pool_unknown_channel(self):
        """Scheduling on an undeclared channel raises ValueError."""
        with ThreadPool(channels={"a": ChannelConfig()}) as pool:
            with self.assertRaises(ValueError):
                pool.schedule(function, channel="b")

    def test_thread_pool_channel_concurrency_limit(self):
        """The channel concurrency limit is never exceeded."""
        recorder = ConcurrencyRecorder()

        with ThreadPool(
            max_workers=4,
            channels={"limited": ChannelConfig(max_concurrency=1)},
        ) as pool:
            limited = [
                pool.schedule(
                    recorder_function, args=(recorder, "limited"),
                    channel="limited",
                )
                for _ in range(8)
            ]
            default = [
                pool.schedule(recorder_function, args=(recorder, "default"))
                for _ in range(8)
            ]

            for future in limited + default:
                future.result(timeout=10)

        self.assertEqual(recorder.max_seen["limited"], 1)
        self.assertGreaterEqual(recorder.max_seen["default"], 2)

    def test_thread_pool_channel_fairness_no_starvation(self):
        """A continuously submitting channel cannot starve another one."""
        recorder = ConcurrencyRecorder()
        stop_flood = threading.Event()
        small_started = threading.Event()

        def small_function():
            small_started.set()

            return "small"

        with ThreadPool(
            max_workers=3,
            channels={
                "big": ChannelConfig(max_concurrency=3, weight=1),
                "small": ChannelConfig(max_concurrency=1, weight=1),
            },
        ) as pool:
            flood_thread = threading.Thread(
                target=flood, args=(pool, recorder, "big", stop_flood)
            )
            flood_thread.start()

            time.sleep(0.2)  # let the big channel saturate the pool

            small_future = pool.schedule(small_function, channel="small")

            # The small channel starts within a bounded time even though
            # the big channel keeps submitting tasks continuously.
            self.assertTrue(small_started.wait(5))
            self.assertEqual(small_future.result(timeout=10), "small")

            stop_flood.set()
            flood_thread.join()

        self.assertIn("big", recorder.order)

    def test_thread_pool_channel_weights(self):
        """Weights define the dispatch order between ready channels."""
        recorder = ConcurrencyRecorder()
        gate = threading.Event()

        with ThreadPool(
            max_workers=1,
            channels={
                "a": ChannelConfig(weight=3),
                "b": ChannelConfig(weight=1),
            },
        ) as pool:
            futures = [
                pool.schedule(
                    recorder_function,
                    args=(recorder, "a", gate, 0),
                    channel="a",
                )
                for _ in range(8)
            ]
            futures += [
                pool.schedule(
                    recorder_function,
                    args=(recorder, "b", gate, 0),
                    channel="b",
                )
                for _ in range(8)
            ]

            time.sleep(0.2)  # let the scheduler order the whole backlog
            gate.set()

            for future in futures:
                future.result(timeout=10)

        self.assertEqual(
            recorder.order,
            ["a", "a", "a", "b", "a", "a", "a", "b", "a", "a",
             "b", "b", "b", "b", "b", "b"],
        )

    def test_thread_pool_map_keeps_channel(self):
        """All the map chunks inherit the channel of the map call."""
        recorder = ConcurrencyRecorder()

        with ThreadPool(
            max_workers=4,
            channels={"mapped": ChannelConfig(max_concurrency=1)},
        ) as pool:
            future = pool.map(
                recorder_function,
                [recorder] * 8,
                ["mapped"] * 8,
                chunksize=2,
                channel="mapped",
            )

            self.assertEqual(list(future.result()), ["mapped"] * 8)

        self.assertEqual(recorder.max_seen["mapped"], 1)

    def test_thread_pool_channel_cancel(self):
        """Cancelling a queued channel task frees its channel slot."""
        with ThreadPool(
            max_workers=1,
            channels={"a": ChannelConfig(max_concurrency=1)},
        ) as pool:
            running = pool.schedule(long_function, channel="a")
            queued = pool.schedule(long_function, channel="a")
            time.sleep(0.1)

            self.assertTrue(queued.cancel())

            with self.assertRaises(CancelledError):
                queued.result()

            # The channel slot is released, the next task is admitted.
            following = pool.schedule(function, args=[1], channel="a")
            self.assertEqual(following.result(timeout=10), 1)
            running.result(timeout=10)

    def test_thread_pool_channel_decorator(self):
        """The concurrent decorator routes tasks to the given channel."""
        recorder = ConcurrencyRecorder()

        with ThreadPool(
            max_workers=4,
            channels={"decorated": ChannelConfig(max_concurrency=1)},
        ) as pool:

            @concurrent.thread(pool=pool, channel="decorated")
            def decorated(recorder, channel):
                return recorder.run(channel)

            futures = [decorated(recorder, "decorated") for _ in range(6)]

            for future in futures:
                self.assertEqual(future.result(timeout=10), "decorated")

        self.assertEqual(recorder.max_seen["decorated"], 1)

    def test_thread_pool_channel_preserved_worker_recycle(self):
        """Channel limits hold when workers are recycled via max_tasks."""
        recorder = ConcurrencyRecorder()

        with ThreadPool(
            max_workers=2,
            max_tasks=2,
            channels={"a": ChannelConfig(max_concurrency=1)},
        ) as pool:
            futures = [
                pool.schedule(
                    recorder_function, args=(recorder, "a"), channel="a"
                )
                for _ in range(6)
            ]

            for future in futures:
                future.result(timeout=10)

        self.assertEqual(recorder.max_seen["a"], 1)
