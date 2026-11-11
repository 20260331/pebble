# This file is part of Pebble.
# Copyright (c) 2013-2026, Matteo Cafasso

# Pebble is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Pebble is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with Pebble.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import time
import logging
import itertools

from queue import Queue, Empty
from enum import IntEnum
from threading import RLock, Lock, Semaphore
from dataclasses import dataclass
from concurrent.futures import Future, TimeoutError, CancelledError
from typing import Iterator, TypeVar, Mapping
from typing import Any, Callable, Generic, Iterable

from pebble.common.types import T
from pebble.common import Result, ResultStatus, ProcessFuture, CONSTS
from pebble.common import launch_thread


class PoolFullError(RuntimeError):
    """Raised when a task is scheduled while the Pool budget is full.

    It is raised only by Pools configured with a finite
    ``max_unfinished_tasks`` budget and ``block=False``.
    Pools blocking on a full budget wait for a slot instead.

    """


class BasePool:
    def __init__(
        self,
        max_workers: int,
        max_tasks: int,
        initializer: Callable | None,
        initargs: Iterable[Any],
        max_unfinished_tasks: int = 0,
        block: bool = True,
    ):
        self._context: PoolContext = PoolContext(
            max_workers,
            max_tasks,
            initializer,
            initargs,
            max_unfinished_tasks,
            block,
        )
        self._loops: Iterable[Any] = ()
        self._task_counter: itertools.count = itertools.count()

    def __enter__(self: PoolT) -> PoolT:
        return self

    def __exit__(self, *_):
        self.close()
        self.join()

    @property
    def active(self) -> bool:
        self._update_pool_status()

        return self._context.status in (PoolStatus.CLOSED, PoolStatus.RUNNING)

    def close(self):
        """Closes the Pool preventing new tasks from being accepted.

        Pending tasks will be completed.
        """
        self._context.status = PoolStatus.CLOSED

    def stop(self):
        """Stops the pool without performing any pending task."""
        self._context.status = PoolStatus.STOPPED

    def join(self, timeout: float | None = None):
        """Joins the pool waiting until all workers exited.

        If *timeout* is set, it block until all workers are done
        or raises TimeoutError.
        """
        if self._context.status == PoolStatus.RUNNING:
            raise RuntimeError("The Pool is still running")
        if self._context.status == PoolStatus.CLOSED:
            self._wait_queue_depletion(timeout)
            self.stop()
            self.join()
        else:
            self._context.task_queue.put(None)  # Pool termination sentinel
            self._stop_pool()

    def _wait_queue_depletion(self, timeout: float | None):
        tick: float = time.time()

        while self.active:
            if timeout is not None and time.time() - tick > timeout:
                raise TimeoutError("Tasks are still being executed")
            elif self._context.task_queue.unfinished_tasks:
                time.sleep(CONSTS.sleep_unit)
            else:
                return

    def _check_pool_status(self):
        self._update_pool_status()

        if self._context.status == PoolStatus.ERROR:
            raise RuntimeError("Unexpected error within the Pool")
        elif self._context.status != PoolStatus.RUNNING:
            raise RuntimeError("The Pool is not active")

    def _update_pool_status(self):
        if self._context.status == PoolStatus.CREATED:
            self._start_pool()

        for loop in self._loops:  # type: ignore[var-annotated]
            if not loop.is_alive():
                self._context.status = PoolStatus.ERROR

    def _acquire_budget(self, blocking: bool | None = None):
        """Admits a task against the pool-wide unfinished task budget.

        Blocks or raises *PoolFullError* depending on the Pool configuration
        unless a different *blocking* behaviour is requested (map feeder).

        """
        self._check_pool_status()
        self._context.budget.acquire(blocking)

    def _bind_budget(self, task: Task):
        """Binds the task budget slot to the Future terminal state.

        The slot is released exactly once no matter how the task terminates:
        completion, error, cancellation, timeout, worker expiration or
        pool shutdown.

        """
        task.future.add_done_callback(
            lambda _future: self._context.budget.release_task(task)
        )

    def _pool_map(
        self,
        function: Callable[..., T],
        iterables: tuple[Iterable[Any], ...],
        timeout: float | None,
        chunksize: int,
        map_future_type: Callable[[], MapFutureT],
        chunk_runner: Callable,
    ) -> MapFutureT:
        """Shared implementation of the Pool map method.

        The input iterables are consumed lazily by a dedicated feeder thread
        which schedules one chunk at a time, applying backpressure against
        the admission budget instead of prefetching unbounded input.

        """
        self._check_pool_status()

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")

        chunks: Iterator = iter_chunks(zip(*iterables), chunksize)
        map_future: MapFutureT = map_future_type()
        chunk_queue: Queue = Queue()
        map_future.set_result(MapResults(chunk_queue, map_future, timeout))
        launch_thread(
            None,
            map_feeder,
            True,
            self,
            function,
            chunks,
            timeout,
            map_future,
            chunk_queue,
            chunk_runner,
        )

        return map_future

    def _schedule_map_chunk(
        self,
        chunk_runner: Callable,
        function: Callable[..., T],
        chunk: tuple,
        timeout: float | None,
    ) -> Future:
        raise NotImplementedError("Not implemented")

    def _start_pool(self):
        raise NotImplementedError("Not implemented")

    def _stop_pool(self):
        raise NotImplementedError("Not implemented")


class PoolContext:
    def __init__(
        self,
        max_workers: int,
        max_tasks: int,
        initializer: Callable | None,
        initargs: Iterable[Any],
        max_unfinished_tasks: int = 0,
        block_on_full: bool = True,
    ):
        self._status: PoolStatus = PoolStatus.CREATED
        self.status_mutex: RLock = RLock()

        self.task_queue: Queue[Task | None] = Queue()
        self.workers: int = max_workers
        self.task_counter: itertools.count = itertools.count()
        self.worker_parameters: Worker = Worker(max_tasks, initializer, initargs)
        self.budget: TaskBudget = TaskBudget(max_unfinished_tasks, block_on_full)

    @property
    def status(self) -> PoolStatus:
        return self._status

    @status.setter
    def status(self, status: PoolStatus):
        with self.status_mutex:
            if self.alive:
                self._status = status

    @property
    def alive(self) -> bool:
        return self.status not in (PoolStatus.ERROR, PoolStatus.STOPPED)


class Task:
    def __init__(
        self,
        identifier: int,
        future: Future,
        timeout: float | None,
        payload: TaskPayload,
    ):
        self.id: int = identifier
        self.future: Future = future
        self.timeout: float | None = timeout
        self.payload: TaskPayload = payload
        self.timestamp: float = 0.0
        self.worker_id: int = 0
        self.budget_released: bool = False

    @property
    def started(self) -> bool:
        return bool(self.timestamp > 0)

    def set_running_or_notify_cancel(self):
        map_future: MapFuture | None = getattr(self.future, "map_future", None)
        if map_future is not None:
            if not map_future.done():
                try:
                    map_future.set_running_or_notify_cancel()
                except RuntimeError:
                    pass

        try:
            self.future.set_running_or_notify_cancel()
        except RuntimeError:
            pass


class MapFuture(Future[T]):
    def __init__(self):
        super().__init__()
        self._futures: list[Future[T]] = []
        self._futures_lock: Lock = Lock()
        self._map_cancelled: bool = False

    @property
    def futures(self) -> tuple[Future[T], ...]:
        with self._futures_lock:
            return tuple(self._futures)

    @property
    def map_cancelled(self) -> bool:
        return self._map_cancelled

    def register(self, future: Future[T]):
        """Registers a chunk Future scheduled by the map feeder."""
        with self._futures_lock:
            self._futures.append(future)
        setattr(future, "map_future", self)

    def cancel(self) -> bool:
        """Cancel the future.

        Stops the map feeder from consuming further input and cancels
        all the scheduled chunk futures.

        Returns True if any of the chunk futures is cancelled.
        False otherwise.
        """
        self._map_cancelled = True

        with self._futures_lock:
            return any(tuple(f.cancel() for f in self._futures))


class ProcessMapFuture(ProcessFuture[T]):
    def __init__(self):
        super().__init__()
        self._futures: list[ProcessFuture[T]] = []
        self._futures_lock: Lock = Lock()
        self._map_cancelled: bool = False

    @property
    def futures(self) -> tuple[ProcessFuture[T], ...]:
        with self._futures_lock:
            return tuple(self._futures)

    @property
    def map_cancelled(self) -> bool:
        return self._map_cancelled

    def register(self, future: ProcessFuture[T]):
        """Registers a chunk Future scheduled by the map feeder."""
        with self._futures_lock:
            self._futures.append(future)
        setattr(future, "map_future", self)

    def cancel(self) -> bool:
        """Cancel the future.

        Stops the map feeder from consuming further input and cancels
        all the scheduled chunk futures.

        Returns True if any of the chunk futures is cancelled.
        False otherwise.
        """
        self._map_cancelled = True

        with self._futures_lock:
            return any(tuple(f.cancel() for f in self._futures))


class MapResults(Result[T]):
    """Lazily consumes the chunk Futures produced by the map feeder."""

    def __init__(
        self,
        chunk_queue: "Queue[Future | BaseException | None]",
        map_future: MapFuture[Any] | ProcessMapFuture[Any],
        timeout: float | None = None,
    ):
        self._chunk_queue: Queue[Future | BaseException | None] = chunk_queue
        self._map_future: MapFuture[Any] | ProcessMapFuture[Any] = map_future
        self._timeout: float | None = timeout
        self._elements: Iterator[Any] = iter(())
        self._cancelled: bool = False

    def __iter__(self) -> Iterator[T]:
        return self

    def next(self) -> T:
        if self._cancelled:  # input not yet scheduled when the map was cancelled
            raise CancelledError()

        while True:
            try:
                result: Any = next(self._elements)
                break
            except StopIteration:
                item = self._chunk_queue.get()

                if item is None:  # map feeder exhausted the input
                    if self._map_future.map_cancelled:
                        self._cancelled = True
                        raise CancelledError()
                    raise
                if isinstance(item, BaseException):  # feeder/scheduling error
                    raise item

                self._elements = iter(chunk_result(item, self._timeout))

        if isinstance(result, Result):
            if result.status == ResultStatus.SUCCESS:
                return result.value

            raise result.value

        raise result  # type: ignore[misc]

    __next__ = next


def map_feeder(
    pool: BasePool,
    function: Callable[..., T],
    chunks: Iterator,
    timeout: float | None,
    map_future: MapFuture[Any] | ProcessMapFuture[Any],
    chunk_queue: "Queue[Future | BaseException | None]",
    chunk_runner: Callable,
):
    """Schedules the map chunks lazily, applying backpressure on the budget."""
    try:
        for chunk in chunks:
            if map_future.map_cancelled:
                break

            future: Future = pool._schedule_map_chunk(
                chunk_runner, function, chunk, timeout
            )
            # each Pool schedules the Future subtype its MapFuture expects
            map_future.register(future)  # type: ignore[arg-type]
            chunk_queue.put(future)
    except BaseException as error:  # budget rejection, pool shutdown...
        chunk_queue.put(error)
    finally:
        chunk_queue.put(None)


def drain_pending_tasks(context: PoolContext):
    """Releases the budget of tasks stranded in the queue on shutdown.

    Workers and scheduler loops do not consume further tasks once the Pool
    is stopped or broken, so their slots are returned explicitly.
    Idempotent with respect of the Future done callbacks.

    """
    task_queue: Queue[Task | None] = context.task_queue

    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            break

        if task is None:  # termination sentinel, hand it back to the loop
            task_queue.put(None)
            break

        task_queue.task_done()
        context.budget.release_task(task)


def iter_chunks(iterable: Iterable[Any], chunksize: int) -> Iterator:
    """Iterates over zipped iterables in chunks."""
    try:
        yield from itertools.batched(iterable, chunksize)  # type: ignore[attr-defined] - > Python 3.12
    except AttributeError:  # < Python 3.12
        while 1:
            chunk: tuple = tuple(itertools.islice(iterable, chunksize))

            if not chunk:
                return

            yield chunk


def chunk_result(future: Future[T], timeout: float | None) -> Any:
    """Returns the results of a processed chunk."""
    try:
        return future.result(timeout=timeout)
    except BaseException as error:
        return (error,)


def run_initializer(initializer: Callable, initargs: Iterable[Any]) -> bool:
    """Runs the Pool initializer dealing with errors."""
    try:
        initializer(*initargs)
        return True
    except BaseException as error:
        logging.exception(error)
        return False


class PoolStatus(IntEnum):
    """Current status of the Pool."""

    CREATED = 0
    RUNNING = 1
    CLOSED = 2
    STOPPED = 3
    ERROR = 4


class TaskBudget:
    """Pool-wide admission budget tracking unfinished tasks.

    A single instance is shared by every task created within the Pool,
    no matter whether it originates from *schedule*, *submit*, *map*
    or the concurrent/asynchronous decorators.

    A slot is acquired upon task admission and released exactly once
    when the related task reaches its terminal state.

    A *limit* of zero disables the budget, preserving the unbounded
    historical behaviour.

    """

    def __init__(self, limit: int, block_on_full: bool):
        if limit < 0:
            raise ValueError("max_unfinished_tasks must be >= 0")

        self.limit: int = limit
        self.block_on_full: bool = block_on_full
        self._semaphore: Semaphore | None = Semaphore(limit) if limit else None
        self._release_lock: Lock = Lock()

    @property
    def limited(self) -> bool:
        return self._semaphore is not None

    def acquire(self, blocking: bool | None = None):
        """Acquires an admission slot.

        When *blocking* is None, the Pool configuration is used.
        Raises *PoolFullError* if a non-blocking acquisition fails.

        """
        if self._semaphore is None:
            return

        if blocking is None:
            blocking = self.block_on_full

        if not self._semaphore.acquire(blocking=blocking):
            raise PoolFullError(
                "The Pool unfinished task budget is exhausted (%d)" % self.limit
            )

    def release(self):
        """Releases a slot which was never bound to a Task."""
        if self._semaphore is not None:
            self._semaphore.release()

    def release_task(self, task: Task):
        """Releases the slot bound to a Task.

        Idempotent: shutdown draining and Future callbacks can race
        without ever returning more slots than were acquired.

        """
        if self._semaphore is None:
            return

        with self._release_lock:
            if task.budget_released:
                return
            task.budget_released = True

        self._semaphore.release()


@dataclass
class Worker:
    """Worker configuration."""

    max_tasks: int
    initializer: Callable | None
    initargs: Iterable[Any]


@dataclass
class TaskPayload(Generic[T]):
    """The work item wrapped within a Task."""

    function: Callable[..., T]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]


PoolT = TypeVar("PoolT", bound=BasePool)
MapFutureT = TypeVar("MapFutureT", bound="MapFuture[Any] | ProcessMapFuture[Any]")
