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
import weakref
import logging
import itertools

from queue import Queue
from enum import IntEnum
from threading import RLock
from dataclasses import dataclass
from concurrent.futures import Future, TimeoutError, CancelledError
from typing import Iterator, TypeVar, Mapping
from typing import Any, Callable, Generic, Iterable

from pebble.common.types import T
from pebble.common import Result, ResultStatus, ProcessFuture, CONSTS


class BasePool:
    def __init__(
        self,
        max_workers: int,
        max_tasks: int,
        initializer: Callable | None,
        initargs: Iterable[Any],
    ):
        self._context: PoolContext = PoolContext(
            max_workers, max_tasks, initializer, initargs
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
        self._context.leases.release_all()

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
    ):
        self._status: PoolStatus = PoolStatus.CREATED
        self.status_mutex: RLock = RLock()

        self.task_queue: Queue[Task | None] = Queue()
        self.workers: int = max_workers
        self.task_counter: itertools.count = itertools.count()
        self.worker_parameters: Worker = Worker(max_tasks, initializer, initargs)
        self.leases: MapLeaseRegistry = MapLeaseRegistry()

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


class MapLease:
    """Consumption lease bound to an individual map batch.

    The lease is armed once the consumer starts retrieving the map
    results and can be explicitly renewed. When it expires or is
    released, the batch tasks which did not start yet are cancelled
    while the running ones follow the Pool cancellation semantics.

    Leases only keep weak references to the batch futures so that
    abandoning the map does not prevent its garbage collection.

    """

    def __init__(self, timeout: float, futures: Iterable[Future]):
        self._timeout: float = timeout
        self._lock: RLock = RLock()
        self._deadline: float | None = None
        self._released: bool = False
        self._futures: tuple[weakref.ref[Future], ...] = tuple(
            weakref.ref(future) for future in futures
        )

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def active(self) -> bool:
        with self._lock:
            return self._deadline is not None and not self._released

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def arm(self) -> None:
        """Start the lease once the result consumption begins."""
        with self._lock:
            if self._deadline is None and not self._released:
                self._deadline = time.monotonic() + self._timeout

    def renew(self, timeout: float | None = None) -> float:
        """Extend the lease, arming it if consumption did not start yet.

        Returns the new lease deadline.

        """
        with self._lock:
            if self._released:
                raise RuntimeError("Map consumption lease already released")

            self._deadline = time.monotonic() + (
                self._timeout if timeout is None else timeout
            )

            return self._deadline

    def expired(self) -> bool:
        with self._lock:
            return (
                self._deadline is not None
                and not self._released
                and time.monotonic() >= self._deadline
            )

    def cancel(self) -> bool:
        """Cancel the batch tasks following the Pool semantics."""
        with self._lock:
            # Materialize the cancellations before evaluating any()
            # to avoid short-circuiting the remaining batch futures.
            return any(
                tuple(
                    future.cancel()
                    for reference in self._futures
                    if (future := reference()) is not None
                )
            )

    def release(self) -> None:
        with self._lock:
            self._released = True


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

    @property
    def started(self) -> bool:
        return bool(self.timestamp > 0)

    def set_running_or_notify_cancel(self):
        map_future_reference: weakref.ref[MapFuture] | None = getattr(
            self.future, "map_future", None
        )
        if map_future_reference is not None:
            map_future = map_future_reference()
            if map_future is not None and not map_future.done():
                try:
                    map_future.set_running_or_notify_cancel()
                except RuntimeError:
                    pass

        try:
            self.future.set_running_or_notify_cancel()
        except RuntimeError:
            pass


class MapFuture(Future[T]):
    def __init__(
        self,
        futures: Iterable[Future[T]],
        lease: MapLease | None = None,
    ):
        super().__init__()
        self._futures: tuple[Future[T], ...] = tuple(futures)
        self._lease: MapLease | None = lease
        self.lease_finalized: bool = False

    @property
    def futures(self) -> tuple[Future[T], ...]:
        return self._futures

    @property
    def lease(self) -> MapLease | None:
        return self._lease

    def result(self, timeout: float | None = None) -> T:
        arm_map_lease(self)

        return super().result(timeout)

    def renew_lease(self, timeout: float | None = None) -> float:
        """Renew the map consumption lease returning its new deadline."""
        return renew_map_lease(self, timeout)

    def cancel(self) -> bool:
        """Cancel the future.

        Returns True if any of the elements of the iterables is cancelled.
        False otherwise.
        """
        if self._lease is not None:
            cancelled = self._lease.cancel()
            finalize_map_lease(self)
        else:
            cancelled = any(tuple(f.cancel() for f in self._futures))

        super().cancel()

        return cancelled

    def _finalize_lease(self) -> None:
        finalize_map_lease(self)

    def peek_result(self) -> Any:
        return self._result if self.done() else None  # type: ignore[attr-defined]

    def clear_futures(self) -> None:
        self._futures = ()


class ProcessMapFuture(ProcessFuture[T]):
    def __init__(
        self,
        futures: Iterable[ProcessFuture[T]],
        lease: MapLease | None = None,
    ):
        super().__init__()
        self._futures: tuple[ProcessFuture[T], ...] = tuple(futures)
        self._lease: MapLease | None = lease
        self.lease_finalized: bool = False

    @property
    def futures(self) -> tuple[ProcessFuture[T], ...]:
        return self._futures

    @property
    def lease(self) -> MapLease | None:
        return self._lease

    def result(self, timeout: float | None = None) -> T:
        arm_map_lease(self)

        return super().result(timeout)

    def renew_lease(self, timeout: float | None = None) -> float:
        """Renew the map consumption lease returning its new deadline."""
        return renew_map_lease(self, timeout)

    def cancel(self) -> bool:
        """Cancel the future.

        Returns True if any of the elements of the iterables is cancelled.
        False otherwise.
        """
        if self._lease is not None:
            cancelled = self._lease.cancel()
            finalize_map_lease(self)
        else:
            cancelled = any(tuple(f.cancel() for f in self._futures))

        super().cancel()

        return cancelled

    def _finalize_lease(self) -> None:
        finalize_map_lease(self)

    def peek_result(self) -> Any:
        return self._result if self.done() else None  # type: ignore[attr-defined]

    def clear_futures(self) -> None:
        self._futures = ()


def arm_map_lease(map_future: MapFuture | ProcessMapFuture) -> None:
    """Arm the consumption lease as the map results start being consumed."""
    if map_future.lease is not None:
        map_future.lease.arm()


def renew_map_lease(
    map_future: MapFuture | ProcessMapFuture, timeout: float | None
) -> float:
    if map_future.lease is None:
        raise RuntimeError("Map has no consumption lease")

    return map_future.lease.renew(timeout)


def finalize_map_lease(map_future: MapFuture | ProcessMapFuture) -> None:
    """Cancel the not yet started batch tasks and release the buffers."""
    lease = map_future.lease
    if lease is None or map_future.lease_finalized:
        return

    map_future.lease_finalized = True
    lease.cancel()
    release_map_buffers(map_future)
    lease.release()


def release_map_buffers(map_future: MapFuture | ProcessMapFuture) -> None:
    """Release the result iterator and the batch futures references."""
    results = map_future.peek_result()
    if isinstance(results, MapResults):
        results.close()

    futures = map_future.futures
    map_future.clear_futures()

    for future in futures:
        try:
            delattr(future, "map_future")
        except AttributeError:
            pass


class MapResults(Result[T]):
    def __init__(
        self,
        futures: Iterable[Future],
        timeout: float | None = None,
        map_future: MapFuture | ProcessMapFuture | None = None,
    ):
        self._results: Iterator[Result | Exception] | None = (
            itertools.chain.from_iterable(chunk_result(f, timeout) for f in futures)
        )
        self._lease: MapLease | None = (
            map_future.lease if map_future is not None else None
        )
        self._map_future: weakref.ref[MapFuture | ProcessMapFuture] | None = (
            weakref.ref(map_future) if map_future is not None else None
        )
        self._lock: RLock = RLock()

        # The lease could already be expired if the batch was abandoned
        # while waiting for all the chunk futures to complete.
        if self._lease is not None and self._lease.released:
            self._results = None

    def __iter__(self) -> Iterator[T]:
        return self

    def next(self) -> T:
        with self._lock:
            results = self._results
            if results is None or (self._lease is not None and self._lease.released):
                raise CancelledError()

            try:
                result: Result | Exception = next(results)
            except StopIteration:
                self._finalize()
                raise

        if isinstance(result, Result):
            if result.status == ResultStatus.SUCCESS:
                return result.value

            result = result.value

        raise result  # type: ignore[misc]

    __next__ = next

    def renew_lease(self, timeout: float | None = None) -> float:
        """Renew the map consumption lease returning its new deadline."""
        if self._lease is None:
            raise RuntimeError("Map has no consumption lease")

        return self._lease.renew(timeout)

    def close(self) -> None:
        """Abandon the map consumption.

        The not yet started batch tasks are cancelled, the running ones
        follow the Pool cancellation semantics and the result buffers
        are released.

        """
        with self._lock:
            self._finalize()

    def __enter__(self) -> MapResults[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _finalize(self) -> None:
        """Release the result chain and finalize the map lease."""
        if self._results is None:
            return

        self._results = None  # release the chain and the chunk generators

        map_future = self._map_future() if self._map_future is not None else None
        if map_future is not None:
            map_future._finalize_lease()


class MapLeaseRegistry:
    """Tracks the consumption leases of the maps scheduled within a Pool.

    Every map owns an independent lease. The registry only polls them,
    an error raised while handling one lease never affects the others.
    Map futures are referenced weakly so batches abandoned by the
    consumer are detected and cleaned up during the next poll.

    """

    def __init__(self):
        self._lock: RLock = RLock()
        self._entries: list[
            tuple[MapLease, weakref.ref[MapFuture | ProcessMapFuture]]
        ] = []

    def register(self, map_future: MapFuture | ProcessMapFuture) -> None:
        if map_future.lease is None:
            return

        with self._lock:
            self._entries.append((map_future.lease, weakref.ref(map_future)))

    def update(self) -> None:
        """Handle expired leases and garbage collected map futures."""
        with self._lock:
            entries: tuple[
                tuple[MapLease, weakref.ref[MapFuture | ProcessMapFuture]],
                ...,
            ] = tuple(self._entries)

        for entry in entries:
            lease, reference = entry
            try:
                map_future = reference()
                if map_future is None:
                    lease.cancel()
                    lease.release()
                    self._discard(entry)
                elif lease.released:
                    self._discard(entry)
                elif lease.expired():
                    map_future._finalize_lease()
                    self._discard(entry)
            except Exception:
                logging.exception("Error while handling map consumption lease")

    def release_all(self) -> None:
        """Release all the registered leases, used on Pool shutdown."""
        with self._lock:
            entries = tuple(self._entries)
            self._entries.clear()

        for lease, reference in entries:
            try:
                map_future = reference()
                if map_future is not None:
                    map_future._finalize_lease()
                else:
                    lease.cancel()
                    lease.release()
            except Exception:
                logging.exception("Error while releasing map consumption lease")

    def _discard(
        self,
        entry: tuple[MapLease, weakref.ref[MapFuture | ProcessMapFuture]],
    ) -> None:
        with self._lock:
            try:
                self._entries.remove(entry)
            except ValueError:
                pass


def map_results(map_future: MapFutureType, timeout: float | None) -> MapFutureType:
    futures: Iterable[Future] = map_future.futures
    # Weak references ensure the batch futures do not keep the map
    # future alive once the consumer abandons it.
    map_future_reference: weakref.ref[MapFutureType] = weakref.ref(map_future)

    if not futures:
        map_future.set_result(MapResults(futures, map_future=map_future))
        return map_future

    def done_map(_):
        owner = map_future_reference()
        if owner is not None and not owner.done():
            owner.set_result(MapResults(futures, timeout=timeout, map_future=owner))

    for future in futures:
        future.add_done_callback(done_map)
        setattr(future, "map_future", map_future_reference)

    return map_future


def validate_lease(lease: float | None) -> None:
    """Validates the map consumption lease parameter."""
    if lease is None:
        return
    if isinstance(lease, bool) or not isinstance(lease, (int, float)):
        raise TypeError("lease expected to be None, integer or float")
    if lease <= 0:
        raise ValueError("lease must be greater than zero")


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
MapFutureType = TypeVar("MapFutureType", bound=MapFuture | ProcessMapFuture)
