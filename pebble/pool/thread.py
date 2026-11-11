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
import threading
import multiprocessing

from queue import Queue
from itertools import count
from threading import Thread
from concurrent.futures import Future
from typing import Any, Callable, Mapping, Iterable, List

from pebble.common import ResultStatus, execute, launch_thread, CONSTS
from pebble.common.types import P, T
from pebble.pool.base_pool import iter_chunks, run_initializer
from pebble.pool.base_pool import MapResults, Worker, Result
from pebble.pool.base_pool import PoolStatus, MapFuture, map_results
from pebble.pool.base_pool import PoolContext, BasePool, Task, TaskPayload
from pebble.pool.receipt import ReceiptEvent, ReceiptOutcome, ReceiptPolicy
from pebble.pool.receipt import emit_receipt, outcome_of, validate_channel


class ThreadPool(BasePool):
    """Allows to schedule jobs within a Pool of Threads.

    max_workers is an integer representing the amount of desired process workers
    managed by the pool.
    If max_tasks is a number greater than zero,
    each worker will be restarted after performing an equal amount of tasks.

    initializer must be callable, if passed, it will be called
    every time a worker is started, receiving initargs as arguments.

    """

    def __init__(
        self,
        max_workers: int = multiprocessing.cpu_count(),
        max_tasks: int = 0,
        initializer: Callable | None = None,
        initargs: Iterable[Any] = (),
        receipts: bool = False,
        receipt_policy: ReceiptPolicy | None = None,
    ):
        super().__init__(
            max_workers,
            max_tasks,
            initializer,
            initargs,
            receipts=receipts,
            receipt_policy=receipt_policy,
        )
        self._pool_manager: PoolManager = PoolManager(self._context)
        self._pool_manager_loop: Thread | None = None

    def _pool_kind(self) -> str:
        return "thread"

    def _start_pool(self):
        with self._context.status_mutex:
            if self._context.status == PoolStatus.CREATED:
                self._pool_manager.start()
                self._pool_manager_loop = launch_thread(
                    None, pool_manager_loop, True, self._pool_manager
                )
                self._loops = (self._pool_manager_loop,)

                self._context.status = PoolStatus.RUNNING
                emit_receipt(self._context, ReceiptEvent.POOL_STARTED)

    def _stop_pool(self):
        if self._pool_manager_loop is not None:
            self._pool_manager_loop.join()
        self._pool_manager.stop()

    def schedule(
        self,
        function: Callable[..., T],
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] = {},
        channel: str | None = None,
    ) -> Future[T]:
        """Schedules *function* to be run the Pool.

        *args* and *kwargs* will be forwareded to the scheduled function
        respectively as arguments and keyword arguments.

        *channel* is an optional label grouping the task receipts so they
        can be queried through ``Pool.receipts(channel=...)``.

        A *concurrent.futures.Future* object is returned.

        """
        self._check_pool_status()

        task_channel: str = validate_channel(channel)
        future: Future = Future()
        payload: TaskPayload = TaskPayload(function, args, kwargs)
        task: Task = Task(
            next(self._task_counter), future, None, payload, channel=task_channel
        )

        emit_receipt(
            self._context,
            ReceiptEvent.SUBMITTED,
            task_id=task.id,
            channel=task_channel,
            function=function,
        )

        self._context.task_queue.put(task)

        return future

    def submit(
        self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """This function is provided for compatibility with
        `asyncio.loop.run_in_executor`.

        For scheduling jobs within the pool use `schedule` instead.

        """
        return self.schedule(function, args=args, kwargs=kwargs)

    def map(
        self,
        function: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1
    ) -> MapFuture[MapResults[T]]:
        """Returns an iterator equivalent to map(function, iterables).

        *chunksize* controls the size of the chunks the iterable will
        be broken into before being passed to the function. If None
        the size will be controlled by the Pool.

        """
        self._check_pool_status()

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")

        futures: List[Future] = [
            self.schedule(process_chunk, args=(function, chunk))
            for chunk in iter_chunks(zip(*iterables), chunksize)
        ]
        future: MapFuture = MapFuture(futures)

        return map_results(future, timeout)


def pool_manager_loop(pool_manager: PoolManager):
    context: PoolContext = pool_manager.context

    while context.alive:
        pool_manager.update_status()
        time.sleep(CONSTS.sleep_unit)


class PoolManager:
    def __init__(self, context: PoolContext):
        self.context: PoolContext = context
        self.workers: List[Thread] = []

    def start(self):
        self.create_workers()

    def stop(self):
        for worker in self.workers:
            self.context.task_queue.put(None)
        for worker in tuple(self.workers):
            self.join_worker(worker)

    def update_status(self):
        expired: Iterable[Thread] = self.inspect_workers()

        for worker in expired:
            self.join_worker(worker)

        self.create_workers()

    def inspect_workers(self) -> tuple:
        return tuple(w for w in self.workers if not w.is_alive())

    def create_workers(self):
        for _ in range(self.context.workers - len(self.workers)):
            worker: Thread = launch_thread(None, worker_thread, True, self.context)

            self.workers.append(worker)

    def join_worker(self, worker: Thread):
        worker.join()
        self.workers.remove(worker)


def worker_thread(context: PoolContext):
    """The worker thread routines."""
    queue: Queue[Task | None] = context.task_queue
    parameters: Worker = context.worker_parameters

    if parameters.initializer is not None:
        if not run_initializer(parameters.initializer, parameters.initargs):
            context.status = PoolStatus.ERROR
            return

    for task in get_next_task(context, parameters.max_tasks):
        execute_next_task(context, task)
        queue.task_done()
        emit_receipt(
            context,
            ReceiptEvent.RESOURCE_RELEASED,
            task_id=task.id,
            channel=task.channel,
            attempt=task.attempt,
            function=task.payload.function,
            worker_id=threading.get_ident(),
            reason="queue_slot",
        )


def get_next_task(context: PoolContext, max_tasks: int):
    counter: count = count()
    queue: Queue[Task | None] = context.task_queue

    while context.alive and (max_tasks == 0 or next(counter) < max_tasks):
        task: Task | None = queue.get()

        if task is not None:
            if task.future.cancelled():
                task.set_running_or_notify_cancel()
                emit_receipt(
                    context,
                    ReceiptEvent.TERMINAL,
                    task_id=task.id,
                    channel=task.channel,
                    function=task.payload.function,
                    outcome=ReceiptOutcome.CANCELLED.value,
                    reason="cancelled_before_start",
                )
                emit_receipt(
                    context,
                    ReceiptEvent.RESOURCE_RELEASED,
                    task_id=task.id,
                    channel=task.channel,
                    function=task.payload.function,
                    reason="queue_slot",
                )
                queue.task_done()
            else:
                yield task


def execute_next_task(context: PoolContext, task: Task):
    payload: TaskPayload = task.payload
    task.timestamp = time.time()
    task.attempt += 1
    task.set_running_or_notify_cancel()

    emit_receipt(
        context,
        ReceiptEvent.STARTED,
        task_id=task.id,
        channel=task.channel,
        attempt=task.attempt,
        function=payload.function,
        worker_id=threading.get_ident(),
    )

    result: Result = execute(payload.function, *payload.args, **payload.kwargs)

    if result.status == ResultStatus.SUCCESS:
        task.future.set_result(result.value)
    else:
        task.future.set_exception(result.value)

    emit_receipt(
        context,
        ReceiptEvent.TERMINAL,
        task_id=task.id,
        channel=task.channel,
        attempt=task.attempt,
        function=payload.function,
        worker_id=threading.get_ident(),
        outcome=outcome_of(result),
        error=None if result.status == ResultStatus.SUCCESS else result.value,
    )


def process_chunk(function: Callable, chunk: list) -> list:
    """Processes a chunk of the iterable passed to map dealing with errors."""
    return [execute(function, *args) for args in chunk]
