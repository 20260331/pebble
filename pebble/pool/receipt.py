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

"""Optional, sanitizable task execution receipts.

When enabled on a Pool, every accepted task generates an ordered trail of
:class:`ReceiptRecord` describing its lifecycle within the Pool:

submission, start, worker replacement/retry, terminal state and the
release of the resources assigned to the task.

Receipts are designed to be safely exportable: function arguments,
keyword arguments, return values, exception messages and tracebacks are
never captured. Only non-sensitive metadata (identifiers, event names,
outcome names, exception *type* names) is recorded and the
:class:`ReceiptPolicy` allows redacting the remaining optional metadata.

"""

from __future__ import annotations

import json
import time
import uuid
import itertools

from enum import Enum
from threading import RLock
from dataclasses import dataclass
from concurrent.futures import CancelledError, TimeoutError
from typing import Any, Callable

from pebble.common.types import Result, ResultStatus


DEFAULT_CHANNEL = "default"
REDACTED = "<redacted>"


class ReceiptEvent(str, Enum):
    """Lifecycle events captured within a task receipt."""

    SUBMITTED = "submitted"
    """The task has been accepted by the Pool."""

    STARTED = "started"
    """A worker started executing the task."""

    WORKER_REPLACED = "worker_replaced"
    """The worker assigned to the task was stopped or expired
    and replaced by a new one."""

    TERMINAL = "terminal"
    """The task reached a terminal state."""

    RESOURCE_RELEASED = "resource_released"
    """The resources assigned to the task (queue slot, worker)
    have been returned to the Pool."""

    POOL_STARTED = "pool_started"
    """Pool-level event: the Pool workers have been started."""

    POOL_STOPPED = "pool_stopped"
    """Pool-level event: the Pool workers have been stopped."""

    CHANNEL_CLOSED = "channel_closed"
    """Pool-level event: the process communication channel was closed."""


class ReceiptOutcome(str, Enum):
    """Terminal outcomes of a task."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ReceiptPolicy:
    """Redaction policy applied to the optional receipt metadata.

    Function arguments, return values, exception messages and tracebacks
    are never recorded regardless of the policy configuration.

    """

    redact_functions: bool = False
    """Replace the function qualified names with a redaction marker."""

    redact_workers: bool = False
    """Replace worker identifiers with stable, per-Pool aliases."""

    redact_channels: bool = False
    """Replace user provided channel names with a redaction marker."""


@dataclass(frozen=True)
class ReceiptRecord:
    """A single, sanitized entry within a task receipt.

    The class only exposes primitive, JSON serializable metadata.
    It never contains function arguments, return values or exception text.

    """

    seq: int
    """Monotonic, Pool-wide sequence number defining the record order."""

    event: str
    """Name of the :class:`ReceiptEvent` being recorded."""

    task_id: int | None
    """Identifier of the task. ``None`` for Pool-level events."""

    channel: str | None
    """Channel the task was scheduled on. ``None`` for Pool-level events."""

    attempt: int
    """Execution attempt within the task, starting from one."""

    function: str | None
    """Qualified name of the scheduled function, if not redacted."""

    worker_id: int | None
    """Identifier of the worker (PID or thread id), if not redacted."""

    outcome: str | None
    """Name of the :class:`ReceiptOutcome` for terminal events."""

    error_type: str | None
    """Name of the exception type for failed tasks, message excluded."""

    reason: str | None
    """Machine readable cause of worker replacement or cancellation."""

    timestamp: float
    """Wall clock time at which the event occurred."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event": self.event,
            "task_id": self.task_id,
            "channel": self.channel,
            "attempt": self.attempt,
            "function": self.function,
            "worker_id": self.worker_id,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class ReceiptStore:
    """Thread-safe, in-memory store for the task receipt records.

    All the records live within the Pool owner process, nothing is sent
    through the worker communication channels.

    """

    def __init__(
        self,
        kind: str,
        policy: ReceiptPolicy | None = None,
        pool_id: str | None = None,
    ):
        self.kind: str = kind
        self.pool_id: str = pool_id or uuid.uuid4().hex
        self.policy: ReceiptPolicy = policy or ReceiptPolicy()
        self._records: list[ReceiptRecord] = []
        self._sequence: itertools.count = itertools.count()
        self._lock: RLock = RLock()
        self._worker_aliases: dict[int, int] = {}

    def record(
        self,
        event: ReceiptEvent,
        *,
        task_id: int | None = None,
        channel: str | None = None,
        attempt: int = 0,
        function: Callable | None = None,
        worker_id: int | None = None,
        outcome: str | None = None,
        error: BaseException | None = None,
        reason: str | None = None,
    ) -> ReceiptRecord:
        """Append a sanitized record to the store.

        Only the exception type name is extracted from ``error`` and only
        the function qualified name is extracted from ``function``.

        """
        with self._lock:
            record = ReceiptRecord(
                seq=next(self._sequence),
                event=event.value,
                task_id=task_id,
                channel=self._channel_name(channel),
                attempt=attempt,
                function=self._function_name(function),
                worker_id=self._worker_alias(worker_id),
                outcome=outcome,
                error_type=type(error).__name__ if error is not None else None,
                reason=reason,
                timestamp=time.time(),
            )
            self._records.append(record)

            return record

    def for_task(self, task_id: int) -> tuple[ReceiptRecord, ...]:
        """Return the receipt records of a task in sequence order."""
        with self._lock:
            return tuple(r for r in self._records if r.task_id == task_id)

    def for_channel(self, channel: str) -> tuple[ReceiptRecord, ...]:
        """Return all the receipt records of a channel in sequence order."""
        name = REDACTED if self.policy.redact_channels else channel

        with self._lock:
            return tuple(r for r in self._records if r.channel == name)

    def records(self) -> tuple[ReceiptRecord, ...]:
        """Return all the receipt records in sequence order."""
        with self._lock:
            return tuple(self._records)

    def to_json(self, status: str) -> str:
        """Serialize the receipts into a deterministic JSON document."""
        with self._lock:
            document = {
                "pool": {
                    "id": self.pool_id,
                    "kind": self.kind,
                    "status": status,
                },
                "records": [record.to_dict() for record in self._records],
            }

        return json.dumps(document, indent=2, sort_keys=True)

    def _channel_name(self, channel: str | None) -> str | None:
        if channel is None:
            return None
        if self.policy.redact_channels:
            return REDACTED

        return channel

    def _function_name(self, function: Callable | None) -> str | None:
        if function is None:
            return None
        if self.policy.redact_functions:
            return REDACTED

        name = getattr(function, "__qualname__", None)
        if name is None:
            name = getattr(function, "__name__", None)

        return name or type(function).__name__

    def _worker_alias(self, worker_id: int | None) -> int | None:
        if worker_id is None:
            return None
        if not self.policy.redact_workers:
            return worker_id

        alias = self._worker_aliases.get(worker_id)
        if alias is None:
            alias = len(self._worker_aliases)
            self._worker_aliases[worker_id] = alias

        return alias


def validate_channel(channel: str | None) -> str:
    """Validate the optional user provided channel name."""
    if channel is None:
        return DEFAULT_CHANNEL
    if not isinstance(channel, str):
        raise TypeError("channel expected to be None or string")

    return channel


def outcome_of(result: Result) -> str:
    """Map a function execution :class:`Result` to a receipt outcome name."""
    if result.status == ResultStatus.SUCCESS:
        return ReceiptOutcome.SUCCESS.value
    if isinstance(result.value, TimeoutError):
        return ReceiptOutcome.TIMEOUT.value
    if isinstance(result.value, CancelledError):
        return ReceiptOutcome.CANCELLED.value
    if result.status == ResultStatus.ERROR:
        return ReceiptOutcome.ERROR.value

    return ReceiptOutcome.FAILURE.value


def emit_receipt(context: Any, event: ReceiptEvent, **kwargs: Any) -> None:
    """Record a receipt event on the Pool context if receipts are enabled."""
    store: ReceiptStore | None = getattr(context, "receipt_store", None)

    if store is not None:
        store.record(event, **kwargs)
