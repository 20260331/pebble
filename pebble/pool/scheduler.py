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

from collections import deque
from threading import Condition
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Deque

if TYPE_CHECKING:
    from pebble.pool.base_pool import Task


# Tasks which do not declare a channel are routed to the default channel.
DEFAULT_CHANNEL: str | None = None


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration of a named workload channel.

    *max_concurrency* is the maximum amount of tasks belonging to the
    channel which are allowed to be in flight (dispatched but not yet
    completed) at any given time. If None, the channel is only bounded
    by the Pool *max_workers*.

    *weight* controls the share of workers assigned to the channel once
    multiple channels have tasks ready to run. A channel with weight 3
    is served three times as often as a channel with weight 1 within
    each scheduling round.

    """

    max_concurrency: int | None = None
    weight: int = 1


class FairScheduler:
    """Weighted fair scheduler enforcing per-channel concurrency limits.

    Tasks are admitted into per-channel FIFO queues and retrieved through
    a Deficit Round Robin discipline: each scheduling round, active
    channels accumulate a deficit proportional to their weight and are
    served as long as they hold deficit and are not blocked by their
    concurrency limit. The round rotation ensures a channel continuously
    submitting tasks cannot starve the other ready channels.

    """

    def __init__(
        self,
        channels: Mapping[str, ChannelConfig] | None = None,
    ):
        self._configs: dict[str | None, ChannelConfig] = {
            DEFAULT_CHANNEL: ChannelConfig()
        }
        self._queues: dict[str | None, Deque[Task]] = {DEFAULT_CHANNEL: deque()}
        self._deficit: dict[str | None, int] = {DEFAULT_CHANNEL: 0}
        self._in_flight: dict[str | None, int] = {DEFAULT_CHANNEL: 0}
        self._active: Deque[str | None] = deque()
        self._active_channels: set[str | None] = set()
        self._condition: Condition = Condition()

        for name, config in (channels or {}).items():
            self._register(name, config)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Names of the configured named channels."""
        return tuple(name for name in self._configs if isinstance(name, str))

    def validate(self, channel: str | None):
        """Raise ValueError if the channel is unknown to the scheduler."""
        if channel not in self._configs:
            names = ", ".join(self.channel_names) or "<none>"

            raise ValueError(
                f"Unknown channel {channel!r}, configured channels: {names}"
            )

    def admit(self, task: Task):
        """Enqueue a task within its channel ready queue."""
        self.validate(task.channel)

        with self._condition:
            self._queues[task.channel].append(task)

            if task.channel not in self._active_channels:
                self._active_channels.add(task.channel)
                self._active.append(task.channel)

            self._condition.notify()

    def retrieve(self) -> Task | None:
        """Return the next task allowed to run or None if none is admissible.

        The per-channel concurrency limits are honored. The Pool-wide
        amount of workers is not enforced here: unlimited channels keep
        flowing into the workers queue as without channels configured.
        Cancelled tasks are filtered out lazily and accounted for by
        the caller through :meth:`release`.

        """
        with self._condition:
            for _ in range(len(self._active)):
                name = self._active[0]
                config = self._configs[name]

                if self._at_concurrency_limit(name, config):
                    # Waiting for a slot, let the other channels run.
                    # The channel deficit is retained for its next turn.
                    self._active.rotate(-1)

                    continue

                if self._deficit[name] <= 0:
                    self._deficit[name] += config.weight

                queue = self._queues[name]

                if not queue:
                    self._deactivate(name)

                    continue

                task = queue.popleft()
                self._deficit[name] -= 1
                self._in_flight[name] += 1

                if not queue:
                    self._deactivate(name)
                elif self._deficit[name] <= 0:
                    # Quantum exhausted, give the other channels a turn.
                    self._active.rotate(-1)

                return task

            return None

    def release(self, task: Task):
        """Release the in-flight slot held by the task.

        Must be invoked once for every task returned by :meth:`retrieve`,
        regardless of whether it was executed, cancelled or rejected.

        """
        with self._condition:
            self._in_flight[task.channel] -= 1
            self._condition.notify()

    def wait(self, timeout: float | None = None):
        """Wait until new tasks are admitted or slots are released."""
        with self._condition:
            self._condition.wait(timeout)

    def in_flight(self, channel: str | None) -> int:
        """Amount of in-flight tasks for the given channel."""
        with self._condition:
            return self._in_flight[channel]

    def _register(self, name: str, config: ChannelConfig):
        if not isinstance(name, str) or not name:
            raise TypeError("Channel names must be non-empty strings")
        if not isinstance(config, ChannelConfig):
            raise TypeError("Channel configuration must be a ChannelConfig")
        if config.max_concurrency is not None and (
            not isinstance(config.max_concurrency, int)
            or isinstance(config.max_concurrency, bool)
            or config.max_concurrency < 1
        ):
            raise ValueError("Channel max_concurrency must be None or >= 1")
        if (
            not isinstance(config.weight, int)
            or isinstance(config.weight, bool)
            or config.weight < 1
        ):
            raise ValueError("Channel weight must be an integer >= 1")
        if name in self._configs:
            raise ValueError(f"Channel {name!r} is configured more than once")

        self._configs[name] = config
        self._queues[name] = deque()
        self._deficit[name] = 0
        self._in_flight[name] = 0

    def _at_concurrency_limit(self, name: str | None, config: ChannelConfig) -> bool:
        return (
            config.max_concurrency is not None
            and self._in_flight[name] >= config.max_concurrency
        )

    def _deactivate(self, name: str | None):
        self._active.popleft()
        self._active_channels.discard(name)
        self._deficit[name] = 0
