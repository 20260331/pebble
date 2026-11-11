import unittest

from concurrent.futures import Future

from pebble.pool.base_pool import Task, TaskPayload
from pebble.pool.scheduler import ChannelConfig, FairScheduler


def _task(identifier: int, channel: str | None = None) -> Task:
    return Task(
        identifier,
        Future(),
        None,
        TaskPayload(lambda: None, (), {}),
        channel,
    )


class TestFairScheduler(unittest.TestCase):
    def test_default_channel_routes_tasks(self):
        """Tasks without channel flow through the default channel."""
        scheduler = FairScheduler()

        first = _task(1)
        second = _task(2)
        scheduler.admit(first)
        scheduler.admit(second)

        self.assertIs(scheduler.retrieve(), first)
        self.assertIs(scheduler.retrieve(), second)
        self.assertIsNone(scheduler.retrieve())

    def test_unknown_channel_rejected(self):
        """Admitting or validating an unknown channel raises ValueError."""
        scheduler = FairScheduler({"a": ChannelConfig()})

        with self.assertRaises(ValueError):
            scheduler.admit(_task(1, "b"))
        with self.assertRaises(ValueError):
            scheduler.validate("b")

        scheduler.validate(None)
        scheduler.validate("a")

    def test_invalid_channel_configurations(self):
        """Invalid channel definitions are rejected at construction."""
        for channels, error in (
            ({"a": ChannelConfig(max_concurrency=0)}, ValueError),
            ({"a": ChannelConfig(max_concurrency=True)}, ValueError),
            ({"a": ChannelConfig(weight=0)}, ValueError),
            ({"a": ChannelConfig(weight=-1)}, ValueError),
            ({"a": ChannelConfig(weight=2.0)}, ValueError),
            ({"": ChannelConfig()}, TypeError),
            ({1: ChannelConfig()}, TypeError),
        ):
            with self.assertRaises(error):
                FairScheduler(channels)

    def test_concurrency_limit_blocks_channel(self):
        """A channel does not dispatch beyond its concurrency limit."""
        scheduler = FairScheduler({"a": ChannelConfig(max_concurrency=1)})

        scheduler.admit(_task(1, "a"))
        scheduler.admit(_task(2, "a"))

        first = scheduler.retrieve()
        self.assertEqual(first.id, 1)
        self.assertEqual(scheduler.in_flight("a"), 1)

        # The channel reached its limit, no further task is admissible.
        self.assertIsNone(scheduler.retrieve())

        scheduler.release(first)
        self.assertEqual(scheduler.in_flight("a"), 0)

        second = scheduler.retrieve()
        self.assertEqual(second.id, 2)

    def test_other_channels_served_while_one_blocked(self):
        """A capped channel does not block the other ready channels."""
        scheduler = FairScheduler(
            {
                "a": ChannelConfig(max_concurrency=1),
                "b": ChannelConfig(max_concurrency=1),
            }
        )

        scheduler.admit(_task(1, "a"))
        scheduler.admit(_task(2, "a"))
        scheduler.admit(_task(3, "b"))

        a_task = scheduler.retrieve()
        self.assertEqual(a_task.channel, "a")

        # "a" is capped, the scheduler rotates and serves "b".
        b_task = scheduler.retrieve()
        self.assertEqual(b_task.channel, "b")

        # Both channels capped now, nothing can be admitted.
        self.assertIsNone(scheduler.retrieve())

        scheduler.release(a_task)

        next_a = scheduler.retrieve()
        self.assertEqual(next_a.id, 2)
        self.assertEqual(next_a.channel, "a")

    def test_weighted_round_robin_sequence(self):
        """Weights control the service ratio within each round."""
        scheduler = FairScheduler(
            {
                "a": ChannelConfig(weight=3),
                "b": ChannelConfig(weight=1),
            }
        )

        for identifier in range(4):
            scheduler.admit(_task(identifier, "a"))
        for identifier in range(4, 8):
            scheduler.admit(_task(identifier, "b"))

        order = []

        while True:
            task = scheduler.retrieve()

            if task is None:
                break

            order.append(task.channel)

        # "a" is served three times, then "b" once before the next round.
        self.assertEqual(order, ["a", "a", "a", "b", "a", "b", "b", "b"])

    def test_continuous_channel_cannot_starve_others(self):
        """A channel admitting tasks continuously cannot monopolize workers."""
        scheduler = FairScheduler(
            {
                "big": ChannelConfig(weight=1),
                "small": ChannelConfig(weight=1),
            }
        )

        for identifier in range(100):
            scheduler.admit(_task(identifier, "big"))
        scheduler.admit(_task(1000, "small"))

        first = scheduler.retrieve()
        second = scheduler.retrieve()

        self.assertEqual(first.channel, "big")
        # The round rotation serves the small channel within the first round.
        self.assertEqual(second.channel, "small")

    def test_default_channel_is_independent(self):
        """The default channel has its own limit counters."""
        scheduler = FairScheduler(
            {"a": ChannelConfig(max_concurrency=1)}
        )

        scheduler.admit(_task(1, "a"))
        scheduler.admit(_task(2, "a"))
        scheduler.admit(_task(3, None))
        scheduler.admit(_task(4, None))

        self.assertEqual(scheduler.retrieve().channel, "a")
        # "a" is capped but the unlimited default channel keeps flowing.
        self.assertEqual(scheduler.retrieve().channel, None)
        self.assertEqual(scheduler.retrieve().channel, None)
        self.assertIsNone(scheduler.retrieve())

    def test_in_flight_accounting_balances_on_release(self):
        """Cancelled tasks returned by retrieve balance out on release."""
        scheduler = FairScheduler({"a": ChannelConfig(max_concurrency=2)})

        first = _task(1, "a")
        second = _task(2, "a")
        scheduler.admit(first)
        scheduler.admit(second)

        self.assertIs(scheduler.retrieve(), first)
        self.assertIs(scheduler.retrieve(), second)
        self.assertEqual(scheduler.in_flight("a"), 2)

        first.future.cancel()
        scheduler.release(first)
        scheduler.release(second)

        self.assertEqual(scheduler.in_flight("a"), 0)


if __name__ == "__main__":
    unittest.main()
