import os
import json
import time
import tempfile
import unittest

from concurrent.futures import TimeoutError

from pebble import ThreadPool, ProcessPool, ProcessExpired
from pebble import ReceiptEvent, ReceiptPolicy


# Module level functions so the ProcessPool can pickle them
# with every multiprocessing start method.


def receipt_function(value):
    return value


def receipt_slow_function(seconds):
    time.sleep(seconds)

    return seconds


def receipt_error_function():
    raise RuntimeError("TOP-SECRET-EXCEPTION-MESSAGE")


def receipt_secret_function(token):
    if token == "fail":
        raise RuntimeError("TOP-SECRET-EXCEPTION-MESSAGE")

    return "TOP-SECRET-RETURN-VALUE"


def receipt_dying_function():
    os._exit(1)


def event_names(records):
    return [record.event for record in records]


class TestThreadPoolReceipts(unittest.TestCase):
    def test_receipts_disabled_by_default(self):
        """Receipt collection is opt-in."""
        with ThreadPool(max_workers=1) as pool:
            self.assertFalse(pool.receipts_enabled)
            pool.schedule(receipt_function, args=[1])

        with self.assertRaises(RuntimeError):
            pool.receipts()
        with self.assertRaises(RuntimeError):
            pool.export_receipts()

    def test_task_lifecycle_events(self):
        """Each task records submission, start, terminal and release."""
        with ThreadPool(max_workers=2, receipts=True) as pool:
            self.assertTrue(pool.receipts_enabled)
            future = pool.schedule(receipt_function, args=[1], channel="chan")
            future.result()

        records = pool.receipts(task_id=0)

        self.assertEqual(
            event_names(records),
            [
                ReceiptEvent.SUBMITTED.value,
                ReceiptEvent.STARTED.value,
                ReceiptEvent.TERMINAL.value,
                ReceiptEvent.RESOURCE_RELEASED.value,
            ],
        )
        self.assertEqual([r.seq for r in records], sorted(r.seq for r in records))
        self.assertTrue(all(r.channel == "chan" for r in records))
        self.assertEqual(records[1].attempt, 1)
        self.assertEqual(records[2].outcome, "success")
        self.assertIsNone(records[2].error_type)
        self.assertEqual(records[3].reason, "queue_slot")
        self.assertIsNotNone(records[1].worker_id)

    def test_failure_records_exception_type_only(self):
        """Failure terminals expose the exception type name, never the text."""
        with ThreadPool(max_workers=1, receipts=True) as pool:
            future = pool.schedule(receipt_error_function)
            self.assertRaises(RuntimeError, future.result)

        terminal = [
            r for r in pool.receipts(task_id=0) if r.event == "terminal"
        ][0]

        self.assertEqual(terminal.outcome, "failure")
        self.assertEqual(terminal.error_type, "RuntimeError")
        self.assertNotIn("TOP-SECRET-EXCEPTION-MESSAGE", pool.export_receipts())

    def test_no_sensitive_content_in_export(self):
        """Arguments, return values and exception text never reach the export."""
        with ThreadPool(max_workers=2, receipts=True) as pool:
            ok = pool.schedule(
                receipt_secret_function, args=["TOP-SECRET-ARGUMENT"]
            )
            bad = pool.schedule(
                receipt_secret_function, args=["fail"]
            )
            self.assertEqual(ok.result(), "TOP-SECRET-RETURN-VALUE")
            self.assertRaises(RuntimeError, bad.result)

        document = pool.export_receipts()

        self.assertNotIn("TOP-SECRET-ARGUMENT", document)
        self.assertNotIn("TOP-SECRET-RETURN-VALUE", document)
        self.assertNotIn("TOP-SECRET-EXCEPTION-MESSAGE", document)

    def test_query_by_task_and_channel(self):
        """Receipts are queryable by task identifier or channel label."""
        with ThreadPool(max_workers=2, receipts=True) as pool:
            pool.schedule(receipt_function, args=[1], channel="a")
            pool.schedule(receipt_function, args=[2], channel="b")
            pool.schedule(receipt_function, args=[3], channel="a")

        channel_a = pool.receipts(channel="a")
        channel_b = pool.receipts(channel="b")

        self.assertEqual({r.task_id for r in channel_a}, {0, 2})
        self.assertEqual({r.task_id for r in channel_b}, {1})
        self.assertEqual(len(pool.receipts(task_id=2)), 4)
        self.assertEqual(pool.receipts(task_id=999), ())
        with self.assertRaises(ValueError):
            pool.receipts(task_id=0, channel="a")

    def test_default_channel(self):
        """Tasks scheduled without a channel land in the default one."""
        with ThreadPool(max_workers=1, receipts=True) as pool:
            pool.schedule(receipt_function, args=[1]).result()

        self.assertEqual(
            {r.channel for r in pool.receipts(task_id=0)}, {"default"}
        )

    def test_invalid_channel_type(self):
        with ThreadPool(max_workers=1, receipts=True) as pool:
            with self.assertRaises(TypeError):
                pool.schedule(receipt_function, channel=123)

    def test_redaction_policy(self):
        """The policy redacts function, worker and channel metadata."""
        policy = ReceiptPolicy(
            redact_functions=True,
            redact_workers=True,
            redact_channels=True,
        )

        with ThreadPool(max_workers=2, receipts=True, receipt_policy=policy) as pool:
            for _ in range(4):
                pool.schedule(receipt_function, args=[1], channel="private")

        document = json.loads(pool.export_receipts())

        for record in document["records"]:
            if record["function"] is not None:
                self.assertEqual(record["function"], "<redacted>")
            if record["channel"] is not None:
                self.assertEqual(record["channel"], "<redacted>")

        worker_ids = {
            r["worker_id"] for r in document["records"] if r["worker_id"] is not None
        }
        self.assertTrue(worker_ids)
        self.assertTrue(all(alias < 2 for alias in worker_ids))

    def test_cancelled_before_start(self):
        """Queued tasks cancelled before dispatch get a terminal receipt."""
        with ThreadPool(max_workers=1, receipts=True) as pool:
            pool.schedule(receipt_slow_function, args=[1])
            future = pool.schedule(receipt_function, args=[1])
            self.assertTrue(future.cancel())

        records = pool.receipts(task_id=1)

        self.assertEqual(
            event_names(records),
            [
                ReceiptEvent.SUBMITTED.value,
                ReceiptEvent.TERMINAL.value,
                ReceiptEvent.RESOURCE_RELEASED.value,
            ],
        )
        self.assertEqual(records[1].outcome, "cancelled")
        self.assertEqual(records[1].reason, "cancelled_before_start")

    def test_export_while_running_raises(self):
        """JSON export is only allowed once the Pool is closed."""
        pool = ThreadPool(max_workers=1, receipts=True)
        pool.schedule(receipt_slow_function, args=[1])

        with self.assertRaises(RuntimeError):
            pool.export_receipts()

        pool.close()
        pool.join()
        self.assertIsInstance(pool.export_receipts(), str)

    def test_export_to_file_is_deterministic(self):
        """The export is valid JSON with stable ordering and can be written."""
        with ThreadPool(max_workers=2, receipts=True) as pool:
            for index in range(5):
                pool.schedule(receipt_function, args=[index], channel="chan")

        first = pool.export_receipts()
        second = pool.export_receipts()

        self.assertEqual(first, second)
        document = json.loads(first)
        self.assertEqual(document["pool"]["kind"], "thread")
        self.assertEqual(document["pool"]["status"], "stopped")

        seqs = [record["seq"] for record in document["records"]]
        self.assertEqual(seqs, sorted(seqs))

        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as fd:
            path = fd.name
        try:
            returned = pool.export_receipts(path)
            with open(path) as file:
                self.assertEqual(file.read(), returned)
        finally:
            os.unlink(path)


class TestProcessPoolReceipts(unittest.TestCase):
    def test_task_lifecycle_events(self):
        """Process tasks record the full lifecycle with a PID worker id."""
        with ProcessPool(max_workers=1, receipts=True) as pool:
            future = pool.schedule(receipt_function, args=[1], channel="chan")
            self.assertEqual(future.result(), 1)

        records = pool.receipts(task_id=0)

        self.assertEqual(
            event_names(records),
            [
                ReceiptEvent.SUBMITTED.value,
                ReceiptEvent.STARTED.value,
                ReceiptEvent.TERMINAL.value,
                ReceiptEvent.RESOURCE_RELEASED.value,
            ],
        )
        self.assertEqual(records[2].outcome, "success")
        self.assertGreater(records[1].worker_id, 0)

        names = [r.event for r in pool.receipts()]
        self.assertEqual(names[0], ReceiptEvent.POOL_STARTED.value)
        self.assertIn(ReceiptEvent.CHANNEL_CLOSED.value, names)
        self.assertEqual(names[-1], ReceiptEvent.POOL_STOPPED.value)

    def test_timeout_records_worker_replacement(self):
        """Timeouts stop the worker and record the replacement."""
        with ProcessPool(max_workers=1, receipts=True) as pool:
            future = pool.schedule(
                receipt_slow_function, args=[2], timeout=0.3
            )
            self.assertRaises(TimeoutError, future.result)

        records = pool.receipts(task_id=0)
        names = event_names(records)

        self.assertEqual(
            names,
            [
                ReceiptEvent.SUBMITTED.value,
                ReceiptEvent.STARTED.value,
                ReceiptEvent.WORKER_REPLACED.value,
                ReceiptEvent.TERMINAL.value,
                ReceiptEvent.RESOURCE_RELEASED.value,
            ],
        )
        replacement = records[2]
        terminal = records[3]

        self.assertEqual(replacement.reason, "timeout")
        self.assertEqual(replacement.worker_id, terminal.worker_id)
        self.assertEqual(terminal.outcome, "timeout")
        self.assertEqual(terminal.error_type, "TimeoutError")

    def test_worker_expiration_records_replacement(self):
        """An expired worker is recorded together with the error terminal."""
        with ProcessPool(max_workers=1, receipts=True) as pool:
            future = pool.schedule(receipt_dying_function)
            self.assertRaises(ProcessExpired, future.result)

        records = pool.receipts(task_id=0)
        names = event_names(records)

        self.assertIn(ReceiptEvent.WORKER_REPLACED.value, names)
        replacement = records[names.index(ReceiptEvent.WORKER_REPLACED.value)]
        terminal = records[names.index(ReceiptEvent.TERMINAL.value)]

        self.assertEqual(replacement.reason, "expired")
        self.assertEqual(terminal.outcome, "error")
        self.assertEqual(terminal.error_type, "ProcessExpired")

    def test_no_sensitive_content_in_export(self):
        """Process pool exports omit arguments, results and exception text."""
        with ProcessPool(max_workers=2, receipts=True) as pool:
            ok = pool.schedule(
                receipt_secret_function,
                args=["TOP-SECRET-ARGUMENT"],
                channel="secret",
            )
            bad = pool.schedule(receipt_secret_function, args=["fail"])
            self.assertEqual(ok.result(), "TOP-SECRET-RETURN-VALUE")
            self.assertRaises(RuntimeError, bad.result)

        document = pool.export_receipts()

        self.assertNotIn("TOP-SECRET-ARGUMENT", document)
        self.assertNotIn("TOP-SECRET-RETURN-VALUE", document)
        self.assertNotIn("TOP-SECRET-EXCEPTION-MESSAGE", document)
        self.assertEqual(
            {r.task_id for r in pool.receipts(channel="secret")}, {0}
        )
        self.assertEqual(json.loads(document)["pool"]["kind"], "process")


if __name__ == "__main__":
    unittest.main()
