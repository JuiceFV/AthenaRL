import unittest

from athena.core.monitors import ValueListMonitor
from athena.core.tracker import trackable


class TestTrackable(unittest.TestCase):
    def test_trackable(self) -> None:
        @trackable(loss=float, name=str)
        class DummyClass:
            def __init__(self, a: int, b: int, c: int = 10) -> None:
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def do_stuff(self, i: float):
                self.notify_trackers(loss=i, name="not_used")

        instance = DummyClass(1, 2)
        self.assertIsInstance(instance, DummyClass)
        self.assertEqual(instance.a, 1)
        self.assertEqual(instance.b, 2)
        self.assertEqual(instance.c, 10)

        monitors = [ValueListMonitor("loss") for _ in range(3)]
        instance.add_trackers(monitors)
        instance.add_tracker(monitors[0])

        for i in range(10):
            instance.do_stuff(float(i))

        for monitor in monitors:
            self.assertEqual(monitor.values, [float(i) for i in range(10)])

    def test_no_trackabke_values(self) -> None:
        try:
            @trackable()
            class NoTrackableValues:
                pass
        except RuntimeError:
            pass
