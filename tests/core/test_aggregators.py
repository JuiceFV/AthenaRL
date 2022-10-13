import unittest

import torch
import athena.core.aggregators as agg


class TestAggregators(unittest.TestCase):
    def test_base(self) -> None:
        aggregator = agg.Aggregator("test")
        self.assertRaisesRegex(
            ValueError, "Got test1; expected test",
            aggregator, "test1", torch.ones(1)
        )

    def test_mean_aggregator(self):
        aggregator = agg.MeanAggregator("test")
        vals = [torch.tensor([1, 2, 3], dtype=float), torch.tensor([5, 6, 7], dtype=float)]
        aggregator("test", vals)
        self.assertListEqual(aggregator.values, [4.0])
        add_vals = [torch.tensor([8, 9, 10], dtype=float)]
        aggregator("test", add_vals)
        self.assertListEqual(aggregator.values, [4.0, 9.0])
