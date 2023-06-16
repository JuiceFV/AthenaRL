import random
import unittest

from athena.episode import sum_tree


class SumTreeTest(unittest.TestCase):
    def setUp(self):
        self._tree = sum_tree.SumTree(capacity=100)

    def test_negative_capacity(self):
        with self.assertRaises(
            ValueError, msg="Sum tree capacity should be positive. Got: -1"
        ):
            sum_tree.SumTree(capacity=-1)

    def test_set_negative_value(self):
        with self.assertRaises(
            ValueError, msg="Sum tree values should be nonnegative. Got -1"
        ):
            self._tree.set(node_index=0, value=-1)

    def test_small_capacity_constructor(self):
        tree = sum_tree.SumTree(capacity=1)
        self.assertEqual(len(tree.nodes), 1)
        tree = sum_tree.SumTree(capacity=2)
        self.assertEqual(len(tree.nodes), 2)

    def test_set_value_small_capacity(self):
        tree = sum_tree.SumTree(capacity=1)
        tree.set(0, 1.5)
        self.assertEqual(tree.get(0), 1.5)

    def test_set_value(self):
        self._tree.set(node_index=0, value=1.0)
        self.assertEqual(self._tree.get(0), 1.0)

        for level in self._tree.nodes:
            self.assertEqual(self._tree.get(0), 1.0)
            node_at_this_depth = len(level)
            for i in range(1, node_at_this_depth):
                self.assertEqual(level[i], 0.0)

    def test_capacity_greater_than_requested(self):
        self.assertGreaterEqual(len(self._tree.nodes[-1]), 100)

    def test_sample_from_empty_tree(self):
        with self.assertRaises(Exception, msg="Cannot sample from an empty sum tree."):
            self._tree.sample()

    def test_sample_with_invalid_query_value(self):
        self._tree.set(node_index=5, value=1.0)
        with self.assertRaises(ValueError, msg="query_value must be in [0, 1]."):
            self._tree.sample(query_value=-0.1)
        with self.assertRaises(ValueError, msg="query_value must be in [0, 1]."):
            self._tree.sample(query_value=1.1)

    def test_sample_singleton(self):
        self._tree.set(node_index=5, value=1.0)
        item = self._tree.sample()

        self.assertEqual(item, 5)

    def test_sample_pair_with_uneven_probabilities(self):
        self._tree.set(node_index=2, value=1.0)
        self._tree.set(node_index=3, value=3.0)

        for _ in range(10000):
            random.seed(1)
            self.assertEqual(self._tree.sample(), 2)

    def test_sample_pair_with_uneven_probabilities_with_query_value(self):
        self._tree.set(node_index=2, value=1.0)
        self._tree.set(node_index=3, value=3.0)

        for _ in range(10000):
            self.assertEqual(self._tree.sample(query_value=0.1), 2)

    def test_sampling_with_seed_does_not_affect_future_calls(self):
        # Setting the seed here will set a deterministic random value r, which will
        # be used when sampling from the tree. Since it is scalled up by the total
        # sum value of the tree, M, we can see that r' * M + m = M, where:
        #   - M  = total sum value of the tree (total_value)
        #   - m  = value of node 3 (max_value)
        #   - r' = r + delta
        # We can then solve for M: M = m / (1 - r'), and we can set the value of the
        # node 2 to r' * M + delta, which will guarantee that
        # r * M < r' * M + delta, thereby guaranteeing that node 2 will always get
        # picked.
        seed = 1
        random.seed(seed)
        deterministic_random_value = random.random()
        max_value = 100
        delta = 0.01
        total_value = max_value / (1 - deterministic_random_value - delta)
        min_value = deterministic_random_value * total_value + delta
        self._tree.set(node_index=2, value=min_value)
        self._tree.set(node_index=3, value=max_value)
        for _ in range(10000):
            random.seed(seed)
            self.assertEqual(self._tree.sample(), 2)
        # The above loop demonstrated that there is 0 probability that node 3 gets
        # selected. The loop below demonstrates that this probability is no longer
        # 0 when the seed is not set explicitly. There is a very low probability
        # that node 2 gets selected, but to avoid flakiness, we simply assert that
        # node 3 gets selected most of the time.
        counts = {2: 0, 3: 0}
        for _ in range(10000):
            counts[self._tree.sample()] += 1
        self.assertLess(counts[2], counts[3])

    def test_stratified_sampling_from_empty_tree(self):
        with self.assertRaises(Exception, msg="Cannot sample from an empty sum tree."):
            self._tree.stratified_sample(5)

    def test_stratified_sampling(self):
        k = 32
        for i in range(k):
            self._tree.set(node_index=i, value=1)
        samples = self._tree.stratified_sample(k)
        self.assertEqual(len(samples), k)
        for i in range(k):
            self.assertEqual(samples[i], i)

    def test_max_recorded_probability(self):
        k = 32
        self._tree.set(node_index=0, value=0)
        self.assertEqual(self._tree.max_recorded_priority, 1)
        for i in range(1, k):
            self._tree.set(node_index=i, value=i)
            self.assertEqual(self._tree.max_recorded_priority, i)
