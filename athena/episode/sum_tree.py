import math
import random
from typing import List, Optional

import numpy as np


class SumTree:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Sum tree capacity should be positive. Got: {capacity}")

        self.nodes: List[List[float]] = []
        tree_depth = int(math.ceil(np.log2(capacity)))
        level_size = 1
        for _ in range(tree_depth + 1):
            nodes_at_this_depth = np.zeros(level_size)
            self.nodes.append(nodes_at_this_depth)
            level_size *= 2
        self.max_recorded_priority = 1.0

    def _total_priority(self) -> float:
        return self.nodes[0][0]

    def sample(self, query_value: Optional[float] = None) -> int:
        if self._total_priority() == 0.0:
            raise Exception("Cannot sample from an empty sum tree.")

        if query_value and (query_value < 0.0 or query_value > 1.0):
            raise ValueError("query_value must be in [0, 1].")

        query_value = random.random() if query_value is None else query_value
        query_value *= self._total_priority()

        node_index = 0
        for node_at_this_depth in self.nodes[1:]:
            left_child = node_index * 2

            left_sum = node_at_this_depth[left_child]
            if query_value < left_sum:
                node_index = left_child
            else:
                node_index = left_child + 1
                query_value -= left_sum
        return node_index

    def stratified_sample(self, batch_size: int) -> List[int]:
        if self._total_priority() == 0.0:
            raise Exception("Cannot sample from an empty sum tree.")

        bounds = np.linspace(0.0, 1.0, batch_size + 1)
        segments = [(bounds[i], bounds[i + 1]) for i in range(batch_size)]
        query_values = [random.uniform(x[0], x[1]) for x in segments]
        return [self.sample(query_value=x) for x in query_values]

    def get(self, node_index: int) -> float:
        return self.nodes[-1][node_index]

    def set(self, node_index: int, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Sum tree values should be nonnegative. Got {value}")
        self.max_recorded_priority = max(value, self.max_recorded_priority)
        delta_value = value - self.nodes[-1][node_index]

        for nodes_at_this_depth in reversed(self.nodes):
            nodes_at_this_depth[node_index] += delta_value
            node_index //= 2

        if node_index != 0:
            raise Exception("Sum tree traversal failed, final node index is not 0.")
