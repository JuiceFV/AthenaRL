import unittest

import numpy as np
from athena.episode import prioritized_replay_buffer

# Default parameters used when creating the replay memory.
SCREEN_SIZE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32
REPLAY_CAPACITY = 100


class PrioritizedReplayBufferTest(unittest.TestCase):
    def create_default_memory(self) -> prioritized_replay_buffer.PrioritizedReplayBuffer:
        return prioritized_replay_buffer.PrioritizedReplayBuffer(
            STACK_SIZE, REPLAY_CAPACITY, BATCH_SIZE, max_sample_attempts=10
        )  # For faster tests.

    def add_blank(
        self,
        memory: prioritized_replay_buffer.PrioritizedReplayBuffer,
        actions: int = 0,
        reward: float = 0.0,
        terminal: int = 0,
        priority: float = 1.0
    ) -> int:
        dummy = np.zeros(SCREEN_SIZE)
        memory.add(
            observation=dummy,
            actions=actions,
            reward=reward,
            terminal=terminal,
            priority=priority,
        )
        index = (memory.iter() - 1) % REPLAY_CAPACITY
        return index

    def test_add_with_and_without_priority(self):
        memory = self.create_default_memory()
        self.assertEqual(memory.iter(), 0)
        zeros = np.zeros(SCREEN_SIZE)

        self.add_blank(memory)
        self.assertEqual(memory.iter(), STACK_SIZE)
        self.assertEqual(memory.add_calls, STACK_SIZE)

        # Check that the prioritized replay buffer expects an additional argument
        # for priority.
        with self.assertRaisesRegex(ValueError, "Expected:"):
            memory.add(observation=zeros, actions=0, reward=0, terminal=0)

    def test_dummy_screens_added_to_new_memory(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        for i in range(index):
            self.assertEqual(memory.sum_tree.get(i), 0.0)

    def test_get_priority_with_invalid_indices(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        with self.assertRaises(TypeError, msg="The indices tensor is not 1-dimensional"):
            memory.get_priority(index)
        with self.assertRaises(TypeError, msg="Indices must be int32s"):
            memory.get_priority(np.array([index]))

    def test_set_and_get_priority(self):
        memory = self.create_default_memory()
        batch_size = 7
        indices = np.zeros(batch_size, dtype=np.int32)
        for index in range(batch_size):
            indices[index] = self.add_blank(memory)
        priorities = np.arange(batch_size)
        memory.set_priority(indices, priorities)
        # We send the indices in reverse order and verify the priorities come back
        # in that same order.
        fetched_priorities = memory.get_priority(np.flip(indices, 0))
        for i in range(batch_size):
            self.assertEqual(priorities[i], fetched_priorities[batch_size - 1 - i])

    def test_new_element_has_high_priority(self):
        memory = self.create_default_memory()
        index = self.add_blank(memory)
        self.assertEqual(memory.get_priority(np.array([index], dtype=np.int32))[0], 1.0)

    def test_low_priority_element_not_frequently_sampled(self):
        memory = self.create_default_memory()
        # Add an item and set its priority to 0.
        self.add_blank(memory, terminal=0, priority=0.0)
        # Now add a few new items.
        for _ in range(3):
            self.add_blank(memory, terminal=1)
        # This test should always pass.
        for _ in range(100):
            batch = memory.sample_transition_batch(batch_size=2)
            # Ensure all terminals are set to 1.
            self.assertTrue((batch.terminal == 1).all())

    def test_sample_index_batch_too_many_failed_retries(self):
        memory = self.create_default_memory()
        # Only adding a single observation is not enough to be able to sample
        # (as it both straddles the cursor and does not pass the
        # `index >= self.cursor() - self._update_horizon` check in
        # circular_replay_buffer.py).
        self.add_blank(memory)
        with self.assertRaises(
            RuntimeError,
            msg="Max sample attempts: Tried 10 times but only sampled 1 valid "
            "indices. Batch size is 2",
        ):
            memory.sample_index_batch(2)

    def test_sample_index_batch(self):
        memory = prioritized_replay_buffer.PrioritizedReplayBuffer(
            STACK_SIZE, REPLAY_CAPACITY, BATCH_SIZE, max_sample_attempts=10
        )
        # This will ensure we end up with cursor == 1.
        for _ in range(REPLAY_CAPACITY - STACK_SIZE + 2):
            self.add_blank(memory)
        self.assertEqual(memory.iter(), 1)
        samples = memory.sample_index_batch(REPLAY_CAPACITY)
        # Because cursor == 1, the invalid range as set by circular_replay_buffer.py
        # will be # [0, 1, 2, 3], resulting in all samples being in
        # [STACK_SIZE, REPLAY_CAPACITY - 1].
        for sample in samples:
            self.assertGreaterEqual(sample, STACK_SIZE)
            self.assertLessEqual(sample, REPLAY_CAPACITY - 1)
