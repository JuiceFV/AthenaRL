import os
import gzip
import tempfile
import unittest

import numpy as np
import numpy.testing as npt
import torch
from athena.episode import circular_replay_buffer

# Default parameters used when creating the replay memory.
OBSERVATION_SHAPE = (84, 84)
OBSERVATION_DTYPE = np.uint8
STACK_SIZE = 4
BATCH_SIZE = 32


class CheckpointableClass:
    def __init__(self):
        self.a = 0


class ReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._test_subdir = self.tmp_dir.name
        num_dims = 10
        self._test_observations = np.ones(num_dims) * 1
        self._test_actions = np.ones(num_dims) * 2
        self._test_reward = np.ones(num_dims) * 3
        self._test_terminal = np.ones(num_dims) * 4
        self._test_add_calls = np.array(7)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_constructor(self):
        memory = circular_replay_buffer.ReplayBuffer(stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE)
        self.assertEqual(memory.add_calls, 0)

    def test_add(self):
        memory = circular_replay_buffer.ReplayBuffer(stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE)
        self.assertEqual(memory.iter(), 0)
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, actions=0, reward=0, terminal=0)
        # Check if the cursor moved STACK_SIZE -1 padding adds + 1, (the one above).
        self.assertEqual(memory.iter(), STACK_SIZE)

    def test_extra_add(self):
        memory = circular_replay_buffer.ReplayBuffer(stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE)
        self.assertEqual(memory.iter(), 0)
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, actions=0, reward=0, terminal=0, extra1=0, extra2=[0, 0])
        with self.assertRaisesRegex(ValueError, "Expected:"):
            memory.add(observation=zeros, actions=0, reward=0, terminal=0)
        # Check if the cursor moved STACK_SIZE -1 zeros adds + 1, (the one above).
        self.assertEqual(memory.iter(), STACK_SIZE)

    def test_low_capacity(self):
        with self.assertRaisesRegex(ValueError, "There is not enough capacity"):
            circular_replay_buffer.ReplayBuffer(
                stack_size=10,
                replay_capacity=10,
                batch_size=BATCH_SIZE,
                update_horizon=1,
                gamma=1.0,
            )

        with self.assertRaisesRegex(ValueError, "There is not enough capacity"):
            circular_replay_buffer.ReplayBuffer(
                stack_size=5,
                replay_capacity=10,
                batch_size=BATCH_SIZE,
                update_horizon=10,
                gamma=1.0,
            )

        # We should be able to create a buffer that contains just enough for a transition.
        circular_replay_buffer.ReplayBuffer(
            stack_size=5,
            replay_capacity=10,
            batch_size=BATCH_SIZE,
            update_horizon=5,
            gamma=1.0,
        )

    def test_nstep_reward_dum(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE,
            replay_capacity=10,
            batch_size=BATCH_SIZE,
            update_horizon=5,
            gamma=1.0
        )

        for i in range(50):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE),
                actions=0,
                reward=2.0,
                terminal=0
            )

        for _ in range(100):
            batch = memory.sample_transition_batch()
            # Total reward is reward per step * update_horizon.
            self.assertEqual(batch[2][0], 10.0)

    def test_sample_transition_batch(self):
        replay_capacity = 10
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1, replay_capacity=replay_capacity, batch_size=2
        )
        num_adds = 50
        for i in range(num_adds):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, OBSERVATION_DTYPE),
                actions=0,
                reward=0,
                terminal=i % 4,  # Every 4 transition is terminal
            )
        # Test sampling with default batch size.
        for _ in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)
        # Test changing batch sizes.
        for _ in range(1000):
            batch = memory.sample_transition_batch(BATCH_SIZE)
            self.assertEqual(batch[0].shape[0], BATCH_SIZE)
        # Verify we revert to default batch size.
        for _ in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)

        # Verify we can specify what indices to sample.
        indices = [1, 2, 3, 5, 8]
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE) for i in indices]
        )
        expected_next_states = (expected_states + 1) % replay_capacity
        # Because the replay buffer is circular, we can exactly compute what the
        # states will be at the specified indices by doing a little mod math:
        expected_states += num_adds - replay_capacity
        expected_next_states += num_adds - replay_capacity
        # This is replicating the formula that was used above to determine what
        # transitions are terminal when adding observation (i % 4).
        expected_terminal = np.expand_dims(
            np.array([min((x + num_adds - replay_capacity) % 4, 1) for x in indices]), 1
        ).astype(bool)
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(batch.actions, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_actions, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_state, expected_next_states)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))

    def test_sample_transition_batch_extra(self):
        replay_capacity = 10
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1, replay_capacity=replay_capacity, batch_size=2
        )
        num_adds = 50  # The number of transitions to add to the memory.
        for i in range(num_adds):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE),
                actions=0,
                reward=0,
                terminal=i % 4,
                extra1=i % 2,
                extra2=[i % 2, 0],
            )
        # Test sampling with default batch size.
        for _ in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)
        # Test changing batch sizes.
        for _ in range(1000):
            batch = memory.sample_transition_batch(BATCH_SIZE)
            self.assertEqual(batch[0].shape[0], BATCH_SIZE)
        # Verify we revert to default batch size.
        for _ in range(1000):
            batch = memory.sample_transition_batch()
            self.assertEqual(batch[0].shape[0], 2)

        # Verify we can specify what indices to sample.
        indices = [1, 2, 3, 5, 8]
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE) for i in indices]
        )
        expected_next_states = (expected_states + 1) % replay_capacity
        # Because the replay buffer is circular, we can exactly compute what the
        # states will be at the specified indices by doing a little mod math:
        expected_states += num_adds - replay_capacity
        expected_next_states += num_adds - replay_capacity
        # This is replicating the formula that was used above to determine what
        # transitions are terminal when adding observation (i % 4).
        expected_terminal = np.expand_dims(
            np.array([min((x + num_adds - replay_capacity) % 4, 1) for x in indices]), 1
        ).astype(bool)
        expected_extra1 = np.expand_dims(
            np.array([(x + num_adds - replay_capacity) % 2 for x in indices]), 1
        )
        expected_next_extra1 = np.expand_dims(
            np.array([(x + 1 + num_adds - replay_capacity) % 2 for x in indices]), 1
        )
        expected_extra2 = np.stack(
            [
                [(x + num_adds - replay_capacity) % 2 for x in indices],
                np.zeros((len(indices),)),
            ],
            axis=1,
        )
        expected_next_extra2 = np.stack(
            [
                [(x + 1 + num_adds - replay_capacity) % 2 for x in indices],
                np.zeros((len(indices),)),
            ],
            axis=1,
        )
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(batch.actions, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_actions, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_reward, np.zeros((len(indices), 1)))
        npt.assert_array_equal(batch.next_state, expected_next_states)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))
        npt.assert_array_equal(batch.extra1, expected_extra1)
        npt.assert_array_equal(batch.next_extra1, expected_next_extra1)
        npt.assert_array_equal(batch.extra2, expected_extra2)
        npt.assert_array_equal(batch.next_extra2, expected_next_extra2)

    def test_sampling_with_terminal_in_trajectory(self):
        replay_capacity = 10
        update_horizon = 3
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=1,
            replay_capacity=replay_capacity,
            batch_size=2,
            update_horizon=update_horizon,
            gamma=1.0,
        )
        for i in range(replay_capacity):
            memory.add(
                observation=np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE),
                actions=i * 2,
                reward=i,
                terminal=1 if i == 3 else 0,
            )
        indices = [2, 3, 4]
        batch = memory.sample_transition_batch(
            batch_size=len(indices), indices=torch.tensor(indices)
        )
        # In commone shape, state is 2-D unless stack_size > 1.
        expected_states = np.array(
            [np.full(OBSERVATION_SHAPE, i, dtype=OBSERVATION_DTYPE) for i in indices]
        )
        # The reward in the replay buffer will be (an asterisk marks the terminal state): [0 1 2 3* 4 5 6 7 8 9]
        # Since we're setting the update_horizon to 3, the accumulated trajectory
        # reward starting at each of the replay buffer positions will be: [3 6 5 3 15 18 21 24]
        # Since indices = [2, 3, 4], our expected reward are [5, 3, 15].
        expected_reward = np.array([[5], [3], [15]])
        # Because update_horizon = 3, both indices 2 and 3 include terminal.
        expected_terminal = np.array([[1], [1], [0]]).astype(bool)
        npt.assert_array_equal(batch.state, expected_states)
        npt.assert_array_equal(batch.actions, np.expand_dims(np.array(indices) * 2, axis=1))
        npt.assert_array_equal(batch.reward, expected_reward)
        npt.assert_array_equal(batch.terminal, expected_terminal)
        npt.assert_array_equal(batch.indices, np.expand_dims(np.array(indices), 1))

    def test_is_transition_valid(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=10, batch_size=2
        )

        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBSERVATION_DTYPE),
            actions=0,
            reward=0,
            terminal=0,
        )
        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBSERVATION_DTYPE),
            actions=0,
            reward=0,
            terminal=0,
        )
        memory.add(
            observation=np.full(OBSERVATION_SHAPE, 0, dtype=OBSERVATION_DTYPE),
            actions=0,
            reward=0,
            terminal=1,
        )

        # These valids account for the automatically applied padding (3 blanks each episode)
        # correct_valids = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        # The above comment is for the original Dopamine buffer, which doesn't
        # account for terminal frames within the update_horizon frames before
        # the cursor. In this case, the frame right before the cursor
        # is terminal, so even though it is within [c-update_horizon, c],
        # it should still be valid for sampling, as next state doesn't matter.
        correct_valids = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # The cursor is:                    ^\
        for i in range(10):
            self.assertEqual(
                correct_valids[i],
                memory.is_valid_transition(i),
                "Index %i should be %s" % (i, bool(correct_valids[i])),
            )

    def test_save(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        memory.observation = self._test_observations
        memory.actions = self._test_actions
        memory.reward = self._test_reward
        memory.terminal = self._test_terminal
        current_iteration = 5
        stale_iteration = current_iteration - circular_replay_buffer.CHECKPOINT_DURATION
        memory.save(self._test_subdir, stale_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            stale_filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, stale_iteration)
            )
            self.assertTrue(os.path.exists(stale_filename))

        memory.save(self._test_subdir, current_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, current_iteration)
            )
            self.assertTrue(os.path.exists(filename))
            # The stale version file should have been deleted.
            self.assertFalse(os.path.exists(stale_filename))

    def test_save_nonndarray_attributes(self):
        # Tests checkpointing an attribute which is not a numpy array.
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )

        # Add some non-numpy data: an int, a string, an object.
        memory.dummy_attribute_1 = 4753849
        memory.dummy_attribute_2 = "String data"
        memory.dummy_attribute_3 = CheckpointableClass()

        current_iteration = 5
        stale_iteration = current_iteration - circular_replay_buffer.CHECKPOINT_DURATION
        memory.save(self._test_subdir, stale_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            stale_filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, stale_iteration)
            )
            self.assertTrue(os.path.exists(stale_filename))

        memory.save(self._test_subdir, current_iteration)
        for attr in memory.__dict__:
            if attr.startswith("_"):
                continue
            filename = os.path.join(
                self._test_subdir, "{}_ckpt.{}.gz".format(attr, current_iteration)
            )
            self.assertTrue(os.path.exists(filename))
            # The stale version file should have been deleted.
            self.assertFalse(os.path.exists(stale_filename))

    def test_load_from_nonexistent_directory(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, actions=0, reward=0, terminal=0)
        # We are trying to load from a non-existent directory, so a NotFoundError will be raised.
        with self.assertRaises(FileNotFoundError):
            memory.load("/does/not/exist", "3")
        self.assertNotEqual(memory._storage["observation"], self._test_observations)
        self.assertNotEqual(memory._storage["actions"], self._test_actions)
        self.assertNotEqual(memory._storage["reward"], self._test_reward)
        self.assertNotEqual(memory._storage["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_calls, self._test_add_calls)

    def test_partial_load_fails(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE, replay_capacity=5, batch_size=BATCH_SIZE
        )
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, actions=0, reward=0, terminal=0)
        self.assertNotEqual(memory._storage["observation"], self._test_observations)
        self.assertNotEqual(memory._storage["actions"], self._test_actions)
        self.assertNotEqual(memory._storage["reward"], self._test_reward)
        self.assertNotEqual(memory._storage["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_calls, self._test_add_calls)
        numpy_arrays = {
            "observation": self._test_observations,
            "actions": self._test_actions,
            "terminal": self._test_terminal,
            "add_calls": self._test_add_calls,
        }
        for attr in numpy_arrays:
            filename = os.path.join(self._test_subdir, "{}_ckpt.3.gz".format(attr))
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    np.save(outfile, numpy_arrays[attr], allow_pickle=False)
        # We are are missing the reward file, so a NotFoundError will be raised.
        with self.assertRaises(FileNotFoundError):
            memory.load(self._test_subdir, "3")
        # Since we are missing the reward file, it should not have loaded any of
        # the other files.
        self.assertNotEqual(memory._storage["observation"], self._test_observations)
        self.assertNotEqual(memory._storage["actions"], self._test_actions)
        self.assertNotEqual(memory._storage["reward"], self._test_reward)
        self.assertNotEqual(memory._storage["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_calls, self._test_add_calls)

    def test_load(self):
        memory = circular_replay_buffer.ReplayBuffer(
            stack_size=STACK_SIZE,
            replay_capacity=5,
            batch_size=BATCH_SIZE,
        )
        zeros = np.zeros(OBSERVATION_SHAPE)
        memory.add(observation=zeros, actions=0, reward=0, terminal=0)
        self.assertNotEqual(memory._storage["observation"], self._test_observations)
        self.assertNotEqual(memory._storage["actions"], self._test_actions)
        self.assertNotEqual(memory._storage["reward"], self._test_reward)
        self.assertNotEqual(memory._storage["terminal"], self._test_terminal)
        self.assertNotEqual(memory.add_calls, self._test_add_calls)
        store_prefix = circular_replay_buffer.STORE_FILENAME_PREFIX
        numpy_arrays = {
            store_prefix + "observation": self._test_observations,
            store_prefix + "actions": self._test_actions,
            store_prefix + "reward": self._test_reward,
            store_prefix + "terminal": self._test_terminal,
            "add_calls": self._test_add_calls,
        }
        for attr in numpy_arrays:
            filename = os.path.join(self._test_subdir, "{}_ckpt.3.gz".format(attr))
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    np.save(outfile, numpy_arrays[attr], allow_pickle=False)
        memory.load(self._test_subdir, "3")
        npt.assert_allclose(memory._storage["observation"], self._test_observations)
        npt.assert_allclose(memory._storage["actions"], self._test_actions)
        npt.assert_allclose(memory._storage["reward"], self._test_reward)
        npt.assert_allclose(memory._storage["terminal"], self._test_terminal)
        self.assertEqual(memory.add_calls, self._test_add_calls)
