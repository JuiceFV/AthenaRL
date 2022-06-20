import unittest

import numpy as np

import athena.episode as ep
from tests.episode.utils import *


class FiniteEpisodicsBufferTest(unittest.TestCase):
    def test_add(self):
        buffer = ep.FiniteEpisodesBuffer(
            episode_capacity=EPISODE_CAPACITY, stack_size=STACK_SIZE, batch_size=BATCH_SIZE
        )
        self.assertEqual(buffer.iter(), 0)
        state = np.zeros(STATE_SHAPE)
        buffer.add(action=0, state=state, score=0., is_last=0)
        self.assertEqual(buffer.iter(), 1)

    def test_low_capacity(self):
        with self.assertRaisesRegex(ValueError, "Expected capacity >= episode_capacity"):
            ep.FiniteEpisodesBuffer(
                episode_capacity=11,
                capacity=10,
                batch_size=BATCH_SIZE,
            )

        with self.assertRaisesRegex(ValueError, "Expected stack_size <= episode_capacity"):
            ep.FiniteEpisodesBuffer(
                stack_size=5,
                episode_capacity=4,
                batch_size=BATCH_SIZE,
            )

        ep.FiniteEpisodesBuffer(
            stack_size=5,
            capacity=10,
            batch_size=BATCH_SIZE,
            episode_capacity=5,
        )
