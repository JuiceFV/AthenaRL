import unittest

import numpy as np

import athena.episode as ep
from tests.episode.utils import *


class CyclicEpisodicsBufferTest(unittest.TestCase):
    def test_add(self):
        buffer = ep.CyclicEpisodesBuffer(
            episode_capacity=EPISODE_CAPACITY, stack_size=STACK_SIZE, batch_size=BATCH_SIZE
        )
        self.assertEqual(buffer.iter(), 0)
        state = np.zeros(STATE_SHAPE)
        buffer.add(action=0, state=state, score=0., is_last=0)
        self.assertEqual(buffer.iter(), STACK_SIZE)

    def test_low_capacity(self):
        with self.assertRaisesRegex(ValueError, "Storage capacity has to be greater than"):
            ep.CyclicEpisodesBuffer(
                stack_size=10,
                capacity=10,
                batch_size=BATCH_SIZE,
                episode_capacity=1,
            )

        with self.assertRaisesRegex(ValueError, "Storage capacity has to be greater than"):
            ep.CyclicEpisodesBuffer(
                stack_size=5,
                capacity=10,
                batch_size=BATCH_SIZE,
                episode_capacity=10,
            )

        ep.CyclicEpisodesBuffer(
            stack_size=5,
            capacity=10,
            batch_size=BATCH_SIZE,
            episode_capacity=5,
        )
