import unittest
import source.episode as ep
from source.test.episode.utils import *


class BaseEpisodicsBufferTest(unittest.TestCase):
    def test_constructor(self):
        buffer = ep.EpisodesBuffer(
            stack_size=STACK_SIZE, episode_capacity=EPISODE_CAPACITY, batch_size=BATCH_SIZE
        )
        self.assertEqual(buffer.add_calls, 0)
