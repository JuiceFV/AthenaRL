"""Temporary module dedicated for the test purposes.
TODO: Remove it afterall
"""
import logging
from typing import Any, Dict, List

import pytorch_lightning as pl
import athena.episode as ep
from athena.sim.features import RandGen
from athena.sim.random_action_state import fill_episodes_buffer
from athena.episode import episodes_buffer_to_df


logger = logging.getLogger(__name__)


def random_batch_simulation(
    pkl_path: str,
    generators: List[RandGen],
    buffer_dict: Dict[str, Any],
    features_num: int = 136,
    gamma: float = 1.0,
    seed: int = 1
):
    buffer_type = getattr(ep, buffer_dict['name'], None)
    if buffer_type is None:
        raise NotImplementedError(
            f"The buffer {buffer_dict['name']} is not implemented, yet"
        )
    buffer_dict['type'] = buffer_type
    return _batch_simulation(
        pkl_path,
        generators,
        buffer_dict,
        features_num,
        gamma,
        seed
    )


def _batch_simulation(
    pkl_path: str,
    generators: List[RandGen],
    buffer_dict: Dict[str, Any],
    features_num: int = 136,
    gamma: float = 1.0,
    seed: int = 1
) -> None:
    pl.seed_everything(seed)
    buffer: ep.EpisodesBuffer = buffer_dict['type'](**buffer_dict['params'])
    print(buffer_dict['params'])
    fill_episodes_buffer(
        buffer,
        buffer.capacity,
        generators,
        buffer.episode_capacity,
        features_num,
        gamma
    )
    df = episodes_buffer_to_df(buffer)
    logger.info(f"Saving dataset with {len(df)} samples to {pkl_path}")
    df.to_pickle(pkl_path)
