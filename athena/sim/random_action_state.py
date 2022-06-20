import logging
import os
from typing import Iterable, List

import numpy as np
from athena.episode import EpisodesBuffer, Sequence, SequenceEntity
from athena.sim.features import RandGen, SimFeature
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_action(seq_id: int) -> int:
    prev_seed = int(os.environ.get(
        "PL_GLOBAL_SEED",
        str(np.random.get_state()[1][0])
    ))
    np.random.seed(seq_id)
    action = np.random.randint(0, np.iinfo(np.uint32).max)
    np.random.seed(prev_seed)
    return action


def _get_random_features(features: Iterable[SimFeature]) -> Iterable[float]:
    return np.array(list(map(lambda f: f.draw(), features)))


def request_seqence(
    buffer: EpisodesBuffer,
    num_items: int,
    seq_id: int,
    features: Iterable[SimFeature],
    gamma: float = 1.0
) -> None:
    sequence = Sequence()
    seq_num = 0
    generated = False
    ground_truth = _get_random_features(features)
    action = _get_action(seq_id)
    while not generated:
        state = _get_random_features(features)
        if seq_num >= (num_items - 1):
            generated = True
        entity = SequenceEntity(
            seq_id=seq_id,
            seq_num=seq_num,
            action=action,
            state=state,
            score=(1 - np.einsum("i,i->", ground_truth, state)),
            is_last=bool(generated)
        )
        sequence.add_enitity(entity)
        sequence.recalculate_positional_scores(gamma)
        seq_num += 1
    sequence.recalculate_positional_scores(gamma)
    for entity in sequence:
        buffer.add(**entity.asdict())


def fill_episodes_buffer(
    buffer: EpisodesBuffer,
    dataset_size: int,
    gens: List[RandGen],
    max_items_bulk: int = 49,
    features_num: int = 136,
    gamma: float = 1.0
):
    if (dataset_size < 0) or (dataset_size > buffer.capacity):
        raise ValueError("Wrong desirable dataset size.")
    if buffer.size > dataset_size:
        raise ValueError(
            "Current buffer already filled with size greater than given"
        )

    with tqdm(
        total=dataset_size - buffer.size,
        desc=f"Filling episode buffer to size {dataset_size}"
    ) as pbar:
        seq_id = 0
        features = np.array([
            SimFeature(i, np.random.choice(gens))
            for i in range(features_num)
        ])
        while buffer.size < dataset_size:
            temp_size = buffer.size
            max_bulk = min(dataset_size - buffer.size, max_items_bulk)
            request_seqence(buffer, max_bulk, seq_id, features, gamma)
            size_delta = buffer.size - temp_size
            pbar.update(n=size_delta)
            seq_id += 1
            if size_delta <= 0:
                break

        if buffer.size >= dataset_size:
            logger.info(
                f"Successfully filled episode buffer to size: {buffer.size}!"
            )
        else:
            logger.info(
                f"Stopped early and filled episode buffer to size: {buffer.size}."
            )
