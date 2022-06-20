
import logging
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from athena.episode.episodes_buffers.base import EpisodesBuffer

logger = logging.getLogger(__name__)


def _dense2sparse(dense: np.ndarray) -> List[Dict[int, float]]:
    assert len(dense.shape) == 2, f"dense shape is {dense.shape}"
    return [{i: v.item() for i, v in enumerate(elem)} for elem in torch.einsum("i...j->j...i", dense)]


def episodes_buffer_to_df(buffer: EpisodesBuffer) -> pd.DataFrame:
    batch = buffer.sample_observation_batch(batch_size=buffer.size)
    seq_id = list(map(str, batch.seq_id.flatten().tolist()))
    seq_num = batch.seq_num.flatten().tolist()
    action_id = list(map(str, batch.action.flatten().tolist()))
    state_features = []
    for sparse_seq_states in [_dense2sparse(seq_states) for seq_states in batch.state.squeeze(1)]:
        state_features.extend(sparse_seq_states)
    score = batch.score.flatten().tolist()
    is_last = batch.is_last.flatten().tolist()
    data = {
        "seq_id": seq_id,
        "seq_num": seq_num,
        "action_id": action_id,
        "state_features": state_features,
        "score": score,
        "is_last": is_last
    }
    return pd.DataFrame.from_dict(data)
