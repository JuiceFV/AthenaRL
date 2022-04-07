import logging
import numpy as np
from tqdm import tqdm
from source.sim.sequence import Sequence, SeqEntity
from source.sim.buffer import SimBuffer


logger = logging.getLogger(__name__)


MAX_ITEMS_BULK = 49


def request_seqence(
    buffer: SimBuffer,
    num_items: int,
    seq_id: int
):
    sequence = Sequence()
    iters = 0
    generated = False
    while not generated:

        if iters >= (num_items - 1):
            generated = True
        entity = SeqEntity(
            seq_id
        )


def fill_simulation_buffer(buffer: SimBuffer, dataset_size: int):
    if (dataset_size < 0) or (dataset_size > buffer.capacity):
        raise ValueError("Wrong desirable dataset size.")
    if buffer.size > dataset_size:
        raise ValueError(
            "Current buffer already filled with size greater than given"
        )

    with tqdm(
        total=dataset_size - buffer.size,
        desc=f"Filling simulation buffer to size {dataset_size}"
    ) as pbar:
        seq_id = 0
        while buffer.size < dataset_size:
            temp_size = buffer.size
            max_bulk = min(dataset_size - buffer.size, MAX_ITEMS_BULK)
            request_seqence(buffer, max_bulk, seq_id)
            size_delta = buffer.size - temp_size
            pbar.update(n=size_delta)
            seq_id += 1
            if size_delta <= 0:
                break

        if buffer.size >= dataset_size:
            logger.info(
                f"Successfully filled simulation buffer to size: {buffer.size}!"
            )
        else:
            logger.info(
                f"Stopped early and filled simulation buffer to size: {buffer.size}."
            )
