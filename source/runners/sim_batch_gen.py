import pytorch_lightning as pl


def random_batch_simulation(
    pickle_path: str,
    dataset_size: int,
    seed: int = 1
):
    return _batch_simulation(
        pickle_path,
        dataset_size,
        seed
    )


def _batch_simulation(
    pickle_path: str,
    dataset_size: int,
    seed: int = 1
) -> None:
    pl.seed_everything(seed)
