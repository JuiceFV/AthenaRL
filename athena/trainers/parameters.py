from athena.core.config import create_config_class
from athena.core.dtypes import BaseDataClass

from athena.trainers.ranking.seq2slate.seq2slate_base import Seq2SlateTrainer


@create_config_class(
    Seq2SlateTrainer.__init__, blocklist=["reinforce_network", "baseline_network", "baseline_warmup_batches"]
)
class Seq2SlateTrainerParameters(BaseDataClass):
    pass
