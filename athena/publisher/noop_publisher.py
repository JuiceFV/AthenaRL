from athena.core.dataclasses import dataclass
from athena.core.dtypes import TrainingOutput
from athena.core.dtypes.results import NoPublishingResults
from athena.model_managers.manager import ModelManager
from athena.publisher.publisher_base import ModelPublisher


@dataclass
class NoPublishing(ModelPublisher):
    def do_publish(self, model_manager: ModelManager, training_output: TrainingOutput) -> NoPublishingResults:
        return NoPublishingResults(success=True)
