import abc
import inspect

from athena.core.registry import RegistryMeta
from athena.core.dtypes.results import PublishingResult, PublishingResultRoster
from athena.model_managers.manager import ModelManager
from athena.core.dtypes import TrainingOutput


class ModelPublisher(metaclass=RegistryMeta):

    def publish(self, model_manager: ModelManager, training_output: TrainingOutput):
        result = self.do_publish(model_manager, training_output)

        result_type = inspect.signature(self.do_publish).return_annotation
        if result_type == inspect.Signature.empty:
            raise TypeError("The result type must be specified.")
        return PublishingResultRoster.make_roster_instance(result, result_type)

    @abc.abstractmethod
    def do_publish(self, model_manager: ModelManager, training_output: TrainingOutput) -> PublishingResult:
        pass
