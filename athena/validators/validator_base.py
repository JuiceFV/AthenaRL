import abc
import inspect
from typing import List, Optional
from athena.core.dtypes import TableSpec, TrainingOutput
from athena.core.dtypes.results import ValidationResult, ValidationResultRoster
from athena.core.registry import RegistryMeta


class ModelValidator(metaclass=RegistryMeta):
    def validate(
        self,
        training_output: TrainingOutput,
        history: Optional[List[TrainingOutput]] = None,
        input_table_spec: Optional[TableSpec] = None
    ):
        result = self.do_validate()

        result_type = inspect.signature(self.do_validate).return_annotation
        if result_type == inspect.Signature.empty:
            raise TypeError("The result type must be specified.")
        return ValidationResultRoster.make_roster_instance(result, result)

    @abc.abstractmethod
    def do_validate(
        self,
        training_output: TrainingOutput,
        history: Optional[List[TrainingOutput]] = None,
        input_table_spec: Optional[TableSpec] = None
    ) -> ValidationResult:
        pass
