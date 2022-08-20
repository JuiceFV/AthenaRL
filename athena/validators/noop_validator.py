from typing import List, Optional

from athena.core.dataclasses import dataclass
from athena.core.dtypes.results import NoValidationResults
from athena.validators.validator_base import ModelValidator
from athena.core.dtypes import TrainingOutput, TableSpec

@dataclass
class NoValidation(ModelValidator):
    def do_validate(
        self, 
        training_output: TrainingOutput, 
        history: Optional[List[TrainingOutput]] = None,
        input_table_spec: Optional[TableSpec] = None
    ) -> NoValidationResults:
        return NoValidationResults(should_publish=True)