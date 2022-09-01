from typing import Dict, List, Optional, Tuple

from athena.core.dataclasses import dataclass, field
from athena.core.registry import RegistryMeta, DiscriminatedUnion


@dataclass
class ValidationResult(metaclass=RegistryMeta):
    should_publish: bool


@dataclass
class NoValidationResults(ValidationResult):
    __registry_name__ = "no_validation_results"


@ValidationResult.register()
class ValidationResultRoster(DiscriminatedUnion):
    pass


@dataclass
class PublishingResult(metaclass=RegistryMeta):
    success: bool


@dataclass
class NoPublishingResults(PublishingResult):
    __registry_name__ = "no_publishing_results"


@PublishingResult.register()
class PublishingResultRoster(DiscriminatedUnion):
    pass


@dataclass
class TrainingReport(metaclass=RegistryMeta):
    pass


@dataclass
class Seq2SlateTrainingReport(TrainingReport):
    __registry_name__ = "seq2slate_report"


class TrainingReportRoster(DiscriminatedUnion):
    pass


@dataclass
class TrainingOutput:
    output_paths: Dict[str, str] = field(default_factory=dict)
    validation_result: Optional[ValidationResultRoster] = None
    publishing_result: Optional[PublishingResultRoster] = None
    training_report: Optional[TrainingReport] = None
    logger_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = field(default_factory=dict)
