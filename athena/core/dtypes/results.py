from typing import Dict, List, Optional, Tuple

from athena.core.dataclasses import dataclass, field
from athena.core.registry import DiscriminatedUnion, RegistryMeta


@dataclass
class ValidationResult(metaclass=RegistryMeta):
    r"""
    Base validation result class.
    """
    #: Whether publish a report or not.
    should_publish: bool


@dataclass
class NoValidationResults(ValidationResult):
    r"""
    No validation results required.
    """
    __registry_name__ = "no_validation_results"


@ValidationResult.register()
class ValidationResultRoster(DiscriminatedUnion):
    r"""
    Roster of validation results for the models.
    """
    pass


@dataclass
class PublishingResult(metaclass=RegistryMeta):
    r"""
    """
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


@TrainingReport.register()
class TrainingReportRoster(DiscriminatedUnion):
    pass


@dataclass
class TrainingOutput:
    r"""
    Generalized training output consists of the follwoing reports:

    1. Validation Results. Summarizes a model's validation results.
    2. Publishing Results.
    3. Training Report.
    """
    output_paths: Dict[str, str] = field(default_factory=dict)
    validation_result: Optional[ValidationResultRoster] = None
    publishing_result: Optional[PublishingResultRoster] = None
    training_report: Optional[TrainingReportRoster] = None
    logger_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = field(default_factory=dict)
