from typing import Dict, List, Tuple

from athena.core.dataclasses import dataclass, field
from athena.core.registry import RegistryMeta, DiscriminatedUnion


@dataclass
class TrainingOutput:
    output_paths: Dict[str, str] = field(default_factory=dict)
    logger_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = field(default_factory=dict)


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