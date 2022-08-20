import torch
from typing import Dict, Optional

from athena.core.dtypes import TrainingOutput
from athena.core.parameters import NormalizationData
from athena.model_managers.manager import ModelManager
from athena.model_managers.roster import ModelManagerRoster
from athena.core.dtypes import TableSpec, AthenaOptions
from athena.validators import ModelValidatorRoster
from athena.publisher import ModelPublisherRoster


def build_and_train(
    input_table_spec: TableSpec,
    model: ModelManagerRoster,
    nepochs: int,
    options: AthenaOptions,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidatorRoster] = None,
    publisher: Optional[ModelPublisherRoster] = None
) -> TrainingOutput:
    resource_options = options.resource_options
    use_gpu = torch.cuda.is_available() if resource_options is None else resource_options.use_gpu

    manager: ModelManager = model.value

    normalization_data_map: Optional[Dict[int, NormalizationData]] = None
    setup_data: Optional[Dict[str, bytes]] = None

    data_module = manager.get_data_module(options=options, input_table_spec=input_table_spec)
    if data_module is not None:
        data_module.prepare_data()
        setup_data = data_module.setup_data
