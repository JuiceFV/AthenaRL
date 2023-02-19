from typing import Dict, Optional, Tuple

from athena.core.dataclasses import dataclass
from athena.core.dtypes import TableSpec
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import (PreprocessingOptions, ReaderOptions,
                                        ResourceOptions, RLOptions)
from athena.core.dtypes.preprocessing.base import InputColumn
from athena.core.dtypes.rl.options import RewardOptions
from athena.core.parameters import NormalizationData, NormalizationKey
from athena.data.athena_datamodule import AthenaDataModule
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.data.manual_datamodule import ManualDataModule
from athena.model_managers.manager import ModelManager
from athena.preprocessing.batch_preprocessor import Seq2SlateBatchPreprocessor
from athena.preprocessing.preprocessor import Preprocessor
from athena.report.seq2slate_reporter import Seq2SlateReporter


@dataclass
class Seq2SlateBase(ModelManager):
    """_summary_

    TODO: add state and candidate features allowlist
    """
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    candidate_preprocessing_options: Optional[PreprocessingOptions] = None

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self._state_preprocessing_options = self.state_preprocessing_options
        self._candidate_preprocessing_options = self.candidate_preprocessing_options
        self.evaluate = self.trainer_params.cpe

    @property
    def should_generate_eval_data(self) -> bool:
        return self.evaluate

    def get_data_module(
        self,
        fapper: FAPper,
        *,
        input_table_spec: Optional[TableSpec] = None,
        rl_options: Optional[RLOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ResourceOptions] = None
    ) -> Optional[AthenaDataModule]:
        return Seq2SlateDataModule(
            fapper=fapper,
            rl_options=rl_options,
            input_table_spec=input_table_spec,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            reader_options=reader_options,
            resource_options=resource_options,
            model_manager=self
        )

    def get_reporter(self) -> Seq2SlateReporter:
        return Seq2SlateReporter()


class Seq2SlateDataModule(ManualDataModule):
    @property
    def should_generate_eval_data(self) -> bool:
        return self.model_manager.should_generate_eval_data

    def run_feature_identification(self, input_table_spec: TableSpec) -> Dict[str, NormalizationData]:
        manager: Seq2SlateBase = self.model_manager
        state_preprocessing_options = manager.state_preprocessing_options or PreprocessingOptions()
        candidate_preprocessing_options = manager.candidate_preprocessing_options or PreprocessingOptions()
        state_normalization_params = self.fapper.identify_normalization_params(
            table_name=input_table_spec.table_name,
            col_name=InputColumn.STATE_FEATURES,
            preprocessing_options=state_preprocessing_options,
        )
        candidate_normalization_params = self.fapper.identify_normalization_params(
            table_name=input_table_spec.table_name,
            col_name=InputColumn.STATE_SEQUENCE_FEATURES,
            preprocessing_options=candidate_preprocessing_options,
        )
        return {
            NormalizationKey.STATE: NormalizationData(dense_normalization_params=state_normalization_params),
            NormalizationKey.CANDIDATE: NormalizationData(dense_normalization_params=candidate_normalization_params)
        }

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        rl_options: RLOptions,
        data_extractor: DataExtractor
    ) -> Dataset:
        reward_options = rl_options.reward_options or RewardOptions()
        return data_extractor.query_data(
            table_name=input_table_spec.table_name,
            discrete_actions=False,
            sample_range=sample_range,
            reward_options=reward_options,
            extra_columns=[InputColumn.STATE_SEQUENCE_FEATURES]
        )

    def build_batch_preprocessor(self) -> Seq2SlateBatchPreprocessor:
        candidate_dim = len(self.candidate_normalization_dict.dense_normalization_params)
        state_preprocessor = Preprocessor(self.state_normalization_dict.dense_normalization_params)
        candidate_preprocessor = Preprocessor(self.candidate_normalization_dict.dense_normalization_params)
        return Seq2SlateBatchPreprocessor(
            num_of_candidates=self.model_manager.num_of_candidates,
            candidate_dim=candidate_dim,
            state_preprocessor=state_preprocessor,
            candidate_preprocessor=candidate_preprocessor,
            on_policy=self.model_manager.trainer_params.params.on_policy
        )
