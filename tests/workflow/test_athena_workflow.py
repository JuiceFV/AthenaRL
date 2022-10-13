import json
import logging
import os
from typing import Dict
import unittest
import zipfile
import torch
from unittest.mock import patch

import entry as cli
from click.testing import CliRunner
from athena.core.parameters import NormalizationData, NormalizationParams
from athena.core.dtypes import Dataset
from tests.core.athena_test_base import AthenaTestBase
from ruamel.yaml import YAML

current_dir = os.path.abspath(os.path.dirname(__file__))

SEQ2SLATE_STATE_NORMALIZATION_JSON = os.path.join(
    current_dir, "test_data/seq2slate/seq2slate_state_normalization.json"
)
SEQ2SLATE_CANDIDATE_NORMALIZATION_JSON = os.path.join(
    current_dir, "test_data/seq2slate/seq2slate_candidate_normalization.json"
)
SEQ2SLATE_WORKFLOW_PARQUET_ZIP = os.path.join(
    current_dir, "test_data/seq2slate/seq2slate_workflow_parquet.zip"
)
SEQ2SLATE_WORKFLOW_CONFIG_YAML = os.path.join(
    current_dir, "sample_config/seq2slate.yaml"
)
SEQ2SLATE_WORKFLOW_PARQUET_REL_PATH = "seq2slate_workflow_parquet"

logger = logging.getLogger(__name__)


def get_test_workflow_config(cfg_path: str, use_gpu: bool):
    yaml = YAML(typ="safe")
    with open(cfg_path, "r") as f:
        config = yaml.load(f)
        config["options"]["resource_options"]["gpu"] = int(use_gpu)
        config["input_table_spec"]["table_sample"] = 50.0
        config["input_table_spec"]["eval_table_sample"] = 50.0
    return config


def mock_seq2slate_normalization() -> Dict[str, NormalizationData]:
    with open(SEQ2SLATE_STATE_NORMALIZATION_JSON, "r") as state_f, open(SEQ2SLATE_CANDIDATE_NORMALIZATION_JSON, "r") as candidate_f:

        str_norms = {
            "state": json.load(state_f),
            "candidate": json.load(candidate_f)
        }

    norm_params = {}
    for col, norm in str_norms.items():
        norm_params[col] = {}
        for fid, str_params in norm.items():
            norm_params[col][int(fid)] = NormalizationParams(**json.loads(str_params))
    return {
        col: NormalizationData(dense_normalization_params=params)
        for col, params in norm_params.items()
    }


class TestAthenaWorkflow(AthenaTestBase):
    def _test_seq2slate_workflow(self, use_gpu: bool = False):
        runner = CliRunner()
        config = get_test_workflow_config(SEQ2SLATE_WORKFLOW_CONFIG_YAML, use_gpu)

        with runner.isolated_filesystem():
            yaml = YAML(typ="safe")
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)

            with zipfile.ZipFile(SEQ2SLATE_WORKFLOW_PARQUET_ZIP, "r") as zip_ds:
                zip_ds.extractall()

            mock_dataset = Dataset(
                parquet_url=f"file://{os.path.abspath(SEQ2SLATE_WORKFLOW_PARQUET_REL_PATH)}"
            )
            mock_normalization = mock_seq2slate_normalization()
            with patch(
                "athena.data.data_extractor.DataExtractor.query_data",
                return_value=mock_dataset
            ), patch(
                "athena.model_managers.seq2slate_base.Seq2SlateDataModule.run_feature_identification",
                return_value=mock_normalization
            ), patch(
                "athena.data.fap.spark.SparkFapper.get_max_sequence_len",
                return_value=50
            ):
                result = runner.invoke(
                    cli.run,
                    [
                        "athena.workflow.training.build_and_train",
                        "config.yaml"
                    ],
                    catch_exceptions=False
                )
            logger.info(result.output)
            assert result.exit_code == 0, f"result = {result}"

    def test_seq2slate_workflow(self):
        self._test_seq2slate_workflow()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_seq2slate_workflow_gpu(self):
        self._test_seq2slate_workflow(use_gpu=True)
