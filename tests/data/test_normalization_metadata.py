import logging
import unittest
import numpy as np
import pytest

from athena.core.dtypes import Ftype
from athena.core.dtypes.preprocessing.options import PreprocessingOptions

from tests.data.athena_sql_test_base import AthenaSQLTestBase

logger = logging.getLogger(__name__)

NROWS = 10000
COL_NAME = "states"
TABLE_NAME = "test_table"


class TestInferNormalizationMeta(AthenaSQLTestBase):
    def setUp(self):
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)

    @pytest.mark.serial
    def test_infer_normalization_meta(self):
        distributions = {}
        distributions["0"] = {"mean": 0, "stdev": 1}
        distributions["1"] = {"mean": 4, "stdev": 3}

        def gen_gauss_feature():
            return {
                k: np.random.normal(loc=params["mean"], scale=params["stdev"])
                for k, params in distributions.items()
            }

        data = [(i, gen_gauss_feature()) for i in range(NROWS)]
        df = self.sc.parallelize(data).toDF(["i", COL_NAME])
        df.show()

        df.createOrReplaceTempView(TABLE_NAME)

        nsamples = NROWS // 2
        preprocessing_options = PreprocessingOptions(nsamples=nsamples)

        normalization_params = self.fapper.identify_normalization_params(
            TABLE_NAME, COL_NAME, preprocessing_options, seed=self.test_class_seed
        )

        logger.info(normalization_params)
        for fid, params in distributions.items():
            logger.info(
                f"Expect {fid} to be normal with mean {params['mean']}, stdev {params['stdev']}."
            )
            assert normalization_params[fid].ftype == Ftype.CONTINUOUS
            assert abs(normalization_params[fid].mean - params["mean"]) < 0.05
            assert abs(normalization_params[fid].stdev - params["stdev"]) < 0.05


if __name__ == "__main__":
    unittest.main()
