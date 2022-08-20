import unittest
from typing import Dict

import athena.preprocessing.normalization as normalization
import numpy as np
import numpy.testing as npt
import torch
from athena.core.dtypes import Ftype
from athena.core.parameters import NormalizationParams
from athena.preprocessing import MISSING_VALUE
from athena.preprocessing.normalization import (identify_param,
                                                sort_features_by_normalization)
from athena.preprocessing.preprocessor import Preprocessor
from scipy import special
from tests.preprocessing.utils import (BOXCOX_FEATURE_ID,
                                       CONTINUOUS_FEATURE_ID, ENUM_FEATURE_ID,
                                       PROBABILITY_FEATURE_ID,
                                       NumpyFeaturePreprocessor, id2type,
                                       variate_data)


class TestPreprocessing(unittest.TestCase):
    def test_prepare_normalization_and_normalize(self):
        fvalue_map = variate_data()
        normalization_params: Dict[str, NormalizationParams] = {}
        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues, 10)
        for fid, norm_params in normalization_params.items():
            if id2type(fid) == Ftype.CONTINUOUS.value:
                self.assertEqual(norm_params.ftype, Ftype.CONTINUOUS)
                self.assertIs(norm_params.boxcox_lambda, None)
                self.assertIs(norm_params.boxcox_shift, None)
            elif id2type(fid) == Ftype.BOXCOX.value:
                self.assertEqual(norm_params.ftype, Ftype.BOXCOX)
                self.assertIsNot(norm_params.boxcox_lambda, None)
                self.assertIsNot(norm_params.boxcox_shift, None)
            else:
                assert norm_params.ftype.value == id2type(fid)

        preprocessor = Preprocessor(normalization_params)
        sorted_features, _ = sort_features_by_normalization(normalization_params)
        input_matrix = torch.zeros([10000, len(sorted_features)])
        for i, fid in enumerate(sorted_features):
            input_matrix[:, i] = torch.from_numpy(fvalue_map[fid])
        normalized_feature_matrix = preprocessor(input_matrix, (input_matrix != MISSING_VALUE))
        normalized_features: Dict[int, torch.Tensor] = {}
        on_column = 0
        for fid in sorted_features:
            norm = normalization_params[fid]
            if norm.ftype == Ftype.ENUM:
                column_size = len(norm.possible_values)
            else:
                column_size = 1
            normalized_features[fid] = normalized_feature_matrix[:, on_column: (on_column + column_size)]
            on_column += column_size

        self.assertTrue(
            all(
                [
                    np.isfinite(param.stdev) and np.isfinite(param.mean)
                    for param in normalization_params.values()
                ]
            )
        )
        for fid, nfvalues in normalized_features.items():
            nfvalues = nfvalues.numpy()
            self.assertTrue(np.all(np.isfinite(nfvalues)))
            ftype = normalization_params[fid].ftype
            if ftype == Ftype.PROBABILITY:
                sigmoidv = special.expit(nfvalues)
                self.assertTrue(np.all(np.logical_and(np.greater(sigmoidv, 0), np.less(sigmoidv, 1))))
            elif ftype == Ftype.ENUM:
                possible_values = normalization_params[fid].possible_values
                self.assertEqual(nfvalues.shape[0], len(fvalue_map[fid]))
                self.assertEqual(nfvalues.shape[1], len(possible_values))
                possible_value_map = {}
                for i, possible_value in enumerate(possible_values):
                    possible_value_map[possible_value] = i
                for i, row in enumerate(nfvalues):
                    original_feature = fvalue_map[fid][i]
                    if abs(original_feature - MISSING_VALUE) < 0.01:
                        self.assertEqual(0.0, np.sum(row))
                    else:
                        self.assertEqual(possible_value_map[original_feature], np.where(row == 1)[0][0])
            elif ftype == Ftype.QUANTILE:
                for i, feature in enumerate(nfvalues[0]):
                    original_feature = fvalue_map[fid][i]
                    expected = NumpyFeaturePreprocessor.value_to_quantile(
                        original_feature, normalization_params[fid].quantiles
                    )
                    self.assertAlmostEqual(feature, expected, 2)
            elif ftype in (Ftype.CONTINUOUS, Ftype.BOXCOX):
                one_stdev = np.isclose(np.std(nfvalues, ddof=1), 1, atol=0.01)
                zero_stdev = np.isclose(np.std(nfvalues, ddof=1), 0, atol=0.01)
                zero_mean = np.isclose(np.mean(nfvalues), 0, atol=0.01)
                self.assertTrue(
                    np.all(zero_mean),
                    f"Mean of feature (id) {fid} is {np.mean(nfvalues)}, but should be 0"
                )
                self.assertTrue(np.logical_or(one_stdev, zero_stdev))
            elif ftype == Ftype.BINARY:
                pass
            else:
                raise NotImplementedError()

    def test_normalize_dense_matrix_enum(self):
        normalization_params = {
            1: NormalizationParams(Ftype.ENUM, None, None, None, None, [12, 4, 2], None, None, None),
            2: NormalizationParams(Ftype.CONTINUOUS, None, 0, 0, 1, None, None, None, None),
            3: NormalizationParams(Ftype.ENUM, None, None, None, None, [15, 3], None, None, None)
        }

        preprocessor = Preprocessor(normalization_params)

        inputs = np.zeros([4, 3], dtype=np.float32)
        fids = [2, 1, 3]
        inputs[:, fids.index(1)] = [12, 4, 2, 2]
        inputs[:, fids.index(2)] = [1.0, 2.0, 3.0, 3.0]
        inputs[:, fids.index(3)] = [15, 3, 15, MISSING_VALUE]
        inputs = torch.from_numpy(inputs)
        normalized_feature_matrix = preprocessor(inputs, (inputs != MISSING_VALUE))

        npt.assert_allclose(
            np.array(
                [
                    [1.0, 1, 0, 0, 1, 0],
                    [2.0, 0, 1, 0, 0, 1],
                    [3.0, 0, 0, 1, 1, 0],
                    [3.0, 0, 0, 1, 0, 0]

                ]
            ),
            normalized_feature_matrix
        )

    def test_persistency(self):
        fvalue_map = variate_data()
        normalization_params: Dict[int, NormalizationParams] = {}
        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues)
            fvalues[0] = MISSING_VALUE

        serialized_normalization_params = normalization.serialize(normalization_params)
        deserialized_normalization_params = normalization.deseialize(serialized_normalization_params)
        self.assertEqual(deserialized_normalization_params.keys(), normalization_params.keys())
        for fid in normalization_params:
            for field in [
                "ftype",
                "possible_values",
                "boxcox_lambda",
                "boxcox_shift",
                "mean",
                "stdev",
                "quantiles",
                "min_value",
                "max_value"
            ]:
                self.assertEqual(
                    getattr(deserialized_normalization_params[fid], field),
                    getattr(normalization_params[fid], field)
                )

    def test_quantile_boundary(self):
        input = torch.tensor([[0.0], [80.0], [100.0]])
        norm_params = NormalizationParams(Ftype.QUANTILE, None, None, 0, 1, None, [0.0, 80.0, 100.0], 0.0, 100.0)
        preprocessor = Preprocessor({1: norm_params})
        preprocessed_input = preprocessor._preprocess_quantile(0, input.float(), [norm_params])

        expected = torch.tensor([[0.0], [0.5], [1.0]])

        self.assertTrue(np.allclose(preprocessed_input, expected))

    def test_preprocessing_network(self):
        fvalue_map = variate_data()

        normalization_params: Dict[int, NormalizationParams] = {}
        fid_preprocessed_blob_map: Dict[int, np.ndarray] = {}

        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues)
            fvalues[0] = MISSING_VALUE

            preprocessor = Preprocessor({fid: normalization_params[fid]})
            fvalue_matrix = torch.from_numpy(np.expand_dims(fvalues, -1))
            normalized_fvalues: torch.Tensor = preprocessor(fvalue_matrix, (fvalue_matrix != MISSING_VALUE))
            fid_preprocessed_blob_map[fid] = normalized_fvalues.numpy()

        test_features = NumpyFeaturePreprocessor.preprocess(fvalue_map, normalization_params)

        for fid in fvalue_map:
            normalized_features = fid_preprocessed_blob_map[fid]
            if fid != ENUM_FEATURE_ID:
                normalized_features = np.squeeze(normalized_features, -1)

            tol = 0.01
            if fid == BOXCOX_FEATURE_ID:
                tol = 0.5

            self.assertTrue(
                np.allclose(
                    normalized_features.flatten(),
                    test_features[fid].flatten(),
                    rtol=tol,
                    atol=tol
                ),
                f"{fid} doesn't match."
            )

    def test_type_override_binary(self):
        fvalue_map = variate_data()
        probability_values = fvalue_map[PROBABILITY_FEATURE_ID]

        param = identify_param(-101, probability_values, ftype=Ftype.BINARY)
        self.assertEqual(param.ftype.value, "binary")

    def test_type_override_continuous(self):
        fvalue_map = variate_data()
        boxcocks_values = fvalue_map[BOXCOX_FEATURE_ID]

        param = identify_param(-101, boxcocks_values, ftype=Ftype.CONTINUOUS)
        self.assertEqual(param.ftype.value, "continuous")

    def test_type_override_boxcox(self):
        fvalue_map = variate_data()
        continuous_values = fvalue_map[CONTINUOUS_FEATURE_ID]

        param = identify_param(-101, continuous_values, ftype=Ftype.BOXCOX)
        self.assertEqual(param.ftype.value, "boxcox")

    def test_type_override_quantile(self):
        fvalue_map = variate_data()
        boxcocks_values = fvalue_map[BOXCOX_FEATURE_ID]

        param = identify_param(-101, boxcocks_values, ftype=Ftype.QUANTILE)
        self.assertEqual(param.ftype.value, "quantile")
