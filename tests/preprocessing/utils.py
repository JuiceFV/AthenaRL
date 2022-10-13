from typing import Dict, Union

import numpy as np
from athena.core.dtypes import Ftype
from athena.core.parameters import NormalizationParams
from athena.preprocessing import BOXCOX_MARGIN, MAX_FVALUE, MIN_FVALUE, MISSING_VALUE
from scipy import stats, special

BINARY_FEATURE_ID = 1
BINARY_FEATURE_ID_2 = 2
BOXCOX_FEATURE_ID = 3
CONTINUOUS_FEATURE_ID = 4
CONTINUOUS_FEATURE_ID_2 = 5
ENUM_FEATURE_ID = 6
PROBABILITY_FEATURE_ID = 7
QUANTILE_FEATURE_ID = 8


def id2type(id: int) -> str:
    if id in (BINARY_FEATURE_ID, BINARY_FEATURE_ID_2):
        return "binary"
    if id == BOXCOX_FEATURE_ID:
        return "boxcox"
    if id in (CONTINUOUS_FEATURE_ID, CONTINUOUS_FEATURE_ID_2):
        return "continuous"
    if id == ENUM_FEATURE_ID:
        return "enum"
    if id == PROBABILITY_FEATURE_ID:
        return "probability"
    if id == QUANTILE_FEATURE_ID:
        return "quantile"
    assert False, f"Invalid feature id: {id}"


def variate_data() -> Dict[int, np.ndarray]:
    np.random.seed(1)
    feature_value_map = {}
    feature_value_map[BINARY_FEATURE_ID] = stats.bernoulli.rvs(0.5, size=10000).astype(np.float32)
    feature_value_map[BINARY_FEATURE_ID_2] = stats.bernoulli.rvs(0.5, size=10000).astype(np.float32)
    feature_value_map[CONTINUOUS_FEATURE_ID] = stats.norm.rvs(size=10000).astype(np.float32)
    feature_value_map[CONTINUOUS_FEATURE_ID_2] = stats.norm.rvs(size=10000).astype(np.float32)
    feature_value_map[BOXCOX_FEATURE_ID] = stats.expon.rvs(size=10000).astype(np.float32)
    feature_value_map[ENUM_FEATURE_ID] = (stats.randint.rvs(0, 10, size=10000) * 1000).astype(np.float32)
    feature_value_map[QUANTILE_FEATURE_ID] = np.concatenate(
        (stats.norm.rvs(size=5000), stats.expon.rvs(size=5000))
    ).astype(np.float32)
    feature_value_map[PROBABILITY_FEATURE_ID] = np.clip(stats.beta.rvs(a=2.0, b=2.0, size=10000), 0.01, 0.99)
    return feature_value_map


class NumpyFeaturePreprocessor(object):
    @staticmethod
    def value_to_quantile(original_value: np.float32, quantiles: np.ndarray) -> np.float32:
        if original_value <= quantiles[0]:
            return 0.0
        if original_value >= quantiles[-1]:
            return 1.0
        nquantiles = float(len(quantiles) - 1)
        right = np.searchsorted(quantiles, original_value)
        left = right - 1
        interpolated = (
            left + (
                (original_value - quantiles[left]) / (quantiles[right] + 1e-6 - quantiles[left])
            )
        ) / nquantiles
        return interpolated

    @classmethod
    def preprocess_feature(cls, feature: np.float32, params: NormalizationParams) -> np.float32:
        is_not_missing = 1 - np.isclose(feature, MISSING_VALUE)
        if params.ftype == Ftype.BINARY:
            return ((feature != 0) * is_not_missing).astype(np.float32)
        if params.boxcox_lambda is not None:
            feature = stats.boxcox(np.maximum(feature + params.boxcox_shift, BOXCOX_MARGIN), params.boxcox_lambda)
        if params.ftype == Ftype.PROBABILITY:
            feature = np.clip(feature, 0.01, 0.99)
            feature = special.logit(feature)
        elif params.ftype == Ftype.QUANTILE:
            transformed_feature = np.zeros_like(feature)
            for i in range(feature.shape[0]):
                transformed_feature[i] = cls.value_to_quantile(feature[i], params.quantiles)
            feature = transformed_feature
        elif params.ftype == Ftype.ENUM:
            possible_values = params.possible_values
            value_feature_mapping: Dict[int, int] = {}
            for i, possible_value in enumerate(possible_values):
                value_feature_mapping[possible_value] = i
            output_feature = np.zeros((len(feature), len(possible_values)))
            for i, value in enumerate(feature):
                if abs(value - MISSING_VALUE) < 1e-2:
                    continue
                output_feature[i][value_feature_mapping[value]] = 1.0
            return output_feature
        else:
            feature = feature - params.mean
            feature /= params.stdev
            feature = np.clip(feature, MIN_FVALUE, MAX_FVALUE)
        feature *= is_not_missing
        return feature

    @classmethod
    def preprocess(
        cls, features: Dict[int, Union[np.float32, int]], params: Dict[int, NormalizationParams]
    ) -> Dict[int, np.float32]:
        res: Dict[int, np.float32] = {}
        for fid in features:
            res[fid] = cls.preprocess_feature(features[fid], params[fid])
        return res

    @classmethod
    def preprocess_array(
        cls, arr: np.ndarray, features: Dict[int, Union[np.float32, int]], params: Dict[int, NormalizationParams]
    ) -> np.ndarray:
        assert len(arr.shape) == 2 and arr.shape[1] == len(features)
        preprocessed_values = [
            cls.preprocess(
                {
                    fid: fvalue
                    for fid, fvalue in zip(features, row)
                }, params
            )
            for row in arr
        ]
        return np.array([[pv[fid] for fid in features] for pv in preprocessed_values], dtype=np.float32)
