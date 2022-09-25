r"""
Defines the DPP options which will be set while configuration.
"""
from typing import Dict, List, Optional

from athena.core.dataclasses import dataclass
from athena.preprocessing import (DEFAULT_MAX_QUANTILE_SIZE,
                                  DEFAULT_MAX_UNIQUE_ENUM, DEFAULT_NSAMPLES,
                                  DEFAULT_QUANTILE_K2_THRESHOLD)


@dataclass
class PreprocessingOptions:
    r"""
    Options used to preprocess and transform data. For the details
    follow :func:`athena.preprocessing.normalization.identify_param`.

    .. warning::

        Currently ``set_missing_value_to_zero``, ``allowed_features``
        ``assert_allowlist_feature_coverage`` not in use. They will
        be implemented in further versions.

        1. ``set_missing_value_to_zero`` - intends to replace missing
        values (``x_presence == False``) with 0. During sparse data
        processing.

        2. ``allowed_features`` - implements "check-box" for features.
        To consider only given features ModelManager must implement
        the logic of processing these features.

        3. ``assert_allowlist_feature_coverage`` - asserts if processing
        features are ``allowed_features``.
    """

    #: Number of samples used to infer statistics from raw data.
    nsamples: int = DEFAULT_NSAMPLES

    #: Maximum unique values of a catigorical feature.
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM

    #: Number of quantiles splits the raw data during transformation.
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE

    #: Skewness (:math:`s^2`) and Kurtosis (:math:`k^2`) threshold
    #: of `normal test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html>`_.
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD

    #: Forcibly skip box cox transformation.
    skip_boxcox: bool = False

    #: Forcibly skip quantile transformation.
    skip_quantiles: bool = True

    #: Optionally pre-defined feature types.
    feature_overrides: Optional[Dict[int, str]] = None

    #: Part of dataset intended to be used in the training process.
    table_sample: Optional[float] = None

    #: Optionally replcae missing values to zero in sparse data.
    set_missing_value_to_zero: Optional[bool] = False

    #: Optionally selected features which will be used during process.
    allowed_features: Optional[List[int]] = None

    #: Optionally check if processing features are the same as ``allowed_features``.
    assert_allowlist_feature_coverage: bool = True
