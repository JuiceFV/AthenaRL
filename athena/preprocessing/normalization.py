import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

import athena.core.dtypes as adt
from athena.core.dtypes import Ftype
from athena.core.parameters import NormalizationParams
from athena.preprocessing import (BOXCOX_MARGIN, BOXCOX_MAX_STDEV,
                                  DEFAULT_MAX_QUANTILE_SIZE,
                                  DEFAULT_MAX_UNIQUE_ENUM,
                                  DEFAULT_QUANTILE_K2_THRESHOLD,
                                  MIN_SAMPLES_TO_IDENTIFY)
from athena.preprocessing.identify_types import identify_type

logger = logging.getLogger(__name__)


def identify_param(
    fname: str,
    fvalues: np.ndarray,
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold: int = DEFAULT_QUANTILE_K2_THRESHOLD,
    skip_boxcox: bool = False,
    skip_quantiles: bool = False,
    ftype: Optional[Ftype] = None
) -> NormalizationParams:
    r"""
    Infers statistics (metadata) from given sample of feature values. Recently,
    Ioffe & Szegedy have shown in their `work <https://arxiv.org/pdf/1502.03167.pdf>`_
    that normalization mitigating issues from varying feature scales and distributions which
    has shown to improve model performance and convergence.

    1. Identify feature type (``float``, ``int``, ``enum`` etc.)
    2. Check if sampled feature values are normally distributed.
    3. In case they not apply three transformations

      * BoxCox

        .. math::
            y = \mathbf{1}_{\lambda \neq 0}\left[\frac{x^{\lambda} - 1}{\lambda}, \log{x}\right]

      * `Quantile <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html>`_
      * Gauss

        .. math::

            y = \frac{x - \mu{(x)}}{\sigma{(x)}}

    While training and serving pre-computed statistics are applied in
    :class:`athena.preprocessing.preprocessor.Preprocessor`.

    Args:
        fname (str): Feature name.
        fvalues (np.ndarray): Sample of feature values.
        max_unique_enum_values (int, optional): Maximum number of unique categories for categorical feature.
          Defaults to ``DEFAULT_MAX_UNIQUE_ENUM``.
        quantile_size (int, optional): Number of quantiles splits the raw data during transformation.
          Defaults to ``DEFAULT_MAX_QUANTILE_SIZE``.
        quantile_k2_threshold (int, optional): Skewness (:math:`s^2`) and Kurtosis (:math:`k^2`) threshold of
          `normal test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html>`_.
          Defaults to ``DEFAULT_QUANTILE_K2_THRESHOLD``.
        skip_boxcox (bool, optional): Forcibly skip box cox transformation. Defaults to ``False``.
        skip_quantiles (bool, optional): Forcibly skip quantile transformation. Defaults to ``False``.
        ftype (Optional[Ftype], optional): Manual type for the feature. Defaults to ``None``.

    Raises:
        TypeError: If feature type is undefined.
        RuntimeError: If number of samples isn't enough to infer some statistics.

    Returns:
        NormalizationParams: Accumulated metadata.
    """
    boxcox_required = ftype == Ftype.BOXCOX
    continuous_required = ftype == Ftype.CONTINUOUS
    quantile_required = ftype == Ftype.QUANTILE

    # If manual type is not given, identify it automatically
    if ftype is None:
        ftype = identify_type(fvalues, max_unique_enum_values)

    boxcox_lambda: Optional[float] = None
    boxcox_shift = 0.0
    mean = 0.0
    stdev = 1.0
    possible_values: Optional[np.ndarray] = None
    quantiles: Optional[np.ndarray] = None

    if ftype not in Ftype:
        raise TypeError(f"Unknown type {ftype.value}")
    if len(fvalues) < MIN_SAMPLES_TO_IDENTIFY:
        raise RuntimeError("Insufficient information to identify parameter.")

    min_fvalue = float(np.min(fvalues))
    max_fvalue = float(np.max(fvalues))

    # If it's not required to normalize feature
    # compute its gauss statistics.
    if ftype == Ftype.DO_NOT_PREPROCESS:
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        # NOTE: Due to computations occures over sample
        # use degree of freedom equal to 1, means that
        # we calculate statistic (normalization factor N-1)
        # not parameter (normalization factor N)
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)

    # If feature is continuous make sure it's normalized
    if ftype == Ftype.CONTINUOUS or boxcox_required or quantile_required:
        # If fake vectors (padding for example) is given then no normalization is applied
        if min_fvalue == max_fvalue and not (boxcox_required or quantile_required):
            return NormalizationParams(Ftype.CONTINUOUS, None, 0, 0, 1, None, None, None, None)

        # identify the difference between current feature distribution
        # and Normal distribution using normal tes. It returns k2 = s^2 + k^2
        # where s^2 - skewness (left/right) and k^2 - kurtosis (peak up/down).
        k2_original, p_original = stats.normaltest(fvalues)

        # apply box cox transformation and perform normal test to the transformed values
        boxcox_shift = float(min_fvalue * -1)
        transformed_fvalues, lambda_ = stats.boxcox(np.maximum(fvalues + boxcox_shift, BOXCOX_MARGIN))
        k2_boxcox, p_boxcox = stats.normaltest(transformed_fvalues)
        logger.info(f"Feature stats; Original K2: {k2_original} P: {p_original} BoxCox K2: {k2_boxcox} P: {p_boxcox}")

        # ===Box Cox Normalizaton===
        # In case transformation is tangible (lambda alter is higher than 0.1) and no other restrictions defined
        # mark that it's been successful and should be applied during training and inference.
        if (lambda_ < 0.9 or lambda_ > 1.1 or boxcox_required) and not (continuous_required or quantile_required):

            # We must be sure that box cox differs enough and it's much more closer to the normal distribution
            # than original values distribution. Besides, it also possible that data variates too much, s.t.
            # quantile normalization is more suitable.
            if (k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold) or boxcox_required:
                stdev = float(np.std(transformed_fvalues, ddof=1))

                # For the few sample the data may be too noise or oposite too dense, so make sure that it's
                # distributed smoothly.
                if (np.isfinite(stdev) and stdev < BOXCOX_MAX_STDEV and not np.isclose(stdev, 0)) or boxcox_required:
                    fvalues = transformed_fvalues
                    boxcox_lambda = float(lambda_)
        if boxcox_lambda is None or skip_boxcox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            ftype = Ftype.BOXCOX

        # ===Quantile Normalization===
        # In case box cox hasn't been applied but original feature
        # distribution is still too far from gauss distribution apply
        # quantile normalization
        if (
            boxcox_lambda is None and
            k2_original > quantile_k2_threshold and
            not skip_quantiles and not continuous_required
        ) or quantile_required:
            ftype = Ftype.QUANTILE
            # Get quantiles emperically extracted from the feature values
            # (alphap, betap) are coefficient of beta distribution which
            # are used to interpolate cdf of given distribution. In this
            # case where alphap = 0, betap = 1 the interpolation is linear.
            quantiles = np.unique(
                stats.mstats.mquantiles(
                    fvalues,
                    np.arange(quantile_size + 1, dtype=np.float64) / float(quantile_size),
                    alphap=0.0,
                    betap=1.0
                )
            ).astype(float).tolist()

    # ===Apply Gauss normalization===
    if ftype in (Ftype.CONTINUOUS, Ftype.BOXCOX):
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)
        if not np.isfinite(stdev):
            logger.info(f"Standard deviation is infinite for feature {fname}")
            return None
        fvalues /= stdev

    # Infer all unique values of catigorical feature
    if ftype == Ftype.ENUM:
        possible_values = np.unique(fvalues.astype(int)).astype(int).tolist()

    return NormalizationParams(
        ftype=ftype,
        boxcox_lambda=boxcox_lambda,
        boxcox_shift=boxcox_shift,
        mean=mean,
        stdev=stdev,
        possible_values=possible_values,
        quantiles=quantiles,
        min_value=min_fvalue,
        max_value=max_fvalue
    )


def sort_features_by_normalization(
    normalization_params: Dict[int, NormalizationParams]
) -> Tuple[List[int], List[int]]:
    sorted_features: List[int] = []
    fheaders: List[int] = []
    if not isinstance(list(normalization_params.keys())[0], int):
        raise TypeError("Feature id must be integer type.")
    for ftype in Ftype:
        fheaders.append(len(sorted_features))
        for fid in sorted(normalization_params.keys()):
            norm = normalization_params[fid]
            if norm.ftype == ftype:
                sorted_features.append(fid)
    return sorted_features, fheaders


def serialize(params: Dict[int, NormalizationParams]) -> Dict[int, str]:
    return {fid: json.dumps(asdict(fparams)) for fid, fparams in params.items()}


def deseialize(params_json: Dict[int, str]) -> Dict[int, NormalizationParams]:
    params: Dict[int, NormalizationParams] = {}
    for fid, fparmas in params_json.items():
        norm_params = NormalizationParams(**json.loads(fparmas))
        if norm_params.ftype == Ftype.ENUM:
            if norm_params.possible_values is None:
                raise RuntimeError(f"Expected values for {Ftype.ENUM} feature type")
        params[int(fid)] = norm_params
    return params


def get_normalization_data_dim(normalization_params: Dict[int, NormalizationParams]) -> int:
    return sum(map(lambda np: len(np.possible_values) if np.ftype == Ftype.ENUM else 1, normalization_params.values()))


def get_feature_config(continuous_features: Optional[List[Tuple[int, str]]]) -> adt.ModelFeatureConfig:
    continuous_features = continuous_features or []
    continuous_features_infos = [
        adt.ContinuousFeatureInfo(name=fname, feature_id=fid)
        for fid, fname in continuous_features
    ]
    return adt.ModelFeatureConfig(continuous_feature_infos=continuous_features_infos)


def get_feature_norm_metadata(fname: str, fvalue_list: List[float], norm_params: Dict[str, Any]):
    logger.info(f"Extracting normalization for feature: {fname}")
    nfeatures = len(fvalue_list)
    if nfeatures < MIN_SAMPLES_TO_IDENTIFY:
        logger.warning("Number of samples isn't enough to extract metadata.")
        return None
    feature_override = None
    if norm_params["feature_overrides"] is not None:
        feature_override = norm_params["feature_overrides"].get(fname, None)
    feature_override = feature_override or norm_params.get("default_feature_override", None)

    fvalues = np.array(fvalue_list, dtype=np.float32)
    if np.any(np.isinf(fvalues)):
        raise ValueError(f"Feature {fname} contains infinity.")
    if np.any(np.isnan(fvalues)):
        raise ValueError(f"Feature {fname} contains NaN.")

    normalization_params = identify_param(
        fname,
        fvalues,
        norm_params["max_unique_enum_values"],
        norm_params["quantile_size"],
        norm_params["quantile_k2_threshold"],
        norm_params["skip_boxcox"],
        norm_params["skip_quantiles"],
        feature_override
    )
    logger.info(f"Feature {fname} normalization {normalization_params}")
    return normalization_params


def infer_normalization(
    max_unique_enum_values: int,
    qunatile_size: int,
    quantile_k2_threshold: float,
    skip_box_cox: bool = False,
    skip_quantiles: bool = False,
    feature_overrides: Optional[Dict[int, str]] = None,
    allowed_features: Optional[List[int]] = None,
    assert_allowlist_feature_coverage: bool = True
) -> Callable[[List], Dict[int, NormalizationParams]]:
    norm_params = {
        "max_unique_enum_values": max_unique_enum_values,
        "quantile_size": qunatile_size,
        "quantile_k2_threshold": quantile_k2_threshold,
        "skip_boxcox": skip_box_cox,
        "skip_quantiles": skip_quantiles,
        "feature_overrides": feature_overrides
    }

    allowed_features = set(allowed_features or [])

    def assert_allowed_features(params: Dict[int, NormalizationParams]) -> None:
        if not allowed_features:
            return
        allowed_features_set = {int(fid) for fid in allowed_features}
        available_features = set(params.keys())
        if allowed_features_set != available_features:
            raise RuntimeError(
                f"Could not identify preprocessing type for the following features: "
                f"{allowed_features_set - available_features}; Extra features: "
                f"{available_features - allowed_features_set};"
            )

    def process(rows: List) -> Dict[int, NormalizationParams]:
        params = {}
        for row in rows:
            if "fname" not in row or "fvalues" not in row:
                raise AttributeError(f"Feature name or/and values are missing; {row}")
            norm_metadata = get_feature_norm_metadata(
                row["fname"], row["fvalues"], norm_params
            )
            if norm_metadata is not None and not allowed_features or row["fname"] in allowed_features:
                params[row["fname"]] = norm_metadata

        if assert_allowlist_feature_coverage:
            assert_allowed_features(params)
        return params

    return process
