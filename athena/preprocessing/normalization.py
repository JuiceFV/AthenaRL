from dataclasses import asdict
import json
import numpy as np
from scipy import stats

from typing import Dict, List, Optional, Tuple
from athena.core.dtypes import Ftype
from athena.core.parameters import NormalizationParams
from athena.preprocessing.identify_types import identify_type, DEFAULT_MAX_UNIQUE_ENUM

BOXCOX_MARGIN = 1e-4
MISSING_VALUE = -228.228
BOXCOX_MAX_STDEV = 1e8
DEFAULT_MAX_QUANTILE_SIZE = 20
DEFAULT_QUANTILE_K2_THRESHOLD = 1000.0
MIN_SAMPLES_TO_IDENTIFY = 20
MAX_FVALUE = 11.513
MIN_FVALUE = MAX_FVALUE * -1


def identify_param(
    fname: int,
    fvalues: np.ndarray,
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold: int = DEFAULT_QUANTILE_K2_THRESHOLD,
    skip_boxcox: bool = False,
    skip_quantiles: bool = False,
    ftype: Optional[Ftype] = None
):
    """_summary_

    Args:
        fname (str): _description_
        fvalues (np.ndarray): _description_
        max_unique_enum_values (int, optional): _description_. Defaults to DEFAULT_MAX_UNIQUE_ENUM.
        quantile_size (int, optional): _description_. Defaults to DEFAULT_MAX_QUANTILE_SIZE.
        quantile_k2_threshold (int, optional): _description_. Defaults to DEFAULT_QUANTILE_K2_THRESHOLD.
        skip_boxcox (bool, optional): _description_. Defaults to False.
        skip_quantiles (bool, optional): _description_. Defaults to False.
        ftype (Optional[Ftype], optional): _description_. Defaults to None.

    Raises:
        TypeError: _description_
        RuntimeError: _description_

    TODO:
        - Add logging (tentatively for the normaltest k2 and p)
        - Add continuous action (like time spent or smth else)

    Returns:
        _type_: _description_
    """
    boxcox_required = ftype == Ftype.BOXCOX
    continuous_required = ftype == Ftype.CONTINUOUS
    quantile_required = ftype == Ftype.QUANTILE
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

    if ftype == Ftype.DO_NOT_PREPROCESS:
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)

    if ftype == Ftype.CONTINUOUS or boxcox_required or quantile_required:
        if min_fvalue == max_fvalue and not (boxcox_required or quantile_required):
            return NormalizationParams(Ftype.CONTINUOUS, None, 0, 0, 1, None, None, None, None)

        k2_original, p_original = stats.normaltest(fvalues)

        boxcox_shift = float(min_fvalue * -1)
        transformed_fvalues, lambda_ = stats.boxcox(np.maximum(fvalues + boxcox_shift, BOXCOX_MARGIN))
        k2_boxcox, p_boxcox = stats.normaltest(transformed_fvalues)

        if (lambda_ < 0.9 or lambda_ > 1.1 or boxcox_required) and not (continuous_required or quantile_required):

            if (k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold) or boxcox_required:
                stdev = float(np.std(transformed_fvalues, ddof=1))

                if (np.isfinite(stdev) and stdev < BOXCOX_MAX_STDEV and not np.isclose(stdev, 0)) or boxcox_required:
                    fvalues = transformed_fvalues
                    boxcox_lambda = float(lambda_)
        if boxcox_lambda is None or skip_boxcox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            ftype = Ftype.BOXCOX
        if (boxcox_lambda is None and k2_original > quantile_k2_threshold and not skip_quantiles and not continuous_required) or quantile_required:
            ftype = Ftype.QUANTILE
            quantiles = np.unique(
                stats.mstats.mquantiles(
                    fvalues,
                    np.arange(quantile_size + 1, dtype=np.float64) / float(quantile_size),
                    alphap=0.0,
                    betap=1.0
                )
            ).astype(float).tolist()

    if ftype in (Ftype.CONTINUOUS, Ftype.BOXCOX):
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)
        if not np.isfinite(stdev):
            return None
        fvalues /= stdev

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


def sort_features_by_normalization(normalization_params: Dict[int, NormalizationParams]) -> Tuple[List[int], List[int]]:
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
