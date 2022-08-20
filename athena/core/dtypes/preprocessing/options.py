from typing import Dict, List, Optional

from athena.core.dataclasses import dataclass
from athena.preprocessing import (DEFAULT_MAX_QUANTILE_SIZE,
                                  DEFAULT_MAX_UNIQUE_ENUM, DEFAULT_NSAMPLES,
                                  DEFAULT_QUANTILE_K2_THRESHOLD)


@dataclass
class PreprocessingOptions:
    nsamples: int = DEFAULT_NSAMPLES
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD
    skip_boxcox: bool = False
    skip_quantiles: bool = True
    feature_overrides: Optional[Dict[int, str]] = None
    table_sample: Optional[float] = None
    set_missing_value_to_zero: Optional[bool] = False
    allowed_features: Optional[List[int]] = None
    assert_allowlist_feature_coverage: bool = True
