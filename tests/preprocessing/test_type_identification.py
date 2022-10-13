import unittest

from athena.preprocessing.identify_types import Ftype, identify_type
from tests.preprocessing.utils import (BINARY_FEATURE_ID, BOXCOX_FEATURE_ID,
                                       CONTINUOUS_FEATURE_ID, ENUM_FEATURE_ID,
                                       PROBABILITY_FEATURE_ID,
                                       QUANTILE_FEATURE_ID, variate_data)


class TestTypeIdentification(unittest.TestCase):
    def test_identification(self):
        feature_value_map = variate_data()

        types = {}
        for name, values in feature_value_map.items():
            types[name] = identify_type(values)

        self.assertEqual(types[BINARY_FEATURE_ID], Ftype.BINARY)
        self.assertEqual(types[CONTINUOUS_FEATURE_ID], Ftype.CONTINUOUS)

        self.assertEqual(types[BOXCOX_FEATURE_ID], Ftype.CONTINUOUS)

        self.assertEqual(types[QUANTILE_FEATURE_ID], Ftype.CONTINUOUS)
        self.assertEqual(types[ENUM_FEATURE_ID], Ftype.ENUM)
        self.assertEqual(types[PROBABILITY_FEATURE_ID], Ftype.PROBABILITY)
