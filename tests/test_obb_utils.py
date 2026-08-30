from __future__ import annotations

import unittest

import numpy as np

from detector.obb_utils import polygon_to_xyxy, xywhr_to_polygon


class OBBUtilsTest(unittest.TestCase):
    def test_axis_aligned_conversion(self) -> None:
        polygon = xywhr_to_polygon([10, 20, 4, 8, 0])
        np.testing.assert_allclose(polygon_to_xyxy(polygon), [8, 16, 12, 24], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

