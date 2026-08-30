from __future__ import annotations

import unittest

from classifier.transforms import MultiViewTransform


class MultiViewTransformTests(unittest.TestCase):
    def test_creates_independent_views(self) -> None:
        calls: list[int] = []

        def transform(_value: object) -> int:
            calls.append(len(calls) + 1)
            return calls[-1]

        result = MultiViewTransform(transform, views=2)(object())
        self.assertEqual(result, (1, 2))
        self.assertEqual(len(calls), 2)

    def test_requires_two_views(self) -> None:
        with self.assertRaises(ValueError):
            MultiViewTransform(lambda value: value, views=1)

    def test_grayscale_rejects_color_only_augmentation(self) -> None:
        from classifier.transforms import build_transform

        config = {
            "color_mode": "grayscale",
            "augmentation": {"saturation": 0.1, "hue": 0.0},
        }
        with self.assertRaisesRegex(ValueError, "saturation=0 and hue=0"):
            build_transform(config, training=True)


if __name__ == "__main__":
    unittest.main()
