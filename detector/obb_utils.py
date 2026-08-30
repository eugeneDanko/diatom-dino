"""Small NumPy-only helpers for oriented bounding boxes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def xywhr_to_polygon(box: Iterable[float]) -> np.ndarray:
    """Convert ``center_x, center_y, width, height, radians`` to four points."""

    cx, cy, width, height, angle = map(float, box)
    corners = np.asarray(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ],
        dtype=np.float32,
    )
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    return corners @ rotation.T + np.asarray([cx, cy], dtype=np.float32)


def polygon_to_xyxy(polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    return np.asarray(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
        dtype=np.float32,
    )


def scale_polygon(polygon: np.ndarray, scale_x: float, scale_y: float | None = None) -> np.ndarray:
    scale_y = scale_x if scale_y is None else scale_y
    result = np.asarray(polygon, dtype=np.float32).copy().reshape(-1, 2)
    result[:, 0] *= float(scale_x)
    result[:, 1] *= float(scale_y)
    return result


def clip_xyxy(box: Iterable[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.asarray(
        [
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        ],
        dtype=np.float32,
    )

