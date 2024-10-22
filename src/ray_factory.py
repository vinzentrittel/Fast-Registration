from enum import IntEnum
from typing import Tuple

from numpy import array, ndarray, isclose, newaxis
from numpy.linalg import norm

from axis import Converter as AxisConverter

def _make_default_rays() -> ndarray:
    offsets = _make_default_points()
    lengths = norm(offsets, axis=-1)[:, newaxis]
    lengths[lengths == 0.0] = 1.0
    return offsets / lengths

def _make_default_points() -> ndarray:
    return array(1.0/3.0 * array(AxisConverter.AxisConstellations))

DefaultPoints = _make_default_points()
DefaultRays = _make_default_rays()

