from enum import IntEnum

from numpy import array, ndarray
from numpy.linalg import norm

class RayFactory:
    class Axis(IntEnum):
        Center = 0
        Left = -1
        Right = 1
        Posterior = -1
        Anterior = 1
        Inferior = -1
        Superior = 1

    @classmethod
    def make_ray(cls, frontal: Axis, sagittal: Axis, longitudinal: Axis) -> ndarray:
        ray = cls.make_default_point(frontal, sagittal, longitudinal)
        length = norm(ray)
        if length != 0.0:
            return ray / length
        else:
            return ray

    @classmethod
    def make_default_point(cls, frontal: Axis, sagittal: Axis, longitudinal: Axis) -> ndarray:
        return array([
            frontal * 1./3.,
            sagittal * 1./3.,
            longitudinal * 1./3.,
        ])

