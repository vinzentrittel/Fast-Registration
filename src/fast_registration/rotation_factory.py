from scipy.spatial.transform import Rotation
from numpy import apply_along_axis, arange, array, ndarray, newaxis

from .axis import Converter as AxisConverter

class RotationFactory:
    """
    # TODO: this should not be a class
    Utility class to match corresponding cube sections.
    For a given rotation matrix from RotationFactory.Rotations
    find the current subcube location axes.

    Usage:
    current_frontal = AxisValue.Left
    current_sagittal = AxisValue.Posterior
    current_longitudinal = AxisValue.Inferior
    n: int = ... # index to your 3x3 rotation matrix of interest
    transform_matrix = RotationFactory.Rotations[n]
    # ...
    # ... do something useful with that matrix
    # ...

    (
     new_frontal,
     new_sagittal,
     new_longitudinal,
    ) = RotationFactory.new_location(
     n,
     current_frontal,
     current_sagittal,
     current_longitudinal,
    )
    """
    Rotations: ndarray

    _RotationOrder = arange(24)[:, newaxis]
    _SectionPoints = array(AxisConverter.AxisConstellations)
    _StartConfigurations = array(tuple(
            Rotation.from_rotvec(n * array([0, 1, 0]), degrees=True).as_matrix()
            for n in (0, 90, 180, 270,)
        ) + tuple(
            Rotation.from_rotvec(n * array([1, 0, 0]), degrees=True).as_matrix()
            for n in (90, 270,)
        )
    )
    _ZRotations = array(tuple(
        Rotation.from_rotvec(n * array([0, 0, 1]), degrees=True).as_matrix()
        for n in (0, 90, 180, 270,)
    ))
    
    RotationIndexCorrespondences: ndarray
    _RotatedLocations: ndarray

    @classmethod
    def correspondence_from_rotations(cls, points: ndarray) -> ndarray:
        """
        For a set of 27 3d points, calculate all possible rotations.
        The resulting points will be rearranged. So that the new values
        lays in the correct subcube. Note: this will only work, if the
        original values were also in the correct subcube.

        For the order, visit: src/fast_registration/axis.py

        Arguments:
        points - numpy.ndarray of shape = (27, 3,), representing
                 a 3d point cloud.
        """
        points = cls._rotate(points)
        return points[cls._RotationOrder, cls.RotationIndexCorrespondences]

    @classmethod
    def _rotate(cls, points: ndarray) -> ndarray:
        """
        Return all rotations for all 'points'. 'points' is a set of three-
        dimensional values. That set of n points should have
        points.shape = (n, 3)

        The result will have ndarray.shape = (24, n, 3)
        """
        return cls.Rotations.dot(points.transpose()).transpose(0, 2, 1)

    @classmethod
    def _make_correspondence_order(cls, new_locations: ndarray) -> ndarray:
        result = apply_along_axis(AxisConverter.to_index, 2, new_locations).argsort()
        return result

    @classmethod
    def _make_rotations(cls) -> ndarray:
        """ Construct all 24 possible rotation matrices. """
        return array(tuple(
            config.dot(variant)
            for config in cls._StartConfigurations
            for variant in cls._ZRotations
        ))

    @classmethod
    def _make_location_correspondences(cls) -> ndarray:
        """ Make index look-up for all 24 possible cube rotations. """
        return apply_along_axis(
            cls._calc_section,
            2,
            cls._rotate(cls._SectionPoints),
        )

    @staticmethod
    def _calc_section(point: ndarray) -> ndarray:
        """
        Return the axis indices for a given one-dimensionam
        ndarray.
        """
        def calc_section_1d(value):
            if value > 0.29:
                return 1
            if value < -0.29:
                return -1
            return 0

        return array(tuple(map(calc_section_1d, point)))

# pylint: disable=protected-access
RotationFactory.Rotations = RotationFactory._make_rotations()
RotationFactory._RotatedLocations = RotationFactory._make_location_correspondences()
# TODO: holy mother of hacks... this is what I needed and I already implemented it :)
RotationFactory.RotationIndexCorrespondences = RotationFactory._make_correspondence_order(
    RotationFactory._RotatedLocations
)

if __name__ == "__main__":
    print(RotationFactory.RotationIndexCorrespondences)
