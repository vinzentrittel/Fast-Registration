from scipy.spatial.transform import Rotation
from numpy import apply_along_axis, arange, array, isclose, ndarray, newaxis, sort

from identity_clipper import Axis

class RotationFactory:
    """
    Utility class to match corresponding cube sections.
    For a given rotation matrix from RotationFactory.Rotations
    find the current subcube location axes.

    Usage:
    current_frontal = Axis.Left
    current_sagittal = Axis.Posterior
    current_longitudinal = Axis.Inferior
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
    _SectionPoints = array(tuple(
        (frontal, sagittal, longitudinal,)
        for frontal in (Axis.Left, Axis.Center, Axis.Right,)
        for sagittal in (Axis.Posterior, Axis.Center, Axis.Anterior,)
        for longitudinal in (Axis.Inferior, Axis.Center, Axis.Superior,)
    ))
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
    
    _CorrespondenceOrder: ndarray
    _RotatedLocations: ndarray

    @classmethod
    def correspondence_from_rotations(cls, points: ndarray) -> ndarray:
        points = cls._rotate(points)
        return points[cls._RotationOrder, cls._CorrespondenceOrder]

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
        result = apply_along_axis(cls.axes_to_index, 2, new_locations).argsort()
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
        Return the Axis indices for a given one-dimensionam
        ndarray.
        """
        def calc_section_1d(value):
            if value > 0.29:
                return 1
            if value < -0.29:
                return -1
            return 0

        return array(tuple(map(calc_section_1d, point)))

    @staticmethod
    def axes_to_index(axes: ndarray) -> int:
        """
        Return index for RotationFactory._RotatedLocations based on the origin
        location axis constellation.
        """
        return 9 * axes[0] + 3 * axes[1] + axes[2] + 13


RotationFactory.Rotations = RotationFactory._make_rotations()
RotationFactory._RotatedLocations = RotationFactory._make_location_correspondences()
RotationFactory._CorrespondenceOrder = RotationFactory._make_correspondence_order(RotationFactory._RotatedLocations)

if __name__ == '__main__':
    print(RotationFactory.correspondence_from_rotations(RotationFactory._SectionPoints))
