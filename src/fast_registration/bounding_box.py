from __future__ import annotations

from dataclasses import dataclass, field
from numpy import (
    abs as abs_,
    array,
    cross,
    identity,
    ndarray,
    newaxis,
    ones,
    zeros,
)
from numpy.linalg import inv, norm, pinv, solve
from vtk import vtkPolyData, vtkSTLReader
from vtk.util.numpy_support import vtk_to_numpy
from slic3r_display import Slic3rBoxRepresentable

from .axis import eigenvectors, extrema
from .util import load_stl

@dataclass
class BoundingBox:
    """
    For a given geometry, create an oriented bounding box around it.
    The bounding box does not obey to the usual metrics. Its orientation
    comes from a projection of an input mesh geometry onto its eigenvectors.

    Considered are the maxima and minima along primary and secondary axis.
    From their projection on these axis an orthogonal ternary axis is constructed.
    Then the secondary axis is recalculated as another orthogonal to primary and
    secondary axis.

    Then the axes are normalized.

    Properties:
    geometry - numpy.ndarray representation of the mesh geometry
    axes - ortho-normal representation of the vectors (see above=
    length - length of the axes, constructing the oriented bounding box
    center - mid point to all outer extrema
    transform_matrix - 4x4 transformation matrix to make the geometry into
                       an identity cube.
    """
    geometry: vtkPolyData
    axes: ndarray = field(default_factory=lambda: identity(3))
    length: ndarray = field(default_factory=lambda: ones(3))
    center: ndarray = field(default_factory=lambda: zeros(3))
    transform_matrix: ndarray = field(default_factory=lambda: identity(3))

    def __post_init__(self) -> None:
        self._points = vtk_to_numpy(self.geometry.GetPoints().GetData())
        eigenvectors_ = eigenvectors(self.geometry)

        extremas = [extrema(self._points, eigenvectors_[id]) for id in range(3)]

        self.axes[0] = self._vector_from_extrema(*extremas[0])
        self.axes[1] = self._vector_from_extrema(*extremas[1])
        self.axes[2] = cross(self.axes[0], self.axes[1])
        self.axes[1] = cross(self.axes[2], self.axes[0])
        self.axes /= norm(self.axes, axis=1)[:, newaxis]

        # assert all axes point in a postive direction
        directions = self.axes.dot(ones(3))
        self.axes *= (directions / abs_(directions))[:, newaxis]

        # assert axis order x, y, z in a right-hand coordinate system
        if cross(self.axes[2], self.axes[0]).dot(self.axes[1]) < 0.0:
            self.axes = self.axes[[1, 0, 2]]

        in_place_projection = self._points.dot(inv(self.axes))
        minimums = in_place_projection.min(axis=0)
        maximums = in_place_projection.max(axis=0)
        self.length = maximums - minimums
        self.center = solve(self.axes, minimums) + self.axes.dot(self.length / 2.0)

        self.transform_matrix = self._calc_normalization_matrix()

    @property
    def inverse_transform_matrix(self) -> ndarray:
        """
        Non trivial 4x4 inversion matrix, that takes translation into account.
        """
        offset = -self.axes.dot(self.length) / 2.0 + self.center
        translation = array([
            [1, 0, 0, offset[0]],
            [0, 1, 0, offset[1]],
            [0, 0, 1, offset[2]],
            [0, 0, 0, 1],
        ])
        scale = identity(4) * [
            1.0 / self.length[0],
            1.0 / self.length[1],
            1.0 / self.length[2],
            1.0,
        ]
        normalization = inv([
            [ *self.axes[:, 0], 0 ],
            [ *self.axes[:, 1], 0 ],
            [ *self.axes[:, 2], 0 ],
            [ 0, 0, 0, 1 ],
        ])
        half_in_size = array([[0.5,0,0,0], [0,0.5,0,0],[0,0,0.5,0],[0,0,0,1]])
        center_to_origin = array([[1,0,0,0.5],[0,1,0,0.5],[0,0,1,0.5],[0,0,0,1]])

        return translation.dot(inv(normalization)).dot(inv(scale)).dot(center_to_origin).dot(half_in_size)

    @staticmethod
    def _normalize(vector: ndarray) -> ndarray:
        return vector / norm(vector)

    @staticmethod
    def _vector_from_extrema(min_: ndarray, max_: ndarray) -> ndarray:
        vector = max_ - min_
        return vector / norm(vector)

    def _calc_center(self, minimums: ndarray, maximums: ndarray) -> ndarray:
        return (0.5 * (minimums + maximums)).dot(inv(self.axes))

    def _calc_normalization_matrix(self) -> ndarray:
        offset = self.axes.dot(self.length) / 2.0 - self.center
        translation = array([
            [1, 0, 0, offset[0]],
            [0, 1, 0, offset[1]],
            [0, 0, 1, offset[2]],
            [0, 0, 0, 1],
        ])
        scale = identity(4) * [
            1.0 / self.length[0],
            1.0 / self.length[1],
            1.0 / self.length[2],
            1.0,
        ]
        normalization = inv([
            [ *self.axes[:, 0], 0 ],
            [ *self.axes[:, 1], 0 ],
            [ *self.axes[:, 2], 0 ],
            [ 0, 0, 0, 1 ],
        ])
        double_in_size = array([[2,0,0,0], [0,2,0,0],[0,0,2,0],[0,0,0,1]])
        center_to_origin = array([[1,0,0,-0.5],[0,1,0,-0.5],[0,0,1,-0.5],[0,0,0,1]])

        return double_in_size.dot(center_to_origin).dot(scale).dot(normalization).dot(translation)
        return scale.dot(translation).dot(normalization) # this kind of works
        return scale.dot(normalization).dot(translation)

if __name__ == "__main__":
    vertebra = load_stl( 'data/L5.stl')
    from timeit import default_timer
    start = default_timer()
    bb = BoundingBox(vertebra)
    end = default_timer()
    print(end - start)
