from __future__ import annotations

from typing import Callable, Tuple

from numpy import (
    array,
    argmax,
    argmin,
    cross,
    identity,
    ndarray,
    pi,
    subtract,
)
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation
from vtk import vtkPolyData
from vtk.util.numpy_support import vtk_to_numpy

class BoundingBox:
    def __init__(self, geometry: vtkPolyData) -> None:
        self.geometry = vtk_to_numpy(geometry.GetPoints().GetData())
        self.axes = identity(3)
        self.primary_normal = self.calc_widest_projection(self.geometry)
        self.secondary_normal = self.calc_outer_points_second_order()

        self.ternary_normal = self.calc_normalized_axis(
            self.secondary_normal,
            self.primary_normal,
            operation=cross,
        )
        self.secondary_normal = self.calc_normalized_axis(
            self.ternary_normal,
            self.primary_normal,
            operation=cross,
        )
        in_place_projection = self.geometry.dot(self.axes)
        minumums = in_place_projection.min(axis=0)
        maximums = in_place_projection.max(axis=0)

        self.length = maximums - minumums
        self.center = self.calc_center(minumums, maximums)
        self.inverse_matrix = self.calc_normalization_matrix()
        self.transform_matrix = self.calc_inverse_normalization_matrix()

        from vtk import vtkSTLWriter, vtkTransform, vtkTransformPolyDataFilter
        transform = vtkTransform()
        transform.SetMatrix(self.inverse_matrix.flatten().tolist())
        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetInputData(geometry)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

    @property
    def primary_normal(self) -> ndarray:
        return self.axes[0, :]
    @primary_normal.setter
    def primary_normal(self, new_vector: ndarray) -> None:
        self.axes[0] = new_vector
    @property
    def secondary_normal(self) -> ndarray:
        return self.axes[1]
    @secondary_normal.setter
    def secondary_normal(self, new_vector: ndarray) -> None:
        self.axes[1] = new_vector
    @property
    def ternary_normal(self) -> ndarray:
        return self.axes[2]
    @ternary_normal.setter
    def ternary_normal(self, new_vector: ndarray) -> None:
        self.axes[2] = new_vector

    @staticmethod
    def calc_normalized_axis(
        first_vector: ndarray,
        second_vector: ndarray,
        operation: Callable[[ndarray, ndarray], ndarray]=subtract,
    ) -> ndarray:
        non_normalized_result = operation(first_vector, second_vector)
        return non_normalized_result / norm(non_normalized_result)

    @classmethod
    def calc_widest_projection(cls, geometry: ndarray, transform: ndarray=None) -> ndarray:
        if transform is None:
            target = geometry
        else:
            target = geometry.dot(transform)
        min_ = argmin(target, axis=0)
        max_ = argmax(target, axis=0)
        widest_dimension = argmax(
            norm(geometry[max_] - geometry[min_], axis=1)
        )
        return cls.calc_normalized_axis(
            geometry[max_[widest_dimension]],
            geometry[min_[widest_dimension]],
        )

    def calc_outer_points_second_order(self) -> ndarray:
        orthogonal_axis = self.calc_normalized_axis(self.primary_normal, array([0, 0, 1]), cross)
        candidate_axes = array(tuple(
            Rotation.from_rotvec(
                45.0 * n * self.primary_normal,
                degrees=True,
            ).apply(
                orthogonal_axis,
            )
            for n in range(4)
        )).transpose()

        return self.calc_widest_projection(self.geometry, transform=candidate_axes)

    def calc_center(self, minimums: ndarray, maximums: ndarray) -> ndarray:
        return (0.5 * (minimums + maximums)).dot(inv(self.axes))

    def calc_normalization_matrix(self) -> ndarray:
        scale = array([
            2.0 / self.length[0],
            2.0 / self.length[1],
            2.0 / self.length[2],
            1.0,
        ]) * identity(4)
        translation = array([
            [1, 0, 0, -self.center[0]],
            [0, 1, 0, -self.center[1]],
            [0, 0, 1, -self.center[2]],
            [0, 0, 0, 1],
        ])
        normalization = array([
            [ *self.axes[:, 0], 0 ],
            [ *self.axes[:, 1], 0 ],
            [ *self.axes[:, 2], 0 ],
            [ 0, 0, 0, 1 ],
        ])

        matrix = scale.dot(normalization).dot(translation)
        return matrix

    def calc_inverse_normalization_matrix(self) -> ndarray:
        return inv(self.inverse_matrix)


if __name__ == "__main__":
    from main import load_stl
    from features import Slicer

    vertebra = load_stl( '../data/L5.stl')
    from timeit import default_timer
    start = default_timer()
    bb = BoundingBox(vertebra)
    end = default_timer()
    print(end - start)
