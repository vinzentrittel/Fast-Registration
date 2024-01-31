from __future__ import annotations

from typing import Tuple

from numpy import array, argmax, argmin, cross, identity, ndarray, pi
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from vtk import vtkPolyData
from vtk.util.numpy_support import vtk_to_numpy

class BoundingBox:
    def __init__(self, geometry: vtkPolyData, center_of_mass: ndarray) -> None:
        self.geometry = vtk_to_numpy(geometry.GetPoints().GetData())
        self.center_of_mass = center_of_mass

        outer_points_first_order = self.calc_outer_points_first_order()
        self.primary_normal = outer_points_first_order[1] - outer_points_first_order[0]
        self.primary_normal /= norm(self.primary_normal)

        outer_points_second_order = self.calc_outer_points_second_order()
        self.secondary_normal = outer_points_second_order[1] - outer_points_second_order[0]
        self.secondary_normal /= norm(self.secondary_normal)

        self.ternary_normal = cross(self.secondary_normal, self.primary_normal)
        self.ternary_normal /= norm(self.ternary_normal)

        self.secondary_normal = cross(self.ternary_normal, self.primary_normal)
        self.secondary_normal /= norm(self.secondary_normal)

        self.length = self.calc_lengths()
        self.inverse_matrix = self.calc_normalization_matrix()
        self.transform_matrix = self.calc_inverse_normalization_matrix()

    def calc_outer_points_first_order(self) -> ndarray:
        min_ = [ argmin(self.geometry[:, dimension]) for dimension in range(3) ]
        max_ = [ argmax(self.geometry[:, dimension]) for dimension in range(3) ]

        min_vertices = array(tuple(self.geometry[vertex] for vertex in min_))
        max_vertices = array(tuple(self.geometry[vertex] for vertex in max_))
        widths = norm(max_vertices - min_vertices, axis=0)
        widest_dimension = argmax(widths)
        return array([
            self.geometry[min_[widest_dimension]],
            self.geometry[max_[widest_dimension]],
        ])

    def calc_outer_points_second_order(self) -> ndarray:
        orthogonal_axis = cross(self.primary_normal, array([0, 0, 1]))
        orthogonal_axis /= norm(orthogonal_axis)

        rotations = [Rotation.from_rotvec(pi/4.0 * n * self.primary_normal) for n in range(4)]
        axis_candidates = [
            rotation.apply(orthogonal_axis) for rotation in rotations
        ]

        projections = array(tuple(self.geometry.dot(axis) for axis in axis_candidates))
        min_ = argmin(projections, axis=1)
        max_ = argmax(projections, axis=1)
        min_vertices = array(tuple(self.geometry[vertex] for vertex in min_))
        max_vertices = array(tuple(self.geometry[vertex] for vertex in max_))
        widths = norm(max_vertices - min_vertices, axis=1)
        widest_axis = argmax(widths)
        return array([
            self.geometry[min_[widest_axis]],
            self.geometry[max_[widest_axis]],
        ])

    def calc_lengths(self) -> ndarray:
        projections = array([self.geometry.dot(axis) for axis in (
            self.primary_normal,
            self.secondary_normal,
            self.ternary_normal,
        )])
        min_ = projections.min(axis=1)
        max_ = projections.max(axis=1)
        return max_ - min_ 

    def calc_normalization_matrix(self) -> ndarray:
        translation = self.center_of_mass
        scale = 2.0 / self.length

        # axes
        primary = scale[0] * self.primary_normal
        secondary = scale[1] * self.secondary_normal
        ternary = scale[2] * self.ternary_normal
        
        return array([
            [
                primary[0],
                secondary[0],
                ternary[0],
                translation[0],
            ], [
                primary[1],
                secondary[1],
                ternary[1],
                translation[1],
            ], [
                primary[2],
                secondary[2],
                ternary[2],
                translation[2],
            ], [ 
                0, 0, 0, 1
            ],
        ])

    def calc_inverse_normalization_matrix(self) -> ndarray:
        rotation = self.inverse_matrix[:3, :3]
        translation = self.inverse_matrix[:3, 3]

        inverse_rotation = rotation.T

        inverse_translation = -inverse_rotation.dot(translation)

        inverse_transformation = identity(4)
        inverse_transformation[:3, :3] = inverse_rotation
        inverse_transformation[:3, 3] = inverse_translation

        return inverse_transformation


if __name__ == "__main__":
    from main import load_stl
    from features import Slicer

    vertebra = load_stl( '../data/L5.stl')
    from timeit import default_timer
    start = default_timer()
    com = Slicer.calculate_center_of_mass(vertebra)
    bb = BoundingBox(vertebra, center_of_mass=com)
    end = default_timer()
    print(end - start)
    print(bb.inverse_matrix)
