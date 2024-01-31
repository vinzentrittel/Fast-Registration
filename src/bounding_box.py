from __future__ import annotations

from typing import Tuple

from numpy import array, argmax, argmin, cross, inf, ndarray, pi
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

        self.ternary_normal = cross(self.primary_normal, self.secondary_normal)
        self.ternary_normal /= norm(self.ternary_normal)

        self.secondary_normal = cross(self.ternary_normal, self.primary_normal)
        self.secondary_normal /= norm(self.secondary_normal)

        self.length = self.calc_lengths()

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
        
    def __len__(self) -> Tuple[float, float, float]:
        return self.length

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
