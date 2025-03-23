from __future__ import annotations
from dataclasses import dataclass
from typing import List

from numpy import absolute, array, ndarray, sum as sum_, tile, ones
from numpy.linalg import inv, norm
from slic3r_display import Slic3rPointRepresentable, Slic3rBoxRepresentable
from vtk import vtkPolyData, vtkTransform, vtkTransformPolyDataFilter
from vtk.util.numpy_support import vtk_to_numpy

from .bounding_box import BoundingBox
from .identity_clipper import IdentityClipper
from .ray_factory import DefaultRays, DefaultPoints
from .rotation_factory import RotationFactory
from .axis import AxisValue, Converter as AxisConverter
from .util import write, transform

class Dissectionable:
    """
    Represents 3D object, that can be divides into 27 sub-objects.
    Subdivisions consists of slicing the geometries oriented bounding box into
    equi-volumetric sub-cubes.
    """
    def __init__(self, geometry: vtkPolyData) -> None:
        self.bounding_box = BoundingBox(geometry)
        normalized_geometry = transform(
            geometry, matrix=self.bounding_box.transform_matrix
        )
        self.clipper = IdentityClipper()
        self.clipper.input_data = normalized_geometry

    @property
    def normalized_geometry(self) -> vtkPolyData:
        return self.clipper.input_data

    def min_err_rotation(self, other: Dissectionable) -> int:
        """
        Return that rotation transformation index, that yields the smallest absolute euklidean
        distance between this object and the 'other' object.

        To resolve the index, use members of rotation_factory.py

        Keyword arguments:
        other - an object to compare this instace of Dissectionable against.

        Usage:

        source = Dissectionable(load_stl("source.stl"))
        target = Dissectionable(load_stl("target.stl"))
        rotation_index = source.min_err_rotation(target)
        transformation_matrix = RotationFactory.Rotations[rotation_index]
        index_correspondeces = RotationFactory.RotationIndexCorrespondences[rotation_index]
        """
        self_projection = self.calc_section_projections()
        other_projection = other.calc_section_projections()

        self_permutations = RotationFactory.correspondence_from_rotations(self_projection)
        other_permutations = tile(other_projection, [len(self_permutations), 1, 1])
        absolute_errors = sum_(
            norm(absolute(self_permutations - other_permutations), axis=2),
            axis=1,
        )
        return absolute_errors.argmin()

    def calc_section_projections(self) -> ndarray:
        """
        This is up for debate.

        For each of the 27 sub-cubes, calculate one point, that represents that subsection in
        a meaningful way. Different approaches shall be compared to get an empirically sound
        projection method.
        """
        projections = []
        for projection, direction, clip in zip(
            DefaultPoints, DefaultRays, self.clipper
        ):
            vertices = clip.GetPoints()
            if vertices is not None and vertices.GetNumberOfPoints() > 0:
                new_point = (
                    vtk_to_numpy(vertices.GetData()).sum(axis=0) / vertices.GetNumberOfPoints()
                )
                projections.append(new_point)
            else:
                projections.append(array(projection))
        back_projections = ones((len(projections), 4,))
        back_projections[:, :3] = array(projections)
        return array(projections)

if __name__ == '__main__':
    from .util import load_stl
    vertebra = load_stl("data/L5.stl")

    from timeit import default_timer
    start = default_timer()
    dissection = Dissectionable(vertebra)
    other = Dissectionable(load_stl("data/L4.stl"))
    print(dissection.min_err_rotation(other))

    end = default_timer()
    print(end - start)
