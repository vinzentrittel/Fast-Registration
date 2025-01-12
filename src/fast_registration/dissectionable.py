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
        self.sections = DefaultPoints.copy()

    @property
    def normalized_geometry(self) -> vtkPolyData:
        return self.clipper.input_data

    def min_err_rotation(self, other: Dissectionable) -> ndarray:
        """
        Return that rotation transformation, that yields the smallest absolute euklidean
        distance between this object and the 'other' object.

        Keyword arguments:
        other - an object to compare this instace of Dissectionable against.
        """
        self_projection = self.calc_section_projections()
        other_projection = other.calc_section_projections()

        self_permutations = RotationFactory.correspondence_from_rotations(self_projection)
        other_permutations = tile(other_projection, [len(self_permutations), 1, 1])
        absolute_errors = sum_(
            norm(absolute(self_permutations - other_permutations), axis=2),
            axis=1,
        )
        return RotationFactory.Rotations[absolute_errors.argmin()]

    def calc_section_projections(self) -> ndarray:
        """
        This is up for debate.

        For each of the 27 sub-cubes, calculate one point, that represents that subsection in
        a meaningful way. Different approaches shall be compared to get an empirically sound
        projection method.
        """
        projections = []
        for projection, direction, clip in zip(
            # TODO: need generator for IdentityClipper
            self.sections, DefaultRays, self.clipper
        ):
            vertices = clip.GetPoints()
            if vertices is not None and vertices.GetNumberOfPoints() > 0:
                vertices = vtk_to_numpy(clip.GetPoints().GetData())
                length = vertices.dot(projection).max()
                new_point = 2.0 * length * direction
                projections.append(new_point)
            else:
                projections.append(array([0, 0, 0]))
        back_projections = ones((len(projections), 4,))
        back_projections[:, :3] = array(projections)
        return array(projections)

    @staticmethod
    def make_sections() -> List[List[List[Section]]]:
        """
        Returns 27 sections of this instance. Sections are created by slicing this
        instance's geometry two times in the three body planes.
        """
        sections = [[3 * [None] for _ in range(3)] for _ in range(3)]
        for frontal in (AxisValue.Left, AxisValue.Center, AxisValue.Right,):
            for sagittal in (AxisValue.Posterior, AxisValue.Center, AxisValue.Anterior,):
                for longitudinal in (AxisValue.Inferior, AxisValue.Center, AxisValue.Superior,):
                    sections[frontal][sagittal][longitudinal] = Section(
                        frontal, sagittal, longitudinal
                    )
        return sections

@dataclass
class Section:
    """
    Data container for attributes of each sub-cube for an Dissectionable object.
    """
    default_point: ndarray
    direction: ndarray
    point: ndarray

    @staticmethod
    def make_section(
        frontal: AxisValue, sagittal: AxisValue, longitudinal: AxisValue
    ) -> Section:
        """
        Create a dummy section, where the projection point is set to a
        sensible default value and the normal (direction)
        points in a sensible direction for that section.
        
        What is sensible is determined by 'frontal', 'sagittal' and
        'longitudinal' parameters.

        Keyword arguments:
        frontal - enum value to indicate placement in the frontal plane.
        sagittal - enum value to indicate placement in the sagittal plane.
        longitudinal - enum value to indicate placement in the longitudinal plane.
        """
        default_point = DefaultPoints[
            AxisConverter.to_index((frontal, sagittal, longitudinal,))
        ].copy()
        direction=DefaultRays[
            AxisConverter.to_index((frontal, sagittal, longitudinal,))
        ].copy()
        return Section(
            default_point=default_point,
            direction=direction,
            point=default_point.copy(),
        )

if __name__ == '__main__':
    from util import load_stl
    vertebra = load_stl("data/L5.stl")

    from timeit import default_timer
    start = default_timer()
    dissection = Dissectionable(vertebra)
    other = Dissectionable(load_stl("data/L4.stl"))
    print(dissection.min_err_rotation(other))

    end = default_timer()
    print(end - start)
