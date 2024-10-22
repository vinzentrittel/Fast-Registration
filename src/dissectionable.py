from __future__ import annotations

from typing import List

from numpy import absolute, array, ndarray, sum as sum_, tile
from numpy.linalg import inv, norm
from vtk import vtkPolyData, vtkTransform, vtkTransformPolyDataFilter
from vtk.util.numpy_support import vtk_to_numpy

from bounding_box import BoundingBox
from identity_clipper import IdentityClipper
from ray_factory import DefaultRays, DefaultPoints
from rotation_factory import RotationFactory
from axis import AxisValue, Converter as AxisConverter

class Dissectionable:
    def __init__(self, geometry: vtkPolyData) -> None:
        self.bounding_box = BoundingBox(geometry)
        normalized_geometry = self.transform(
            geometry, matrix=self.bounding_box.transform_matrix
        )
        self.clipper = IdentityClipper()
        self.clipper.input_data = normalized_geometry
        self.sections = DefaultPoints.copy()

    def min_err_rotation(self, other: Dissectionable) -> ndarray:
        self_projection = self.calc_section_projections()
        other_projection = other.calc_section_projections()

        self_permutations = RotationFactory.correspondence_from_rotations(self_projection)
        other_permutations = tile(other_projection, [len(self_permutations), 1, 1])
        absolute_errors = sum_(norm(absolute(self_permutations - other_permutations), axis=2), axis=1)
        return absolute_errors.argmin()
        return RotationFactory.Rotations[absolute_errors.argmin()]


    def calc_section_projections(self) -> ndarray:
        projections = list()
        for projection, direction, clip in zip(
            self.sections, DefaultRays, self.clipper # TODO: need generator for IdentityClipper
        ):
            vertices = clip.GetPoints()
            if not vertices is None:
                vertices = vtk_to_numpy(clip.GetPoints().GetData())
                length = vertices.dot(projection).max()
                new_point = 2.0 * length * direction
                projections.append(new_point)
            else:
                projections.append(array([0, 0, 0]))
        return array(projections)

    @staticmethod
    def make_sections() -> List[List[List[Section]]]:
        sections = [[3 * [None] for _ in range(3)] for _ in range(3)]
        for frontal in (AxisValue.Left, AxisValue.Center, AxisValue.Right,):
            for sagittal in (AxisValue.Posterior, AxisValue.Center, AxisValue.Anterior,):
                for longitudinal in (AxisValue.Inferior, AxisValue.Center, AxisValue.Superior,):
                    sections[frontal][sagittal][longitudinal] = Section(
                        frontal, sagittal, longitudinal
                    )
        return sections

    @staticmethod
    def transform(geometry: vtkPolyData, matrix: ndarray) -> vtkPolyData:
        transformation = vtkTransform()
        transformation.SetMatrix(matrix.flatten().tolist())

        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetInputData(geometry)
        transform_filter.SetTransform(transformation)
        transform_filter.Update()

        return transform_filter.GetOutput()


class Section:
    def __init__(
        self,
        frontal: AxisValue,
        sagittal: AxisValue,
        longitudinal: AxisValue
    ) -> None:
        self.default_point = DefaultPoints[
            AxisConverter.to_index((frontal, sagittal, longitudinal,))
        ]
        self.direction = DefaultRays[
            AxisConverter.to_index((frontal, sagittal, longitudinal,))
        ]
        self.point = self.default_point.copy()

if __name__ == '__main__':
    from util import load_stl
    vertebra = load_stl("data/L5_shifted.stl")
    rotated_vertebra = load_stl("data/L5_rotated.stl")

    from timeit import default_timer
    start = default_timer()
    dissection = Dissectionable(vertebra)
    rotated_dissection = Dissectionable(rotated_vertebra)

    dissection.calc_section_projections()
    print(dissection.min_err_rotation(rotated_dissection))
    end = default_timer()
    print(end - start)

    write(dissection.clipper.input_data, "data/L5.debug.stl")
    write(rotated_dissection.clipper.input_data, "data/L5_rotated.debug.stl")

