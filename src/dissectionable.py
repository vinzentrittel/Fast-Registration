from __future__ import annotations

from numpy import array, ndarray
from numpy.linalg import inv
from vtk import vtkPolyData, vtkTransform, vtkTransformPolyDataFilter
from vtk.util.numpy_support import vtk_to_numpy

from bounding_box import BoundingBox
from identity_clipper import IdentityClipper
from ray_factory import RayFactory

Axis = RayFactory.Axis

class Dissectionable:
    def __init__(self, geometry: vtkPolyData) -> None:
        self.bounding_box = BoundingBox(geometry)
        normalized_geometry = self.transform(
            geometry, matrix=self.bounding_box.transform_matrix
        )
        from vtk import vtkSTLWriter
        writer = vtkSTLWriter()
        writer.SetFileName('../data/L5.debug.stl')
        writer.SetInputData(normalized_geometry)
        writer.Update()

        self.clipper = IdentityClipper()
        self.clipper.input_data = normalized_geometry
        self.sections = self.make_sections()

    def calc_section_projections(self) -> None:
        poi = list()
        for frontal in (Axis.Left, Axis.Center, Axis.Right,):
            for sagittal in (Axis.Posterior, Axis.Center, Axis.Anterior,):
                for longitudinal in (Axis.Inferior, Axis.Center, Axis.Superior,):
                    section = self.sections[frontal][sagittal][longitudinal]
                    section_geometry = self.clipper(frontal, sagittal, longitudinal)

                    points = section_geometry.GetPoints()
                    if not points is None:
                        vertices = vtk_to_numpy(points.GetData())
                        projection = vertices.dot(section.direction).argmax()
                        new_point = projection * section.direction
                        #alternative: new_point = vertices[projection]
                        new_point = array([new_point[0], new_point[1], new_point[2], 1.0])
                        section.point = inv(self.bounding_box.transform_matrix).dot(new_point)

    @staticmethod
    def make_sections() -> List[List[List[Section]]]:
        sections = [[3 * [None] for _ in range(3)] for _ in range(3)]
        for frontal in (Axis.Left, Axis.Center, Axis.Right,):
            for sagittal in (Axis.Posterior, Axis.Center, Axis.Anterior,):
                for longitudinal in (Axis.Inferior, Axis.Center, Axis.Superior,):
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
        frontal: Axis,
        sagittal: Axis,
        longitudinal: Axis
    ) -> None:
        self.default_point = RayFactory.make_default_point(frontal, sagittal, longitudinal)
        self.direction = RayFactory.make_ray(frontal, sagittal, longitudinal)
        self.point = self.default_point.copy()

if __name__ == '__main__':
    from main import load_stl
    vertebra = load_stl("../data/L5.stl")
    from timeit import default_timer
    start = default_timer()
    dissection = Dissectionable(vertebra)
    dissection.calc_section_projections()
    end = default_timer()
    print(end - start)
