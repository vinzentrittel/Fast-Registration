__version__ = "0.1"

from timeit import default_timer
from typing import Any

from numpy import array, identity, matmul, ndarray, ones
from numpy.linalg import inv
from slic3r_display import Slic3rPointRepresentable
from vtk import (
    vtkMatrix4x4,
    vtkPointSet,
    vtkPolyData,
    vtkTransform,
    vtkTransformFilter,
    vtkTransformPolyDataFilter,
)
from vtk.util.numpy_support import vtk_to_numpy

from .dissectionable import Dissectionable
from .landmarking import register_elastically
from .rotation_factory import RotationFactory
from .util import load_markers, load_stl, numpy_to_points, PointMode, remesh, write
    
def register(
    source: vtkPolyData,
    target: vtkPolyData,
    landmarks: ndarray,
    landmark_normals: ndarray,
    source_pois: ndarray,
) -> Any:
    # calculate identity projection
    dissectionable_source = Dissectionable(source)
    start = default_timer()
    dissectionable_target = Dissectionable(target)

    rotation_index = dissectionable_target.min_err_rotation(dissectionable_source)
    rotation_matrix = identity(4)
    rotation_matrix[:3, :3] = RotationFactory.Rotations[rotation_index]

    transformation_matrix = (
        dissectionable_source.bounding_box.inverse_transform_matrix
    ).dot(
        rotation_matrix
    ).dot(
        dissectionable_target.bounding_box.transform_matrix
    )
    transform = vtkTransform()
    transform.SetMatrix(transformation_matrix.flatten())
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(target)
    transform_filter.Update()

    target_landmarks = register_elastically(
        transform_filter.GetOutput(), landmarks, landmark_normals, source_pois
    )
    point_set = vtkPointSet()
    point_set.SetPoints(numpy_to_points(target_landmarks))
    #transform.Inverse()
    inverse_transform_matrix = (
        dissectionable_target.bounding_box.inverse_transform_matrix
    ).dot(inv(rotation_matrix)).dot(
        dissectionable_source.bounding_box.transform_matrix
    )
    print(default_timer() - start)
    #transform.SetMatrix(dissectionable_source.bounding_box.inverse_transform_matrix.dot(inv(rotation_matrix)).dot(dissectionable_target.bounding_box.transform_matrix).flatten())
    transform.SetMatrix(inverse_transform_matrix.flatten())
    transform_filter = vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(point_set)
    transform_filter.Update()
    #Slic3rPointRepresentable.write_from([transform_filter.GetOutput().GetPoint(n) for n in range(transform_filter.GetOutput().GetNumberOfPoints())], "output.mrk.json")

def dubious(geometry: vtkPolyData) -> bool:
    original_bounds = geometry.GetBounds()
    identity_transform = vtkTransform()
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(identity_transform)
    transform_filter.SetInputData(geometry)
    transform_filter.Update()
    return original_bounds != transform_filter.GetOutput().GetBounds()

if __name__ == "__main__":
    from sys import argv
    from .util import to_lps
    (
        landmarks,
        landmark_normals,
        landmark_curvatures,
        landmark_weighted_curvatures,
    ) = load_markers(f"{argv[2]}.csv", PointMode.SCALE_HANDLE)
    source_pois, *_ = load_markers(f"{argv[2]}.csv", PointMode.POI)
    register(
        source=load_stl(argv[2]),
        target=load_stl(argv[1]),
        landmarks=array(landmarks),
        landmark_normals=array(landmark_normals),
        source_pois=array(source_pois),
    )
