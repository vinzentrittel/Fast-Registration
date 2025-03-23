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
    scale_handle_points: ndarray,
    scale_handle_normals: ndarray,
    landmarks: ndarray,
) -> Any:
    # calculate identity projection
    dissectionable_target = Dissectionable(target)
    dissectionable_source = Dissectionable(source)

    rotation_index = dissectionable_source.min_err_rotation(dissectionable_target)
    rotation_matrix = identity(4)
    rotation_matrix[:3, :3] = RotationFactory.Rotations[rotation_index]

    transformation_matrix = (
        dissectionable_target.bounding_box.inverse_transform_matrix
    ).dot(
        rotation_matrix
    ).dot(
        dissectionable_source.bounding_box.transform_matrix
    )
    transform = vtkTransform()
    transform.SetMatrix(transformation_matrix.flatten())
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(source)
    transform_filter.Update()

    source_landmarks = register_elastically(
        transform_filter.GetOutput(), scale_handle_points, scale_handle_normals, landmarks
    )
    point_set = vtkPointSet()
    point_set.SetPoints(numpy_to_points(source_landmarks))
    #transform.Inverse()
    transform.SetMatrix(dissectionable_source.bounding_box.inverse_transform_matrix.dot(inv(rotation_matrix)).dot(dissectionable_target.bounding_box.transform_matrix).flatten())
    transform_filter = vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(point_set)
    transform_filter.Update()

    back_transform_landmarks = vtk_to_numpy(transform_filter.GetOutput().GetPoints().GetData())
    transform.SetMatrix(dissectionable_source.bounding_box.inverse_transform_matrix.flatten())
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(dissectionable_source.normalized_geometry)
    transform_filter.Update()
    write(transform_filter.GetOutput(), "output.stl")
    Slic3rPointRepresentable.write_from(back_transform_landmarks.tolist(), "output.mrk.json")

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
    scale_handle_points, scale_handle_normals = load_markers(f"{argv[2]}.csv", PointMode.SCALE_HANDLE)
    target_landmarks, _ = load_markers(f"{argv[2]}.csv", PointMode.POI)
    write(load_stl(argv[1]), argv[1])
    register(
        source=load_stl(argv[1]),
        target=load_stl(argv[2]),
        scale_handle_points=array(scale_handle_points),
        scale_handle_normals=array(scale_handle_normals),
        landmarks=array(target_landmarks),
    )
