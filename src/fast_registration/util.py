from pathlib import Path
from typing import Any

from numpy import multiply, ndarray, newaxis
from slic3r_display import Slic3rBoxRepresentable, Slic3rPointRepresentable
from vtk import (
    vtkIterativeClosestPointTransform,
    vtkPolyData,
    vtkSTLReader,
    vtkSTLWriter,
    vtkTransform,
    vtkTransformPolyDataFilter,
)

def load_stl(filename: str) -> vtkPolyData:
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def write(obj: Any, filename: Path) -> None:
    from .bounding_box import BoundingBox

    if isinstance(obj, BoundingBox):
        Slic3rBoxRepresentable.write_from(
            origin=Slic3rBoxRepresentable.center_to_origin(obj.center, obj.axes, obj.length),
            axes=multiply(obj.axes, obj.length[:, newaxis]),
            filename=filename,
        )
    elif isinstance(obj, vtkPolyData):
        writer = vtkSTLWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(obj)
        writer.Update()
    elif isinstance(obj, ndarray) and obj.ndim == 2 and obj.shape[1] == 3:
        Slic3rPointRepresentable.write_from(obj, filename)
    else:
        raise NotImplementedError(type(obj), "has no appropriate write function")

def transform(geometry: vtkPolyData, matrix: ndarray) -> vtkPolyData:
    """
    Return the transformed 'geometry' as dictated by the transformation 'matrix'.

    Keyword arguments:
    geometry - original polygon mesh as a vtkPolyData instance.
    matrix - 3x3 transformation matrix as a numpy ndarray instance.
    """
    transformation = vtkTransform()
    transformation.SetMatrix(matrix.flatten().tolist())

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputData(geometry)
    transform_filter.SetTransform(transformation)
    transform_filter.Update()

    return transform_filter.GetOutput()

def calculate_icp_transformation(
    source_geometry: vtkPolyData, target_geometry: vtkPolyData
) -> vtkIterativeClosestPointTransform:
    """
    Calculate a transformation matrix, to rigidly register a 'source_geometry'
    onto the position of 'target_geometry'. Transformation algorithm is the
    iterative closest point search.

    Keyword arguments:
    source_geometry - a vtkPolyData mesh, that should be moved to align with
                      'target_geometry'.
    target_geometry - the destination of a 'source_geometry'
    """
    iterative_closest_point = vtkIterativeClosestPointTransform()
    iterative_closest_point.SetSource(source_geometry)
    iterative_closest_point.SetTarget(target_geometry)
    iterative_closest_point.GetLandmarkTransform().SetModeToRigidBody()
    iterative_closest_point.SetMaximumNumberOfLandmarks(200)
    iterative_closest_point.SetMaximumMeanDistance(0.00001)
    iterative_closest_point.SetMaximumNumberOfIterations(25)
    #iterative_closest_point.CheckMeanDistanceOn()
    #iterative_closest_point.StartByMatchingCentroidsOn()
    iterative_closest_point.Update()

    return iterative_closest_point
