from csv import DictReader
from enum import auto, Enum
from os.path import isfile
from pathlib import Path
from typing import Any, List, Tuple

from numpy import multiply, ndarray, newaxis, sum as sum_
from pyacvd import Clustering
from pyvista import wrap
from slic3r_display import Slic3rBoxRepresentable, Slic3rPointRepresentable
from vtk import (
    vtkCurvatures,
    vtkDataArray,
    vtkIterativeClosestPointTransform,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataNormals,
    vtkSTLReader,
    vtkSTLWriter,
    vtkTransform,
    vtkTransformPolyDataFilter,
)
from vtk.util.numpy_support import vtk_to_numpy

def load_stl(filename: str) -> vtkPolyData:
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

POINTS_HEADER = "x", "y", "z", "nx", "ny", "nz", "kind"

class PointMode(Enum):
    """
    Mode of landmarks.
    """
    POI = 0
    SCALE_HANDLE = auto()

def load_points(
    filename: Path, mode: PointMode, points: vtkPoints=None, normals: vtkDataArray=None
) -> List[Tuple[float, float, float]]:
    """
    Load points (Point of interest [POI], scale handle) and there respective normals from CSV files.
    Returns 3D points as a list of 3-tuples.

    Keyword arguments:
    filename - file path to a CSV file, containing locations as generated
               by fast_registration/marker.py.
    mode - enum value describing the role of a location in loaded file.
    points - [optional] vtkPoint object, where the points shall be appended to.
    normals - [optional] vtkDataArray object, where the normals shall be appended to.
    """
    if (points is None or normals is None) and points != normals:
        raise ValueError("Provide vtkPoints and vtkDataArray to write to parameter objects")
    if not isfile(filename):
        rows = []
    else:
        with open(filename, "r", encoding="utf-8") as csv_point_file:
            csv = DictReader(csv_point_file)
            rows = list(csv)
    result = [
        (
            (float(row_dict["x"]), float(row_dict["y"]), float(row_dict["z"]),),
            (float(row_dict["nx"]), float(row_dict["ny"]), float(row_dict["nz"]),),
        )
        for row_dict in rows
        if row_dict["kind"] == mode.name
    ]
    if not points is None:
        for point, normal in result:
            points.InsertNextPoint(point)
            normals.InsertNextTuple(normal)
    return result


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

def point_correspondences(
    source: vtkPolyData, target_points: ndarray, target_normals: ndarray, cluster_count: int=2000
) -> ndarray:
    """
    Return points on the source mesh, that have a similar location and orientation
    as sparse target_points and target_normals.

    Beware! Only points on "sharp" or "pointy" locations are considered.

    Keyword arguments:
    source - geometry that should be inspected for correspondences.
    target_points - 3D landmarks to identify on the source
    target_normals - orientation normal to be expected of a POI on the source mesh.
    cluster_count - number of clusters for remeshing
    """
    clustering = Clustering(wrap(source))
    clustering.cluster(cluster_count)
    source = clustering.create_mesh()

    curvature_type = 'Gauss_Curvature'
    curvatures = vtkCurvatures()
    curvatures.SetCurvatureTypeToGaussian()
    curvatures.SetInputData(source)
    curvatures.Update()
    curvatures = vtk_to_numpy(
        curvatures.GetOutput().GetPointData().GetAbstractArray(curvature_type)
    )
    sharp_point_indices = (curvatures > 0.05).nonzero()

    normals = vtkPolyDataNormals()
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.SetInputData(source)
    normals.Update()
    normals = vtk_to_numpy(normals.GetOutput().GetPointData().GetNormals())
    normals = normals[sharp_point_indices]
    normal_weights = target_normals.dot(normals.T)

    points = vtk_to_numpy(source.GetPoints().GetData())
    sharp_points = points[sharp_point_indices]
    squared_distances_to_target = sum_(
        (target_points[None, :, :] - sharp_points[:, None, :])**2, axis=2
    ).T
    weights = normal_weights / squared_distances_to_target

    return points[weights.argmax(axis=1)]
