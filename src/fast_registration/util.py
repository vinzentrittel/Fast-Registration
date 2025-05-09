"""
Collection of useful functions for this project.

load_stl(filename: str) - load geometry from drive.

load_points(
    filename: Path, mode: PointMode, points: vtkPoints=None, normals: vtkDataArray=None
) - load landmark points from csv file.

write(obj: Any, filename: Path) - write various objects to drive.

transform(geometry: vtkPolyData, matrix: ndarray) - rigidly modify a VTK geometry by a 4x4 matrix.

calculate_curvature(source: vtkPolyData)
    - calculate degree of curvature for all vertices of a VTK geometry.

n_greatest_values(array: vtkDoubleArray, n: int) - mark all n greatest values of a VTK array.

remesh(geometry: vtkPolyData, cluster_count: int)
    - remesh a VTK geometry to a pre-stated number of vertices.

point_correspondences(
    source: vtkPolyData, target_points: ndarray, target_normals: ndarray, cluster_count: int=2000
)
    - find points of source geometry, that share most similar properties with points of target
      geometry.
"""
from csv import DictReader
from enum import auto, Enum
from os.path import isfile
from pathlib import Path
from typing import Any, List, Tuple, Union

from numpy import (
    arange,
    argmax,
    argpartition,
    argsort,
    array,
    multiply,
    ndarray,
    newaxis,
    zeros,
)
from numpy.linalg import norm
from pyacvd import Clustering
from pyvista import wrap
from slic3r_display import Slic3rBoxRepresentable, Slic3rPointRepresentable
from vtk import ( # pylint: disable=no-name-in-module
    vtkAlgorithmOutput,
    vtkCellDataToPointData,
    vtkCurvatures,
    vtkDataArray,
    vtkDecimatePro,
    vtkDoubleArray,
    vtkIdList,
    vtkIdTypeArray,
    vtkPointDataToCellData,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataNormals,
    vtkSTLReader,
    vtkSTLWriter,
    vtkTransform,
    vtkTransformPolyDataFilter,
)
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy # pylint: disable=import-error,no-name-in-module

def to_lps(geometry: vtkPolyData) -> vtkPolyData:
    """
    Some geometries are in a different coordinate system.
    This function fixes the issue.
    """
    transform_ = vtkTransform()
    transform_.Scale(-1, -1, 1)
    transform_.RotateX(-90)

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform_)
    transform_filter.SetInputData(geometry)
    transform_filter.Update()
    return transform_filter.GetOutput()

def load_stl(filename: str) -> vtkPolyData:
    """
    Load a VTK surface geometry from an STL file.

    Keyword arguments:
    filename - path to where the STL file is stored at.
    """
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

POINTS_HEADER = "x", "y", "z", "nx", "ny", "nz", "kind"
CURVATURE_TYPE =  "Mean_Curvature" # or "Gauss_Curvature"

class PointMode(Enum):
    """
    Mode of landmarks.
    """
    POI = 0
    SCALE_HANDLE = auto()

def load_markers(
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
    return tuple(zip(*result)) if result else ([], [],)


def write(obj: Any, filename: Path) -> None:
    """
    Utility to save various objects to disc.

    Keyword arguments:
    obj - instance to save (can be
              bounding_box.BoundingBox,
              vtkPolyData,
              numpy.ndarray with shape==(n, 3,),
          )
    filename - path to store the file.
    """
    from .bounding_box import BoundingBox # pylint: disable=import-outside-toplevel
    print(type(obj))

    if isinstance(obj, BoundingBox):
        Slic3rBoxRepresentable.write_from(
            origin=Slic3rBoxRepresentable.center_to_origin(obj.center, obj.axes, obj.length),
            axes=multiply(obj.axes, obj.length[:, newaxis]),
            filename=filename,
        )
    elif isinstance(obj, vtkPolyData):
        writer = vtkSTLWriter()
        writer.SetFileName(str(filename))
        writer.SetFileTypeToBinary()
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

def calculate_curvature(source: vtkPolyData) -> vtkDoubleArray:
    """
    Return VTK array containing the degree of curvature for each source's vertex in order.

    Keyword arguments:
    source - VTK surface geometry.
    """
    curvatures = vtkCurvatures()
    if "Gauss" in CURVATURE_TYPE:
        curvatures.SetCurvatureTypeToGaussian()
    else:
        curvatures.SetCurvatureTypeToMean()
    curvatures.SetInputData(source)

    averager = vtkPointDataToCellData()
    averager.SetInputConnection(curvatures.GetOutputPort())
    back_averager = vtkCellDataToPointData()
    back_averager.SetInputConnection(averager.GetOutputPort())
    back_averager.Update()
    #return curvatures.GetOutput().GetPointData().GetAbstractArray(CURVATURE_TYPE)
    return back_averager.GetOutput().GetPointData().GetAbstractArray(CURVATURE_TYPE)

def generate_neighbor_list(geometry: vtkPolyData) -> List[Tuple[int, int]]:
    neighbors = [[] for _ in range(geometry.GetNumberOfPoints())]
    geometry.BuildPointLocator()
    locator = geometry.GetPointLocator()
    for point_id in range(geometry.GetNumberOfPoints()):
        neighbor_ids = vtkIdList()
        locator.FindClosestNPoints(25, geometry.GetPoint(point_id), neighbor_ids)
        for n in range(neighbor_ids.GetNumberOfIds()):
            neighbor_id = neighbor_ids.GetId(n)
            neighbors[point_id].append(neighbor_id)
            neighbors[neighbor_id].append(point_id)

    neighbors[:] = [list(set(n)) for n in neighbors]
    return [
        (first, second,)
        for first, neighbors in enumerate(neighbors)
        for second in neighbors
    ]

def n_greatest_values(array_: vtkDoubleArray, n: int) -> vtkDoubleArray:
    """
    For a given VTK array, marke each of the n greatest values with 1.0.
    Else 0.0.
    """
    if not isinstance(array_, ndarray):
        array_ = vtk_to_numpy(array_)
    filter_mask = zeros(len(array_))
    filter_mask[argpartition(-array_, n)[:n]] = 1.0
    return numpy_to_vtk(filter_mask)

def remesh(geometry: vtkPolyData, cluster_count: int, decimate: bool=False) -> vtkPolyData:
    """
    Given an updated number of new vertices, this method returns mesh, so that is consists
    of triangles only of roughly equal area.

    Keyword arguments:
    geometry - VTK surface geometry.
    cluster_count - maximum number of vertices, the resulting geometry will have.
    decimate - flag to trigger decimation to minimal geometry that preserves topology.
    """
    clustering = Clustering(wrap(geometry))
    clustering.cluster(cluster_count)
    if not decimate:
        return clustering.create_mesh()

    decimation = vtkDecimatePro()
    decimation.SetInputData(clustering.create_mesh())
    decimation.PreserveTopologyOn()
    decimation.Update()
    return decimation.GetOutput()

def smooth_normals(input_: Union[vtkPolyData, vtkAlgorithmOutput]) -> vtkPolyData:
    """
    Interpolate the "Normals" VTK data array by their surrounding cells.
    To work properly the normals must have previously been calculated and be available
    for the "input_"'s PointData.GetNormals().

    Return a whole new vtkPolyData object. The input is left untouched.

    Keyword Arguments:
    input_ - vtkPolyData object or an algorithms output (see GetOutputPort()).
    """
    result = _average_point_data(input_, data_array_name="Normals")
    normals = vtk_to_numpy(result.GetPointData().GetNormals())
    normals /= norm(normals, axis=1)[:, newaxis]
    #normals = numpy_to_vtk(normals)
    result.GetPointData().SetNormals(numpy_to_vtk(normals))

    return result

def average_curvature(input_: vtkPolyData) -> vtkPolyData:
    return _average_point_data(input_, data_array_name=CURVATURE_TYPE)

def _average_point_data(
    input_: Union[vtkPolyData, vtkAlgorithmOutput], data_array_name: str
) -> vtkPolyData:
    """
    Interpolate the VTK data array values of the array 'data_array_name' by their surrounding cells.
    To work properly the data array must have previously been calculated and be available
    for the "input_"'s PointData instance.

    Return a whole new vtkPolyData object. The input is left untouched.

    Keyword Arguments:
    input_ - vtkPolyData object or an algorithms output (see GetOutputPort()).
    """
    assert isinstance(input_, (vtkPolyData, vtkAlgorithmOutput,))
    averager = vtkPointDataToCellData()
    averager.ProcessAllArraysOff()
    averager.PassPointDataOn()
    if isinstance(input_, vtkPolyData):
        averager.SetInputData(input_)
    else:
        averager.SetInputConnection(input_)
    averager.AddPointDataArray(data_array_name)

    averager2 = vtkCellDataToPointData()
    averager2.ProcessAllArraysOff()
    averager2.PassCellDataOn()
    averager2.SetInputConnection(averager.GetOutputPort())
    averager2.AddCellDataArray(data_array_name)
    averager2.Update()

    return averager2.GetOutput()

def id_list_to_array(id_list: vtkIdList) -> vtkIdTypeArray:
    array_ = vtkIdTypeArray()
    array_.SetNumberOfComponents(1)
    array_.SetNumberOfTuples(id_list.GetNumberOfIds())
    for index in range(id_list.GetNumberOfIds()):
        array_.InsertNextValue(id_list.GetId(index))
    return array_

def numpy_to_points(points: ndarray) -> vtkPoints:
    result = vtkPoints()
    for coordinate in points:
        result.InsertNextPoint(coordinate)
    return result

def points_to_numpy(points: vtkPoints) -> ndarray:
    pass
