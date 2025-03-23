"""
The final landmark registration happens here.
"""
from math import sqrt
from numpy import (
    arange,
    argmax,
    argsort,
    array,
    ndarray,
    newaxis,
)
from numpy.linalg import norm
from slic3r_display import Slic3rPointRepresentable
from vtk import ( # pylint: disable=no-name-in-module
    mutable,
    reference,
    vtkGenericCell,
    vtkPointSet,
    vtkPolyData,
    vtkPolyDataNormals,
    vtkThinPlateSplineTransform,
    vtkTransformFilter,
    vtkTransformPolyDataFilter,
)
from vtk.util.numpy_support import vtk_to_numpy # pylint: disable=import-error,no-name-in-module

from .util import (
    load_markers,
    load_stl,
    remesh,
    n_greatest_values,
    calculate_curvature,
    smooth_normals,
    numpy_to_points,
    PointMode,
    write,
)

def register_elastically(
    source: vtkPolyData, target_points: ndarray, target_normals: ndarray, landmarks: ndarray
) -> ndarray:
    approximation = thin_plate_spline_transform(source, target_points, target_normals, landmarks)
    return move_to_surface(approximation, source)

def thin_plate_spline_transform(
    source: vtkPolyData, target_points: ndarray, target_normals: ndarray, landmarks: ndarray
) -> ndarray:
    """
    Perform a thin plate spline transformation to a set of landmark points.

    The transformation is based on 'target_points' and there 'target_normals' as well as a
    'source' geometry. Each point and normal leads the transformation to a locally similar
    point on the 'source'. 'landmarks' are then transformed and returned.

    Keyword arguments:
    source - source geometry, that guides the transformation and where the landmarks are coming
             from.
    target_points - numpy ndarray of shape=(n,3) containing 3d coordinates on a target geometry.
    target_normals - numpy ndarray of shape=(n,3) containing 3d vectors describing the direction
                     for each one of the 'target_points'.
    landmarks - numpy ndarray of shape=(m,3) containing 3d coordinates of points of interest on
                the 'source' geometry. These points will be transformed and returned. The original
                ndarray remains unaltered.
    """
    source_points = numpy_to_points(point_correspondences(
        source, target_points=target_points, target_normals=target_normals
    ))
    target_points = numpy_to_points(target_points)
    target_landmarks = vtkPointSet()
    target_landmarks.SetPoints(numpy_to_points(landmarks))

    transform = vtkThinPlateSplineTransform()
    transform.SetBasisToR() # 3D data
    transform.SetSourceLandmarks(target_points)
    transform.SetTargetLandmarks(source_points)

    filter_ = vtkTransformFilter()
    filter_.SetTransform(transform)
    filter_.SetInputData(target_landmarks)
    filter_.Update()
    return vtk_to_numpy(filter_.GetOutput().GetPoints().GetData())

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
    source = remesh(source, cluster_count=cluster_count, decimate=False)
    curvatures = calculate_curvature(source)
    # TODO: investigate if filtering by curvature does even provide a performance boost at this point (n = size/2)
    great_curvature_mask = vtk_to_numpy(
        n_greatest_values(curvatures, int(source.GetNumberOfPoints() / 2))
    )
    sharp_point_indices = array(
        [index for index, flag in enumerate(great_curvature_mask) if flag == 1.0]
    )

    points = vtk_to_numpy(source.GetPoints().GetData())
    candidate_distances = norm(
        points[sharp_point_indices] - target_points[:, newaxis, :],
        axis=2,
    )
    candidate_indices = sharp_point_indices[
        argsort(candidate_distances, axis=1)[:, :int(len(points) / 100)]
    ]

    normals = vtkPolyDataNormals()
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.SetInputData(source)
    normals = vtk_to_numpy(smooth_normals(normals.GetOutputPort()).GetPointData().GetNormals())
    projections = target_normals.dot(normals.T)
    candidate_projections = projections[arange(len(target_points))[:, newaxis], candidate_indices]
    best_candidate_indices = candidate_indices[
        arange(len(target_points)), argmax(candidate_projections, axis=1)
    ]
    return points[best_candidate_indices]

def move_to_surface(landmarks: ndarray, target: vtkPolyData) -> ndarray:
    """
    Return corrected position for landmark points, that were previously close to 'target' surface.
    The corrected position will reside right on the surface now.

    This is achieved by finding the closest face of the 'target' surface.
    Then the intersection of a ray from initial 'landmark' and face normal with the closes mesh
    gives the corrected position.

    This is done for every 'landmark' point.

    Keyword arguments:
    landmarks - Numpy array of shape=(n,3) representing an arbitrary number of 3D points, that
                should be trivially corrected to reside on the surface.
    target - VTK surface mesh, where the landmarks should ideally reside on.
    """
    target_normals = vtkPolyDataNormals()
    target_normals.ComputeCellNormalsOn()
    target_normals.ComputePointNormalsOff()
    target_normals.SplittingOff()
    target_normals.SetInputData(target)
    target_normals.Update()
    target = target_normals.GetOutput()
    cell_normals = vtk_to_numpy(target.GetCellData().GetNormals())

    target.BuildLocator()
    target.BuildCellLocator()
    target_locator = target.GetCellLocator()

    surface_points = []
    estimate_surface_point = [0.0, 0.0, 0.0]
    surface_cell_id = mutable(0)
    distance2 = mutable(0.0)
    for landmark in landmarks:
        target_locator.FindClosestPoint(
            landmark,
            estimate_surface_point,
            vtkGenericCell(),
            surface_cell_id,
            mutable(0),
            distance2,
        )

        direction = cell_normals[surface_cell_id]
        distance = sqrt(distance2)
        p1 = landmark - distance * direction
        p2 = landmark + distance * direction

        x = [0.0, 0.0, 0.0]
        target_locator.IntersectWithLine(p1, p2, 0.0001, mutable(0), x, [0,0,0], mutable(0))
        surface_points.append(x)

    return array(surface_points)

if __name__ == "__main__":
    SOURCE = load_stl("data/thin_plate_spline/L4_manual_adjust.stl")
    TARGET_POINTS, TARGET_NORMALS = load_markers(
        "data/thin_plate_spline/L5.stl.csv",
        mode=PointMode.SCALE_HANDLE,
    )

    registered_points = thin_plate_spline_transform(
        SOURCE, array(TARGET_POINTS), array(TARGET_NORMALS), landmarks=array([[-0.53,31.29,1.29]])
    )
    corrected_points = move_to_surface(registered_points, target=SOURCE)
    print(corrected_points)
