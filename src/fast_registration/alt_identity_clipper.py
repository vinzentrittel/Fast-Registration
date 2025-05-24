from collections import namedtuple
from math import sqrt
from typing import List, Tuple

from numpy import argmax, array, multiply, sum as sum_, unique, zeros
from slic3r_display import Slic3rPointRepresentable
from vtk import ( # pylint: disable=no-name-in-module
    vtkBoundingBox,
    vtkCurvatures,
    vtkDoubleArray,
    vtkExtractSelection,
    vtkIdList,
    vtkIdTypeArray,
    vtkPolyData,
    vtkSelection,
    vtkSelectionNode,
    vtkStaticPointLocator,
    vtkUnstructuredGrid,
    vtkWindowedSincPolyDataFilter,
)
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy # pylint: disable=import-error,no-name-in-module

from .util import (
    average_curvature,
    calculate_curvature,
    CURVATURE_TYPE,
    generate_neighbor_list,
    id_list_to_array,
    load_stl,
    n_greatest_values,
    remesh,
    write,
)

WEIGHTED_CURVATURE_TYPE = "Curvature"

Bounds = namedtuple("Bounds", ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max",))
Point = namedtuple("Point", ("id", "location",))

def _calculate_subcube_points(bounds: Bounds, intervals: int):
    """
    Return tuple of eight boundary 3D points, that each define a subcube for
    a grid from 'min_' to 'max_' and a count of elements per dimension,
    equal to 'intervals'
    """
    assert (
        bounds.x_max > bounds.x_min     \
        and bounds.y_max > bounds.y_min \
        and bounds.z_max > bounds.z_min
    )
    assert intervals > 0

    step = [
        (bounds.x_max - bounds.x_min) / intervals,
        (bounds.y_max - bounds.y_min) / intervals,
        (bounds.z_max - bounds.z_min) / intervals,
    ]
    values = [
        tuple(bounds.x_min + step[0] * v for v in range(intervals + 1)),
        tuple(bounds.y_min + step[1] * v for v in range(intervals + 1)),
        tuple(bounds.z_min + step[2] * v for v in range(intervals + 1)),
    ]
    cube_indices = [
        (x, y, z,)
        for x in range(intervals)
        for y in range(intervals)
        for z in range(intervals)
    ]
    cube_bounds = [
        (values[0][x], values[0][x+1], values[1][y], values[1][y+1], values[2][z], values[2][z+1],)
        for x, y, z in cube_indices
    ]
    return cube_bounds

def voxelize(
    geometry: vtkPolyData, voxel_per_dimension: int, bounds: Bounds
) -> List[vtkUnstructuredGrid]:
    """
    Return an unstructured grid, split into 'voxel_per_dimension'³ elements, in a grid like manner.
    Each grid element contains all points of 'geometry', that are part of a plane residing in that sub-cube -
    fully or partially.

    Keyword arguments:
    geometry - surface mesh, that is to be split.
    voxel_per_dimension - number of elements along each axis.
    limits - minimum and maximum boundary of the grid. Faces that are outside these bounds
             will not be considered.
    """
    locator = vtkStaticPointLocator()
    locator.SetDataSet(geometry)
    locator.BuildLocator()

    selection_node = vtkSelectionNode()
    selection_node.SetFieldType(vtkSelectionNode.POINT)
    selection_node.SetContentType(vtkSelectionNode.INDICES)
    selection = vtkSelection()
    selection.AddNode(selection_node)
    extractor = vtkExtractSelection()
    extractor.SetInputData(0, geometry)
    extractor.SetInputData(1, selection)

    subcube = vtkBoundingBox()
    sub_geometries: List[vtkUnstructuredGrid] = []
    radius = sqrt(
        (0.5 * (bounds.x_max - bounds.x_min) / voxel_per_dimension) ** 2   \
        + (0.5 * (bounds.y_max - bounds.y_min) / voxel_per_dimension) ** 2 \
        + (0.5 * (bounds.z_max - bounds.z_min) / voxel_per_dimension) ** 2 \
    )

    for local_bounds in _calculate_subcube_points(
        bounds, voxel_per_dimension
    ):
        center = (
            0.5 * (local_bounds[0] + local_bounds[1]),
            0.5 * (local_bounds[2] + local_bounds[3]),
            0.5 * (local_bounds[4] + local_bounds[5]),
        )
        point_ids = vtkIdList()
        locator.FindPointsWithinRadius(radius, center, point_ids)
        id_array = vtkIdTypeArray()
        subcube.SetBounds(local_bounds)
        for point_id in [point_ids.GetId(n) for n in range(point_ids.GetNumberOfIds())]:
            if subcube.ContainsPoint(geometry.GetPoint(point_id)):
                id_array.InsertNextValue(point_id)
        selection_node.SetSelectionList(id_array)
        extractor.Update()

        output = vtkUnstructuredGrid()
        output.DeepCopy(extractor.GetOutput())
        sub_geometries.append(output)

    return sub_geometries

def _smooth(geometry: vtkPolyData) -> vtkPolyData:
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(geometry)
    smoother.SetNumberOfIterations(20)
    smoother.SetPassBand(0.005)
    smoother.FeatureEdgeSmoothingOn()
    smoother.BoundarySmoothingOn()
    smoother.Update()
    return smoother.GetOutput()

def _calculate_distance_weighted_curvature(geometry):
    """
    Return list of (1 / norm(x, y)^2)
    """
    curvatures = vtk_to_numpy(geometry.GetPointData().GetScalars(CURVATURE_TYPE))
    neighbor_list = generate_neighbor_list(geometry)
    weights = [0 for _ in range(geometry.GetNumberOfPoints())]
    scales = [0 for _ in range(geometry.GetNumberOfPoints())]
    for first, second in neighbor_list:
        if first == second:
            continue
        first_point = array(geometry.GetPoint(first))
        second_point = array(geometry.GetPoint(second))
        delta = second_point - first_point
        distance2 = sum_(multiply(delta, delta))

        weights[first] += (curvatures[first] - curvatures[second]) / distance2
        scales += 1 / distance2
    return [weight / scale for weight, scale in zip(weights, scales)]

def _calculate_weighted_curvatures(geometry: vtkPolyData) -> vtkDoubleArray:
    smooth_geometry = _smooth(geometry)
    curvatures = vtkCurvatures()
    curvatures.SetInputData(smooth_geometry)
    curvatures.SetCurvatureTypeToMean()
    curvatures.Update()
    geometry.GetPointData().SetScalars(curvatures.GetOutput().GetPointData().GetScalars(CURVATURE_TYPE))
    geometry.GetPointData().AddArray(curvatures.GetOutput().GetPointData().GetScalars(CURVATURE_TYPE))
    return numpy_to_vtk(_calculate_distance_weighted_curvature(curvatures.GetOutput()))

def _find_winner_in_grid_sections(
    geometry: vtkPolyData, section_count: int
) -> List[Point]:
    curvatures = _calculate_weighted_curvatures(geometry)
    curvatures.SetName(WEIGHTED_CURVATURE_TYPE)
    geometry.GetPointData().AddArray(curvatures)
    geometry.BuildPointLocator()
    locator = geometry.GetPointLocator()

    bounds: List[float] = [ 0.0 ] * 6
    locator.GetBounds(bounds)
    point_candidates = []
    for voxel in voxelize(geometry, section_count, Bounds(*bounds)):
        if voxel.GetNumberOfPoints() == 0:
            continue
        curvature = vtk_to_numpy(voxel.GetPointData().GetAbstractArray(WEIGHTED_CURVATURE_TYPE))
        maximum_curvature_id = argmax(curvature)
        point_candidates.append(voxel.GetPoint(maximum_curvature_id))

    maximum_curvature_locations = unique(point_candidates, axis=0)
    maximum_curvature_point_ids = list(set(
        locator.FindClosestPoint(l) for l in maximum_curvature_locations
    ))

    return [Point(id_, geometry.GetPoint(id_)) for id_ in maximum_curvature_point_ids]

def _find_winner_in_neighborhoods(
    geometry: vtkPolyData, points: List[Point], neighbor_count: int
) -> List[Point]:
    geometry.BuildPointLocator()
    locator = geometry.GetPointLocator()
    winners, losers = [], []
    point_candidates = [p.id for p in points]
    for _, location in points:
        neighbor_ids = vtkIdList()
        locator.FindClosestNPoints(neighbor_count, location, neighbor_ids)
        rivals = [
            neighbor_ids.GetId(n)
            for n in range(neighbor_ids.GetNumberOfIds())
            if neighbor_ids.GetId(n) in point_candidates
        ]
        assert len(rivals) > 0
        winner = rivals[argmax([
            geometry.GetPointData().GetAbstractArray(WEIGHTED_CURVATURE_TYPE).GetTuple(r)[0]
            for r in rivals
        ])]
        if winner in losers:
            continue
        winners.append(
            winner
        )
        rivals.remove(winner)
        losers += rivals

    winners = list(set(winners))
    return [Point(id_, geometry.GetPoint(id_)) for id_ in winners]

def calculate_curved_sections(geometry: vtkPolyData):
    candidate_points = _find_winner_in_grid_sections(geometry, section_count=7)
    candidate_points = _find_winner_in_neighborhoods(
        geometry,
        points=candidate_points,
        neighbor_count=int(geometry.GetNumberOfPoints() / 100),
    )

    geometry.BuildPointLocator()
    locator = geometry.GetPointLocator()
    candidate_points = [
        locator.FindClosestPoint(point.location)
        for point in candidate_points
    ]
    mask = numpy_to_vtk([
        1 if p in candidate_points else 0
        for p in range(geometry.GetNumberOfPoints())
    ])
    mask.SetName("CurvatureMask")
    geometry.GetPointData().AddArray(mask)

    curvatures = calculate_curvature(geometry)
    geometry.GetPointData().AddArray(curvatures)
    return mask, geometry.GetPointData().GetAbstractArray(WEIGHTED_CURVATURE_TYPE), curvatures
