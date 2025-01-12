from __future__ import annotations

from typing import Tuple
from enum import auto, IntEnum

from numpy import argmax, argmin, array, ndarray, diag, newaxis
from numpy.linalg import norm
from vtk import (
    vtkDoubleArray,
    vtkPCAStatistics,
    vtkPolyData,
    vtkStatisticsAlgorithm,
    vtkSTLReader,
    vtkTable,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

class AxisValue(IntEnum):
    Center = 0
    Left = -1
    Right = 1
    Posterior = -1
    Anterior = 1
    Inferior = -1
    Superior = 1

class AxisIndex(IntEnum):
    Left = 0
    Right = auto()
    Posterior = auto()
    Anterior = auto()
    Inferior = auto()
    Superior = auto()

class Converter:
    AxisConstellations = (
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    )

    @classmethod
    def from_index(cls, index: int) -> Tuple[AxisIndex, AxisIndex, AxisIndex]:
        return cls.AxisConstellations[index]

    @staticmethod
    def to_index(axes: Tuple[AxisIndex]) -> int:
        return 9 * axes[0] + 3 * axes[1] + axes[2] + 13

def eigenvectors(geometry: vtkPolyData, scale_by_eigenvalue: bool=False) -> ndarray:
    vtk_eigenvectors = vtkDoubleArray()
    pca = _make_pca_algorithm(geometry)
    pca.GetEigenvectors(vtk_eigenvectors)

    if not scale_by_eigenvalue:
        return array(vtk_eigenvectors)

    vtk_eigenvalues = vtkDoubleArray()
    pca.GetEigenvalues(vtk_eigenvalues)
    return array(vtk_eigenvectors)  @ diag(array(vtk_eigenvalues))

def extrema(points: ndarray, axis: ndarray) -> ndarray:
    projection = points.dot(axis)
    min_ = argmin(projection)
    max_ = argmax(projection)
    return array([points[max_], points[min_]])

def _make_pca_algorithm(geometry: vtkPolyData) -> vtkPCAStatistics:
    pca = vtkPCAStatistics()
    pca.SetInputData(
        vtkStatisticsAlgorithm.INPUT_DATA,
        _convert_points_to_table(geometry),
    )
    pca.SetColumnStatus("x", 1)
    pca.SetColumnStatus("y", 1)
    pca.SetColumnStatus("z", 1)
    pca.RequestSelectedColumns()
    pca.SetDeriveOption(True)
    pca.Update()
    return pca

def _convert_points_to_table(geometry: vtkPolyData) -> vtkTable:
    np_geo = vtk_to_numpy(geometry.GetPoints().GetData())
    x_array = numpy_to_vtk(np_geo[:, 0])
    x_array.SetName("x")
    y_array = numpy_to_vtk(np_geo[:, 1])
    y_array.SetName("y")
    z_array = numpy_to_vtk(np_geo[:, 2])
    z_array.SetName("z")

    table = vtkTable()
    table.AddColumn(x_array)
    table.AddColumn(y_array)
    table.AddColumn(z_array)
    return table

if __name__ == "__main__":
    from vtk.util.numpy_support import vtk_to_numpy
    from slic3r_display import Slic3rLineRepresentable
    from util import load_stl

    geometry = load_stl("data/c7.stl")
    axes = eigenvectors(geometry)

    points = vtk_to_numpy(geometry.GetPoints().GetData())
    mnm = [extrema(points, axes[id]).tolist() for id in range(3)]
    Slic3rLineRepresentable.write_from(mnm, "c7.mrk.json")
    lines = [[[0.0, 0.0, 0.0], eigenvectors(geometry, scale_by_eigenvalue=True)[id].tolist()] for id in range(3)]
    Slic3rLineRepresentable.write_from(lines, "eigen.mrk.json")
