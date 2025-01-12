"""
this is a test
"""
# pylint: disable=missing-function-docstring
import pytest # pylint: disable=unused-import

from collections import namedtuple # pylint: disable=wrong-import-order

from numpy import identity, ndarray, sum as sum_
from vtk import vtkPolyData
from vtkmodules.util.numpy_support import vtk_to_numpy

from fast_registration.dissectionable import Dissectionable
from fast_registration.util import transform, load_stl, write

@pytest.fixture
def rotated_dissectionable_l4() -> Dissectionable:
    Data = namedtuple("Data", ["original", "rotated"])
    return Data(
        Dissectionable(load_stl("data/L4.stl")), Dissectionable(load_stl("data/L4_rot_x90.stl"))
    )

@pytest.fixture
def rotated_dissectionable_l5() -> Dissectionable:
    Data = namedtuple("Data", ["original", "rotated"])
    return Data(
        Dissectionable(load_stl("data/L5.stl")),
        Dissectionable(load_stl("data/L5_rot_x90.stl")),
    )

def transformation_error(
    source: Dissectionable, target: Dissectionable, transformation: ndarray
) -> float:
    transformation4x4 = identity(4)
    transformation4x4[:3, :3] = transformation[:3, :3]

    calculated_geometry = transform(source.normalized_geometry, transformation4x4)
    calculated_geometry = vtk_to_numpy(calculated_geometry.GetPoints().GetData())
    target = vtk_to_numpy(target.normalized_geometry.GetPoints().GetData())
    return abs(sum_(target - calculated_geometry))

def test_dissectionable_integration(
    rotated_dissectionable_l4,
    rotated_dissectionable_l5,
): # pylint: disable=redefined-outer-name
    original, rotated = rotated_dissectionable_l4
    rotation = original.min_err_rotation(rotated)
    assert transformation_error(original, rotated, rotation) < 0.001

    original, rotated = rotated_dissectionable_l5
    rotation = original.min_err_rotation(rotated)
    assert transformation_error(original, rotated, rotation) < 0.001
