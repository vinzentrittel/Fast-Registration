import pytest
from typing import Tuple

from numpy import ndarray
from vtk import vtkPlane

from plane_factory import (
    LeftPlane,
    RightPlane,
    PosteriorPlane,
    AnteriorPlane,
    InferiorPlane,
    SuperiorPlane,
    make_plane,
)

def test_make_plane():
    def to_tuple(plane: vtkPlane) -> Tuple[ndarray]:
        origin = plane.GetOrigin()
        normal = plane.GetNormal()
        return origin, normal

    left_plane = make_plane(LeftPlane)
    right_plane = make_plane(RightPlane)
    assert to_tuple(left_plane)[0][0] == -to_tuple(right_plane)[0][0]
    assert to_tuple(left_plane)[1] == to_tuple(right_plane)[1]

    posterior_plane = make_plane(PosteriorPlane)
    anterior_plane = make_plane(AnteriorPlane)
    assert to_tuple(posterior_plane)[0][1] == -to_tuple(anterior_plane)[0][1]
    assert to_tuple(posterior_plane)[1] == to_tuple(anterior_plane)[1]

    inferior_plane = make_plane(InferiorPlane)
    superior_plane = make_plane(SuperiorPlane)
    assert to_tuple(inferior_plane)[0][2] == -to_tuple(superior_plane)[0][2]
    assert to_tuple(inferior_plane)[1] == to_tuple(superior_plane)[1]
